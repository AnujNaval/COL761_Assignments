"""
train_A.py  -  COL761 Assignment 3
Dataset A: Node classification (7 classes, Accuracy).

  * Transductive full-batch training, loss masked to train nodes.
  * Label indexing: logits[labeled_nodes][mask] vs y[mask].
  * Class-balanced cross-entropy (auto-disabled if all weights ~= 1).
  * EMA decay 0.99 with 50-epoch warmup.
  * PlainGCN ensemble backbone + NodeNet variants for diversity.
"""

import argparse
import math
import os
import sys
import time
import torch
import torch.nn.functional as F

from load_dataset import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_A import NodeNet, PlainGCN, EnsembleNodeNet, ModelEMA


@torch.no_grad()
def evaluate(model, x, edge_index, labeled_nodes, mask, y):
    model.eval()
    logits = model(x, edge_index)
    preds  = logits[labeled_nodes][mask].argmax(dim=1)
    targets = y[mask]
    return (preds == targets).float().mean().item()


def train_epoch(model, x, edge_index, labeled_nodes, train_mask, y,
                optimizer, class_weights=None):
    model.train()
    optimizer.zero_grad()
    logits = model(x, edge_index)
    train_logits  = logits[labeled_nodes][train_mask]
    train_targets = y[train_mask]
    loss = F.cross_entropy(train_logits, train_targets, weight=class_weights)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def clone_architecture(model):
    struct_dim  = 4 if model.use_struct else 0
    in_channels = model.input_mlp[0].in_features - struct_dim
    if isinstance(model, PlainGCN):
        return PlainGCN(
            in_channels     = in_channels,
            hidden_channels = model.hidden,
            num_classes     = model.num_classes,
            dropout         = model.dropout,
            use_struct      = model.use_struct,
            num_layers      = model.num_layers,
        )
    return NodeNet(
        in_channels     = in_channels,
        hidden_channels = model.hidden,
        num_classes     = model.num_classes,
        dropout         = model.dropout,
        num_layers      = model.num_layers,
        edge_drop       = model.edge_drop,
        use_jk          = model.use_jk,
        use_struct      = model.use_struct,
        input_feat_drop = model.input_feat_drop,
        conv_type       = model.conv_type,
    )


def run(name, model, x, edge_index, labeled_nodes, train_mask, val_mask, y,
        lr, wd, epochs, patience, deadline, start_time,
        class_weights=None, ema_decay=0.99, ema_warmup=50):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    warmup_sched = 50
    def lr_lambda(step):
        if step < warmup_sched:
            return (step + 1) / warmup_sched
        progress = (step - warmup_sched) / max(1, epochs - warmup_sched)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ema        = ModelEMA(model, decay=ema_decay)
    eval_model = clone_architecture(model).to(x.device)

    best_acc   = 0.0
    best_state = None
    best_src   = 'raw'
    pat_count  = 0

    print(f'\n  [{name}]  lr={lr}  wd={wd}  '
          f'type={type(model).__name__}  '
          f'h={model.hidden}  L={model.num_layers}')

    for epoch in range(1, epochs + 1):
        if time.time() > deadline:
            print(f'    Time budget reached at epoch {epoch}')
            break

        loss = train_epoch(model, x, edge_index, labeled_nodes,
                           train_mask, y, optimizer,
                           class_weights=class_weights)
        scheduler.step()

        if epoch <= ema_warmup:
            ema.reset_from(model)
        else:
            ema.update(model)

        a_raw = evaluate(model, x, edge_index, labeled_nodes, val_mask, y)
        ema.copy_to(eval_model)
        a_ema = evaluate(eval_model, x, edge_index, labeled_nodes, val_mask, y)

        if a_raw >= a_ema:
            acc, src_state, src_label = a_raw, model.state_dict(), 'raw'
        else:
            acc, src_state, src_label = a_ema, ema.state_dict(), 'ema'

        if epoch % 25 == 0 or epoch == 1:
            elapsed = (time.time() - start_time) / 60
            cur_lr  = optimizer.param_groups[0]['lr']
            print(f'    Epoch {epoch:4d} | Loss: {loss:.4f} | '
                  f'Acc raw/ema: {a_raw:.4f}/{a_ema:.4f} | '
                  f'lr: {cur_lr:.5f} | Elapsed: {elapsed:.1f}m')

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.detach().clone() for k, v in src_state.items()}
            best_src   = src_label
            pat_count  = 0
        else:
            pat_count += 1

        if pat_count >= patience:
            print(f'    Early stop at epoch {epoch} '
                  f'(best Acc={best_acc:.4f} from {best_src})')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f'  --> best Acc: {best_acc:.4f}')
    return best_acc


def build_model(in_ch, num_classes, cfg):
    cls = cfg.get('model_class', 'NodeNet')
    if cls == 'PlainGCN':
        return PlainGCN(
            in_channels     = in_ch,
            hidden_channels = cfg['hidden'],
            num_classes     = num_classes,
            dropout         = cfg['dropout'],
            use_struct      = cfg.get('use_struct', False),
            num_layers      = cfg.get('num_layers', 2),
        )
    return NodeNet(
        in_channels     = in_ch,
        hidden_channels = cfg['hidden'],
        num_classes     = num_classes,
        dropout         = cfg['dropout'],
        num_layers      = cfg['num_layers'],
        edge_drop       = cfg['edge_drop'],
        use_jk          = cfg.get('use_jk', True),
        use_struct      = cfg.get('use_struct', True),
        input_feat_drop = cfg.get('input_feat_drop', 0.0),
        conv_type       = cfg.get('conv_type', 'gcn+sage'),
    )


def compute_class_weights(y_train, num_classes):
    counts = torch.bincount(y_train, minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    w = 1.0 / counts
    w = w * (num_classes / w.sum())
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--model_dir',   required=True)
    parser.add_argument('--kerberos',    required=True)
    parser.add_argument('--epochs',      type=int, default=300)
    parser.add_argument('--patience',    type=int, default=100)
    parser.add_argument('--time_budget', type=int, default=1800)
    parser.add_argument('--n_ensemble',  type=int, default=5)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--no_class_weights', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Dataset A...')
    ds   = load_dataset('A', args.data_dir)
    data = ds[0]

    x             = data.x.to(device)
    edge_index    = data.edge_index.to(device)
    y             = data.y.to(device)
    labeled_nodes = data.labeled_nodes.to(device)
    train_mask    = data.train_mask.to(device)
    val_mask      = data.val_mask.to(device)
    num_classes   = ds.num_classes
    in_ch         = x.shape[1]

    print(f'  Device={device}  Nodes={data.num_nodes}  Feats={in_ch}  '
          f'Classes={num_classes}')
    print(f'  labeled={labeled_nodes.shape[0]}  '
          f'train={train_mask.sum().item()}  val={val_mask.sum().item()}')

    if args.no_class_weights:
        class_weights = None
    else:
        y_train = y[train_mask]
        class_weights = compute_class_weights(y_train, num_classes).to(device)
        print(f'  class_weights = {class_weights.cpu().numpy().round(3).tolist()}')

    # PlainGCN backbone (proven ~81% on Cora) + NodeNet for diversity
    arch_configs = [
        dict(name='PlainGCN-16-2L',   model_class='PlainGCN',
             hidden=16, num_layers=2,
             dropout=0.5, edge_drop=0.0,
             lr=1e-2, wd=5e-4, use_struct=False),
        dict(name='PlainGCN-64-2L',   model_class='PlainGCN',
             hidden=64, num_layers=2,
             dropout=0.5, edge_drop=0.0,
             lr=1e-2, wd=5e-4, use_struct=False),
        dict(name='PlainGCN-128-2L',  model_class='PlainGCN',
             hidden=128, num_layers=2,
             dropout=0.5, edge_drop=0.0,
             lr=1e-2, wd=5e-4, use_struct=False),
        dict(name='PlainGCN-64-struct', model_class='PlainGCN',
             hidden=64, num_layers=2,
             dropout=0.5, edge_drop=0.0,
             lr=1e-2, wd=5e-4, use_struct=True),
        dict(name='NodeNet-GCN+SAGE-128-2L', model_class='NodeNet',
             conv_type='gcn+sage',
             hidden=128, num_layers=2,
             dropout=0.5, edge_drop=0.0, input_feat_drop=0.0,
             lr=5e-3, wd=5e-4, use_struct=True, use_jk=False),
    ]

    n = min(args.n_ensemble, len(arch_configs))
    time_per_model = args.time_budget // n
    start_time     = time.time()

    trained_models = []
    individual_acc = []

    for i in range(n):
        cfg  = arch_configs[i]
        seed = args.seed + 1000 * i
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        deadline = start_time + (i + 1) * time_per_model
        print(f'\n===== Ensemble member {i+1}/{n}   '
              f'config={cfg["name"]}   seed={seed} =====')

        model = build_model(in_ch, num_classes, cfg).to(device)
        acc = run(
            f'{cfg["name"]}-s{seed}', model, x, edge_index,
            labeled_nodes, train_mask, val_mask, y,
            lr=cfg['lr'], wd=cfg['wd'],
            epochs=args.epochs, patience=args.patience,
            deadline=deadline, start_time=start_time,
            class_weights=class_weights,
        )
        trained_models.append(model)
        individual_acc.append(acc)

    print('\n===== Greedy ensemble selection =====')
    order = sorted(range(n), key=lambda i: -individual_acc[i])
    kept = [order[0]]
    best_ens_acc = individual_acc[order[0]]
    print(f'  Seed ensemble: member {order[0]+1} (Acc={best_ens_acc:.4f})')
    for idx in order[1:]:
        candidate_idxs = kept + [idx]
        candidate = EnsembleNodeNet(
            [trained_models[j] for j in candidate_idxs]).to(device)
        candidate.eval()
        a = evaluate(candidate, x, edge_index, labeled_nodes, val_mask, y)
        status = 'KEEP' if a > best_ens_acc else 'drop'
        print(f'  + member {idx+1} -> Acc={a:.4f}  [{status}]')
        if a > best_ens_acc:
            kept = candidate_idxs
            best_ens_acc = a

    print(f'\nIndividual Acc: {[f"{a:.4f}" for a in individual_acc]}')
    print(f'Kept members  : {[i+1 for i in kept]}')
    print(f'Final ensemble: Acc={best_ens_acc:.4f}')

    if len(kept) == 1:
        final_model = trained_models[kept[0]]
        print(f'Saving single best model (#{kept[0]+1})')
    else:
        final_model = EnsembleNodeNet(
            [trained_models[j] for j in kept]).to(device)
        print(f'Saving ensemble of {len(kept)} models')

    total_time = (time.time() - start_time) / 60
    print(f'\nTotal time: {total_time:.1f} minutes')

    final_model.eval()
    model_path = os.path.join(args.model_dir, f'{args.kerberos}_model_A.pt')
    torch.save(final_model, model_path)
    print(f'Best Acc     : {best_ens_acc:.4f}  ({best_ens_acc*100:.2f}%)')
    print(f'Model saved  : {model_path}')


if __name__ == '__main__':
    main()
