"""
train_C.py  -  COL761 Assignment 3  -  Dataset C (Link Prediction)
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
from models_C import SAGENet, EnsembleSAGENet, ModelEMA


@torch.no_grad()
def hits_at_k(model, x, edge_index, pos_edges, neg_edges, k=50):
    model.eval()
    z = model.encode(x, edge_index)
    pos_scores = model.decode(z, pos_edges)

    if neg_edges.dim() == 3:
        P, Kn, _ = neg_edges.shape
        neg_scores = model.decode(z, neg_edges.reshape(P * Kn, 2)).view(P, Kn)
        n_higher = (neg_scores >= pos_scores.unsqueeze(1)).sum(dim=1)
        return (n_higher < k).float().mean().item()
    else:
        neg_scores = model.decode(z, neg_edges)
        neg_sorted, _ = torch.sort(neg_scores)
        idx = torch.searchsorted(neg_sorted, pos_scores, right=True)
        n_higher = neg_sorted.numel() - idx
        return (n_higher < k).float().mean().item()


def train_epoch(model, x, edge_index, train_pos, num_nodes, optimizer):
    model.train()
    optimizer.zero_grad()

    z = model.encode(x, edge_index)
    pos_scores = model.decode(z, train_pos)

    M = train_pos.shape[0]
    device = x.device
    src = train_pos[:, 0]
    dst = train_pos[:, 1]
    rand_a = torch.randint(0, num_nodes, (M,), device=device)
    rand_b = torch.randint(0, num_nodes, (M,), device=device)
    neg_a = torch.stack([src, rand_a], dim=1)
    neg_b = torch.stack([rand_b, dst], dim=1)
    neg_edges = torch.cat([neg_a, neg_b], dim=0)
    neg_scores = model.decode(z, neg_edges)

    loss = (
        F.binary_cross_entropy_with_logits(pos_scores,
                                           torch.ones_like(pos_scores))
      + F.binary_cross_entropy_with_logits(neg_scores,
                                           torch.zeros_like(neg_scores))
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def clone_architecture(model):
    struct_dim = 4 if model.use_struct else 0
    in_channels = model.input_mlp[0].in_features - struct_dim
    return SAGENet(
        in_channels     = in_channels,
        hidden_channels = model.hidden,
        dropout         = model.dropout,
        num_layers      = model.num_layers,
        edge_drop       = model.edge_drop,
        use_jk          = model.use_jk,
        use_struct      = model.use_struct,
        input_feat_drop = model.input_feat_drop,
        conv_type       = model.conv_type,
    )


def run(name, model, x, edge_index, train_pos,
        valid_pos, valid_neg, lr, wd, epochs, patience,
        deadline, start_time, num_nodes,
        ema_decay=0.99, ema_warmup=50):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    warmup_sched = 50
    def lr_lambda(step):
        if step < warmup_sched:
            return (step + 1) / warmup_sched
        progress = (step - warmup_sched) / max(1, epochs - warmup_sched)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ema = ModelEMA(model, decay=ema_decay)
    eval_model = clone_architecture(model).to(x.device)

    best_hits  = 0.0
    best_state = None
    best_src   = 'raw'
    pat_count  = 0

    print(f'\n  [{name}]  lr={lr}  wd={wd}  '
          f'conv={model.conv_type}  '
          f'h={model.hidden} L={model.num_layers}')

    for epoch in range(1, epochs + 1):
        if time.time() > deadline:
            print(f'    Time budget reached at epoch {epoch}')
            break

        loss = train_epoch(model, x, edge_index, train_pos,
                           num_nodes, optimizer)
        scheduler.step()

        if epoch <= ema_warmup:
            ema.reset_from(model)
        else:
            ema.update(model)

        h_raw = hits_at_k(model, x, edge_index, valid_pos, valid_neg, k=50)
        ema.copy_to(eval_model)
        h_ema = hits_at_k(eval_model, x, edge_index, valid_pos, valid_neg, k=50)

        if h_raw >= h_ema:
            h50 = h_raw
            src_state = model.state_dict()
            src_label = 'raw'
        else:
            h50 = h_ema
            src_state = ema.state_dict()
            src_label = 'ema'

        if epoch % 25 == 0 or epoch == 1:
            elapsed = (time.time() - start_time) / 60
            cur_lr  = optimizer.param_groups[0]['lr']
            print(f'    Epoch {epoch:4d} | Loss: {loss:.4f} | '
                  f'Hits@50 raw/ema: {h_raw:.4f}/{h_ema:.4f} | '
                  f'lr: {cur_lr:.5f} | Elapsed: {elapsed:.1f}m')

        if h50 > best_hits:
            best_hits  = h50
            best_state = {k: v.detach().clone() for k, v in src_state.items()}
            best_src   = src_label
            pat_count  = 0
        else:
            pat_count += 1

        if pat_count >= patience:
            print(f'    Early stop at epoch {epoch} '
                  f'(best Hits@50={best_hits:.4f} from {best_src})')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f'  --> best Hits@50: {best_hits:.4f}')
    return best_hits


def build_model(in_ch, cfg):
    return SAGENet(
        in_channels     = in_ch,
        hidden_channels = cfg['hidden'],
        dropout         = cfg['dropout'],
        num_layers      = cfg['num_layers'],
        edge_drop       = cfg['edge_drop'],
        use_jk          = cfg.get('use_jk', True),
        use_struct      = cfg.get('use_struct', True),
        input_feat_drop = cfg.get('input_feat_drop', 0.0),
        conv_type       = cfg.get('conv_type', 'gcn+sage'),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--model_dir',   required=True)
    parser.add_argument('--kerberos',    required=True)
    parser.add_argument('--epochs',      type=int, default=600)
    parser.add_argument('--patience',    type=int, default=100)
    parser.add_argument('--time_budget', type=int, default=2400)
    parser.add_argument('--n_ensemble',  type=int, default=5)
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Dataset C...')
    ds         = load_dataset('C', args.data_dir)
    x          = ds.x.to(device)
    edge_index = ds.edge_index.to(device)
    train_pos  = ds.train_pos.to(device)
    valid_pos  = ds.valid_pos.to(device)
    valid_neg  = ds.valid_neg.to(device)
    num_nodes  = ds.num_nodes
    in_ch      = x.shape[1]

    print(f'  Device={device}  Nodes={num_nodes}  Feats={in_ch}')
    print(f'  train_pos={tuple(train_pos.shape)}  '
          f'valid_pos={tuple(valid_pos.shape)}  '
          f'valid_neg={tuple(valid_neg.shape)}')

    arch_configs = [
        dict(name='GCN+SAGE-256-3L', conv_type='gcn+sage',
             hidden=256, num_layers=3,
             dropout=0.2, edge_drop=0.2, input_feat_drop=0.0,
             lr=5e-3, wd=5e-4),
        dict(name='GCN+SAGE-512-3L', conv_type='gcn+sage',
             hidden=512, num_layers=3,
             dropout=0.3, edge_drop=0.2, input_feat_drop=0.0,
             lr=3e-3, wd=1e-4),
        dict(name='SAGE-pure-256-3L', conv_type='sage',
             hidden=256, num_layers=3,
             dropout=0.2, edge_drop=0.2, input_feat_drop=0.0,
             lr=5e-3, wd=5e-4),
        dict(name='GCN+SAGE-384-2L', conv_type='gcn+sage',
             hidden=384, num_layers=2,
             dropout=0.2, edge_drop=0.2, input_feat_drop=0.0,
             lr=5e-3, wd=5e-4),
        dict(name='GAT+SAGE-256-3L', conv_type='gat+sage',
             hidden=256, num_layers=3,
             dropout=0.2, edge_drop=0.2, input_feat_drop=0.0,
             lr=3e-3, wd=5e-4),
    ]

    n = min(args.n_ensemble, len(arch_configs))
    time_per_model = args.time_budget // n
    start_time     = time.time()

    trained_models = []
    individual_h50 = []

    for i in range(n):
        cfg  = arch_configs[i]
        seed = args.seed + 1000 * i
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        deadline = start_time + (i + 1) * time_per_model
        print(f'\n===== Ensemble member {i+1}/{n}   '
              f'config={cfg["name"]}   seed={seed} =====')

        model = build_model(in_ch, cfg).to(device)
        h50 = run(
            f'{cfg["name"]}-s{seed}', model, x, edge_index, train_pos,
            valid_pos, valid_neg,
            lr=cfg['lr'], wd=cfg['wd'],
            epochs=args.epochs, patience=args.patience,
            deadline=deadline, start_time=start_time,
            num_nodes=num_nodes,
        )
        trained_models.append(model)
        individual_h50.append(h50)

    print('\n===== Greedy ensemble selection =====')
    order = sorted(range(n), key=lambda i: -individual_h50[i])
    kept = [order[0]]
    best_ens_h50 = individual_h50[order[0]]
    print(f'  Seed ensemble: member {order[0]+1} '
          f'(Hits@50={best_ens_h50:.4f})')
    for idx in order[1:]:
        candidate_idxs = kept + [idx]
        candidate = EnsembleSAGENet(
            [trained_models[j] for j in candidate_idxs]).to(device)
        candidate.eval()
        h = hits_at_k(candidate, x, edge_index, valid_pos, valid_neg, k=50)
        status = 'KEEP' if h > best_ens_h50 else 'drop'
        print(f'  + member {idx+1} -> Hits@50={h:.4f}  [{status}]')
        if h > best_ens_h50:
            kept = candidate_idxs
            best_ens_h50 = h

    print(f'\nIndividual Hits@50: {[f"{h:.4f}" for h in individual_h50]}')
    print(f'Kept members     : {[i+1 for i in kept]}')
    print(f'Final ensemble   : Hits@50={best_ens_h50:.4f}')

    if len(kept) == 1:
        final_model = trained_models[kept[0]]
        print(f'Saving single best model (#{kept[0]+1})')
    else:
        final_model = EnsembleSAGENet(
            [trained_models[j] for j in kept]).to(device)
        print(f'Saving ensemble of {len(kept)} models')

    total_time = (time.time() - start_time) / 60
    print(f'\nTotal time: {total_time:.1f} minutes')

    final_model.eval()
    model_path = os.path.join(args.model_dir, f'{args.kerberos}_model_C.pt')
    torch.save(final_model, model_path)
    print(f'Best Hits@50 : {best_ens_h50:.4f}  ({best_ens_h50*100:.2f}%)')
    print(f'Model saved  : {model_path}')


if __name__ == '__main__':
    main()
