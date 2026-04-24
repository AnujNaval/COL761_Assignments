"""
train_B.py  -  COL761 Assignment 3
Dataset B: Binary Node Classification (AUC-ROC)
"""

import argparse
import os
import sys
import time
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_B import SGC_B

def load_data_raw(data_dir):
    path = os.path.join(data_dir, 'B', 'data.pt')
    print(f'  Loading {path} ...')
    return torch.load(path, weights_only=False)

def compute_class_weights(y_train):
    y_train_long = y_train.long()
    n_class = torch.bincount(y_train_long)
    weights = len(y_train_long) / (len(n_class) * n_class.float())
    return weights

def train_epoch(model, loader, optimizer, class_weights, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        # Cast FP16 back to FP32 on the GPU for training
        x_batch = x_batch.to(device).to(torch.float32)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model.mlp(x_batch) 
        loss = F.cross_entropy(logits, y_batch, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_auc(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device).to(torch.float32)
        logits = model.mlp(x_batch)
        scores = torch.softmax(logits, dim=1)[:, 1]
        all_scores.extend(scores.cpu().tolist())
        all_labels.extend(y_batch.tolist())
    
    if len(set(all_labels)) < 2:
        return 0.5
    return roc_auc_score(all_labels, all_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--model_dir',   required=True)
    parser.add_argument('--kerberos',    required=True)
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--patience',    type=int,   default=15)
    parser.add_argument('--time_budget', type=int,   default=6000)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print('Loading Dataset B...')
    data = load_data_raw(args.data_dir)
    in_ch = data.x.shape[1]

    # --- AGGRESSIVE RAM MANAGEMENT ---
    print('Compressing raw features to FP16 to save RAM...')
    data.x = data.x.to(torch.float16)
    gc.collect()

    print('\nPrecomputing SGC Features (A^K X). Takes ~1 minute...')
    temp_model = SGC_B(in_ch, K=2)
    with torch.no_grad():
        edge_index_gpu = data.edge_index.to(device)
        x_sgc_cpu = temp_model.precompute_SGC(data.x, edge_index_gpu)

    # Nuke the original data.x from orbit to free up another 6.8 GB of RAM
    print('Precomputation complete. Freeing raw feature memory...')
    del data.x
    gc.collect()

    # --- PREPARE TABULAR DATALOADERS ---
    train_nodes = data.labeled_nodes[data.train_mask]
    val_nodes   = data.labeled_nodes[data.val_mask]

    x_train = x_sgc_cpu[train_nodes]
    y_train = data.y[data.train_mask].long()
    x_val   = x_sgc_cpu[val_nodes]
    y_val   = data.y[data.val_mask].long()

    class_weights = compute_class_weights(y_train).to(device)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=2048, shuffle=True)
    val_loader   = DataLoader(TensorDataset(x_val, y_val), batch_size=2048, shuffle=False)

    # --- EXPERIMENTS ---
    configs = [
        {'name': 'SGC-K2-H256', 'K': 2, 'hidden': 256, 'lr': 0.005, 'wd': 1e-4},
        {'name': 'SGC-K2-H512', 'K': 2, 'hidden': 512, 'lr': 0.001, 'wd': 1e-4},
    ]

    best_overall_auc = 0.0
    best_overall_model = None
    start_time = time.time()

    for cfg in configs:
        print(f"\n--- Training {cfg['name']} ---")
        model = SGC_B(in_channels=in_ch, hidden_channels=cfg['hidden'], K=cfg['K']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        best_auc = 0.0
        best_state = None
        pat_count = 0

        for epoch in range(1, args.epochs + 1):
            if (time.time() - start_time) > args.time_budget:
                print("Time budget exceeded!")
                break

            loss = train_epoch(model, train_loader, optimizer, class_weights, device)
            auc  = evaluate_auc(model, val_loader, device)
            scheduler.step(auc)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | AUC: {auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                pat_count = 0
            else:
                pat_count += 1

            if pat_count >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        print(f"  --> Best AUC for {cfg['name']}: {best_auc:.4f}")
        
        if best_auc > best_overall_auc:
            best_overall_auc = best_auc
            model.load_state_dict(best_state)
            best_overall_model = model

    # --- SAVE MODEL ---
    best_overall_model.eval()
    model_path = os.path.join(args.model_dir, f'{args.kerberos}_model_B.pt')
    torch.save(best_overall_model, model_path)

    print(f'\nBest Overall AUC : {best_overall_auc:.4f}')
    print(f'Model saved      : {model_path}')

if __name__ == '__main__':
    main()
