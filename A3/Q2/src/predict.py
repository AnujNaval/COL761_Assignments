"""
predict.py  –  COL761 Assignment 3 prediction script
(Patched to handle massive Dataset B constraints)
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_dataset import COL761NodeDataset, COL761LinkDataset, load_dataset, _load_edge_list

def load_model(model_path: str) -> torch.nn.Module:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = torch.load(model_path, weights_only=False, map_location="cpu")
    model.eval()
    return model

def _random_A(dataset) -> torch.Tensor:
    return torch.randint(0, dataset.num_classes, (dataset[0].num_nodes,))

def _random_B(dataset) -> torch.Tensor:
    return torch.rand(dataset[0].num_nodes)

def _random_C(V: int, K: int) -> tuple:
    return torch.rand(V), torch.rand(V, K)

@torch.no_grad()
def predict_A(model: torch.nn.Module, dataset) -> torch.Tensor:
    data = dataset[0]
    device = next(model.parameters()).device
    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    logits = model(x, edge_index)
    return logits.argmax(dim=1).cpu()

@torch.no_grad()
def predict_B(model: torch.nn.Module, dataset) -> torch.Tensor:
    data = dataset[0]
    device = next(model.parameters()).device
    
    # Dataset B Patch: LEAVE X ON CPU!
    x          = data.x 
    edge_index = data.edge_index.to(device)

    logits = model(x, edge_index)    

    if logits.shape[1] == 1:
        return torch.sigmoid(logits).squeeze(1).cpu()
    return torch.softmax(logits, dim=1)[:, 1].cpu()

@torch.no_grad()
def predict_C(model: torch.nn.Module, dataset, test_dir: str = None):
    device = next(model.parameters()).device
    x          = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)

    if test_dir is None:
        pos, neg, split = dataset.valid_pos.to(device), dataset.valid_neg.to(device), "valid"
    else:
        pos   = _load_edge_list(os.path.join(test_dir, "test_pos.txt")).to(device)
        npy   = os.path.join(test_dir, "test_neg_hard.npy")
        with open(npy, "rb") as f:
            neg = torch.from_numpy(np.load(f)).to(device)
        split = "test"

    P, K, _ = neg.shape
    pos_scores = model(x, edge_index, pos)

    CHUNK = 5000
    neg_flat  = neg.view(P * K, 2)
    neg_score_parts = []
    for start in range(0, P * K, CHUNK):
        end   = min(start + CHUNK, P * K)
        chunk = model(x, edge_index, neg_flat[start:end])
        neg_score_parts.append(chunk)
    neg_scores = torch.cat(neg_score_parts, dim=0).view(P, K)

    return pos_scores.cpu(), neg_scores.cpu(), split

def predict_and_save(dataset_name: str, data_dir: str, model_path: str, out_dir: str, test_dir: str = None, kerberos: str = "student") -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading dataset {dataset_name} ...")

    # FIX: Bypass PyG cache for Dataset B to avoid disk blowouts
    if dataset_name == "B":
        print("  Bypassing PyG cache for Dataset B to save disk space & RAM...")
        data_path = os.path.join(data_dir, 'B', 'data.pt')
        data = torch.load(data_path, weights_only=False, map_location='cpu')
        data.x = data.x.to(torch.float16) 
        
        class DummyDataset:
            def __init__(self, d): self.data = d
            def __getitem__(self, idx): return self.data
        ds = DummyDataset(data)
    else:
        ds = load_dataset(dataset_name, data_dir)

    if model_path is not None:
        print(f"Loading model from {model_path} ...")
        model = load_model(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    else:
        print("No --model_path given — using random predictions.")
        model = None

    if dataset_name == "A":
        y_pred = predict_A(model, ds) if model else _random_A(ds)
        out_path = os.path.join(out_dir, f"{kerberos}_predictions_A.pt")
        torch.save({"y_pred": y_pred}, out_path)
        print(f"Saved {out_path}  shape={y_pred.shape}")

    elif dataset_name == "B":
        y_score = predict_B(model, ds) if model else _random_B(ds)
        out_path = os.path.join(out_dir, f"{kerberos}_predictions_B.pt")
        torch.save({"y_score": y_score}, out_path)
        print(f"Saved {out_path}  shape={y_score.shape}")

    elif dataset_name == "C":
        if model:
            pos_scores, neg_scores, split = predict_C(model, ds, test_dir=test_dir)
        else:
            if test_dir or not hasattr(ds, "valid_pos"):
                pos, neg, split = ds.test_pos, ds.test_neg, "test"
            else:
                pos, neg, split = ds.valid_pos, ds.valid_neg, "valid"
            V, K = pos.shape[0], neg.shape[1]
            pos_scores, neg_scores = _random_C(V, K)

        out_path = os.path.join(out_dir, f"{kerberos}_predictions_C.pt")
        torch.save({"pos_scores": pos_scores, "neg_scores": neg_scores, "split": split}, out_path)
        print(f"Saved {out_path}  split={split}")

def main():
    parser = argparse.ArgumentParser(description="Generate predictions for COL761 A3 datasets.")
    parser.add_argument("--dataset",    required=True, choices=["A", "B", "C"])
    parser.add_argument("--task",       required=True, choices=["node", "link"])
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--model_dir",  default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--kerberos",   required=True)
    parser.add_argument("--test_dir",   default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    model_path = None
    if args.model_dir is not None:
        model_path = os.path.join(args.model_dir, f"{args.kerberos}_model_{args.dataset}.pt")

    predict_and_save(
        args.dataset, args.data_dir, model_path, args.output_dir,
        test_dir=args.test_dir, kerberos=args.kerberos,
    )

if __name__ == "__main__":
    main()