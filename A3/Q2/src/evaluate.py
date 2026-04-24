"""
evaluate.py  –  COL761 Assignment 3 evaluation script
(Patched to handle massive Dataset B constraints)
"""

import argparse
import os
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from load_dataset import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def hits_at_k(pos_scores: torch.Tensor, neg_scores: torch.Tensor, k: int) -> float:
    n_neg_higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    return (n_neg_higher < k).float().mean().item()

# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset evaluators
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_A(pred_path: str, split: str, data_dir: str, gt_dir: str) -> float:
    pred  = torch.load(pred_path, weights_only=False)
    y_pred = pred["y_pred"]

    if split == "val":
        data        = load_dataset("A", data_dir)[0]
        node_idx    = data.labeled_nodes[data.val_mask]
        true_labels = data.y[data.val_mask]
    else:
        gt          = torch.load(os.path.join(gt_dir, "A", "test.pt"), weights_only=False)
        node_idx    = gt["test_node_idx"]
        true_labels = gt["test_labels"]

    return accuracy_score(true_labels.numpy(), y_pred[node_idx].numpy())

def evaluate_B(pred_path: str, split: str, data_dir: str, gt_dir: str) -> float:
    pred    = torch.load(pred_path, weights_only=False)
    y_score = pred["y_score"]

    if split == "val":
        # PATCH: Bypass disk-heavy processing for Dataset B
        data_path = os.path.join(data_dir, 'B', 'data.pt')
        data = torch.load(data_path, weights_only=False, map_location='cpu')
        node_idx    = data.labeled_nodes[data.val_mask]
        true_labels = data.y[data.val_mask]
    else:
        gt          = torch.load(os.path.join(gt_dir, "B", "test.pt"), weights_only=False)
        node_idx    = gt["test_node_idx"]
        true_labels = gt["test_labels"]

    return roc_auc_score(true_labels.numpy(), y_score[node_idx].numpy())

def evaluate_C(pred_path: str, k: int = 50) -> tuple[float, str]:
    pred       = torch.load(pred_path, weights_only=False)
    pos_scores = pred["pos_scores"]
    neg_scores = pred["neg_scores"]
    split      = pred.get("split", "unknown")
    return hits_at_k(pos_scores, neg_scores, k=k), split

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate COL761 A3 predictions.")
    parser.add_argument("--dataset",    required=True, choices=["A", "B", "C"])
    parser.add_argument("--task",       required=True, choices=["node", "link"])
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--kerberos",   required=True)
    parser.add_argument("--split",   default="val", choices=["val", "test"], help=argparse.SUPPRESS)
    parser.add_argument("--gt_dir",  default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    pred_file = os.path.join(args.output_dir, f"{args.kerberos}_predictions_{args.dataset}.pt")
    if not os.path.isfile(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    if args.dataset in ("A", "B"):
        if args.split == "test" and args.gt_dir is None:
            args.gt_dir = args.data_dir

    print(f"\nEvaluating dataset {args.dataset} ...")

    if args.dataset == "A":
        acc = evaluate_A(pred_file, args.split, args.data_dir, args.gt_dir)
        print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

    elif args.dataset == "B":
        auc = evaluate_B(pred_file, args.split, args.data_dir, args.gt_dir)
        print(f"  AUC-ROC  : {auc:.4f}")

    elif args.dataset == "C":
        h50, split = evaluate_C(pred_file, k=50)
        print(f"  Hits@50  : {h50:.4f}  ({h50*100:.2f}%)")

if __name__ == "__main__":
    main()