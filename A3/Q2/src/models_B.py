"""
models_B.py  -  COL761 Assignment 3
Model for Dataset B (Binary Node Classification, AUC-ROC).
"""

import torch
import torch.nn as nn
import torch_geometric.utils as pyg_utils

class SGC_B(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, K=2, dropout=0.3):
        super().__init__()
        self.K = K
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)
        )

    def precompute_SGC(self, x, edge_index):
        N = x.size(0)
        device = edge_index.device
        
        edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=N)
        row, col = edge_index
        deg = pyg_utils.degree(col, N, dtype=torch.float32)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        adj = torch.sparse_coo_tensor(edge_index, edge_weight, (N, N), device=device)

        # Store the massive matrix in compressed FP16 on the CPU! (~6.8 GB instead of ~13.6 GB)
        out = torch.zeros(N, x.size(1), dtype=torch.float16, device='cpu')
        
        chunk_size = 100 
        for start in range(0, x.size(1), chunk_size):
            end = min(start + chunk_size, x.size(1))
            
            # Move to GPU and upscale to FP32 for stable math
            h_chunk = x[:, start:end].to(device).to(torch.float32)
            
            for _ in range(self.K):
                h_chunk = torch.sparse.mm(adj, h_chunk)
                
            # Downscale back to FP16 and send to CPU RAM
            out[:, start:end] = h_chunk.to(torch.float16).cpu()
            
        return out

    def forward(self, x, edge_index):
        with torch.no_grad():
            h = self.precompute_SGC(x, edge_index)

            device = next(self.parameters()).device
            logits = torch.zeros(h.size(0), 2, device=device) 
            chunk_size = 50000
            
            for i in range(0, h.size(0), chunk_size):
                end = min(i + chunk_size, h.size(0))
                # Cast the FP16 chunks back to FP32 for the MLP
                h_chunk = h[i:end].to(device).to(torch.float32)
                logits[i:end] = self.mlp(h_chunk)

            return logits
