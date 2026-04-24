"""
models_C.py  -  COL761 Assignment 3
Dataset C (Link Prediction).

Fixes over the previous version:
  * EMA decay lowered and EMA starts AFTER warmup (handled in train_C.py) -
    the previous 0.999 decay needed thousands of steps to catch up and was
    being dragged by the random init.
  * Regularisation dialled back to the levels that produced the 80% run.
  * Structural features and symmetric Hadamard decoder are kept - both are
    pure wins.
  * Architecture options parameterised so ensemble members can be made
    genuinely diverse (not just different seeds of the same net).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import dropout_edge, degree


def compute_structural_features(edge_index, num_nodes):
    deg = degree(edge_index[0], num_nodes=num_nodes).float()
    feats = torch.stack([
        deg / (deg.mean() + 1e-6),
        torch.log1p(deg),
        1.0 / torch.sqrt(deg.clamp(min=1.0)),
        (deg > deg.median()).float(),
    ], dim=1).to(edge_index.device)
    return feats


class SAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.2,
                 num_layers=2, edge_drop=0.2, use_jk=True,
                 use_struct=True, input_feat_drop=0.0,
                 conv_type='gcn+sage'):
        super().__init__()
        self.dropout         = dropout
        self.input_feat_drop = input_feat_drop
        self.edge_drop       = edge_drop
        self.num_layers      = num_layers
        self.use_jk          = use_jk
        self.use_struct      = use_struct
        self.conv_type       = conv_type
        self.hidden          = hidden_channels

        struct_dim = 4 if use_struct else 0
        total_in   = in_channels + struct_dim

        self.input_mlp = nn.Sequential(
            nn.Linear(total_in, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            kind = self._layer_kind(i)
            if kind == 'gcn':
                self.convs.append(GCNConv(hidden_channels, hidden_channels,
                                          cached=False, add_self_loops=True,
                                          normalize=True))
            elif kind == 'sage':
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            elif kind == 'gat':
                heads = 4
                self.convs.append(GATConv(hidden_channels, hidden_channels // heads,
                                          heads=heads, concat=True,
                                          dropout=dropout))
            else:
                raise ValueError(f'Unknown layer kind: {kind}')
            self.norms.append(nn.LayerNorm(hidden_channels))

        jk_dim = hidden_channels * (num_layers + 1) if use_jk else hidden_channels
        self.jk_proj = nn.Linear(jk_dim, hidden_channels)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.register_buffer('_struct_feats', torch.zeros(1, 4),
                             persistent=False)
        self._struct_ready = False

        self._reset_parameters()

    def _layer_kind(self, i):
        if self.conv_type == 'gcn+sage':
            return 'gcn' if i == 0 else 'sage'
        if self.conv_type == 'sage':
            return 'sage'
        if self.conv_type == 'gcn':
            return 'gcn'
        if self.conv_type == 'gat+sage':
            return 'gat' if i == 0 else 'sage'
        return 'sage'

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _prepare_features(self, x, edge_index):
        x = F.normalize(x, p=2, dim=1)
        if self.use_struct:
            if (not self._struct_ready
                    or self._struct_feats.shape[0] != x.shape[0]):
                self._struct_feats = compute_structural_features(
                    edge_index, x.shape[0]).to(x.device)
                self._struct_ready = True
            x = torch.cat([x, self._struct_feats], dim=1)
        return x

    def encode(self, x, edge_index):
        x = self._prepare_features(x, edge_index)
        h = self.input_mlp(x)

        if self.training and self.input_feat_drop > 0:
            h = F.dropout(h, p=self.input_feat_drop, training=True)

        ei = edge_index
        if self.training and self.edge_drop > 0:
            ei, _ = dropout_edge(edge_index, p=self.edge_drop,
                                 force_undirected=True)

        layer_outputs = [h]
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, ei)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new
            layer_outputs.append(h)

        if self.use_jk:
            h = torch.cat(layer_outputs, dim=1)
            h = self.jk_proj(h)

        return h

    def decode(self, z, edge_pairs):
        src = z[edge_pairs[:, 0]]
        dst = z[edge_pairs[:, 1]]
        h = src * dst
        return self.decoder(h).squeeze(-1)

    def forward(self, x, edge_index, edge_pairs):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_pairs)


class ModelEMA:
    def __init__(self, model, decay=0.99):
        self.decay  = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(),
                                                     alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    @torch.no_grad()
    def reset_from(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].copy_(v.detach())

    def copy_to(self, model):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


class EnsembleSAGENet(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def encode(self, x, edge_index):
        return [m.encode(x, edge_index) for m in self.models]

    def decode(self, z_or_list, edge_pairs):
        if not isinstance(z_or_list, (list, tuple)):
            return self.models[0].decode(z_or_list, edge_pairs)
        logits = [m.decode(z, edge_pairs)
                  for m, z in zip(self.models, z_or_list)]
        return torch.stack(logits, dim=0).mean(dim=0)

    def forward(self, x, edge_index, edge_pairs):
        z_list = self.encode(x, edge_index)
        return self.decode(z_list, edge_pairs)
