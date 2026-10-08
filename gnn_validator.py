"""
GNN Validator Architecture.
Implements the Multi-Layer Graph Isomorphism Network with Edge features (GINE)
and multi-scale global readout pooling formalized in Section III-C (Eq. 13, 14, 15).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, GlobalAttention


class MultiScaleGINEValidator(nn.Module):
    """
    Generalized Zero-Cost GNN Validator.
    
    Features:
      - Input normalization layer handling heterogeneous feature distributions
      - Multi-layer GINE with edge feature projection W_e and learnable eps (Eq. 13)
      - Multi-scale global readout pooling concatenating representations
        across all layers k = 0, 1, ..., n (Eq. 14)
      - Regression head Psi predicting scalar performance proxy (Eq. 15)
      - Optional family-specific readout heads for multi-paradigm meta-training (Eq. 16)
    """
    def __init__(
        self,
        num_node_features=23,
        num_edge_features=4,
        hidden_dim=64,
        num_layers=4,
        use_attention_readout=False,
        head_names=None,
    ):
        super(MultiScaleGINEValidator, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_attention_readout = use_attention_readout

        # Input feature normalization & projection
        self.input_bn = nn.BatchNorm1d(num_node_features)
        self.node_proj = nn.Linear(num_node_features, hidden_dim)

        # Message passing layers (GINEConv with edge attributes)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # train_eps=True implements (1 + eps^(k)) in Eq. (13)
            conv = GINEConv(mlp, train_eps=True, edge_dim=num_edge_features)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Global readout: concatenates k=0..n representations -> (n + 1) * hidden_dim
        readout_dim = hidden_dim * (num_layers + 1)

        if use_attention_readout:
            gate_nn = nn.Sequential(nn.Linear(hidden_dim, 1))
            self.attn_pool = GlobalAttention(gate_nn=gate_nn)

        # Unified regression head Psi (Eq. 15)
        self.mlp_head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Family-specific heads (if specified)
        self.heads = nn.ModuleDict()
        if head_names:
            for name in head_names:
                self.heads[name] = nn.Sequential(
                    nn.Linear(readout_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )

    def extract_graph_embedding(self, x, edge_index, edge_attr, batch):
        """
        Executes GINE message passing and multi-scale concatenation (Eq. 13-14).
        Returns comprehensive graph embedding h_G.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Normalize heterogeneous node features
        if x.size(0) > 1 or not self.training:
            x_norm = self.input_bn(x)
        else:
            x_norm = x

        h = self.node_proj(x_norm)
        # k = 0 initial embedding
        layer_pools = [global_add_pool(h, batch)]

        # Message passing iterations k = 1..n
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr=edge_attr)
            h = self.bns[i](h)
            h = F.relu(h)
            layer_pools.append(global_add_pool(h, batch))

        # h_G = CONCAT( sum_v h_v^(k) for k=0..n ) (Eq. 14)
        h_G = torch.cat(layer_pools, dim=-1)
        return h_G

    def forward(self, x, edge_index, edge_attr=None, batch=None, head=None):
        """
        Forward pass predicting scalar performance proxy y_hat.
        """
        if edge_attr is None:
            # Fallback for empty / missing edge features
            edge_attr = torch.zeros((edge_index.size(1), 4), device=x.device, dtype=x.dtype)

        h_G = self.extract_graph_embedding(x, edge_index, edge_attr, batch)

        if head is not None and head in self.heads:
            out = self.heads[head](h_G)
        else:
            out = self.mlp_head(h_G)

        return out.squeeze(-1)


# Backward-compatible alias
ZeroCostGNNValidator = MultiScaleGINEValidator