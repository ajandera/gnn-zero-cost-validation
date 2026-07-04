import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINConv, global_add_pool

class ZeroCostGNNValidator(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=64, num_layers=4):
        super(ZeroCostGNNValidator, self).__init__()
        
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(self._build_gin_layer(num_node_features, hidden_dim))
        self.bns.append(BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(self._build_gin_layer(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm1d(hidden_dim))
            
        # Regression head (Predicts the scalar performance proxy)
        self.mlp_head = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, 1)
        )

    def _build_gin_layer(self, in_dim, out_dim):
        mlp = Sequential(
            Linear(in_dim, out_dim),
            ReLU(),
            Linear(out_dim, out_dim)
        )
        # train_eps=True matches the (1 + epsilon) learnable parameter in your paper
        return GINConv(mlp, train_eps=True) 

    def forward(self, x, edge_index, batch):
        h = x
        
        # Message passing iterations
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            
        # Global readout pooling (aggregates structural states)
        h_graph = global_add_pool(h, batch)
        
        # Final performance proxy regression
        out = self.mlp_head(h_graph)
        return out.squeeze()