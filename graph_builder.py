import torch
import numpy as np
from torch_geometric.data import Data

def extract_layer_statistics(weight_tensor):
    """
    Extracts statistical features from a layer's weight tensor.
    Approximates the features defined in the manuscript: [mean, variance, sparsity].
    """
    if weight_tensor is None or weight_tensor.numel() == 0:
        return [0.0, 0.0, 0.0]
    
    mean = weight_tensor.mean().item()
    var = weight_tensor.var().item() if weight_tensor.numel() > 1 else 0.0
    
    # Calculate sparsity (percentage of weights close to zero)
    threshold = 1e-5
    zeros = torch.sum(torch.abs(weight_tensor) < threshold).item()
    sparsity = zeros / weight_tensor.numel()
    
    # Note: srank (stable rank) calculation via SVD is computationally heavy.
    # For a full implementation, you would add: (torch.norm(W, p='fro')**2) / (torch.norm(W, p=2)**2)
    return [mean, var, sparsity]

def build_model_graph(model, performance_score=None):
    """
    Converts a PyTorch model into a PyTorch Geometric Data object.
    (Simplified sequential assumption for demonstration).
    """
    node_features = []
    edges_src = []
    edges_dst = []
    
    layers = list(model.modules())[1:] # Skip the container itself
    
    for i, layer in enumerate(layers):
        # 1. Layer Type (Simplified One-Hot Encoding: [Linear, Conv, RNN, Other])
        l_type = [0, 0, 0, 0]
        if isinstance(layer, torch.nn.Linear): l_type[0] = 1
        elif isinstance(layer, torch.nn.Conv2d): l_type[1] = 1
        elif isinstance(layer, (torch.nn.LSTM, torch.nn.GRU)): l_type[2] = 1
        else: l_type[3] = 1
            
        # 2. Extract weights if parameter-based layer
        weights = layer.weight if hasattr(layer, 'weight') else None
        stats = extract_layer_statistics(weights)
        
        # Combine features: [type_0, type_1, type_2, type_3, mean, var, sparsity]
        node_features.append(l_type + stats)
        
        # 3. Build edges (Assuming sequential data flow for this template)
        if i < len(layers) - 1:
            edges_src.append(i)
            edges_dst.append(i + 1)
            
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    
    # Target variable (e.g., WER or Accuracy)
    y = torch.tensor([performance_score], dtype=torch.float) if performance_score else None
    
    return Data(x=x, edge_index=edge_index, y=y)