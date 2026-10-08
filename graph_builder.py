"""
Graph Representation Builder for Neural Network Architectures.
Implements the unified structural-statistical feature encoding formalized in Section III:
  - Theoretical stable rank (Eq. 8) via truncated power iteration (Eq. 9)
  - Explicit weight sparsity (Eq. 10)
  - Unified node attribute vector x_i (Eq. 11)
  - Edge attribute vector e_{u, v_i} (Eq. 12)
  - Normalized Graph Laplacian L_norm and spectral gap lambda_2 (Eq. 6-7)
"""

import math
import torch
import numpy as np
from torch_geometric.data import Data

# 11 Operation Classes (e_type)
OPERATION_CLASSES = [
    "conv",              # Standard Convolution (Conv1d, Conv2d)
    "dw_conv",           # Depthwise Separable Convolution
    "linear",            # Dense / Linear
    "lstm",              # LSTM recurrent cell
    "gru",               # GRU recurrent cell
    "mha",               # Multi-Head Attention
    "ffn",               # Feed-Forward Network Block
    "layer_norm",        # Layer Normalization
    "batch_norm",        # Batch Normalization
    "pooling",           # Max/Avg Pooling
    "identity"           # Identity / Skip connection
]

NUM_OP_CLASSES = len(OPERATION_CLASSES)
OP_TO_IDX = {op: i for i, op in enumerate(OPERATION_CLASSES)}

# Feature names for interpretability / GNNExplainer
FEATURE_NAMES = [
    # Operation classes (11)
    "Op: Conv",
    "Op: Depthwise Conv",
    "Op: Linear",
    "Op: LSTM",
    "Op: GRU",
    "Op: Multi-Head Attention",
    "Op: FFN Block",
    "Op: LayerNorm",
    "Op: BatchNorm",
    "Op: Pooling",
    "Op: Identity",
    # Architectural dimensions (5)
    "Input Dimension (d_in)",
    "Output Dimension (d_out)",
    "Kernel Spatial Size (k)",
    "Attention Heads (h)",
    "Layer Parameter Count (log P_i)",
    # Weight statistics (4)
    "Weight Mean (mu)",
    "Weight Variance (sigma^2)",
    "Stable Rank (srank)",
    "Parameter Sparsity (sp)",
    # Hyperparameters & regularization (3)
    "Activation Function Type",
    "Dropout Rate (p_drop)",
    "Normalization Running Stat"
]

NUM_NODE_FEATURES = len(FEATURE_NAMES)
NUM_EDGE_FEATURES = 4  # [t_dim, delta_skip, delta_stride, delta_recur]


def truncated_power_iteration(weight_tensor, num_iters=5):
    """
    Approximates the spectral norm ||W_i||_2 = sigma_1 using truncated power iteration
    with linear complexity O(T * d_in * d_out), matching Eq. (9) of the manuscript.
    """
    if weight_tensor is None or weight_tensor.numel() == 0:
        return 1.0

    W = weight_tensor.detach().float()
    if W.dim() > 2:
        W = W.flatten(1)
    elif W.dim() == 1:
        return float(torch.norm(W, p=2).item())

    d_out, d_in = W.shape
    if d_out == 0 or d_in == 0:
        return 1.0

    # Initialize random vector
    v = torch.randn(d_in, 1, device=W.device, dtype=W.dtype)
    v_norm = torch.norm(v, p=2).clamp(min=1e-8)
    v = v / v_norm

    for _ in range(num_iters):
        # u = W * v
        u = torch.matmul(W, v)
        u_norm = torch.norm(u, p=2).clamp(min=1e-8)
        u = u / u_norm

        # v = W^T * u
        v = torch.matmul(W.t(), u)
        v_norm = torch.norm(v, p=2).clamp(min=1e-8)
        v = v / v_norm

    sigma_1 = torch.norm(torch.matmul(W, v), p=2).item()
    return max(sigma_1, 1e-8)


def compute_stable_rank(weight_tensor, num_iters=5):
    """
    Computes the stable rank srank(W_i) = ||W_i||_F^2 / ||W_i||_2^2 (Eq. 8).
    Acts as a continuous, robust proxy for effective latent dimensionality.
    """
    if weight_tensor is None or weight_tensor.numel() == 0:
        return 1.0

    W = weight_tensor.detach().float()
    fro_norm_sq = float((torch.norm(W, p="fro") ** 2).item())
    sigma_1 = truncated_power_iteration(W, num_iters=num_iters)
    spectral_norm_sq = sigma_1 ** 2

    return fro_norm_sq / (spectral_norm_sq + 1e-8)


def compute_parameter_sparsity(weight_tensor, threshold=1e-5):
    """
    Computes explicit parameter sparsity sp(W_i) = 1 - ||W_i||_0 / (d_out * d_in) (Eq. 10).
    Reflects the extent of structural pruning.
    """
    if weight_tensor is None or weight_tensor.numel() == 0:
        return 0.0

    W = weight_tensor.detach()
    num_zeros = float(torch.sum(torch.abs(W) < threshold).item())
    return num_zeros / float(W.numel())


def extract_weight_statistics(weight_tensor):
    """
    Extracts statistical footprint: [mean, variance, stable_rank, sparsity].
    """
    if weight_tensor is None or weight_tensor.numel() == 0:
        return [0.0, 0.0, 1.0, 0.0]

    W = weight_tensor.detach().float()
    mean = float(W.mean().item())
    var = float(W.var().item()) if W.numel() > 1 else 0.0
    srank = compute_stable_rank(W, num_iters=5)
    sparsity = compute_parameter_sparsity(W)

    return [mean, var, srank, sparsity]


def encode_node_features(
    op_type="linear",
    d_in=128,
    d_out=64,
    kernel_size=1,
    num_heads=1,
    param_count=0,
    weight_tensor=None,
    activation_type=1,  # 0: None, 1: ReLU, 2: GELU, 3: SiLU, 4: Tanh
    dropout_rate=0.0,
    norm_stat=0.0,
):
    """
    Encodes the unified localized feature vector for node v_i (Eq. 11):
    x_i = [e_type(v_i), d_arch(v_i), mu(W_i), sigma^2(W_i), srank(W_i), sp(W_i), h_reg(v_i)]^T
    """
    # 1. Operation class one-hot e_type
    e_type = [0.0] * NUM_OP_CLASSES
    idx = OP_TO_IDX.get(op_type, OP_TO_IDX["linear"])
    e_type[idx] = 1.0

    # 2. Architectural capacity d_arch
    # Scaled logarithmic parameter count and normalized dimensions
    log_params = math.log10(max(1, param_count))
    d_arch = [
        float(d_in) / 512.0,
        float(d_out) / 512.0,
        float(kernel_size),
        float(num_heads),
        log_params / 6.0,
    ]

    # 3. Weight statistics [mu, sigma^2, srank, sp]
    stats = extract_weight_statistics(weight_tensor)

    # 4. Hyperparameters & regularization h_reg
    h_reg = [
        float(activation_type) / 4.0,
        float(dropout_rate),
        float(norm_stat),
    ]

    return e_type + d_arch + stats + h_reg


def encode_edge_attributes(t_dim=1.0, is_skip=0.0, stride=1.0, is_recur=0.0):
    """
    Endows directed edge (u, v_i) with edge attribute vector (Eq. 12):
    e_{u, v_i} = [t_dim, delta_skip, delta_stride, delta_recur]^T
    """
    return [float(t_dim), float(is_skip), float(stride), float(is_recur)]


def compute_laplacian_spectral_gap(edge_index, num_nodes):
    """
    Computes normalized graph Laplacian L_norm = I - D^{-1/2} A D^{-1/2} (Eq. 6)
    and algebraic connectivity / spectral gap lambda_2 (Eq. 7).
    """
    if num_nodes <= 1:
        return 0.0

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    for s, d in zip(src, dst):
        adj[s, d] = 1.0
        adj[d, s] = 1.0  # undirected connectivity for Laplacian spectrum

    deg = np.sum(adj, axis=1)
    with np.errstate(divide="ignore"):
        deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0

    D_mat = np.diag(deg_inv_sqrt)
    L_norm = np.eye(num_nodes) - D_mat @ adj @ D_mat
    eigvals = np.sort(np.linalg.eigvalsh(L_norm))

    spectral_gap = float(eigvals[1]) if len(eigvals) > 1 else 0.0
    return max(0.0, spectral_gap)


def build_model_graph(model, performance_score=None, task_type="regression"):
    """
    Inspects a PyTorch model and converts it into an attributed graph Data object
    with full node features (Eq. 11) and edge attributes (Eq. 12).
    """
    node_features = []
    edges_src = []
    edges_dst = []
    edge_attributes = []

    layers = [m for m in model.modules() if len(list(m.children())) == 0]
    if len(layers) == 0:
        layers = [model]

    dims = []
    for i, layer in enumerate(layers):
        op_type = "linear"
        d_in, d_out = 64, 64
        kernel_size = 1
        num_heads = 1
        act_type = 0
        dropout_p = 0.0
        norm_stat = 0.0

        weight = getattr(layer, "weight", None)
        param_count = sum(p.numel() for p in layer.parameters()) if hasattr(layer, "parameters") else 0

        if isinstance(layer, torch.nn.Linear):
            op_type = "linear"
            d_in, d_out = layer.in_features, layer.out_features
        elif isinstance(layer, torch.nn.Conv2d):
            op_type = "dw_conv" if (layer.groups == layer.in_channels and layer.groups > 1) else "conv"
            d_in, d_out = layer.in_channels, layer.out_channels
            kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        elif isinstance(layer, torch.nn.LSTM):
            op_type = "lstm"
            d_in, d_out = layer.input_size, layer.hidden_size
        elif isinstance(layer, torch.nn.GRU):
            op_type = "gru"
            d_in, d_out = layer.input_size, layer.hidden_size
        elif isinstance(layer, torch.nn.MultiheadAttention):
            op_type = "mha"
            d_in, d_out = layer.embed_dim, layer.embed_dim
            num_heads = layer.num_heads
        elif isinstance(layer, torch.nn.LayerNorm):
            op_type = "layer_norm"
            d_in = layer.normalized_shape[0] if isinstance(layer.normalized_shape, tuple) else layer.normalized_shape
            d_out = d_in
        elif isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            op_type = "batch_norm"
            d_in, d_out = layer.num_features, layer.num_features
            if layer.running_mean is not None:
                norm_stat = float(layer.running_mean.abs().mean().item())
        elif isinstance(layer, (torch.nn.ReLU, torch.nn.ReLU6)):
            op_type = "identity"
            act_type = 1
        elif isinstance(layer, torch.nn.GELU):
            op_type = "identity"
            act_type = 2
        elif isinstance(layer, torch.nn.SiLU):
            op_type = "identity"
            act_type = 3
        elif isinstance(layer, (torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.AdaptiveAvgPool2d)):
            op_type = "pooling"
        elif isinstance(layer, torch.nn.Dropout):
            op_type = "identity"
            dropout_p = layer.p

        dims.append((d_in, d_out))
        node_feats = encode_node_features(
            op_type=op_type,
            d_in=d_in,
            d_out=d_out,
            kernel_size=kernel_size,
            num_heads=num_heads,
            param_count=param_count,
            weight_tensor=weight,
            activation_type=act_type,
            dropout_rate=dropout_p,
            norm_stat=norm_stat,
        )
        node_features.append(node_feats)

    # Build sequential and skip edge pathways
    num_nodes = len(layers)
    for i in range(num_nodes - 1):
        edges_src.append(i)
        edges_dst.append(i + 1)
        prev_dout = dims[i][1]
        next_din = dims[i + 1][0]
        tdim = float(prev_dout) / max(1.0, float(next_din))
        edge_attributes.append(encode_edge_attributes(t_dim=tdim, is_skip=0.0, stride=1.0, is_recur=0.0))

    x = torch.tensor(node_features, dtype=torch.float)
    if len(edges_src) > 0:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_EDGE_FEATURES), dtype=torch.float)

    y = torch.tensor([performance_score], dtype=torch.float) if performance_score is not None else None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_raw_graph(node_list, edge_list, performance_score=None):
    """
    Constructs a PyG Data object directly from explicit node and edge definitions.
    Used for DAG cells (e.g. NAS-Bench-201) and custom architectures.
    
    Each node in node_list is a dict of kwargs passed to encode_node_features.
    Each edge in edge_list is a tuple: (u, v, t_dim, is_skip, stride, is_recur).
    """
    node_features = [encode_node_features(**kwargs) for kwargs in node_list]
    edges_src = []
    edges_dst = []
    edge_attributes = []

    for edge in edge_list:
        u, v = edge[0], edge[1]
        t_dim = edge[2] if len(edge) > 2 else 1.0
        is_skip = edge[3] if len(edge) > 3 else 0.0
        stride = edge[4] if len(edge) > 4 else 1.0
        is_recur = edge[5] if len(edge) > 5 else 0.0

        edges_src.append(u)
        edges_dst.append(v)
        edge_attributes.append(encode_edge_attributes(t_dim=t_dim, is_skip=is_skip, stride=stride, is_recur=is_recur))

    x = torch.tensor(node_features, dtype=torch.float)
    if len(edges_src) > 0:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_EDGE_FEATURES), dtype=torch.float)

    y = torch.tensor([performance_score], dtype=torch.float) if performance_score is not None else None
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)