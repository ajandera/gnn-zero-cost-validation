# Generalized Zero-Cost Validation of Neural Networks via Graph-Based Meta-Representation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper:
> **Generalized Zero-Cost Validation of Neural Networks via Graph-Based Meta-Representation**  
> *Ales Jandera, Zuzana Sarosiova, and Tomas Skovranek*

---

## 📌 Abstract

Traditional validation of deep neural networks demands computationally intensive forward passes over downstream datasets, requiring access to labeled test data and substantial compute. This paper introduces a **generalized zero-cost validation approach** using graph-based meta-representations and Graph Neural Networks (GNNs). 

Our method evaluates the prospective quality of a target neural architecture **without requiring any data-driven training, test-set forward passes, or gradient computations**. We encode neural architectures as directed attributed graphs capturing both **structural topology** (e.g., skip paths, recurrence, graph Laplacian spectrum) and **macro-statistical weight footprints** (e.g., stable rank, empirical parameter sparsity, spectral distribution). A multi-scale **Graph Isomorphism Network with Edge features (GINE)** is trained across heterogeneous model families to predict performance metrics such as Word Error Rate (WER) and Top-1 Accuracy in milliseconds.

```
Target Neural Architecture               Graph Representation                    GNN Validator                       Predicted Score ŷ
(RNN, CNN, Transformer, NAS)  ───────►  Builder (X_v, X_e)          ───────►  (Multi-Scale GINE)        ───────►  (Quality / Error Proxy)
                                           ▲                                                                              │
                                           │ offline                                                                      ▼
                                   Perturbation Engine                                                          Accept / Reject / Rank
                                 (Offline Meta-Dataset)
```

---

## 🔬 Theoretical Formulation

### 1. Structural and Statistical Feature Encoding

A target neural network $\mathcal{N}$ is formalized as a directed attributed graph $\mathcal{G} = (V, E, X_v, X_e)$:

#### **Node Attribute Vector $x_i \in \mathbb{R}^{d_v}$ (Eq. 11):**
$$x_i = \left[ e_{type}(v_i), \; d_{arch}(v_i), \; \mu(W_i), \; \sigma^2(W_i), \; srank(W_i), \; sp(W_i), \; h_{reg}(v_i) \right]^T$$

- **Operation Class $e_{type}(v_i) \in \{0, 1\}^{11}$:** One-hot encoding of layer operation (`Conv`, `Depthwise Conv`, `Linear`, `LSTM`, `GRU`, `Multi-Head Attention`, `FFN`, `LayerNorm`, `BatchNorm`, `Pooling`, `Identity`).
- **Architectural Dimensions $d_{arch}(v_i)$:** Input/output dimensions $(d_{in}, d_{out})$, spatial kernel size $k$, attention heads $h$, and logarithmic parameter count $\log_{10}(1 + P_i)$.
- **Weight Tensor Footprint:** Empirical mean $\mu(W_i)$ and variance $\sigma^2(W_i)$.
- **Stable Rank $srank(W_i)$ (Eq. 8):**
  $$srank(W_i) = \frac{\|W_i\|_F^2}{\|W_i\|_2^2} = \frac{\sum_{k=1}^r \sigma_k^2}{\sigma_1^2}$$
  Computed via **truncated power iteration** (Eq. 9) within $T = 3\text{--}5$ iterations with linear complexity $\mathcal{O}(T \cdot d_{in} d_{out})$ without incurring full SVD complexity.
- **Explicit Weight Sparsity $sp(W_i)$ (Eq. 10):**
  $$sp(W_i) = 1 - \frac{\|W_i\|_0}{d_{out} \times d_{in}}$$
- **Hyperparameters & Regularization $h_{reg}(v_i)$:** Activation function encoding (ReLU, GELU, SiLU, Tanh), dropout rate $p_{drop}$, and normalization running statistics.

#### **Edge Attribute Vector $e_{u, v_i} \in \mathbb{R}^4$ (Eq. 12):**
$$e_{u, v_i} = \left[ t_{dim}, \; \delta_{skip}, \; \delta_{stride}, \; \delta_{recur} \right]^T$$
Capturing dimension transition ratio $t_{dim}$, residual/skip flags $\delta_{skip}$, subsampling strides $\delta_{stride}$, and recurrent feedback pathways $\delta_{recur}$.

---

### 2. Multi-Scale GINE Message Passing Architecture

The validator employs a Multi-Scale Graph Isomorphism Network with Edge features (GINE) (Eq. 13):
$$h_{v_i}^{(k)} = \text{MLP}^{(k)} \left( (1 + \epsilon^{(k)}) h_{v_i}^{(k-1)} + \sum_{u \in \mathcal{N}(v_i)} \text{ReLU}\left(h_u^{(k-1)} + W_e^{(k)} e_{u, v_i}\right) \right)$$

Information across all structural scales is integrated via global readout concatenation (Eq. 14):
$$h_G = \text{CONCAT}\left( \sum_{v_i \in V} h_{v_i}^{(k)} \;\Bigg|\; k = 0, 1, \dots, n \right)$$

And passed through regression head $\Psi$ (Eq. 15) to predict the scalar performance proxy $\hat{y} = \Psi(h_G; \theta_\Psi)$.

---

## 📊 Benchmark Results

### Table III: Performance on RNN Speech Architectures (LibriSpeech clean subset, WER)
| Method | MSE $\downarrow$ | Spearman $\rho$ $\uparrow$ | Kendall $\tau$ $\uparrow$ | Pairwise Acc. (%) | Top-5 Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GNN Validator (ours)** | **0.018** | **0.91** | **0.73** | **87%** | **100%** |
| Jacobian Norm Proxy* | 0.054 | 0.63 | 0.46 | 71% | 80% |
| NTK Condition Number* | 0.061 | 0.58 | 0.42 | 68% | 72% |
| Random Baseline | 0.095 | 0.02 | 0.01 | 50% | 40% |

*\*Zero-cost proxy scores are linearly calibrated via OLS on $\mathcal{D}_{train}$ (Eq. 5).*

---

### Table V: Performance on CNN Architectures (CIFAR-10 Top-1 Accuracy)
| Method | MSE $\downarrow$ | Spearman $\rho$ $\uparrow$ | Kendall $\tau$ $\uparrow$ | Pairwise Acc. (%) | Top-5 Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GNN Validator (ours)** | **0.021** | **0.88** | **0.70** | **85%** | **98%** |
| Jacobian Norm Proxy* | 0.057 | 0.61 | 0.44 | 72% | 82% |
| NTK Condition Number* | 0.064 | 0.59 | 0.41 | 68% | 79% |
| Random Baseline | 0.096 | 0.03 | 0.02 | 50% | 40% |

---

### Table VII: Performance on Transformer Architectures (GLUE / ViT)
| Method | MSE $\downarrow$ | Spearman $\rho$ $\uparrow$ | Kendall $\tau$ $\uparrow$ | Pairwise Acc. (%) | Top-5 Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GNN Validator (ours)** | **0.026** | **0.85** | **0.67** | **83%** | **96%** |
| Jacobian Norm Proxy* | 0.061 | 0.58 | 0.41 | 70% | 81% |
| NTK Condition Number* | 0.068 | 0.56 | 0.39 | 67% | 77% |
| Random Baseline | 0.097 | 0.04 | 0.02 | 51% | 43% |

---

### Table IX: Performance Comparison on NAS-Bench-201 Benchmark ($N=100$ training budget, $<0.65\%$ of search space)
| Method | CIFAR-10 ($\rho$ / $\tau$ / Test Acc) | CIFAR-100 ($\rho$ / $\tau$ / Test Acc) | ImageNet-16-120 ($\rho$ / $\tau$ / Test Acc) |
| :--- | :---: | :---: | :---: |
| Random Search | 0.00 / 0.00 / 93.48% | 0.00 / 0.00 / 70.62% | 0.00 / 0.00 / 44.32% |
| GradNorm | 0.58 / 0.42 / 92.68% | 0.56 / 0.40 / 68.32% | 0.53 / 0.38 / 43.15% |
| SNIP | 0.61 / 0.44 / 92.89% | 0.58 / 0.42 / 69.11% | 0.55 / 0.39 / 43.80% |
| TE-NAS (NTK) | 0.65 / 0.47 / 93.90% | 0.63 / 0.45 / 71.24% | 0.60 / 0.42 / 45.43% |
| SynFlow | 0.74 / 0.54 / 93.61% | 0.72 / 0.52 / 71.40% | 0.70 / 0.50 / 45.10% |
| NASWOT (Jacobian) | 0.77 / 0.58 / 93.64% | 0.76 / 0.57 / 71.38% | 0.72 / 0.53 / 45.20% |
| **GNN Validator (Ours, $N=100$)** | **0.86 / 0.70 / 94.18%** | **0.84 / 0.68 / 73.22%** | **0.81 / 0.65 / 46.85%** |
| *Global Optimum (Ceiling)* | *1.00 / 1.00 / 94.37%* | *1.00 / 1.00 / 73.51%* | *1.00 / 1.00 / 47.31%* |

---

### Table X: Structural Optimization, Coarsening & Compression Pareto Dynamics
| Optimization Technique | FLOPs | Prediction MSE | Primary Paradigm |
| :--- | :---: | :---: | :--- |
| Weight Sharing & Quantization | 28,000 | 0.034 | Parameter-level |
| Redundancy Removal | 29,000 | 0.036 | Heuristic Pruning |
| Structured Pruning | 30,000 | 0.032 | Topological Pruning |
| **Neuron Clustering (Graph Coarsening)** | **32,000** | **0.030** | **Graph Coarsening** |
| Activation Sparsity | 33,000 | 0.031 | Functional Sparsity |
| Post-Training Quantization | 35,000 | 0.035 | Numerical Precision |
| Layer Dependency Minimization | 36,000 | 0.033 | Connectivity Pruning |
| Low-Rank Factorization | 38,000 | 0.031 | Spectral Decomposition |
| Knowledge Distillation | 40,000 | 0.029 | Student Transfer |
| Graph Partitioning | 42,000 | 0.028 | Sub-graph Modularity |
| Neural Architecture Search (NAS) | 45,000 | 0.027 | Topology Search |

---

## 🚀 Quick Start

### 1. Installation

Clone repository and install dependencies:
```bash
git clone https://github.com/ajandera/gnn-zero-cost-validation.git
cd gnn-zero-cost-validation

# Recommended using uv or venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Reproduce All Experiments

To run the complete suite across all paradigms (RNN, CNN, Transformer, NAS-Bench-201, and Compression Pareto study) and generate all publication PDF figures:
```bash
python main.py --paradigm all
```

### 3. Evaluate Individual Neural Network Paradigms

- **RNN Speech Recognition Validation (Table III, Figures 2–4):**
  ```bash
  python main.py --paradigm rnn --epochs 30 --save_plots --run_explainer
  ```

- **CNN Vision Architecture Validation (Table V, Figures 5–6):**
  ```bash
  python main.py --paradigm cnn --epochs 30 --save_plots --run_explainer
  ```

- **Transformer Architecture Validation (Table VII, Figures 7–8):**
  ```bash
  python main.py --paradigm transformer --epochs 30 --save_plots --run_explainer
  ```

- **NAS-Bench-201 Cell Search Benchmark (Table IX, $N=100$ budget):**
  ```bash
  python main.py --paradigm nasbench201 --epochs 30
  ```

- **Complexity vs. Predictive Fidelity Pareto Study (Table X, Figure 9):**
  ```bash
  python main.py --paradigm compression --save_plots
  ```

---

## 📁 Repository Structure

```
gnn-zero-cost-validation/
├── dataset_generator.py      # Systematic perturbation engines for RNN, CNN, Transformer & NAS
├── graph_builder.py          # Unified structural & statistical feature extractor (Eq. 6-12)
├── gnn_validator.py          # Multi-Scale GINEValidator network (Eq. 13-15)
├── baselines.py              # Zero-cost proxies (NASWOT, TE-NAS, SynFlow, SNIP) & OLS calibration
├── metrics.py                # Validation metrics: MSE, Spearman ρ, Kendall τ, Pairwise Acc, Top-5 Recall
├── feature_importance.py     # GNNExplainer attribution and qualitative ranking (Tables IV, VI, VIII)
├── visualization.py          # Publication-grade plotting for Figures 2, 3, 5, 6, 7, 8, 9
├── main.py                   # Master CLI experiment runner reproducing manuscript benchmarks
├── requirements.txt          # Python dependencies
└── README.md                 # Complete documentation
```

---

## 📖 Citation

If you use this codebase or method in your research, please cite our paper:

```bibtex
@article{jandera2026generalized,
  title={Generalized Zero-Cost Validation of Neural Networks via Graph-Based Meta-Representation},
  author={Jandera, Ales and Sarosiova, Zuzana and Skovranek, Tomas},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
