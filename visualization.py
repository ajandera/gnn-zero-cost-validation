"""
Visualization Module for Generalized Zero-Cost Validation.
Generates publication-quality figures matching the manuscript:
  - Figures 2, 5, 7: Scatter plots (Predicted vs True Performance with ideal y=x line)
  - Figures 3, 6, 8: Model Ranking trajectories (Sorted by ground truth)
  - Figure 9: Complexity (FLOPs) vs Predictive Fidelity (MSE) Pareto Curve
  - Feature Importance attributions (matching Tables IV, VI, VIII)
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_scatter(
    preds,
    trues,
    title="Predicted vs True Performance",
    filename="figure_scatter.pdf",
    xlabel="Predicted Performance",
    ylabel="True Performance",
):
    """
    Generates a scatter plot of predicted vs true values with the ideal line,
    matching the exact style of Figures 2, 5, and 7 in the manuscript.
    """
    preds = np.asarray(preds).flatten()
    trues = np.asarray(trues).flatten()
    n_models = len(preds)

    plt.figure(figsize=(6, 5))
    plt.scatter(
        preds,
        trues,
        color="#1f77b4",
        edgecolors="navy",
        alpha=0.75,
        s=35,
        label=f"Predictions (N= {n_models})",
    )

    min_val = min(float(np.min(preds)), float(np.min(trues)))
    max_val = max(float(np.max(preds)), float(np.max(trues)))
    padding = (max_val - min_val) * 0.05
    line_start = min_val - padding
    line_end = max_val + padding

    plt.plot(
        [line_start, line_end],
        [line_start, line_end],
        color="#d62728",
        linestyle="-",
        linewidth=1.8,
        label="Ideal Line",
    )

    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved scatter plot to {filename}")


def plot_ranking(
    preds,
    trues,
    title="Model Ranking Correlation",
    filename="figure_ranking.pdf",
    xlabel="Model Index (sorted by true performance)",
    ylabel="Performance Metric",
    rho=None,
):
    """
    Generates a line plot showing predicted vs true ranking trajectories,
    matching the exact style of Figures 3, 6, and 8 in the manuscript.
    """
    preds = np.asarray(preds).flatten()
    trues = np.asarray(trues).flatten()

    sorted_indices = np.argsort(trues)
    trues_sorted = trues[sorted_indices]
    preds_sorted = preds[sorted_indices]
    x_axis = np.arange(1, len(trues) + 1)

    plt.figure(figsize=(6, 5))
    pred_label = f"Predicted (rho= {rho:.2f})" if rho is not None else "Predicted"
    plt.plot(x_axis, preds_sorted, color="#1f77b4", linestyle="-", linewidth=1.8, label=pred_label)
    plt.plot(x_axis, trues_sorted, color="#d62728", linestyle="--", linewidth=1.8, alpha=0.85, label="True")

    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved ranking plot to {filename}")


def plot_pareto_tradeoff(records, filename="figure_pareto.pdf"):
    """
    Generates Figure 9: Validation error (MSE) versus computational complexity (FLOPs)
    across structural optimization, graph coarsening, and compression techniques.
    """
    plt.figure(figsize=(7.5, 5.5))

    flops = [r["flops"] for r in records]
    mses = [r["mse"] for r in records]
    names = [r["technique"] for r in records]

    # Color highlights
    colors = []
    for r in records:
        if "Coarsening" in r["technique"] or "Clustering" in r["technique"] or "Partitioning" in r["technique"]:
            colors.append("#2ca02c")  # Green for graph-aware
        elif "Quantization" in r["technique"] or "Redundancy" in r["technique"]:
            colors.append("#d62728")  # Red for localized perturbations
        else:
            colors.append("#1f77b4")  # Blue standard

    plt.scatter(flops, mses, c=colors, s=70, edgecolors="black", zorder=3)

    # Annotate key techniques
    for r in records:
        name = r["technique"]
        x, y = r["flops"], r["mse"]
        offset_y = 0.0004
        offset_x = -500
        if "Clustering" in name:
            plt.annotate("Neuron Clustering (Coarsening)", (x, y), textcoords="offset points", xytext=(-60, -14),
                         fontweight="bold", color="#1b631b", fontsize=9)
        elif "Redundancy" in name:
            plt.annotate("Redundancy Removal", (x, y), textcoords="offset points", xytext=(-40, 8),
                         fontweight="bold", color="#941616", fontsize=9)
        elif "Partitioning" in name:
            plt.annotate("Graph Partitioning", (x, y), textcoords="offset points", xytext=(-40, -14),
                         color="#1b631b", fontsize=8.5)
        elif "NAS" in name:
            plt.annotate("NAS (Topology Search)", (x, y), textcoords="offset points", xytext=(-60, 8), fontsize=8.5)
        elif "Sharing" in name:
            plt.annotate("Weight Sharing & Quant.", (x, y), textcoords="offset points", xytext=(-20, 8), fontsize=8.5)

    plt.xlabel("Computational Complexity (FLOPs)", fontsize=11)
    plt.ylabel("Prediction MSE", fontsize=11)
    plt.title("Complexity vs. Predictive Fidelity Across Compression Paradigms", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved Pareto trade-off plot to {filename}")


def plot_feature_importance_bar(sorted_importance, title="Relative Feature Importance", filename="figure_features.pdf"):
    """
    Generates horizontal bar chart of relative feature importance scores.
    """
    # Top 10 features
    top_items = sorted_importance[:10]
    names = [item[0] for item in reversed(top_items)]
    scores = [item[1] for item in reversed(top_items)]

    plt.figure(figsize=(7, 4.5))
    bars = plt.barh(names, scores, color="#1f77b4", edgecolor="navy", alpha=0.85)

    for bar, score in zip(bars, scores):
        plt.text(score + 1.0, bar.get_y() + bar.get_height() / 2, f"{score:.1f}%",
                 va="center", ha="left", fontsize=9)

    plt.xlim(0, max(scores) * 1.15)
    plt.xlabel("Relative Importance (%)", fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to {filename}")