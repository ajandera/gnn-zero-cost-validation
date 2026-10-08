"""
Evaluation metrics for Generalized Zero-Cost Validation.
Implements:
  - Mean Squared Error (MSE)
  - Spearman rank correlation (rho)
  - Kendall rank correlation (tau)
  - Pairwise Ranking Accuracy (%)
  - Top-k Recall (%)
as formalized in Sections III and IV of the manuscript.
"""

import numpy as np
from scipy.stats import spearmanr, kendalltau


def compute_mse(preds, trues):
    """Computes Mean Squared Error between predicted and true scores."""
    preds = np.asarray(preds).flatten()
    trues = np.asarray(trues).flatten()
    return float(np.mean((preds - trues) ** 2))


def compute_spearman_rho(preds, trues):
    """Computes Spearman rank correlation coefficient (rho)."""
    preds = np.asarray(preds).flatten()
    trues = np.asarray(trues).flatten()
    rho, _ = spearmanr(preds, trues)
    return float(rho) if not np.isnan(rho) else 0.0


def compute_kendall_tau(preds, trues):
    """Computes Kendall rank correlation coefficient (tau)."""
    preds = np.asarray(preds).flatten()
    trues = np.asarray(trues).flatten()
    tau, _ = kendalltau(preds, trues)
    return float(tau) if not np.isnan(tau) else 0.0


def compute_pairwise_accuracy(preds, trues):
    """
    Computes Pairwise Ranking Accuracy (%):
    Fraction of candidate pairs (i, j) where the predicted relative
    ordering matches the ground-truth ordering.
    """
    preds = np.asarray(preds).flatten()
    trues = np.asarray(trues).flatten()
    n = len(trues)
    if n < 2:
        return 100.0

    concordant = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            true_diff = trues[i] - trues[j]
            pred_diff = preds[i] - preds[j]
            if true_diff != 0:
                total_pairs += 1
                if (true_diff > 0 and pred_diff > 0) or (true_diff < 0 and pred_diff < 0):
                    concordant += 1
                elif pred_diff == 0:
                    concordant += 0.5

    return (concordant / total_pairs * 100.0) if total_pairs > 0 else 0.0


def compute_top_k_recall(preds, trues, k=5, higher_is_better=True):
    """
    Computes Top-k Recall (%):
    Percentage of true top-k models that are included in the predicted top-k set.
    """
    preds = np.asarray(preds).flatten()
    trues = np.asarray(trues).flatten()
    n = len(trues)
    k = min(k, n)
    if k == 0:
        return 0.0

    if higher_is_better:
        true_top_k = set(np.argsort(trues)[-k:])
        pred_top_k = set(np.argsort(preds)[-k:])
    else:
        true_top_k = set(np.argsort(trues)[:k])
        pred_top_k = set(np.argsort(preds)[:k])

    intersection = len(true_top_k.intersection(pred_top_k))
    return (intersection / k) * 100.0


def evaluate_all_metrics(preds, trues, higher_is_better=True, k=5):
    """
    Computes the complete suite of validation metrics reported in the paper:
    MSE, Spearman rho, Kendall tau, Pairwise Accuracy, and Top-k Recall.
    """
    return {
        "mse": compute_mse(preds, trues),
        "spearman_rho": compute_spearman_rho(preds, trues),
        "kendall_tau": compute_kendall_tau(preds, trues),
        "pairwise_acc": compute_pairwise_accuracy(preds, trues),
        "top_k_recall": compute_top_k_recall(preds, trues, k=k, higher_is_better=higher_is_better),
    }
