import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(preds, trues, title="Predicted vs True Performance", filename="scatter_plot.pdf"):
    """
    Generates a scatter plot of predicted vs true values with an ideal line,
    matching the style of Figures 1, 3, and 5 in the manuscript.
    """
    plt.figure(figsize=(6, 5))
    
    # Scatter points
    plt.scatter(preds, trues, color='blue', label='Predictions', alpha=0.7)
    
    # Ideal y=x line
    min_val = min(min(preds), min(trues))
    max_val = max(max(preds), max(trues))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', linewidth=1, label='Ideal Line')
    
    plt.xlabel('Predicted Value')
    plt.ylabel('True Value')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved scatter plot to {filename}")

def plot_ranking(preds, trues, title="Ranking Correlation", filename="ranking_plot.pdf"):
    """
    Generates a line plot showing the predicted vs true ranking of models,
    matching the style of Figures 2, 4, and 6 in the manuscript.
    """
    # Sort both arrays based on the TRUE values to create a ranking order
    sorted_indices = np.argsort(trues)
    trues_sorted = np.array(trues)[sorted_indices]
    preds_sorted = np.array(preds)[sorted_indices]
    
    x_axis = np.arange(1, len(trues) + 1)
    
    plt.figure(figsize=(6, 5))
    
    # Plot sorted true values and the corresponding predictions
    plt.plot(x_axis, preds_sorted, color='blue', linestyle='-', label='Predicted', linewidth=1.5)
    plt.plot(x_axis, trues_sorted, color='red', linestyle='--', label='True', linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Model Index (sorted by true performance)')
    plt.ylabel('Performance Metric')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved ranking plot to {filename}")