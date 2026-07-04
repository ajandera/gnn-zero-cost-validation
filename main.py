import torch
from torch_geometric.loader import DataLoader
from scipy.stats import spearmanr
import numpy as np

from dataset_generator import generate_perturbed_models
from gnn_validator import ZeroCostGNNValidator

def train():
    print("1. Generating Perturbation Graph Dataset...")
    dataset = generate_perturbed_models(num_samples=300)
    
    # Train/Test Split (80/20)
    train_size = int(0.8 * len(dataset))
    train_loader = DataLoader(dataset[:train_size], batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset[train_size:], batch_size=32, shuffle=False)
    
    print("2. Initializing GNN Validator...")
    # 7 input features: [4 one-hot types, mean, var, sparsity]
    model = ZeroCostGNNValidator(num_node_features=7, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.MSELoss()
    
    print("3. Training Model...")
    model.train()
    for epoch in range(1, 51):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train MSE: {total_loss / len(train_loader.dataset):.4f}")
            
    print("4. Evaluating on Test Set...")
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            preds.extend(out.cpu().numpy())
            trues.extend(batch.y.cpu().numpy())
            
    mse = np.mean((np.array(preds) - np.array(trues))**2)
    spearman_rho, _ = spearmanr(preds, trues)
    
    print("\n--- Final Results ---")
    print(f"Test MSE:           {mse:.4f}")
    print(f"Spearman Corr (ρ):  {spearman_rho:.4f}")
    
    # --- NEW: VISUALIZATION STEP ---
    print("\n5. Generating Plots...")
    plot_scatter(preds, trues, title="Predicted vs True Accuracy", filename="figure_scatter.pdf")
    plot_ranking(preds, trues, title="Model Ranking Correlation", filename="figure_ranking.pdf")

if __name__ == "__main__":
    train()