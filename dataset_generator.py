import torch
import random
from graph_builder import build_model_graph

def generate_perturbed_models(num_samples=100):
    """
    Generates a dataset of graphs by applying random perturbations to a base model.
    Assigns a synthetic 'performance score' based on the degradation.
    """
    dataset = []
    
    for _ in range(num_samples):
        # Base architecture (e.g., simple feed-forward/CNN)
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 10)
        )
        
        base_accuracy = 0.90
        degradation = 0.0
        
        # Apply random weight noise perturbation
        for param in model.parameters():
            if len(param.shape) > 1 and random.random() > 0.5:
                noise_std = random.uniform(0.01, 0.2)
                param.data += torch.randn_like(param) * noise_std
                degradation += noise_std * 0.5 # Higher noise = worse performance
                
        # Calculate simulated true performance
        simulated_accuracy = max(0.1, base_accuracy - degradation)
        
        # Convert to graph
        graph_data = build_model_graph(model, performance_score=simulated_accuracy)
        dataset.append(graph_data)
        
    return dataset