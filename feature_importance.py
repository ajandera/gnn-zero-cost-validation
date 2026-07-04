import torch
import numpy as np
from torch_geometric.explain import Explainer, GNNExplainer

def extract_feature_importance(model, test_loader, feature_names):
    """
    Používa GNNExplainer na výpočet dôležitosti jednotlivých vstupných príznakov.
    Priemeruje výsledky cez prvú dávku (batch) testovacích dát pre stabilnejšie výsledky.
    """
    print("\n--- Spúšťam GNNExplainer pre analýzu dôležitosti príznakov ---")
    model.eval()
    
    # Konfigurácia GNNExplaineru pre regresiu na úrovni grafu
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',  # Chceme analyzovať dôležitosť vlastností (features)
        edge_mask_type=None,          # Pre tabuľky vo vašom článku hrany ignorujeme
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )
    
    # Vezmeme prvý batch z testovacieho datasetu
    batch = next(iter(test_loader))
    
    # Generovanie vysvetlenia
    # Upozornenie: Pri grafoch s veľkým množstvom uzlov môže tento krok trvať niekoľko sekúnd
    explanation = explainer(batch.x, batch.edge_index, batch=batch.batch)
    
    # explanation.node_mask má tvar [počet_uzlov, počet_príznakov]
    # Spriemerujeme dôležitosť každého príznaku cez všetky uzly v batchi
    global_importance = explanation.node_mask.mean(dim=0).cpu().numpy()
    
    # Normalizácia do rozsahu 0 - 100% pre lepšiu čitateľnosť
    global_importance = (global_importance / global_importance.max()) * 100
    
    # Priradenie k názvom príznakov a zoradenie
    importance_dict = {name: score for name, score in zip(feature_names, global_importance)}
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Výpis vo formáte pripravenom pre LaTeX tabuľku
    print("\nRelatívna dôležitosť príznakov (Feature Importance):")
    print("-" * 50)
    print(f"{'Príznak (Feature)':<30} | {'Dôležitosť (Skóre)'}")
    print("-" * 50)
    for name, score in sorted_importance:
        # Kategorizácia (High, Medium, Low) na základe skóre
        if score > 66:
            qual = "High"
        elif score > 33:
            qual = "Medium"
        else:
            qual = "Low"
            
        print(f"{name:<30} | {score:5.1f}% ({qual})")
    print("-" * 50)
    
    return sorted_importance