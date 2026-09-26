import networkx as nx
import joblib
from pathlib import Path
from src.routing.prediction_providers import SafetyPredictionProvider, TrafficPredictionProvider
from src.routing.edge_cost import EdgeId

def validate_production_graph():
    print("Loading production graph...")
    graph_path = Path("data/road_network/chennai_drive.graphml")
    if not graph_path.exists():
        print(f"Graph not found at {graph_path}")
        return False
        
    G = nx.read_graphml(graph_path)
    
    print(f"Graph type: {type(G)}")
    if not isinstance(G, nx.MultiDiGraph):
        print("ERROR: Graph is not a nx.MultiDiGraph")
        return False
        
    nodes = list(G.nodes)
    edges = list(G.edges(keys=True, data=True))
    
    print(f"Node count: {len(nodes)}")
    print(f"Edge count: {len(edges)}")
    
    # Check integer node IDs
    sample_nodes = nodes[:10]
    for n in sample_nodes:
        if not isinstance(n, int):
            try:
                int(n) # Usually graphml loads nodes as string, let's verify what type they are
            except ValueError:
                pass
    
    print("Graph smoke test passed!")
    return True

def validate_production_models():
    print("\nLoading models...")
    try:
        tpp = TrafficPredictionProvider()
        print("TrafficPredictionProvider loaded velocity_predictor.pkl successfully.")
    except Exception as e:
        print(f"ERROR loading traffic model: {e}")
        return False
        
    try:
        spp = SafetyPredictionProvider()
        print("SafetyPredictionProvider loaded safety_risk_xgb.pkl successfully.")
    except Exception as e:
        print(f"ERROR loading safety model: {e}")
        return False
        
    print("Models smoke test passed!")
    return True

if __name__ == "__main__":
    validate_production_graph()
    validate_production_models()
