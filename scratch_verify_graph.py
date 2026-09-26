import sys
import logging
from src.graph.osm_loader import load_road_network

logging.basicConfig(level=logging.INFO)

def main():
    try:
        graph = load_road_network("data/road_network/chennai.graphml")
    except Exception as e:
        print(f"Failed to load graph: {e}")
        sys.exit(1)
        
    print(f"Graph type: {type(graph)}")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    
    # Check 10 random edges
    import random
    edges = list(graph.edges(keys=True, data=True))
    random.seed(42)
    sample_edges = random.sample(edges, min(10, len(edges)))
    
    missing_length = 0
    negative_length = 0
    non_numeric_length = 0
    
    print("\nSample edges:")
    for u, v, k, data in sample_edges:
        length = data.get("length")
        print(f"({u}, {v}, {k}) -> length: {length} (type: {type(length)})")
        if length is None:
            missing_length += 1
        elif not isinstance(length, (int, float)):
            non_numeric_length += 1
        elif length < 0:
            negative_length += 1
            
    print(f"\nStats over sample: Missing: {missing_length}, Negative: {negative_length}, Non-numeric: {non_numeric_length}")

if __name__ == "__main__":
    main()
