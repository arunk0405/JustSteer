import os
import sys

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from graph.osm_loader import download_road_network

def main():
    place = "Chennai, Tamil Nadu, India"
    print(f"Downloading {place} road network...")
    
    try:
        graph = download_road_network(place)
        
        print("\nGraph type:", type(graph))
        print("Nodes:", graph.number_of_nodes())
        print("Edges:", graph.number_of_edges())
        
        # Sample node
        node_id = list(graph.nodes())[0]
        node_data = graph.nodes[node_id]
        print("\nSample node:")
        print(f"ID: {node_id}")
        print(f"x: {node_data.get('x')}")
        print(f"y: {node_data.get('y')}")
        
        # Sample edge
        # Find an edge that has geometry if possible, or just the first one
        u, v, key, edge_data = list(graph.edges(data=True, keys=True))[0]
        print("\nSample edge:")
        print(f"source: {u}")
        print(f"target: {v}")
        print(f"key: {key}")
        print(f"length: {edge_data.get('length')}")
        print(f"highway: {edge_data.get('highway')}")
        print(f"oneway: {edge_data.get('oneway')}")
        print(f"geometry: {edge_data.get('geometry')}")
        
        # Step 3: Validate the downloaded graph
        print("\nRunning graph validation...")
        from graph.graph_validator import validate_road_network, RoadNetworkValidationError
        from graph.osm_loader import save_road_network, load_road_network
        import time
        from pathlib import Path
        
        try:
            validate_road_network(graph)
            print("\nRoad network validation: PASS")
            print(f"Graph type: {type(graph).__name__}")
            print(f"Nodes: {graph.number_of_nodes()}")
            print(f"Edges: {graph.number_of_edges()}")
            
            # Step 4: Save and Load
            save_path = Path("data/road_network/chennai_drive.graphml")
            print(f"\nSaving graph to {save_path}...")
            start_save = time.time()
            save_road_network(graph, save_path, overwrite=True)
            save_time = time.time() - start_save
            print(f"Save complete in {save_time:.2f} seconds. File size: {save_path.stat().st_size / (1024*1024):.2f} MB")
            
            print(f"\nLoading graph from {save_path} (without hitting OSM)...")
            start_load = time.time()
            loaded_graph = load_road_network(save_path)
            load_time = time.time() - start_load
            print(f"Load and validation complete in {load_time:.2f} seconds.")
            
            print(f"\nReloaded graph verification:")
            print(f"Graph type: {type(loaded_graph).__name__}")
            print(f"Nodes: {loaded_graph.number_of_nodes()} (Original: {graph.number_of_nodes()})")
            print(f"Edges: {loaded_graph.number_of_edges()} (Original: {graph.number_of_edges()})")
            
            if graph.number_of_nodes() == loaded_graph.number_of_nodes() and graph.number_of_edges() == loaded_graph.number_of_edges():
                print("Count verification: PASS")
            else:
                print("Count verification: FAIL")
            
        except RoadNetworkValidationError as ve:
            print(f"\nRoad network validation: FAIL")
            print(f"Error: {ve}")
            
    except Exception as e:
        print(f"\nError during download or execution: {e}")

if __name__ == "__main__":
    main()
