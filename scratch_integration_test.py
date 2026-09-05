import networkx as nx
import time
from src.routing.astar import AStarRoutingEngine

def run_integration_test():
    G = nx.MultiDiGraph()
    
    # Adding nodes with coordinates
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=10.0, y=10.0)
    G.add_node("C", x=20.0, y=0.0)
    
    # Adding parallel edges A->B
    G.add_edge("A", "B", key=0, final_cost=0.5, length=150.0, geometry=[(0,0), (5,5), (10,10)])
    G.add_edge("A", "B", key=1, final_cost=0.2, length=200.0, geometry=[(0,0), (0,10), (10,10)])
    
    # Adding edge B->C
    G.add_edge("B", "C", key=0, final_cost=0.3, length=100.0, geometry=[(10,10), (15,5), (20,0)])
    
    # Initialize Engine
    engine = AStarRoutingEngine(G, cost_attribute="final_cost")
    
    # Route from A to C
    result = engine.route("A", "C")
    
    print("Source:", result.source_node)
    print("Destination:", result.destination_node)
    print("Path nodes:", result.path_nodes)
    print("Selected edge keys:", [k for u,v,k in result.path_edges])
    print("Total cost:", result.total_cost)
    print("Total distance:", result.distance_m)
    print("Expanded nodes:", result.expanded_nodes)
    print("Execution time ms:", result.execution_time_ms)
    print("Geometry:", result.geometry)

if __name__ == "__main__":
    run_integration_test()
