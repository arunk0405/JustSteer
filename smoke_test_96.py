import os
import networkx as nx
from datetime import datetime

from src.routing.astar import AStarRoutingEngine
from src.routing.cost_provider import DynamicStoreCostProvider, StaticGraphCostProvider
from src.routing.edge_cost import EdgeId, EdgeCost
from src.routing.contracts import CombinedEdgeCost
from src.routing.edge_cost_store import EdgeCostStore
from src.routing.exceptions import RouteNotFoundError

def run_smoke_test():
    print("Loading production graph...")
    graph_path = "data/road_network/chennai_drive.graphml"
    
    # Load graph and ensure nodes are integers (as done in Step 9.5 smoke test)
    G_raw = nx.read_graphml(graph_path)
    # Convert node labels to integers
    mapping = {node: int(node) for node in G_raw.nodes()}
    G = nx.relabel_nodes(G_raw, mapping)
    
    # Take a snapshot of a few edges to ensure immutability
    sample_edges = list(G.edges(keys=True, data=True))[:10]
    
    print(f"Graph loaded. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # 1. Test Static Fallback (Legacy)
    # The production graph has 'length' instead of 'final_cost', but we can use 'length'
    # just for the smoke test to prove the static provider reads the graph.
    print("Running static route test...")
    static_provider = StaticGraphCostProvider(G, cost_attribute="length")
    engine_static = AStarRoutingEngine(G, cost_provider=static_provider)
    
    # Find a valid edge to route across
    u, v, key = list(G.edges(keys=True))[0]
    res_static = engine_static.route(u, v)
    print(f"Static Route success: {res_static.success}, cost: {res_static.total_cost}")
    assert res_static.success
    
    # 2. Test Dynamic Provider
    print("Running dynamic route test...")
    store = EdgeCostStore()
    edge_id = EdgeId(source=u, target=v, key=key)
    
    store.set(edge_id, EdgeCost(
        edge_id=edge_id,
        cost_data=CombinedEdgeCost(
            edge_id=str(edge_id),
            predicted_travel_time=100.0,
            raw_safety_risk=50.0,
            normalized_travel_time=0.4,
            normalized_safety_risk=0.4,
            travel_time_weight=0.5,
            safety_weight=0.5,
            combined_cost=0.4
        ),
        generated_at=datetime.now(),
        traffic_model_version=None,
        safety_model_version=None
    ))
    
    dynamic_provider = DynamicStoreCostProvider(store)
    engine_dynamic = AStarRoutingEngine(G, cost_provider=dynamic_provider)
    
    res_dynamic = engine_dynamic.route(u, v)
    print(f"Dynamic Route success: {res_dynamic.success}, cost: {res_dynamic.total_cost}")
    assert res_dynamic.success
    assert res_dynamic.total_cost == 0.4
    
    # 3. Test Missing Cost (Edge unavailable)
    print("Running missing cost test...")
    # Find a neighbor of v, say w
    successors = list(G.successors(v))
    if successors:
        w = successors[0]
        try:
            engine_dynamic.route(v, w)
            print("ERROR: Should have raised RouteNotFoundError")
            assert False
        except RouteNotFoundError:
            print("Missing cost correctly skipped.")
    
    # 4. Verify Immutability
    print("Verifying graph immutability...")
    current_sample_edges = list(G.edges(keys=True, data=True))[:10]
    assert sample_edges == current_sample_edges
    print("Graph is unchanged.")
    
    print("Smoke test PASSED.")

if __name__ == "__main__":
    run_smoke_test()
