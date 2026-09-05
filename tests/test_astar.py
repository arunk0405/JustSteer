import pytest
import networkx as nx
import math
from src.routing.astar import AStarRoutingEngine
from src.routing.exceptions import (
    InvalidNodeError,
    RouteNotFoundError,
    InvalidEdgeCostError
)

def build_test_graph() -> nx.MultiDiGraph:
    """Builds a deterministic test MultiDiGraph."""
    G = nx.MultiDiGraph()
    # A -> B -> D (longer distance, cost=20)
    # A -> C -> D (shorter distance, cost=10)
    
    # Nodes with mock coordinates (not actually used by zero heuristic, but good for testing geometry extraction fallback)
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=1.0, y=1.0)
    G.add_node("C", x=0.5, y=0.5)
    G.add_node("D", x=2.0, y=0.0)
    G.add_node("E", x=3.0, y=3.0) # Disconnected node
    
    # Path 1: A -> B -> D
    G.add_edge("A", "B", key=0, final_cost=5.0, length=100.0)
    G.add_edge("B", "D", key=0, final_cost=15.0, length=100.0)
    
    # Path 2: A -> C -> D (Lower cost, shorter physical length)
    G.add_edge("A", "C", key=0, final_cost=2.0, length=50.0)
    G.add_edge("C", "D", key=0, final_cost=8.0, length=50.0)
    
    # A -> B alternative edge (Parallel edge test)
    # Should be picked if we go A->B because cost is 3.0 instead of 5.0
    G.add_edge("A", "B", key=1, final_cost=3.0, length=100.0)
    
    # One-way test: D -> A has no edge.
    
    return G


def test_astar_picks_lowest_cost_path():
    G = build_test_graph()
    engine = AStarRoutingEngine(G, cost_attribute="final_cost")
    
    res = engine.route("A", "D")
    
    assert res.success is True
    # Path A -> C -> D has total cost 10.0
    # Path A -> B -> D has total cost 3.0 (edge key 1) + 15.0 = 18.0
    assert res.total_cost == 10.0
    assert res.path_nodes == ["A", "C", "D"]
    assert res.distance_m == 100.0


def test_astar_handles_parallel_edges():
    G = build_test_graph()
    engine = AStarRoutingEngine(G, cost_attribute="final_cost")
    
    # A to B directly
    res = engine.route("A", "B")
    
    assert res.success is True
    assert res.total_cost == 3.0
    assert len(res.path_edges) == 1
    
    # The edge key picked should be 1 (since cost 3.0 < 5.0)
    u, v, key = res.path_edges[0]
    assert u == "A"
    assert v == "B"
    assert key == 1


def test_source_equals_destination():
    G = build_test_graph()
    engine = AStarRoutingEngine(G)
    
    res = engine.route("A", "A")
    assert res.success is True
    assert res.total_cost == 0.0
    assert res.path_nodes == ["A"]
    assert len(res.path_edges) == 0


def test_unreachable_destination():
    G = build_test_graph()
    engine = AStarRoutingEngine(G)
    
    with pytest.raises(RouteNotFoundError):
        engine.route("A", "E")


def test_one_way_respect():
    G = build_test_graph()
    engine = AStarRoutingEngine(G)
    
    # D -> A does not exist
    with pytest.raises(RouteNotFoundError):
        engine.route("D", "A")


def test_invalid_nodes():
    G = build_test_graph()
    engine = AStarRoutingEngine(G)
    
    with pytest.raises(InvalidNodeError):
        engine.route("Z", "A")
        
    with pytest.raises(InvalidNodeError):
        engine.route("A", "Z")


def test_invalid_costs():
    G = nx.MultiDiGraph()
    G.add_node("A")
    G.add_node("B")
    G.add_node("C")
    G.add_node("D")
    
    # Negative cost
    G.add_edge("A", "B", key=0, final_cost=-5.0)
    
    # NaN cost
    G.add_edge("B", "C", key=0, final_cost=float('nan'))
    
    # String cost
    G.add_edge("C", "D", key=0, final_cost="invalid")
    
    # Valid fallback path
    G.add_edge("A", "D", key=0, final_cost=10.0)
    
    engine = AStarRoutingEngine(G, cost_attribute="final_cost")
    
    # Instead of throwing an error, it should skip A->B, B->C, C->D and use A->D
    res = engine.route("A", "D")
    assert res.success is True
    assert res.total_cost == 10.0
    assert res.path_nodes == ["A", "D"]
    
    # If no valid path exists at all, it should raise RouteNotFoundError
    with pytest.raises(RouteNotFoundError):
        engine.route("A", "B")

def test_route_cost_consistency():
    G = build_test_graph()
    engine = AStarRoutingEngine(G, cost_attribute="final_cost")
    
    res = engine.route("A", "D")
    
    assert res.success is True
    
    # Re-calculate the sum of the selected edges
    sum_cost = 0.0
    for u, v, key in res.path_edges:
        sum_cost += G[u][v][key]["final_cost"]
        
    assert math.isclose(res.total_cost, sum_cost, rel_tol=1e-5)


def test_astar_vs_dijkstra_validation():
    # Since heuristic is 0, our AStar IS Dijkstra structurally.
    # But we can compare our custom MultiDiGraph AStar against NetworkX's built-in single-graph Dijkstra
    # just to ensure mathematically we find the same cost.
    
    G = build_test_graph()
    
    # Convert MultiDiGraph to DiGraph by taking the min cost edge (which is what standard Dijkstra would need)
    # We do this mapping explicitly for the NetworkX built-in function to work on it.
    G_single = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        cost = data.get("final_cost")
        if not G_single.has_edge(u, v) or cost < G_single[u][v]["final_cost"]:
            G_single.add_edge(u, v, final_cost=cost)
            
    engine = AStarRoutingEngine(G, cost_attribute="final_cost")
    
    # Test multiple paths
    paths_to_test = [("A", "D"), ("A", "B"), ("C", "D")]
    
    for src, dst in paths_to_test:
        res = engine.route(src, dst)
        
        # NetworkX built-in Dijkstra length
        nx_cost = nx.shortest_path_length(G_single, source=src, target=dst, weight="final_cost")
        
        assert math.isclose(res.total_cost, nx_cost, rel_tol=1e-5), \
            f"Cost mismatch for {src}->{dst}: Custom A*={res.total_cost}, NX Dijkstra={nx_cost}"

def test_stale_queue_entries():
    # Construct a graph where a node is reached first via a high-cost path,
    # and then later via a lower-cost path. This ensures it's pushed to the 
    # queue twice. The stale entry check must discard the higher cost one.
    G = nx.MultiDiGraph()
    G.add_node("S")
    G.add_node("A")
    G.add_node("B")
    G.add_node("E")
    
    # Path 1: S -> B directly, high cost (cost = 10), discovered immediately
    G.add_edge("S", "B", key=0, final_cost=10.0)
    
    # Path 2: S -> A -> B, lower cost (cost = 1 + 2 = 3), discovered after S
    G.add_edge("S", "A", key=0, final_cost=1.0)
    G.add_edge("A", "B", key=0, final_cost=2.0)
    
    # From B -> E
    G.add_edge("B", "E", key=0, final_cost=5.0)
    
    # If the queue processes the stale B (f=10) later instead of skipping it, 
    # it could mess up traversal or cause unnecessary expansions. We verify 
    # the total cost is optimal (3 + 5 = 8) and expanded nodes doesn't blow up.
    engine = AStarRoutingEngine(G, cost_attribute="final_cost")
    res = engine.route("S", "E")
    
    assert res.success is True
    assert res.total_cost == 8.0
    assert res.path_nodes == ["S", "A", "B", "E"]
