import pytest
import networkx as nx
from datetime import datetime

from src.routing.astar import AStarRoutingEngine
from src.routing.cost_provider import StaticGraphCostProvider, DynamicStoreCostProvider
from src.routing.edge_cost import EdgeId, EdgeCost
from src.routing.contracts import CombinedEdgeCost
from src.routing.edge_cost_store import EdgeCostStore
from src.routing.exceptions import RouteNotFoundError

@pytest.fixture
def test_graph():
    G = nx.MultiDiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_node(4)
    # 1 -> 2 has 3 parallel edges
    G.add_edge(1, 2, key=0, final_cost=10.0)
    G.add_edge(1, 2, key=1, final_cost=20.0)
    G.add_edge(1, 2, key=2, final_cost=30.0)
    # 2 -> 3
    G.add_edge(2, 3, key=0, final_cost=10.0)
    # 1 -> 4
    G.add_edge(1, 4, key=0, final_cost=10.0)
    # 4 -> 3
    G.add_edge(4, 3, key=0, final_cost=10.0)
    return G

def create_mock_cost(edge_id: EdgeId, cost_value: float) -> EdgeCost:
    return EdgeCost(
        edge_id=edge_id,
        cost_data=CombinedEdgeCost(
            edge_id=str(edge_id),
            predicted_travel_time=100.0,
            raw_safety_risk=50.0,
            normalized_travel_time=cost_value,
            normalized_safety_risk=cost_value,
            travel_time_weight=0.5,
            safety_weight=0.5,
            combined_cost=cost_value
        ),
        generated_at=datetime.now(),
        traffic_model_version=None,
        safety_model_version=None
    )

def test_1_dynamic_cost_hit(test_graph):
    """Test 1 — Dynamic cost hit"""
    store = EdgeCostStore()
    edge_id = EdgeId(1, 2, 0)
    store.set(edge_id, create_mock_cost(edge_id, 0.5))
    
    provider = DynamicStoreCostProvider(store)
    cost = provider.get_cost(edge_id)
    assert cost == 0.5

def test_2_dynamic_cost_missing(test_graph):
    """Test 2 — Dynamic cost missing"""
    store = EdgeCostStore()
    edge_id = EdgeId(1, 2, 0)
    # Cost is missing
    provider = DynamicStoreCostProvider(store)
    cost = provider.get_cost(edge_id)
    assert cost is None
    
    # AStar skips edge if missing cost
    engine = AStarRoutingEngine(test_graph, cost_provider=provider)
    with pytest.raises(RouteNotFoundError):
        engine.route(1, 2)

def test_3_all_parallel_edges_missing(test_graph):
    """Test 3 — All parallel edges missing"""
    store = EdgeCostStore()
    provider = DynamicStoreCostProvider(store)
    engine = AStarRoutingEngine(test_graph, cost_provider=provider)
    
    # Neither key 0, 1, or 2 have costs for 1 -> 2
    with pytest.raises(RouteNotFoundError):
        engine.route(1, 2)

def test_4_partial_parallel_edge_availability(test_graph):
    """Test 4 — Partial parallel-edge availability"""
    store = EdgeCostStore()
    # key 0 is missing
    # key 1 is 0.25
    # key 2 is 0.75
    store.set(EdgeId(1, 2, 1), create_mock_cost(EdgeId(1, 2, 1), 0.25))
    store.set(EdgeId(1, 2, 2), create_mock_cost(EdgeId(1, 2, 2), 0.75))
    
    provider = DynamicStoreCostProvider(store)
    engine = AStarRoutingEngine(test_graph, cost_provider=provider)
    
    result = engine.route(1, 2)
    assert result.success is True
    assert result.total_cost == 0.25
    assert result.path_edges[0][2] == 1  # best_edge_key is 1

def test_5_zero_dynamic_cost(test_graph):
    """Test 5 — Zero dynamic cost"""
    store = EdgeCostStore()
    edge_id = EdgeId(1, 2, 0)
    store.set(edge_id, create_mock_cost(edge_id, 0.0))
    
    provider = DynamicStoreCostProvider(store)
    engine = AStarRoutingEngine(test_graph, cost_provider=provider)
    
    result = engine.route(1, 2)
    assert result.total_cost == 0.0

def test_6_invalid_dynamic_cost(test_graph):
    """Test 6 — Invalid dynamic cost"""
    store = EdgeCostStore()
    edge_id_1 = EdgeId(1, 2, 0)
    edge_id_2 = EdgeId(1, 2, 1)
    edge_id_3 = EdgeId(1, 2, 2)
    
    store.set(edge_id_1, create_mock_cost(edge_id_1, -0.1))
    store.set(edge_id_2, create_mock_cost(edge_id_2, 1.5))
    store.set(edge_id_3, create_mock_cost(edge_id_3, float('nan')))
    
    provider = DynamicStoreCostProvider(store)
    engine = AStarRoutingEngine(test_graph, cost_provider=provider)
    
    # All are invalid, route should not be found
    with pytest.raises(RouteNotFoundError):
        engine.route(1, 2)

def test_7_static_provider(test_graph):
    """Test 7 — Static provider"""
    provider = StaticGraphCostProvider(test_graph, "final_cost")
    engine = AStarRoutingEngine(test_graph, cost_provider=provider)
    
    result = engine.route(1, 2)
    assert result.total_cost == 10.0
    assert result.path_edges[0][2] == 0

def test_8_provider_injection(test_graph):
    """Test 8 — Provider injection"""
    class MockProvider:
        def get_cost(self, edge_id: EdgeId) -> float:
            return 42.0
            
    engine = AStarRoutingEngine(test_graph, cost_provider=MockProvider())
    result = engine.route(1, 2)
    assert result.total_cost == 42.0

def test_9_dynamic_route(test_graph):
    """Test 9 — Dynamic route overrides static route"""
    store = EdgeCostStore()
    # Staticaly: 1->2->3 costs 20. 1->4->3 costs 20.
    # Dynamically: 1->2 (0.9), 2->3 (0.9) = 1.8
    # 1->4 (0.1), 4->3 (0.1) = 0.2
    store.set(EdgeId(1, 2, 0), create_mock_cost(EdgeId(1, 2, 0), 0.9))
    store.set(EdgeId(2, 3, 0), create_mock_cost(EdgeId(2, 3, 0), 0.9))
    store.set(EdgeId(1, 4, 0), create_mock_cost(EdgeId(1, 4, 0), 0.1))
    store.set(EdgeId(4, 3, 0), create_mock_cost(EdgeId(4, 3, 0), 0.1))
    
    provider = DynamicStoreCostProvider(store)
    engine = AStarRoutingEngine(test_graph, cost_provider=provider)
    
    result = engine.route(1, 3)
    assert result.path_nodes == [1, 4, 3]
    assert result.total_cost == 0.2

def test_10_missing_cost_alternative_route(test_graph):
    """Test 10 — Missing-cost alternative route"""
    store = EdgeCostStore()
    # 1->2 missing
    store.set(EdgeId(1, 4, 0), create_mock_cost(EdgeId(1, 4, 0), 0.1))
    store.set(EdgeId(4, 3, 0), create_mock_cost(EdgeId(4, 3, 0), 0.1))
    
    provider = DynamicStoreCostProvider(store)
    engine = AStarRoutingEngine(test_graph, cost_provider=provider)
    
    result = engine.route(1, 3)
    assert result.path_nodes == [1, 4, 3]
    assert result.total_cost == 0.2

def test_11_no_dynamically_feasible_route(test_graph):
    """Test 11 — No dynamically feasible route"""
    store = EdgeCostStore()
    # 1->2->3 missing, 1->4 missing, 4->3 (0.1)
    store.set(EdgeId(4, 3, 0), create_mock_cost(EdgeId(4, 3, 0), 0.1))
    
    provider = DynamicStoreCostProvider(store)
    engine = AStarRoutingEngine(test_graph, cost_provider=provider)
    
    with pytest.raises(RouteNotFoundError):
        engine.route(1, 3)

def test_12_request_isolation(test_graph):
    """Test 12 — Request isolation"""
    store_a = EdgeCostStore()
    store_a.set(EdgeId(1, 2, 0), create_mock_cost(EdgeId(1, 2, 0), 0.1))
    
    store_b = EdgeCostStore()
    store_b.set(EdgeId(1, 2, 0), create_mock_cost(EdgeId(1, 2, 0), 0.9))
    
    provider_a = DynamicStoreCostProvider(store_a)
    engine_a = AStarRoutingEngine(test_graph, cost_provider=provider_a)
    result_a = engine_a.route(1, 2)
    
    provider_b = DynamicStoreCostProvider(store_b)
    engine_b = AStarRoutingEngine(test_graph, cost_provider=provider_b)
    result_b = engine_b.route(1, 2)
    
    assert result_a.total_cost == 0.1
    assert result_b.total_cost == 0.9

def test_13_parallel_edge_reconstruction(test_graph):
    """Test 13 — Parallel-edge reconstruction"""
    store = EdgeCostStore()
    store.set(EdgeId(1, 2, 0), create_mock_cost(EdgeId(1, 2, 0), 0.8))
    store.set(EdgeId(1, 2, 1), create_mock_cost(EdgeId(1, 2, 1), 0.3))
    
    provider = DynamicStoreCostProvider(store)
    engine = AStarRoutingEngine(test_graph, cost_provider=provider)
    
    result = engine.route(1, 2)
    assert result.path_edges[0][2] == 1

def test_14_graph_immutability(test_graph):
    """Test 14 — Graph immutability"""
    import copy
    graph_before = copy.deepcopy(test_graph)
    
    store = EdgeCostStore()
    store.set(EdgeId(1, 2, 0), create_mock_cost(EdgeId(1, 2, 0), 0.5))
    
    provider = DynamicStoreCostProvider(store)
    engine = AStarRoutingEngine(test_graph, cost_provider=provider)
    engine.route(1, 2)
    
    assert nx.is_isomorphic(test_graph, graph_before, edge_match=lambda e1, e2: e1 == e2)
    assert list(test_graph.edges(keys=True, data=True)) == list(graph_before.edges(keys=True, data=True))
