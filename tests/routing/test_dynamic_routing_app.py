import pytest
import networkx as nx
from datetime import datetime
from typing import Dict, Any

from src.routing.edge_cost import EdgeId, EdgeCost
from src.routing.feature_resolver import FeatureResolver, RoutingContext, WeatherContext, ResolvedEdgeFeatures
from src.routing.prediction_orchestrator import PredictionOrchestrator, EdgeCostPredictionResult
from src.routing.contracts import TrafficPrediction, SafetyPrediction, CombinedEdgeCost
from src.routing.candidate_selector import CandidateEdgeSelector
from src.routing.dynamic_routing_app import DynamicRoutingApp
from src.routing.exceptions import RouteNotFoundError

# ---------------------------------------------------------
# Mocks
# ---------------------------------------------------------

class FakeFeatureResolver(FeatureResolver):
    def __init__(self, should_fail_for=None):
        super().__init__(nx.MultiDiGraph())
        self.should_fail_for = should_fail_for or set()

    def resolve(self, edge_id: EdgeId, context: RoutingContext) -> ResolvedEdgeFeatures:
        if edge_id in self.should_fail_for:
            raise ValueError(f"Forced feature resolution failure for {edge_id}")
            
        return ResolvedEdgeFeatures(
            edge_id=edge_id,
            traffic_features={"mock": 1},
            safety_features={"mock": 2}
        )

class FakePredictionOrchestrator(PredictionOrchestrator):
    def __init__(self, costs: Dict[EdgeId, float], should_fail_for=None):
        self.costs = costs
        self.should_fail_for = should_fail_for or set()
        self.predict_calls = 0

    def predict_edge(
        self,
        edge_id: EdgeId,
        graph: nx.MultiDiGraph,
        traffic_features: Dict[str, Any],
        safety_features: Dict[str, Any]
    ) -> EdgeCostPredictionResult:
        self.predict_calls += 1
        
        if edge_id in self.should_fail_for:
            raise RuntimeError(f"Forced prediction failure for {edge_id}")
            
        # Default to a valid combined cost (1.0) if not explicitly mocked
        cost = self.costs.get(edge_id, 1.0)
        
        return EdgeCostPredictionResult(
            edge_id=edge_id,
            traffic_prediction=TrafficPrediction(predicted_travel_time=10.0),
            safety_prediction=SafetyPrediction(risk_score=20.0),
            travel_time_minutes=10.0,
            combined_cost=CombinedEdgeCost(
                edge_id=str(edge_id),
                predicted_travel_time=10.0,
                raw_safety_risk=20.0,
                normalized_travel_time=0.5,
                normalized_safety_risk=0.2,
                travel_time_weight=0.7,
                safety_weight=0.3,
                combined_cost=cost
            )
        )

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def mock_graph():
    G = nx.MultiDiGraph()
    G.add_node(1, x=0.0, y=0.0)
    G.add_node(2, x=0.0, y=0.1)
    G.add_node(3, x=0.1, y=0.1)
    G.add_node(4, x=0.1, y=0.0)
    
    # Topology: 1 -> 2 -> 3 -> 4 is optimal if costs are low
    # Decoy edges: 1 -> 3, 1 -> 4, etc.
    G.add_edge(1, 2, key=0, length=100.0)
    G.add_edge(2, 3, key=0, length=100.0)
    G.add_edge(3, 4, key=0, length=100.0)
    G.add_edge(1, 3, key=0, length=100.0)
    G.add_edge(1, 4, key=0, length=100.0)
    G.add_edge(2, 4, key=0, length=100.0)
    # Complete the graph for full matrix evaluation (BruteForceSequenceOptimizer requires valid permutations)
    G.add_edge(3, 2, key=0, length=100.0)
    G.add_edge(4, 3, key=0, length=100.0)
    G.add_edge(4, 2, key=0, length=100.0)
    
    return G

@pytest.fixture
def routing_context():
    weather = WeatherContext(
        temperature_c=30.0,
        rain_1h_mm=0.0,
        wind_speed_kmh=10.0,
        visibility_m=10000.0,
        humidity_pct=60.0,
        rain_risk_score=0.1,
        rain_forecast_hours=0
    )
    return RoutingContext(
        timestamp=datetime(2023, 10, 1, 12, 0, 0),
        weather=weather,
        congestion_index=1.0,
        safety_history=None
    )

# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

def test_full_end_to_end_routing(mock_graph, routing_context):
    # Test 14 - End-to-end multi-stop flow
    # Test 1 - RoutingContext creation/passing
    # Test 2 - Candidate generation
    # Test 3 - Dynamic generation
    
    costs = {
        EdgeId(1, 2, 0): 0.1,
        EdgeId(2, 3, 0): 0.1,
        EdgeId(3, 4, 0): 0.1,
        # Decoys are 1.0 by default
    }
    
    resolver = FakeFeatureResolver()
    orchestrator = FakePredictionOrchestrator(costs)
    selector = CandidateEdgeSelector(mock_graph, buffer_m=500000.0) # Huge buffer to include all
    
    app = DynamicRoutingApp(mock_graph, resolver, orchestrator, selector)
    
    result = app.route(1, (3, 2, 4), routing_context)
    
    assert result.success is True
    # The optimal sequence should be 1 -> 2 -> 3 -> 4
    assert result.delivery_sequence.sequence == (2, 3, 4)
    assert len(result.routes) == 3
    assert pytest.approx(result.total_cost) == 0.3
    
    # Test 15 - Dynamic generation count
    # Predict should be called exactly once per candidate edge in the bounding box
    # For this tiny graph, 9 edges total.
    assert orchestrator.predict_calls == 9

def test_request_isolation(mock_graph, routing_context):
    # Test 4 - Request-scoped store
    # Test 8 - Request isolation
    
    costs_A = {
        EdgeId(1, 2, 0): 0.1,
        EdgeId(2, 3, 0): 0.1,
    }
    
    costs_B = {
        EdgeId(1, 2, 0): 1.0, # Much worse for request B
        EdgeId(1, 3, 0): 0.1,  # B prefers this
        EdgeId(3, 2, 0): 0.1,  # B goes 1 -> 3 -> 2
    }
    
    resolver = FakeFeatureResolver()
    orchestrator_A = FakePredictionOrchestrator(costs_A)
    orchestrator_B = FakePredictionOrchestrator(costs_B)
    selector = CandidateEdgeSelector(mock_graph, buffer_m=500000.0)
    
    app_A = DynamicRoutingApp(mock_graph, resolver, orchestrator_A, selector)
    app_B = DynamicRoutingApp(mock_graph, resolver, orchestrator_B, selector)
    
    # Run request A
    result_A = app_A.route(1, (2, 3), routing_context)
    assert result_A.delivery_sequence.sequence == (2, 3)
    assert pytest.approx(result_A.total_cost) == 0.2
    
    # Run request B (even though it's a different app instance, the isolation is what matters.
    # In reality, they could share the same app instance!)
    app_shared = DynamicRoutingApp(mock_graph, resolver, orchestrator_B, selector)
    result_B = app_shared.route(1, (2, 3), routing_context)
    assert result_B.delivery_sequence.sequence == (3, 2)
    assert pytest.approx(result_B.total_cost) == 0.2

def test_missing_costs_and_unreachable_pairs(mock_graph, routing_context):
    # Test 10 - Missing costs
    # Test 11 - Unreachable pair
    
    # Fail prediction for edge 1->2. Thus 1->2 is missing.
    should_fail = {EdgeId(1, 2, 0)}
    resolver = FakeFeatureResolver()
    orchestrator = FakePredictionOrchestrator(costs={}, should_fail_for=should_fail)
    selector = CandidateEdgeSelector(mock_graph, buffer_m=500000.0)
    
    app = DynamicRoutingApp(mock_graph, resolver, orchestrator, selector)
    
    # Without 1->2, and with all others defaulting to 1.0:
    # 1 -> 3 -> 2 will cost 2.0
    result = app.route(1, (2, 3), routing_context)
    
    assert result.success is True
    # It still finds a route, but it must use 1->3->2 (since 1->2 is dead)
    assert pytest.approx(result.total_cost) == 2.0
    assert result.delivery_sequence.sequence == (3, 2)

def test_parallel_edges_preservation(mock_graph, routing_context):
    # Test 9 - Parallel edges
    mock_graph.add_edge(1, 2, key=1, length=100.0)
    
    costs = {
        EdgeId(1, 2, 0): 1.0,
        EdgeId(1, 2, 1): 0.1, # Second parallel edge is much cheaper
    }
    
    resolver = FakeFeatureResolver()
    orchestrator = FakePredictionOrchestrator(costs)
    selector = CandidateEdgeSelector(mock_graph, buffer_m=500000.0)
    app = DynamicRoutingApp(mock_graph, resolver, orchestrator, selector)
    
    result = app.route(1, (2,), routing_context)
    
    assert result.success is True
    assert pytest.approx(result.total_cost) == 0.1
    
    # Verify the specific parallel edge was chosen
    assert len(result.routes) == 1
    path_edges = result.routes[0].path_edges
    assert len(path_edges) == 1
    u, v, key = path_edges[0]
    assert u == 1 and v == 2 and key == 1

def test_graph_immutability(mock_graph, routing_context):
    # Test 13 - Graph immutability
    import copy
    graph_snapshot = copy.deepcopy(mock_graph)
    
    resolver = FakeFeatureResolver()
    orchestrator = FakePredictionOrchestrator({})
    selector = CandidateEdgeSelector(mock_graph, buffer_m=500000.0)
    app = DynamicRoutingApp(mock_graph, resolver, orchestrator, selector)
    
    app.route(1, (2, 3), routing_context)
    
    assert list(mock_graph.nodes(data=True)) == list(graph_snapshot.nodes(data=True))
    assert list(mock_graph.edges(data=True, keys=True)) == list(graph_snapshot.edges(data=True, keys=True))
