import pytest
import networkx as nx
from datetime import datetime
from typing import Dict, Any

from src.routing.edge_cost import EdgeId, EdgeCost
from src.routing.edge_cost_store import EdgeCostStore
from src.routing.feature_resolver import FeatureResolver, RoutingContext, WeatherContext, SafetyHistory, ResolvedEdgeFeatures
from src.routing.prediction_orchestrator import PredictionOrchestrator, EdgeCostPredictionResult
from src.routing.dynamic_cost_generator import DynamicCostGenerator, GenerationResult
from src.routing.contracts import TrafficPrediction, SafetyPrediction, CombinedEdgeCost

class FakeFeatureResolver(FeatureResolver):
    def __init__(self, should_fail_for=None):
        super().__init__(nx.MultiDiGraph())
        self.should_fail_for = should_fail_for or set()
        self.last_context = None

    def resolve(self, edge_id: EdgeId, context: RoutingContext) -> ResolvedEdgeFeatures:
        self.last_context = context
        if edge_id in self.should_fail_for:
            raise ValueError(f"Forced feature resolution failure for {edge_id}")
            
        return ResolvedEdgeFeatures(
            edge_id=edge_id,
            traffic_features={"mock": 1},
            safety_features={"mock": 2}
        )

class FakePredictionOrchestrator:
    def __init__(self, should_fail_for=None):
        self.should_fail_for = should_fail_for or set()

    def predict_edge(
        self,
        edge_id: EdgeId,
        graph: nx.MultiDiGraph,
        traffic_features: Dict[str, Any],
        safety_features: Dict[str, Any]
    ) -> EdgeCostPredictionResult:
        if edge_id in self.should_fail_for:
            raise RuntimeError(f"Forced prediction failure for {edge_id}")
            
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
                combined_cost=0.7
            )
        )

@pytest.fixture
def mock_graph():
    G = nx.MultiDiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(1, 2, key=0, length=100.0, highway='residential')
    G.add_edge(1, 2, key=1, length=120.0, highway='arterial')
    G.add_edge(2, 3, key=0, length=150.0, highway='residential')
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

def test_single_candidate(mock_graph, routing_context):
    store = EdgeCostStore()
    resolver = FakeFeatureResolver()
    orchestrator = FakePredictionOrchestrator()
    
    generator = DynamicCostGenerator(mock_graph, resolver, orchestrator, store)
    
    edge_id = EdgeId(1, 2, 0)
    gen_time = datetime(2023, 10, 1, 12, 5, 0)
    
    result = generator.generate([edge_id], routing_context, generation_time=gen_time)
    
    assert result.processed_count == 1
    assert result.successful_edges == [edge_id]
    assert len(result.failed_edges) == 0
    
    # Verify Store
    assert store.contains(edge_id)
    cost = store.get(edge_id)
    assert cost.edge_id == edge_id
    assert cost.generated_at == gen_time
    assert cost.cost_data.combined_cost == 0.7

def test_multiple_candidates(mock_graph, routing_context):
    store = EdgeCostStore()
    resolver = FakeFeatureResolver()
    orchestrator = FakePredictionOrchestrator()
    generator = DynamicCostGenerator(mock_graph, resolver, orchestrator, store)
    
    candidates = [EdgeId(1, 2, 0), EdgeId(1, 2, 1), EdgeId(2, 3, 0)]
    result = generator.generate(candidates, routing_context)
    
    assert result.processed_count == 3
    assert len(result.successful_edges) == 3
    
    for c in candidates:
        assert store.contains(c)

def test_parallel_edges(mock_graph, routing_context):
    """Test 3 - Parallel edges independently processed."""
    store = EdgeCostStore()
    resolver = FakeFeatureResolver()
    orchestrator = FakePredictionOrchestrator()
    generator = DynamicCostGenerator(mock_graph, resolver, orchestrator, store)
    
    e1 = EdgeId(1, 2, 0)
    e2 = EdgeId(1, 2, 1)
    
    result = generator.generate([e1, e2], routing_context)
    
    assert store.contains(e1)
    assert store.contains(e2)
    assert store.get(e1) is not store.get(e2) # Distinct EdgeCost objects
    assert store.get(e1).edge_id == e1
    assert store.get(e2).edge_id == e2

def test_existing_store_replacement(mock_graph, routing_context):
    """Test 4 - Store replacement semantics."""
    store = EdgeCostStore()
    e1 = EdgeId(1, 2, 0)
    old_cost = EdgeCost(
        edge_id=e1,
        cost_data=CombinedEdgeCost(
            edge_id=str(e1),
            predicted_travel_time=10.0,
            raw_safety_risk=20.0,
            normalized_travel_time=0.1,
            normalized_safety_risk=0.1,
            travel_time_weight=0.5,
            safety_weight=0.5,
            combined_cost=0.2
        ),
        generated_at=datetime(2023, 1, 1)
    )
    store.set(e1, old_cost)
    
    resolver = FakeFeatureResolver()
    orchestrator = FakePredictionOrchestrator()
    generator = DynamicCostGenerator(mock_graph, resolver, orchestrator, store)
    
    gen_time = datetime(2023, 2, 2)
    generator.generate([e1], routing_context, generation_time=gen_time)
    
    new_cost = store.get(e1)
    assert new_cost is not old_cost
    assert new_cost.generated_at == gen_time
    assert new_cost.cost_data.combined_cost == 0.7

def test_missing_edge_feature_resolution_failure(mock_graph, routing_context):
    """Test 5 - Feature resolution failure gracefully handled (Partial Success)."""
    e1 = EdgeId(1, 2, 0)
    e2 = EdgeId(1, 2, 1)
    
    store = EdgeCostStore()
    resolver = FakeFeatureResolver(should_fail_for={e1})
    orchestrator = FakePredictionOrchestrator()
    generator = DynamicCostGenerator(mock_graph, resolver, orchestrator, store)
    
    result = generator.generate([e1, e2], routing_context)
    
    assert result.processed_count == 2
    assert e1 in result.failed_edges
    assert e2 in result.successful_edges
    assert e1 in result.errors
    assert "Forced feature resolution failure" in result.errors[e1]
    
    assert not store.contains(e1)
    assert store.contains(e2)

def test_prediction_failure(mock_graph, routing_context):
    """Test 6 - Prediction orchestrator failure."""
    e1 = EdgeId(1, 2, 0)
    e2 = EdgeId(1, 2, 1)
    
    store = EdgeCostStore()
    resolver = FakeFeatureResolver()
    orchestrator = FakePredictionOrchestrator(should_fail_for={e2})
    generator = DynamicCostGenerator(mock_graph, resolver, orchestrator, store)
    
    result = generator.generate([e1, e2], routing_context)
    
    assert e2 in result.failed_edges
    assert "Forced prediction failure" in result.errors[e2]
    
    assert store.contains(e1)
    assert not store.contains(e2)

def test_empty_candidate_set(mock_graph, routing_context):
    """Test 8 - Empty candidate set."""
    store = EdgeCostStore()
    generator = DynamicCostGenerator(mock_graph, FakeFeatureResolver(), FakePredictionOrchestrator(), store)
    
    result = generator.generate([], routing_context)
    assert result.processed_count == 0
    assert len(result.successful_edges) == 0

def test_context_propagation(mock_graph, routing_context):
    """Test 9 - Context propagation."""
    resolver = FakeFeatureResolver()
    generator = DynamicCostGenerator(mock_graph, resolver, FakePredictionOrchestrator(), EdgeCostStore())
    
    generator.generate([EdgeId(1, 2, 0)], routing_context)
    assert resolver.last_context is routing_context

def test_graph_immutability(mock_graph, routing_context):
    """Test 11 - Graph immutability."""
    import copy
    graph_snapshot = copy.deepcopy(mock_graph)
    
    store = EdgeCostStore()
    generator = DynamicCostGenerator(mock_graph, FakeFeatureResolver(), FakePredictionOrchestrator(), store)
    
    generator.generate([EdgeId(1, 2, 0)], routing_context)
    
    # Assert nodes and edges perfectly match
    assert list(mock_graph.nodes(data=True)) == list(graph_snapshot.nodes(data=True))
    assert list(mock_graph.edges(data=True, keys=True)) == list(graph_snapshot.edges(data=True, keys=True))
