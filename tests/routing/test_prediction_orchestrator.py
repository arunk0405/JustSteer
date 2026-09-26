import math
import pytest
import networkx as nx
from unittest.mock import MagicMock

from src.routing.edge_cost import EdgeId
from src.routing.contracts import TrafficPrediction, SafetyPrediction, CombinedEdgeCost
from src.routing.weight_calculator import WeightCalculator, NormalizationConfig
from src.routing.travel_time import InvalidPhysicalQuantityError
from src.routing.prediction_providers import TrafficPredictionProvider, SafetyPredictionProvider
from src.routing.prediction_orchestrator import (
    PredictionOrchestrator,
    FeatureValidationError,
    EdgeCostPredictionResult
)

# --- Test Fixtures ---

@pytest.fixture
def mock_graph():
    G = nx.MultiDiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(1, 2, key=0, length=1000.0)
    G.add_edge(1, 2, key=1, length=2000.0)
    G.add_edge(2, 3, key=0, length=1500.0)
    return G

@pytest.fixture
def base_traffic_features():
    return {f: 1.0 for f in TrafficPredictionProvider.EXPECTED_FEATURES}

@pytest.fixture
def base_safety_features():
    return {f: 1.0 for f in SafetyPredictionProvider.EXPECTED_FEATURES}

@pytest.fixture
def orchestrator():
    traffic_provider = MagicMock(spec=TrafficPredictionProvider)
    traffic_provider.EXPECTED_FEATURES = TrafficPredictionProvider.EXPECTED_FEATURES
    traffic_provider.predict_speed.return_value = 60.0
    
    safety_provider = MagicMock(spec=SafetyPredictionProvider)
    safety_provider.EXPECTED_FEATURES = SafetyPredictionProvider.EXPECTED_FEATURES
    safety_provider.predict.return_value = SafetyPrediction(risk_score=50.0)
    
    config = NormalizationConfig(
        travel_time_p5=1.0, travel_time_p95=5.0,
        travel_time_weight=0.5, safety_weight=0.5
    )
    weight_calc = WeightCalculator(config)
    
    return PredictionOrchestrator(
        traffic_provider=traffic_provider,
        safety_provider=safety_provider,
        weight_calculator=weight_calc
    )


# --- Tests ---

def test_1_valid_features_end_to_end(orchestrator, mock_graph, base_traffic_features, base_safety_features):
    edge_id = EdgeId(1, 2, 0)
    
    result = orchestrator.predict_edge(
        edge_id=edge_id,
        graph=mock_graph,
        traffic_features=base_traffic_features,
        safety_features=base_safety_features
    )
    
    assert isinstance(result, EdgeCostPredictionResult)
    assert result.edge_id == edge_id
    # 1000m @ 60km/h -> 1 minute travel time
    assert math.isclose(result.travel_time_minutes, 1.0, rel_tol=1e-5)
    assert result.traffic_prediction.predicted_travel_time == 1.0
    assert result.safety_prediction.risk_score == 50.0
    assert isinstance(result.combined_cost, CombinedEdgeCost)


def test_2_correct_edge_identity(orchestrator, mock_graph, base_traffic_features, base_safety_features):
    # Test parallel edges are correctly distinguished
    edge_0 = EdgeId(1, 2, 0) # 1000m
    edge_1 = EdgeId(1, 2, 1) # 2000m
    
    res_0 = orchestrator.predict_edge(edge_0, mock_graph, base_traffic_features, base_safety_features)
    assert res_0.edge_id == edge_0
    assert res_0.travel_time_minutes == 1.0
    
    res_1 = orchestrator.predict_edge(edge_1, mock_graph, base_traffic_features, base_safety_features)
    assert res_1.edge_id == edge_1
    assert res_1.travel_time_minutes == 2.0


def test_3_correct_edge_length(orchestrator, mock_graph, base_traffic_features, base_safety_features):
    edge_id = EdgeId(2, 3, 0) # 1500m
    # length = 1500m, speed = 60km/h = 1000m/min -> 1.5 mins
    res = orchestrator.predict_edge(edge_id, mock_graph, base_traffic_features, base_safety_features)
    assert math.isclose(res.travel_time_minutes, 1.5, rel_tol=1e-5)


def test_4_speed_to_time_correctness(orchestrator, mock_graph, base_traffic_features, base_safety_features):
    orchestrator.traffic_provider.predict_speed.return_value = 30.0
    # 1000m @ 30km/h = 500m/min -> 2 mins
    res = orchestrator.predict_edge(EdgeId(1, 2, 0), mock_graph, base_traffic_features, base_safety_features)
    assert math.isclose(res.travel_time_minutes, 2.0, rel_tol=1e-5)


def test_5_safety_mapping(orchestrator, mock_graph, base_traffic_features, base_safety_features):
    orchestrator.safety_provider.predict.return_value = SafetyPrediction(risk_score=80.0)
    res = orchestrator.predict_edge(EdgeId(1, 2, 0), mock_graph, base_traffic_features, base_safety_features)
    assert res.safety_prediction.risk_score == 80.0
    assert res.combined_cost.raw_safety_risk == 80.0


def test_6_missing_traffic_feature(orchestrator, mock_graph, base_traffic_features, base_safety_features):
    del base_traffic_features["temperature_c"]
    with pytest.raises(FeatureValidationError, match="Missing required traffic feature: temperature_c"):
        orchestrator.predict_edge(EdgeId(1, 2, 0), mock_graph, base_traffic_features, base_safety_features)
    orchestrator.traffic_provider.predict_speed.assert_not_called()


def test_7_missing_safety_feature(orchestrator, mock_graph, base_traffic_features, base_safety_features):
    del base_safety_features["lanes"]
    with pytest.raises(FeatureValidationError, match="Missing required safety feature: lanes"):
        orchestrator.predict_edge(EdgeId(1, 2, 0), mock_graph, base_traffic_features, base_safety_features)
    orchestrator.safety_provider.predict.assert_not_called()


def test_8_invalid_numeric_feature(orchestrator, mock_graph, base_traffic_features, base_safety_features):
    # NaN
    base_traffic_features["wind_speed_kmh"] = float('nan')
    with pytest.raises(FeatureValidationError, match="NaN or Infinity"):
        orchestrator.predict_edge(EdgeId(1, 2, 0), mock_graph, base_traffic_features, base_safety_features)
    
    # Infinity
    base_traffic_features["wind_speed_kmh"] = float('inf')
    with pytest.raises(FeatureValidationError, match="NaN or Infinity"):
        orchestrator.predict_edge(EdgeId(1, 2, 0), mock_graph, base_traffic_features, base_safety_features)
        
    # Non-numeric
    base_traffic_features["wind_speed_kmh"] = "fast"
    with pytest.raises(FeatureValidationError, match="invalid type str"):
        orchestrator.predict_edge(EdgeId(1, 2, 0), mock_graph, base_traffic_features, base_safety_features)


def test_9_invalid_traffic_prediction(orchestrator, mock_graph, base_traffic_features, base_safety_features):
    # NaN
    orchestrator.traffic_provider.predict_speed.return_value = float('nan')
    with pytest.raises(InvalidPhysicalQuantityError, match="finite non-negative"):
        orchestrator.predict_edge(EdgeId(1, 2, 0), mock_graph, base_traffic_features, base_safety_features)
        
    # Negative
    orchestrator.traffic_provider.predict_speed.return_value = -10.0
    with pytest.raises(InvalidPhysicalQuantityError, match="finite non-negative"):
        orchestrator.predict_edge(EdgeId(1, 2, 0), mock_graph, base_traffic_features, base_safety_features)


def test_10_zero_speed(orchestrator, mock_graph, base_traffic_features, base_safety_features):
    orchestrator.traffic_provider.predict_speed.return_value = 0.0
    with pytest.raises(InvalidPhysicalQuantityError, match="Predicted speed is zero"):
        orchestrator.predict_edge(EdgeId(1, 2, 0), mock_graph, base_traffic_features, base_safety_features)


def test_11_weight_calculator_invocation(orchestrator, mock_graph, base_traffic_features, base_safety_features):
    orchestrator.weight_calculator.calculate = MagicMock(return_value=CombinedEdgeCost(
        edge_id="EdgeId(source=1, target=2, key=0)", predicted_travel_time=1.0, raw_safety_risk=50.0,
        normalized_travel_time=0.0, normalized_safety_risk=0.5, travel_time_weight=0.5, safety_weight=0.5, combined_cost=0.25
    ))
    
    orchestrator.predict_edge(EdgeId(1, 2, 0), mock_graph, base_traffic_features, base_safety_features)
    
    orchestrator.weight_calculator.calculate.assert_called_once()
    args, kwargs = orchestrator.weight_calculator.calculate.call_args
    assert kwargs["edge_id"] == "EdgeId(source=1, target=2, key=0)"
    assert kwargs["traffic_pred"].predicted_travel_time == 1.0
    assert kwargs["safety_pred"].risk_score == 50.0


def test_12_no_graph_mutation(orchestrator, mock_graph, base_traffic_features, base_safety_features):
    edge_id = EdgeId(1, 2, 0)
    original_data = dict(mock_graph[1][2][0])
    
    orchestrator.predict_edge(edge_id, mock_graph, base_traffic_features, base_safety_features)
    
    post_data = dict(mock_graph[1][2][0])
    assert original_data == post_data
    assert "final_cost" not in post_data
    assert "traffic_prediction" not in post_data
    assert "safety_prediction" not in post_data
    assert "travel_time" not in post_data
