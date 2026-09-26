import math
import pytest
import networkx as nx

from src.routing.edge_cost import EdgeId
from src.routing.contracts import TrafficPrediction, SafetyPrediction, CombinedEdgeCost
from src.routing.weight_calculator import WeightCalculator, NormalizationConfig
from src.routing.travel_time import (
    TravelTimeCalculator,
    EdgeLengthAccessor,
    InvalidPhysicalQuantityError
)

# ==========================================
# TravelTimeCalculator Tests
# ==========================================

def test_basic_conversion():
    # 1000 m @ 30 km/h = 1 km @ 30 km/h = 1/30 hours = 2 minutes
    time = TravelTimeCalculator.calculate(1000.0, 30.0)
    assert math.isclose(time, 2.0, rel_tol=1e-5)

def test_different_edge_lengths():
    time1 = TravelTimeCalculator.calculate(1000.0, 30.0)
    time2 = TravelTimeCalculator.calculate(2000.0, 30.0)
    assert math.isclose(time2, time1 * 2, rel_tol=1e-5)

def test_different_speeds():
    time1 = TravelTimeCalculator.calculate(1000.0, 30.0)
    time2 = TravelTimeCalculator.calculate(1000.0, 60.0)
    assert math.isclose(time2, time1 / 2, rel_tol=1e-5)

def test_floating_point_values():
    # 137.5 m @ 27.3 km/h
    # 27.3 km/h = 27300 m/hr = 455 m/min
    # time = 137.5 / 455 = 0.3021978...
    time = TravelTimeCalculator.calculate(137.5, 27.3)
    expected = 137.5 / (27.3 * 1000 / 60)
    assert math.isclose(time, expected, rel_tol=1e-5)

def test_zero_speed():
    with pytest.raises(InvalidPhysicalQuantityError, match="speed is zero"):
        TravelTimeCalculator.calculate(1000.0, 0.0)

def test_negative_speed():
    with pytest.raises(InvalidPhysicalQuantityError, match="finite non-negative"):
        TravelTimeCalculator.calculate(1000.0, -10.0)

def test_negative_length():
    with pytest.raises(InvalidPhysicalQuantityError, match="finite non-negative"):
        TravelTimeCalculator.calculate(-1000.0, 30.0)

def test_nan_values():
    with pytest.raises(InvalidPhysicalQuantityError, match="finite non-negative"):
        TravelTimeCalculator.calculate(float('nan'), 30.0)
    with pytest.raises(InvalidPhysicalQuantityError, match="finite non-negative"):
        TravelTimeCalculator.calculate(1000.0, float('nan'))

def test_infinity_values():
    with pytest.raises(InvalidPhysicalQuantityError, match="finite non-negative"):
        TravelTimeCalculator.calculate(float('inf'), 30.0)
    with pytest.raises(InvalidPhysicalQuantityError, match="finite non-negative"):
        TravelTimeCalculator.calculate(1000.0, float('inf'))

def test_missing_values():
    with pytest.raises(InvalidPhysicalQuantityError, match="must be a number"):
        TravelTimeCalculator.calculate(None, 30.0)
    with pytest.raises(InvalidPhysicalQuantityError, match="must be a number"):
        TravelTimeCalculator.calculate(1000.0, None)

def test_non_numeric_values():
    with pytest.raises(InvalidPhysicalQuantityError, match="must be a number"):
        TravelTimeCalculator.calculate("1000", 30.0)
    with pytest.raises(InvalidPhysicalQuantityError, match="must be a number"):
        TravelTimeCalculator.calculate(1000.0, "30")

def test_zero_length():
    # 0 length should safely return 0 time without doing unnecessary math
    time = TravelTimeCalculator.calculate(0.0, 30.0)
    assert time == 0.0


# ==========================================
# EdgeLengthAccessor Tests
# ==========================================

@pytest.fixture
def mock_graph():
    G = nx.MultiDiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    
    # Normal edge
    G.add_edge(1, 2, key=0, length=150.5)
    
    # Parallel edges
    G.add_edge(2, 3, key=0, length=100.0)
    G.add_edge(2, 3, key=1, length=250.0)
    
    # Edge with missing length
    G.add_edge(1, 3, key=0, other_attr="hello")
    
    # Edge with invalid length (negative)
    G.add_edge(3, 1, key=0, length=-50.0)
    
    # Edge with NaN length
    G.add_edge(3, 2, key=0, length=float('nan'))
    
    return G

def test_accessor_normal_edge(mock_graph):
    accessor = EdgeLengthAccessor(mock_graph)
    edge = EdgeId(source=1, target=2, key=0)
    assert accessor.get_length_m(edge) == 150.5

def test_accessor_parallel_edges(mock_graph):
    accessor = EdgeLengthAccessor(mock_graph)
    edge_0 = EdgeId(source=2, target=3, key=0)
    edge_1 = EdgeId(source=2, target=3, key=1)
    
    assert accessor.get_length_m(edge_0) == 100.0
    assert accessor.get_length_m(edge_1) == 250.0

def test_accessor_missing_edge(mock_graph):
    accessor = EdgeLengthAccessor(mock_graph)
    edge = EdgeId(source=1, target=2, key=99)
    with pytest.raises(KeyError, match="Edge key 99 not found"):
        accessor.get_length_m(edge)

def test_accessor_missing_length_attribute(mock_graph):
    accessor = EdgeLengthAccessor(mock_graph)
    edge = EdgeId(source=1, target=3, key=0)
    with pytest.raises(InvalidPhysicalQuantityError, match="missing 'length' attribute"):
        accessor.get_length_m(edge)

def test_accessor_invalid_length_negative(mock_graph):
    accessor = EdgeLengthAccessor(mock_graph)
    edge = EdgeId(source=3, target=1, key=0)
    with pytest.raises(InvalidPhysicalQuantityError, match="finite and non-negative"):
        accessor.get_length_m(edge)

def test_accessor_invalid_length_nan(mock_graph):
    accessor = EdgeLengthAccessor(mock_graph)
    edge = EdgeId(source=3, target=2, key=0)
    with pytest.raises(InvalidPhysicalQuantityError, match="finite and non-negative"):
        accessor.get_length_m(edge)


# ==========================================
# Integration Test
# ==========================================

def test_weight_calculator_integration():
    """
    Verifies complete local chain:
    EdgeId -> Mock Graph -> TravelTimeCalculator -> WeightCalculator
    """
    # 1. Setup mock graph with edge
    G = nx.MultiDiGraph()
    G.add_edge(1, 2, key=0, length=1000.0)
    edge_id = EdgeId(source=1, target=2, key=0)
    
    # 2. Access length
    accessor = EdgeLengthAccessor(G)
    length_m = accessor.get_length_m(edge_id)
    
    # 3. Mocked traffic speed (e.g. from TrafficPredictionProvider)
    predicted_speed_kmh = 30.0
    
    # 4. Calculate travel time
    travel_time_minutes = TravelTimeCalculator.calculate(length_m, predicted_speed_kmh)
    
    # 5. Integrate with WeightCalculator
    config = NormalizationConfig(
        travel_time_p5=1.0,     # min travel time
        travel_time_p95=5.0,    # max travel time
        travel_time_weight=0.6,
        safety_weight=0.4
    )
    calculator = WeightCalculator(config)
    
    # Simulated safety risk from SafetyPredictionProvider
    risk_score = 50.0  
    
    combined_cost = calculator.calculate(
        edge_id=edge_id,
        traffic_pred=TrafficPrediction(predicted_travel_time=travel_time_minutes),
        safety_pred=SafetyPrediction(risk_score=risk_score)
    )
    
    # Assertions
    assert travel_time_minutes == 2.0
    
    # Validation against WeightCalculator expected math:
    # t_norm = (2.0 - 1.0) / (5.0 - 1.0) = 1/4 = 0.25
    # s_norm = 50.0 / 100.0 = 0.5
    # combined = (0.6 * 0.25) + (0.4 * 0.5) = 0.15 + 0.20 = 0.35
    
    assert math.isclose(combined_cost.combined_cost, 0.35, rel_tol=1e-5)
    assert combined_cost.predicted_travel_time == 2.0
    assert combined_cost.raw_safety_risk == 50.0
    assert combined_cost.normalized_travel_time == 0.25
    assert combined_cost.normalized_safety_risk == 0.5
