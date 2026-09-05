import pytest
import math
from src.routing.contracts import TrafficPrediction, SafetyPrediction
from src.routing.weight_calculator import NormalizationConfig, WeightCalculator

def test_weight_validation_success():
    config = NormalizationConfig(
        travel_time_p5=2.0, travel_time_p95=20.0,
        travel_time_weight=0.7, safety_weight=0.3
    )
    assert config.travel_time_weight + config.safety_weight == 1.0

def test_weight_validation_failure():
    with pytest.raises(ValueError, match="Weights must sum to 1.0"):
        NormalizationConfig(
            travel_time_p5=2.0, travel_time_p95=20.0,
            travel_time_weight=0.8, safety_weight=0.3
        )

def test_weight_validation_negative():
    with pytest.raises(ValueError, match="Weights cannot be negative"):
        NormalizationConfig(
            travel_time_p5=2.0, travel_time_p95=20.0,
            travel_time_weight=1.1, safety_weight=-0.1
        )

def test_percentile_validation():
    with pytest.raises(ValueError, match="travel_time_p95 cannot be less than"):
        NormalizationConfig(
            travel_time_p5=20.0, travel_time_p95=2.0,
            travel_time_weight=0.7, safety_weight=0.3
        )

@pytest.fixture
def calculator():
    config = NormalizationConfig(
        travel_time_p5=2.0, travel_time_p95=20.0,
        travel_time_weight=0.7, safety_weight=0.3
    )
    return WeightCalculator(config)

def test_missing_predictions(calculator):
    with pytest.raises(ValueError, match="Traffic prediction is missing"):
        calculator.calculate("E1", None, SafetyPrediction(20))
        
    with pytest.raises(ValueError, match="Safety prediction is missing"):
        calculator.calculate("E1", TrafficPrediction(5), None)

def test_nan_infinity_handling(calculator):
    with pytest.raises(ValueError, match="Invalid travel time prediction: nan"):
        calculator.calculate("E1", TrafficPrediction(math.nan), SafetyPrediction(20))
        
    with pytest.raises(ValueError, match="Invalid travel time prediction: inf"):
        calculator.calculate("E1", TrafficPrediction(math.inf), SafetyPrediction(20))
        
    with pytest.raises(ValueError, match="Invalid safety risk prediction: nan"):
        calculator.calculate("E1", TrafficPrediction(5), SafetyPrediction(math.nan))

def test_negative_travel_time(calculator):
    with pytest.raises(ValueError, match="Travel time must be greater than zero"):
        calculator.calculate("E1", TrafficPrediction(-5.0), SafetyPrediction(20))

def test_zero_denominator_fallback():
    # p5 == p95 -> no variance in reference dataset
    config = NormalizationConfig(
        travel_time_p5=10.0, travel_time_p95=10.0,
        travel_time_weight=0.7, safety_weight=0.3
    )
    calc = WeightCalculator(config)
    
    # normalized travel time should fallback to 0.5
    result = calc.calculate("E1", TrafficPrediction(15.0), SafetyPrediction(20))
    assert math.isclose(result.normalized_travel_time, 0.5)

def test_outlier_clipping(calculator):
    # Below P5
    result_low = calculator.calculate("E1", TrafficPrediction(1.0), SafetyPrediction(20))
    assert math.isclose(result_low.normalized_travel_time, 0.0) # Clipped to 2.0 -> (2-2)/18 = 0
    
    # Above P95
    result_high = calculator.calculate("E1", TrafficPrediction(120.0), SafetyPrediction(20))
    assert math.isclose(result_high.normalized_travel_time, 1.0) # Clipped to 20.0 -> (20-2)/18 = 1

def test_boundary_values(calculator):
    # Safety bounds
    result_safe = calculator.calculate("E1", TrafficPrediction(10.0), SafetyPrediction(-10))
    assert math.isclose(result_safe.normalized_safety_risk, 0.0)
    
    result_danger = calculator.calculate("E1", TrafficPrediction(10.0), SafetyPrediction(150))
    assert math.isclose(result_danger.normalized_safety_risk, 1.0)

def test_monotonicity_safety(calculator):
    # Increasing safety risk must strictly increase cost if weight > 0
    res1 = calculator.calculate("E1", TrafficPrediction(10.0), SafetyPrediction(20))
    res2 = calculator.calculate("E1", TrafficPrediction(10.0), SafetyPrediction(70))
    
    assert res2.combined_cost > res1.combined_cost

def test_monotonicity_travel_time(calculator):
    # Increasing travel time must strictly increase cost if weight > 0
    res1 = calculator.calculate("E1", TrafficPrediction(5.0), SafetyPrediction(40))
    res2 = calculator.calculate("E1", TrafficPrediction(15.0), SafetyPrediction(40))
    
    assert res2.combined_cost > res1.combined_cost

def test_weight_extremes():
    # α = 1, β = 0
    config1 = NormalizationConfig(
        travel_time_p5=0.0, travel_time_p95=10.0,
        travel_time_weight=1.0, safety_weight=0.0
    )
    calc1 = WeightCalculator(config1)
    res1 = calc1.calculate("E1", TrafficPrediction(5.0), SafetyPrediction(100))
    assert math.isclose(res1.combined_cost, 0.5) # safety 100 is ignored
    
    # α = 0, β = 1
    config2 = NormalizationConfig(
        travel_time_p5=0.0, travel_time_p95=10.0,
        travel_time_weight=0.0, safety_weight=1.0
    )
    calc2 = WeightCalculator(config2)
    res2 = calc2.calculate("E1", TrafficPrediction(5.0), SafetyPrediction(100))
    assert math.isclose(res2.combined_cost, 1.0) # travel time 5 is ignored

def test_example_calculation_matches_prompt(calculator):
    # Road A: TT=5, Safety=20
    resA = calculator.calculate("RoadA", TrafficPrediction(5), SafetyPrediction(20))
    
    # T_norm = (5 - 2) / (20 - 2) = 3 / 18 = 0.1666...
    assert math.isclose(resA.normalized_travel_time, 0.166666, rel_tol=1e-4)
    # S_norm = 20 / 100 = 0.2
    assert math.isclose(resA.normalized_safety_risk, 0.2)
    
    # Combined = 0.7 * 0.1666... + 0.3 * 0.2 = 0.1166... + 0.06 = 0.1766...
    assert math.isclose(resA.combined_cost, 0.176666, rel_tol=1e-4)
    
    # Expected range: cost must be between 0 and 1
    assert 0.0 <= resA.combined_cost <= 1.0
