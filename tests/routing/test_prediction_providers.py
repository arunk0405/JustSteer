import pytest
from unittest.mock import MagicMock, patch

from src.routing.prediction_providers import (
    SafetyPredictionProvider,
    TrafficPredictionProvider,
    ModelLoadingError
)
from src.routing.contracts import SafetyPrediction, TrafficPrediction
from src.routing.weight_calculator import WeightCalculator, NormalizationConfig


@pytest.fixture
def valid_safety_features():
    return {
        "accidents_total_historical": 10,
        "fatal_accidents_historical": 0,
        "injuries_historical": 2,
        "historical_risk_score": 5,
        "road_type_enc": 1,
        "lanes": 2,
        "speed_limit": 40,
        "lighting_enc": 0,
        "surface_enc": 1,
        "traffic_exposure_enc": 2
    }


@pytest.fixture
def valid_traffic_features():
    return {
        "rain_1h_mm": 0.0,
        "temperature_c": 30.0,
        "humidity_pct": 65.0,
        "wind_speed_kmh": 10.0,
        "visibility_m": 10000.0,
        "rain_risk_score": 0.0,
        "rain_forecast_hours": 0.0,
        "is_raining": 0,
        "is_foggy": 0,
        "is_high_wind": 0,
        "is_ne_monsoon": 0,
        "congestion_index": 0.5,
        "freeflow_speed_kmph": 40.0,
        "hour_of_day": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "month": 5,
        "zone_type_enc": 1,
        "road_type_enc": 2
    }


@patch("src.routing.prediction_providers.Path.exists")
@patch("src.routing.prediction_providers.joblib.load")
def test_safety_predictor_success(mock_load, mock_exists, valid_safety_features):
    """Test valid safety prediction returning the exact expected domain contract."""
    mock_exists.return_value = True
    
    # Mock model predict method
    mock_model = MagicMock()
    # Predict returns array [1] -> Penalty 50
    mock_model.predict.return_value = [1]
    mock_load.return_value = mock_model
    
    provider = SafetyPredictionProvider(model_path="dummy.pkl")
    prediction = provider.predict(valid_safety_features)
    
    assert isinstance(prediction, SafetyPrediction)
    assert prediction.risk_score == 50.0
    
    # Verify the features were passed exactly as expected in the dataframe
    called_df = mock_model.predict.call_args[0][0]
    assert list(called_df.columns) == SafetyPredictionProvider.EXPECTED_FEATURES
    assert called_df.iloc[0]["lanes"] == 2


@patch("src.routing.prediction_providers.Path.exists")
@patch("src.routing.prediction_providers.joblib.load")
def test_safety_predictor_invalid_class(mock_load, mock_exists, valid_safety_features):
    """Test safety prediction when the model outputs an unknown class."""
    mock_exists.return_value = True
    
    mock_model = MagicMock()
    # Predict returns array [99] -> Unknown
    mock_model.predict.return_value = [99]
    mock_load.return_value = mock_model
    
    provider = SafetyPredictionProvider(model_path="dummy.pkl")
    
    with pytest.raises(ValueError, match="Unknown predicted class"):
        provider.predict(valid_safety_features)


@patch("src.routing.prediction_providers.Path.exists")
def test_safety_predictor_missing_model(mock_exists):
    mock_exists.return_value = False
    
    with pytest.raises(ModelLoadingError, match="artifact missing"):
        SafetyPredictionProvider(model_path="dummy.pkl")


@patch("src.routing.prediction_providers.Path.exists")
@patch("src.routing.prediction_providers.joblib.load")
def test_safety_predictor_missing_features(mock_load, mock_exists, valid_safety_features):
    mock_exists.return_value = True
    provider = SafetyPredictionProvider(model_path="dummy.pkl")
    
    del valid_safety_features["speed_limit"]
    
    with pytest.raises(ValueError, match="Missing required feature"):
        provider.predict(valid_safety_features)


@patch("src.routing.prediction_providers.Path.exists")
@patch("src.routing.prediction_providers.joblib.load")
def test_traffic_predict_speed_success(mock_load, mock_exists, valid_traffic_features):
    """Test valid traffic speed extraction."""
    mock_exists.return_value = True
    
    mock_model = MagicMock()
    mock_model.predict.return_value = [35.5]
    mock_load.return_value = mock_model
    
    provider = TrafficPredictionProvider(model_path="dummy.pkl")
    speed = provider.predict_speed(valid_traffic_features)
    
    assert speed == 35.5
    
    called_df = mock_model.predict.call_args[0][0]
    assert list(called_df.columns) == TrafficPredictionProvider.EXPECTED_FEATURES
    assert called_df.iloc[0]["rain_1h_mm"] == 0.0


@patch("src.routing.prediction_providers.Path.exists")
@patch("src.routing.prediction_providers.joblib.load")
def test_traffic_predict_not_implemented(mock_load, mock_exists, valid_traffic_features):
    """Verify that domain contract conversion explicitly fails due to missing conversion rules."""
    mock_exists.return_value = True
    mock_model = MagicMock()
    mock_model.predict.return_value = [35.5]
    mock_load.return_value = mock_model
    
    provider = TrafficPredictionProvider(model_path="dummy.pkl")
    
    with pytest.raises(NotImplementedError, match="convert predicted speed .* into travel time"):
        provider.predict(valid_traffic_features)


def test_weight_calculator_compatibility():
    """
    Test 12: Compatibility Test With WeightCalculator.
    Validates that a correctly formatted SafetyPrediction and a 
    TrafficPrediction (if provided manually bypassing the error)
    are strictly compatible with the existing WeightCalculator.
    """
    config = NormalizationConfig(
        travel_time_p5=2.0,
        travel_time_p95=10.0,
        travel_time_weight=0.7,
        safety_weight=0.3
    )
    calculator = WeightCalculator(config)
    
    # We simulate a successful domain contract instantiation
    safe_pred = SafetyPrediction(risk_score=50.0)
    traffic_pred = TrafficPrediction(predicted_travel_time=5.0)
    
    combined = calculator.calculate(
        edge_id="test-edge",
        traffic_pred=traffic_pred,
        safety_pred=safe_pred
    )
    
    assert combined.edge_id == "test-edge"
    # travel time 5.0 -> norm = (5-2)/(10-2) = 3/8 = 0.375
    # safety 50.0 -> norm = 0.5
    # combined = 0.375 * 0.7 + 0.5 * 0.3 = 0.2625 + 0.15 = 0.4125
    assert abs(combined.combined_cost - 0.4125) < 1e-5
    assert combined.predicted_travel_time == 5.0
    assert combined.raw_safety_risk == 50.0
