import joblib
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import warnings

# Suppress sklearn/xgboost warning verbosity during standard inference
warnings.filterwarnings("ignore", category=UserWarning)

from src.routing.contracts import SafetyPrediction, TrafficPrediction

class ModelLoadingError(Exception):
    """Raised when an ML model artifact is missing or fails to deserialize."""
    pass

class SafetyPredictionProvider:
    """
    Inference adapter for the trained safety risk model.
    Transforms raw edge features into the required exact model format, executes 
    inference, and maps the output (class 0/1/2) to the 0-100 risk scale established 
    in train_safety.py.
    """
    
    EXPECTED_FEATURES = [
        "accidents_total_historical",
        "fatal_accidents_historical",
        "injuries_historical",
        "historical_risk_score",
        "road_type_enc",
        "lanes",
        "speed_limit",
        "lighting_enc",
        "surface_enc",
        "traffic_exposure_enc"
    ]
    
    # Established in train_safety.py: PENALTY_MAP = {0: 20, 1: 50, 2: 80}
    PENALTY_MAP = {0: 20, 1: 50, 2: 80}
    
    def __init__(self, model_path: str = "models/safety/safety_risk_xgb.pkl"):
        path = Path(model_path)
        if not path.exists():
            raise ModelLoadingError(f"Safety model artifact missing at {path}")
            
        try:
            self.model = joblib.load(path)
        except Exception as e:
            raise ModelLoadingError(f"Failed to deserialize safety model: {e}")
            
    def predict(self, feature_dict: Dict[str, Any]) -> SafetyPrediction:
        """
        Predict the safety risk penalty for an edge.
        Returns a domain-compliant SafetyPrediction.
        """
        for f in self.EXPECTED_FEATURES:
            if f not in feature_dict:
                raise ValueError(f"Missing required feature for safety prediction: {f}")
                
        # Construct DataFrame in expected order
        df = pd.DataFrame([feature_dict])[self.EXPECTED_FEATURES]
        
        try:
            prediction_enc = int(self.model.predict(df)[0])
        except Exception as e:
            raise RuntimeError(f"Safety model inference failed: {e}")
            
        if prediction_enc not in self.PENALTY_MAP:
            raise ValueError(f"Unknown predicted class from safety model: {prediction_enc}")
            
        risk_score = self.PENALTY_MAP[prediction_enc]
        return SafetyPrediction(risk_score=float(risk_score))


class TrafficPredictionProvider:
    """
    Inference adapter for the trained velocity prediction model.
    """
    EXPECTED_FEATURES = [
        "rain_1h_mm", "temperature_c", "humidity_pct", "wind_speed_kmh", 
        "visibility_m", "rain_risk_score", "rain_forecast_hours", "is_raining", 
        "is_foggy", "is_high_wind", "is_ne_monsoon", "congestion_index", 
        "freeflow_speed_kmph", "hour_of_day", "day_of_week", "is_weekend", 
        "month", "zone_type_enc", "road_type_enc"
    ]
    
    def __init__(self, model_path: str = "models/velocity_predictor.pkl"):
        path = Path(model_path)
        if not path.exists():
            raise ModelLoadingError(f"Velocity model artifact missing at {path}")
            
        try:
            self.model = joblib.load(path)
        except Exception as e:
            raise ModelLoadingError(f"Failed to deserialize velocity model: {e}")
            
    def predict_speed(self, feature_dict: Dict[str, Any]) -> float:
        """
        Executes inference directly, returning the raw traffic_speed_kmh value.
        """
        for f in self.EXPECTED_FEATURES:
            if f not in feature_dict:
                raise ValueError(f"Missing required feature for traffic prediction: {f}")
                
        df = pd.DataFrame([feature_dict])[self.EXPECTED_FEATURES]
        
        try:
            speed_kmh = float(self.model.predict(df)[0])
        except Exception as e:
            raise RuntimeError(f"Traffic model inference failed: {e}")
            
        return speed_kmh

    def predict(self, feature_dict: Dict[str, Any]) -> TrafficPrediction:
        """
        Attempt to return a domain-compliant TrafficPrediction.
        """
        # We can successfully extract the speed:
        _speed_kmh = self.predict_speed(feature_dict)
        
        # However, the domain contract requires travel_time (minutes).
        # We cannot safely convert speed -> travel time without distance
        # AND without an explicitly documented conversion formula.
        raise NotImplementedError(
            "The repository does not currently define a formula to convert "
            "predicted speed (km/h) into travel time (minutes). "
            "Prediction cannot be safely integrated with WeightCalculator until this is resolved."
        )
