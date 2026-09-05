import sys
import os
import math
import pandas as pd
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.routing.contracts import TrafficPrediction, SafetyPrediction
from src.routing.weight_calculator import NormalizationConfig, WeightCalculator

def verify_pipeline():
    print("==================================================")
    print("PIPELINE END-TO-END VERIFICATION")
    print("==================================================\n")
    
    # Load ML Model 1 (Velocity Predictor)
    velocity_model_path = "models/velocity_predictor.pkl"
    if not os.path.exists(velocity_model_path):
        print(f"Error: {velocity_model_path} not found. Train ML Model 1 first.")
        sys.exit(1)
        
    velocity_model = joblib.load(velocity_model_path)
    print("[OK] ML Model 1 (Velocity Predictor) loaded.")
    
    # Load ML Model 2 (Safety Risk XGBoost)
    safety_model_path = "models/safety/safety_risk_xgb.pkl"
    if not os.path.exists(safety_model_path):
        print(f"Error: {safety_model_path} not found. Train ML Model 2 first.")
        sys.exit(1)
        
    safety_model = joblib.load(safety_model_path)
    print("[OK] ML Model 2 (Safety Risk Predictor) loaded.")
    
    # 1. Prepare dummy input for ML Model 1 (19 features)
    feature_cols = [
        "rain_1h_mm", "temperature_c", "humidity_pct", "wind_speed_kmh", 
        "visibility_m", "rain_risk_score", "rain_forecast_hours", "is_raining",
        "is_foggy", "is_high_wind", "is_ne_monsoon", "congestion_index",
        "freeflow_speed_kmph", "hour_of_day", "day_of_week", "is_weekend",
        "month", "zone_type_enc", "road_type_enc"
    ]
    
    # Fake edge parameters (arterial road, slightly rainy)
    fake_edge_traffic = pd.DataFrame([{
        "rain_1h_mm": 2.5, "temperature_c": 30.0, "humidity_pct": 80.0, "wind_speed_kmh": 15.0, 
        "visibility_m": 5000, "rain_risk_score": 0.3, "rain_forecast_hours": 2, "is_raining": 1,
        "is_foggy": 0, "is_high_wind": 0, "is_ne_monsoon": 1, "congestion_index": 0.6,
        "freeflow_speed_kmph": 50.0, "hour_of_day": 14, "day_of_week": 2, "is_weekend": 0,
        "month": 11, "zone_type_enc": 1, "road_type_enc": 1
    }])[feature_cols] # Ensure column order
    
    # 2. Predict Travel Time
    # ML Model 1 outputs velocity (km/h). Travel time depends on edge length.
    # Let's assume the edge is 1 km long. travel_time (hours) = 1 / velocity. (minutes) = 60 / velocity.
    predicted_velocity = velocity_model.predict(fake_edge_traffic)[0]
    predicted_travel_time_min = 60.0 / max(predicted_velocity, 1.0)
    print(f"[*] ML Model 1 Output: {predicted_velocity:.2f} km/h -> {predicted_travel_time_min:.2f} minutes")
    
    # 3. Prepare dummy input for ML Model 2 (10 features)
    safety_features = [
        "accidents_total_historical", "fatal_accidents_historical",
        "injuries_historical", "historical_risk_score", "road_type_enc",
        "lanes", "speed_limit", "lighting_enc", "surface_enc", "traffic_exposure_enc"
    ]
    
    fake_edge_safety = pd.DataFrame([{
        "accidents_total_historical": 5,
        "fatal_accidents_historical": 0,
        "injuries_historical": 2,
        "historical_risk_score": 2,
        "road_type_enc": 1, # Arterial
        "lanes": 2,
        "speed_limit": 50,
        "lighting_enc": 1,
        "surface_enc": 0,
        "traffic_exposure_enc": 2
    }])[safety_features]
    
    # 4. Predict Safety Risk
    # XGBoost in our script outputs class (0, 1, 2) and probabilities.
    # Let's use the probability of HIGH risk (class 2) * 100 as the 0-100 risk score, or mapping 0->20, 1->50, 2->80
    predicted_safety_class = safety_model.predict(fake_edge_safety)[0]
    PENALTY_MAP = {0: 20, 1: 50, 2: 80}
    predicted_safety_score = PENALTY_MAP[predicted_safety_class]
    print(f"[*] ML Model 2 Output: Class {predicted_safety_class} -> Risk Score {predicted_safety_score}")
    
    # 5. Route calculation
    config = NormalizationConfig(
        travel_time_p5=2.0,
        travel_time_p95=20.0,
        travel_time_weight=0.7,
        safety_weight=0.3
    )
    calculator = WeightCalculator(config)
    
    combined = calculator.calculate(
        edge_id="EDGE_123",
        traffic_pred=TrafficPrediction(predicted_travel_time_min),
        safety_pred=SafetyPrediction(predicted_safety_score)
    )
    
    print("\n--------------------------------------------------")
    print("FINAL A* EDGE COST")
    print("--------------------------------------------------")
    print(f"Edge ID:           {combined.edge_id}")
    print(f"Travel Time (min): {combined.predicted_travel_time:.2f} (norm: {combined.normalized_travel_time:.3f})")
    print(f"Safety Risk (0-100): {combined.raw_safety_risk:.2f} (norm: {combined.normalized_safety_risk:.3f})")
    print(f"Combined Cost:     {combined.combined_cost:.4f}")
    
    print("\n[OK] Pipeline Verified.")

if __name__ == "__main__":
    verify_pipeline()
