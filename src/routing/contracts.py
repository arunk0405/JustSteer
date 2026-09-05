from dataclasses import dataclass
from typing import Optional

@dataclass
class TrafficPrediction:
    """
    Explicit contract for ML Model 1 (Traffic / Travel Time Model)
    """
    predicted_travel_time: float
    unit: str = "minutes"

@dataclass
class SafetyPrediction:
    """
    Explicit contract for ML Model 2 (Safety Model)
    """
    risk_score: float
    scale: str = "0-100"

@dataclass
class CombinedEdgeCost:
    """
    The final output format of the Weight Calculation Engine, 
    designed for consumption by the future A* algorithm.
    """
    edge_id: str
    
    predicted_travel_time: float
    raw_safety_risk: float
    
    normalized_travel_time: float
    normalized_safety_risk: float
    
    travel_time_weight: float
    safety_weight: float
    
    combined_cost: float
