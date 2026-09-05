import math
import logging
from dataclasses import dataclass
from typing import Optional
from src.routing.contracts import TrafficPrediction, SafetyPrediction, CombinedEdgeCost

logger = logging.getLogger(__name__)

@dataclass
class NormalizationConfig:
    """
    Configuration parameters for normalizing edge costs.
    Parameters should be calibrated on a reference dataset and persisted.
    """
    # Travel Time Normalization parameters (Min-Max with Percentile Clipping)
    travel_time_p5: float
    travel_time_p95: float
    
    # Weight Preferences (must sum to 1.0)
    travel_time_weight: float
    safety_weight: float
    
    def __post_init__(self):
        self.validate()
        
    def validate(self):
        if self.travel_time_weight < 0 or self.safety_weight < 0:
            raise ValueError("Weights cannot be negative.")
            
        total_weight = self.travel_time_weight + self.safety_weight
        if not math.isclose(total_weight, 1.0, rel_tol=1e-5):
            raise ValueError(f"Weights must sum to 1.0. Current sum is {total_weight}")
            
        if self.travel_time_p95 < self.travel_time_p5:
            raise ValueError("travel_time_p95 cannot be less than travel_time_p5.")


class WeightCalculator:
    """
    Calculates combined routing edge costs for A* by normalizing and weighting
    the independent predictions from Traffic ML and Safety ML models.
    """
    def __init__(self, config: NormalizationConfig):
        self.config = config

    def validate_predictions(
        self, 
        traffic_pred: TrafficPrediction, 
        safety_pred: SafetyPrediction
    ) -> None:
        """
        Validates the raw predictions from ML models.
        """
        # Validate Traffic Prediction
        if traffic_pred is None:
            raise ValueError("Traffic prediction is missing.")
            
        tt = traffic_pred.predicted_travel_time
        if tt is None or math.isnan(tt) or math.isinf(tt):
            raise ValueError(f"Invalid travel time prediction: {tt}")
            
        if tt <= 0:
            raise ValueError(f"Travel time must be greater than zero, got {tt}")
            
        # Validate Safety Prediction
        if safety_pred is None:
            raise ValueError("Safety prediction is missing.")
            
        sr = safety_pred.risk_score
        if sr is None or math.isnan(sr) or math.isinf(sr):
            raise ValueError(f"Invalid safety risk prediction: {sr}")
            
        if sr < 0 or sr > 100:
            logger.warning(f"Safety risk {sr} is outside the expected 0-100 scale.")

    def normalize_travel_time(self, travel_time: float) -> float:
        """
        Normalizes travel time to a [0, 1] scale using percentile-based min-max scaling.
        Clips outliers to robust percentiles to prevent extreme values from compressing the scale.
        """
        p5 = self.config.travel_time_p5
        p95 = self.config.travel_time_p95
        
        # Handle zero division if all candidate travel times are identical (p5 == p95)
        if math.isclose(p5, p95, rel_tol=1e-5):
            # If there's no variance in the reference dataset, return 0.5 (neutral)
            return 0.5
            
        # Clip outlier values
        clipped_t = max(p5, min(travel_time, p95))
        
        # Min-Max Normalization
        t_norm = (clipped_t - p5) / (p95 - p5)
        
        return t_norm

    def normalize_safety(self, risk_score: float) -> float:
        """
        Normalizes the safety risk score (assumed 0-100) to a [0, 1] scale.
        0 = safest, 1 = highest risk.
        """
        # Clip to ensure bounds even if model acts unexpectedly
        risk_score = max(0.0, min(risk_score, 100.0))
        return risk_score / 100.0

    def calculate_combined_cost(self, t_norm: float, s_norm: float) -> float:
        """
        Combines normalized travel time and safety risk using the configured weights.
        """
        cost = (self.config.travel_time_weight * t_norm) + (self.config.safety_weight * s_norm)
        return cost

    def calculate(
        self, 
        edge_id: str, 
        traffic_pred: TrafficPrediction, 
        safety_pred: SafetyPrediction
    ) -> CombinedEdgeCost:
        """
        Main orchestration function.
        Validates, normalizes, and combines predictions into a final routing cost.
        Preserves raw values for debugging and explainability.
        """
        self.validate_predictions(traffic_pred, safety_pred)
        
        t_norm = self.normalize_travel_time(traffic_pred.predicted_travel_time)
        s_norm = self.normalize_safety(safety_pred.risk_score)
        
        combined_cost = self.calculate_combined_cost(t_norm, s_norm)
        
        return CombinedEdgeCost(
            edge_id=edge_id,
            predicted_travel_time=traffic_pred.predicted_travel_time,
            raw_safety_risk=safety_pred.risk_score,
            normalized_travel_time=t_norm,
            normalized_safety_risk=s_norm,
            travel_time_weight=self.config.travel_time_weight,
            safety_weight=self.config.safety_weight,
            combined_cost=combined_cost
        )
