import math
import networkx as nx
from typing import Dict, Any
from dataclasses import dataclass

from src.routing.edge_cost import EdgeId
from src.routing.contracts import TrafficPrediction, SafetyPrediction, CombinedEdgeCost
from src.routing.prediction_providers import TrafficPredictionProvider, SafetyPredictionProvider
from src.routing.travel_time import EdgeLengthAccessor, TravelTimeCalculator
from src.routing.weight_calculator import WeightCalculator

@dataclass(frozen=True)
class EdgeCostPredictionResult:
    """
    Immutable result containing the complete ML-derived state for a single candidate edge.
    Intended for persistence into the EdgeCostStore in the next stage.
    """
    edge_id: EdgeId
    traffic_prediction: TrafficPrediction
    safety_prediction: SafetyPrediction
    travel_time_minutes: float
    combined_cost: CombinedEdgeCost


class FeatureValidationError(ValueError):
    """Raised when an API-supplied model feature is missing, NaN, Inf, or otherwise un-inferable."""
    pass


class PredictionOrchestrator:
    """
    Clean architectural boundary for single-edge dynamic cost resolution.
    
    Responsibilities:
    1. Validate API-supplied ML features.
    2. Extract physical edge geometry from the OSM graph.
    3. Invoke prediction providers (Traffic and Safety).
    4. Convert Traffic speed into Travel Time.
    5. Evaluate final CombinedEdgeCost via WeightCalculator.
    """
    
    def __init__(
        self,
        traffic_provider: TrafficPredictionProvider,
        safety_provider: SafetyPredictionProvider,
        weight_calculator: WeightCalculator
    ):
        self.traffic_provider = traffic_provider
        self.safety_provider = safety_provider
        self.weight_calculator = weight_calculator
        
    def _validate_features(self, features: Dict[str, Any], expected_keys: list, context: str) -> None:
        """
        Lightweight feature validation before passing to pandas/sklearn.
        Ensures strict domain compliance without inventing missing values.
        """
        for key in expected_keys:
            if key not in features:
                raise FeatureValidationError(f"Missing required {context} feature: {key}")
                
            value = features[key]
            
            if value is None:
                raise FeatureValidationError(f"{context} feature '{key}' is None.")
                
            if not isinstance(value, (int, float, bool)):
                # Must be strictly numeric or bool. Do not parse strings silently.
                raise FeatureValidationError(f"{context} feature '{key}' is of invalid type {type(value).__name__}.")
                
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value):
                    raise FeatureValidationError(f"{context} feature '{key}' is NaN or Infinity: {value}")

    def predict_edge(
        self,
        edge_id: EdgeId,
        graph: nx.MultiDiGraph,
        traffic_features: Dict[str, Any],
        safety_features: Dict[str, Any]
    ) -> EdgeCostPredictionResult:
        """
        Executes the complete prediction and cost evaluation pipeline for a single edge.
        """
        # 1. Validate Feature Dictionaries
        self._validate_features(
            traffic_features, 
            self.traffic_provider.EXPECTED_FEATURES, 
            "traffic"
        )
        self._validate_features(
            safety_features, 
            self.safety_provider.EXPECTED_FEATURES, 
            "safety"
        )
        
        # 2. Geometry Resolution (Read-only from OSM graph)
        accessor = EdgeLengthAccessor(graph)
        length_m = accessor.get_length_m(edge_id)
        
        # 3. Traffic Speed Prediction
        speed_kmh = self.traffic_provider.predict_speed(traffic_features)
        
        # 4. Travel Time Resolution
        travel_time_minutes = TravelTimeCalculator.calculate(
            length_m=length_m, 
            speed_kmh=speed_kmh
        )
        
        traffic_prediction = TrafficPrediction(predicted_travel_time=travel_time_minutes)
        
        # 5. Safety Risk Prediction
        safety_prediction = self.safety_provider.predict(safety_features)
        
        # 6. Combined Cost Resolution
        # Note: edge_id is passed as a string representation to match WeightCalculator contract
        combined_cost = self.weight_calculator.calculate(
            edge_id=str(edge_id),
            traffic_pred=traffic_prediction,
            safety_pred=safety_prediction
        )
        
        return EdgeCostPredictionResult(
            edge_id=edge_id,
            traffic_prediction=traffic_prediction,
            safety_prediction=safety_prediction,
            travel_time_minutes=travel_time_minutes,
            combined_cost=combined_cost
        )
