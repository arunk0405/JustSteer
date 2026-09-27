from datetime import datetime
from typing import List, Any, Optional
from pydantic import BaseModel, Field

class Coordinate(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)

class MultiStopRouteRequest(BaseModel):
    origin: Coordinate
    deliveries: List[Coordinate] = Field(..., max_length=8)
    timestamp: Optional[datetime] = None

class RouteLeg(BaseModel):
    start_location: Coordinate
    end_location: Coordinate
    distance_m: float
    estimated_duration_s: float
    geometry: Optional[dict] = None  # GeoJSON LineString format

class RouteSummary(BaseModel):
    total_distance_m: float
    estimated_duration_s: float
    stop_count: int

class TrafficInfo(BaseModel):
    average_predicted_travel_time_s: float

class SafetyInfo(BaseModel):
    average_risk: float

class WeightsInfo(BaseModel):
    travel_time: float
    safety: float

class ModelInfo(BaseModel):
    traffic: str
    safety: str

class ExplainabilityMetadata(BaseModel):
    candidate_edges_evaluated: int
    a_star_expanded_nodes: int
    evaluated_permutations: int
    traffic: TrafficInfo
    safety: SafetyInfo
    weights: WeightsInfo
    models: ModelInfo

class MultiStopRouteResponse(BaseModel):
    success: bool
    summary: RouteSummary
    optimized_stops: List[Coordinate]
    legs: List[RouteLeg]
    explainability: Optional[ExplainabilityMetadata] = None
