import math
from datetime import datetime
from typing import Any, Dict, Optional
import networkx as nx
from dataclasses import dataclass

from src.routing.edge_cost import EdgeId


class MissingSafetyDataError(ValueError):
    """Raised when historical safety data is not available from the runtime source."""
    pass


class FeatureResolutionError(ValueError):
    """Raised when required features cannot be resolved or derived."""
    pass


@dataclass(frozen=True)
class WeatherContext:
    temperature_c: float
    rain_1h_mm: float
    wind_speed_kmh: float
    visibility_m: float
    humidity_pct: float
    rain_risk_score: float
    rain_forecast_hours: int


@dataclass(frozen=True)
class SafetyHistory:
    accidents_total_historical: int
    fatal_accidents_historical: int
    injuries_historical: int
    historical_risk_score: float


@dataclass(frozen=True)
class RoutingContext:
    timestamp: datetime
    weather: WeatherContext
    congestion_index: float
    safety_history: Optional[SafetyHistory] = None


@dataclass(frozen=True)
class ResolvedEdgeFeatures:
    edge_id: EdgeId
    traffic_features: Dict[str, Any]
    safety_features: Dict[str, Any]


class FeatureResolver:
    """
    Architectural boundary that isolates external dynamic state (RoutingContext)
    and static topology (OSM graph) to assemble the exact feature dictionaries 
    required by PredictionOrchestrator.
    """
    
    ROAD_TYPE_MAP = {"highway": 0, "arterial": 1, "residential": 2, "service": 3}
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    def resolve(self, edge_id: EdgeId, context: RoutingContext) -> ResolvedEdgeFeatures:
        if not self.graph.has_edge(edge_id.source, edge_id.target, key=edge_id.key):
            raise FeatureResolutionError(f"Edge {edge_id} does not exist in graph.")
            
        edge_data = self.graph[edge_id.source][edge_id.target][edge_id.key]
        
        # 1. Parse OSM Attributes
        road_type_enc = self._parse_road_type(edge_data.get('highway', 'residential'))
        freeflow_speed = self._parse_speed(edge_data.get('maxspeed', 40.0))
        lanes = self._parse_lanes(edge_data.get('lanes', 1))
        
        # OSM doesn't natively provide zone_type directly on edges typically
        zone_type_enc = 0 # Default: residential
        
        # Missing real data source for these synthetic encodings in OSM
        lighting_enc = 0
        surface_enc = 0
        traffic_exposure_enc = 0
        
        # 2. Derive Timestamp Features
        dt = context.timestamp
        hour_of_day = dt.hour
        day_of_week = dt.weekday()
        is_weekend = int(day_of_week >= 5)
        month = dt.month
        is_ne_monsoon = int(month in [10, 11, 12])
        
        # 3. Derive Weather Features (mirroring train.py rules exactly)
        w = context.weather
        # Validate raw weather values before logical derivation
        rain_1h_mm = self._validate_numeric_feature("rain_1h_mm", w.rain_1h_mm)
        visibility_m = self._validate_numeric_feature("visibility_m", w.visibility_m)
        wind_speed_kmh = self._validate_numeric_feature("wind_speed_kmh", w.wind_speed_kmh)
        
        is_raining = int(rain_1h_mm > 0.1)
        is_foggy = int(visibility_m < 1000)
        is_high_wind = int(wind_speed_kmh > 36.0)
        
        # 4. Construct Traffic Features
        traffic_features = {
            "rain_1h_mm": rain_1h_mm,
            "temperature_c": self._validate_numeric_feature("temperature_c", w.temperature_c),
            "humidity_pct": self._validate_numeric_feature("humidity_pct", w.humidity_pct),
            "wind_speed_kmh": wind_speed_kmh,
            "visibility_m": visibility_m,
            "rain_risk_score": self._validate_numeric_feature("rain_risk_score", w.rain_risk_score),
            "rain_forecast_hours": self._validate_numeric_feature("rain_forecast_hours", w.rain_forecast_hours, as_int=True),
            "is_raining": is_raining,
            "is_foggy": is_foggy,
            "is_high_wind": is_high_wind,
            "is_ne_monsoon": is_ne_monsoon,
            "congestion_index": self._validate_numeric_feature("congestion_index", context.congestion_index),
            "freeflow_speed_kmph": self._validate_numeric_feature("freeflow_speed_kmph", freeflow_speed),
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "month": month,
            "zone_type_enc": zone_type_enc,
            "road_type_enc": road_type_enc
        }
        
        # 5. Construct Safety Features
        if context.safety_history is None:
            raise MissingSafetyDataError(
                "Safety history is currently trained on synthetic data and has no real runtime source. "
                "Cannot invent fallback values."
            )
            
        sh = context.safety_history
        safety_features = {
            "accidents_total_historical": self._validate_numeric_feature("accidents_total_historical", sh.accidents_total_historical, as_int=True),
            "fatal_accidents_historical": self._validate_numeric_feature("fatal_accidents_historical", sh.fatal_accidents_historical, as_int=True),
            "injuries_historical": self._validate_numeric_feature("injuries_historical", sh.injuries_historical, as_int=True),
            "historical_risk_score": self._validate_numeric_feature("historical_risk_score", sh.historical_risk_score),
            "road_type_enc": road_type_enc,
            "lanes": self._validate_numeric_feature("lanes", lanes, as_int=True),
            "speed_limit": self._validate_numeric_feature("speed_limit", freeflow_speed),
            "lighting_enc": lighting_enc,
            "surface_enc": surface_enc,
            "traffic_exposure_enc": traffic_exposure_enc
        }
        
        return ResolvedEdgeFeatures(
            edge_id=edge_id,
            traffic_features=traffic_features,
            safety_features=safety_features
        )

    def _parse_road_type(self, highway_tag: Any) -> int:
        if isinstance(highway_tag, list):
            highway_tag = highway_tag[0]
        if not isinstance(highway_tag, str):
            return self.ROAD_TYPE_MAP["residential"]
            
        highway_tag = highway_tag.lower()
        if highway_tag in ["motorway", "motorway_link", "trunk", "trunk_link"]:
            return self.ROAD_TYPE_MAP["highway"]
        elif highway_tag in ["primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link"]:
            return self.ROAD_TYPE_MAP["arterial"]
        elif highway_tag in ["residential", "unclassified", "living_street"]:
            return self.ROAD_TYPE_MAP["residential"]
        elif highway_tag in ["service", "track"]:
            return self.ROAD_TYPE_MAP["service"]
        return self.ROAD_TYPE_MAP["residential"]

    def _parse_speed(self, maxspeed_tag: Any) -> float:
        if isinstance(maxspeed_tag, list):
            maxspeed_tag = maxspeed_tag[0]
        try:
            val = float(maxspeed_tag)
            if math.isfinite(val) and val >= 0:
                return val
        except (ValueError, TypeError):
            pass
        return 40.0

    def _parse_lanes(self, lanes_tag: Any) -> int:
        if isinstance(lanes_tag, list):
            lanes_tag = lanes_tag[0]
        try:
            val = int(lanes_tag)
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass
        return 1

    def _validate_numeric_feature(self, name: str, value: Any, as_int: bool = False) -> Any:
        if value is None:
            raise FeatureResolutionError(f"Feature '{name}' is None")
            
        if isinstance(value, bool):
            raise FeatureResolutionError(f"Feature '{name}' is a boolean, expected numeric")
            
        try:
            parsed = int(value) if as_int else float(value)
        except (ValueError, TypeError):
            raise FeatureResolutionError(f"Feature '{name}' is invalid non-numeric value: {value}")
            
        if isinstance(parsed, float):
            if not math.isfinite(parsed):
                raise FeatureResolutionError(f"Feature '{name}' is not finite: {parsed}")
                
        return parsed
