import math
import pytest
import networkx as nx
from datetime import datetime

from src.routing.edge_cost import EdgeId
from src.routing.prediction_providers import TrafficPredictionProvider, SafetyPredictionProvider
from src.routing.feature_resolver import (
    FeatureResolver,
    RoutingContext,
    WeatherContext,
    SafetyHistory,
    FeatureResolutionError,
    MissingSafetyDataError
)


@pytest.fixture
def mock_graph():
    G = nx.MultiDiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(1, 2, key=0, highway="primary", maxspeed="60", lanes="2")
    G.add_edge(1, 2, key=1, highway="residential", maxspeed="30", lanes="1")
    G.add_edge(2, 3, key=0, highway=["motorway", "primary"], maxspeed=["80", "60"], lanes=["3", "2"])
    return G

@pytest.fixture
def base_context():
    weather = WeatherContext(
        temperature_c=30.0,
        rain_1h_mm=0.0,
        wind_speed_kmh=10.0,
        visibility_m=10000.0,
        humidity_pct=60.0,
        rain_risk_score=0.1,
        rain_forecast_hours=0
    )
    safety = SafetyHistory(
        accidents_total_historical=10,
        fatal_accidents_historical=1,
        injuries_historical=5,
        historical_risk_score=45.0
    )
    return RoutingContext(
        timestamp=datetime(2023, 11, 15, 14, 30), # Nov (Monsoon), Wed, 14:30
        weather=weather,
        congestion_index=0.5,
        safety_history=safety
    )

@pytest.fixture
def resolver(mock_graph):
    return FeatureResolver(mock_graph)


def test_1_valid_feature_resolution(resolver, base_context):
    edge_id = EdgeId(1, 2, 0)
    features = resolver.resolve(edge_id, base_context)
    
    # Check shape
    assert features.edge_id == edge_id
    assert isinstance(features.traffic_features, dict)
    assert isinstance(features.safety_features, dict)
    
    # Check derivations
    tf = features.traffic_features
    assert tf["freeflow_speed_kmph"] == 60.0
    assert tf["road_type_enc"] == 1 # arterial
    assert tf["hour_of_day"] == 14
    assert tf["is_weekend"] == 0
    assert tf["is_ne_monsoon"] == 1
    
    sf = features.safety_features
    assert sf["speed_limit"] == 60.0
    assert sf["lanes"] == 2
    assert sf["accidents_total_historical"] == 10


def test_2_parallel_edges(resolver, base_context):
    edge_0 = EdgeId(1, 2, 0)
    edge_1 = EdgeId(1, 2, 1)
    
    f0 = resolver.resolve(edge_0, base_context)
    f1 = resolver.resolve(edge_1, base_context)
    
    assert f0.traffic_features["road_type_enc"] == 1 # arterial
    assert f0.traffic_features["freeflow_speed_kmph"] == 60.0
    
    assert f1.traffic_features["road_type_enc"] == 2 # residential
    assert f1.traffic_features["freeflow_speed_kmph"] == 30.0


def test_3_missing_edge(resolver, base_context):
    with pytest.raises(FeatureResolutionError, match="does not exist in graph"):
        resolver.resolve(EdgeId(99, 100, 0), base_context)


def test_4_missing_required_feature(resolver, base_context):
    # Instead of omitting the field (dataclass prevents it), we pass None to a required field
    # which we can do by mutating the dataclass using __dict__ trick or object.__setattr__
    # Alternatively, create a context with a None value.
    invalid_weather = WeatherContext(
        temperature_c=None, # type: ignore
        rain_1h_mm=0.0,
        wind_speed_kmh=10.0,
        visibility_m=10000.0,
        humidity_pct=60.0,
        rain_risk_score=0.1,
        rain_forecast_hours=0
    )
    context = RoutingContext(
        timestamp=base_context.timestamp,
        weather=invalid_weather,
        congestion_index=0.5,
        safety_history=base_context.safety_history
    )
    with pytest.raises(FeatureResolutionError, match="Feature 'temperature_c' is None"):
        resolver.resolve(EdgeId(1, 2, 0), context)


def test_5_nan_rejection(resolver, base_context):
    invalid_weather = WeatherContext(
        temperature_c=float('nan'),
        rain_1h_mm=0.0,
        wind_speed_kmh=10.0,
        visibility_m=10000.0,
        humidity_pct=60.0,
        rain_risk_score=0.1,
        rain_forecast_hours=0
    )
    context = RoutingContext(
        timestamp=base_context.timestamp,
        weather=invalid_weather,
        congestion_index=0.5,
        safety_history=base_context.safety_history
    )
    with pytest.raises(FeatureResolutionError, match="is not finite"):
        resolver.resolve(EdgeId(1, 2, 0), context)


def test_6_infinity_rejection(resolver, base_context):
    invalid_weather = WeatherContext(
        temperature_c=float('inf'),
        rain_1h_mm=0.0,
        wind_speed_kmh=10.0,
        visibility_m=10000.0,
        humidity_pct=60.0,
        rain_risk_score=0.1,
        rain_forecast_hours=0
    )
    context = RoutingContext(
        timestamp=base_context.timestamp,
        weather=invalid_weather,
        congestion_index=0.5,
        safety_history=base_context.safety_history
    )
    with pytest.raises(FeatureResolutionError, match="is not finite"):
        resolver.resolve(EdgeId(1, 2, 0), context)


def test_7_timestamp_derived_features(resolver, mock_graph):
    # Sunday, May 21 2023, 08:15 AM
    dt = datetime(2023, 5, 21, 8, 15)
    weather = WeatherContext(30.0, 0.0, 10.0, 10000.0, 60.0, 0.1, 0)
    safety = SafetyHistory(10, 1, 5, 45.0)
    context = RoutingContext(timestamp=dt, weather=weather, congestion_index=0.5, safety_history=safety)
    
    features = resolver.resolve(EdgeId(1, 2, 0), context)
    tf = features.traffic_features
    
    assert tf["hour_of_day"] == 8
    assert tf["day_of_week"] == 6 # Sunday
    assert tf["is_weekend"] == 1
    assert tf["month"] == 5
    assert tf["is_ne_monsoon"] == 0


def test_8_no_system_clock_dependency(resolver, base_context):
    f1 = resolver.resolve(EdgeId(1, 2, 0), base_context)
    f2 = resolver.resolve(EdgeId(1, 2, 0), base_context)
    
    assert f1.traffic_features["hour_of_day"] == f2.traffic_features["hour_of_day"]
    assert f1.traffic_features["day_of_week"] == f2.traffic_features["day_of_week"]


def test_9_safety_data_unavailable(resolver, base_context):
    context = RoutingContext(
        timestamp=base_context.timestamp,
        weather=base_context.weather,
        congestion_index=0.5,
        safety_history=None
    )
    with pytest.raises(MissingSafetyDataError, match="has no real runtime source"):
        resolver.resolve(EdgeId(1, 2, 0), context)


def test_10_no_graph_mutation(resolver, base_context):
    edge_id = EdgeId(2, 3, 0)
    original_data = dict(resolver.graph[2][3][0])
    
    resolver.resolve(edge_id, base_context)
    
    post_data = dict(resolver.graph[2][3][0])
    assert original_data == post_data


def test_11_feature_schema_compatibility(resolver, base_context):
    features = resolver.resolve(EdgeId(1, 2, 0), base_context)
    
    # Check that the keys exactly match the EXPECTED_FEATURES of the providers
    assert set(features.traffic_features.keys()) == set(TrafficPredictionProvider.EXPECTED_FEATURES)
    assert set(features.safety_features.keys()) == set(SafetyPredictionProvider.EXPECTED_FEATURES)
