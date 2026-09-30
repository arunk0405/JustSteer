import sys
import os
import json
import networkx as nx
import osmnx as ox
from datetime import datetime

sys.path.append(os.path.abspath("."))

from src.routing.feature_resolver import FeatureResolver, RoutingContext, WeatherContext, SafetyHistory
from src.routing.prediction_orchestrator import PredictionOrchestrator
from src.routing.prediction_providers import SafetyPredictionProvider, TrafficPredictionProvider
from src.routing.weight_calculator import WeightCalculator, NormalizationConfig
from src.routing.dynamic_routing_app import DynamicRoutingApp
from src.routing.candidate_selector import CandidateEdgeSelector

print("=== START VERIFICATION ===")

graph_path = os.getenv("GRAPH_PATH", "data/road_network/chennai_drive.graphml")
graph = ox.load_graphml(graph_path)

safety_model_path = os.getenv("SAFETY_MODEL_PATH", "models/safety/safety_risk_xgb.pkl")
traffic_model_path = os.getenv("TRAFFIC_MODEL_PATH", "models/velocity_predictor.pkl")

safety_provider = SafetyPredictionProvider(safety_model_path)
traffic_provider = TrafficPredictionProvider(traffic_model_path)

norm_config = NormalizationConfig(
    travel_time_p5=0.5,
    travel_time_p95=10.0,
    travel_time_weight=0.6,
    safety_weight=0.4
)
weight_calculator = WeightCalculator(norm_config)

prediction_orchestrator = PredictionOrchestrator(
    traffic_provider=traffic_provider,
    safety_provider=safety_provider,
    weight_calculator=weight_calculator
)

feature_resolver = FeatureResolver(graph)

# Phase 8 & 9: Edge-Aware Inference and Dynamic Cost
edges = list(graph.edges(data=True))
# Pick 3 edges with different road types
primary_edge = next((u, v, d) for u, v, d in edges if d.get("highway") in ["primary", "trunk", "motorway"])
secondary_edge = next((u, v, d) for u, v, d in edges if d.get("highway") in ["secondary", "tertiary", "secondary_link"])
residential_edge = next((u, v, d) for u, v, d in edges if d.get("highway") in ["residential", "unclassified"])

# Dummy context
w_context = WeatherContext(temperature_c=30.0, rain_1h_mm=0.0, wind_speed_kmh=5.0, visibility_m=10000.0, humidity_pct=50.0, rain_risk_score=0.1, rain_forecast_hours=0)
s_history = SafetyHistory(accidents_total_historical=10, fatal_accidents_historical=0, injuries_historical=2, historical_risk_score=0.2)
context = RoutingContext(timestamp=datetime.utcnow(), weather=w_context, congestion_index=0.5, safety_history=s_history)

print("\n--- Edge Verification ---")
for name, edge in [("Highway", primary_edge), ("Arterial", secondary_edge), ("Residential", residential_edge)]:
    u, v, d = edge
    edge_id = f"{u}-{v}"
    from src.routing.edge_cost import EdgeId
    edge_obj = EdgeId(source=u, target=v, key=d.get('key', 0) if 'key' in d else list(graph[u][v].keys())[0])
    features = feature_resolver.resolve(edge_obj, context)
    cost_res = prediction_orchestrator.predict_edge(
        edge_id=edge_obj, 
        graph=graph, 
        traffic_features=features.traffic_features, 
        safety_features=features.safety_features
    )
    
    print(f"Edge: {name} ({edge_id})")
    print(f"Road Type: {d.get('highway')} (enc: {features.traffic_features['road_type_enc']})")
    print(f"Freeflow: {features.traffic_features['freeflow_speed_kmph']}")
    print(f"Traffic Pred Speed: (derived from {cost_res.travel_time_minutes:.4f} min travel time)")
    print(f"Safety Pred Score: {cost_res.safety_prediction.risk_score}")
    print(f"Edge length: {d.get('length', 0):.2f} m")
    print(f"Travel time: {cost_res.travel_time_minutes:.4f} min")
    print(f"Combined cost: {cost_res.combined_cost.combined_cost:.4f}")
    print("---")

print("\n--- Route Recommendation ---")
# Phase 10: Routing test
origin_coords = (13.0012, 80.2167)
dest_coords = (13.0100, 80.2200)

origin_node = ox.distance.nearest_nodes(graph, origin_coords[1], origin_coords[0])
dest_node = ox.distance.nearest_nodes(graph, dest_coords[1], dest_coords[0])

candidate_selector = CandidateEdgeSelector(graph, buffer_m=2000.0)

app = DynamicRoutingApp(
    graph=graph,
    feature_resolver=feature_resolver,
    prediction_orchestrator=prediction_orchestrator,
    candidate_selector=candidate_selector
)

try:
    route = app.route(origin_node, (dest_node,), context)
    print(f"Route success: {route.success}")
    print(f"Distance: {route.total_distance_m:.2f}")
    print(f"Estimated travel time: {route.total_duration_s/60.0:.2f} min")
    print(f"Total cost: {route.total_cost:.4f}")
except Exception as e:
    print(f"Route failed: {e}")

print("\n--- Multi-Stop Routing ---")
stop1_node = ox.distance.nearest_nodes(graph, 80.2180, 13.0050)
stop2_node = ox.distance.nearest_nodes(graph, 80.2190, 13.0080)

try:
    multi_route = app.route(origin_node, (stop1_node, stop2_node), context)
    print(f"Success: {multi_route.success}")
    print(f"Leg count: {len(multi_route.routes)}")
    print(f"Selected sequence: {multi_route.delivery_sequence}")
    print(f"Distance: {multi_route.total_distance_m:.2f}")
except Exception as e:
    print(f"Multi-stop failed: {e}")

print("=== END VERIFICATION ===")
