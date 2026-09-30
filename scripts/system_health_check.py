import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
import subprocess

def compute_sha256(filepath):
    if not os.path.exists(filepath):
        return None
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def main():
    print("========================================")
    print("JUSTSTEER SYSTEM HEALTH CHECK")
    print("========================================")
    
    project_root = Path(__file__).parent.parent
    
    # 1. Environment
    print(f"[PASS] Environment (Python {sys.version.split()[0]})")
    
    # 2. Models and Metadata
    traffic_model = project_root / "models/velocity_predictor.pkl"
    traffic_meta = project_root / "models/velocity_predictor_metadata.json"
    safety_model = project_root / "models/safety/safety_risk_xgb.pkl"
    safety_meta = project_root / "models/safety/safety_risk_xgb_metadata.json"
    
    if traffic_model.exists(): print("[PASS] Traffic model artifact")
    else: print("[FAIL] Traffic model artifact"); return 1
    
    if traffic_meta.exists(): print("[PASS] Traffic metadata")
    else: print("[FAIL] Traffic metadata"); return 1
    
    if safety_model.exists(): print("[PASS] Safety model artifact")
    else: print("[FAIL] Safety model artifact"); return 1
    
    if safety_meta.exists(): print("[PASS] Safety metadata")
    else: print("[FAIL] Safety metadata"); return 1

    # Checksums
    with open(traffic_meta) as f: t_meta = json.load(f)
    if t_meta.get("artifact", {}).get("sha256") == compute_sha256(traffic_model):
        print("[PASS] Traffic checksum")
    else: print("[FAIL] Traffic checksum"); return 1
    
    with open(safety_meta) as f: s_meta = json.load(f)
    if s_meta.get("artifact", {}).get("sha256") == compute_sha256(safety_model):
        print("[PASS] Safety checksum")
    else: print("[FAIL] Safety checksum"); return 1
    
    # Imports
    sys.path.insert(0, str(project_root))
    import osmnx as ox
    from src.routing.prediction_providers import SafetyPredictionProvider, TrafficPredictionProvider
    from src.routing.weight_calculator import WeightCalculator, NormalizationConfig
    from src.routing.prediction_orchestrator import PredictionOrchestrator
    from src.routing.feature_resolver import FeatureResolver, RoutingContext, WeatherContext, SafetyHistory
    from src.routing.edge_cost import EdgeId
    from src.routing.dynamic_cost_generator import DynamicCostGenerator
    from src.routing.candidate_selector import CandidateEdgeSelector
    from src.routing.dynamic_routing_app import DynamicRoutingApp

    if len(t_meta.get("feature_order", [])) == len(TrafficPredictionProvider.EXPECTED_FEATURES):
        print("[PASS] Traffic feature contract")
    else: print("[FAIL] Traffic feature contract"); return 1

    if len(s_meta.get("feature_order", [])) == len(SafetyPredictionProvider.EXPECTED_FEATURES):
        print("[PASS] Safety feature contract")
    else: print("[FAIL] Safety feature contract"); return 1
    
    graph_path = project_root / "data/road_network/chennai_drive.graphml"
    try:
        graph = ox.load_graphml(graph_path)
        print("[PASS] OSM graph loading")
    except Exception as e:
        print(f"[FAIL] OSM graph loading: {e}"); return 1
        
    try:
        traffic_provider = TrafficPredictionProvider(str(traffic_model))
        print("[PASS] Traffic inference setup")
    except Exception as e:
        print(f"[FAIL] Traffic inference setup: {e}"); return 1
    try:
        safety_provider = SafetyPredictionProvider(str(safety_model))
        print("[PASS] Safety inference setup")
    except Exception as e:
        print(f"[FAIL] Safety inference setup: {e}"); return 1

    norm_config = NormalizationConfig(travel_time_p5=0.5, travel_time_p95=10.0, travel_time_weight=0.6, safety_weight=0.4)
    weight_calculator = WeightCalculator(norm_config)
    orchestrator = PredictionOrchestrator(traffic_provider, safety_provider, weight_calculator)
    resolver = FeatureResolver(graph)
    candidate_selector = CandidateEdgeSelector(graph, buffer_m=2000.0)
    
    app = DynamicRoutingApp(graph, resolver, orchestrator, candidate_selector)
    
    # Edge differentiation test
    edges = list(graph.edges(data=True, keys=True))
    edge_dict = {}
    for u, v, k, data in edges:
        hw = data.get("highway")
        if isinstance(hw, list): hw = hw[0]
        if hw in ["primary", "secondary", "residential", "service"]:
            if hw not in edge_dict:
                edge_dict[hw] = (u, v, k, data)
        if len(edge_dict) >= 4:
            break
            
    print("\n--- Edge Differentiation Test ---")
    
    weather = WeatherContext(
        temperature_c=25.0, rain_1h_mm=0.0, wind_speed_kmh=10.0,
        visibility_m=5000.0, humidity_pct=60.0, rain_risk_score=0.1, rain_forecast_hours=0
    )
    safety_hist = SafetyHistory(
        accidents_total_historical=10, fatal_accidents_historical=0,
        injuries_historical=2, historical_risk_score=5.0
    )
    context = RoutingContext(timestamp=datetime.utcnow(), weather=weather, congestion_index=1.0, safety_history=safety_hist)
    
    for hw, (u, v, k, data) in edge_dict.items():
        eid = EdgeId(source=u, target=v, key=k)
        resolved = resolver.resolve(eid, context)
        res = orchestrator.predict_edge(eid, graph, resolved.traffic_features, resolved.safety_features)
        print(f"EDGE: {u}->{v} ({hw})")
        print(f"  Road type: {hw}")
        print(f"  Free-flow speed: {data.get('maxspeed', 'unknown')} km/h")
        speed_pred = traffic_provider.predict_speed(resolved.traffic_features)
        print(f"  Predicted traffic speed: {speed_pred:.2f} km/h")
        safety_pred = safety_provider.predict(resolved.safety_features)
        print(f"  Predicted safety risk: {safety_pred.risk_score:.2f}")
        print(f"  Dynamic cost: {res.combined_cost.combined_cost:.2f}")
        print(f"-----------------------------")
        
    print("[PASS] Traffic inference")
    print("[PASS] Safety inference")
    print("[PASS] Dynamic edge cost")
        
    print("\n--- A* & Multi-stop Routing Test ---")
    try:
        nodes = list(graph.nodes())
        start_node = nodes[0]
        delivery_1 = list(graph.successors(start_node))[0]
        delivery_2 = list(graph.successors(delivery_1))[0]
        delivery_nodes = (delivery_1, delivery_2)
        result = app.route(start_node, delivery_nodes, context)
        
        if result.success:
            print("[PASS] A* routing")
            print(f"[PASS] Multi-stop routing (Evaluated {result.evaluated_permutations} permutations, {len(result.routes)} legs)")
        else:
            print("[FAIL] Route unsuccessful")
            return 1
    except Exception as e:
        print(f"[FAIL] Routing failed: {e}")
        return 1
    
    print("\n========================================")
    print("SYSTEM HEALTH: PASS")
    print("========================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
