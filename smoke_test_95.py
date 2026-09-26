import networkx as nx
from datetime import datetime
from src.routing.edge_cost import EdgeId
from src.routing.edge_cost_store import EdgeCostStore
from src.routing.feature_resolver import FeatureResolver, RoutingContext, WeatherContext, SafetyHistory
from src.routing.prediction_providers import TrafficPredictionProvider, SafetyPredictionProvider
from src.routing.weight_calculator import WeightCalculator, NormalizationConfig
from src.routing.prediction_orchestrator import PredictionOrchestrator
from src.routing.dynamic_cost_generator import DynamicCostGenerator

def run_smoke_test():
    print("Loading graph...")
    graph = nx.read_graphml("data/road_network/chennai_drive.graphml")
    graph = nx.relabel_nodes(graph, int)
    
    # Convert edge lengths to floats (graphml stores them as strings)
    for u, v, k, d in graph.edges(keys=True, data=True):
        if 'length' in d:
            d['length'] = float(d['length'])
            
    print(f"Graph loaded with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
    
    print("Initializing components...")
    resolver = FeatureResolver(graph)
    traffic_provider = TrafficPredictionProvider()
    safety_provider = SafetyPredictionProvider()
    weight_calculator = WeightCalculator(NormalizationConfig(travel_time_p5=1.0, travel_time_p95=10.0, travel_time_weight=0.5, safety_weight=0.5))
    orchestrator = PredictionOrchestrator(traffic_provider, safety_provider, weight_calculator)
    store = EdgeCostStore()
    
    generator = DynamicCostGenerator(graph, resolver, orchestrator, store)
    
    # Pick a few sample edges
    edges_list = list(graph.edges(keys=True))
    sample_edges = edges_list[:10]
    candidate_edges = [EdgeId(int(u), int(v), k) for u, v, k in sample_edges]
    
    # Context with synthetic safety data since it's required
    weather = WeatherContext(
        temperature_c=30.0,
        rain_1h_mm=0.0,
        wind_speed_kmh=10.0,
        visibility_m=10000.0,
        humidity_pct=60.0,
        rain_risk_score=0.1,
        rain_forecast_hours=0
    )
    safety_history = SafetyHistory(
        accidents_total_historical=1,
        fatal_accidents_historical=0,
        injuries_historical=0,
        historical_risk_score=20.0
    )
    context = RoutingContext(
        timestamp=datetime.now(),
        weather=weather,
        congestion_index=1.0,
        safety_history=safety_history
    )
    
    print(f"Running generation for {len(candidate_edges)} candidate edges...")
    start_time = datetime.now()
    result = generator.generate(candidate_edges, context)
    duration = (datetime.now() - start_time).total_seconds()
    
    print(f"Processed {result.processed_count} edges in {duration:.2f} seconds.")
    print(f"Successful: {len(result.successful_edges)}")
    print(f"Failed: {len(result.failed_edges)}")
    
    if result.failed_edges:
        print("Errors:")
        for edge, err in result.errors.items():
            print(f"  {edge}: {err}")
            
    # Verify graph mutation
    print("Verifying graph immutability...")
    for u, v, k in sample_edges:
        data = graph[u][v][k]
        if 'final_cost' in data or 'dynamic_cost' in data:
            print(f"ERROR: Graph mutated on edge ({u}, {v}, {k})!")
            return False
            
    print("Graph mutation check passed.")
    print("Smoke test completed successfully!")
    return True

if __name__ == "__main__":
    run_smoke_test()
