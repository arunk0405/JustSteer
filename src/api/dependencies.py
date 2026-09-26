import os
import networkx as nx
import osmnx as ox
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

from src.routing.feature_resolver import FeatureResolver
from src.routing.prediction_orchestrator import PredictionOrchestrator
from src.routing.prediction_providers import SafetyPredictionProvider, TrafficPredictionProvider
from src.routing.weight_calculator import WeightCalculator, NormalizationConfig
from src.routing.candidate_selector import CandidateEdgeSelector
from src.routing.dynamic_routing_app import DynamicRoutingApp
from src.context.zone_registry import ZoneRegistry
from src.context.weather_fetcher import WeatherFetcher
from src.context.traffic_fetcher import TrafficFetcher
from src.context.context_aggregator import ContextAggregator

class AppState:
    graph: nx.MultiDiGraph
    dynamic_routing_app: DynamicRoutingApp
    context_aggregator: ContextAggregator

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load Graph (read-only, shared)
    graph_path = os.getenv("GRAPH_PATH", "data/road_network/chennai_drive.graphml")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph not found at {graph_path}")
    
    graph = ox.load_graphml(graph_path)
    
    # 2. Init ML Providers & Orchestrator
    safety_model_path = os.getenv("SAFETY_MODEL_PATH", "models/safety/safety_risk_xgb.pkl")
    traffic_model_path = os.getenv("TRAFFIC_MODEL_PATH", "models/velocity_predictor.pkl")
    
    # Instantiate ML models once at startup
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
    
    # 3. Candidate selector (read-only bounding box logic)
    candidate_selector = CandidateEdgeSelector(graph, buffer_m=2000.0)
    
    # 4. Instantiate singleton DynamicRoutingApp orchestrator
    app.state.dynamic_routing_app = DynamicRoutingApp(
        graph=graph,
        feature_resolver=feature_resolver,
        prediction_orchestrator=prediction_orchestrator,
        candidate_selector=candidate_selector
    )
    app.state.graph = graph
    
    # 5. Initialize Context Aggregator
    zone_yaml = os.getenv("ZONE_YAML_PATH", "configs/zones/chennai.yaml")
    zone_registry = ZoneRegistry(zone_yaml)
    
    # Use dry_run=True by default for API if no keys provided, to safely fall back to heuristics
    weather_fetcher = WeatherFetcher(dry_run=True)
    traffic_fetcher = TrafficFetcher(dry_run=True)
    
    app.state.context_aggregator = ContextAggregator(
        registry=zone_registry,
        weather_fetcher=weather_fetcher,
        traffic_fetcher=traffic_fetcher
    )
    
    yield
    # Cleanup on shutdown (if any needed in the future)

def get_routing_app(request: Request) -> DynamicRoutingApp:
    return request.app.state.dynamic_routing_app

def get_context_aggregator(request: Request) -> ContextAggregator:
    return request.app.state.context_aggregator

def get_graph(request: Request) -> nx.MultiDiGraph:
    return request.app.state.graph
