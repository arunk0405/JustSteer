from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Tuple
import networkx as nx

from src.api.schemas import (
    MultiStopRouteRequest,
    MultiStopRouteResponse,
    RouteLeg
)
from src.api.dependencies import get_routing_app, get_context_aggregator, get_graph
from src.routing.dynamic_routing_app import DynamicRoutingApp
from src.context.context_aggregator import ContextAggregator
from src.graph.coordinate_mapper import find_nearest_node, CoordinateValidationError, NodeMappingError
from src.routing.exceptions import RouteNotFoundError, InvalidNodeError, GraphUnavailableError
from src.routing.feature_resolver import RoutingContext, WeatherContext, SafetyHistory

router = APIRouter()

def _delivery_context_to_routing_context(delivery_ctx, requested_timestamp) -> RoutingContext:
    # Build weather context
    if delivery_ctx.weather:
        weather_ctx = WeatherContext(
            temperature_c=delivery_ctx.weather.temp_celsius,
            rain_1h_mm=delivery_ctx.weather.rain_1h_mm,
            wind_speed_kmh=delivery_ctx.weather.wind_speed_mps * 3.6, # Convert m/s to km/h
            visibility_m=delivery_ctx.weather.visibility_m,
            humidity_pct=delivery_ctx.weather.humidity_pct,
            rain_risk_score=delivery_ctx.weather.rain_risk_score,
            rain_forecast_hours=delivery_ctx.weather.rain_forecast_hours
        )
    else:
        # Fallback if weather is completely unavailable
        weather_ctx = WeatherContext(
            temperature_c=25.0,
            rain_1h_mm=0.0,
            wind_speed_kmh=10.0,
            visibility_m=10000.0,
            humidity_pct=50.0,
            rain_risk_score=0.0,
            rain_forecast_hours=0
        )
        
    # Build traffic index
    congestion_index = 0.85 # default to free flow
    if delivery_ctx.traffic:
        congestion_index = delivery_ctx.traffic.congestion_index
        
    # Build safety history - F7 Demo Readiness Mock
    safety_history = SafetyHistory(
        accidents_total_historical=5,
        fatal_accidents_historical=0,
        injuries_historical=2,
        historical_risk_score=45.0
    )

    return RoutingContext(
        timestamp=requested_timestamp,
        weather=weather_ctx,
        congestion_index=congestion_index,
        safety_history=safety_history
    )

def _format_geojson_geometry(geometry: List[Tuple[float, float]]):
    if not geometry:
        return None
    return {
        "type": "LineString",
        "coordinates": [[lon, lat] for lat, lon in geometry] # GeoJSON needs [lon, lat]
    }

# We define this synchronously. FastAPI will run it in a threadpool to prevent blocking the event loop.
@router.post("/v1/route/multi-stop", response_model=MultiStopRouteResponse)
def route_multi_stop(
    request: MultiStopRouteRequest,
    routing_app: DynamicRoutingApp = Depends(get_routing_app),
    context_aggregator: ContextAggregator = Depends(get_context_aggregator),
    graph: nx.MultiDiGraph = Depends(get_graph)
):
    try:
        # 1. Snap coordinates to graph nodes
        origin_node = find_nearest_node(graph, request.origin.latitude, request.origin.longitude).node_id
        
        delivery_nodes = []
        # We maintain a mapping from node -> List[Coordinate] to map the optimized route back to the user's input.
        node_to_coords = {origin_node: [request.origin]}
        
        for d in request.deliveries:
            node = find_nearest_node(graph, d.latitude, d.longitude).node_id
            delivery_nodes.append(node)
            if node not in node_to_coords:
                node_to_coords[node] = []
            node_to_coords[node].append(d)
            
        delivery_nodes_tuple = tuple(delivery_nodes)
        
        # 2. Context Aggregation (from origin coordinate)
        # We aggregate context once using the origin to form the single request context.
        req_time = request.timestamp or datetime.now(timezone.utc)
        delivery_ctx = context_aggregator.get_context(request.origin.latitude, request.origin.longitude)
        
        routing_ctx = _delivery_context_to_routing_context(delivery_ctx, req_time)
        
        # 3. Dynamic Routing Execution
        result = routing_app.route(
            start_node=origin_node,
            delivery_nodes=delivery_nodes_tuple,
            context=routing_ctx
        )
        
        # 4. Result Serialization
        # Map node IDs back to requested coordinates using our pool
        mapping_pool = {k: list(v) for k, v in node_to_coords.items()}
        # We don't pop origin from the pool since it's the start, just the deliveries
        if origin_node in mapping_pool:
            mapping_pool[origin_node].pop(0) # Remove origin so it doesn't get consumed as a delivery incorrectly if origin == delivery
            
        optimized_stops = []
        for node in result.delivery_sequence.sequence:
            coord = mapping_pool[node].pop(0)
            optimized_stops.append(coord)
            
        serialized_routes = []
        current_coord = request.origin
        
        for i, leg in enumerate(result.routes):
            leg_end_coord = optimized_stops[i]
            
            geo_json = _format_geojson_geometry(leg.geometry)
            
            serialized_routes.append(
                RouteLeg(
                    start_location=current_coord,
                    end_location=leg_end_coord,
                    distance_m=leg.distance_m,
                    estimated_duration_s=leg.duration_s,
                    geometry=geo_json
                )
            )
            current_coord = leg_end_coord
            
        from src.api.schemas import RouteSummary, ExplainabilityMetadata, TrafficInfo, SafetyInfo, WeightsInfo, ModelInfo
        summary = RouteSummary(
            total_distance_m=result.total_distance_m,
            estimated_duration_s=result.total_duration_s,
            stop_count=len(optimized_stops)
        )
            
        explainability = None
        if result.explainability_metrics:
            metrics = result.explainability_metrics
            explainability = ExplainabilityMetadata(
                candidate_edges_evaluated=metrics["candidate_edges_evaluated"],
                a_star_expanded_nodes=result.expanded_nodes,
                evaluated_permutations=result.evaluated_permutations,
                traffic=TrafficInfo(average_predicted_travel_time_s=metrics["average_predicted_travel_time_s"]),
                safety=SafetyInfo(average_risk=metrics["average_safety_risk"]),
                weights=WeightsInfo(
                    travel_time=metrics["average_travel_time_weight"],
                    safety=metrics["average_safety_weight"]
                ),
                models=ModelInfo(
                    traffic=metrics["models"]["traffic"],
                    safety=metrics["models"]["safety"]
                )
            )

        return MultiStopRouteResponse(
            success=result.success,
            summary=summary,
            optimized_stops=optimized_stops,
            legs=serialized_routes,
            explainability=explainability
        )

    except (CoordinateValidationError, NodeMappingError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except InvalidNodeError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RouteNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except GraphUnavailableError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Graph infrastructure unavailable")
    except Exception as e:
        # Prevent internal details from leaking on unexpected exceptions
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during routing")
