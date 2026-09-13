import math
from dataclasses import dataclass
import networkx as nx
import osmnx as ox
from typing import Optional

class CoordinateValidationError(ValueError):
    """Raised when input geographic coordinates are invalid."""
    pass

class NodeMappingError(ValueError):
    """Raised when node mapping fails or exceeds snap distance limits."""
    pass

@dataclass(frozen=True)
class NodeMatch:
    """Represents a matched node in the OSM road network."""
    node_id: int
    input_latitude: float
    input_longitude: float
    matched_latitude: float
    matched_longitude: float
    snap_distance_m: float

def find_nearest_node(
    graph: nx.MultiDiGraph,
    latitude: float,
    longitude: float,
    max_snap_distance_m: Optional[float] = None
) -> NodeMatch:
    """
    Finds the nearest OSM node in the graph for a given geographic coordinate.
    
    External coordinate convention:
    (latitude, longitude)
    
    OSM node convention:
    x = longitude
    y = latitude
    
    Args:
        graph: The validated NetworkX MultiDiGraph (from OSMnx).
        latitude: Input geographic latitude (-90 to 90).
        longitude: Input geographic longitude (-180 to 180).
        max_snap_distance_m: Optional maximum allowed snap distance in meters.
                             If provided and nearest node is further, raises NodeMappingError.
                             
    Returns:
        NodeMatch: The matched node details including distance.
        
    Raises:
        CoordinateValidationError: If latitude or longitude is invalid.
        NodeMappingError: If mapping fails or exceeds max_snap_distance_m.
        RoadNetworkValidationError: If the provided graph is fundamentally invalid.
    """
    # 1. Validate coordinates
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        raise CoordinateValidationError("Coordinates must be numeric.")
        
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise CoordinateValidationError("Coordinates must be finite numbers.")
        
    if not (-90.0 <= latitude <= 90.0):
        raise CoordinateValidationError(f"Invalid latitude: {latitude}. Must be between -90 and 90.")
        
    if not (-180.0 <= longitude <= 180.0):
        raise CoordinateValidationError(f"Invalid longitude: {longitude}. Must be between -180 and 180.")
        
    if max_snap_distance_m is not None:
        if not isinstance(max_snap_distance_m, (int, float)) or max_snap_distance_m < 0:
            raise ValueError("max_snap_distance_m must be a non-negative number.")
            
    # 2. Validate graph (lightweight)
    from graph.graph_validator import validate_road_network
    # Use existing validation. In a real system, the graph is expected to be validated on load,
    # but we enforce it here as requested to reuse existing validation.
    validate_road_network(graph)
    
    if "crs" not in graph.graph:
        # OSMnx nearest_nodes requires CRS for correct geographic distance calculation.
        # Fallback to WGS84 if it's missing (though validate_road_network could technically enforce this)
        graph.graph["crs"] = "epsg:4326"
    
    # 3. Use OSMnx nearest node calculation
    # X = longitude, Y = latitude
    try:
        node_id, dist = ox.distance.nearest_nodes(
            graph, 
            X=longitude, 
            Y=latitude, 
            return_dist=True
        )
    except Exception as e:
        raise NodeMappingError(f"Failed to find nearest node using OSMnx: {e}") from e
        
    # 4. Check snap distance constraint
    if max_snap_distance_m is not None and dist > max_snap_distance_m:
        raise NodeMappingError(
            f"Nearest node ({dist:.2f}m) exceeds maximum allowed snap distance ({max_snap_distance_m}m)."
        )
        
    # 5. Extract matched coordinates (x=lon, y=lat)
    matched_lon = graph.nodes[node_id]["x"]
    matched_lat = graph.nodes[node_id]["y"]
    
    return NodeMatch(
        node_id=node_id,
        input_latitude=latitude,
        input_longitude=longitude,
        matched_latitude=matched_lat,
        matched_longitude=matched_lon,
        snap_distance_m=dist
    )
