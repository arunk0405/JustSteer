import math
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class RoadNetworkValidationError(ValueError):
    """Exception raised when a road network graph fails structural validation."""
    pass

def validate_road_network(graph: nx.MultiDiGraph) -> None:
    """
    Validates the structure of a drivable road network graph.
    
    OSM graph
       ↓
    graph validation
       ↓
    validated static road network

    Key validations:
    - Type is nx.MultiDiGraph (directed, allows parallel edges)
    - Not empty
    - Nodes contain finite geographic coordinates (x=longitude, y=latitude)
    - Edges contain finite, non-negative length
    
    Note:
    - This validation is read-only and does NOT mutate the graph.
    - Optional OSM attributes (name, highway, etc.) may be missing and are not required.
    - Traffic/safety/routing costs are not validated at this stage.
    """
    # 1. Validate graph type
    if not isinstance(graph, nx.MultiDiGraph):
        raise RoadNetworkValidationError(f"Road network must be a NetworkX MultiDiGraph. Got {type(graph)}")

    # 2. Validate graph is not empty
    if graph.number_of_nodes() == 0:
        raise RoadNetworkValidationError("Road network contains no nodes.")
    if graph.number_of_edges() == 0:
        raise RoadNetworkValidationError("Road network contains no edges.")

    # 3. Validate node structure (coordinates)
    for node, data in graph.nodes(data=True):
        if "x" not in data or "y" not in data:
            raise RoadNetworkValidationError(f"Node {node} is missing coordinate 'x' or 'y'.")
        
        x = data["x"]
        y = data["y"]
        
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise RoadNetworkValidationError(f"Node {node} has non-numeric coordinates: x={x}, y={y}.")
            
        if not math.isfinite(x) or not math.isfinite(y):
            raise RoadNetworkValidationError(f"Node {node} has non-finite coordinates: x={x}, y={y}.")
            
        if not (-180.0 <= x <= 180.0):
            raise RoadNetworkValidationError(f"Node {node} has invalid longitude 'x': {x}.")
            
        if not (-90.0 <= y <= 90.0):
            raise RoadNetworkValidationError(f"Node {node} has invalid latitude 'y': {y}.")

    # 4. Validate edge structure and references
    for u, v, key, data in graph.edges(keys=True, data=True):
        # Implicitly, u and v are valid node references if we can iterate over them, 
        # but we explicitly double check they exist in the node set just to be absolutely certain.
        if u not in graph:
            raise RoadNetworkValidationError(f"Edge ({u}, {v}, {key}) references a missing source node: {u}.")
        if v not in graph:
            raise RoadNetworkValidationError(f"Edge ({u}, {v}, {key}) references a missing target node: {v}.")
            
        # Validate edge length
        if "length" not in data:
            raise RoadNetworkValidationError(f"Edge ({u}, {v}, {key}) is missing required 'length' attribute.")
            
        length = data["length"]
        if not isinstance(length, (int, float)):
            raise RoadNetworkValidationError(f"Edge ({u}, {v}, {key}) has non-numeric length: {length}.")
            
        if not math.isfinite(length):
            raise RoadNetworkValidationError(f"Edge ({u}, {v}, {key}) has non-finite length: {length}.")
            
        if length < 0:
            raise RoadNetworkValidationError(f"Edge ({u}, {v}, {key}) has negative length: {length}.")

    logger.info("Road network validation passed successfully.")
