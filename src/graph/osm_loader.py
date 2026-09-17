import logging
import osmnx as ox
import networkx as nx
from pathlib import Path

logger = logging.getLogger(__name__)

def download_road_network(place: str) -> nx.MultiDiGraph:
    """
    Downloads the drivable road network for a given place from OpenStreetMap.
    
    Args:
        place: A string identifying the place (e.g., "Chennai, Tamil Nadu, India").
        
    Returns:
        A NetworkX MultiDiGraph containing the road network.
        
    Raises:
        Exception: If the OSM road network download fails or the place cannot be resolved.
    """
    logger.info(f"Downloading drivable road network for '{place}'...")
    try:
        # Use OSMnx to download the graph for driving
        graph = ox.graph_from_place(place, network_type="drive")
        logger.info(f"Successfully downloaded road network for '{place}'. Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        return graph
    except Exception as e:
        logger.error(f"Failed to download road network for '{place}': {e}")
        # Raise a clear exception as requested
        raise RuntimeError(f"OSM road network download failed for place: '{place}'. Details: {e}") from e

def save_road_network(graph: nx.MultiDiGraph, path: str | Path, overwrite: bool = False) -> None:
    """
    Validates and saves the road network to disk.
    
    Args:
        graph: The NetworkX MultiDiGraph to save.
        path: The file path to save the graph to (e.g., data/road_network/chennai.graphml).
        overwrite: If True, overwrites the file if it exists. If False, raises FileExistsError.
        
    Raises:
        RoadNetworkValidationError: If the graph is invalid.
        FileExistsError: If the file exists and overwrite is False.
    """
    from .graph_validator import validate_road_network
    
    # 1. Validate before saving
    validate_road_network(graph)
    
    path_obj = Path(path)
    
    # 2. Check if file exists and handle overwrite logic
    if path_obj.exists() and not overwrite:
        raise FileExistsError(f"File already exists at {path_obj}. Use overwrite=True to replace it.")
        
    # 3. Create parent directories if they don't exist
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # 4. Save using OSMnx's graphml exporter (preserves MultiDiGraph, parallel edges, geometry)
    try:
        ox.save_graphml(graph, filepath=path_obj)
        logger.info(f"Successfully saved road network to {path_obj}")
    except Exception as e:
        logger.error(f"Failed to save road network to {path_obj}: {e}")
        raise RuntimeError(f"Failed to save road network to {path_obj}: {e}") from e

def load_road_network(path: str | Path) -> nx.MultiDiGraph:
    """
    Loads a road network from disk and validates it.
    
    Args:
        path: The file path to load the graph from.
        
    Returns:
        The validated NetworkX MultiDiGraph.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file is unreadable or corrupt.
        RoadNetworkValidationError: If the loaded graph fails structural validation.
    """
    from .graph_validator import validate_road_network
    
    path_obj = Path(path)
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"Graph file not found at {path_obj}")
        
    logger.info(f"Loading road network from {path_obj}...")
    try:
        graph = ox.load_graphml(filepath=path_obj)
    except Exception as e:
        logger.error(f"Failed to load road network from {path_obj}: {e}")
        raise RuntimeError(f"Failed to load road network from {path_obj}: {e}") from e
        
    # Ensure it's a MultiDiGraph (OSMnx load_graphml should return MultiDiGraph, but verify)
    if not isinstance(graph, nx.MultiDiGraph):
        raise RuntimeError(f"Loaded graph is not a MultiDiGraph, got {type(graph)}")
        
    # Validate the loaded graph before returning
    validate_road_network(graph)
    
    logger.info(f"Successfully loaded and validated road network from {path_obj}. Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    return graph
