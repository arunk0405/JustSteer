import math
import logging
from typing import Set, Tuple
import networkx as nx

from src.routing.edge_cost import EdgeId
from src.routing.exceptions import InvalidNodeError, GraphUnavailableError

logger = logging.getLogger(__name__)

class CandidateEdgeSelector:
    """
    Selects a geographic subset of physical OSM edges relevant to a delivery request.
    This component identifies candidate edges for future dynamic cost generation.
    It does not calculate routes, dynamic costs, or mutate the underlying graph.
    """

    def __init__(self, graph: nx.MultiDiGraph, buffer_m: float = 2000.0):
        if graph is None:
            raise GraphUnavailableError("Cannot initialize CandidateEdgeSelector with None graph.")
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError(f"Graph must be a NetworkX MultiDiGraph, got {type(graph).__name__}.")
        
        if not isinstance(buffer_m, (int, float)) or isinstance(buffer_m, bool):
            raise TypeError("buffer_m must be a finite non-negative number.")
        if not math.isfinite(buffer_m) or buffer_m < 0:
            raise ValueError("buffer_m must be a finite non-negative number.")
            
        self.graph = graph
        self.buffer_m = float(buffer_m)

    def _meters_to_degrees(self, buffer_m: float, max_abs_lat: float) -> Tuple[float, float]:
        """
        Converts a buffer in meters to approximate degree deltas for latitude and longitude.
        Assumes spherical Earth. 1 degree of latitude is approx 111,320 meters.
        1 degree of longitude varies with latitude (approx 111,320 * cos(latitude)).
        """
        lat_m_per_degree = 111320.0
        lat_delta = buffer_m / lat_m_per_degree
        
        # Avoid division by zero near the poles by capping latitude
        capped_lat = min(abs(max_abs_lat), 89.9)
        lon_m_per_degree = lat_m_per_degree * math.cos(math.radians(capped_lat))
        lon_delta = buffer_m / lon_m_per_degree
        
        return lat_delta, lon_delta

    def select(self, start_node: int, delivery_nodes: Tuple[int, ...]) -> Set[EdgeId]:
        """
        Identify the physical directed edges located within the buffered geographic region
        containing the start_node and all delivery_nodes.
        
        Args:
            start_node: The starting node ID.
            delivery_nodes: A tuple of delivery node IDs. Can be empty.
            
        Returns:
            A set of EdgeId objects representing the candidate directed edges.
            
        Raises:
            InvalidNodeError: If any of the provided nodes do not exist in the graph.
            ValueError: If a provided node is missing coordinate attributes 'x' or 'y'.
        """
        nodes_to_check = (start_node,) + delivery_nodes
        
        lats = []
        lons = []
        
        for node in nodes_to_check:
            if node not in self.graph:
                raise InvalidNodeError(f"Node {node} not found in the graph.")
                
            node_data = self.graph.nodes[node]
            if "y" not in node_data or "x" not in node_data:
                raise ValueError(f"Node {node} is missing coordinate 'x' or 'y'.")
                
            lats.append(node_data["y"])
            lons.append(node_data["x"])
            
        min_lat = min(lats)
        max_lat = max(lats)
        min_lon = min(lons)
        max_lon = max(lons)
        
        max_abs_lat = max(abs(min_lat), abs(max_lat))
        lat_delta, lon_delta = self._meters_to_degrees(self.buffer_m, max_abs_lat)
        
        lat_min_buffered = min_lat - lat_delta
        lat_max_buffered = max_lat + lat_delta
        lon_min_buffered = min_lon - lon_delta
        lon_max_buffered = max_lon + lon_delta
        
        candidate_edges = set()
        
        # We perform a full scan of the edges. While O(E), it is very fast in raw Python.
        # Edges are included only if BOTH endpoints fall within the buffered bounding box.
        for u, v, key in self.graph.edges(keys=True):
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]
            
            u_y, u_x = u_data.get("y"), u_data.get("x")
            v_y, v_x = v_data.get("y"), v_data.get("x")
            
            # Skip edges that reference nodes with incomplete coordinate data
            if u_y is None or u_x is None or v_y is None or v_x is None:
                continue
                
            u_in_box = (lat_min_buffered <= u_y <= lat_max_buffered) and (lon_min_buffered <= u_x <= lon_max_buffered)
            v_in_box = (lat_min_buffered <= v_y <= lat_max_buffered) and (lon_min_buffered <= v_x <= lon_max_buffered)
            
            if u_in_box and v_in_box:
                candidate_edges.add(EdgeId(source=u, target=v, key=key))
                
        return candidate_edges
