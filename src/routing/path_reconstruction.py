from typing import Any, Dict, List, Tuple
import networkx as nx

def reconstruct_path(
    came_from: Dict[Any, Tuple[Any, Any]], 
    current: Any
) -> Tuple[List[Any], List[Tuple[Any, Any, Any]]]:
    """
    Reconstructs the node and edge path from the came_from map.
    
    Args:
        came_from: Map of node -> (predecessor_node, selected_edge_key)
        current: The destination node to work backwards from
        
    Returns:
        Tuple of (path_nodes, path_edges)
        path_nodes: [node1, node2, ..., nodeN]
        path_edges: [(u, v, key), ...]
    """
    path_nodes = [current]
    path_edges = []
    
    while current in came_from:
        prev_node, edge_key = came_from[current]
        
        path_nodes.append(prev_node)
        path_edges.append((prev_node, current, edge_key))
        
        current = prev_node
        
    path_nodes.reverse()
    path_edges.reverse()
    
    return path_nodes, path_edges

def extract_geometry_and_distance(
    graph: nx.MultiDiGraph, 
    path_edges: List[Tuple[Any, Any, Any]]
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Extracts total geographic distance and combined line geometry.
    
    Args:
        graph: The OSM networkx graph (MultiDiGraph)
        path_edges: The sequence of selected edges (u, v, key)
        
    Returns:
        (route_geometry, total_distance)
        route_geometry: List of (lon, lat) coordinates without duplicates
        total_distance: Total physical distance in meters (if length exists)
    """
    geometry = []
    total_distance = 0.0
    
    for idx, (u, v, key) in enumerate(path_edges):
        edge_data = graph.edges[u, v, key]
        
        # Distance
        length = edge_data.get("length", 0.0)
        total_distance += float(length)
        
        # Geometry extraction
        # If 'geometry' exists (shapely LineString or list of coords)
        # OSMnx puts shapely LineString in 'geometry'
        edge_geom = []
        if "geometry" in edge_data:
            geom = edge_data["geometry"]
            # If shapely LineString
            if hasattr(geom, "coords"):
                edge_geom = list(geom.coords)
            elif isinstance(geom, list) or isinstance(geom, tuple):
                edge_geom = list(geom)
        else:
            # Fallback to straight line between u and v
            # Expecting nodes to have 'x' and 'y' for lon/lat
            u_node = graph.nodes[u]
            v_node = graph.nodes[v]
            if "x" in u_node and "y" in u_node and "x" in v_node and "y" in v_node:
                edge_geom = [(u_node["x"], u_node["y"]), (v_node["x"], v_node["y"])]
                
        # Append to master geometry, avoiding duplicate endpoints
        if not geometry:
            geometry.extend(edge_geom)
        else:
            # If the first coordinate of the new edge matches the last of the route, skip it
            # (to maintain continuous Linestring without repeated points)
            if edge_geom:
                # Ensure direction corresponds to u -> v
                u_node = graph.nodes.get(u, {})
                v_node = graph.nodes.get(v, {})
                
                # Check if it needs reversing by comparing distances to u
                if "x" in u_node and "y" in u_node:
                    u_coord = (u_node["x"], u_node["y"])
                    
                    # distance from start of geom to u
                    start_dist = _sq_dist(edge_geom[0], u_coord)
                    # distance from end of geom to u
                    end_dist = _sq_dist(edge_geom[-1], u_coord)
                    
                    if end_dist < start_dist:
                        # Geometry is stored backwards relative to u->v traversal
                        edge_geom.reverse()
                        
                if _coords_match(geometry[-1], edge_geom[0]):
                    geometry.extend(edge_geom[1:])
                else:
                    geometry.extend(edge_geom)
                
    return geometry, total_distance

def _sq_dist(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    return (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2

def _coords_match(c1: Tuple[float, float], c2: Tuple[float, float], tol: float=1e-6) -> bool:
    return abs(c1[0] - c2[0]) < tol and abs(c1[1] - c2[1]) < tol
