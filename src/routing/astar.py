import heapq
import time
import math
import networkx as nx
from typing import Any, Dict, Tuple, Optional

from src.routing.exceptions import (
    GraphUnavailableError, 
    InvalidNodeError, 
    RouteNotFoundError,
    InvalidEdgeCostError
)
from src.routing.route_result import RouteResult
from src.routing.heuristic import get_heuristic
from src.routing.path_reconstruction import reconstruct_path, extract_geometry_and_distance
from src.routing.edge_cost import EdgeId
from src.routing.cost_provider import EdgeCostProvider, StaticGraphCostProvider
class AStarRoutingEngine:
    def __init__(
        self, 
        graph: nx.MultiDiGraph, 
        cost_attribute: str = "final_cost", 
        heuristic_mode: str = "zero",
        max_expansions: int = 1000000,
        cost_provider: Optional[EdgeCostProvider] = None
    ):
        """
        Initializes the A* routing engine.
        
        Args:
            graph: The OSM road network (must be a NetworkX MultiDiGraph).
            cost_attribute: The edge attribute containing the final ML composite cost (legacy fallback).
            heuristic_mode: The heuristic to use (default: 'zero', which acts as Dijkstra).
            max_expansions: Safeguard against infinite loops / extremely large searches.
            cost_provider: Request-scoped provider for dynamic costs. If None, uses static graph attribute.
        """
        if graph is None:
            raise GraphUnavailableError("Cannot initialize AStarRoutingEngine with None graph.")
            
        self.graph = graph
        self.cost_attribute = cost_attribute
        self.heuristic_mode = heuristic_mode
        self.max_expansions = max_expansions
        self.heuristic_fn = get_heuristic(heuristic_mode)
        self.cost_provider = cost_provider or StaticGraphCostProvider(graph, cost_attribute)

    def route(self, source: Any, destination: Any) -> RouteResult:
        """
        Finds the lowest-cost path from source to destination.
        
        Args:
            source: Source node ID in the graph.
            destination: Destination node ID in the graph.
            
        Returns:
            RouteResult containing path, geometry, and cost.
            
        Raises:
            InvalidNodeError: If source or destination is not in the graph.
            RouteNotFoundError: If no path exists.
            InvalidEdgeCostError: If a negative or NaN cost is encountered.
        """
        start_time = time.time()
        
        # 1. Validate Nodes
        if source not in self.graph:
            raise InvalidNodeError(f"Source node {source} not found in graph.")
        if destination not in self.graph:
            raise InvalidNodeError(f"Destination node {destination} not found in graph.")
            
        # 2. Source == Destination
        if source == destination:
            return RouteResult(
                success=True,
                source_node=source,
                destination_node=destination,
                path_nodes=[source],
                path_edges=[],
                total_cost=0.0,
                distance_m=0.0,
                expanded_nodes=0,
                execution_time_ms=(time.time() - start_time) * 1000,
                geometry=[]
            )
            
        # 3. Initialize Data Structures
        open_set = []  # Priority queue: (f_score, counter, node)
        counter = 0    # Tie-breaker for stable sorting when f_scores are equal
        
        heapq.heappush(open_set, (0.0, counter, source))
        
        # Maps node -> (predecessor_node, selected_edge_key)
        came_from: Dict[Any, Tuple[Any, Any]] = {}
        
        # Maps node -> cheapest known cost from source to node
        g_score: Dict[Any, float] = {source: 0.0}
        
        expansions = 0
        
        # 4. Main Search Loop
        while open_set:
            if expansions >= self.max_expansions:
                raise RouteNotFoundError(f"Exceeded max expansions ({self.max_expansions}) before finding a route.")
                
            current_f, _, current = heapq.heappop(open_set)
            
            # Stale priority queue entry check
            # We compare current_f against the optimal f-score derived from the best known g_score.
            # (current_f > g_score[current] + h(current))
            optimal_f = g_score.get(current, float('inf')) + self.heuristic_fn(current, destination, self.graph)
            
            # Using a tiny tolerance for floating point stability
            if current_f > optimal_f + 1e-9:
                continue
            
            if current == destination:
                break
                
            expansions += 1
            
            # Evaluate all directed outgoing neighbors
            for neighbor in self.graph.successors(current):
                # A MultiDiGraph can have multiple parallel edges to the same neighbor.
                # We need to evaluate all of them and pick the one with the lowest valid cost.
                best_edge_key = None
                best_edge_cost = float('inf')
                
                # self.graph[current][neighbor] is a dict mapping edge_key -> edge_data
                for key, edge_data in self.graph[current][neighbor].items():
                    edge_id = EdgeId(source=current, target=neighbor, key=key)
                    cost = self.cost_provider.get_cost(edge_id)
                    
                    if cost is None:
                        continue
                        
                    if cost < best_edge_cost:
                        best_edge_cost = cost
                        best_edge_key = key
                        
                # No valid edge found? Skip this neighbor
                if best_edge_key is None or best_edge_cost == float('inf'):
                    continue
                    
                tentative_g_score = g_score[current] + best_edge_cost
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = (current, best_edge_key)
                    g_score[neighbor] = tentative_g_score
                    
                    h_score = self.heuristic_fn(neighbor, destination, self.graph)
                    f_score = tentative_g_score + h_score
                    
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    
        # 5. Check if we reached the destination
        if destination not in came_from:
            raise RouteNotFoundError(f"No directed path found from {source} to {destination}.")
            
        # 6. Reconstruct Path
        path_nodes, path_edges = reconstruct_path(came_from, destination)
        geometry, total_distance = extract_geometry_and_distance(self.graph, path_edges)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return RouteResult(
            success=True,
            source_node=source,
            destination_node=destination,
            path_nodes=path_nodes,
            path_edges=path_edges,
            total_cost=g_score[destination],
            distance_m=total_distance,
            expanded_nodes=expansions,
            execution_time_ms=execution_time_ms,
            geometry=geometry
        )
