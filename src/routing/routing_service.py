"""Pairwise routing service.

This module provides a clean orchestration layer over the underlying routing algorithm.
It exposes a simple node-to-node routing abstraction for future consumers (e.g., Sequence Optimizer)
so they don't have to know about A* internals or graph initialization.
"""

from typing import Any
from src.routing.astar import AStarRoutingEngine
from src.routing.route_result import RouteResult

class PairwiseRoutingService:
    """A service that provides pairwise node-to-node routing on a road graph.

    This service delegates the actual route calculation to the existing A* routing engine.
    It does not own the graph, calculate ML costs, or perform multi-stop optimization.
    """

    def __init__(self, astar_engine: AStarRoutingEngine):
        """Initialize the routing service with an existing routing engine.

        Args:
            astar_engine: An initialized AStarRoutingEngine configured with a graph.
        """
        self.astar_engine = astar_engine

    def route(self, origin_node: Any, destination_node: Any) -> RouteResult:
        """Find the optimal route between two OSM nodes.

        Args:
            origin_node: The starting node ID.
            destination_node: The destination node ID.

        Returns:
            RouteResult containing the selected path, edges, cost, and geometry.
            
        Raises:
            InvalidNodeError: If origin_node or destination_node is not in the graph.
            RouteNotFoundError: If no path exists between the nodes.
            InvalidEdgeCostError: If a route expansion encounters invalid costs.
        """
        return self.astar_engine.route(origin_node, destination_node)
