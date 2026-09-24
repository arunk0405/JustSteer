"""Multi-stop route execution.

This module provides the execution orchestration layer that transforms a
selected delivery sequence into a complete, routed multi-stop journey.
"""

from dataclasses import dataclass
from typing import Any, Tuple

from src.routing.delivery_sequence import DeliverySequence
from src.routing.route_result import RouteResult
from src.routing.routing_service import PairwiseRoutingService


@dataclass(frozen=True)
class MultiStopRouteResult:
    """Represents the complete result of a multi-stop delivery sequence execution.
    
    Attributes:
        success: True if the entire sequence was successfully routed.
        delivery_sequence: The underlying requested delivery sequence.
        routes: An ordered tuple of the individual pairwise RouteResults for each leg.
        total_cost: The sum of the total_cost of all routed legs.
        total_distance_m: The sum of the physical distance (in meters) of all routed legs.
    """
    success: bool
    delivery_sequence: DeliverySequence
    routes: Tuple[RouteResult, ...]
    total_cost: float
    total_distance_m: float


class MultiStopRouteExecutor:
    """Orchestrates the routing of a multi-stop delivery sequence.
    
    This executor uses an existing PairwiseRoutingService to generate the 
    consecutive routing legs required to satisfy the delivery sequence.
    """
    
    def __init__(self, pairwise_service: PairwiseRoutingService):
        """Initialize the executor with a pairwise routing dependency.
        
        Args:
            pairwise_service: The service to delegate individual routing legs to.
        """
        self.pairwise_service = pairwise_service
        
    def execute(self, start_node: Any, sequence: DeliverySequence) -> MultiStopRouteResult:
        """Execute the delivery sequence by routing between consecutive stops.
        
        Args:
            start_node: The starting node ID.
            sequence: The explicit order of delivery nodes to visit.
            
        Returns:
            MultiStopRouteResult containing the aggregated route information.
            
        Raises:
            RouteNotFoundError: If any pairwise leg cannot be routed.
            InvalidNodeError: If any node is missing from the graph.
        """
        routes = []
        total_cost = 0.0
        total_distance_m = 0.0
        
        current_node = start_node
        
        for next_node in sequence.sequence:
            # Delegate entirely to the pairwise service.
            # Even if current_node == next_node, the service handles it.
            leg_result = self.pairwise_service.route(current_node, next_node)
            
            routes.append(leg_result)
            total_cost += leg_result.total_cost
            
            # Aggregate distance, respecting RouteResult semantics
            if leg_result.distance_m is not None:
                total_distance_m += leg_result.distance_m
                
            current_node = next_node
            
        return MultiStopRouteResult(
            success=True,
            delivery_sequence=sequence,
            routes=tuple(routes),
            total_cost=total_cost,
            total_distance_m=total_distance_m
        )
