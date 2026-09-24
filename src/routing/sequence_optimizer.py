"""Sequence optimization for multi-stop delivery routes.

This module provides the optimization logic to determine the most efficient
order to visit a set of delivery nodes.
"""

import itertools
from typing import Any, List, Optional, Tuple

from src.routing.delivery_sequence import (
    DeliverySequence,
    DeliverySequenceRequest,
    DeliverySequenceResult
)
from src.routing.pairwise_route_matrix import PairwiseRouteMatrixBuilder
from src.routing.exceptions import RouteNotFoundError


class BruteForceSequenceOptimizer:
    """Optimizes a delivery sequence by evaluating all possible permutations.
    
    Given the typical small number of active deliveries (e.g. 4-5), an exhaustive
    evaluation provides the guaranteed optimal sequence without complex heuristics.
    """
    
    def __init__(self, matrix_builder: PairwiseRouteMatrixBuilder):
        """Initialize with a pairwise route matrix builder.
        
        Args:
            matrix_builder: The builder used to pre-calculate all necessary pairwise routes.
        """
        self.matrix_builder = matrix_builder
        
    def optimize(self, request: DeliverySequenceRequest) -> DeliverySequenceResult:
        """Find the lowest-cost valid delivery sequence.
        
        Args:
            request: The sequence request containing start node and delivery nodes.
            
        Returns:
            A DeliverySequenceResult containing the optimal sequence, cost, and legs.
            
        Raises:
            RouteNotFoundError: If no valid sequence can reach all delivery nodes.
        """
        if not request.delivery_nodes:
            return DeliverySequenceResult(
                delivery_sequence=DeliverySequence(sequence=()),
                total_cost=0.0,
                routes=()
            )
            
        # 1. Build the matrix of all required pairwise routes
        matrix = self.matrix_builder.build(request)
        
        # Determine the deliveries that actually need sequencing (excluding start if it's there)
        deliveries_to_sequence = tuple(
            node for node in request.delivery_nodes if node != request.start_node
        )
        
        best_cost = float('inf')
        best_sequence: Optional[Tuple[Any, ...]] = None
        best_routes: Optional[Tuple[Any, ...]] = None
        
        # 2. Evaluate all permutations of the deliveries
        for perm in itertools.permutations(deliveries_to_sequence):
            # If the start node was explicitly in the deliveries, it must be the first stop.
            # But wait, domain rule: start node delivery is zero-cost if handled gracefully.
            # The prompt said: "Start = 100, Deliveries = [100, 200]. Do not silently remove delivery 100."
            # So if start node is in deliveries, it should logically just be first.
            if request.start_node in request.delivery_nodes:
                full_sequence = (request.start_node,) + perm
            else:
                full_sequence = perm
                
            current_node = request.start_node
            current_cost = 0.0
            current_routes = []
            is_valid = True
            
            for next_node in full_sequence:
                if current_node == next_node:
                    # Self-pairs are not in the matrix, so generate it via the service
                    route = self.matrix_builder.routing_service.route(current_node, next_node)
                else:
                    route = matrix.get_route(current_node, next_node)
                    
                if route is None:
                    is_valid = False
                    break
                    
                current_routes.append(route)
                current_cost += route.total_cost
                current_node = next_node
                
                # Fast fail if already worse than best known
                if current_cost >= best_cost:
                    is_valid = False
                    break
                    
            if is_valid and current_cost < best_cost:
                best_cost = current_cost
                best_sequence = full_sequence
                best_routes = tuple(current_routes)
                
        if best_sequence is None:
            raise RouteNotFoundError("No valid route sequence connects all delivery nodes.")
            
        return DeliverySequenceResult(
            delivery_sequence=DeliverySequence(sequence=best_sequence),
            total_cost=best_cost,
            routes=best_routes
        )
