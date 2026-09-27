"""Application-level orchestration for multi-stop routing.

This module provides the final integration layer for Step 8, uniting the
domain contract, sequence optimization, and route execution into a single,
clean public interface.
"""

from typing import Any, Tuple

from src.routing.delivery_sequence import DeliverySequenceRequest
from src.routing.multi_stop_routing import MultiStopRouteExecutor, MultiStopRouteResult
from src.routing.sequence_optimizer import BruteForceSequenceOptimizer


class MultiStopRouter:
    """Application-facing entry point for complete multi-stop delivery routing.
    
    This orchestrator cleanly separates sequence optimization from physical route
    execution while providing a single high-level API to consumers.
    """
    
    def __init__(
        self,
        sequence_optimizer: BruteForceSequenceOptimizer,
        route_executor: MultiStopRouteExecutor
    ):
        """Initialize the router with its required dependencies.
        
        Args:
            sequence_optimizer: Component responsible for determining the visitation order.
            route_executor: Component responsible for executing the selected order.
        """
        self.sequence_optimizer = sequence_optimizer
        self.route_executor = route_executor
        
    def route(self, start_node: Any, delivery_nodes: Tuple[Any, ...]) -> MultiStopRouteResult:
        """Plan and execute a complete multi-stop route.
        
        Args:
            start_node: The current location node ID.
            delivery_nodes: A collection of distinct delivery node IDs.
            
        Returns:
            A MultiStopRouteResult containing the selected sequence and detailed route legs.
            
        Raises:
            RouteNotFoundError: If the deliveries cannot be logically sequenced or 
                                if a physical route leg is unreachable.
            ValueError: If domain invariants are violated (e.g. duplicate deliveries).
        """
        # 1. Establish the domain contract
        request = DeliverySequenceRequest(
            start_node=start_node,
            delivery_nodes=delivery_nodes
        )
        
        # 2. Sequence Optimization (Step 8.2)
        # We retrieve the optimal order, delegating algorithmic complexity to the optimizer.
        optimization_result = self.sequence_optimizer.optimize(request)
        
        # 3. Route Execution (Step 8.3)
        # We strictly respect the chosen sequence and execute it as the authoritative plan.
        execution_result = self.route_executor.execute(
            start_node=request.start_node,
            sequence=optimization_result.delivery_sequence
        )
        
        # 4. Return Final Integrated Result (Step 8.4)
        return MultiStopRouteResult(
            success=execution_result.success,
            delivery_sequence=execution_result.delivery_sequence,
            routes=execution_result.routes,
            total_cost=execution_result.total_cost,
            total_distance_m=execution_result.total_distance_m,
            total_duration_s=execution_result.total_duration_s,
            expanded_nodes=execution_result.expanded_nodes,
            evaluated_permutations=optimization_result.evaluated_permutations
        )
