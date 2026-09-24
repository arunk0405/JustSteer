"""Pairwise route matrix for delivery sequence optimization.

This module provides request-local caching and organization of the directed pairwise 
RouteResults required by the sequence optimization layer.
"""

from typing import Any, Dict, Optional, Tuple

from src.routing.delivery_sequence import DeliverySequenceRequest
from src.routing.route_result import RouteResult
from src.routing.routing_service import PairwiseRoutingService
from src.routing.exceptions import RouteNotFoundError


class PairwiseRouteMatrix:
    """A request-local collection of directed pairwise routes.
    
    This matrix organizes the RouteResults calculated for a specific delivery sequence request.
    It provides deterministic lookups and explicitly distinguishes between valid routes
    and unreachable paths.
    """
    def __init__(self, routes: Dict[Tuple[Any, Any], Optional[RouteResult]]):
        """Initialize with an already computed dictionary of routes.
        
        Args:
            routes: A dictionary mapping (source, destination) node pairs to 
                    their RouteResult. None signifies an explicitly unreachable pair.
        """
        # We store it privately so consumers don't mutate it
        self._routes = dict(routes)
        
    def get_route(self, source: Any, destination: Any) -> Optional[RouteResult]:
        """Retrieve the pairwise route between two nodes.
        
        Args:
            source: The origin node ID.
            destination: The destination node ID.
            
        Returns:
            The original RouteResult if the route is valid, or None if the route
            was evaluated but determined to be unreachable.
            
        Raises:
            KeyError: If the (source, destination) pair was never requested/evaluated
                      (e.g. a self-pair or return-to-start pair).
        """
        pair = (source, destination)
        if pair not in self._routes:
            raise KeyError(f"Directed pair {pair} was not evaluated in this matrix.")
            
        return self._routes[pair]


class PairwiseRouteMatrixBuilder:
    """Builder responsible for requesting required pairwise routes.
    
    This builder orchestrates calls to the PairwiseRoutingService. It generates
    the minimal necessary set of directed routes for a DeliverySequenceRequest.
    """
    def __init__(self, routing_service: PairwiseRoutingService):
        """Initialize the matrix builder.
        
        Args:
            routing_service: The service responsible for delegating node-to-node routing.
        """
        self.routing_service = routing_service
        
    def build(self, request: DeliverySequenceRequest) -> PairwiseRouteMatrix:
        """Construct the pairwise route matrix for a specific sequence request.
        
        Args:
            request: The validated delivery sequence request.
            
        Returns:
            A populated PairwiseRouteMatrix containing required pairwise RouteResults.
        """
        routes: Dict[Tuple[Any, Any], Optional[RouteResult]] = {}
        
        # Determine the required pairs deterministically
        # 1. Start node to each delivery
        for delivery in request.delivery_nodes:
            if request.start_node == delivery:
                continue
                
            self._evaluate_pair(request.start_node, delivery, routes)
            
        # 2. Each delivery to every other delivery
        for source in request.delivery_nodes:
            for destination in request.delivery_nodes:
                if source == destination:
                    continue
                    
                self._evaluate_pair(source, destination, routes)
                
        return PairwiseRouteMatrix(routes)
        
    def _evaluate_pair(
        self, 
        source: Any, 
        destination: Any, 
        routes: Dict[Tuple[Any, Any], Optional[RouteResult]]
    ) -> None:
        """Evaluate a single directed pair and store the result."""
        pair = (source, destination)
        
        if pair in routes:
            return
            
        try:
            result = self.routing_service.route(source, destination)
            routes[pair] = result
        except RouteNotFoundError:
            routes[pair] = None
