"""Application entry point for dynamic routing.

This module provides the highest-level orchestrator that connects the static graph,
dynamic candidate selection, cost generation, and isolated request-scoped routing execution.
"""

from typing import Any, Tuple
import networkx as nx

from src.routing.edge_cost_store import EdgeCostStore
from src.routing.feature_resolver import FeatureResolver, RoutingContext
from src.routing.prediction_orchestrator import PredictionOrchestrator
from src.routing.candidate_selector import CandidateEdgeSelector
from src.routing.dynamic_cost_generator import DynamicCostGenerator
from src.routing.cost_provider import DynamicStoreCostProvider
from src.routing.astar import AStarRoutingEngine
from src.routing.routing_service import PairwiseRoutingService
from src.routing.pairwise_route_matrix import PairwiseRouteMatrixBuilder
from src.routing.sequence_optimizer import BruteForceSequenceOptimizer
from src.routing.multi_stop_routing import MultiStopRouteExecutor, MultiStopRouteResult
from src.routing.multi_stop_router import MultiStopRouter


class DynamicRoutingApp:
    """Orchestrates the entire Step 9.7 dynamic multi-stop routing lifecycle.
    
    This class ensures that every routing request receives a strictly isolated,
    request-scoped EdgeCostStore. Costs are generated exactly once for the candidate
    region, and the resulting request-scoped routing dependencies are safely
    discarded after the request completes.
    """
    
    def __init__(
        self,
        graph: nx.MultiDiGraph,
        feature_resolver: FeatureResolver,
        prediction_orchestrator: PredictionOrchestrator,
        candidate_selector: CandidateEdgeSelector
    ):
        """Initialize the long-lived application dependencies.
        
        Args:
            graph: The static OSM road network.
            feature_resolver: Authority for assembling ML features.
            prediction_orchestrator: Authority for executing ML predictions.
            candidate_selector: Authority for selecting geographic bounds.
        """
        self.graph = graph
        self.feature_resolver = feature_resolver
        self.prediction_orchestrator = prediction_orchestrator
        self.candidate_selector = candidate_selector
        
    def route(
        self, 
        start_node: Any, 
        delivery_nodes: Tuple[Any, ...], 
        context: RoutingContext
    ) -> MultiStopRouteResult:
        """Execute a complete dynamically-costed multi-stop routing request.
        
        Args:
            start_node: The origin node ID.
            delivery_nodes: A tuple of destination node IDs.
            context: The request-specific routing context (weather, time, safety).
            
        Returns:
            MultiStopRouteResult containing the final optimized sequence and route legs.
        """
        # 1. Candidate Selection
        candidate_edges = self.candidate_selector.select(start_node, delivery_nodes)
        
        # 2. Request-scoped State Creation
        store = EdgeCostStore()
        
        # 3. Dynamic Cost Generation
        # This generator lives only for this request
        cost_generator = DynamicCostGenerator(
            graph=self.graph,
            feature_resolver=self.feature_resolver,
            prediction_orchestrator=self.prediction_orchestrator,
            edge_cost_store=store
        )
        # Generate costs EXACTLY ONCE for the request
        cost_generator.generate(candidate_edges, context)
        
        # 4. Request-scoped Routing Dependencies
        provider = DynamicStoreCostProvider(store)
        astar = AStarRoutingEngine(self.graph, cost_provider=provider)
        
        # The PairwiseRoutingService wraps the A* engine bound to this request's store
        pairwise_service = PairwiseRoutingService(astar)
        
        # 5. Optimization & Execution Pipelines
        matrix_builder = PairwiseRouteMatrixBuilder(pairwise_service)
        optimizer = BruteForceSequenceOptimizer(matrix_builder)
        executor = MultiStopRouteExecutor(pairwise_service)
        
        # 6. Final Orchestration
        router = MultiStopRouter(sequence_optimizer=optimizer, route_executor=executor)
        
        return router.route(start_node, delivery_nodes)
