import networkx as nx
from typing import Iterable, List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass

from src.routing.edge_cost import EdgeId, EdgeCost
from src.routing.edge_cost_store import EdgeCostStore
from src.routing.feature_resolver import FeatureResolver, RoutingContext
from src.routing.prediction_orchestrator import PredictionOrchestrator

@dataclass(frozen=True)
class GenerationResult:
    """
    Immutable result summarizing the dynamic cost generation operation.
    """
    successful_edges: List[EdgeId]
    failed_edges: List[EdgeId]
    errors: Dict[EdgeId, str]
    
    @property
    def processed_count(self) -> int:
        return len(self.successful_edges) + len(self.failed_edges)


class DynamicCostGenerator:
    """
    Step 9.5 orchestration boundary.
    Coordinates candidate edge feature resolution, cost prediction, and storage.
    """
    
    def __init__(
        self,
        graph: nx.MultiDiGraph,
        feature_resolver: FeatureResolver,
        prediction_orchestrator: PredictionOrchestrator,
        edge_cost_store: EdgeCostStore
    ):
        self.graph = graph
        self.feature_resolver = feature_resolver
        self.prediction_orchestrator = prediction_orchestrator
        self.store = edge_cost_store

    def generate(
        self, 
        candidate_edges: Iterable[EdgeId], 
        context: RoutingContext,
        generation_time: Optional[datetime] = None
    ) -> GenerationResult:
        """
        Generates and stores dynamic costs for the provided candidate edges.
        
        Failure Policy (Partial Success):
        If a candidate edge fails to resolve features or predict, it is skipped
        and the error is captured in the result. No invalid cost is stored.
        Successfully predicted edges are persisted in the EdgeCostStore replacing
        any previous state for that EdgeId.
        """
        # Determine the generation timestamp for all costs in this batch
        gen_time = generation_time if generation_time is not None else datetime.now()
        
        successful_edges = []
        failed_edges = []
        errors = {}
        
        # We use a set internally to avoid redundant work if candidate_edges contains duplicates
        unique_edges = set(candidate_edges)
        
        if not unique_edges:
            return GenerationResult(
                successful_edges=[],
                failed_edges=[],
                errors={}
            )
            
        for edge_id in unique_edges:
            try:
                # 1. Feature Resolution
                resolved_features = self.feature_resolver.resolve(edge_id, context)
                
                # 2. Prediction Orchestration
                prediction_result = self.prediction_orchestrator.predict_edge(
                    edge_id=edge_id,
                    graph=self.graph,
                    traffic_features=resolved_features.traffic_features,
                    safety_features=resolved_features.safety_features
                )
                
                # 3. EdgeCost Construction
                edge_cost = EdgeCost(
                    edge_id=edge_id,
                    cost_data=prediction_result.combined_cost,
                    generated_at=gen_time,
                    traffic_model_version=None,  # Providers currently do not expose version info
                    safety_model_version=None
                )
                
                # 4. Persistence
                self.store.set(edge_id, edge_cost)
                
                successful_edges.append(edge_id)
                
            except Exception as e:
                failed_edges.append(edge_id)
                errors[edge_id] = str(e)
                
        return GenerationResult(
            successful_edges=successful_edges,
            failed_edges=failed_edges,
            errors=errors
        )
