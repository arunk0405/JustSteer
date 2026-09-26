import math
from typing import Protocol, Optional
import networkx as nx

from src.routing.edge_cost import EdgeId
from src.routing.edge_cost_store import EdgeCostStore

class EdgeCostProvider(Protocol):
    """
    Protocol for providing traversal costs to A*.
    """
    def get_cost(self, edge_id: EdgeId) -> Optional[float]:
        ...

class StaticGraphCostProvider:
    """
    Legacy provider that reads static costs directly from graph attributes.
    """
    def __init__(self, graph: nx.MultiDiGraph, cost_attribute: str = "final_cost"):
        self.graph = graph
        self.cost_attribute = cost_attribute

    def get_cost(self, edge_id: EdgeId) -> Optional[float]:
        if not self.graph.has_edge(edge_id.source, edge_id.target, key=edge_id.key):
            return None
            
        edge_data = self.graph[edge_id.source][edge_id.target][edge_id.key]
        raw_cost = edge_data.get(self.cost_attribute)
        
        if raw_cost is None:
            return None
            
        try:
            cost = float(raw_cost)
        except (ValueError, TypeError):
            return None
            
        if math.isnan(cost) or math.isinf(cost):
            return None
            
        if cost < 0:
            return None
            
        return cost

class DynamicStoreCostProvider:
    """
    Provider that reads request-scoped dynamic ML costs from EdgeCostStore.
    """
    def __init__(self, store: EdgeCostStore):
        self.store = store

    def get_cost(self, edge_id: EdgeId) -> Optional[float]:
        edge_cost = self.store.get(edge_id)
        if edge_cost is None:
            return None
            
        cost = edge_cost.cost_data.combined_cost
        
        if cost is None:
            return None
            
        try:
            cost = float(cost)
        except (ValueError, TypeError):
            return None
            
        if math.isnan(cost) or math.isinf(cost):
            return None
            
        if cost < 0 or cost > 1:
            return None
            
        return cost
