from typing import Optional, Dict
from src.routing.edge_cost import EdgeId, EdgeCost

class EdgeCostStore:
    """
    In-memory domain store for dynamic edge costs.
    Maps a canonical EdgeId to its most recent EdgeCost state.
    Does not depend on NetworkX, the graph topology, or coordinates.
    """
    def __init__(self):
        self._store: Dict[EdgeId, EdgeCost] = {}

    def set(self, edge_id: EdgeId, cost: EdgeCost) -> None:
        """
        Stores or replaces the cost for a given edge.
        Follows update semantics (latest cost overwrites).
        """
        if not isinstance(edge_id, EdgeId):
            raise TypeError("edge_id must be an instance of EdgeId")
        if not isinstance(cost, EdgeCost):
            raise TypeError("cost must be an instance of EdgeCost")
        self._store[edge_id] = cost

    def get(self, edge_id: EdgeId) -> Optional[EdgeCost]:
        """
        Retrieves the dynamic cost for the given edge.
        Returns None if no cost is currently stored (it may not have been predicted yet).
        """
        if not isinstance(edge_id, EdgeId):
            raise TypeError("edge_id must be an instance of EdgeId")
        return self._store.get(edge_id)

    def contains(self, edge_id: EdgeId) -> bool:
        """Checks if a cost is currently stored for the given edge."""
        if not isinstance(edge_id, EdgeId):
            raise TypeError("edge_id must be an instance of EdgeId")
        return edge_id in self._store

    def remove(self, edge_id: EdgeId) -> None:
        """Removes the cost for the given edge if it exists."""
        if not isinstance(edge_id, EdgeId):
            raise TypeError("edge_id must be an instance of EdgeId")
        self._store.pop(edge_id, None)

    def clear(self) -> None:
        """Clears all dynamic costs from the store."""
        self._store.clear()
