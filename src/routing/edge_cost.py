from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime
from src.routing.contracts import CombinedEdgeCost

@dataclass(frozen=True)
class EdgeId:
    """
    Canonical identity for an edge in an OSM MultiDiGraph.
    Because MultiDiGraph supports multiple edges between nodes,
    a (source, target) tuple is insufficient.
    """
    source: int
    target: int
    key: Any

    def __post_init__(self):
        # Enforce that source and target are integers (and not bools)
        if not isinstance(self.source, int) or isinstance(self.source, bool):
            raise TypeError(f"Source node must be an integer, got {type(self.source).__name__}")
        if not isinstance(self.target, int) or isinstance(self.target, bool):
            raise TypeError(f"Target node must be an integer, got {type(self.target).__name__}")
        
        # Valid edge keys in OSMnx/NetworkX can be ints or strings.
        if not isinstance(self.key, (int, str)) or isinstance(self.key, bool):
            raise TypeError(f"Edge key must be an integer or string, got {type(self.key).__name__}")


@dataclass(frozen=True)
class EdgeCost:
    """
    Represents the dynamic cost state of one edge in the routing graph.
    This separates the static OSM topology from request-specific/time-varying dynamic predictions.
    """
    edge_id: EdgeId
    cost_data: CombinedEdgeCost
    generated_at: datetime
    traffic_model_version: Optional[str] = None
    safety_model_version: Optional[str] = None
