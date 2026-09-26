from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional

@dataclass
class RouteResult:
    """
    Standardized result structure for a routing query.
    """
    success: bool
    source_node: Any
    destination_node: Any
    path_nodes: List[Any] = field(default_factory=list)
    path_edges: List[Tuple[Any, Any, Any]] = field(default_factory=list)  # (u, v, key)
    total_cost: float = 0.0
    distance_m: float = 0.0
    duration_s: float = 0.0
    expanded_nodes: int = 0
    execution_time_ms: float = 0.0
    geometry: Optional[List[Tuple[float, float]]] = None  # [(lon, lat), ...]
    
    @property
    def is_valid(self) -> bool:
        return self.success and len(self.path_nodes) > 0
