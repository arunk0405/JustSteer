from typing import Any

def get_heuristic(mode: str = "zero"):
    """
    Returns a heuristic function compatible with the A* engine.
    
    Args:
        mode: The heuristic mode. Only 'zero' is recommended for 
              normalized/dimensionless ML composite costs.
    """
    if mode == "zero":
        return zero_heuristic
    
    # Fallback to zero if unknown
    return zero_heuristic

def zero_heuristic(u: Any, v: Any, graph: Any) -> float:
    """
    Zero heuristic turns A* into Dijkstra's shortest path.
    This is the only mathematically safe (admissible) heuristic when the 
    edge costs are a normalized/dimensionless composite of ML models [0, 1].
    Any physical distance (e.g. Haversine) would heavily overestimate the 
    remaining cost and return sub-optimal routes.
    """
    return 0.0
