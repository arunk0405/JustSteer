import math
import networkx as nx
from typing import Union
from src.routing.edge_cost import EdgeId

class InvalidPhysicalQuantityError(ValueError):
    """Raised when a physical measurement (length or speed) is invalid for routing calculations."""
    pass

class EdgeLengthAccessor:
    """
    Narrowly scoped read-only component to retrieve physical edge length from the OSM graph.
    It separates graph access from dynamic travel time calculation.
    """
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        
    def get_length_m(self, edge_id: EdgeId) -> float:
        """
        Retrieves the physical length of the edge in meters.
        
        Args:
            edge_id: Canonical EdgeId preserving source, target, and key.
            
        Returns:
            The physical edge length in meters (as float).
            
        Raises:
            KeyError: If the edge does not exist in the graph.
            InvalidPhysicalQuantityError: If the length attribute is missing, non-numeric, negative, infinite, or NaN.
        """
        if edge_id.source not in self.graph:
            raise KeyError(f"Source node {edge_id.source} not in graph.")
            
        if edge_id.target not in self.graph[edge_id.source]:
            raise KeyError(f"Target node {edge_id.target} not reachable from {edge_id.source}.")
            
        if edge_id.key not in self.graph[edge_id.source][edge_id.target]:
            raise KeyError(f"Edge key {edge_id.key} not found between {edge_id.source} and {edge_id.target}.")
            
        edge_data = self.graph[edge_id.source][edge_id.target][edge_id.key]
        
        if "length" not in edge_data:
            raise InvalidPhysicalQuantityError(f"Edge {edge_id} is missing 'length' attribute.")
            
        length = edge_data["length"]
        
        if length is None or not isinstance(length, (int, float)):
            raise InvalidPhysicalQuantityError(f"Edge {edge_id} length must be numeric, got: {length} ({type(length)})")
            
        if not math.isfinite(length) or length < 0:
            raise InvalidPhysicalQuantityError(f"Edge {edge_id} length must be finite and non-negative, got: {length}")
            
        return float(length)


class TravelTimeCalculator:
    """
    Pure calculation component that determines predicted travel time.
    Separates the static physical graph geometry from dynamic ML predictions.
    """
    @staticmethod
    def calculate(length_m: Union[int, float], speed_kmh: Union[int, float]) -> float:
        """
        Converts physical edge length and predicted traffic speed into travel time in minutes.
        
        Formula:
            speed_m_per_minute = speed_kmh * 1000 / 60
            travel_time_minutes = length_m / speed_m_per_minute
        
        Args:
            length_m: Edge length in meters (from OSM graph)
            speed_kmh: Predicted traffic speed in km/h (from TrafficPredictionProvider)
            
        Returns:
            Travel time in minutes (compatible with WeightCalculator)
            
        Raises:
            InvalidPhysicalQuantityError: If inputs are missing, negative, infinite, NaN, non-numeric, or speed is zero.
        """
        # 1. Validate length
        if length_m is None or not isinstance(length_m, (int, float)) or isinstance(length_m, bool):
            raise InvalidPhysicalQuantityError(f"Length must be a number, got: {length_m}")
            
        if not math.isfinite(length_m) or length_m < 0:
            raise InvalidPhysicalQuantityError(f"Length must be a finite non-negative number, got: {length_m}")
            
        # 2. Validate speed
        if speed_kmh is None or not isinstance(speed_kmh, (int, float)) or isinstance(speed_kmh, bool):
            raise InvalidPhysicalQuantityError(f"Speed must be a number, got: {speed_kmh}")
            
        if not math.isfinite(speed_kmh) or speed_kmh < 0:
            raise InvalidPhysicalQuantityError(f"Speed must be a finite non-negative number, got: {speed_kmh}")
            
        if math.isclose(speed_kmh, 0.0, abs_tol=1e-9):
            raise InvalidPhysicalQuantityError("Predicted speed is zero. Cannot calculate travel time.")
            
        # 3. Handle zero length cleanly (prevents doing unnecessary math)
        if math.isclose(length_m, 0.0, abs_tol=1e-9):
            return 0.0
            
        # 4. Convert speed to meters per minute
        speed_m_per_minute = float(speed_kmh) * 1000.0 / 60.0
        
        # 5. Calculate travel time in minutes
        travel_time_minutes = float(length_m) / speed_m_per_minute
        
        return travel_time_minutes
