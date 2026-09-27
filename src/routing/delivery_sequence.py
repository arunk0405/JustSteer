"""Delivery sequence domain model and contract.

This module defines the domain-level representations for delivery sequence optimization.
It establishes the input and output contracts without implementing the optimization logic itself.
"""

from dataclasses import dataclass
from typing import Any, Tuple

from src.routing.route_result import RouteResult


@dataclass(frozen=True)
class DeliverySequenceRequest:
    """Represents a request to optimize a delivery sequence.
    
    Attributes:
        start_node: The graph node ID where the sequence begins (the partner's location).
        delivery_nodes: A collection of distinct graph node IDs representing deliveries.
    """
    start_node: Any
    delivery_nodes: Tuple[Any, ...]

    def __post_init__(self):
        """Validate domain invariants for the request."""
        # Ensure no duplicate deliveries as sequence optimization treats them uniquely
        if len(set(self.delivery_nodes)) != len(self.delivery_nodes):
            raise ValueError("Duplicate delivery nodes are not permitted.")


@dataclass(frozen=True)
class DeliverySequence:
    """Represents an ordered sequence of delivery nodes.
    
    This object defines the explicit order in which deliveries should be visited.
    It does not contain the start node, only the deliveries themselves.
    
    Attributes:
        sequence: The ordered tuple of delivery node IDs.
    """
    sequence: Tuple[Any, ...]


@dataclass(frozen=True)
class DeliverySequenceResult:
    """Represents the output contract of a sequence optimization.
    
    Attributes:
        delivery_sequence: The explicit order of deliveries chosen by the optimizer.
        total_cost: The combined cost of the sequence, used to rank candidate sequences.
        routes: The ordered collection of RouteResults representing the actual paths
                between the consecutive sequence stops (e.g. start -> A, A -> B).
    """
    delivery_sequence: DeliverySequence
    total_cost: float
    routes: Tuple[RouteResult, ...]
    evaluated_permutations: int = 0
