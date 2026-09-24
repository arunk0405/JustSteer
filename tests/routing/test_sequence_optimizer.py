import pytest
from typing import Any, Tuple

from src.routing.delivery_sequence import DeliverySequenceRequest
from src.routing.route_result import RouteResult
from src.routing.routing_service import PairwiseRoutingService
from src.routing.pairwise_route_matrix import PairwiseRouteMatrixBuilder
from src.routing.sequence_optimizer import BruteForceSequenceOptimizer
from src.routing.exceptions import RouteNotFoundError

class FakePairwiseRoutingService:
    def __init__(self, costs: dict):
        self.costs = costs
        
    def route(self, source: Any, destination: Any) -> RouteResult:
        if (source, destination) not in self.costs:
            raise RouteNotFoundError(f"No route {source}->{destination}")
        cost = self.costs[(source, destination)]
        return RouteResult(
            success=True,
            source_node=source,
            destination_node=destination,
            total_cost=cost
        )


def test_optimize_empty():
    service = FakePairwiseRoutingService({})
    builder = PairwiseRouteMatrixBuilder(service)
    optimizer = BruteForceSequenceOptimizer(builder)
    
    req = DeliverySequenceRequest("A", ())
    res = optimizer.optimize(req)
    
    assert res.total_cost == 0.0
    assert len(res.delivery_sequence.sequence) == 0


def test_optimize_single():
    service = FakePairwiseRoutingService({("A", "B"): 5.0})
    builder = PairwiseRouteMatrixBuilder(service)
    optimizer = BruteForceSequenceOptimizer(builder)
    
    req = DeliverySequenceRequest("A", ("B",))
    res = optimizer.optimize(req)
    
    assert res.total_cost == 5.0
    assert res.delivery_sequence.sequence == ("B",)


def test_optimize_multiple():
    # A->B=1, B->C=2 (cost 3)
    # A->C=4, C->B=5 (cost 9)
    service = FakePairwiseRoutingService({
        ("A", "B"): 1.0,
        ("A", "C"): 4.0,
        ("B", "C"): 2.0,
        ("C", "B"): 5.0
    })
    builder = PairwiseRouteMatrixBuilder(service)
    optimizer = BruteForceSequenceOptimizer(builder)
    
    req = DeliverySequenceRequest("A", ("B", "C"))
    res = optimizer.optimize(req)
    
    assert res.total_cost == 3.0
    assert res.delivery_sequence.sequence == ("B", "C")


def test_optimize_with_start_in_deliveries():
    # Start=A, deliveries=[A, B]
    service = FakePairwiseRoutingService({
        ("A", "A"): 0.0,
        ("A", "B"): 3.0,
    })
    builder = PairwiseRouteMatrixBuilder(service)
    optimizer = BruteForceSequenceOptimizer(builder)
    
    req = DeliverySequenceRequest("A", ("A", "B"))
    res = optimizer.optimize(req)
    
    assert res.total_cost == 3.0
    assert res.delivery_sequence.sequence == ("A", "B")


def test_optimize_unreachable():
    service = FakePairwiseRoutingService({
        ("A", "B"): 1.0,
        # missing B->C, C->B
    })
    builder = PairwiseRouteMatrixBuilder(service)
    optimizer = BruteForceSequenceOptimizer(builder)
    
    req = DeliverySequenceRequest("A", ("B", "C"))
    
    with pytest.raises(RouteNotFoundError):
        optimizer.optimize(req)
