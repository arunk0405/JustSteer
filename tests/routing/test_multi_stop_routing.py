import pytest
import networkx as nx
from typing import Any, List, Tuple

from src.routing.delivery_sequence import DeliverySequence
from src.routing.route_result import RouteResult
from src.routing.multi_stop_routing import MultiStopRouteExecutor, MultiStopRouteResult
from src.routing.exceptions import RouteNotFoundError
from src.routing.astar import AStarRoutingEngine
from src.routing.routing_service import PairwiseRoutingService


class FakePairwiseRoutingService:
    def __init__(self):
        self.calls: List[Tuple[Any, Any]] = []
        self.unreachable_pairs = set()
        self.mock_routes = {}
        
    def route(self, source: Any, destination: Any) -> RouteResult:
        self.calls.append((source, destination))
        
        if (source, destination) in self.unreachable_pairs:
            raise RouteNotFoundError(f"No route {source} -> {destination}")
            
        if (source, destination) in self.mock_routes:
            return self.mock_routes[(source, destination)]
            
        return RouteResult(
            success=True,
            source_node=source,
            destination_node=destination,
            total_cost=0.0,
            distance_m=0.0,
            path_nodes=[source, destination]
        )


@pytest.fixture
def fake_service():
    return FakePairwiseRoutingService()


@pytest.fixture
def executor(fake_service):
    return MultiStopRouteExecutor(fake_service)


def test_basic_multi_stop_execution(executor, fake_service):
    # Test 1, Test 2 - Basic Execution & Correct Ordering
    seq = DeliverySequence(sequence=("B", "C", "D"))
    result = executor.execute("A", seq)
    
    assert result.success is True
    assert len(fake_service.calls) == 3
    assert fake_service.calls == [("A", "B"), ("B", "C"), ("C", "D")]
    
    assert len(result.routes) == 3
    assert result.routes[0].source_node == "A"
    assert result.routes[0].destination_node == "B"
    assert result.routes[1].source_node == "B"
    assert result.routes[1].destination_node == "C"
    assert result.routes[2].source_node == "C"
    assert result.routes[2].destination_node == "D"


def test_aggregate_cost(executor, fake_service):
    # Test 3 - Aggregate Cost
    r1 = RouteResult(success=True, source_node="A", destination_node="B", total_cost=0.2)
    r2 = RouteResult(success=True, source_node="B", destination_node="C", total_cost=0.4)
    r3 = RouteResult(success=True, source_node="C", destination_node="D", total_cost=0.3)
    
    fake_service.mock_routes = {
        ("A", "B"): r1,
        ("B", "C"): r2,
        ("C", "D"): r3
    }
    
    seq = DeliverySequence(sequence=("B", "C", "D"))
    result = executor.execute("A", seq)
    
    # 0.2 + 0.4 + 0.3 = 0.9
    assert pytest.approx(result.total_cost) == 0.9


def test_aggregate_distance(executor, fake_service):
    # Test 4 - Aggregate Distance
    r1 = RouteResult(success=True, source_node="A", destination_node="B", distance_m=100)
    r2 = RouteResult(success=True, source_node="B", destination_node="C", distance_m=200)
    r3 = RouteResult(success=True, source_node="C", destination_node="D", distance_m=150)
    
    fake_service.mock_routes = {
        ("A", "B"): r1,
        ("B", "C"): r2,
        ("C", "D"): r3
    }
    
    seq = DeliverySequence(sequence=("B", "C", "D"))
    result = executor.execute("A", seq)
    
    # 100 + 200 + 150 = 450
    assert pytest.approx(result.total_distance_m) == 450


def test_empty_sequence(executor, fake_service):
    # Test 5 - Empty Sequence
    seq = DeliverySequence(sequence=())
    result = executor.execute("A", seq)
    
    assert len(fake_service.calls) == 0
    assert result.success is True
    assert len(result.routes) == 0
    assert result.total_cost == 0.0
    assert result.total_distance_m == 0.0


def test_single_delivery(executor, fake_service):
    # Test 6 - Single Delivery
    seq = DeliverySequence(sequence=("B",))
    result = executor.execute("A", seq)
    
    assert len(fake_service.calls) == 1
    assert fake_service.calls[0] == ("A", "B")
    assert len(result.routes) == 1


def test_delivery_at_current_location(executor, fake_service):
    # Test 7 - Delivery at Current Location
    seq = DeliverySequence(sequence=("A", "B"))
    result = executor.execute("A", seq)
    
    assert len(fake_service.calls) == 2
    assert fake_service.calls[0] == ("A", "A")
    assert fake_service.calls[1] == ("A", "B")


def test_routing_failure(executor, fake_service):
    # Test 8, Test 9 - Routing Failure & No Silent Skipping
    fake_service.unreachable_pairs.add(("B", "C"))
    
    seq = DeliverySequence(sequence=("B", "C", "D"))
    
    with pytest.raises(RouteNotFoundError, match="No route B -> C"):
        executor.execute("A", seq)


def test_sequence_immutability(executor, fake_service):
    # Test 10 - Sequence Immutability
    tup = ("B", "C")
    seq = DeliverySequence(sequence=tup)
    
    executor.execute("A", seq)
    
    assert seq.sequence is tup
    assert seq.sequence == ("B", "C")


def test_routeresult_preservation(executor, fake_service):
    # Test 11 - RouteResult Preservation
    r1 = RouteResult(success=True, source_node="A", destination_node="B")
    fake_service.mock_routes = {("A", "B"): r1}
    
    seq = DeliverySequence(sequence=("B",))
    result = executor.execute("A", seq)
    
    assert result.routes[0] is r1


def test_dependency_injection(executor):
    # Test 12 - Dependency Injection
    # Verified by the fact we just executed all above tests entirely using FakePairwiseRoutingService.
    assert isinstance(executor.pairwise_service, FakePairwiseRoutingService)


def test_synthetic_end_to_end_routing():
    # 34. Integration Test
    # Create a synthetic graph A -> B -> C -> D
    G = nx.MultiDiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=0.0, y=1.0)
    G.add_node("C", x=0.0, y=2.0)
    G.add_node("D", x=0.0, y=3.0)
    
    G.add_edge("A", "B", final_cost=0.2, length=100.0)
    G.add_edge("B", "C", final_cost=0.3, length=200.0)
    G.add_edge("C", "D", final_cost=0.4, length=150.0)
    
    astar = AStarRoutingEngine(graph=G, cost_attribute="final_cost")
    service = PairwiseRoutingService(astar_engine=astar)
    executor = MultiStopRouteExecutor(pairwise_service=service)
    
    seq = DeliverySequence(sequence=("B", "C", "D"))
    result = executor.execute("A", seq)
    
    assert result.success is True
    assert len(result.routes) == 3
    
    assert result.routes[0].source_node == "A"
    assert result.routes[0].destination_node == "B"
    assert pytest.approx(result.routes[0].total_cost) == 0.2
    
    assert result.routes[1].source_node == "B"
    assert result.routes[1].destination_node == "C"
    assert pytest.approx(result.routes[1].total_cost) == 0.3
    
    assert result.routes[2].source_node == "C"
    assert result.routes[2].destination_node == "D"
    assert pytest.approx(result.routes[2].total_cost) == 0.4
    
    assert pytest.approx(result.total_cost) == 0.9
    assert pytest.approx(result.total_distance_m) == 450.0
