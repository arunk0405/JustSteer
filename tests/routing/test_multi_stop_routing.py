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
    seq = DeliverySequence(sequence=(2, 3, 4))
    result = executor.execute(1, seq)
    
    assert result.success is True
    assert len(fake_service.calls) == 3
    assert fake_service.calls == [(1, 2), (2, 3), (3, 4)]
    
    assert len(result.routes) == 3
    assert result.routes[0].source_node == 1
    assert result.routes[0].destination_node == 2
    assert result.routes[1].source_node == 2
    assert result.routes[1].destination_node == 3
    assert result.routes[2].source_node == 3
    assert result.routes[2].destination_node == 4


def test_aggregate_cost(executor, fake_service):
    # Test 3 - Aggregate Cost
    r1 = RouteResult(success=True, source_node=1, destination_node=2, total_cost=0.2)
    r2 = RouteResult(success=True, source_node=2, destination_node=3, total_cost=0.4)
    r3 = RouteResult(success=True, source_node=3, destination_node=4, total_cost=0.3)
    
    fake_service.mock_routes = {
        (1, 2): r1,
        (2, 3): r2,
        (3, 4): r3
    }
    
    seq = DeliverySequence(sequence=(2, 3, 4))
    result = executor.execute(1, seq)
    
    # 0.2 + 0.4 + 0.3 = 0.9
    assert pytest.approx(result.total_cost) == 0.9


def test_aggregate_distance(executor, fake_service):
    # Test 4 - Aggregate Distance
    r1 = RouteResult(success=True, source_node=1, destination_node=2, distance_m=100)
    r2 = RouteResult(success=True, source_node=2, destination_node=3, distance_m=200)
    r3 = RouteResult(success=True, source_node=3, destination_node=4, distance_m=150)
    
    fake_service.mock_routes = {
        (1, 2): r1,
        (2, 3): r2,
        (3, 4): r3
    }
    
    seq = DeliverySequence(sequence=(2, 3, 4))
    result = executor.execute(1, seq)
    
    # 100 + 200 + 150 = 450
    assert pytest.approx(result.total_distance_m) == 450


def test_empty_sequence(executor, fake_service):
    # Test 5 - Empty Sequence
    seq = DeliverySequence(sequence=())
    result = executor.execute(1, seq)
    
    assert len(fake_service.calls) == 0
    assert result.success is True
    assert len(result.routes) == 0
    assert result.total_cost == 0.0
    assert result.total_distance_m == 0.0


def test_single_delivery(executor, fake_service):
    # Test 6 - Single Delivery
    seq = DeliverySequence(sequence=(2,))
    result = executor.execute(1, seq)
    
    assert len(fake_service.calls) == 1
    assert fake_service.calls[0] == (1, 2)
    assert len(result.routes) == 1


def test_delivery_at_current_location(executor, fake_service):
    # Test 7 - Delivery at Current Location
    seq = DeliverySequence(sequence=(1, 2))
    result = executor.execute(1, seq)
    
    assert len(fake_service.calls) == 2
    assert fake_service.calls[0] == (1, 1)
    assert fake_service.calls[1] == (1, 2)


def test_routing_failure(executor, fake_service):
    # Test 8, Test 9 - Routing Failure & No Silent Skipping
    fake_service.unreachable_pairs.add((2, 3))
    
    seq = DeliverySequence(sequence=(2, 3, 4))
    
    with pytest.raises(RouteNotFoundError, match="No route 2 -> 3"):
        executor.execute(1, seq)


def test_sequence_immutability(executor, fake_service):
    # Test 10 - Sequence Immutability
    tup = (2, 3)
    seq = DeliverySequence(sequence=tup)
    
    executor.execute(1, seq)
    
    assert seq.sequence is tup
    assert seq.sequence == (2, 3)


def test_routeresult_preservation(executor, fake_service):
    # Test 11 - RouteResult Preservation
    r1 = RouteResult(success=True, source_node=1, destination_node=2)
    fake_service.mock_routes = {(1, 2): r1}
    
    seq = DeliverySequence(sequence=(2,))
    result = executor.execute(1, seq)
    
    assert result.routes[0] is r1


def test_dependency_injection(executor):
    # Test 12 - Dependency Injection
    # Verified by the fact we just executed all above tests entirely using FakePairwiseRoutingService.
    assert isinstance(executor.pairwise_service, FakePairwiseRoutingService)


def test_synthetic_end_to_end_routing():
    # 34. Integration Test
    # Create a synthetic graph A -> B -> C -> D
    G = nx.MultiDiGraph()
    G.add_node(1, x=0.0, y=0.0)
    G.add_node(2, x=0.0, y=1.0)
    G.add_node(3, x=0.0, y=2.0)
    G.add_node(4, x=0.0, y=3.0)
    
    G.add_edge(1, 2, final_cost=0.2, length=100.0)
    G.add_edge(2, 3, final_cost=0.3, length=200.0)
    G.add_edge(3, 4, final_cost=0.4, length=150.0)
    
    astar = AStarRoutingEngine(graph=G, cost_attribute="final_cost")
    service = PairwiseRoutingService(astar_engine=astar)
    executor = MultiStopRouteExecutor(pairwise_service=service)
    
    seq = DeliverySequence(sequence=(2, 3, 4))
    result = executor.execute(1, seq)
    
    assert result.success is True
    assert len(result.routes) == 3
    
    assert result.routes[0].source_node == 1
    assert result.routes[0].destination_node == 2
    assert pytest.approx(result.routes[0].total_cost) == 0.2
    
    assert result.routes[1].source_node == 2
    assert result.routes[1].destination_node == 3
    assert pytest.approx(result.routes[1].total_cost) == 0.3
    
    assert result.routes[2].source_node == 3
    assert result.routes[2].destination_node == 4
    assert pytest.approx(result.routes[2].total_cost) == 0.4
    
    assert pytest.approx(result.total_cost) == 0.9
    assert pytest.approx(result.total_distance_m) == 450.0
