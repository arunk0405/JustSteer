import pytest
from typing import Any, List, Tuple

from src.routing.delivery_sequence import DeliverySequenceRequest
from src.routing.route_result import RouteResult
from src.routing.pairwise_route_matrix import PairwiseRouteMatrixBuilder
from src.routing.exceptions import RouteNotFoundError

class FakePairwiseRoutingService:
    def __init__(self):
        self.calls: List[Tuple[Any, Any]] = []
        self.unreachable_pairs = set()
        
    def route(self, source: Any, destination: Any) -> RouteResult:
        self.calls.append((source, destination))
        
        if (source, destination) in self.unreachable_pairs:
            raise RouteNotFoundError(f"No route {source} -> {destination}")
            
        return RouteResult(
            success=True,
            source_node=source,
            destination_node=destination,
            total_cost=float(len(str(source)) + len(str(destination))),
            path_nodes=[source, destination]
        )

@pytest.fixture
def fake_service():
    return FakePairwiseRoutingService()

@pytest.fixture
def builder(fake_service):
    return PairwiseRouteMatrixBuilder(fake_service)

def test_zero_deliveries(builder, fake_service):
    # Test 1 - Zero Deliveries
    request = DeliverySequenceRequest(start_node="P", delivery_nodes=())
    matrix = builder.build(request)
    
    assert len(fake_service.calls) == 0
    with pytest.raises(KeyError):
        matrix.get_route("P", "P")

def test_one_delivery(builder, fake_service):
    # Test 2 - One Delivery
    request = DeliverySequenceRequest(start_node="P", delivery_nodes=("A",))
    matrix = builder.build(request)
    
    assert len(fake_service.calls) == 1
    assert fake_service.calls[0] == ("P", "A")
    
    assert matrix.get_route("P", "A") is not None
    
    with pytest.raises(KeyError):
        matrix.get_route("A", "P")
        
    with pytest.raises(KeyError):
        matrix.get_route("A", "A")

def test_two_deliveries(builder, fake_service):
    # Test 3 - Two Deliveries
    request = DeliverySequenceRequest(start_node="P", delivery_nodes=("A", "B"))
    builder.build(request)
    assert len(fake_service.calls) == 4
    assert set(fake_service.calls) == {("P", "A"), ("P", "B"), ("A", "B"), ("B", "A")}

def test_three_deliveries(builder, fake_service):
    # Test 4 - Three Deliveries
    request = DeliverySequenceRequest(start_node="P", delivery_nodes=("A", "B", "C"))
    builder.build(request)
    assert len(fake_service.calls) == 9

def test_four_deliveries(builder, fake_service):
    # Test 5 - Four Deliveries
    request = DeliverySequenceRequest(start_node="P", delivery_nodes=("A", "B", "C", "D"))
    builder.build(request)
    assert len(fake_service.calls) == 16
    for u, v in fake_service.calls:
        assert u != v  # Verify no self pairs

def test_five_deliveries(builder, fake_service):
    # Test 6 - Five Deliveries
    request = DeliverySequenceRequest(start_node="P", delivery_nodes=("A", "B", "C", "D", "E"))
    builder.build(request)
    assert len(fake_service.calls) == 25

def test_directionality(builder):
    # Test 7 - Directionality
    request = DeliverySequenceRequest(start_node="P", delivery_nodes=("A", "B"))
    matrix = builder.build(request)
    
    res_ab = matrix.get_route("A", "B")
    res_ba = matrix.get_route("B", "A")
    
    assert res_ab is not res_ba
    assert res_ab.source_node == "A"
    assert res_ba.source_node == "B"

def test_unreachable_direction(builder, fake_service):
    # Test 8, 17 - One-way / Unreachable Direction & Handling
    fake_service.unreachable_pairs.add(("B", "A"))
    
    request = DeliverySequenceRequest(start_node="P", delivery_nodes=("A", "B"))
    matrix = builder.build(request)
    
    # A -> B should be valid
    assert matrix.get_route("A", "B") is not None
    
    # B -> A should explicitly return None to signal evaluated but unreachable
    assert matrix.get_route("B", "A") is None

def test_exact_routeresult_preservation(builder, fake_service):
    # Test 9 - Exact RouteResult Preservation
    request = DeliverySequenceRequest(start_node="P", delivery_nodes=("A",))
    matrix = builder.build(request)
    
    # Let's get the specific object it returned
    # The service just creates it on the fly, let's inject a specific object
    expected_route = RouteResult(success=True, source_node="P", destination_node="A")
    
    def mocked_route(src, dst):
        return expected_route
        
    fake_service.route = mocked_route
    matrix2 = builder.build(request)
    
    assert matrix2.get_route("P", "A") is expected_route

def test_no_duplicate_routing_calls(builder, fake_service):
    # Test 10 - No Duplicate Routing Calls
    request = DeliverySequenceRequest(start_node="P", delivery_nodes=("A", "B", "C"))
    builder.build(request)
    assert len(fake_service.calls) == 9
    assert len(set(fake_service.calls)) == 9  # All calls are unique

def test_self_pairs_not_requested(builder, fake_service):
    # Test 11 - Self-Pairs Are Not Requested
    request = DeliverySequenceRequest(start_node="P", delivery_nodes=("A", "B"))
    builder.build(request)
    
    for u, v in fake_service.calls:
        assert u != v

def test_return_to_start_not_requested(builder, fake_service):
    # Test 12 - Return-to-start is not requested
    request = DeliverySequenceRequest(start_node="P", delivery_nodes=("A", "B", "C"))
    builder.build(request)
    
    for u, v in fake_service.calls:
        assert v != "P"

def test_deterministic_call_order(builder, fake_service):
    # Test 13, 20 - Input order is preserved / Deterministic Call Order
    request = DeliverySequenceRequest(start_node="P", delivery_nodes=("C", "A", "B"))
    builder.build(request)
    
    expected_order = [
        ("P", "C"),
        ("P", "A"),
        ("P", "B"),
        ("C", "A"),
        ("C", "B"),
        ("A", "C"),
        ("A", "B"),
        ("B", "C"),
        ("B", "A")
    ]
    
    assert fake_service.calls == expected_order

def test_request_not_mutated(builder):
    # Test 14 - Request is not mutated
    request = DeliverySequenceRequest(start_node="P", delivery_nodes=("A", "B", "C"))
    builder.build(request)
    
    assert request.start_node == "P"
    assert request.delivery_nodes == ("A", "B", "C")

def test_request_local_isolation(builder, fake_service):
    # Test 15 - Request-local isolation
    req1 = DeliverySequenceRequest(start_node="P", delivery_nodes=("A",))
    m1 = builder.build(req1)
    
    req2 = DeliverySequenceRequest(start_node="P", delivery_nodes=("B",))
    m2 = builder.build(req2)
    
    assert m1.get_route("P", "A") is not None
    with pytest.raises(KeyError):
        m2.get_route("P", "A")
        
    assert m2.get_route("P", "B") is not None
    with pytest.raises(KeyError):
        m1.get_route("P", "B")

def test_same_pair_in_separate_requests(builder, fake_service):
    # Test 16 - Same Pair in Separate Requests
    req1 = DeliverySequenceRequest(start_node="P", delivery_nodes=("A",))
    builder.build(req1)
    
    assert fake_service.calls == [("P", "A")]
    fake_service.calls.clear()
    
    req2 = DeliverySequenceRequest(start_node="P", delivery_nodes=("A",))
    builder.build(req2)
    
    # Fake service should be called again because the matrix is request-local
    assert fake_service.calls == [("P", "A")]

def test_empty_partial_matrix_behavior(builder, fake_service):
    # Test 18 - Empty / Partial matrix behavior
    fake_service.unreachable_pairs.add(("A", "B"))
    
    req = DeliverySequenceRequest(start_node="P", delivery_nodes=("A", "B"))
    matrix = builder.build(req)
    
    # A->B was evaluated but is unreachable
    assert matrix.get_route("A", "B") is None
    
    # A->P was NEVER evaluated
    with pytest.raises(KeyError):
        matrix.get_route("A", "P")

def test_dependency_injection(builder):
    # Test 19 - Dependency Injection
    # The builder successfully used FakePairwiseRoutingService without importing A* or OSMnx.
    req = DeliverySequenceRequest(start_node="P", delivery_nodes=("A",))
    builder.build(req)
    assert isinstance(builder.routing_service, FakePairwiseRoutingService)

def test_start_node_equals_delivery_node(builder, fake_service):
    # Test 21+ Edge Cases - Start Node == Delivery Node
    request = DeliverySequenceRequest(start_node="A", delivery_nodes=("A", "B", "C"))
    matrix = builder.build(request)
    
    # Should not request self pair A->A
    for u, v in fake_service.calls:
        assert u != v
        
    # Start -> deliveries (A->B, A->C)
    # Delivery -> deliveries (B->A, B->C, C->A, C->B)
    # Total calls: 6
    assert len(fake_service.calls) == 6
    
    assert matrix.get_route("A", "B") is not None
    assert matrix.get_route("C", "B") is not None
    
    # A->A should not be requested/evaluated
    with pytest.raises(KeyError):
        matrix.get_route("A", "A")
