import pytest
from dataclasses import FrozenInstanceError

from src.routing.delivery_sequence import (
    DeliverySequenceRequest,
    DeliverySequence,
    DeliverySequenceResult,
)
from src.routing.route_result import RouteResult

def test_valid_multi_delivery_request():
    # Test 1 & 4 - Valid multi-delivery request
    request = DeliverySequenceRequest(
        start_node="P",
        delivery_nodes=(1, 2, 3, 4)
    )
    assert request.start_node == "P"
    assert request.delivery_nodes == (1, 2, 3, 4)

def test_single_delivery_request():
    # Test 2 - Single delivery
    request = DeliverySequenceRequest(
        start_node="P",
        delivery_nodes=(1,)
    )
    assert request.start_node == "P"
    assert request.delivery_nodes == (1,)

def test_empty_delivery_list():
    # Test 3 - Empty delivery list
    # Explicitly chosen behavior: Valid request, representing nothing to sequence.
    request = DeliverySequenceRequest(
        start_node="P",
        delivery_nodes=()
    )
    assert request.start_node == "P"
    assert request.delivery_nodes == ()
    
    seq = DeliverySequence(sequence=())
    assert seq.sequence == ()

def test_duplicate_delivery_validation():
    # Test 5 - Duplicate delivery validation
    # Duplicate deliveries should be explicitly rejected.
    with pytest.raises(ValueError, match="Duplicate delivery nodes"):
        DeliverySequenceRequest(
            start_node="P",
            delivery_nodes=(1, 2, 1)
        )

def test_start_node_validation():
    # Test 6 - Start node validation
    # The start node CAN be within the delivery nodes
    request = DeliverySequenceRequest(
        start_node=1,
        delivery_nodes=(1, 2, 3)
    )
    assert request.start_node == 1
    assert request.delivery_nodes == (1, 2, 3)

def test_delivery_node_validation():
    # Test 7 - Delivery node validation
    # Domain representations should just hold the IDs without external validation.
    # Ensuring basic types can be stored.
    request = DeliverySequenceRequest(start_node=1, delivery_nodes=(2, 3, 4))
    assert request.start_node == 1
    assert request.delivery_nodes == (2, 3, 4)

def test_sequence_preserves_order():
    # Test 8 - Sequence preserves order
    # It should not sort or reorder the tuple.
    seq = DeliverySequence(sequence=(3, 1, 4, 2))
    assert seq.sequence == (3, 1, 4, 2)

def test_immutability():
    # Test 9 - Immutability
    request = DeliverySequenceRequest(start_node="P", delivery_nodes=(1, 2))
    
    with pytest.raises(FrozenInstanceError):
        request.start_node = "X" # type: ignore
        
    seq = DeliverySequence(sequence=(1, 2))
    with pytest.raises(FrozenInstanceError):
        seq.sequence = (2, 1) # type: ignore

def test_result_contract():
    # Test 10 - Result contract
    # Verify the struct can hold standard route results without making fake logic
    seq = DeliverySequence(sequence=(1, 2))
    
    route_pa = RouteResult(
        success=True, 
        source_node="P", 
        destination_node=1, 
        total_cost=10.0,
        path_nodes=["P", 1]
    )
    route_ab = RouteResult(
        success=True, 
        source_node=1, 
        destination_node=2, 
        total_cost=5.0,
        path_nodes=[1, 2]
    )
    
    result = DeliverySequenceResult(
        delivery_sequence=seq,
        total_cost=15.0,
        routes=(route_pa, route_ab)
    )
    
    assert result.delivery_sequence.sequence == (1, 2)
    assert result.total_cost == 15.0
    assert len(result.routes) == 2
    assert result.routes[0].source_node == "P"
    assert result.routes[1].destination_node == 2
    
    # Verify immutability
    with pytest.raises(FrozenInstanceError):
        result.total_cost = 20.0 # type: ignore
