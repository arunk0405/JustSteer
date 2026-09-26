import pytest
from datetime import datetime, timezone
from src.routing.edge_cost import EdgeId, EdgeCost
from src.routing.edge_cost_store import EdgeCostStore
from src.routing.contracts import CombinedEdgeCost

@pytest.fixture
def store():
    return EdgeCostStore()

@pytest.fixture
def mock_cost_data():
    return CombinedEdgeCost(
        edge_id="dummy",
        predicted_travel_time=5.0,
        raw_safety_risk=20.0,
        normalized_travel_time=0.5,
        normalized_safety_risk=0.2,
        travel_time_weight=0.7,
        safety_weight=0.3,
        combined_cost=0.41
    )

def create_mock_cost(edge_id: EdgeId, cost_data: CombinedEdgeCost) -> EdgeCost:
    return EdgeCost(
        edge_id=edge_id,
        cost_data=cost_data,
        generated_at=datetime.now(timezone.utc)
    )

def test_set_and_get(store, mock_cost_data):
    edge_id = EdgeId(1, 2, 0)
    cost = create_mock_cost(edge_id, mock_cost_data)
    
    store.set(edge_id, cost)
    retrieved = store.get(edge_id)
    
    assert retrieved is cost
    assert store.contains(edge_id) is True

def test_get_missing(store):
    edge_id = EdgeId(1, 2, 0)
    assert store.get(edge_id) is None
    assert store.contains(edge_id) is False

def test_update_semantics(store, mock_cost_data):
    edge_id = EdgeId(1, 2, 0)
    cost1 = create_mock_cost(edge_id, mock_cost_data)
    
    # modify data slightly for the second update
    mock_cost_data2 = CombinedEdgeCost(**{**mock_cost_data.__dict__, "combined_cost": 0.99})
    cost2 = create_mock_cost(edge_id, mock_cost_data2)
    
    store.set(edge_id, cost1)
    store.set(edge_id, cost2)
    
    retrieved = store.get(edge_id)
    assert retrieved is cost2
    assert retrieved.cost_data.combined_cost == 0.99

def test_multiple_parallel_edges(store, mock_cost_data):
    # Two parallel edges between the same nodes
    edge1 = EdgeId(100, 200, 0)
    edge2 = EdgeId(100, 200, 1)
    
    cost1 = create_mock_cost(edge1, mock_cost_data)
    
    mock_cost_data2 = CombinedEdgeCost(**{**mock_cost_data.__dict__, "combined_cost": 0.88})
    cost2 = create_mock_cost(edge2, mock_cost_data2)
    
    store.set(edge1, cost1)
    store.set(edge2, cost2)
    
    assert store.get(edge1) is cost1
    assert store.get(edge2) is cost2
    assert store.get(edge1).cost_data.combined_cost == 0.41
    assert store.get(edge2).cost_data.combined_cost == 0.88

def test_remove(store, mock_cost_data):
    edge_id = EdgeId(1, 2, 0)
    cost = create_mock_cost(edge_id, mock_cost_data)
    
    store.set(edge_id, cost)
    assert store.contains(edge_id)
    
    store.remove(edge_id)
    assert not store.contains(edge_id)
    assert store.get(edge_id) is None

def test_clear(store, mock_cost_data):
    edge1 = EdgeId(1, 2, 0)
    edge2 = EdgeId(2, 3, 0)
    
    store.set(edge1, create_mock_cost(edge1, mock_cost_data))
    store.set(edge2, create_mock_cost(edge2, mock_cost_data))
    
    assert store.contains(edge1)
    assert store.contains(edge2)
    
    store.clear()
    
    assert not store.contains(edge1)
    assert not store.contains(edge2)

def test_invalid_types(store):
    with pytest.raises(TypeError):
        store.set("invalid_id", None) # type: ignore
        
    edge_id = EdgeId(1, 2, 0)
    with pytest.raises(TypeError):
        store.set(edge_id, "invalid_cost") # type: ignore
        
    with pytest.raises(TypeError):
        store.get("invalid_id") # type: ignore
        
    with pytest.raises(TypeError):
        store.contains("invalid_id") # type: ignore
        
    with pytest.raises(TypeError):
        store.remove("invalid_id") # type: ignore
