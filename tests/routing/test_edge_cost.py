import pytest
from datetime import datetime, timezone
from src.routing.edge_cost import EdgeId, EdgeCost
from src.routing.contracts import CombinedEdgeCost

def test_edge_id_valid_int():
    edge_id = EdgeId(source=1, target=2, key=0)
    assert edge_id.source == 1
    assert edge_id.target == 2
    assert edge_id.key == 0

def test_edge_id_valid_str_key():
    edge_id = EdgeId(source=1, target=2, key="main")
    assert edge_id.key == "main"

def test_edge_id_invalid_types():
    with pytest.raises(TypeError):
        EdgeId(source="1", target=2, key=0)
    with pytest.raises(TypeError):
        EdgeId(source=1, target="2", key=0)
    with pytest.raises(TypeError):
        EdgeId(source=1, target=2, key=1.5)
    with pytest.raises(TypeError):
        EdgeId(source=True, target=2, key=0)
    with pytest.raises(TypeError):
        EdgeId(source=1, target=False, key=0)
    with pytest.raises(TypeError):
        EdgeId(source=1, target=2, key=None)
    with pytest.raises(TypeError):
        EdgeId(source=1, target=2, key=True)

def test_edge_id_equality():
    id1 = EdgeId(1, 2, 0)
    id2 = EdgeId(1, 2, 0)
    id3 = EdgeId(1, 2, 1)
    id4 = EdgeId(2, 1, 0)

    assert id1 == id2
    assert id1 != id3
    assert id1 != id4

def test_edge_id_hashability():
    id1 = EdgeId(1, 2, 0)
    id2 = EdgeId(1, 2, 0)
    
    edge_map = {id1: "value"}
    assert id2 in edge_map
    assert edge_map[id2] == "value"

def test_edge_cost_creation():
    edge_id = EdgeId(100, 200, 0)
    cost_data = CombinedEdgeCost(
        edge_id="100_200_0",
        predicted_travel_time=5.0,
        raw_safety_risk=20.0,
        normalized_travel_time=0.5,
        normalized_safety_risk=0.2,
        travel_time_weight=0.7,
        safety_weight=0.3,
        combined_cost=0.41
    )
    generated_at = datetime.now(timezone.utc)
    
    edge_cost = EdgeCost(
        edge_id=edge_id,
        cost_data=cost_data,
        generated_at=generated_at,
        traffic_model_version="v1.0",
        safety_model_version="v2.1"
    )
    
    assert edge_cost.edge_id == edge_id
    assert edge_cost.cost_data.combined_cost == 0.41
    assert edge_cost.generated_at == generated_at
    assert edge_cost.traffic_model_version == "v1.0"
