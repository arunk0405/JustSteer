import math
import pytest
import networkx as nx
from src.graph.graph_validator import validate_road_network, RoadNetworkValidationError

def create_valid_graph():
    """Creates a small valid synthetic graph."""
    G = nx.MultiDiGraph()
    G.add_node(1, x=80.1, y=13.1)
    G.add_node(2, x=80.2, y=13.2)
    G.add_node(3, x=80.3, y=13.3)
    G.add_edge(1, 2, key=0, length=15.0)
    G.add_edge(2, 3, key=0, length=20.5)
    return G

def test_valid_graph():
    G = create_valid_graph()
    # Should not raise an exception
    validate_road_network(G)

def test_empty_graph():
    G = nx.MultiDiGraph()
    with pytest.raises(RoadNetworkValidationError, match="Road network contains no nodes"):
        validate_road_network(G)

def test_wrong_graph_type():
    G = nx.Graph()
    with pytest.raises(RoadNetworkValidationError, match="must be a NetworkX MultiDiGraph"):
        validate_road_network(G)

def test_missing_node_coordinates():
    G = create_valid_graph()
    G.add_node(4, x=80.4) # Missing y
    with pytest.raises(RoadNetworkValidationError, match="missing coordinate"):
        validate_road_network(G)

def test_invalid_coordinates():
    G = create_valid_graph()
    G.add_node(4, x=200.0, y=13.4) # Invalid longitude
    with pytest.raises(RoadNetworkValidationError, match="invalid longitude"):
        validate_road_network(G)

def test_invalid_edge_reference():
    G = create_valid_graph()
    # Forcefully add an edge with a non-existent target.
    # NetworkX maintains integrity (remove_node deletes edges), so we must manually 
    # corrupt the internal dictionary to test the validator's reference checking.
    G._adj[1][4] = {0: {'length': 10.0}}
    with pytest.raises(RoadNetworkValidationError, match="references a missing target node"):
        validate_road_network(G)

def test_invalid_edge_length():
    G = create_valid_graph()
    G.add_edge(1, 3, key=0, length=-5.0) # Negative length
    with pytest.raises(RoadNetworkValidationError, match="negative length"):
        validate_road_network(G)

def test_missing_optional_attributes():
    G = create_valid_graph()
    # Edge missing name, maxspeed, etc. but has length
    G.add_edge(1, 3, key=0, length=10.0)
    # Should pass
    validate_road_network(G)

def test_parallel_edges_preserved():
    G = create_valid_graph()
    G.add_edge(1, 2, key=1, length=12.0)
    validate_road_network(G)
    assert G.number_of_edges() == 3 # 2 original + 1 parallel

def test_directional_edges_preserved():
    G = create_valid_graph()
    # A -> B exists, B -> A does not
    assert not G.has_edge(2, 1)
    validate_road_network(G)
    assert not G.has_edge(2, 1)

def test_validation_is_read_only():
    G = create_valid_graph()
    before_nodes = G.number_of_nodes()
    before_edges = G.number_of_edges()
    
    validate_road_network(G)
    
    assert G.number_of_nodes() == before_nodes
    assert G.number_of_edges() == before_edges
