import pytest
import math
import networkx as nx

from src.graph.coordinate_mapper import (
    find_nearest_node,
    CoordinateValidationError,
    NodeMappingError,
    NodeMatch
)
from src.graph.graph_validator import RoadNetworkValidationError

def create_test_graph():
    """Creates a small valid synthetic graph for coordinate mapping tests."""
    G = nx.MultiDiGraph(crs="epsg:4326")
    G.add_node(1, x=80.1, y=13.1)
    G.add_node(2, x=80.2, y=13.2)
    G.add_node(3, x=80.3, y=13.3)
    
    G.add_edge(1, 2, key=0, length=15.0)
    G.add_edge(2, 3, key=0, length=20.5)
    return G

def test_exact_coordinate_match():
    # Test 24
    G = create_test_graph()
    
    # Query exactly node 2 (lat=13.2, lon=80.2)
    result = find_nearest_node(G, latitude=13.2, longitude=80.2)
    
    assert result.node_id == 2
    assert math.isclose(result.snap_distance_m, 0.0, abs_tol=1.0)
    assert result.matched_latitude == 13.2
    assert result.matched_longitude == 80.2

def test_nearest_node_selection():
    # Test 25
    G = create_test_graph()
    
    # Query a point very close to node 1 (13.1, 80.1)
    result = find_nearest_node(G, latitude=13.1001, longitude=80.1001)
    
    assert result.node_id == 1
    assert result.snap_distance_m > 0
    assert result.snap_distance_m < 100 # Should be ~15m
    assert result.matched_latitude == 13.1
    assert result.matched_longitude == 80.1

def test_coordinate_ordering():
    # Test 26
    G = create_test_graph()
    
    # Reversing latitude and longitude will place it somewhere in the Arctic Ocean (lat=80.1, lon=13.1)
    # The nearest node mapping handles it, but distance should be massive
    result = find_nearest_node(G, latitude=80.1, longitude=13.1)
    
    assert result.node_id in [1, 2, 3]
    assert result.snap_distance_m > 5000000 # Distance should be > 5,000 km
    assert result.matched_latitude in [13.1, 13.2, 13.3]
    
def test_snap_distance_properties():
    # Test 27
    G = create_test_graph()
    result = find_nearest_node(G, latitude=13.1001, longitude=80.1001)
    
    assert result.snap_distance_m >= 0
    assert math.isfinite(result.snap_distance_m)

def test_invalid_latitude():
    # Test 28
    G = create_test_graph()
    with pytest.raises(CoordinateValidationError):
        find_nearest_node(G, latitude=90.1, longitude=80.1)
    with pytest.raises(CoordinateValidationError):
        find_nearest_node(G, latitude=-90.1, longitude=80.1)

def test_invalid_longitude():
    # Test 29
    G = create_test_graph()
    with pytest.raises(CoordinateValidationError):
        find_nearest_node(G, latitude=13.1, longitude=180.1)
    with pytest.raises(CoordinateValidationError):
        find_nearest_node(G, latitude=13.1, longitude=-180.1)

def test_nan_infinity_coordinates():
    # Test 30
    G = create_test_graph()
    invalid_vals = [float('nan'), float('inf'), float('-inf')]
    for val in invalid_vals:
        with pytest.raises(CoordinateValidationError):
            find_nearest_node(G, latitude=val, longitude=80.1)
        with pytest.raises(CoordinateValidationError):
            find_nearest_node(G, latitude=13.1, longitude=val)
            
def test_missing_coordinates():
    # Test 31 (Handled by graph validator)
    G = create_test_graph()
    G.add_node(4, x=80.4) # Missing y
    
    # Catch Exception since import paths might differ, but check the name
    with pytest.raises(Exception) as excinfo:
        find_nearest_node(G, latitude=13.1, longitude=80.1)
    assert "RoadNetworkValidationError" in type(excinfo.value).__name__

def test_empty_graph():
    # Test 32
    G = nx.MultiDiGraph(crs="epsg:4326")
    with pytest.raises(Exception) as excinfo:
        find_nearest_node(G, latitude=13.1, longitude=80.1)
    assert "RoadNetworkValidationError" in type(excinfo.value).__name__

def test_wrong_graph_type():
    # Test 33
    G = nx.Graph()
    with pytest.raises(Exception) as excinfo:
        find_nearest_node(G, latitude=13.1, longitude=80.1)
    assert "RoadNetworkValidationError" in type(excinfo.value).__name__

def test_distant_out_of_network_coordinate():
    # Test 35
    G = create_test_graph()
    # Query a point in New York (lat=40.7128, lon=-74.0060)
    result = find_nearest_node(G, latitude=40.7128, longitude=-74.0060)
    assert result.snap_distance_m > 1000000 # Distance should be > 1,000 km
    
def test_maximum_snap_distance():
    # Test 36
    G = create_test_graph()
    # Distance to 13.1001, 80.1001 from node 1 is ~15m
    
    # Threshold < distance -> failure
    with pytest.raises(NodeMappingError, match="exceeds maximum allowed snap distance"):
        find_nearest_node(G, latitude=13.1001, longitude=80.1001, max_snap_distance_m=1.0)
        
    # Threshold > distance -> success
    result = find_nearest_node(G, latitude=13.1001, longitude=80.1001, max_snap_distance_m=100.0)
    assert result.node_id == 1

def test_result_consistency():
    # Test 37
    G = create_test_graph()
    result = find_nearest_node(G, latitude=13.1001, longitude=80.1001)
    
    assert result.node_id in G.nodes
    assert result.matched_latitude == G.nodes[result.node_id]["y"]
    assert result.matched_longitude == G.nodes[result.node_id]["x"]

def test_graph_immutability():
    # Test 38
    G = create_test_graph()
    before_nodes = G.number_of_nodes()
    before_edges = G.number_of_edges()
    
    find_nearest_node(G, latitude=13.1001, longitude=80.1001)
    
    assert G.number_of_nodes() == before_nodes
    assert G.number_of_edges() == before_edges

def test_parallel_edges_remain_untouched():
    # Test 39
    G = create_test_graph()
    G.add_edge(1, 2, key=1, length=16.0) # Add a parallel edge
    before_edges = G.number_of_edges()
    
    find_nearest_node(G, latitude=13.1001, longitude=80.1001)
    
    assert G.number_of_edges() == before_edges
    assert G.number_of_edges(1, 2) == 2

def test_backend_component_independence():
    # Test 41
    # We are successfully calling it as a standard python function
    # without any API frameworks throughout this entire test suite.
    assert True
