import math
import pytest
import networkx as nx

from src.routing.edge_cost import EdgeId
from src.routing.exceptions import InvalidNodeError, GraphUnavailableError
from src.routing.candidate_selector import CandidateEdgeSelector

@pytest.fixture
def synthetic_graph():
    """
    Creates a small synthetic graph for testing candidate selection.
    Nodes are arranged in a grid-like manner.
    0,0 (Node 1)    ---   0,1 (Node 2)
      |                     |
    1,0 (Node 3)    ---   1,1 (Node 4)
    """
    G = nx.MultiDiGraph()
    # Add nodes with x (lon) and y (lat)
    # Using small values for simplicity.
    G.add_node(1, y=0.0, x=0.0)
    G.add_node(2, y=0.0, x=1.0)
    G.add_node(3, y=1.0, x=0.0)
    G.add_node(4, y=1.0, x=1.0)
    
    # Add directed edges
    G.add_edge(1, 2, key=0)
    G.add_edge(2, 4, key=0)
    G.add_edge(1, 3, key=0)
    G.add_edge(3, 4, key=0)
    
    # Add reverse edge
    G.add_edge(4, 2, key=0)
    
    # Add parallel edge
    G.add_edge(1, 2, key=1)
    
    # Add a long edge outside expected bounds
    G.add_node(5, y=10.0, x=10.0)
    G.add_edge(4, 5, key=0)
    
    # Node missing coordinates
    G.add_node(6)
    G.add_edge(4, 6, key=0)
    
    return G

def test_basic_candidate_extraction(synthetic_graph):
    """Test 1: Given a small graph and one delivery node, verify expected EdgeIds."""
    # buffer of ~ 150km, which is slightly more than 1 degree.
    # Start: 1 (0,0), Delivery: 3 (1,0)
    # Bbox: lat 0 to 1, lon 0 to 0. 
    # With large buffer, it should cover up to lat 2 and lon 1.
    selector = CandidateEdgeSelector(synthetic_graph, buffer_m=200000.0)
    candidates = selector.select(1, (3,))
    
    # Nodes in buffer: 1(0,0), 2(0,1), 3(1,0), 4(1,1)
    # Node 5(10,10) is out. Node 6 has no coordinates.
    # Therefore, expected edges: (1,2,0), (1,2,1), (2,4,0), (1,3,0), (3,4,0), (4,2,0)
    expected = {
        EdgeId(1, 2, 0),
        EdgeId(1, 2, 1),
        EdgeId(2, 4, 0),
        EdgeId(1, 3, 0),
        EdgeId(3, 4, 0),
        EdgeId(4, 2, 0),
    }
    assert candidates == expected

def test_multiple_delivery_nodes(synthetic_graph):
    """Test 2: Verify that the bounding region includes all requested locations."""
    # Start: 1(0,0), Deliveries: 2(0,1) and 3(1,0). No buffer.
    # Bbox: lat 0 to 1, lon 0 to 1.
    # This covers nodes 1, 2, 3, 4 perfectly even with buffer=0.
    selector = CandidateEdgeSelector(synthetic_graph, buffer_m=0.0)
    candidates = selector.select(1, (2, 3))
    
    expected = {
        EdgeId(1, 2, 0),
        EdgeId(1, 2, 1),
        EdgeId(2, 4, 0),
        EdgeId(1, 3, 0),
        EdgeId(3, 4, 0),
        EdgeId(4, 2, 0),
    }
    assert candidates == expected

def test_parallel_edges_preserved(synthetic_graph):
    """Test 3: Verify that edge keys are preserved independently."""
    selector = CandidateEdgeSelector(synthetic_graph, buffer_m=0.0)
    candidates = selector.select(1, (2,))
    assert EdgeId(1, 2, 0) in candidates
    assert EdgeId(1, 2, 1) in candidates

def test_directed_edges_preserved(synthetic_graph):
    """Test 4: Verify that reverse edges remain distinct."""
    selector = CandidateEdgeSelector(synthetic_graph, buffer_m=0.0)
    candidates = selector.select(2, (4,))
    assert EdgeId(2, 4, 0) in candidates
    assert EdgeId(4, 2, 0) in candidates

def test_buffer_behavior(synthetic_graph):
    """Test 5: Use two different buffer sizes and verify candidate-region changes."""
    # Nodes: 1(0,0), 5(10,10)
    # Start: 1, Delivery: 1. Box: lat 0, lon 0.
    # Buffer: 0
    selector_small = CandidateEdgeSelector(synthetic_graph, buffer_m=0.0)
    candidates_small = selector_small.select(1, ())
    
    # Edges within [0,0]: none have both endpoints at 0,0.
    assert len(candidates_small) == 0
    
    # Buffer large enough to reach node 2 (0,1) but not 3 (1,0)
    # 1 degree lon at lat 0 is ~111,320m.
    selector_medium = CandidateEdgeSelector(synthetic_graph, buffer_m=115000.0)
    candidates_medium = selector_medium.select(1, ())
    # This buffer extends lat to [-1, 1], lon to [-1, 1].
    # Both endpoints of 1->2, 1->3, etc. are in the box.
    assert EdgeId(1, 2, 0) in candidates_medium
    
    # Node 5 is at 10,10. Buffer of 2,000,000m (~18 degrees) reaches node 5.
    selector_large = CandidateEdgeSelector(synthetic_graph, buffer_m=2000000.0)
    candidates_large = selector_large.select(1, ())
    assert EdgeId(4, 5, 0) in candidates_large

def test_zero_buffer(synthetic_graph):
    """Test 6: Verify deterministic behavior when buffer_m=0."""
    selector = CandidateEdgeSelector(synthetic_graph, buffer_m=0.0)
    # Start: 1(0,0), Delivery: 4(1,1). Bbox exactly covers 1, 2, 3, 4.
    candidates = selector.select(1, (4,))
    assert len(candidates) == 6

def test_invalid_start_node(synthetic_graph):
    """Test 7: Verify an explicit error."""
    selector = CandidateEdgeSelector(synthetic_graph)
    with pytest.raises(InvalidNodeError, match="not found in the graph"):
        selector.select(999, (2,))

def test_invalid_delivery_node(synthetic_graph):
    """Test 8: Verify an explicit error."""
    selector = CandidateEdgeSelector(synthetic_graph)
    with pytest.raises(InvalidNodeError, match="not found in the graph"):
        selector.select(1, (2, 999))

def test_invalid_buffer(synthetic_graph):
    """Test 9: Verify negative, NaN, Infinity, non-numeric."""
    with pytest.raises(ValueError):
        CandidateEdgeSelector(synthetic_graph, buffer_m=-100.0)
    with pytest.raises(ValueError):
        CandidateEdgeSelector(synthetic_graph, buffer_m=float('inf'))
    with pytest.raises(ValueError):
        CandidateEdgeSelector(synthetic_graph, buffer_m=float('nan'))
    with pytest.raises(TypeError):
        CandidateEdgeSelector(synthetic_graph, buffer_m="100")

def test_missing_coordinates(synthetic_graph):
    """Test 10: Verify explicit handling."""
    # Node 6 has no coordinates. Asking for it should raise error.
    selector = CandidateEdgeSelector(synthetic_graph)
    with pytest.raises(ValueError, match="is missing coordinate 'x' or 'y'"):
        selector.select(6, ())
        
    # But if node 6 is just in the graph, it should skip edges involving it gracefully.
    candidates = selector.select(1, (4,))
    # Edge (4, 6, 0) should be skipped.
    assert EdgeId(4, 6, 0) not in candidates

def test_graph_immutability(synthetic_graph):
    """Test 11: Take a snapshot of graph nodes/edges/attributes before selection and verify that nothing changes afterward."""
    nodes_before = list(synthetic_graph.nodes(data=True))
    edges_before = list(synthetic_graph.edges(keys=True, data=True))
    
    selector = CandidateEdgeSelector(synthetic_graph)
    selector.select(1, (2, 3, 4))
    
    nodes_after = list(synthetic_graph.nodes(data=True))
    edges_after = list(synthetic_graph.edges(keys=True, data=True))
    
    assert nodes_before == nodes_after
    assert edges_before == edges_after

def test_dynamic_cost_independence(synthetic_graph):
    """Test 12: Construct edges containing fake final_cost values and verify candidate selection ignores them."""
    synthetic_graph[1][2][0]["final_cost"] = 999.9
    synthetic_graph[1][2][1]["traffic_score"] = 5.0
    
    selector = CandidateEdgeSelector(synthetic_graph, buffer_m=0.0)
    candidates = selector.select(1, (4,))
    
    # Should still select the edges regardless of attributes
    assert EdgeId(1, 2, 0) in candidates
    assert EdgeId(1, 2, 1) in candidates

def test_geographic_conversion():
    """Test 26: Because the buffer is expressed in metres, test the actual geographic effect."""
    # Construct points whose distance is approximately inside/outside buffer.
    # 1 degree lat = 111,320m.
    G = nx.MultiDiGraph()
    G.add_node(1, y=0.0, x=0.0)
    G.add_node(2, y=0.009, x=0.0) # ~ 1001.88 meters away north
    G.add_node(3, y=0.011, x=0.0) # ~ 1224.52 meters away north
    G.add_edge(1, 2, key=0)
    G.add_edge(1, 3, key=0)
    
    # 1000m buffer
    selector = CandidateEdgeSelector(G, buffer_m=1000.0)
    candidates = selector.select(1, ())
    
    # Edge 1->2 is outside the 1000m bounding box (0.009 degree * 111320 = 1001.88m, so lat_max_buffered is 0.00898).
    # Wait, exactly: 1000 / 111320 = 0.008983. Max buffered lat = 0.008983. Node 2 is at 0.009 > 0.008983.
    # Therefore, 1->2 is OUTSIDE.
    assert len(candidates) == 0
    
    # 1100m buffer
    selector_1100 = CandidateEdgeSelector(G, buffer_m=1100.0)
    candidates_1100 = selector_1100.select(1, ())
    
    # 1100 / 111320 = 0.00988. Node 2 is IN. Node 3 is OUT.
    assert EdgeId(1, 2, 0) in candidates_1100
    assert EdgeId(1, 3, 0) not in candidates_1100

def test_geographic_conversion_longitude_scaling():
    """Verify longitude scaling adjusts properly at higher latitudes."""
    G = nx.MultiDiGraph()
    G.add_node(1, y=60.0, x=0.0) # At 60 degrees north, cos(60) = 0.5. 1 deg lon = ~55,660m.
    G.add_node(2, y=60.0, x=0.019) # ~ 1057m away east
    G.add_node(3, y=60.0, x=0.021) # ~ 1168m away east
    G.add_edge(1, 2, key=0)
    G.add_edge(1, 3, key=0)
    
    # 1100m buffer. lon_delta = 1100 / (111320 * 0.5) = 1100 / 55660 = 0.01976 degrees.
    selector = CandidateEdgeSelector(G, buffer_m=1100.0)
    candidates = selector.select(1, ())
    
    # Node 2 is at 0.019 < 0.01976. Node 3 is at 0.021 > 0.01976.
    assert EdgeId(1, 2, 0) in candidates
    assert EdgeId(1, 3, 0) not in candidates

def test_integration_lightweight():
    """Test 27: Lightweight integration test using a small synthetic OSM-like graph."""
    G = nx.MultiDiGraph()
    G.add_node(100, y=13.0, x=80.0)
    G.add_node(101, y=13.001, x=80.0)
    G.add_node(102, y=13.002, x=80.0)
    G.add_node(200, y=13.01, x=80.0)
    
    G.add_edge(100, 101, key=0)
    G.add_edge(101, 102, key=0)
    G.add_edge(102, 200, key=0) # Path towards delivery
    
    start_node = 100
    delivery_nodes = (200,)
    
    selector = CandidateEdgeSelector(G, buffer_m=500.0)
    candidates = selector.select(start_node, delivery_nodes)
    
    # Start: (13.0, 80.0), Delivery: (13.01, 80.0).
    # Box spans 13.0 to 13.01. Buffer expands it further.
    # Therefore, all edges should be included.
    expected = {
        EdgeId(100, 101, 0),
        EdgeId(101, 102, 0),
        EdgeId(102, 200, 0),
    }
    assert candidates == expected
