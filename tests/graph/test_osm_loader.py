import pytest
import networkx as nx
from unittest.mock import patch
from src.graph.osm_loader import download_road_network

def create_synthetic_osm_graph() -> nx.MultiDiGraph:
    """Creates a tiny synthetic graph resembling an OSM network."""
    G = nx.MultiDiGraph()
    G.add_node(101, x=80.123, y=13.123)
    G.add_node(102, x=80.124, y=13.124)
    # Add an edge with a length attribute (typical of OSMnx)
    G.add_edge(101, 102, key=0, length=15.5, highway="residential", oneway=False)
    return G

@patch("src.graph.osm_loader.ox.graph_from_place")
def test_download_road_network_return_type(mock_graph_from_place):
    # Setup mock
    mock_graph_from_place.return_value = create_synthetic_osm_graph()
    
    # Execute
    graph = download_road_network("Fake City, Fake Country")
    
    # Assert type
    assert isinstance(graph, nx.MultiDiGraph)

@patch("src.graph.osm_loader.ox.graph_from_place")
def test_download_road_network_not_empty(mock_graph_from_place):
    mock_graph_from_place.return_value = create_synthetic_osm_graph()
    
    graph = download_road_network("Fake City, Fake Country")
    
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0

@patch("src.graph.osm_loader.ox.graph_from_place")
def test_download_road_network_node_structure(mock_graph_from_place):
    mock_graph_from_place.return_value = create_synthetic_osm_graph()
    
    graph = download_road_network("Fake City, Fake Country")
    
    # Get a sample node
    node_id = list(graph.nodes())[0]
    node_data = graph.nodes[node_id]
    
    assert "x" in node_data
    assert "y" in node_data
    assert isinstance(node_data["x"], (int, float))
    assert isinstance(node_data["y"], (int, float))

@patch("src.graph.osm_loader.ox.graph_from_place")
def test_download_road_network_edge_structure(mock_graph_from_place):
    mock_graph_from_place.return_value = create_synthetic_osm_graph()
    
    graph = download_road_network("Fake City, Fake Country")
    
    # Get a sample edge
    u, v, key, data = list(graph.edges(data=True, keys=True))[0]
    
    assert "length" in data
    assert data["length"] > 0

@patch("src.graph.osm_loader.ox.graph_from_place")
def test_download_road_network_failure(mock_graph_from_place):
    # Setup mock to raise an exception simulating a network or OSMnx failure
    mock_graph_from_place.side_effect = Exception("OSM Overpass API timeout")
    
    # It must raise a RuntimeError per our implementation, not swallow it
    with pytest.raises(RuntimeError, match="OSM road network download failed for place"):
        download_road_network("Nowhere City")
