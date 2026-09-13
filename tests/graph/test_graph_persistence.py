import pytest
import networkx as nx
from pathlib import Path
import math

from src.graph.osm_loader import save_road_network, load_road_network
from src.graph.graph_validator import RoadNetworkValidationError

def create_valid_graph():
    """Creates a small valid synthetic graph for persistence testing."""
    G = nx.MultiDiGraph()
    G.add_node(1, x=80.1, y=13.1)
    G.add_node(2, x=80.2, y=13.2)
    G.add_node(3, x=80.3, y=13.3)
    
    # Add parallel edges
    G.add_edge(1, 2, key=0, length=15.0, highway="residential", oneway=True)
    G.add_edge(1, 2, key=1, length=16.0, highway="primary", oneway=False)
    
    # Add directional edge (2 -> 3 exists, 3 -> 2 does not)
    G.add_edge(2, 3, key=0, length=20.5, geometry="LINESTRING(80.2 13.2, 80.3 13.3)")
    return G

def test_save_and_load_valid_graph(tmp_path):
    # Test 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 (Round trip tests)
    G = create_valid_graph()
    file_path = tmp_path / "test_network.graphml"
    
    # Save
    save_road_network(G, file_path)
    assert file_path.exists()
    
    # Load
    loaded_G = load_road_network(file_path)
    
    # Test 2: Type preserved
    assert isinstance(loaded_G, nx.MultiDiGraph)
    
    # Test 3 & 4: Counts preserved
    assert loaded_G.number_of_nodes() == G.number_of_nodes()
    assert loaded_G.number_of_edges() == G.number_of_edges()
    
    # Test 5: Node IDs preserved (GraphML parses IDs as strings if originally int, unless configured, but networkx usually handles it. Wait, OSMnx load_graphml preserves int IDs!)
    assert set(loaded_G.nodes()) == set(G.nodes()) or set(loaded_G.nodes()) == {str(n) for n in G.nodes()}
    
    # Let's check node mapping
    original_nodes = {str(n): n for n in G.nodes()}
    loaded_nodes = list(loaded_G.nodes())
    # Assuming IDs are preserved as string or int, let's normalize to string for comparison
    assert {str(n) for n in loaded_nodes} == set(original_nodes.keys())
    
    # Test 6: Node coordinates preserved
    for n in loaded_G.nodes():
        orig_n = int(n) if str(n).isdigit() else n # Handle potential string parsing
        assert math.isclose(loaded_G.nodes[n]['x'], G.nodes[orig_n]['x'], rel_tol=1e-5)
        assert math.isclose(loaded_G.nodes[n]['y'], G.nodes[orig_n]['y'], rel_tol=1e-5)
        
    # Test 7, 8, 9, 10, 11: Edge properties, parallel edges, directionality, attributes, geometry
    # Check parallel edges between 1 and 2
    u, v = (1, 2) if 1 in loaded_G else ('1', '2')
    assert loaded_G.number_of_edges(u, v) == 2
    
    edges_uv = loaded_G.get_edge_data(u, v)
    # Check keys (GraphML might parse keys as strings or ints)
    keys = list(edges_uv.keys())
    
    # One of the parallel edges should be length 15.0 and residential
    edge_15 = [e for k, e in edges_uv.items() if math.isclose(float(e['length']), 15.0)][0]
    assert edge_15['highway'] == 'residential'
    # Check oneway parsing (might be string 'True' or boolean True)
    assert str(edge_15.get('oneway', '')).lower() == 'true'
    
    # Test directionality (2 -> 3 exists, 3 -> 2 does not)
    u2, v3 = (2, 3) if 2 in loaded_G else ('2', '3')
    assert loaded_G.has_edge(u2, v3)
    assert not loaded_G.has_edge(v3, u2)
    
    # Test geometry
    edge_23 = loaded_G.get_edge_data(u2, v3)[0] if 0 in loaded_G.get_edge_data(u2, v3) else loaded_G.get_edge_data(u2, v3)['0']
    geom = edge_23['geometry']
    geom_str = str(geom) if not isinstance(geom, str) else geom
    assert geom_str == "LINESTRING (80.2 13.2, 80.3 13.3)" or geom_str == "LINESTRING(80.2 13.2, 80.3 13.3)"

def test_missing_file(tmp_path):
    # Test 12
    file_path = tmp_path / "does_not_exist.graphml"
    with pytest.raises(FileNotFoundError):
        load_road_network(file_path)

def test_corrupt_file(tmp_path):
    # Test 13
    file_path = tmp_path / "corrupt.graphml"
    file_path.write_text("<xml>This is not a valid graphml</xml>")
    with pytest.raises(RuntimeError, match="Failed to load road network"):
        load_road_network(file_path)

def test_invalid_graph_cannot_be_saved(tmp_path):
    # Test 14
    G = nx.MultiDiGraph() # Empty graph is invalid
    file_path = tmp_path / "invalid.graphml"
    
    # Catch Exception since import paths might differ, but check the name
    with pytest.raises(Exception) as excinfo:
        save_road_network(G, file_path)
    assert "RoadNetworkValidationError" in type(excinfo.value).__name__
    
    # Ensure it wasn't saved
    assert not file_path.exists()

def test_loaded_invalid_graph_is_rejected(tmp_path):
    # Test 15
    # Create a graph that passes OSMnx save (so we bypass validation during save manually)
    G = nx.MultiDiGraph()
    G.add_node(1, x=80.1, y=13.1)
    G.add_node(2, x=200.0, y=13.2) # Invalid longitude
    G.add_edge(1, 2, key=0, length=10.0)
    
    file_path = tmp_path / "invalid_loaded.graphml"
    import osmnx as ox
    ox.save_graphml(G, file_path) # Force save it
    
    # Loading it should trigger validation and fail
    with pytest.raises(Exception) as excinfo:
        load_road_network(file_path)
    assert "RoadNetworkValidationError" in type(excinfo.value).__name__

def test_graph_is_not_mutated_by_saving(tmp_path):
    # Test 16
    G = create_valid_graph()
    before_nodes = G.number_of_nodes()
    before_edges = G.number_of_edges()
    
    file_path = tmp_path / "test_mutation.graphml"
    save_road_network(G, file_path)
    
    assert G.number_of_nodes() == before_nodes
    assert G.number_of_edges() == before_edges

def test_overwrite_behavior(tmp_path):
    G = create_valid_graph()
    file_path = tmp_path / "test_overwrite.graphml"
    
    # Save once
    save_road_network(G, file_path)
    
    # Save again without overwrite should fail
    with pytest.raises(FileExistsError):
        save_road_network(G, file_path)
        
    # Save again with overwrite should succeed
    save_road_network(G, file_path, overwrite=True)
