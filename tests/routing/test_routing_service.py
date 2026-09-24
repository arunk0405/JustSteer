import pytest
import networkx as nx

from src.routing.astar import AStarRoutingEngine
from src.routing.routing_service import PairwiseRoutingService
from src.routing.exceptions import InvalidNodeError, RouteNotFoundError
from src.routing.route_result import RouteResult

def create_synthetic_graph() -> nx.MultiDiGraph:
    r"""Create a basic synthetic graph for tests.
    A -> B -> C
     \        ^
      v       |
        -> D -
    """
    G = nx.MultiDiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=1.0, y=0.0)
    G.add_node("C", x=2.0, y=0.0)
    G.add_node("D", x=1.0, y=-1.0)
    
    # A -> B -> C cost = 10 + 10 = 20
    G.add_edge("A", "B", key=0, final_cost=10.0, length=100.0, geometry=[(0.0, 0.0), (1.0, 0.0)])
    G.add_edge("B", "C", key=0, final_cost=10.0, length=100.0, geometry=[(1.0, 0.0), (2.0, 0.0)])
    
    # A -> D -> C cost = 5 + 5 = 10
    G.add_edge("A", "D", key=0, final_cost=5.0, length=100.0, geometry=[(0.0, 0.0), (1.0, -1.0)])
    G.add_edge("D", "C", key=0, final_cost=5.0, length=100.0, geometry=[(1.0, -1.0), (2.0, 0.0)])
    
    return G

@pytest.fixture
def basic_graph():
    return create_synthetic_graph()

@pytest.fixture
def astar_engine(basic_graph):
    return AStarRoutingEngine(basic_graph)

@pytest.fixture
def routing_service(astar_engine):
    return PairwiseRoutingService(astar_engine)

def test_basic_pairwise_route(routing_service):
    # Test 1 - Basic pairwise route (A -> C should prefer A->D->C due to cost 10 vs 20)
    result = routing_service.route("A", "C")
    assert result.path_nodes == ["A", "D", "C"]
    assert result.success is True

def test_uses_astar(basic_graph, monkeypatch):
    # Test 2 - Uses A*
    engine = AStarRoutingEngine(basic_graph)
    
    called = []
    original_route = engine.route
    
    def mocked_route(origin, destination):
        called.append((origin, destination))
        return original_route(origin, destination)
        
    monkeypatch.setattr(engine, "route", mocked_route)
    
    service = PairwiseRoutingService(engine)
    service.route("A", "B")
    
    assert called == [("A", "B")]

def test_returns_exact_routeresult(routing_service):
    # Test 3 - Returns Exact RouteResult
    result = routing_service.route("A", "B")
    assert isinstance(result, RouteResult)

def test_source_equals_destination(routing_service):
    # Test 4 - Source Equals Destination
    result = routing_service.route("A", "A")
    assert result.success is True
    assert result.path_nodes == ["A"]
    assert result.total_cost == 0.0

def test_unreachable_destination(basic_graph, routing_service):
    # Test 5 - Unreachable Destination
    basic_graph.add_node("Isolated")
    with pytest.raises(RouteNotFoundError):
        routing_service.route("A", "Isolated")

def test_invalid_origin_node(routing_service):
    # Test 6 - Invalid Origin Node
    with pytest.raises(InvalidNodeError):
        routing_service.route("NonExistent", "B")

def test_invalid_destination_node(routing_service):
    # Test 7 - Invalid Destination Node
    with pytest.raises(InvalidNodeError):
        routing_service.route("A", "NonExistent")

def test_one_way_graph(routing_service):
    # Test 8 - One-Way Graph
    # A -> B works, B -> A fails
    res = routing_service.route("A", "B")
    assert res.success is True
    
    with pytest.raises(RouteNotFoundError):
        routing_service.route("B", "A")

def test_multidigraph_parallel_edges(basic_graph, routing_service):
    # Test 9 - MultiDiGraph Parallel Edges
    # Add a parallel edge from A -> B with lower cost
    basic_graph.add_edge("A", "B", key=1, final_cost=1.0, length=50.0)
    result = routing_service.route("A", "B")
    assert result.total_cost == 1.0
    assert result.path_edges[0] == ("A", "B", 1)

def test_graph_immutability(basic_graph, routing_service):
    # Test 10 - Graph Immutability
    before_nodes = set(basic_graph.nodes)
    before_edges = list(basic_graph.edges(data=True, keys=True))
    
    routing_service.route("A", "C")
    
    assert set(basic_graph.nodes) == before_nodes
    assert list(basic_graph.edges(data=True, keys=True)) == before_edges

def test_different_edge_costs(routing_service):
    # Test 11 - Different Edge Costs
    # A->D->C cost is 10, A->B->C is 20
    result = routing_service.route("A", "C")
    assert result.total_cost == 10.0
    assert result.path_nodes == ["A", "D", "C"]

def test_distance_and_cost_preservation(routing_service):
    # Test 12 - Distance and Cost Preservation
    result = routing_service.route("A", "B")
    assert result.total_cost == 10.0
    assert result.distance_m == 100.0

def test_geometry_preservation(routing_service):
    # Test 13 - Geometry Preservation
    result = routing_service.route("A", "B")
    assert result.geometry == [(0.0, 0.0), (1.0, 0.0)]

def test_dependency_graph_miswiring():
    # Test 14 - Dependency/Graph Miswiring
    # A router configured for one graph shouldn't be overridden. 
    # Since PairwiseRoutingService only takes node IDs, graph injection miswiring 
    # at the route call is structurally impossible. We just verify the router's graph is used.
    G = create_synthetic_graph()
    engine = AStarRoutingEngine(G)
    service = PairwiseRoutingService(engine)
    assert service.astar_engine.graph is G

def test_service_independence():
    # Test 15 - Service Independence
    # We successfully instantiated and used PairwiseRoutingService without OSM, APIs, FastAPI, Database.
    G = nx.MultiDiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_edge(1, 2, key=0, final_cost=5.0)
    
    engine = AStarRoutingEngine(G)
    service = PairwiseRoutingService(engine)
    res = service.route(1, 2)
    
    assert res.success is True
    assert res.path_nodes == [1, 2]
    assert res.total_cost == 5.0
