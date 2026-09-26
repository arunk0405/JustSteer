import pytest
import networkx as nx

from src.routing.delivery_sequence import DeliverySequenceRequest
from src.routing.route_result import RouteResult
from src.routing.exceptions import RouteNotFoundError
from src.routing.pairwise_route_matrix import PairwiseRouteMatrixBuilder
from src.routing.sequence_optimizer import BruteForceSequenceOptimizer
from src.routing.multi_stop_routing import MultiStopRouteExecutor, MultiStopRouteResult
from src.routing.multi_stop_router import MultiStopRouter
from src.routing.astar import AStarRoutingEngine
from src.routing.routing_service import PairwiseRoutingService


# ---------------------------------------------------------
# Fakes for isolated testing
# ---------------------------------------------------------

class FakeRoutingService:
    """A fake pairwise routing service with deterministic costs and fail points."""
    def __init__(self, costs, distance_m=100.0):
        self.costs = costs
        self.distance_m = distance_m
        
    def route(self, source, destination) -> RouteResult:
        if (source, destination) not in self.costs:
            raise RouteNotFoundError(f"No route {source}->{destination}")
            
        cost = self.costs[(source, destination)]
        return RouteResult(
            success=True,
            source_node=source,
            destination_node=destination,
            total_cost=cost,
            distance_m=self.distance_m
        )


@pytest.fixture
def fake_costs():
    # Costs designed so that A -> B -> C -> D is the optimal path
    return {
        (1, 2): 1.0,
        (1, 3): 10.0,
        (1, 4): 10.0,
        (2, 3): 1.0,
        (2, 4): 10.0,
        (3, 4): 1.0,
        (3, 2): 10.0,
        (4, 2): 10.0,
        (1, 1): 0.0,
    }


@pytest.fixture
def router(fake_costs):
    service = FakeRoutingService(fake_costs)
    builder = PairwiseRouteMatrixBuilder(service)
    optimizer = BruteForceSequenceOptimizer(builder)
    executor = MultiStopRouteExecutor(service)
    return MultiStopRouter(optimizer, executor)


# ---------------------------------------------------------
# Required Tests 1-14
# ---------------------------------------------------------

def test_full_successful_flow(router):
    # 1. Full successful flow
    # 4. Multiple deliveries
    # 8. Total cost aggregation
    # 9. Total distance aggregation
    # 10. Sequence preservation
    # 11. Route ordering
    
    # Even if requested in a weird order (C, B, D), optimizer should pick B, C, D
    # since A->B=1, B->C=1, C->D=1 is the cheapest (cost 3.0)
    result = router.route(1, (3, 2, 4))
    
    assert result.success is True
    assert result.delivery_sequence.sequence == (2, 3, 4)
    
    assert len(result.routes) == 3
    assert result.routes[0].source_node == 1
    assert result.routes[0].destination_node == 2
    assert result.routes[1].source_node == 2
    assert result.routes[1].destination_node == 3
    assert result.routes[2].source_node == 3
    assert result.routes[2].destination_node == 4
    
    assert pytest.approx(result.total_cost) == 3.0
    assert pytest.approx(result.total_distance_m) == 300.0


def test_empty_delivery_list(router):
    # 2. Empty delivery list
    result = router.route(1, ())
    
    assert result.success is True
    assert len(result.delivery_sequence.sequence) == 0
    assert len(result.routes) == 0
    assert result.total_cost == 0.0
    assert result.total_distance_m == 0.0


def test_one_delivery(router):
    # 3. One delivery
    result = router.route(1, (3,))
    
    assert result.success is True
    assert result.delivery_sequence.sequence == (3,)
    assert len(result.routes) == 1
    assert result.routes[0].destination_node == 3


def test_delivery_at_current_node(router):
    # 5. Delivery at current node
    result = router.route(1, (1, 2))
    
    assert result.success is True
    assert result.delivery_sequence.sequence == (1, 2)
    assert len(result.routes) == 2
    assert result.routes[0].source_node == 1
    assert result.routes[0].destination_node == 1
    assert result.routes[1].source_node == 1
    assert result.routes[1].destination_node == 2


def test_optimizer_failure():
    # 6. Optimizer failure
    service = FakeRoutingService({(1, 2): 1.0}) # Missing B->C
    builder = PairwiseRouteMatrixBuilder(service)
    optimizer = BruteForceSequenceOptimizer(builder)
    executor = MultiStopRouteExecutor(service)
    router = MultiStopRouter(optimizer, executor)
    
    with pytest.raises(RouteNotFoundError):
        router.route(1, (2, 3))


def test_route_execution_failure():
    # 7. Route execution failure
    # If the optimizer succeeds but execution fails...
    # In this architecture, they both use the same PairwiseRoutingService instance
    # so they will fail identically. We can simulate it by injecting an executor
    # that always fails.
    
    class FailingExecutor:
        def execute(self, start_node, sequence):
            raise RouteNotFoundError("Execution failed unexpectedly")
            
    service = FakeRoutingService({(1, 2): 1.0})
    builder = PairwiseRouteMatrixBuilder(service)
    optimizer = BruteForceSequenceOptimizer(builder)
    failing_executor = FailingExecutor()
    router = MultiStopRouter(optimizer, failing_executor)
    
    with pytest.raises(RouteNotFoundError, match="Execution failed unexpectedly"):
        router.route(1, (2,))


def test_individual_routeresult_preservation(router):
    # 12. Individual RouteResult preservation
    result = router.route(1, (2,))
    
    assert len(result.routes) == 1
    assert isinstance(result.routes[0], RouteResult)
    assert result.routes[0].source_node == 1
    assert result.routes[0].destination_node == 2


def test_input_immutability(router):
    # 13. Input immutability
    deliveries = (3, 2, 4)
    router.route(1, deliveries)
    assert deliveries == (3, 2, 4)


def test_dependency_injection(router):
    # 14. Dependency injection
    assert isinstance(router.sequence_optimizer, BruteForceSequenceOptimizer)
    assert isinstance(router.route_executor, MultiStopRouteExecutor)


# ---------------------------------------------------------
# 15. Synthetic real A* integration
# ---------------------------------------------------------

def test_synthetic_end_to_end_routing():
    # 58. Final Verification Scenario
    
    # Create a synthetic graph
    G = nx.MultiDiGraph()
    for node in [1, 2, 3, 4, 5]:
        G.add_node(node, x=0.0, y=0.0)
        
    # Desired optimal route: A -> C -> E -> B -> D
    # We assign low costs to these edges
    G.add_edge(1, 3, final_cost=0.1, length=100.0)
    G.add_edge(3, 5, final_cost=0.1, length=100.0)
    G.add_edge(5, 2, final_cost=0.1, length=100.0)
    G.add_edge(2, 4, final_cost=0.1, length=100.0)
    
    # Add decoy edges with high costs
    G.add_edge(1, 2, final_cost=10.0, length=100.0)
    G.add_edge(3, 4, final_cost=10.0, length=100.0)
    G.add_edge(2, 3, final_cost=10.0, length=100.0)
    # Must add full connectivity for permutation to evaluate completely without crashing prematurely,
    # or it will just raise RouteNotFoundError on unroutable permutations.
    # Actually, our BruteForceSequenceOptimizer catches `route is None` and gracefully marks
    # that permutation invalid without crashing, so partial connectivity is fine!
    
    astar = AStarRoutingEngine(G, cost_attribute="final_cost")
    service = PairwiseRoutingService(astar)
    builder = PairwiseRouteMatrixBuilder(service)
    optimizer = BruteForceSequenceOptimizer(builder)
    executor = MultiStopRouteExecutor(service)
    
    router = MultiStopRouter(
        sequence_optimizer=optimizer,
        route_executor=executor
    )
    
    # Request out-of-order deliveries
    result = router.route(1, (2, 3, 4, 5))
    
    assert result.success is True
    assert result.delivery_sequence.sequence == (3, 5, 2, 4)
    
    assert len(result.routes) == 4
    assert result.routes[0].source_node == 1
    assert result.routes[0].destination_node == 3
    assert result.routes[1].source_node == 3
    assert result.routes[1].destination_node == 5
    assert result.routes[2].source_node == 5
    assert result.routes[2].destination_node == 2
    assert result.routes[3].source_node == 2
    assert result.routes[3].destination_node == 4
    
    assert pytest.approx(result.total_cost) == 0.4
    assert pytest.approx(result.total_distance_m) == 400.0
