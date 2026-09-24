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
        ("A", "B"): 1.0,
        ("A", "C"): 10.0,
        ("A", "D"): 10.0,
        ("B", "C"): 1.0,
        ("B", "D"): 10.0,
        ("C", "D"): 1.0,
        ("C", "B"): 10.0,
        ("D", "B"): 10.0,
        ("A", "A"): 0.0,
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
    result = router.route("A", ("C", "B", "D"))
    
    assert result.success is True
    assert result.delivery_sequence.sequence == ("B", "C", "D")
    
    assert len(result.routes) == 3
    assert result.routes[0].source_node == "A"
    assert result.routes[0].destination_node == "B"
    assert result.routes[1].source_node == "B"
    assert result.routes[1].destination_node == "C"
    assert result.routes[2].source_node == "C"
    assert result.routes[2].destination_node == "D"
    
    assert pytest.approx(result.total_cost) == 3.0
    assert pytest.approx(result.total_distance_m) == 300.0


def test_empty_delivery_list(router):
    # 2. Empty delivery list
    result = router.route("A", ())
    
    assert result.success is True
    assert len(result.delivery_sequence.sequence) == 0
    assert len(result.routes) == 0
    assert result.total_cost == 0.0
    assert result.total_distance_m == 0.0


def test_one_delivery(router):
    # 3. One delivery
    result = router.route("A", ("C",))
    
    assert result.success is True
    assert result.delivery_sequence.sequence == ("C",)
    assert len(result.routes) == 1
    assert result.routes[0].destination_node == "C"


def test_delivery_at_current_node(router):
    # 5. Delivery at current node
    result = router.route("A", ("A", "B"))
    
    assert result.success is True
    assert result.delivery_sequence.sequence == ("A", "B")
    assert len(result.routes) == 2
    assert result.routes[0].source_node == "A"
    assert result.routes[0].destination_node == "A"
    assert result.routes[1].source_node == "A"
    assert result.routes[1].destination_node == "B"


def test_optimizer_failure():
    # 6. Optimizer failure
    service = FakeRoutingService({("A", "B"): 1.0}) # Missing B->C
    builder = PairwiseRouteMatrixBuilder(service)
    optimizer = BruteForceSequenceOptimizer(builder)
    executor = MultiStopRouteExecutor(service)
    router = MultiStopRouter(optimizer, executor)
    
    with pytest.raises(RouteNotFoundError):
        router.route("A", ("B", "C"))


def test_route_execution_failure():
    # 7. Route execution failure
    # If the optimizer succeeds but execution fails...
    # In this architecture, they both use the same PairwiseRoutingService instance
    # so they will fail identically. We can simulate it by injecting an executor
    # that always fails.
    
    class FailingExecutor:
        def execute(self, start_node, sequence):
            raise RouteNotFoundError("Execution failed unexpectedly")
            
    service = FakeRoutingService({("A", "B"): 1.0})
    builder = PairwiseRouteMatrixBuilder(service)
    optimizer = BruteForceSequenceOptimizer(builder)
    failing_executor = FailingExecutor()
    router = MultiStopRouter(optimizer, failing_executor)
    
    with pytest.raises(RouteNotFoundError, match="Execution failed unexpectedly"):
        router.route("A", ("B",))


def test_individual_routeresult_preservation(router):
    # 12. Individual RouteResult preservation
    result = router.route("A", ("B",))
    
    assert len(result.routes) == 1
    assert isinstance(result.routes[0], RouteResult)
    assert result.routes[0].source_node == "A"
    assert result.routes[0].destination_node == "B"


def test_input_immutability(router):
    # 13. Input immutability
    deliveries = ("C", "B", "D")
    router.route("A", deliveries)
    assert deliveries == ("C", "B", "D")


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
    for node in ["A", "B", "C", "D", "E"]:
        G.add_node(node, x=0.0, y=0.0)
        
    # Desired optimal route: A -> C -> E -> B -> D
    # We assign low costs to these edges
    G.add_edge("A", "C", final_cost=0.1, length=100.0)
    G.add_edge("C", "E", final_cost=0.1, length=100.0)
    G.add_edge("E", "B", final_cost=0.1, length=100.0)
    G.add_edge("B", "D", final_cost=0.1, length=100.0)
    
    # Add decoy edges with high costs
    G.add_edge("A", "B", final_cost=10.0, length=100.0)
    G.add_edge("C", "D", final_cost=10.0, length=100.0)
    G.add_edge("B", "C", final_cost=10.0, length=100.0)
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
    result = router.route("A", ("B", "C", "D", "E"))
    
    assert result.success is True
    assert result.delivery_sequence.sequence == ("C", "E", "B", "D")
    
    assert len(result.routes) == 4
    assert result.routes[0].source_node == "A"
    assert result.routes[0].destination_node == "C"
    assert result.routes[1].source_node == "C"
    assert result.routes[1].destination_node == "E"
    assert result.routes[2].source_node == "E"
    assert result.routes[2].destination_node == "B"
    assert result.routes[3].source_node == "B"
    assert result.routes[3].destination_node == "D"
    
    assert pytest.approx(result.total_cost) == 0.4
    assert pytest.approx(result.total_distance_m) == 400.0
