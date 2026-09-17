"""Offline integration tests for coordinate snapping followed by A* routing."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import networkx as nx
import pytest

from src.graph.coordinate_mapper import CoordinateValidationError
from src.graph.graph_validator import RoadNetworkValidationError
from src.routing.astar import AStarRoutingEngine
from src.routing.coordinate_to_route import CoordinateRouteResult, route_coordinates
from src.routing.exceptions import RouteNotFoundError


def create_routing_graph() -> nx.MultiDiGraph:
    """Create a deterministic WGS84 graph with directed and parallel edges."""
    graph = nx.MultiDiGraph(crs="epsg:4326")
    graph.add_node(100, x=80.0000, y=13.0000)
    graph.add_node(200, x=80.0100, y=13.0000)
    graph.add_node(300, x=80.0200, y=13.0000)
    graph.add_node(400, x=80.0200, y=13.0200)

    # Test-only deterministic routing costs; no production graph is modified.
    graph.add_edge(100, 200, key=0, length=100.0, final_cost=5.0)
    graph.add_edge(100, 200, key=1, length=110.0, final_cost=1.0)
    graph.add_edge(200, 300, key=0, length=200.0, final_cost=2.0)
    return graph


def route(
    graph: nx.MultiDiGraph,
    origin: tuple[float, float],
    destination: tuple[float, float],
):
    """Invoke the integration boundary with the existing A* engine."""
    return route_coordinates(
        graph,
        origin_latitude=origin[0],
        origin_longitude=origin[1],
        destination_latitude=destination[0],
        destination_longitude=destination[1],
        router=AStarRoutingEngine(graph, cost_attribute="final_cost"),
    )


def test_basic_coordinate_to_route_maps_nodes_and_returns_existing_route_result() -> None:
    result = route(create_routing_graph(), (13.0001, 80.0001), (13.0001, 80.0199))

    assert isinstance(result, CoordinateRouteResult)
    assert result.origin_match.node_id == 100
    assert result.destination_match.node_id == 300
    assert result.route.path_nodes == [100, 200, 300]
    assert result.route.path_edges == [(100, 200, 1), (200, 300, 0)]


def test_exact_coordinates_preserve_zero_snap_distances() -> None:
    result = route(create_routing_graph(), (13.0, 80.0), (13.0, 80.02))

    assert result.origin_match.node_id == 100
    assert result.destination_match.node_id == 300
    assert result.origin_match.snap_distance_m == pytest.approx(0.0, abs=1e-6)
    assert result.destination_match.snap_distance_m == pytest.approx(0.0, abs=1e-6)


def test_nearby_coordinates_select_expected_nodes_and_preserve_snap_distances() -> None:
    result = route(create_routing_graph(), (13.0001, 80.0001), (13.0001, 80.0199))

    assert result.origin_match.node_id == 100
    assert result.destination_match.node_id == 300
    assert 0 < result.origin_match.snap_distance_m < 100
    assert 0 < result.destination_match.snap_distance_m < 100


def test_coordinate_order_is_latitude_then_longitude() -> None:
    graph = create_routing_graph()
    graph.add_node(500, x=13.0, y=80.0)
    graph.add_edge(500, 300, key=0, length=1.0, final_cost=1.0)

    result = route(graph, (13.0, 80.0), (13.0, 80.02))

    assert result.origin_match.node_id == 100


def test_one_way_routing_respects_existing_astar_behavior() -> None:
    with pytest.raises(RouteNotFoundError):
        route(create_routing_graph(), (13.0, 80.02), (13.0, 80.0))


def test_parallel_edges_are_selected_by_existing_astar_engine() -> None:
    result = route(create_routing_graph(), (13.0, 80.0), (13.0, 80.01))

    assert result.route.path_edges == [(100, 200, 1)]
    assert result.route.total_cost == 1.0


def test_unreachable_destination_propagates_existing_routing_exception() -> None:
    with pytest.raises(RouteNotFoundError):
        route(create_routing_graph(), (13.0, 80.0), (13.02, 80.02))


def test_same_snapped_node_uses_existing_astar_same_node_contract() -> None:
    result = route(create_routing_graph(), (13.00001, 80.00001), (13.00002, 80.00002))

    assert result.origin_match.node_id == result.destination_match.node_id == 100
    assert result.route.path_nodes == [100]
    assert result.route.path_edges == []
    assert result.route.total_cost == 0.0


@pytest.mark.parametrize(
    ("origin_latitude", "origin_longitude", "destination_latitude", "destination_longitude"),
    [
        (90.1, 80.0, 13.0, 80.02),
        (13.0, 180.1, 13.0, 80.02),
        (13.0, 80.0, None, 80.02),
        (13.0, 80.0, 13.0, "invalid"),
        (float("nan"), 80.0, 13.0, 80.02),
        (13.0, 80.0, float("inf"), 80.02),
    ],
)
def test_invalid_coordinates_are_not_bypassed(
    origin_latitude: object,
    origin_longitude: object,
    destination_latitude: object,
    destination_longitude: object,
) -> None:
    graph = create_routing_graph()
    with pytest.raises(CoordinateValidationError):
        route_coordinates(
            graph,
            origin_latitude=origin_latitude,  # type: ignore[arg-type]
            origin_longitude=origin_longitude,  # type: ignore[arg-type]
            destination_latitude=destination_latitude,  # type: ignore[arg-type]
            destination_longitude=destination_longitude,  # type: ignore[arg-type]
            router=AStarRoutingEngine(graph),
        )


def test_empty_graph_returns_coordinate_mapper_domain_error() -> None:
    graph = nx.MultiDiGraph(crs="epsg:4326")
    with pytest.raises(RoadNetworkValidationError, match="contains no nodes"):
        route_coordinates(graph, 13.0, 80.0, 13.1, 80.1, router=AStarRoutingEngine(graph))


def test_graph_is_read_only_during_mapping_and_routing() -> None:
    graph = create_routing_graph()
    before_graph = copy.deepcopy(graph.graph)
    before_nodes = {node: copy.deepcopy(data) for node, data in graph.nodes(data=True)}
    before_edges = {
        (source, target, key): copy.deepcopy(data)
        for source, target, key, data in graph.edges(keys=True, data=True)
    }

    route(graph, (13.0, 80.0), (13.0, 80.02))

    assert graph.graph == before_graph
    assert {node: data for node, data in graph.nodes(data=True)} == before_nodes
    assert {
        (source, target, key): data
        for source, target, key, data in graph.edges(keys=True, data=True)
    } == before_edges


def test_route_cost_and_geometry_are_from_the_existing_astar_result() -> None:
    graph = create_routing_graph()
    result = route(graph, (13.0, 80.0), (13.0, 80.02))

    selected_cost = sum(graph.edges[u, v, key]["final_cost"] for u, v, key in result.route.path_edges)
    assert result.route.total_cost == selected_cost == 3.0
    assert result.route.geometry == [(80.0, 13.0), (80.01, 13.0), (80.02, 13.0)]


def test_router_and_graph_must_be_injected_as_a_matching_pair() -> None:
    graph = create_routing_graph()
    other_graph = create_routing_graph()

    with pytest.raises(ValueError, match="same graph instance"):
        route_coordinates(
            graph,
            13.0,
            80.0,
            13.0,
            80.02,
            router=AStarRoutingEngine(other_graph),
        )


def test_integration_module_has_no_application_or_external_dependencies() -> None:
    module_path = Path(__file__).resolve().parents[2] / "src/routing/coordinate_to_route.py"
    source = module_path.read_text(encoding="utf-8").lower()
    forbidden_references = (
        "fastapi",
        "pydantic",
        "flask",
        "django",
        "requests",
        "tomtom",
        "openweather",
        "load_road_network",
        "save_road_network",
        "download_road_network",
    )
    assert not any(reference in source for reference in forbidden_references)
