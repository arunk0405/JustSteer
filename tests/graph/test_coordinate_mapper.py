"""Offline tests for the coordinate-to-nearest-node mapping boundary."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import networkx as nx
import pytest

from src.graph.coordinate_mapper import (
    CoordinateValidationError,
    NodeMappingError,
    find_nearest_node,
)
from src.graph.graph_validator import RoadNetworkValidationError


def create_test_graph() -> nx.MultiDiGraph:
    """Create a small WGS84 graph whose nearest nodes are unambiguous."""
    graph = nx.MultiDiGraph(crs="epsg:4326")
    graph.add_node(1, x=80.0000, y=13.0000)
    graph.add_node(2, x=80.0200, y=13.0000)
    graph.add_node(3, x=80.0000, y=13.0200)
    graph.add_edge(1, 2, key=0, length=20.0)
    graph.add_edge(2, 3, key=0, length=25.0)
    return graph


def test_exact_coordinate_match_returns_matching_node_and_zero_distance() -> None:
    result = find_nearest_node(create_test_graph(), latitude=13.0, longitude=80.0)

    assert result.node_id == 1
    assert result.matched_latitude == 13.0
    assert result.matched_longitude == 80.0
    assert result.snap_distance_m == pytest.approx(0.0, abs=1e-6)


def test_nearby_coordinate_selects_expected_nearest_node_and_distance() -> None:
    result = find_nearest_node(create_test_graph(), latitude=13.0001, longitude=80.0001)

    assert result.node_id == 1
    assert result.snap_distance_m == pytest.approx(15.5, abs=1.5)


def test_multiple_candidate_nodes_choose_the_unambiguous_nearest_node() -> None:
    result = find_nearest_node(create_test_graph(), latitude=13.0185, longitude=80.0002)

    assert result.node_id == 3


def test_public_latitude_longitude_order_maps_to_osm_y_x() -> None:
    graph = nx.MultiDiGraph(crs="epsg:4326")
    graph.add_node(10, x=80.0, y=13.0)
    graph.add_node(20, x=13.0, y=80.0)
    graph.add_edge(10, 20, key=0, length=1.0)

    result = find_nearest_node(graph, latitude=13.0, longitude=80.0)

    assert result.node_id == 10
    assert (result.matched_latitude, result.matched_longitude) == (13.0, 80.0)


@pytest.mark.parametrize("latitude", [90.1, -90.1, None, "invalid", [], {}, True])
def test_invalid_latitude_is_rejected(latitude: object) -> None:
    with pytest.raises(CoordinateValidationError):
        find_nearest_node(create_test_graph(), latitude=latitude, longitude=80.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("longitude", [180.1, -180.1, None, "invalid", [], {}, False])
def test_invalid_longitude_is_rejected(longitude: object) -> None:
    with pytest.raises(CoordinateValidationError):
        find_nearest_node(create_test_graph(), latitude=13.0, longitude=longitude)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_coordinates_are_rejected(value: float) -> None:
    graph = create_test_graph()
    with pytest.raises(CoordinateValidationError):
        find_nearest_node(graph, latitude=value, longitude=80.0)
    with pytest.raises(CoordinateValidationError):
        find_nearest_node(graph, latitude=13.0, longitude=value)


def test_empty_graph_is_rejected() -> None:
    with pytest.raises(RoadNetworkValidationError, match="contains no nodes"):
        find_nearest_node(nx.MultiDiGraph(crs="epsg:4326"), latitude=13.0, longitude=80.0)


@pytest.mark.parametrize("graph", [nx.Graph(), nx.DiGraph(), nx.MultiGraph()])
def test_unsuitable_graph_types_are_rejected(graph: nx.Graph) -> None:
    with pytest.raises(RoadNetworkValidationError, match="MultiDiGraph"):
        find_nearest_node(graph, latitude=13.0, longitude=80.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("missing_attribute", ["x", "y"])
def test_missing_node_coordinates_are_rejected(missing_attribute: str) -> None:
    graph = nx.MultiDiGraph(crs="epsg:4326")
    node_data = {"x": 80.0, "y": 13.0}
    del node_data[missing_attribute]
    graph.add_node(1, **node_data)

    with pytest.raises(RoadNetworkValidationError, match="coordinate attributes"):
        find_nearest_node(graph, latitude=13.0, longitude=80.0)


def test_graph_without_crs_is_rejected_without_being_repaired() -> None:
    graph = create_test_graph()
    del graph.graph["crs"]

    with pytest.raises(RoadNetworkValidationError, match="missing graph CRS"):
        find_nearest_node(graph, latitude=13.0, longitude=80.0)
    assert "crs" not in graph.graph


def test_distant_coordinate_returns_nearest_node_and_large_distance_by_default() -> None:
    result = find_nearest_node(create_test_graph(), latitude=40.7128, longitude=-74.0060)

    assert result.node_id in {1, 2, 3}
    assert result.snap_distance_m > 1_000_000


def test_optional_maximum_snap_distance_is_caller_controlled() -> None:
    graph = create_test_graph()
    result = find_nearest_node(graph, latitude=13.0001, longitude=80.0001, max_snap_distance_m=100.0)
    assert result.node_id == 1

    with pytest.raises(NodeMappingError, match="exceeds maximum allowed snap distance"):
        find_nearest_node(graph, latitude=13.0001, longitude=80.0001, max_snap_distance_m=1.0)


@pytest.mark.parametrize("limit", [-1, float("nan"), float("inf"), "100", True])
def test_invalid_maximum_snap_distance_is_rejected(limit: object) -> None:
    with pytest.raises(CoordinateValidationError):
        find_nearest_node(create_test_graph(), 13.0, 80.0, max_snap_distance_m=limit)  # type: ignore[arg-type]


def test_result_is_consistent_with_the_matched_graph_node() -> None:
    graph = create_test_graph()
    result = find_nearest_node(graph, latitude=13.0001, longitude=80.0001)

    assert result.node_id in graph
    assert result.matched_latitude == graph.nodes[result.node_id]["y"]
    assert result.matched_longitude == graph.nodes[result.node_id]["x"]
    assert math.isfinite(result.snap_distance_m)
    assert result.snap_distance_m >= 0


def test_mapping_is_strictly_read_only_including_graph_metadata_and_attributes() -> None:
    graph = create_test_graph()
    before_graph = copy.deepcopy(graph.graph)
    before_nodes = {node: copy.deepcopy(data) for node, data in graph.nodes(data=True)}
    before_edges = {
        (source, target, key): copy.deepcopy(data)
        for source, target, key, data in graph.edges(keys=True, data=True)
    }

    find_nearest_node(graph, latitude=13.0001, longitude=80.0001)

    assert graph.graph == before_graph
    assert {node: data for node, data in graph.nodes(data=True)} == before_nodes
    assert {
        (source, target, key): data
        for source, target, key, data in graph.edges(keys=True, data=True)
    } == before_edges


def test_parallel_edges_are_preserved() -> None:
    graph = create_test_graph()
    graph.add_edge(1, 2, key=1, length=21.0)

    find_nearest_node(graph, latitude=13.0001, longitude=80.0001)

    assert graph.number_of_edges(1, 2) == 2
    assert set(graph[1][2]) == {0, 1}


def test_mapper_has_no_api_routing_or_graph_lifecycle_dependencies() -> None:
    mapper_path = Path(__file__).resolve().parents[2] / "src/graph/coordinate_mapper.py"
    source = mapper_path.read_text(encoding="utf-8").lower()
    forbidden_references = (
        "fastapi",
        "pydantic",
        "flask",
        "django",
        "load_road_network",
        "save_road_network",
        "download_road_network",
        "astar",
        "traffic",
        "safety",
    )
    assert not any(reference in source for reference in forbidden_references)
