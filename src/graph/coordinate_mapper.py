"""Map geographic coordinates to nodes in an already-loaded road graph.

This module is deliberately limited to coordinate-to-node mapping. It does
not manage a graph's lifecycle, perform routing, or depend on app frameworks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
import networkx as nx
import osmnx as ox

from .graph_validator import RoadNetworkValidationError


class CoordinateValidationError(ValueError):
    """Raised when a latitude, longitude, or snap-distance limit is invalid."""


class NodeMappingError(ValueError):
    """Raised when a valid coordinate cannot be mapped to a graph node."""


@dataclass(frozen=True)
class NodeMatch:
    """The result of snapping a geographic coordinate to an OSM graph node.

    ``input_latitude`` and ``input_longitude`` are the public input values.
    ``matched_latitude`` and ``matched_longitude`` are the matched node's OSM
    ``y`` and ``x`` attributes. ``snap_distance_m`` is their distance in metres.
    """

    node_id: int
    input_latitude: float
    input_longitude: float
    matched_latitude: float
    matched_longitude: float
    snap_distance_m: float


def find_nearest_node(
    graph: nx.MultiDiGraph,
    latitude: float,
    longitude: float,
    max_snap_distance_m: float | None = None,
) -> NodeMatch:
    """Return the nearest OSM node to ``(latitude, longitude)``.

    Public coordinates use ``(latitude, longitude)`` order. OSM node
    attributes use the inverse attribute names: ``y`` is latitude and ``x``
    is longitude. The graph is read only; this function neither loads nor
    downloads a graph, and makes no graph, file, network, or routing changes.

    The supplied graph is expected to have been fully validated by the graph
    lifecycle layer. This function performs only mapping-specific lightweight
    checks, avoiding a full node-and-edge validation scan for each GPS lookup.
    """
    latitude = _validate_coordinate("latitude", latitude, -90.0, 90.0)
    longitude = _validate_coordinate("longitude", longitude, -180.0, 180.0)
    max_snap_distance_m = _validate_max_snap_distance(max_snap_distance_m)
    _validate_mapping_graph(graph)

    try:
        # OSMnx expects X=longitude and Y=latitude. For a geographic graph it
        # uses a great-circle distance and returns that distance in metres.
        node_id, distance_m = ox.distance.nearest_nodes(
            graph, X=longitude, Y=latitude, return_dist=True
        )
    except KeyError as error:
        raise RoadNetworkValidationError(
            "Road network nodes must provide required coordinate attributes 'x' and 'y'."
        ) from error
    except Exception as error:
        raise NodeMappingError(f"Failed to find the nearest OSM node: {error}") from error

    try:
        distance_m = float(distance_m)
    except (TypeError, ValueError) as error:
        raise NodeMappingError("Nearest-node lookup returned a non-numeric distance.") from error

    if not math.isfinite(distance_m) or distance_m < 0:
        raise NodeMappingError("Nearest-node lookup returned an invalid snap distance.")
    if max_snap_distance_m is not None and distance_m > max_snap_distance_m:
        raise NodeMappingError(
            f"Nearest node ({distance_m:.2f} m) exceeds maximum allowed snap "
            f"distance ({max_snap_distance_m:.2f} m)."
        )

    node_data = graph.nodes[node_id]
    return NodeMatch(
        node_id=node_id,
        input_latitude=latitude,
        input_longitude=longitude,
        matched_latitude=float(node_data["y"]),
        matched_longitude=float(node_data["x"]),
        snap_distance_m=distance_m,
    )


def _validate_coordinate(name: str, value: object, minimum: float, maximum: float) -> float:
    """Validate one public geographic coordinate without coercing input."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CoordinateValidationError(f"{name.capitalize()} must be a finite numeric value.")

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise CoordinateValidationError(f"{name.capitalize()} must be finite.")
    if not minimum <= numeric_value <= maximum:
        raise CoordinateValidationError(
            f"Invalid {name}: {numeric_value}. Must be between {minimum:g} and {maximum:g}."
        )
    return numeric_value


def _validate_max_snap_distance(value: float | None) -> float | None:
    """Validate the optional caller-provided mapping limit."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CoordinateValidationError("max_snap_distance_m must be a finite non-negative number.")

    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < 0:
        raise CoordinateValidationError("max_snap_distance_m must be a finite non-negative number.")
    return numeric_value


def _validate_mapping_graph(graph: nx.MultiDiGraph) -> None:
    """Perform only the checks required by an OSMnx nearest-node lookup.

    Full road-network validation, including node-coordinate and edge
    validation, belongs to graph loading/persistence. OSMnx validates the
    supplied node-coordinate columns when building its nearest-node index.
    """
    if not isinstance(graph, nx.MultiDiGraph):
        raise RoadNetworkValidationError(
            f"Road network must be a NetworkX MultiDiGraph. Got {type(graph)}"
        )
    if graph.number_of_nodes() == 0:
        raise RoadNetworkValidationError("Road network contains no nodes.")
    if "crs" not in graph.graph:
        raise RoadNetworkValidationError(
            "Road network is missing graph CRS metadata required for nearest-node lookup."
        )
