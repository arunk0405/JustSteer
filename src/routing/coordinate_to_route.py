"""Single-leg orchestration from geographic coordinates to an A* route.

The module connects the graph-domain coordinate mapper with the existing A*
engine. It deliberately does not own graph lifecycle, edge-cost calculation,
or any application-interface concerns.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from src.graph.coordinate_mapper import NodeMatch, find_nearest_node
from src.routing.astar import AStarRoutingEngine
from src.routing.route_result import RouteResult


@dataclass(frozen=True)
class CoordinateRouteResult:
    """A single-leg route together with the coordinate snaps that produced it.

    ``origin_match`` and ``destination_match`` retain the node IDs and snap
    distances needed by a future application layer. ``route`` is the existing
    A* ``RouteResult`` and is not altered by this integration layer.
    """

    origin_match: NodeMatch
    destination_match: NodeMatch
    route: RouteResult


def route_coordinates(
    graph: nx.MultiDiGraph,
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
    *,
    router: AStarRoutingEngine,
) -> CoordinateRouteResult:
    """Route one coordinate pair to another using supplied domain components.

    Inputs use ``(latitude, longitude)`` order. The coordinate mapper snaps
    each input to an OSM node (where ``y`` is latitude and ``x`` is longitude),
    then the supplied existing A* router computes a directed, single-leg route
    between those node IDs.

    The graph and router are injected by the caller. This function neither
    loads nor validates a graph's full structure, creates edge costs, nor
    performs any routing itself. Coordinate and routing exceptions propagate
    unchanged from their respective domain components.

    Raises:
        ValueError: If ``router`` is configured for a different graph object.
        CoordinateValidationError: If either coordinate is invalid.
        RoadNetworkValidationError: If the graph cannot be used for mapping.
        NodeMappingError: If a coordinate cannot be snapped to a node.
        RouteNotFoundError: If no directed path exists between the snapped nodes.
    """
    if router.graph is not graph:
        raise ValueError("The supplied router and graph must reference the same graph instance.")

    origin_match = find_nearest_node(
        graph,
        latitude=origin_latitude,
        longitude=origin_longitude,
    )
    destination_match = find_nearest_node(
        graph,
        latitude=destination_latitude,
        longitude=destination_longitude,
    )
    route = router.route(origin_match.node_id, destination_match.node_id)

    return CoordinateRouteResult(
        origin_match=origin_match,
        destination_match=destination_match,
        route=route,
    )
