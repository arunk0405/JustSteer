class RoutingError(Exception):
    """Base class for all routing-related exceptions."""
    pass


class InvalidNodeError(RoutingError):
    """Raised when a provided source or destination node does not exist in the graph."""
    pass


class RouteNotFoundError(RoutingError):
    """Raised when no valid path can be found between source and destination (e.g. unreachable components)."""
    pass


class InvalidEdgeCostError(RoutingError):
    """Raised when an edge contains a negative, NaN, or otherwise unparseable cost."""
    pass


class GraphUnavailableError(RoutingError):
    """Raised when the routing engine attempts to operate on a missing or uninitialized graph."""
    pass
