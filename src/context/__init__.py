"""
src/context/__init__.py
-----------------------
Public API for the context module.

Usage:

    from src.context import ZoneRegistry, WeatherFetcher, TrafficFetcher, ContextAggregator

    registry   = ZoneRegistry("configs/zones/chennai.yaml")
    weather    = WeatherFetcher(api_key="YOUR_OWM_KEY")
    traffic    = TrafficFetcher(api_key="YOUR_TOMTOM_KEY")
    aggregator = ContextAggregator(registry, weather, traffic)

    ctx = aggregator.get_context(lat=13.0418, lon=80.2341)
    # ctx.zone_name        -> "T. Nagar"
    # ctx.weather          -> WeatherSnapshot(...)
    # ctx.traffic          -> TrafficSnapshot(...)  (nearest corridor or None)
"""

from src.context.zone_registry import ZoneRegistry, AreaZone, CorridorZone
from src.context.weather_fetcher import WeatherFetcher, WeatherSnapshot
from src.context.traffic_fetcher import TrafficFetcher, TrafficSnapshot
from src.context.context_aggregator import ContextAggregator, DeliveryContext

__all__ = [
    "ZoneRegistry",
    "AreaZone",
    "CorridorZone",
    "WeatherFetcher",
    "WeatherSnapshot",
    "TrafficFetcher",
    "TrafficSnapshot",
    "ContextAggregator",
    "DeliveryContext",
]
