"""
src/context/context_aggregator.py
----------------------------------
Maps a delivery coordinate (lat, lon) to its nearest zone, fetches
weather and traffic context, and returns a unified DeliveryContext object.

This is the single entry point that route optimization logic (and the
training data enrichment pipeline) should call. It hides all the caching
and API details behind a clean interface:

    ctx = aggregator.get_context(lat=13.0418, lon=80.2341)

    ctx.zone_name          -> "T. Nagar"
    ctx.zone_code          -> "CHN_TNAGAR"
    ctx.weather            -> WeatherSnapshot(...)
    ctx.traffic            -> TrafficSnapshot(...)  or None
    ctx.to_feature_dict()  -> flat dict ready to be a DataFrame row
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.context.zone_registry import ZoneRegistry, AreaZone, CorridorZone
from src.context.weather_fetcher import WeatherFetcher, WeatherSnapshot
from src.context.traffic_fetcher import TrafficFetcher, TrafficSnapshot

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DeliveryContext:
    """
    Enriched context for a specific delivery coordinate.

    Attributes
    ----------
    lat, lon         : the input coordinates
    area             : the nearest AreaZone from the registry
    corridor         : the nearest CorridorZone (None if no corridor is close)
    weather          : WeatherSnapshot from OWM (None if API unavailable)
    traffic          : TrafficSnapshot from TomTom (None if API unavailable
                       or no corridor assigned)
    """
    lat: float
    lon: float
    area: AreaZone
    corridor: Optional[CorridorZone]
    weather: Optional[WeatherSnapshot]
    traffic: Optional[TrafficSnapshot]

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def zone_name(self) -> str:
        return self.area.name

    @property
    def zone_code(self) -> str:
        return self.area.code

    @property
    def zone_type(self) -> str:
        return self.area.zone_type

    @property
    def corridor_name(self) -> Optional[str]:
        return self.corridor.name if self.corridor else None

    @property
    def corridor_code(self) -> Optional[str]:
        return self.corridor.code if self.corridor else None

    # ── Feature export ─────────────────────────────────────────────────────────

    def to_feature_dict(self) -> dict:
        """
        Returns a flat dictionary of all context fields.
        Suitable for direct use as a pandas DataFrame row or CSV record.

        Missing values (when API is unavailable) are filled with None so
        they can be handled downstream by imputation.
        """
        base = {
            "input_lat":       self.lat,
            "input_lon":       self.lon,
            "zone_name":       self.zone_name,
            "zone_code":       self.zone_code,
            "zone_type":       self.zone_type,
            "zone_tags":       "|".join(self.area.tags),  # pipe-separated for CSV
            "corridor_name":   self.corridor_name,
            "corridor_code":   self.corridor_code,
        }

        weather_dict: dict = {}
        if self.weather:
            weather_dict = self.weather.to_feature_dict()
        else:
            # Placeholder keys so the schema is consistent even without API
            weather_dict = {
                "weather_zone_code": None,
                "weather_condition": None,
                "temp_celsius":      None,
                "humidity_pct":      None,
                "wind_speed_mps":    None,
                "visibility_m":      None,
                "rain_1h_mm":        None,
                "is_raining":        None,
                "is_foggy":          None,
                "is_high_wind":      None,
            }

        traffic_dict: dict = {}
        if self.traffic:
            traffic_dict = self.traffic.to_feature_dict()
        else:
            traffic_dict = {
                "traffic_corridor_code": None,
                "current_speed_kmph":    None,
                "freeflow_speed_kmph":   None,
                "congestion_index":      None,
                "congestion_level":      None,
                "travel_time_ratio":     None,
            }

        return {**base, **weather_dict, **traffic_dict}

    def __repr__(self) -> str:
        weather_str = self.weather.condition if self.weather else "no weather"
        traffic_str = (
            f"{self.traffic.congestion_level}" if self.traffic else "no traffic"
        )
        return (
            f"DeliveryContext(zone={self.zone_name!r}, "
            f"corridor={self.corridor_name!r}, "
            f"weather={weather_str!r}, "
            f"traffic={traffic_str!r})"
        )


# ── Aggregator ────────────────────────────────────────────────────────────────

class ContextAggregator:
    """
    Combines ZoneRegistry + WeatherFetcher + TrafficFetcher into a single
    get_context() call.

    Parameters
    ----------
    registry : ZoneRegistry
        Loaded zone registry for a city.
    weather_fetcher : WeatherFetcher
        Configured OWM weather fetcher.
    traffic_fetcher : TrafficFetcher
        Configured TomTom traffic fetcher.
    corridor_max_distance_km : float
        How close a corridor must be to the point before it's assigned.
        Default: 5.0 km.

    Example
    -------
    >>> registry   = ZoneRegistry("configs/zones/chennai.yaml")
    >>> weather    = WeatherFetcher(dry_run=True)
    >>> traffic    = TrafficFetcher(dry_run=True)
    >>> aggregator = ContextAggregator(registry, weather, traffic)
    >>> ctx = aggregator.get_context(lat=13.0067, lon=80.2572)
    >>> print(ctx)
    DeliveryContext(zone='Adyar', corridor='Sardar Patel Road (Adyar–Guindy)', ...)
    """

    def __init__(
        self,
        registry: ZoneRegistry,
        weather_fetcher: WeatherFetcher,
        traffic_fetcher: TrafficFetcher,
        corridor_max_distance_km: float = 5.0,
    ) -> None:
        self._registry = registry
        self._weather = weather_fetcher
        self._traffic = traffic_fetcher
        self._corridor_max_km = corridor_max_distance_km

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_context(self, lat: float, lon: float) -> DeliveryContext:
        """
        Returns enriched delivery context for the given coordinates.

        Steps:
        1. Find nearest AreaZone from the registry.
        2. Find nearest CorridorZone within corridor_max_distance_km.
        3. Fetch weather for the area zone centroid.
        4. Fetch traffic for the corridor (if one was found).

        Parameters
        ----------
        lat, lon : float
            Delivery coordinate (pickup or drop location).
        """
        # Step 1: zone lookup (always succeeds — returns best match)
        area = self._registry.nearest_area(lat, lon)
        logger.debug(
            "ContextAggregator: (%s, %s) → area=%s", lat, lon, area.code
        )

        # Step 2: corridor lookup (may return None)
        corridor = self._registry.nearest_corridor(
            lat, lon, max_distance_km=self._corridor_max_km
        )
        logger.debug(
            "ContextAggregator: (%s, %s) → corridor=%s",
            lat, lon, corridor.code if corridor else "none",
        )

        # Step 3: weather (use area centroid as the query point)
        weather = self._weather.fetch(
            zone_code=area.code,
            lat=area.lat,
            lon=area.lon,
        )

        # Step 4: traffic (only if a corridor was found)
        traffic: Optional[TrafficSnapshot] = None
        if corridor is not None:
            traffic = self._traffic.fetch(
                corridor_code=corridor.code,
                waypoints=corridor.waypoints,
            )

        return DeliveryContext(
            lat=lat,
            lon=lon,
            area=area,
            corridor=corridor,
            weather=weather,
            traffic=traffic,
        )

    def enrich_batch(
        self, coordinates: list[tuple[float, float]]
    ) -> list[DeliveryContext]:
        """
        Convenience wrapper: get_context() for a list of (lat, lon) pairs.
        Caching means repeated zones are not re-fetched.

        Parameters
        ----------
        coordinates : list of (lat, lon)
            e.g. [(13.04, 80.23), (12.98, 80.22), ...]
        """
        return [self.get_context(lat, lon) for lat, lon in coordinates]

    def to_dataframe(self, coordinates: list[tuple[float, float]]):
        """
        Enriches a list of coordinates and returns a pandas DataFrame.
        Each row contains zone + weather + traffic features.
        Requires pandas to be installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pandas"
            ) from exc

        contexts = self.enrich_batch(coordinates)
        rows = [ctx.to_feature_dict() for ctx in contexts]
        return pd.DataFrame(rows)
