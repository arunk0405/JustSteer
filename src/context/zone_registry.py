"""
src/context/zone_registry.py
----------------------------
Loads the city zone configuration from a YAML file and provides
fast spatial lookup:

  - nearest_area(lat, lon)     -> AreaZone
  - nearest_corridor(lat, lon) -> CorridorZone | None

Distance calculation uses the Haversine formula (no external deps).
All zone metadata (tags, radius, freeflow speed) is preserved for
downstream use by the fetchers and context aggregator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AreaZone:
    """Represents a residential / commercial / industrial area zone."""
    name: str
    code: str
    zone_type: str                          # residential | commercial | industrial | coastal
    lat: float
    lon: float
    radius_km: float
    tags: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"AreaZone(code={self.code!r}, name={self.name!r})"


@dataclass
class CorridorZone:
    """Represents a named road corridor made up of ordered waypoints."""
    name: str
    code: str
    corridor_type: str                      # highway | urban_road
    description: str
    waypoints: List[Tuple[float, float]]    # list of (lat, lon)
    typical_freeflow_kmph: float
    tags: List[str] = field(default_factory=list)

    @property
    def anchor_lat(self) -> float:
        """Midpoint latitude of the corridor (used for proximity checks)."""
        return sum(wp[0] for wp in self.waypoints) / len(self.waypoints)

    @property
    def anchor_lon(self) -> float:
        """Midpoint longitude of the corridor (used for proximity checks)."""
        return sum(wp[1] for wp in self.waypoints) / len(self.waypoints)

    def __repr__(self) -> str:
        return f"CorridorZone(code={self.code!r}, name={self.name!r})"


# ── Haversine helper ──────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Returns the great-circle distance in kilometres between two
    (lat, lon) points using the Haversine formula.
    Accurate to within ~0.3% for distances under 500 km.
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _min_distance_to_corridor(
    lat: float, lon: float, corridor: CorridorZone
) -> float:
    """
    Returns the minimum Haversine distance from (lat, lon) to any
    waypoint on the corridor. This is an approximation — accurate
    enough for assigning a delivery point to a nearby corridor.
    """
    return min(
        _haversine_km(lat, lon, wp_lat, wp_lon)
        for wp_lat, wp_lon in corridor.waypoints
    )


# ── Registry ──────────────────────────────────────────────────────────────────

class ZoneRegistry:
    """
    Loads zone data from a YAML file and exposes spatial lookup methods.

    Parameters
    ----------
    yaml_path : str | Path
        Path to the city zones YAML (e.g. configs/zones/chennai.yaml).

    Example
    -------
    >>> registry = ZoneRegistry("configs/zones/chennai.yaml")
    >>> zone = registry.nearest_area(13.0418, 80.2341)
    >>> print(zone.name)
    T. Nagar
    """

    def __init__(self, yaml_path: str | Path) -> None:
        self._path = Path(yaml_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Zone config not found: {self._path}")

        with self._path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        self.city: str = raw.get("city", "Unknown")
        self.city_code: str = raw.get("city_code", "UNK")
        self.timezone: str = raw.get("timezone", "UTC")

        self.areas: List[AreaZone] = self._parse_areas(raw.get("areas", []))
        self.corridors: List[CorridorZone] = self._parse_corridors(raw.get("corridors", []))

        # Build lookup indices keyed by zone code for O(1) access
        self._area_by_code: dict[str, AreaZone] = {z.code: z for z in self.areas}
        self._corridor_by_code: dict[str, CorridorZone] = {c.code: c for c in self.corridors}

    # ── Parsing helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_areas(raw_list: list) -> List[AreaZone]:
        zones = []
        for item in raw_list:
            zones.append(AreaZone(
                name=item["name"],
                code=item["code"],
                zone_type=item.get("type", "residential"),
                lat=float(item["lat"]),
                lon=float(item["lon"]),
                radius_km=float(item.get("radius_km", 2.0)),
                tags=item.get("tags", []),
            ))
        return zones

    @staticmethod
    def _parse_corridors(raw_list: list) -> List[CorridorZone]:
        corridors = []
        for item in raw_list:
            raw_waypoints = item.get("waypoints", [])
            waypoints: List[Tuple[float, float]] = [
                (float(wp[0]), float(wp[1])) for wp in raw_waypoints
            ]
            corridors.append(CorridorZone(
                name=item["name"],
                code=item["code"],
                corridor_type=item.get("type", "urban_road"),
                description=item.get("description", ""),
                waypoints=waypoints,
                typical_freeflow_kmph=float(item.get("typical_freeflow_kmph", 40.0)),
                tags=item.get("tags", []),
            ))
        return corridors

    # ── Spatial lookup ─────────────────────────────────────────────────────────

    def nearest_area(self, lat: float, lon: float) -> AreaZone:
        """
        Returns the AreaZone whose centroid is closest to (lat, lon).

        Preference is given to zones whose radius_km contains the point;
        if no zone contains the point, the closest centroid is returned.
        """
        if not self.areas:
            raise RuntimeError("No area zones loaded — check the YAML file.")

        # First pass: zones that contain the point within radius
        containing = [
            z for z in self.areas
            if _haversine_km(lat, lon, z.lat, z.lon) <= z.radius_km
        ]

        candidates = containing if containing else self.areas

        return min(
            candidates,
            key=lambda z: _haversine_km(lat, lon, z.lat, z.lon),
        )

    def nearest_corridor(
        self, lat: float, lon: float, max_distance_km: float = 5.0
    ) -> Optional[CorridorZone]:
        """
        Returns the CorridorZone closest to (lat, lon), or None if
        no corridor falls within max_distance_km.

        Distance is measured to the nearest waypoint of each corridor.
        """
        if not self.corridors:
            return None

        best: Optional[CorridorZone] = None
        best_dist = float("inf")

        for corridor in self.corridors:
            dist = _min_distance_to_corridor(lat, lon, corridor)
            if dist < best_dist:
                best_dist = dist
                best = corridor

        return best if best_dist <= max_distance_km else None

    def get_area_by_code(self, code: str) -> Optional[AreaZone]:
        """Look up an AreaZone by its code (e.g. 'CHN_TNAGAR')."""
        return self._area_by_code.get(code)

    def get_corridor_by_code(self, code: str) -> Optional[CorridorZone]:
        """Look up a CorridorZone by its code (e.g. 'CHN_GST')."""
        return self._corridor_by_code.get(code)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        return (
            f"ZoneRegistry({self.city}) — "
            f"{len(self.areas)} areas, {len(self.corridors)} corridors"
        )

    def __repr__(self) -> str:
        return self.summary()
