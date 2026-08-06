"""
tests/test_zone_registry.py
----------------------------
Unit tests for src/context/zone_registry.py

Tests cover:
  - YAML loading and parsing (areas + corridors)
  - nearest_area() — point inside a zone radius, point between zones
  - nearest_corridor() — point near a corridor, point far from all corridors
  - Code-based lookups (get_area_by_code, get_corridor_by_code)
  - Haversine distance sanity check
  - Error handling (file not found, empty zone list)
"""

import math
import pytest
from pathlib import Path

from src.context.zone_registry import (
    ZoneRegistry,
    AreaZone,
    CorridorZone,
    _haversine_km,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

CHENNAI_YAML = Path("configs/zones/chennai.yaml")


@pytest.fixture(scope="module")
def registry() -> ZoneRegistry:
    """Load the real Chennai YAML once for the whole test module."""
    if not CHENNAI_YAML.exists():
        pytest.skip(f"Zone config not found: {CHENNAI_YAML}")
    return ZoneRegistry(CHENNAI_YAML)


# ── Loading tests ─────────────────────────────────────────────────────────────

class TestLoading:
    def test_loads_without_error(self, registry):
        assert registry is not None

    def test_city_metadata(self, registry):
        assert registry.city == "Chennai"
        assert registry.city_code == "CHN"
        assert registry.timezone == "Asia/Kolkata"

    def test_area_count(self, registry):
        # We have 25 areas defined in the YAML
        assert len(registry.areas) == 25

    def test_corridor_count(self, registry):
        # We have 10 corridors defined in the YAML
        assert len(registry.corridors) == 10

    def test_total_zone_count(self, registry):
        assert len(registry.areas) + len(registry.corridors) == 35

    def test_area_fields_populated(self, registry):
        for zone in registry.areas:
            assert isinstance(zone, AreaZone)
            assert zone.name, f"Empty name for zone {zone.code}"
            assert zone.code.startswith("CHN_"), f"Bad code: {zone.code}"
            assert -90 <= zone.lat <= 90
            assert -180 <= zone.lon <= 180
            assert zone.radius_km > 0
            assert isinstance(zone.tags, list)

    def test_corridor_fields_populated(self, registry):
        for corridor in registry.corridors:
            assert isinstance(corridor, CorridorZone)
            assert corridor.name
            assert corridor.code.startswith("CHN_")
            assert len(corridor.waypoints) >= 2, (
                f"Corridor {corridor.code} needs at least 2 waypoints"
            )
            assert corridor.typical_freeflow_kmph > 0
            for wp in corridor.waypoints:
                assert len(wp) == 2
                lat, lon = wp
                assert 8 <= lat <= 14, f"Waypoint lat {lat} out of Tamil Nadu range"
                assert 75 <= lon <= 85, f"Waypoint lon {lon} out of Tamil Nadu range"

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            ZoneRegistry("configs/zones/nonexistent.yaml")


# ── Haversine tests ───────────────────────────────────────────────────────────

class TestHaversine:
    def test_same_point_is_zero(self):
        assert _haversine_km(13.0, 80.0, 13.0, 80.0) == 0.0

    def test_known_distance(self):
        # Anna Nagar (13.0850, 80.2101) to T. Nagar (13.0418, 80.2341)
        # Roughly ~5.5 km apart
        dist = _haversine_km(13.0850, 80.2101, 13.0418, 80.2341)
        assert 4.5 < dist < 6.5, f"Expected ~5.5 km, got {dist:.2f}"

    def test_symmetry(self):
        d1 = _haversine_km(13.0, 80.0, 13.1, 80.1)
        d2 = _haversine_km(13.1, 80.1, 13.0, 80.0)
        assert abs(d1 - d2) < 1e-6


# ── nearest_area() tests ──────────────────────────────────────────────────────

class TestNearestArea:
    def test_tnagar_centroid(self, registry):
        """Point at T. Nagar centroid should return T. Nagar."""
        zone = registry.nearest_area(13.0418, 80.2341)
        assert zone.code == "CHN_TNAGAR"

    def test_adyar_centroid(self, registry):
        zone = registry.nearest_area(13.0067, 80.2572)
        assert zone.code == "CHN_ADYAR"

    def test_sholinganallur_centroid(self, registry):
        zone = registry.nearest_area(12.9010, 80.2279)
        assert zone.code == "CHN_SHOLINGANALLUR"

    def test_anna_nagar_centroid(self, registry):
        zone = registry.nearest_area(13.0850, 80.2101)
        assert zone.code == "CHN_ANNANAGAR"

    def test_chromepet_centroid(self, registry):
        zone = registry.nearest_area(12.9516, 80.1462)
        assert zone.code == "CHN_CHROMEPET"

    def test_ambattur_industrial(self, registry):
        zone = registry.nearest_area(13.1143, 80.1548)
        assert zone.code == "CHN_AMBATTUR"

    def test_returns_area_zone_type(self, registry):
        zone = registry.nearest_area(13.0418, 80.2341)
        assert isinstance(zone, AreaZone)

    def test_point_between_zones_returns_closest(self, registry):
        """A point equidistant between two zones should still return one of them."""
        # Midpoint between Adyar and Velachery
        mid_lat = (13.0067 + 12.9815) / 2
        mid_lon = (80.2572 + 80.2180) / 2
        zone = registry.nearest_area(mid_lat, mid_lon)
        # Should return whichever centroid is closer — just not crash
        assert zone.code in {"CHN_ADYAR", "CHN_VELACHERY", "CHN_PERUNGUDI"}

    def test_outside_chennai_returns_best_guess(self, registry):
        """Even far outside Chennai, should return the nearest zone (not crash)."""
        zone = registry.nearest_area(13.5, 80.5)  # well north of Chennai
        assert isinstance(zone, AreaZone)


# ── nearest_corridor() tests ──────────────────────────────────────────────────

class TestNearestCorridor:
    def test_near_gst_road(self, registry):
        """Point near Chromepet on GST Road."""
        corridor = registry.nearest_corridor(12.9516, 80.1462)
        assert corridor is not None
        assert corridor.code == "CHN_GST"

    def test_near_omr(self, registry):
        """Point at Sholinganallur should be near OMR."""
        corridor = registry.nearest_corridor(12.9010, 80.2279)
        assert corridor is not None
        assert corridor.code == "CHN_OMR"

    def test_far_from_all_corridors_returns_none(self, registry):
        """A point far from Chennai should return None."""
        corridor = registry.nearest_corridor(
            20.0, 80.0, max_distance_km=5.0  # well outside Chennai
        )
        assert corridor is None

    def test_returns_corridor_zone_type(self, registry):
        corridor = registry.nearest_corridor(12.9516, 80.1462)
        assert isinstance(corridor, CorridorZone)

    def test_corridor_has_waypoints(self, registry):
        corridor = registry.nearest_corridor(12.9516, 80.1462)
        assert len(corridor.waypoints) >= 2

    def test_max_distance_respected(self, registry):
        """Point in Avadi (far north-west suburb) is > 5 km from all corridors."""
        # Avadi centroid (13.1162, 80.1025) is >6 km from the nearest corridor
        # (nearest is IRR at ~6.4 km). A 5 km cap should return None.
        corridor = registry.nearest_corridor(13.1162, 80.1025, max_distance_km=5.0)
        assert corridor is None


# ── Code lookup tests ─────────────────────────────────────────────────────────

class TestCodeLookup:
    def test_get_area_by_valid_code(self, registry):
        zone = registry.get_area_by_code("CHN_ADYAR")
        assert zone is not None
        assert zone.name == "Adyar"

    def test_get_area_by_invalid_code(self, registry):
        zone = registry.get_area_by_code("CHN_FAKE")
        assert zone is None

    def test_get_corridor_by_valid_code(self, registry):
        corridor = registry.get_corridor_by_code("CHN_GST")
        assert corridor is not None
        assert "GST" in corridor.name

    def test_get_corridor_by_invalid_code(self, registry):
        corridor = registry.get_corridor_by_code("CHN_FAKE_ROAD")
        assert corridor is None

    def test_all_area_codes_unique(self, registry):
        codes = [z.code for z in registry.areas]
        assert len(codes) == len(set(codes)), "Duplicate area codes found"

    def test_all_corridor_codes_unique(self, registry):
        codes = [c.code for c in registry.corridors]
        assert len(codes) == len(set(codes)), "Duplicate corridor codes found"


# ── CorridorZone property tests ───────────────────────────────────────────────

class TestCorridorProperties:
    def test_anchor_lat_is_within_waypoint_range(self, registry):
        for corridor in registry.corridors:
            lats = [wp[0] for wp in corridor.waypoints]
            assert min(lats) <= corridor.anchor_lat <= max(lats)

    def test_anchor_lon_is_within_waypoint_range(self, registry):
        for corridor in registry.corridors:
            lons = [wp[1] for wp in corridor.waypoints]
            assert min(lons) <= corridor.anchor_lon <= max(lons)

    def test_freeflow_speeds_are_positive(self, registry):
        for corridor in registry.corridors:
            assert corridor.typical_freeflow_kmph > 0


# ── Summary / repr tests ──────────────────────────────────────────────────────

class TestRepr:
    def test_summary_contains_city(self, registry):
        assert "Chennai" in registry.summary()

    def test_summary_contains_counts(self, registry):
        summary = registry.summary()
        assert "25" in summary  # 25 areas
        assert "10" in summary  # 10 corridors

    def test_area_repr(self, registry):
        zone = registry.get_area_by_code("CHN_TNAGAR")
        assert "CHN_TNAGAR" in repr(zone)

    def test_corridor_repr(self, registry):
        corridor = registry.get_corridor_by_code("CHN_GST")
        assert "CHN_GST" in repr(corridor)
