"""
tests/test_context_integration.py
----------------------------------
Phase 2 integration tests.

All tests use dry_run=True (no API keys required).
Covers the full stack: WeatherFetcher → TrafficFetcher → ContextAggregator →
Enricher, including the new Phase 2.2/2.3 features.

Test groups:
  TestWeatherFetcherPhase22  - rain_risk_score, NE monsoon, hourly forecast
  TestTrafficFetcherPhase23  - heuristic fallback, TTR clamping, peak-hour logic
  TestContextAggregator      - get_context(), to_feature_dict(), enrich_batch()
  TestEnricher               - run_enrichment() against real processed_data.csv
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone

from src.context.zone_registry import ZoneRegistry
from src.context.weather_fetcher import WeatherFetcher, WeatherSnapshot, HourlyForecast
from src.context.traffic_fetcher import TrafficFetcher, TrafficSnapshot
from src.context.context_aggregator import ContextAggregator, DeliveryContext
from src.context.enricher import run_enrichment, EnricherConfig

# ── Shared fixtures ───────────────────────────────────────────────────────────

CHENNAI_YAML    = Path("configs/zones/chennai.yaml")
CONTEXT_CONFIG  = Path("configs/context_config.yaml")
PROCESSED_CSV   = Path("output/processed_data.csv")
ENRICHED_CSV    = Path("output/enriched_data.csv")

# Representative Chennai coordinates
TNAGAR     = (13.0418, 80.2341)
ADYAR      = (13.0067, 80.2572)
CHROMEPET  = (12.9516, 80.1462)
AMBATTUR   = (13.1143, 80.1548)
SHOLING    = (12.9010, 80.2279)


@pytest.fixture(scope="module")
def registry() -> ZoneRegistry:
    if not CHENNAI_YAML.exists():
        pytest.skip(f"Zone config not found: {CHENNAI_YAML}")
    return ZoneRegistry(CHENNAI_YAML)


@pytest.fixture(scope="module")
def weather() -> WeatherFetcher:
    return WeatherFetcher(dry_run=True)


@pytest.fixture(scope="module")
def traffic() -> TrafficFetcher:
    return TrafficFetcher(dry_run=True)


@pytest.fixture(scope="module")
def aggregator(registry, weather, traffic) -> ContextAggregator:
    return ContextAggregator(registry, weather, traffic)


# ── WeatherFetcher Phase 2.2 tests ────────────────────────────────────────────

class TestWeatherFetcherPhase22:

    def test_fetch_returns_snapshot(self, weather):
        snap = weather.fetch("CHN_TNAGAR", lat=13.0418, lon=80.2341)
        assert isinstance(snap, WeatherSnapshot)

    def test_snapshot_has_rain_risk_score(self, weather):
        snap = weather.fetch("CHN_ADYAR", lat=13.0067, lon=80.2572)
        assert hasattr(snap, "rain_risk_score")
        assert 0.0 <= snap.rain_risk_score <= 1.0

    def test_snapshot_has_ne_monsoon_flag(self, weather):
        snap = weather.fetch("CHN_ADYAR", lat=13.0067, lon=80.2572)
        assert hasattr(snap, "is_ne_monsoon")
        assert isinstance(snap.is_ne_monsoon, bool)

    def test_snapshot_has_hourly_forecast(self, weather):
        snap = weather.fetch("CHN_CHROMEPET", lat=12.9516, lon=80.1462)
        assert hasattr(snap, "hourly_forecast")
        assert isinstance(snap.hourly_forecast, list)

    def test_hourly_forecast_slots_have_correct_type(self, weather):
        snap = weather.fetch("CHN_SHOLINGANALLUR", lat=12.9010, lon=80.2279)
        for slot in snap.hourly_forecast:
            assert isinstance(slot, HourlyForecast)
            assert isinstance(slot.condition, str)
            assert isinstance(slot.rain_3h_mm, float)

    def test_rain_risk_score_zero_for_clear(self):
        """Clear sky → risk score should be 0.0"""
        fetcher = WeatherFetcher(dry_run=True)
        snap = fetcher._mock_snapshot("CHN_TEST")
        # Mock returns Clear condition with 0 rain
        assert snap.rain_risk_score == 0.0

    def test_rain_risk_score_positive_for_rain(self):
        """Simulate a rain condition and verify risk score > 0."""
        fetcher = WeatherFetcher(dry_run=True)
        score = fetcher._compute_rain_risk(
            rain_mm=8.0, condition="Rain", is_ne_monsoon=False
        )
        assert score > 0.0

    def test_rain_risk_score_higher_during_ne_monsoon(self):
        """Same rain intensity → higher score during NE monsoon."""
        fetcher = WeatherFetcher(dry_run=True)
        base  = fetcher._compute_rain_risk(5.0, "Rain", is_ne_monsoon=False)
        ne    = fetcher._compute_rain_risk(5.0, "Rain", is_ne_monsoon=True)
        assert ne > base

    def test_rain_risk_score_clamped_to_1(self):
        fetcher = WeatherFetcher(dry_run=True)
        score = fetcher._compute_rain_risk(
            rain_mm=100.0, condition="Thunderstorm", is_ne_monsoon=True
        )
        assert score <= 1.0

    def test_ne_monsoon_true_in_october(self):
        fetcher = WeatherFetcher(dry_run=True)
        oct_dt = datetime(2026, 10, 15, tzinfo=timezone.utc)
        with patch("src.context.weather_fetcher.datetime") as mock_dt:
            mock_dt.now.return_value = oct_dt
            mock_dt.fromtimestamp = datetime.fromtimestamp
            snap = fetcher._parse_current("CHN_TEST", _mock_owm_rain_response())
        assert snap.is_ne_monsoon is True

    def test_ne_monsoon_false_in_july(self):
        fetcher = WeatherFetcher(dry_run=True)
        jul_dt = datetime(2026, 7, 15, tzinfo=timezone.utc)
        with patch("src.context.weather_fetcher.datetime") as mock_dt:
            mock_dt.now.return_value = jul_dt
            mock_dt.fromtimestamp = datetime.fromtimestamp
            snap = fetcher._parse_current("CHN_TEST", _mock_owm_clear_response())
        assert snap.is_ne_monsoon is False

    def test_cache_hit_on_second_fetch(self, weather):
        weather.fetch("CHN_KILPAUK", lat=13.0827, lon=80.2297)
        info_before = weather.cache_info()["current_cached_zones"]
        weather.fetch("CHN_KILPAUK", lat=13.0827, lon=80.2297)
        info_after  = weather.cache_info()["current_cached_zones"]
        assert info_after == info_before  # no new entry

    def test_to_feature_dict_has_new_fields(self, weather):
        snap = weather.fetch("CHN_EGMORE", lat=13.0732, lon=80.2609)
        d = snap.to_feature_dict()
        assert "rain_risk_score"     in d
        assert "is_ne_monsoon"       in d
        assert "rain_forecast_hours" in d

    def test_no_api_key_returns_none_without_dry_run(self):
        fetcher = WeatherFetcher(api_key="", dry_run=False)
        snap = fetcher.fetch("CHN_TEST", lat=13.0, lon=80.0)
        assert snap is None


# ── TrafficFetcher Phase 2.3 tests ────────────────────────────────────────────

class TestTrafficFetcherPhase23:

    def test_heuristic_returns_snapshot(self, traffic, registry):
        corridor = registry.get_corridor_by_code("CHN_GST")
        snap = traffic.fetch("CHN_GST", waypoints=corridor.waypoints, hour=9)
        assert isinstance(snap, TrafficSnapshot)

    def test_heuristic_source_is_heuristic(self, traffic, registry):
        corridor = registry.get_corridor_by_code("CHN_ORR")
        snap = traffic.fetch("CHN_ORR", waypoints=corridor.waypoints, hour=14)
        assert snap.source == "heuristic"

    def test_morning_peak_is_more_congested_than_off_peak(self, traffic, registry):
        corridor = registry.get_corridor_by_code("CHN_ANNASALAI")
        peak    = traffic.fetch("CHN_ANNASALAI_PEAK",
                                waypoints=corridor.waypoints, hour=9, is_weekend=False)
        offpeak = traffic.fetch("CHN_ANNASALAI_OFF",
                                waypoints=corridor.waypoints, hour=14, is_weekend=False)
        # Lower congestion_index = more congested (1.0 = free flow, 0.0 = standstill)
        assert peak.congestion_index <= offpeak.congestion_index

    def test_weekend_less_congested_than_weekday_during_peak(self, traffic, registry):
        corridor = registry.get_corridor_by_code("CHN_GST")
        wkday = traffic.fetch("CHN_GST_WD", waypoints=corridor.waypoints,
                              hour=8, is_weekend=False)
        wkend = traffic.fetch("CHN_GST_WE", waypoints=corridor.waypoints,
                              hour=8, is_weekend=True)
        assert wkday.congestion_index <= wkend.congestion_index

    def test_night_time_free_flow(self, traffic, registry):
        corridor = registry.get_corridor_by_code("CHN_OMR")
        night = traffic.fetch("CHN_OMR_NIGHT", waypoints=corridor.waypoints,
                              hour=2, is_weekend=False)
        assert night.congestion_level in {"free_flow", "moderate"}

    def test_travel_time_ratio_is_clamped(self):
        """A very low congestion_index should produce a clamped ratio, not infinity."""
        fetcher = TrafficFetcher(dry_run=True)
        snap = TrafficSnapshot(
            corridor_code="TEST",
            current_speed_kmph=5.0,
            freeflow_speed_kmph=60.0,
            congestion_index=0.08,    # would give ratio=12.5 without clamping
            congestion_level="severe",
            confidence=0.9,
            num_waypoints_sampled=1,
            source="tomtom",
        )
        assert snap.travel_time_ratio <= 5.0

    def test_travel_time_ratio_zero_index_clamped(self):
        snap = TrafficSnapshot(
            corridor_code="TEST",
            current_speed_kmph=0.0,
            freeflow_speed_kmph=60.0,
            congestion_index=0.0,
            congestion_level="severe",
            confidence=0.0,
            num_waypoints_sampled=1,
        )
        assert snap.travel_time_ratio <= 5.0
        assert snap.travel_time_ratio != float("inf")

    def test_to_feature_dict_has_traffic_source(self, traffic, registry):
        corridor = registry.get_corridor_by_code("CHN_ECR")
        snap = traffic.fetch("CHN_ECR", waypoints=corridor.waypoints, hour=10)
        d = snap.to_feature_dict()
        assert "traffic_source" in d
        assert d["traffic_source"] in {"tomtom", "heuristic"}

    def test_congestion_level_valid_values(self, traffic, registry):
        valid_levels = {"free_flow", "moderate", "heavy", "severe"}
        corridor = registry.get_corridor_by_code("CHN_IRR")
        for hour in [8, 12, 18, 23]:
            snap = traffic.fetch(f"CHN_IRR_{hour}", waypoints=corridor.waypoints,
                                 hour=hour, is_weekend=False)
            assert snap.congestion_level in valid_levels

    def test_no_api_key_heuristic_enabled_returns_snapshot(self):
        # Explicitly patch the environment to ensure no API key is available,
        # even if TOMTOM_API_KEY is set in .env
        import os
        from unittest.mock import patch
        env_without_key = {k: v for k, v in os.environ.items() if k != "TOMTOM_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            fetcher = TrafficFetcher(
                api_key="",
                dry_run=False,
                config={"peak_hour_heuristic": {"enabled": True}},
            )
            snap = fetcher.fetch("CHN_HEURISTIC_TEST", waypoints=[(13.0, 80.0), (12.9, 80.1)])
        assert snap is not None
        assert snap.source == "heuristic"


# ── ContextAggregator tests ───────────────────────────────────────────────────

class TestContextAggregator:

    def test_get_context_returns_delivery_context(self, aggregator):
        ctx = aggregator.get_context(*TNAGAR)
        assert isinstance(ctx, DeliveryContext)

    def test_zone_correctly_resolved(self, aggregator):
        ctx = aggregator.get_context(*ADYAR)
        assert ctx.zone_code == "CHN_ADYAR"

    def test_corridor_assigned_near_gst(self, aggregator):
        ctx = aggregator.get_context(*CHROMEPET)
        assert ctx.corridor is not None
        assert ctx.corridor.code == "CHN_GST"

    def test_weather_snapshot_present(self, aggregator):
        ctx = aggregator.get_context(*TNAGAR)
        assert ctx.weather is not None
        assert isinstance(ctx.weather, WeatherSnapshot)

    def test_traffic_snapshot_present_near_corridor(self, aggregator):
        ctx = aggregator.get_context(*CHROMEPET)
        assert ctx.traffic is not None
        assert isinstance(ctx.traffic, TrafficSnapshot)

    def test_to_feature_dict_has_all_expected_keys(self, aggregator):
        ctx = aggregator.get_context(*TNAGAR)
        d = ctx.to_feature_dict()
        required_keys = [
            "zone_name", "zone_code", "zone_type", "zone_tags",
            "corridor_name", "corridor_code",
            "weather_condition", "rain_risk_score", "is_ne_monsoon",
            "rain_forecast_hours",
            "congestion_index", "congestion_level", "travel_time_ratio",
            "traffic_source",
        ]
        for key in required_keys:
            assert key in d, f"Missing key in feature dict: {key}"

    def test_to_feature_dict_none_when_no_corridor(self, aggregator):
        # Avadi is far from all corridors
        ctx = aggregator.get_context(13.1162, 80.1025)
        d = ctx.to_feature_dict()
        assert d["corridor_name"] is None
        assert d["congestion_index"] is None

    def test_enrich_batch_returns_list(self, aggregator):
        points = [TNAGAR, ADYAR, CHROMEPET, SHOLING, AMBATTUR]
        results = aggregator.enrich_batch(points)
        assert len(results) == 5
        assert all(isinstance(r, DeliveryContext) for r in results)

    def test_to_dataframe_shape(self, aggregator):
        points = [TNAGAR, ADYAR, CHROMEPET]
        df = aggregator.to_dataframe(points)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "zone_name" in df.columns

    def test_caching_does_not_change_results(self, aggregator):
        ctx1 = aggregator.get_context(*TNAGAR)
        ctx2 = aggregator.get_context(*TNAGAR)
        assert ctx1.zone_code == ctx2.zone_code
        assert ctx1.weather.condition == ctx2.weather.condition


# ── Enricher integration tests ────────────────────────────────────────────────

class TestEnricher:

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Remove enriched CSV before and after each test."""
        if ENRICHED_CSV.exists():
            ENRICHED_CSV.unlink()
        yield
        if ENRICHED_CSV.exists():
            ENRICHED_CSV.unlink()

    def test_run_enrichment_creates_output(self):
        if not PROCESSED_CSV.exists():
            pytest.skip(f"processed_data.csv not found at {PROCESSED_CSV}")
        cfg = EnricherConfig(
            input_csv=str(PROCESSED_CSV),
            output_csv=str(ENRICHED_CSV),
            verbose=False,
        )
        result = run_enrichment(cfg)
        assert Path(result.output_path).exists()

    def test_enriched_csv_has_more_columns_than_original(self):
        if not PROCESSED_CSV.exists():
            pytest.skip(f"processed_data.csv not found at {PROCESSED_CSV}")
        original_cols = len(pd.read_csv(PROCESSED_CSV).columns)
        cfg = EnricherConfig(
            input_csv=str(PROCESSED_CSV),
            output_csv=str(ENRICHED_CSV),
            verbose=False,
        )
        result = run_enrichment(cfg)
        enriched_cols = len(pd.read_csv(result.output_path).columns)
        assert enriched_cols > original_cols

    def test_enriched_csv_preserves_original_rows(self):
        if not PROCESSED_CSV.exists():
            pytest.skip(f"processed_data.csv not found at {PROCESSED_CSV}")
        original_rows = len(pd.read_csv(PROCESSED_CSV))
        cfg = EnricherConfig(
            input_csv=str(PROCESSED_CSV),
            output_csv=str(ENRICHED_CSV),
            verbose=False,
        )
        result = run_enrichment(cfg)
        enriched = pd.read_csv(result.output_path)
        assert len(enriched) == original_rows

    def test_enriched_csv_has_zone_columns(self):
        if not PROCESSED_CSV.exists():
            pytest.skip(f"processed_data.csv not found at {PROCESSED_CSV}")
        cfg = EnricherConfig(
            input_csv=str(PROCESSED_CSV),
            output_csv=str(ENRICHED_CSV),
            verbose=False,
        )
        result = run_enrichment(cfg)
        df = pd.read_csv(result.output_path)
        assert "zone_name" in df.columns
        assert "zone_code" in df.columns

    def test_enriched_csv_has_weather_columns(self):
        if not PROCESSED_CSV.exists():
            pytest.skip(f"processed_data.csv not found at {PROCESSED_CSV}")
        cfg = EnricherConfig(
            input_csv=str(PROCESSED_CSV),
            output_csv=str(ENRICHED_CSV),
            verbose=False,
        )
        result = run_enrichment(cfg)
        df = pd.read_csv(result.output_path)
        for col in ["rain_risk_score", "is_ne_monsoon", "weather_condition"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_enriched_csv_has_traffic_columns(self):
        if not PROCESSED_CSV.exists():
            pytest.skip(f"processed_data.csv not found at {PROCESSED_CSV}")
        cfg = EnricherConfig(
            input_csv=str(PROCESSED_CSV),
            output_csv=str(ENRICHED_CSV),
            verbose=False,
        )
        result = run_enrichment(cfg)
        df = pd.read_csv(result.output_path)
        for col in ["congestion_level", "congestion_index", "travel_time_ratio"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_enrichment_result_columns_added_list(self):
        if not PROCESSED_CSV.exists():
            pytest.skip(f"processed_data.csv not found at {PROCESSED_CSV}")
        cfg = EnricherConfig(
            input_csv=str(PROCESSED_CSV),
            output_csv=str(ENRICHED_CSV),
            verbose=False,
        )
        result = run_enrichment(cfg)
        assert isinstance(result.columns_added, list)
        assert len(result.columns_added) > 0

    def test_enricher_handles_missing_input_gracefully(self):
        cfg = EnricherConfig(
            input_csv="output/nonexistent.csv",
            output_csv=str(ENRICHED_CSV),
            verbose=False,
        )
        with pytest.raises(FileNotFoundError):
            run_enrichment(cfg)


# ── Helper: OWM mock responses ────────────────────────────────────────────────

def _mock_owm_rain_response() -> dict:
    return {
        "weather": [{"main": "Rain", "description": "moderate rain"}],
        "main": {"temp": 28.0, "feels_like": 32.0, "humidity": 85},
        "wind": {"speed": 5.0},
        "visibility": 5000,
        "rain": {"1h": 6.0},
        "dt": 1700000000,
    }


def _mock_owm_clear_response() -> dict:
    return {
        "weather": [{"main": "Clear", "description": "clear sky"}],
        "main": {"temp": 34.0, "feels_like": 38.0, "humidity": 60},
        "wind": {"speed": 3.0},
        "visibility": 10000,
        "dt": 1700000000,
    }
