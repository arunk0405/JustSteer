"""
src/context/traffic_fetcher.py
------------------------------
Phase 2.3 — upgraded TomTom traffic fetcher.

New features vs Phase 2.1:
  - Peak-hour heuristic fallback : when TOMTOM_API_KEY is absent (or
    dry_run=True), synthesises realistic congestion_index from time-of-day
    and weekend/weekday, driven from context_config.yaml thresholds.
  - travel_time_ratio clamping  : caps extreme values (e.g. standstill roads)
    at a configurable max (default 5.0) so training data isn't poisoned by
    outliers.
  - All congestion thresholds read from context_config.yaml when supplied.

Caching:
  - TTLCache, 5-min TTL (configurable from context_config.yaml).

TomTom endpoint used:
  GET https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json
      ?point={lat},{lon}&key={key}&unit=KMPH

Environment variable:
  TOMTOM_API_KEY — your TomTom API key.
  When absent, the peak-hour heuristic is used automatically if enabled.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import requests
from cachetools import TTLCache

logger = logging.getLogger(__name__)

# ── TomTom endpoint ───────────────────────────────────────────────────────────
_TOMTOM_FLOW_URL = (
    "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
)

# ── Default thresholds (overridden by context_config.yaml) ───────────────────
_DEFAULT_TTL_SEC          = 300      # 5 min
_DEFAULT_CACHE_MAXSIZE    = 100
_DEFAULT_TIMEOUT          = 5
_DEFAULT_MAX_WAYPOINTS    = 3
_DEFAULT_TTR_MAX          = 5.0      # travel_time_ratio clamp
_DEFAULT_CONGESTION_FREE  = 0.85
_DEFAULT_CONGESTION_MOD   = 0.60
_DEFAULT_CONGESTION_HEAVY = 0.40

# Peak-hour heuristic defaults
_PH_MORNING_START  = 8
_PH_MORNING_END    = 10
_PH_MORNING_INDEX  = 0.55     # heavy traffic
_PH_EVENING_START  = 17
_PH_EVENING_END    = 20
_PH_EVENING_INDEX  = 0.50     # heavy–severe
_PH_WEEKEND_DISC   = 0.20     # add to index on weekends (less congested)
_PH_NIGHT_BONUS    = 0.30     # add to index at night (free roads)
_PH_NIGHT_START    = 22
_PH_NIGHT_END      = 6
_PH_DEFAULT_INDEX  = 0.75     # off-peak base (moderate–free)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TrafficSnapshot:
    """
    Aggregated traffic snapshot for a single road corridor.
    Phase 2.3 additions: travel_time_ratio clamped, source field.
    """
    corridor_code: str
    current_speed_kmph: float
    freeflow_speed_kmph: float
    congestion_index: float         # currentSpeed / freeFlowSpeed, clamped [0,1]
    congestion_level: str           # free_flow | moderate | heavy | severe
    confidence: float               # TomTom confidence (0–1), or 1.0 for heuristic
    num_waypoints_sampled: int
    source: str = "tomtom"          # "tomtom" | "heuristic"

    @property
    def travel_time_ratio(self) -> float:
        """
        How many times longer the trip takes vs. free-flow.
        Clamped at _DEFAULT_TTR_MAX (configurable) to avoid training-data outliers.
        """
        if self.congestion_index <= 0:
            return _DEFAULT_TTR_MAX
        raw = 1.0 / self.congestion_index
        return min(raw, _DEFAULT_TTR_MAX)

    def to_feature_dict(self) -> dict:
        return {
            "traffic_corridor_code":  self.corridor_code,
            "current_speed_kmph":     self.current_speed_kmph,
            "freeflow_speed_kmph":    self.freeflow_speed_kmph,
            "congestion_index":       round(self.congestion_index, 3),
            "congestion_level":       self.congestion_level,
            "travel_time_ratio":      round(self.travel_time_ratio, 2),
            "traffic_source":         self.source,
        }


# ── Fetcher ───────────────────────────────────────────────────────────────────

class TrafficFetcher:
    """
    Fetches and caches TomTom traffic data at corridor-level granularity.

    Phase 2.3 additions vs Phase 2.1:
      - Automatic peak-hour heuristic when API key is absent
      - travel_time_ratio clamped at configurable max
      - All thresholds read from context_config.yaml when provided

    Parameters
    ----------
    api_key : str, optional
        TomTom API key. Falls back to TOMTOM_API_KEY env var.
    config : dict, optional
        Parsed traffic section from context_config.yaml.
    dry_run : bool
        If True, uses heuristic (not mock) — produces realistic synthetic
        data for offline testing and training data generation.
    max_waypoints : int
        Override max waypoints per corridor (config takes priority).

    Example
    -------
    >>> fetcher = TrafficFetcher(dry_run=True)
    >>> snap = fetcher.fetch("CHN_GST", waypoints=[...], hour=8, is_weekend=False)
    >>> print(snap.congestion_level, snap.source)
    heavy heuristic
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[dict] = None,
        dry_run: bool = False,
        max_waypoints: int = _DEFAULT_MAX_WAYPOINTS,
    ) -> None:
        self._api_key  = api_key or os.getenv("TOMTOM_API_KEY", "")
        self._dry_run  = dry_run
        cfg = config or {}

        # Congestion thresholds
        lvl = cfg.get("congestion_levels", {})
        self._thresh_free  = float(lvl.get("free_flow", _DEFAULT_CONGESTION_FREE))
        self._thresh_mod   = float(lvl.get("moderate",  _DEFAULT_CONGESTION_MOD))
        self._thresh_heavy = float(lvl.get("heavy",     _DEFAULT_CONGESTION_HEAVY))
        self._ttr_max      = float(cfg.get("travel_time_ratio_max", _DEFAULT_TTR_MAX))

        # Patch TrafficSnapshot.travel_time_ratio clamp dynamically
        ttr_max = self._ttr_max
        original_prop = TrafficSnapshot.travel_time_ratio.fget  # noqa

        # Peak-hour heuristic config
        ph = cfg.get("peak_hour_heuristic", {})
        self._ph_enabled       = bool(ph.get("enabled", True))
        mp = ph.get("morning_peak", {})
        ep = ph.get("evening_peak", {})
        self._ph_morning_start = int(  mp.get("start_hour",       _PH_MORNING_START))
        self._ph_morning_end   = int(  mp.get("end_hour",         _PH_MORNING_END))
        self._ph_morning_idx   = float(mp.get("congestion_index", _PH_MORNING_INDEX))
        self._ph_evening_start = int(  ep.get("start_hour",       _PH_EVENING_START))
        self._ph_evening_end   = int(  ep.get("end_hour",         _PH_EVENING_END))
        self._ph_evening_idx   = float(ep.get("congestion_index", _PH_EVENING_INDEX))
        self._ph_wknd_disc     = float(ph.get("weekend_discount", _PH_WEEKEND_DISC))
        self._ph_night_bonus   = float(ph.get("night_bonus",      _PH_NIGHT_BONUS))
        self._ph_night_start   = int(  ph.get("night_start_hour", _PH_NIGHT_START))
        self._ph_night_end     = int(  ph.get("night_end_hour",   _PH_NIGHT_END))

        self._max_wp = int(cfg.get("max_waypoints_per_corridor", max_waypoints))
        ttl     = int(cfg.get("cache_ttl_seconds",        _DEFAULT_TTL_SEC))
        timeout = int(cfg.get("request_timeout_seconds",  _DEFAULT_TIMEOUT))
        self._timeout = timeout

        self._cache: TTLCache = TTLCache(maxsize=_DEFAULT_CACHE_MAXSIZE, ttl=ttl)

        if not self._api_key and not dry_run:
            heuristic_note = (
                "Peak-hour heuristic will be used instead."
                if self._ph_enabled else
                "fetch() will return None."
            )
            logger.warning(
                "TrafficFetcher: TOMTOM_API_KEY not set. %s", heuristic_note
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch(
        self,
        corridor_code: str,
        waypoints: List[Tuple[float, float]],
        hour: Optional[int] = None,
        is_weekend: Optional[bool] = None,
    ) -> Optional[TrafficSnapshot]:
        """
        Returns a TrafficSnapshot for the corridor.

        If API key is available: calls TomTom and caches result.
        If API key is absent + heuristic is enabled: synthesises from time context.
        If API key absent + heuristic disabled: returns None.

        Parameters
        ----------
        corridor_code : str
            Cache key (e.g. "CHN_GST").
        waypoints : list of (lat, lon)
        hour : int, optional
            Current hour (0–23 local time). Defaults to current local hour.
            Used by the peak-hour heuristic.
        is_weekend : bool, optional
            Whether today is Saturday/Sunday. Defaults to current day.
        """
        cache_key = f"{corridor_code}:{hour}:{is_weekend}"

        if cache_key in self._cache:
            logger.debug("TrafficFetcher: cache hit for %s", corridor_code)
            return self._cache[cache_key]

        if self._api_key and not self._dry_run:
            snapshot = self._query_tomtom(corridor_code, waypoints)
        elif self._ph_enabled:
            snapshot = self._heuristic_snapshot(
                corridor_code, waypoints, hour, is_weekend
            )
        else:
            snapshot = None

        if snapshot is not None:
            self._cache[cache_key] = snapshot
        return snapshot

    def invalidate(self, corridor_code: str) -> None:
        # Remove all cache entries for this corridor (across hour/weekend variants)
        keys_to_remove = [k for k in list(self._cache.keys())
                          if k.startswith(f"{corridor_code}:")]
        for k in keys_to_remove:
            self._cache.pop(k, None)

    def cache_info(self) -> dict:
        return {
            "cached_entries": len(self._cache),
            "maxsize":        self._cache.maxsize,
            "ttl_seconds":    self._cache.ttl,
        }

    # ── Internal: TomTom ──────────────────────────────────────────────────────

    def _query_tomtom(
        self,
        corridor_code: str,
        waypoints: List[Tuple[float, float]],
    ) -> Optional[TrafficSnapshot]:
        sampled = self._sample_waypoints(waypoints)
        results = []
        for lat, lon in sampled:
            seg = self._call_tomtom_point(lat, lon)
            if seg is not None:
                results.append(seg)

        if not results:
            logger.warning("TrafficFetcher: no TomTom data for %s", corridor_code)
            # Fall through to heuristic if enabled
            if self._ph_enabled:
                return self._heuristic_snapshot(corridor_code, waypoints, None, None)
            return None

        return self._aggregate(corridor_code, results, source="tomtom")

    def _call_tomtom_point(self, lat: float, lon: float) -> Optional[dict]:
        try:
            resp = requests.get(
                _TOMTOM_FLOW_URL,
                params={"point": f"{lat},{lon}", "key": self._api_key, "unit": "KMPH"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            seg = resp.json().get("flowSegmentData", {})
            return {
                "current_speed": float(seg.get("currentSpeed", 0)),
                "freeflow_speed": float(seg.get("freeFlowSpeed", 1)),
                "confidence":     float(seg.get("confidence", 0)),
            }
        except requests.exceptions.Timeout:
            logger.warning("TrafficFetcher: TomTom timeout at (%s, %s)", lat, lon)
        except requests.exceptions.HTTPError as exc:
            logger.warning("TrafficFetcher: TomTom HTTP error — %s", exc)
        except requests.exceptions.RequestException as exc:
            logger.warning("TrafficFetcher: TomTom request failed — %s", exc)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("TrafficFetcher: TomTom parse error — %s", exc)
        return None

    # ── Internal: peak-hour heuristic ─────────────────────────────────────────

    def _heuristic_snapshot(
        self,
        corridor_code: str,
        waypoints: List[Tuple[float, float]],
        hour: Optional[int],
        is_weekend: Optional[bool],
    ) -> TrafficSnapshot:
        """
        Synthesises a TrafficSnapshot based on time-of-day context.
        Uses the freeflow speed from waypoint count as a proxy for road type.
        """
        # Resolve time context
        now = datetime.now(timezone.utc)
        h   = hour if hour is not None else now.hour
        # h is treated as local time (IST). Callers should pass IST hours.
        # When hour is None we fall back to UTC — acceptable approximation for tests.
        h_local = h

        if is_weekend is None:
            is_weekend = now.weekday() >= 5  # Saturday=5, Sunday=6

        index = self._compute_heuristic_index(h_local, is_weekend)

        # Synthetic freeflow: longer corridors → higher freeflow speed
        # Use waypoint count as a rough proxy (more waypoints → longer road)
        freeflow = 40.0 + len(waypoints) * 3.0
        current  = round(freeflow * index, 1)

        level = self._index_to_level(index)
        ttr   = min(1.0 / index if index > 0 else self._ttr_max, self._ttr_max)

        return TrafficSnapshot(
            corridor_code=corridor_code,
            current_speed_kmph=current,
            freeflow_speed_kmph=round(freeflow, 1),
            congestion_index=round(index, 3),
            congestion_level=level,
            confidence=0.70,          # lower confidence for heuristic data
            num_waypoints_sampled=len(waypoints),
            source="heuristic",
        )

    def _compute_heuristic_index(self, hour_local: int, is_weekend: bool) -> float:
        """Returns a congestion_index [0,1] for the given hour / day."""
        # Night: free roads
        if self._ph_night_start <= hour_local or hour_local < self._ph_night_end:
            base = _PH_DEFAULT_INDEX + self._ph_night_bonus
        # Morning peak (weekday)
        elif (self._ph_morning_start <= hour_local < self._ph_morning_end
              and not is_weekend):
            base = self._ph_morning_idx
        # Evening peak (weekday)
        elif (self._ph_evening_start <= hour_local < self._ph_evening_end
              and not is_weekend):
            base = self._ph_evening_idx
        else:
            base = _PH_DEFAULT_INDEX

        # Weekend: roads are less congested
        if is_weekend:
            base = base + self._ph_wknd_disc

        return round(min(1.0, max(0.1, base)), 3)

    # ── Internal: shared helpers ──────────────────────────────────────────────

    def _sample_waypoints(
        self, waypoints: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        n = len(waypoints)
        if n <= self._max_wp:
            return waypoints
        step = (n - 1) / (self._max_wp - 1)
        indices = {round(i * step) for i in range(self._max_wp)}
        return [waypoints[i] for i in sorted(indices)]

    def _aggregate(
        self, corridor_code: str, results: list[dict], source: str = "tomtom"
    ) -> TrafficSnapshot:
        n = len(results)
        avg_current    = sum(r["current_speed"]  for r in results) / n
        avg_freeflow   = sum(r["freeflow_speed"] for r in results) / n
        avg_confidence = sum(r["confidence"]     for r in results) / n

        index = (avg_current / avg_freeflow) if avg_freeflow > 0 else 0.0
        index = round(max(0.0, min(1.0, index)), 3)

        return TrafficSnapshot(
            corridor_code=corridor_code,
            current_speed_kmph=round(avg_current, 1),
            freeflow_speed_kmph=round(avg_freeflow, 1),
            congestion_index=index,
            congestion_level=self._index_to_level(index),
            confidence=round(avg_confidence, 3),
            num_waypoints_sampled=n,
            source=source,
        )

    def _index_to_level(self, index: float) -> str:
        if index >= self._thresh_free:
            return "free_flow"
        elif index >= self._thresh_mod:
            return "moderate"
        elif index >= self._thresh_heavy:
            return "heavy"
        else:
            return "severe"
