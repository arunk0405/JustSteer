"""
src/context/weather_fetcher.py
------------------------------
Phase 2.2 — upgraded OWM weather fetcher.

New features vs Phase 2.1:
  - rain_risk_score : float 0.0–1.0 derived from rain_1h_mm + condition
  - is_ne_monsoon   : bool True during October–December (NE monsoon season for
                      Tamil Nadu / Chennai), with configurable risk multiplier
  - hourly_forecast : list of HourlyForecast snapshots (up to N hours ahead)
                      fetched from the OWM Forecast API in one extra API call
  - All thresholds driven from configs/context_config.yaml via ContextConfig

Caching:
  - Current conditions: TTLCache, 15 min TTL (configurable)
  - Hourly forecast:    TTLCache, 60 min TTL (forecast changes slowly)

OWM endpoints used:
  Current: GET https://api.openweathermap.org/data/2.5/weather
  Forecast: GET https://api.openweathermap.org/data/2.5/forecast
            (3-hourly slots; we take first N slots ≈ forecast_hours)

Environment variable:
  OWM_API_KEY — your OpenWeatherMap API key.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import requests
from cachetools import TTLCache

logger = logging.getLogger(__name__)

# ── OWM endpoints ─────────────────────────────────────────────────────────────
_OWM_CURRENT_URL  = "https://api.openweathermap.org/data/2.5/weather"
_OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
_REQUEST_TIMEOUT  = 5

# ── Default thresholds (overridden by context_config.yaml) ───────────────────
_DEFAULT_RAIN_LOW_MM    = 2.5
_DEFAULT_RAIN_MOD_MM    = 7.5
_DEFAULT_RAIN_HIGH_MM   = 15.0
_DEFAULT_FOG_VIS_M      = 1000
_DEFAULT_WIND_MPS       = 10.0
_DEFAULT_NE_START       = 10    # October
_DEFAULT_NE_END         = 12    # December
_DEFAULT_NE_MULTIPLIER  = 1.5
_DEFAULT_FORECAST_HOURS = 6
_CURRENT_TTL_SEC        = 900   # 15 min
_FORECAST_TTL_SEC       = 3600  # 60 min
_CACHE_MAXSIZE          = 200


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class HourlyForecast:
    """A single 3-hourly forecast slot from OWM."""
    valid_at: datetime          # UTC timestamp of this forecast slot
    condition: str
    temp_celsius: float
    rain_3h_mm: float           # mm of rain in the 3-hour window
    wind_speed_mps: float
    humidity_pct: int

    @property
    def is_rainy(self) -> bool:
        return self.condition in {"Rain", "Drizzle", "Thunderstorm"}


@dataclass
class WeatherSnapshot:
    """
    Parsed weather snapshot for a single zone.
    Phase 2.2 additions: rain_risk_score, is_ne_monsoon, hourly_forecast.
    """
    zone_code: str
    condition: str              # OWM main condition: Rain, Clear, Clouds, etc.
    description: str            # OWM description: "moderate rain", etc.
    temp_celsius: float
    feels_like_celsius: float
    humidity_pct: int
    wind_speed_mps: float
    visibility_m: int
    rain_1h_mm: float
    is_raining: bool
    is_foggy: bool
    is_high_wind: bool

    # ── Phase 2.2 additions ────────────────────────────────────────────────────
    rain_risk_score: float = 0.0    # 0.0 (none) → 1.0 (severe)
    is_ne_monsoon: bool = False     # True during Oct–Dec
    hourly_forecast: List[HourlyForecast] = field(default_factory=list)

    @property
    def risk_flags(self) -> list[str]:
        flags = []
        if self.is_raining:
            flags.append(f"rain ({self.rain_1h_mm:.1f} mm/h, risk={self.rain_risk_score:.2f})")
        if self.is_foggy:
            flags.append(f"fog (visibility {self.visibility_m} m)")
        if self.is_high_wind:
            flags.append(f"high_wind ({self.wind_speed_mps:.1f} m/s)")
        if self.is_ne_monsoon:
            flags.append("NE_monsoon_season")
        return flags

    @property
    def rain_forecast_hours(self) -> int:
        """How many of the upcoming forecast slots have rain."""
        return sum(1 for h in self.hourly_forecast if h.is_rainy)

    def to_feature_dict(self) -> dict:
        """Flat dict for a DataFrame row. Includes all Phase 2.2 fields."""
        return {
            "weather_zone_code":      self.zone_code,
            "weather_condition":      self.condition,
            "temp_celsius":           self.temp_celsius,
            "humidity_pct":           self.humidity_pct,
            "wind_speed_mps":         self.wind_speed_mps,
            "visibility_m":           self.visibility_m,
            "rain_1h_mm":             self.rain_1h_mm,
            "rain_risk_score":        round(self.rain_risk_score, 3),
            "is_raining":             int(self.is_raining),
            "is_foggy":               int(self.is_foggy),
            "is_high_wind":           int(self.is_high_wind),
            "is_ne_monsoon":          int(self.is_ne_monsoon),
            "rain_forecast_hours":    self.rain_forecast_hours,
        }


# ── Fetcher ───────────────────────────────────────────────────────────────────

class WeatherFetcher:
    """
    Fetches and caches OWM weather data at zone-level granularity.

    Phase 2.2 additions vs Phase 2.1:
      - Computes rain_risk_score (0–1) from rain intensity + NE monsoon flag
      - Fetches hourly forecast (up to forecast_hours ahead) in one extra call
      - All thresholds read from context_config.yaml when provided

    Parameters
    ----------
    api_key : str, optional
        OWM API key. Falls back to OWM_API_KEY environment variable.
    config : dict, optional
        Parsed weather section from context_config.yaml. If None, uses defaults.
    dry_run : bool
        If True, skip HTTP calls and return mock data (tests/offline use).

    Example
    -------
    >>> fetcher = WeatherFetcher(api_key="abc123")
    >>> snap = fetcher.fetch("CHN_ADYAR", lat=13.0067, lon=80.2572)
    >>> print(snap.rain_risk_score, snap.is_ne_monsoon, snap.rain_forecast_hours)
    0.72 True 3
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[dict] = None,
        dry_run: bool = False,
    ) -> None:
        self._api_key = api_key or os.getenv("OWM_API_KEY", "")
        self._dry_run = dry_run
        cfg = config or {}

        # Thresholds — fall back to defaults if config not provided
        rain_cfg = cfg.get("rain_risk", {})
        self._rain_low   = float(rain_cfg.get("low_mm_per_hour",      _DEFAULT_RAIN_LOW_MM))
        self._rain_mod   = float(rain_cfg.get("moderate_mm_per_hour", _DEFAULT_RAIN_MOD_MM))
        self._rain_high  = float(rain_cfg.get("high_mm_per_hour",     _DEFAULT_RAIN_HIGH_MM))
        self._fog_vis    = int(  cfg.get("fog_visibility_threshold_m", _DEFAULT_FOG_VIS_M))
        self._wind_mps   = float(cfg.get("high_wind_threshold_mps",    _DEFAULT_WIND_MPS))

        ne_cfg = cfg.get("ne_monsoon", {})
        self._ne_start  = int(  ne_cfg.get("start_month",      _DEFAULT_NE_START))
        self._ne_end    = int(  ne_cfg.get("end_month",         _DEFAULT_NE_END))
        self._ne_mult   = float(ne_cfg.get("risk_multiplier",  _DEFAULT_NE_MULTIPLIER))

        self._forecast_hours = int(cfg.get("forecast_hours", _DEFAULT_FORECAST_HOURS))
        ttl = int(cfg.get("cache_ttl_seconds", _CURRENT_TTL_SEC))
        timeout = int(cfg.get("request_timeout_seconds", _REQUEST_TIMEOUT))
        self._timeout = timeout

        self._cache: TTLCache = TTLCache(maxsize=_CACHE_MAXSIZE, ttl=ttl)
        self._forecast_cache: TTLCache = TTLCache(
            maxsize=_CACHE_MAXSIZE, ttl=_FORECAST_TTL_SEC
        )

        if not self._api_key and not dry_run:
            logger.warning(
                "WeatherFetcher: OWM_API_KEY not set — fetch() returns None. "
                "Set OWM_API_KEY or use dry_run=True for offline use."
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch(
        self,
        zone_code: str,
        lat: float,
        lon: float,
        include_forecast: bool = True,
    ) -> Optional[WeatherSnapshot]:
        """
        Returns a WeatherSnapshot, from cache or OWM. Returns None on API error.

        Parameters
        ----------
        zone_code : str
            Cache key (e.g. "CHN_ADYAR").
        lat, lon : float
            Zone centroid coordinates.
        include_forecast : bool
            Whether to also fetch the hourly forecast. Default True.
            Set False to halve API calls when forecast is not needed.
        """
        if zone_code in self._cache:
            logger.debug("WeatherFetcher: cache hit (current) for %s", zone_code)
            return self._cache[zone_code]

        if self._dry_run:
            snapshot = self._mock_snapshot(zone_code)
        else:
            snapshot = self._call_owm_current(zone_code, lat, lon)

        if snapshot is None:
            return None

        # Attach hourly forecast
        if include_forecast:
            forecast = self._fetch_forecast(zone_code, lat, lon)
            snapshot.hourly_forecast = forecast

        self._cache[zone_code] = snapshot
        return snapshot

    def invalidate(self, zone_code: str) -> None:
        self._cache.pop(zone_code, None)
        self._forecast_cache.pop(zone_code, None)

    def cache_info(self) -> dict:
        return {
            "current_cached_zones":  len(self._cache),
            "forecast_cached_zones": len(self._forecast_cache),
            "ttl_current_sec":       self._cache.ttl,
            "ttl_forecast_sec":      self._forecast_cache.ttl,
        }

    # ── Internal: current weather ──────────────────────────────────────────────

    def _call_owm_current(
        self, zone_code: str, lat: float, lon: float
    ) -> Optional[WeatherSnapshot]:
        if not self._api_key:
            return None
        try:
            resp = requests.get(
                _OWM_CURRENT_URL,
                params={"lat": lat, "lon": lon, "appid": self._api_key, "units": "metric"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return self._parse_current(zone_code, resp.json())
        except requests.exceptions.Timeout:
            logger.warning("WeatherFetcher: OWM current timed out for %s", zone_code)
        except requests.exceptions.HTTPError as exc:
            logger.warning("WeatherFetcher: OWM HTTP error for %s — %s", zone_code, exc)
        except requests.exceptions.RequestException as exc:
            logger.warning("WeatherFetcher: OWM request failed for %s — %s", zone_code, exc)
        except (KeyError, ValueError) as exc:
            logger.warning("WeatherFetcher: parse error for %s — %s", zone_code, exc)
        return None

    def _parse_current(self, zone_code: str, data: dict) -> WeatherSnapshot:
        weather_list = data.get("weather", [{}])
        condition   = weather_list[0].get("main", "Unknown")
        description = weather_list[0].get("description", "")
        main        = data.get("main", {})
        wind        = data.get("wind", {})
        rain        = data.get("rain", {})

        wind_speed  = float(wind.get("speed", 0.0))
        visibility  = int(data.get("visibility", 10000))
        rain_1h     = float(rain.get("1h", 0.0))
        is_raining  = condition in {"Rain", "Drizzle", "Thunderstorm"}

        # Determine NE monsoon from current UTC month
        now_month   = datetime.now(timezone.utc).month
        is_ne       = self._ne_start <= now_month <= self._ne_end

        rain_risk   = self._compute_rain_risk(rain_1h, condition, is_ne)

        return WeatherSnapshot(
            zone_code=zone_code,
            condition=condition,
            description=description,
            temp_celsius=float(main.get("temp", 0.0)),
            feels_like_celsius=float(main.get("feels_like", 0.0)),
            humidity_pct=int(main.get("humidity", 0)),
            wind_speed_mps=wind_speed,
            visibility_m=visibility,
            rain_1h_mm=rain_1h,
            is_raining=is_raining,
            is_foggy=visibility < self._fog_vis,
            is_high_wind=wind_speed > self._wind_mps,
            rain_risk_score=rain_risk,
            is_ne_monsoon=is_ne,
        )

    # ── Internal: hourly forecast ──────────────────────────────────────────────

    def _fetch_forecast(
        self, zone_code: str, lat: float, lon: float
    ) -> List[HourlyForecast]:
        if zone_code in self._forecast_cache:
            logger.debug("WeatherFetcher: cache hit (forecast) for %s", zone_code)
            return self._forecast_cache[zone_code]

        if self._dry_run:
            slots = self._mock_forecast(zone_code)
        else:
            slots = self._call_owm_forecast(zone_code, lat, lon)

        self._forecast_cache[zone_code] = slots
        return slots

    def _call_owm_forecast(
        self, zone_code: str, lat: float, lon: float
    ) -> List[HourlyForecast]:
        if not self._api_key:
            return []
        # OWM forecast gives 3-hourly slots; CNT controls how many we fetch.
        # ceil(forecast_hours / 3) slots covers the requested window.
        cnt = max(1, -(-self._forecast_hours // 3))  # ceiling division
        try:
            resp = requests.get(
                _OWM_FORECAST_URL,
                params={
                    "lat": lat, "lon": lon,
                    "appid": self._api_key,
                    "units": "metric",
                    "cnt": cnt,
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return self._parse_forecast(resp.json(), cnt)
        except requests.exceptions.Timeout:
            logger.warning("WeatherFetcher: OWM forecast timed out for %s", zone_code)
        except requests.exceptions.HTTPError as exc:
            logger.warning("WeatherFetcher: OWM forecast HTTP error for %s — %s", zone_code, exc)
        except requests.exceptions.RequestException as exc:
            logger.warning("WeatherFetcher: OWM forecast request failed — %s", exc)
        except (KeyError, ValueError) as exc:
            logger.warning("WeatherFetcher: forecast parse error for %s — %s", zone_code, exc)
        return []

    @staticmethod
    def _parse_forecast(data: dict, max_slots: int) -> List[HourlyForecast]:
        slots = []
        for item in data.get("list", [])[:max_slots]:
            weather = item.get("weather", [{}])[0]
            main    = item.get("main", {})
            rain    = item.get("rain", {})
            dt      = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
            slots.append(HourlyForecast(
                valid_at=dt,
                condition=weather.get("main", "Unknown"),
                temp_celsius=float(main.get("temp", 0.0)),
                rain_3h_mm=float(rain.get("3h", 0.0)),
                wind_speed_mps=float(item.get("wind", {}).get("speed", 0.0)),
                humidity_pct=int(main.get("humidity", 0)),
            ))
        return slots

    # ── Internal: rain risk scoring ────────────────────────────────────────────

    def _compute_rain_risk(
        self, rain_mm: float, condition: str, is_ne_monsoon: bool
    ) -> float:
        """
        Returns a 0–1 risk score:
          0.00       — no rain, clear conditions
          0.25       — drizzle / low rain (< low_mm threshold)
          0.50       — moderate rain
          0.75       — heavy rain
          1.00       — severe / thunderstorm
        Multiplied by ne_multiplier during NE monsoon season (clamped to 1.0).
        """
        if condition == "Thunderstorm":
            base = 1.0
        elif condition in {"Rain", "Drizzle"}:
            if rain_mm >= self._rain_high:
                base = 0.75
            elif rain_mm >= self._rain_mod:
                base = 0.50
            elif rain_mm >= self._rain_low:
                base = 0.25
            else:
                base = 0.15   # condition says rain but mm is low (drizzle)
        else:
            base = 0.0

        if is_ne_monsoon and base > 0:
            base = min(1.0, base * self._ne_mult)

        return round(base, 3)

    # ── Mock data (dry_run / tests) ────────────────────────────────────────────

    def _mock_snapshot(self, zone_code: str) -> WeatherSnapshot:
        now_month = datetime.now(timezone.utc).month
        is_ne     = self._ne_start <= now_month <= self._ne_end
        return WeatherSnapshot(
            zone_code=zone_code,
            condition="Clear",
            description="clear sky (mock)",
            temp_celsius=32.0,
            feels_like_celsius=36.0,
            humidity_pct=72,
            wind_speed_mps=3.5,
            visibility_m=10000,
            rain_1h_mm=0.0,
            is_raining=False,
            is_foggy=False,
            is_high_wind=False,
            rain_risk_score=0.0,
            is_ne_monsoon=is_ne,
        )

    @staticmethod
    def _mock_forecast(zone_code: str) -> List[HourlyForecast]:
        """Returns 2 mock forecast slots (clear sky)."""
        base = datetime.now(timezone.utc)
        return [
            HourlyForecast(
                valid_at=base,
                condition="Clear",
                temp_celsius=32.0,
                rain_3h_mm=0.0,
                wind_speed_mps=3.5,
                humidity_pct=72,
            ),
            HourlyForecast(
                valid_at=base,
                condition="Clouds",
                temp_celsius=31.0,
                rain_3h_mm=0.0,
                wind_speed_mps=4.0,
                humidity_pct=75,
            ),
        ]
