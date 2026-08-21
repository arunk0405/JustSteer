"""
sample_data/generate_synthetic_data.py
---------------------------------------
Generates a 50,000-row synthetic training dataset that matches the exact
schema of output/enriched_data.csv, with realistic Chennai-specific
traffic and weather distributions.

Output: sample_data/synthetic_training_data.csv
        (NEVER mixed into output/enriched_data.csv or processed_data.csv)

Run:
    python sample_data/generate_synthetic_data.py
    python sample_data/generate_synthetic_data.py --rows 50000 --seed 42
"""

from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Reproducibility ───────────────────────────────────────────────────────────
DEFAULT_SEED = 42
DEFAULT_ROWS = 50_000
OUTPUT_PATH  = Path("sample_data/synthetic_training_data.csv")

# ── Date range: synthetic data spans a full calendar year (2024) ─────────────
# Kept entirely in 2024 so there is zero timestamp overlap with real data
# collected from 2025 onward.
SIM_START = datetime(2024, 1, 1, 0, 0, 0)
SIM_END   = datetime(2024, 12, 31, 23, 59, 59)
SIM_RANGE_SEC = int((SIM_END - SIM_START).total_seconds())

# ── Road types (matching enriched_data.csv values) ───────────────────────────
ROAD_TYPES = ["highway", "arterial", "residential", "service"]
ROAD_TYPE_WEIGHTS = [0.20, 0.30, 0.30, 0.20]

# ── Time-of-day slots ─────────────────────────────────────────────────────────
# Derived from the enricher's time_column convention
def _hour_to_slot(hour: int) -> str:
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"

# ── Risk labels ───────────────────────────────────────────────────────────────
RISK_LABELS = ["low", "medium", "high"]

# ── Chennai zones (25 area zones from chennai.yaml) ──────────────────────────
ZONES = [
    # (name, code, type, tags, centroid_lat, centroid_lon)
    ("Anna Nagar",       "CHN_ANNANAGAR",      "residential", "high_density|waterlogging_risk",                    13.0850, 80.2101),
    ("Kilpauk",          "CHN_KILPAUK",         "residential", "hospital_zone|narrow_roads",                        13.0827, 80.2297),
    ("Perambur",         "CHN_PERAMBUR",        "industrial",  "railway_proximity|industrial",                      13.1166, 80.2352),
    ("Tondiarpet",       "CHN_TONDIARPET",      "residential", "coastal_proximity|dense_traffic",                   13.1189, 80.2947),
    ("Royapuram",        "CHN_ROYAPURAM",       "coastal",     "fishing_harbour|coastal|narrow_streets",            13.1114, 80.2966),
    ("Egmore",           "CHN_EGMORE",          "commercial",  "railway_station|high_footfall",                     13.0732, 80.2609),
    ("T. Nagar",         "CHN_TNAGAR",          "commercial",  "shopping_hub|extreme_congestion",                   13.0418, 80.2341),
    ("Nungambakkam",     "CHN_NUNGAMBAKKAM",    "commercial",  "office_zone|embassies",                             13.0569, 80.2425),
    ("Mylapore",         "CHN_MYLAPORE",        "residential", "heritage_area|temple_traffic",                      13.0368, 80.2676),
    ("Royapettah",       "CHN_ROYAPETTAH",      "commercial",  "hospital_zone|mixed_traffic",                       13.0510, 80.2630),
    ("Adyar",            "CHN_ADYAR",           "residential", "waterlogging_risk|river_estuary|upscale",           13.0067, 80.2572),
    ("Velachery",        "CHN_VELACHERY",       "residential", "waterlogging_risk|it_corridor_proximity|high_density", 12.9815, 80.2180),
    ("Pallikaranai",     "CHN_PALLIKARANAI",    "residential", "marsh_proximity|waterlogging_risk|new_development", 12.9383, 80.2132),
    ("Sholinganallur",   "CHN_SHOLINGANALLUR",  "commercial",  "it_corridor|office_parks|peak_hour_congestion",     12.9010, 80.2279),
    ("Perungudi",        "CHN_PERUNGUDI",       "commercial",  "it_corridor|tech_parks",                            12.9630, 80.2452),
    ("Chromepet",        "CHN_CHROMEPET",       "residential", "airport_proximity|leather_industry",                12.9516, 80.1462),
    ("Pallavaram",       "CHN_PALLAVARAM",      "residential", "airport_proximity|military_area",                   12.9675, 80.1491),
    ("Tambaram",         "CHN_TAMBARAM",        "residential", "suburban|railway_junction|gst_road_access",         12.9249, 80.1000),
    ("Ambattur",         "CHN_AMBATTUR",        "industrial",  "industrial_estate|logistics_hub|heavy_vehicles",    13.1143, 80.1548),
    ("Poonamallee",      "CHN_POONAMALLEE",     "residential", "highway_town|suburban",                             13.0474, 80.1162),
    ("Porur",            "CHN_PORUR",           "residential", "it_corridor_west|orr_junction",                     13.0358, 80.1567),
    ("Vadapalani",       "CHN_VADAPALANI",      "commercial",  "shopping|metro_connectivity",                       13.0521, 80.2121),
    ("Avadi",            "CHN_AVADI",           "industrial",  "defence_township|outer_suburb",                     13.1162, 80.1025),
    ("Madhavaram",       "CHN_MADHAVARAM",      "residential", "north_suburb|developing",                           13.1485, 80.2333),
    ("Thiruvottiyur",    "CHN_THIRUVOTTIYUR",   "industrial",  "port_proximity|industrial|coastal",                 13.1627, 80.3043),
]

# ── Corridors (10 road corridors from chennai.yaml) ──────────────────────────
CORRIDORS = [
    # (name, code, typical_freeflow_kmph)
    ("GST Road",                           "CHN_GST",            55),
    ("Outer Ring Road",                    "CHN_ORR",            65),
    ("Anna Salai (Mount Road)",            "CHN_ANNASALAI",      35),
    ("Inner Ring Road (IRR)",              "CHN_IRR",            40),
    ("Rajiv Gandhi Salai (IT Corridor / OMR)", "CHN_OMR",        60),
    ("ECR (East Coast Road)",              "CHN_ECR",            70),
    ("Poonamallee High Road",              "CHN_POONAMALLEE_HRD",38),
    ("NH 48 (Chennai–Bangalore Highway)",  "CHN_NH48",           80),
    ("Sardar Patel Road (Adyar–Guindy)",   "CHN_SPR",            35),
    ("Medavakkam–Sholinganallur Road",     "CHN_MEDSHO",         32),
]

# ── Weather conditions and base probabilities (Chennai climate) ───────────────
# Clear/Clouds dominate most of the year; Rain spikes Oct–Dec (NE monsoon)
WEATHER_CONDITIONS = ["Clear", "Clouds", "Rain", "Drizzle", "Thunderstorm", "Haze", "Mist", "Fog"]
WEATHER_BASE_PROBS = [0.25,    0.35,     0.15,   0.10,      0.05,           0.05,   0.03,   0.02]

# NE monsoon months: Oct, Nov, Dec — rain probability doubles
NE_MONSOON_MONTHS = {10, 11, 12}



# ═════════════════════════════════════════════════════════════════════════════
# HELPER: Weather generation
# ═════════════════════════════════════════════════════════════════════════════

def _weather_condition(month: int, rng: np.random.Generator) -> str:
    """Sample a weather condition realistic for Chennai in the given month."""
    probs = WEATHER_BASE_PROBS.copy()
    if month in NE_MONSOON_MONTHS:
        # Boost Rain + Drizzle + Thunderstorm during NE monsoon
        probs[2] += 0.20   # Rain
        probs[3] += 0.08   # Drizzle
        probs[4] += 0.05   # Thunderstorm
        probs[0] -= 0.15   # Less Clear
        probs[1] -= 0.18   # Less Clouds
    # Re-normalise
    total = sum(probs)
    probs = [p / total for p in probs]
    return rng.choice(WEATHER_CONDITIONS, p=probs)


def _weather_features(condition: str, month: int, hour: int,
                       rng: np.random.Generator) -> dict:
    """
    Generate correlated weather features for a given condition and time.
    Values are tuned to Chennai's actual climate norms.
    """
    is_ne_monsoon = 1 if month in NE_MONSOON_MONTHS else 0

    # ── Temperature (°C) ─────────────────────────────────────────────────────
    # Chennai range: ~24–42°C. Hottest Apr–Jun, coolest Jan–Feb.
    base_temp = {
        1: 28, 2: 29, 3: 32, 4: 35, 5: 37, 6: 35,
        7: 33, 8: 33, 9: 33, 10: 30, 11: 28, 12: 27,
    }[month]
    # Night cooler, afternoon hotter
    hour_delta = -2 if (22 <= hour or hour < 6) else (2 if 12 <= hour < 16 else 0)
    if condition in ("Rain", "Drizzle", "Thunderstorm"):
        base_temp -= 3
    temp = float(np.clip(rng.normal(base_temp + hour_delta, 1.5), 22, 43))

    # ── Humidity (%) ──────────────────────────────────────────────────────────
    # Chennai is coastal; typically 60–90%. Peaks during monsoon.
    base_hum = 75 if month in NE_MONSOON_MONTHS else 65
    if condition in ("Rain", "Drizzle", "Thunderstorm", "Mist", "Fog"):
        base_hum += 12
    humidity = int(np.clip(rng.normal(base_hum, 7), 40, 99))

    # ── Wind speed (m/s) ─────────────────────────────────────────────────────
    wind_mean = 7.0 if condition == "Thunderstorm" else 4.5
    wind_speed = float(np.clip(rng.exponential(wind_mean / 2.0), 0.5, 18.0))
    is_high_wind = 1 if wind_speed > 10.0 else 0

    # ── Visibility (m) ───────────────────────────────────────────────────────
    if condition == "Fog":
        visibility = int(rng.uniform(100, 800))
    elif condition == "Mist":
        visibility = int(rng.uniform(500, 2500))
    elif condition in ("Rain", "Thunderstorm"):
        visibility = int(rng.uniform(2000, 7000))
    else:
        visibility = int(rng.uniform(7000, 10000))
    is_foggy = 1 if visibility < 1000 else 0

    # ── Rain (mm/h) ───────────────────────────────────────────────────────────
    if condition == "Clear":
        rain_1h = 0.0
    elif condition == "Clouds":
        rain_1h = 0.0
    elif condition == "Haze":
        rain_1h = 0.0
    elif condition == "Drizzle":
        rain_1h = float(rng.uniform(0.2, 2.4))
    elif condition == "Mist":
        rain_1h = float(rng.uniform(0.0, 1.0))
    elif condition == "Fog":
        rain_1h = 0.0
    elif condition == "Rain":
        rain_1h = float(rng.uniform(2.5, 20.0))
    else:  # Thunderstorm
        rain_1h = float(rng.uniform(10.0, 40.0))
    is_raining = 1 if rain_1h > 0.1 else 0

    # ── Rain risk score (0–1) ────────────────────────────────────────────────
    if rain_1h == 0:
        base_score = 0.0
    elif rain_1h < 2.5:
        base_score = 0.1 + rain_1h / 25.0
    elif rain_1h < 7.5:
        base_score = 0.3 + (rain_1h - 2.5) / 25.0
    elif rain_1h < 15.0:
        base_score = 0.5 + (rain_1h - 7.5) / 30.0
    else:
        base_score = 0.75 + min(rain_1h / 160.0, 0.25)
    rain_risk_score = round(min(base_score * (1.5 if is_ne_monsoon else 1.0), 1.0), 3)

    # ── Forecast hours with rain (0–6, for 6-hour window) ───────────────────
    if condition in ("Rain", "Thunderstorm"):
        rain_forecast = int(rng.integers(2, 7))
    elif condition == "Drizzle":
        rain_forecast = int(rng.integers(0, 4))
    else:
        rain_forecast = int(rng.integers(0, 2))

    return {
        "weather_condition":  condition,
        "temp_celsius":       round(temp, 2),
        "humidity_pct":       humidity,
        "wind_speed_mps":     round(wind_speed, 2),
        "visibility_m":       visibility,
        "rain_1h_mm":         round(rain_1h, 2),
        "rain_risk_score":    rain_risk_score,
        "is_raining":         is_raining,
        "is_foggy":           is_foggy,
        "is_high_wind":       is_high_wind,
        "is_ne_monsoon":      is_ne_monsoon,
        "rain_forecast_hours": rain_forecast,
    }


# ═════════════════════════════════════════════════════════════════════════════
# HELPER: Traffic generation
# ═════════════════════════════════════════════════════════════════════════════

def _congestion_index(hour: int, is_weekend: int, road_type: str,
                       condition: str, rng: np.random.Generator) -> float:
    """
    Produce a realistic congestion index (0–1) based on time, day, and road.
    Mirrors the peak-hour heuristic in context_config.yaml.
    """
    # Base index by road type
    base = {
        "highway":    0.80,
        "arterial":   0.70,
        "residential":0.85,
        "service":    0.88,
    }[road_type]

    # Peak-hour penalty
    if 8 <= hour < 10 or 17 <= hour < 20:        # morning / evening peak
        base -= 0.25
    elif 22 <= hour or hour < 6:                  # night — freer roads
        base += 0.12
    elif 10 <= hour < 17:                         # midday
        base -= 0.05

    # Weekend relief
    if is_weekend:
        base += 0.18

    # Rain degrades speed
    if condition in ("Rain", "Thunderstorm"):
        base -= 0.15
    elif condition in ("Drizzle", "Fog", "Mist"):
        base -= 0.07

    # Add Gaussian noise and clamp
    index = float(np.clip(rng.normal(base, 0.07), 0.05, 1.0))
    return round(index, 3)


def _index_to_level(index: float) -> str:
    if index >= 0.85:
        return "free_flow"
    elif index >= 0.60:
        return "moderate"
    elif index >= 0.40:
        return "heavy"
    else:
        return "severe"


def _traffic_features(corridor_code: str, freeflow_kmph: int,
                       hour: int, is_weekend: int, road_type: str,
                       condition: str, rng: np.random.Generator) -> dict:
    """Generate correlated traffic features for a corridor + context."""
    congestion_index = _congestion_index(hour, is_weekend, road_type,
                                          condition, rng)
    congestion_level = _index_to_level(congestion_index)

    # current speed derived from index × freeflow, with small jitter
    current_speed = round(
        float(np.clip(
            rng.normal(congestion_index * freeflow_kmph, 2.5),
            5.0, freeflow_kmph * 1.05,
        )), 1
    )
    freeflow_speed = round(
        float(np.clip(rng.normal(freeflow_kmph, 3.0), freeflow_kmph * 0.8, freeflow_kmph * 1.2)),
        1
    )
    # travel_time_ratio = freeflow / current, clamped at 5.0
    travel_time_ratio = round(
        min(freeflow_speed / max(current_speed, 1.0), 5.0), 3
    )

    return {
        "traffic_corridor_code": corridor_code,
        "current_speed_kmph":    current_speed,
        "freeflow_speed_kmph":   freeflow_speed,
        "congestion_index":      congestion_index,
        "congestion_level":      congestion_level,
        "travel_time_ratio":     travel_time_ratio,
        "traffic_source":        "heuristic",
    }



# ═════════════════════════════════════════════════════════════════════════════
# HELPER: Risk label derivation
# ═════════════════════════════════════════════════════════════════════════════

def _derive_risk_label(congestion_index: float, rain_risk_score: float,
                        is_foggy: int, is_high_wind: int,
                        visibility_m: int, road_type: str) -> str:
    """
    Deterministically derive a risk label from the generated features so the
    label is consistent with the feature values (no random noise on the label).

    Thresholds mirror the enricher's intended semantics:
      low    — good conditions, free-flowing traffic
      medium — moderate congestion OR light rain
      high   — severe congestion OR heavy rain / fog / high wind
    """
    score = 0.0
    # Congestion contribution
    if congestion_index < 0.40:
        score += 2.0    # severe congestion
    elif congestion_index < 0.60:
        score += 1.0    # heavy
    elif congestion_index < 0.85:
        score += 0.3    # moderate

    # Rain contribution
    score += rain_risk_score * 2.0

    # Environmental hazards
    if is_foggy:
        score += 1.5
    if is_high_wind:
        score += 0.8
    if visibility_m < 2000:
        score += 0.5

    # Road type modifier (service roads have less overall risk tolerance)
    if road_type == "service":
        score += 0.2

    if score >= 2.5:
        return "high"
    elif score >= 1.0:
        return "medium"
    else:
        return "low"


# ═════════════════════════════════════════════════════════════════════════════
# HELPER: avg_speed_kmph (original Phase 1 feature, kept for schema match)
# ═════════════════════════════════════════════════════════════════════════════

def _avg_speed(road_type: str, hour: int, is_weekend: int,
               condition: str, rng: np.random.Generator) -> float:
    """
    Historical average speed for the road segment.
    This is the pre-enrichment column from processed_data.csv, kept as-is.
    """
    base = {"highway": 52, "arterial": 32, "residential": 28, "service": 20}[road_type]
    if 8 <= hour < 10 or 17 <= hour < 20:
        base = int(base * 0.60)
    elif 22 <= hour or hour < 6:
        base = int(base * 1.15)
    if is_weekend:
        base = int(base * 1.12)
    if condition in ("Rain", "Thunderstorm"):
        base = int(base * 0.80)
    return round(float(np.clip(rng.normal(base, base * 0.12), 5.0, 90.0)), 1)


# ═════════════════════════════════════════════════════════════════════════════
# HELPER: rainfall_mm (original Phase 1 column)
# ═════════════════════════════════════════════════════════════════════════════

def _rainfall_mm(rain_1h: float, rng: np.random.Generator) -> float:
    """Accumulated rainfall on the road segment (correlated with rain_1h_mm)."""
    if rain_1h == 0.0:
        return round(float(rng.uniform(0.0, 1.5)), 2)
    return round(float(np.clip(rng.normal(rain_1h * 0.9, rain_1h * 0.2), 0.0, 60.0)), 2)


# ═════════════════════════════════════════════════════════════════════════════
# ROW BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def _build_row(i: int, rng: np.random.Generator) -> dict:
    """Build a single synthetic row. All features are internally consistent."""

    # ── Timestamp ────────────────────────────────────────────────────────────
    # Distribute evenly across the 2024 calendar year
    offset_sec = int(rng.integers(0, SIM_RANGE_SEC))
    ts = SIM_START + timedelta(seconds=offset_sec)
    hour      = ts.hour
    dow       = ts.weekday()          # 0=Mon … 6=Sun
    is_weekend = 1 if dow >= 5 else 0
    month     = ts.month
    time_slot = _hour_to_slot(hour)

    # ── Zone and corridor ────────────────────────────────────────────────────
    zone_idx  = int(rng.integers(0, len(ZONES)))
    zone_name, zone_code, zone_type, zone_tags, z_lat, z_lon = ZONES[zone_idx]

    corr_idx  = int(rng.integers(0, len(CORRIDORS)))
    corr_name, corr_code, freeflow_kmph = CORRIDORS[corr_idx]

    # ── Road type ────────────────────────────────────────────────────────────
    road_type = rng.choice(ROAD_TYPES, p=ROAD_TYPE_WEIGHTS)

    # ── Weather ──────────────────────────────────────────────────────────────
    condition = _weather_condition(month, rng)
    weather   = _weather_features(condition, month, hour, rng)

    # ── Traffic ──────────────────────────────────────────────────────────────
    traffic = _traffic_features(
        corr_code, freeflow_kmph, hour, is_weekend, road_type, condition, rng
    )

    # ── Derived Phase-1 columns ───────────────────────────────────────────────
    avg_speed   = _avg_speed(road_type, hour, is_weekend, condition, rng)
    rainfall_mm = _rainfall_mm(weather["rain_1h_mm"], rng)

    # ── Risk label (derived, not randomly assigned) ───────────────────────────
    risk_label = _derive_risk_label(
        congestion_index = traffic["congestion_index"],
        rain_risk_score  = weather["rain_risk_score"],
        is_foggy         = weather["is_foggy"],
        is_high_wind     = weather["is_high_wind"],
        visibility_m     = weather["visibility_m"],
        road_type        = road_type,
    )

    # ── Assemble row in exact enriched_data.csv column order ─────────────────
    return {
        "road_id":               f"SYN{i:06d}",   # "SYN" prefix → never clashes with real RD*
        "road_type":             road_type,
        "time_of_day":           time_slot,
        "rainfall_mm":           rainfall_mm,
        "avg_speed_kmph":        avg_speed,
        "is_weekend":            bool(is_weekend),
        "timestamp":             ts.strftime("%Y-%m-%d %H:%M:%S"),
        "risk_label":            risk_label,
        # Zone columns
        "zone_name":             zone_name,
        "zone_code":             zone_code,
        "zone_type":             zone_type,
        "zone_tags":             zone_tags,
        "corridor_name":         corr_name,
        "corridor_code":         corr_code,
        # Weather columns
        "weather_zone_code":     zone_code,
        "weather_condition":     weather["weather_condition"],
        "temp_celsius":          weather["temp_celsius"],
        "humidity_pct":          weather["humidity_pct"],
        "wind_speed_mps":        weather["wind_speed_mps"],
        "visibility_m":          weather["visibility_m"],
        "rain_1h_mm":            weather["rain_1h_mm"],
        "rain_risk_score":       weather["rain_risk_score"],
        "is_raining":            weather["is_raining"],
        "is_foggy":              weather["is_foggy"],
        "is_high_wind":          weather["is_high_wind"],
        "is_ne_monsoon":         weather["is_ne_monsoon"],
        "rain_forecast_hours":   weather["rain_forecast_hours"],
        # Traffic columns
        "traffic_corridor_code": traffic["traffic_corridor_code"],
        "current_speed_kmph":    traffic["current_speed_kmph"],
        "freeflow_speed_kmph":   traffic["freeflow_speed_kmph"],
        "congestion_index":      traffic["congestion_index"],
        "congestion_level":      traffic["congestion_level"],
        "travel_time_ratio":     traffic["travel_time_ratio"],
        "traffic_source":        traffic["traffic_source"],
    }



# ═════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def generate(n_rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED,
             output_path: Path = OUTPUT_PATH) -> Path:
    """
    Generate `n_rows` synthetic rows and write to `output_path`.
    Returns the resolved output path.
    """
    print(f"[synthetic] Generating {n_rows:,} rows  (seed={seed}) ...")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    rows = [_build_row(i, rng) for i in range(n_rows)]
    df   = pd.DataFrame(rows)

    # ── Sanity checks before writing ─────────────────────────────────────────
    assert len(df) == n_rows, "Row count mismatch"
    assert df["road_id"].str.startswith("SYN").all(), \
        "road_id must start with SYN to avoid overlap with real data"
    assert not df["timestamp"].str.startswith("202[56789]").any(), \
        "Timestamps must be in 2024 — no overlap with real collected data"

    # ── Write ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"[synthetic] Written -> {output_path.resolve()}")
    print(f"[synthetic] Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n[synthetic] Risk label distribution:")
    for label, count in df["risk_label"].value_counts().items():
        print(f"  {label:<8} {count:>6,}  ({count/n_rows*100:.1f}%)")

    print(f"\n[synthetic] Congestion level distribution:")
    for lvl, count in df["congestion_level"].value_counts().items():
        print(f"  {lvl:<12} {count:>6,}  ({count/n_rows*100:.1f}%)")

    print(f"\n[synthetic] Weather condition distribution:")
    for cond, count in df["weather_condition"].value_counts().items():
        print(f"  {cond:<14} {count:>6,}  ({count/n_rows*100:.1f}%)")

    print(f"\n[synthetic] NE monsoon rows (Oct–Dec): "
          f"{df['is_ne_monsoon'].sum():,}  "
          f"({df['is_ne_monsoon'].mean()*100:.1f}%)")

    print(f"\n[synthetic] Numeric column ranges:")
    num_cols = ["temp_celsius", "humidity_pct", "wind_speed_mps",
                "rain_1h_mm", "congestion_index", "current_speed_kmph",
                "freeflow_speed_kmph", "travel_time_ratio"]
    for col in num_cols:
        print(f"  {col:<25}  min={df[col].min():.2f}  "
              f"max={df[col].max():.2f}  mean={df[col].mean():.2f}")

    return output_path


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic Chennai traffic/weather training data."
    )
    p.add_argument("--rows",   type=int,  default=DEFAULT_ROWS,
                   help=f"Number of rows to generate (default: {DEFAULT_ROWS})")
    p.add_argument("--seed",   type=int,  default=DEFAULT_SEED,
                   help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    p.add_argument("--output", type=Path, default=OUTPUT_PATH,
                   help=f"Output CSV path (default: {OUTPUT_PATH})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out  = generate(n_rows=args.rows, seed=args.seed, output_path=args.output)
    print(f"\nDone. Synthetic dataset saved to: {out}")
