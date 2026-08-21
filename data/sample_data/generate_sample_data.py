"""
Generates a synthetic, deliberately messy delivery-route dataset so the
ingestion engine can be tested end-to-end without needing the real Kaggle
dataset. Includes missing values, a duplicate row, an identifier column,
a boolean column, and a datetime column — one of each type the profiler
needs to detect correctly.

Run: python sample_data/generate_sample_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)
n = 200

road_types = ["highway", "arterial", "residential", "service"]
time_of_day = ["morning", "afternoon", "evening", "night"]

df = pd.DataFrame(
    {
        "road_id": [f"RD{i:05d}" for i in range(n)],  # identifier column
        "road_type": rng.choice(road_types, size=n),
        "time_of_day": rng.choice(time_of_day, size=n),
        "rainfall_mm": rng.gamma(shape=1.5, scale=3.0, size=n).round(2),
        "avg_speed_kmph": rng.normal(loc=35, scale=8, size=n).round(1),
        "is_weekend": rng.choice([True, False], size=n),
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h").astype(str),
        "risk_label": rng.choice(["low", "medium", "high"], size=n, p=[0.5, 0.35, 0.15]),
    }
)

# Inject missing values to exercise the cleaner
for col, frac in [("rainfall_mm", 0.08), ("avg_speed_kmph", 0.05), ("road_type", 0.03)]:
    idx = rng.choice(n, size=int(n * frac), replace=False)
    df.loc[idx, col] = np.nan

# Inject one exact duplicate row to exercise duplicate-dropping
df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

df.to_csv("sample_data/sample_delivery_data.csv", index=False)
print(f"Wrote sample_data/sample_delivery_data.csv with {len(df)} rows")
