import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

def generate_data(num_segments=500, years=[2022, 2023, 2024, 2025], seed=42):
    """
    Generate synthetic safety data replicating Chennai's road network history.
    """
    np.random.seed(seed)
    
    # Road Characteristics (Static per segment)
    road_types = ["Highway", "Arterial", "Residential", "Service"]
    surfaces = ["Asphalt", "Concrete", "Unpaved"]
    lightings = ["Good", "Moderate", "Poor", "None"]
    traffic_levels = ["Low", "Medium", "High", "Severe"]
    
    data = []
    
    # Generate static characteristics for each segment
    segments = []
    for i in range(num_segments):
        segment_id = f"R{i+1:04d}"
        road_type = np.random.choice(road_types, p=[0.1, 0.3, 0.4, 0.2])
        
        # Lanes and speed depend on road type
        if road_type == "Highway":
            lanes = np.random.choice([4, 6])
            speed_limit = np.random.choice([80, 100])
            surface = "Asphalt"
        elif road_type == "Arterial":
            lanes = np.random.choice([2, 3, 4])
            speed_limit = np.random.choice([50, 60])
            surface = np.random.choice(["Asphalt", "Concrete"], p=[0.8, 0.2])
        elif road_type == "Residential":
            lanes = np.random.choice([1, 2])
            speed_limit = np.random.choice([30, 40])
            surface = np.random.choice(["Asphalt", "Concrete", "Unpaved"], p=[0.6, 0.3, 0.1])
        else:
            lanes = 1
            speed_limit = 20
            surface = np.random.choice(["Asphalt", "Unpaved"], p=[0.5, 0.5])
            
        lighting = np.random.choice(lightings)
        traffic_exposure = np.random.choice(traffic_levels)
        
        # Inherent segment risk base score (latent variable)
        base_risk = 0
        if road_type in ["Highway", "Arterial"]: base_risk += 1
        if lighting in ["Poor", "None"]: base_risk += 1.5
        if traffic_exposure in ["High", "Severe"]: base_risk += 2
        if surface == "Unpaved": base_risk += 0.5
        
        segments.append({
            "segment_id": segment_id,
            "road_type": road_type,
            "lanes": lanes,
            "speed_limit": speed_limit,
            "surface": surface,
            "lighting": lighting,
            "traffic_exposure": traffic_exposure,
            "base_risk": base_risk
        })
        
    # Generate yearly accident data for each segment
    for seg in segments:
        for year in years:
            # Add some temporal noise and a slight increasing trend over years
            year_trend = (year - min(years)) * 0.1
            current_risk = seg["base_risk"] + year_trend + np.random.normal(0, 0.5)
            
            # Poisson distribution for counts based on current_risk
            lam = max(0.1, current_risk)
            
            accidents_total = np.random.poisson(lam)
            
            # Distribute into severity
            if accidents_total == 0:
                fatal = 0
                grievous = 0
                minor = 0
            else:
                # Severity probabilities depend slightly on speed limit
                fatal_prob = 0.01 + (seg["speed_limit"] / 1000)
                grievous_prob = 0.1 + (seg["speed_limit"] / 500)
                minor_prob = 1.0 - (fatal_prob + grievous_prob)
                
                severities = np.random.choice(
                    ["fatal", "grievous", "minor"], 
                    size=accidents_total, 
                    p=[fatal_prob, grievous_prob, minor_prob]
                )
                fatal = sum(severities == "fatal")
                grievous = sum(severities == "grievous")
                minor = sum(severities == "minor")
                
            data.append({
                "segment_id": seg["segment_id"],
                "year": year,
                "accidents_total": accidents_total,
                "fatal_accidents": fatal,
                "grievous_injuries": grievous,
                "minor_injuries": minor,
                "road_type": seg["road_type"],
                "lanes": seg["lanes"],
                "speed_limit": seg["speed_limit"],
                "surface": seg["surface"],
                "lighting": seg["lighting"],
                "traffic_exposure": seg["traffic_exposure"]
            })
            
    return pd.DataFrame(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic safety data.")
    parser.add_argument("--output", type=str, default="data/raw/synthetic_road_safety.csv")
    parser.add_argument("--segments", type=int, default=500)
    args = parser.parse_args()
    
    print(f"Generating synthetic data for {args.segments} road segments...")
    df = generate_data(num_segments=args.segments)
    
    # Ensure directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")
