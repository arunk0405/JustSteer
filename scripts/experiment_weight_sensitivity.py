import sys
import os
# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.routing.contracts import TrafficPrediction, SafetyPrediction
from src.routing.weight_calculator import NormalizationConfig, WeightCalculator
import pandas as pd

def run_sensitivity_experiment():
    print("==================================================")
    print("WEIGHT SENSITIVITY ANALYSIS EXPERIMENT")
    print("==================================================\n")
    
    # Candidate routes for an A to B delivery
    # Route A: Fast but very risky (Highway, heavy traffic, bad weather)
    # Route B: Balanced (Arterial, moderate traffic, average safety)
    # Route C: Slow but extremely safe (Residential, light traffic)
    
    candidates = {
        "Route_A_Fast_Risky": {"tt": 5.0, "risk": 85.0},
        "Route_B_Balanced": {"tt": 12.0, "risk": 40.0},
        "Route_C_Slow_Safe": {"tt": 22.0, "risk": 10.0}
    }
    
    # Normalization parameters (calibrated from hypothetical dataset)
    p5 = 2.0
    p95 = 30.0
    
    weight_profiles = [
        (0.90, 0.10), # Time is critical
        (0.70, 0.30), # Default preference
        (0.50, 0.50), # Equal balance
        (0.30, 0.70), # Safety first
        (0.10, 0.90)  # Extreme safety priority
    ]
    
    results = []
    
    for w_tt, w_sf in weight_profiles:
        config = NormalizationConfig(
            travel_time_p5=p5,
            travel_time_p95=p95,
            travel_time_weight=w_tt,
            safety_weight=w_sf
        )
        calculator = WeightCalculator(config)
        
        costs = {}
        for route_id, params in candidates.items():
            result = calculator.calculate(
                edge_id=route_id,
                traffic_pred=TrafficPrediction(params["tt"]),
                safety_pred=SafetyPrediction(params["risk"])
            )
            costs[route_id] = result.combined_cost
            
        # Find the route with the minimum cost (preferred)
        best_route = min(costs, key=costs.get)
        
        results.append({
            "Weights (TT/Safe)": f"{int(w_tt*100)}/{int(w_sf*100)}",
            "Cost_Route_A": round(costs["Route_A_Fast_Risky"], 3),
            "Cost_Route_B": round(costs["Route_B_Balanced"], 3),
            "Cost_Route_C": round(costs["Route_C_Slow_Safe"], 3),
            "Preferred_Route": best_route
        })
        
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print("\nInterpretation:")
    print("- At 90/10, Route A wins because travel time completely dominates.")
    print("- At 70/30, Route B wins because Route A's extreme risk penalty finally outweighs its speed advantage.")
    print("- At 10/90, Route C wins because safety completely dominates, and 22 mins travel time penalty is acceptable.")
    
if __name__ == "__main__":
    run_sensitivity_experiment()
