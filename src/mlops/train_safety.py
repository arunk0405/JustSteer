"""
Safety Prediction ML Model (Model 2)
Predicts the historical/future risk level of a road segment and converts it into a 
normalized risk score used by the route optimizer (A* edge penalty).
"""

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import f1_score
import xgboost as xgb

def load_data(filepath="data/raw/synthetic_road_safety.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please run scripts/generate_synthetic_safety_data.py first.")
    return pd.read_csv(filepath)

def calculate_severity(fatal, grievous, minor):
    """
    Calculate severity score based on modeling assumptions.
    Fatal = 5, Grievous = 3, Minor = 1
    """
    return 5 * fatal + 3 * grievous + 1 * minor

def engineer_features(df):
    """
    Stage 3 — Feature Engineering
    Creates historical cumulative features to predict the target year.
    """
    df = df.copy()
    
    # Sort strictly by segment and time to ensure historical shift is correct
    df.sort_values(by=["segment_id", "year"], inplace=True)
    
    # Target: Severity for the CURRENT year
    df["severity_score_current"] = calculate_severity(
        df["fatal_accidents"], df["grievous_injuries"], df["minor_injuries"]
    )
    
    # Historical Features: shifted cumulations to ensure we don't leak current year data
    hist_cols = ["accidents_total", "fatal_accidents", "grievous_injuries", "minor_injuries"]
    
    for col in hist_cols:
        df[f"{col}_historical"] = df.groupby("segment_id")[col].transform(
            lambda x: x.shift(1).cumsum().fillna(0)
        )
        
    df["injuries_historical"] = df["grievous_injuries_historical"] + df["minor_injuries_historical"]
    df["historical_risk_score"] = calculate_severity(
        df["fatal_accidents_historical"], 
        df["grievous_injuries_historical"], 
        df["minor_injuries_historical"]
    )
    
    # Road characteristics (static) - Encode categorical variables
    encoders = {}
    for col in ["road_type", "surface", "lighting", "traffic_exposure"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Filter out the first year (2022) because it has 0 history (shifted value is NaN/0)
    min_year = df["year"].min()
    df = df[df["year"] > min_year].copy()
    
    # ── Target Variable (Approach A: Risk Classification) ──
    # Low Risk -> bottom 50%
    # Medium Risk -> 50-80%
    # High Risk -> top 20%
    q50 = df["severity_score_current"].quantile(0.50)
    q80 = df["severity_score_current"].quantile(0.80)
    
    def classify_risk(score):
        if score <= q50:
            return "LOW"
        elif score <= q80:
            return "MEDIUM"
        else:
            return "HIGH"
            
    df["risk_class"] = df["severity_score_current"].apply(classify_risk)
    
    # Encode Target (0=LOW, 1=MEDIUM, 2=HIGH)
    target_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    df["risk_class_enc"] = df["risk_class"].map(target_map)
    
    return df, encoders

def train_safety_model(df, model_out_dir="models/safety"):
    """
    Stage 4 — Train Risk Model (Temporal Split)
    """
    latest_year = df["year"].max()
    print(f"[*] Temporal Split: Train on data < {latest_year}, Test on {latest_year}\n")
    
    train_df = df[df["year"] < latest_year]
    test_df = df[df["year"] == latest_year]
    
    features = [
        "accidents_total_historical",
        "fatal_accidents_historical",
        "injuries_historical",
        "historical_risk_score",
        "road_type_enc",
        "lanes",
        "speed_limit",
        "lighting_enc",
        "surface_enc",
        "traffic_exposure_enc"
    ]
    
    X_train = train_df[features]
    y_train = train_df["risk_class_enc"]
    
    X_test = test_df[features]
    y_test = test_df["risk_class_enc"]
    
    print(f"[*] Training Random Forest Classifier on {len(X_train)} samples...")
    # Baseline Random Forest Model
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced")
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    rf_f1 = f1_score(y_test, rf_pred, average="macro")
    
    print(f"[*] Training XGBoost Classifier on {len(X_train)} samples...")
    # Advanced XGBoost Model
    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, use_label_encoder=False, eval_metric="mlogloss")
    xgb_clf.fit(X_train, y_train)
    xgb_pred = xgb_clf.predict(X_test)
    xgb_f1 = f1_score(y_test, xgb_pred, average="macro")
    
    print("-" * 50)
    print("Model Evaluation (Future Prediction on Test Set)")
    print("-" * 50)
    print(f"Random Forest Macro F1: {rf_f1:.4f}")
    print(f"XGBoost Macro F1:       {xgb_f1:.4f}")
    
    if xgb_f1 >= rf_f1:
        print("\n=> Selecting XGBoost as the best model.")
        best_clf = xgb_clf
        y_pred = xgb_pred
        # Feature Importance (Explainability) for XGBoost
        importances = pd.Series(best_clf.feature_importances_, index=features).sort_values(ascending=False)
        model_name = "safety_risk_xgb.pkl"
    else:
        print("\n=> Selecting Random Forest as the best model.")
        best_clf = rf_clf
        y_pred = rf_pred
        # Feature Importance (Explainability) for Random Forest
        importances = pd.Series(best_clf.feature_importances_, index=features).sort_values(ascending=False)
        model_name = "safety_risk_rf.pkl"
        
    print("\n--- Final Model Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["LOW (0)", "MEDIUM (1)", "HIGH (2)"]))
    
    print("\n--- Feature Importance (Explainability) ---")
    print(importances.round(4))
    print("\n")
    
    # Stage 5 — Output Risk Score for A*
    PENALTY_MAP = {0: 20, 1: 50, 2: 80}
    test_df_results = test_df.copy()
    test_df_results["predicted_class_enc"] = y_pred
    test_df_results["predicted_class"] = test_df_results["predicted_class_enc"].map({0: "LOW", 1: "MEDIUM", 2: "HIGH"})
    test_df_results["a_star_penalty"] = test_df_results["predicted_class_enc"].map(PENALTY_MAP)
    
    print("-" * 50)
    print("Example Output: Segment Risk Scores for A* (Normalized 0-100)")
    print("-" * 50)
    display_cols = ["segment_id", "historical_risk_score", "predicted_class", "a_star_penalty"]
    print(test_df_results[display_cols].sample(10, random_state=42).to_string(index=False))
    print("\n")
    
    # Save the model
    os.makedirs(model_out_dir, exist_ok=True)
    model_path = os.path.join(model_out_dir, model_name)
    joblib.dump(best_clf, model_path)
    print(f"[*] Saved best safety risk model to {model_path}")
    
    return best_clf, test_df_results

if __name__ == "__main__":
    df_raw = load_data()
    df_processed, encoders = engineer_features(df_raw)
    clf, results = train_safety_model(df_processed)
