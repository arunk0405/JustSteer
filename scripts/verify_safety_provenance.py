import os
import json
import hashlib
import sys
import importlib.util
from pathlib import Path

def compute_sha256(filepath):
    if not os.path.exists(filepath):
        return None
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    print("========================================")
    print("SAFETY MODEL PROVENANCE VERIFICATION")
    print("========================================")
    
    project_root = Path(__file__).parent.parent
    metadata_path = project_root / "models" / "safety" / "safety_risk_xgb_metadata.json"
    
    if not metadata_path.exists():
        # Fallback to RF if that was selected
        metadata_path = project_root / "models" / "safety" / "safety_risk_rf_metadata.json"
        
    if not metadata_path.exists():
        print(f"[FAIL] Metadata exists (neither xgb nor rf metadata found)")
        return 1
    print("[PASS] Metadata exists")
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    model_path = project_root / metadata.get("artifact", {}).get("path", "")
    if not model_path.exists():
        print(f"[FAIL] Model artifact exists ({model_path} not found)")
        return 1
    print("[PASS] Model artifact exists")
        
    actual_hash = compute_sha256(str(model_path))
    expected_hash = metadata.get("artifact", {}).get("sha256", "")
    if actual_hash != expected_hash:
        print(f"[FAIL] SHA-256 matches (Expected: {expected_hash}, Actual: {actual_hash})")
        return 1
    print("[PASS] SHA-256 matches")
    
    # Verify consistency between training feature schema and runtime feature schema
    sys.path.insert(0, str(project_root))
    
    try:
        from src.routing.prediction_providers import SafetyPredictionProvider
        runtime_features = SafetyPredictionProvider.EXPECTED_FEATURES
    except ImportError as e:
        print(f"[FAIL] Could not import SafetyPredictionProvider: {e}")
        return 1
        
    training_features = metadata.get("feature_order", [])
    
    if len(training_features) != len(runtime_features):
        print(f"[FAIL] Safety feature count matches (Train: {len(training_features)}, Runtime: {len(runtime_features)})")
        return 1
    print("[PASS] Safety feature count matches")
    
    if set(training_features) != set(runtime_features):
        print("[FAIL] Safety feature names match")
        return 1
    print("[PASS] Safety feature names match")
    
    if training_features != runtime_features:
        print("[FAIL] Safety feature order matches")
        return 1
    print("[PASS] Safety feature order matches")
    
    # Ensure it's correctly flagged as synthetic
    if metadata.get("data_origin") != "synthetic":
        print("[FAIL] Safety data origin is 'synthetic'")
        return 1
    print("[PASS] Safety data explicitly flagged as 'synthetic'")
    
    print("\nSAFETY PROVENANCE: PASS")
    print("========================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
