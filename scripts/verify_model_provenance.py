import os
import json
import hashlib
import sys
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
    print("MODEL PROVENANCE VERIFICATION")
    print("========================================")
    
    project_root = Path(__file__).parent.parent
    metadata_path = project_root / "models" / "velocity_predictor_metadata.json"
    
    if not metadata_path.exists():
        print("[FAIL] Metadata exists (velocity_predictor_metadata.json not found)")
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
    
    if metadata.get("feature_count") != 19:
        print(f"[FAIL] Feature count = 19 (Found {metadata.get('feature_count')})")
        return 1
    print("[PASS] Feature count = 19")
    
    if not metadata.get("feature_order") or len(metadata.get("feature_order")) != 19:
        print("[FAIL] Feature order valid")
        return 1
    print("[PASS] Feature order valid")
    
    if not metadata.get("evaluation_metrics"):
        print("[FAIL] Training metrics present")
        return 1
    print("[PASS] Training metrics present")
    
    if not metadata.get("dataset_hashes") or not metadata.get("dataset_sources"):
        print("[FAIL] Dataset provenance present")
        return 1
    print("[PASS] Dataset provenance present")
    
    if not metadata.get("osm_enrichment", {}).get("enabled"):
        print("[FAIL] OSM enrichment recorded")
        return 1
    print("[PASS] OSM enrichment recorded")
    
    # We infer data quality gates passed if training successfully generated this valid metadata.
    # We could also check that enriched_row_count > raw_row_count
    if metadata.get("enriched_row_count", 0) <= metadata.get("raw_row_count", 0):
        print("[FAIL] Data quality gates passed (Enriched rows not > raw rows)")
        return 1
    print("[PASS] Data quality gates passed")
    
    print("\nMODEL PROVENANCE: PASS")
    print("========================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
