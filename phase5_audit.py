import json
from src.api.main import app
from fastapi.testclient import TestClient

def run_tests():
    print("Starting Phase 5 Audit...")
    with TestClient(app) as client:
        def test_case(name, method, url, **kwargs):
            print(f"\n--- {name} ---")
            res = client.request(method, url, **kwargs)
            print(f"Status: {res.status_code}")
            try:
                print(f"Response: {res.json()}")
            except:
                print(f"Response (text): {res.text}")
            return res

        # 1. Input Validation
        test_case("Missing required fields", "POST", "/v1/route/multi-stop", json={})
        test_case("Wrong data types", "POST", "/v1/route/multi-stop", json={"origin": "string"})
        test_case("Max deliveries exceeded", "POST", "/v1/route/multi-stop", json={
            "origin": {"latitude": 13.0807843, "longitude": 80.2651669},
            "deliveries": [{"latitude": 13.0, "longitude": 80.0}] * 9,
            "timestamp": "2023-10-01T12:00:00Z"
        })
        test_case("Latitude < -90", "POST", "/v1/route/multi-stop", json={
            "origin": {"latitude": -91.0, "longitude": 80.2651669},
            "deliveries": [{"latitude": 13.0, "longitude": 80.0}],
            "timestamp": "2023-10-01T12:00:00Z"
        })
        test_case("Longitude > 180", "POST", "/v1/route/multi-stop", json={
            "origin": {"latitude": 13.0, "longitude": 181.0},
            "deliveries": [{"latitude": 13.0, "longitude": 80.0}],
            "timestamp": "2023-10-01T12:00:00Z"
        })

        # 2. HTTP Boundary
        test_case("Unsupported Method (GET)", "GET", "/v1/route/multi-stop")
        test_case("Unknown Route", "POST", "/v1/route/unknown")
        test_case("Malformed JSON", "POST", "/v1/route/multi-stop", content="invalid{json", headers={"Content-Type": "application/json"})
        
        # 3. Failure Isolation
        valid_payload = {
            "origin": {"latitude": 13.0807843, "longitude": 80.2651669},
            "deliveries": [{"latitude": 13.0806923, "longitude": 80.2652332}],
            "timestamp": "2023-10-01T12:00:00Z"
        }
        test_case("Valid Request 1", "POST", "/v1/route/multi-stop", json=valid_payload)
        test_case("Invalid coordinates mapping", "POST", "/v1/route/multi-stop", json={
            "origin": {"latitude": 89.9, "longitude": 80.2651669},
            "deliveries": [{"latitude": 13.0806923, "longitude": 80.2652332}],
            "timestamp": "2023-10-01T12:00:00Z"
        })
        test_case("Valid Request 2", "POST", "/v1/route/multi-stop", json=valid_payload)

if __name__ == "__main__":
    run_tests()
