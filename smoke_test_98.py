"""
Smoke test for Step 9.8 (API Application Boundary)
Simulates a real HTTP request hitting the FastAPI app.
"""
from fastapi.testclient import TestClient
from src.api.main import app

# Use the real application (lifespan will load graph and models)
client = TestClient(app)

def run_smoke_test():
    print("--- STARTING STEP 9.8 SMOKE TEST ---")
    
    # We use some real coordinates in Chennai
    # Origin: Near T. Nagar
    # Delivery: Near Guindy
    payload = {
        "origin": {
            "latitude": 13.0807843,
            "longitude": 80.2651669
        },
        "deliveries": [
            {
                "latitude": 13.0806923,
                "longitude": 80.2652332
            }
        ]
    }
    
    # Issue HTTP POST request to the API
    with client:
        response = client.post("/v1/route/multi-stop", json=payload)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Total Cost: {data['total_cost']}")
        print(f"Total Distance (m): {data['total_distance_m']}")
        print(f"Number of Route Legs: {len(data['routes'])}")
        
        # Verify GeoJSON and edge serialization
        leg = data['routes'][0]
        print(f"Path Nodes: {len(leg['path_nodes'])}")
        print(f"Path Edges: {len(leg['path_edges'])}")
        
        if leg['path_edges']:
            edge = leg['path_edges'][0]
            print(f"Sample Edge Object: {edge}")
            
        print(f"GeoJSON Type: {leg['geometry']['type']}")
        if leg['geometry']['coordinates']:
            coord = leg['geometry']['coordinates'][0]
            print(f"Sample GeoJSON Coordinate: {coord} (should be [lon, lat])")
            
        print("--- SMOKE TEST PASSED ---")
    else:
        print(f"Response Body: {response.text}")
        print("--- SMOKE TEST FAILED ---")

if __name__ == "__main__":
    run_smoke_test()
