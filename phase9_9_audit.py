import asyncio
import httpx
from httpx import AsyncClient, ASGITransport
from src.api.main import app

PAYLOAD_A = {
    "origin": {"latitude": 13.0807843, "longitude": 80.2651669},
    "deliveries": [{"latitude": 13.0806923, "longitude": 80.2652332}],
    "timestamp": "2023-10-01T12:00:00Z"
}

async def main():
    print("Starting Step 9.9 Frontend Consumability Audit...")
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            res = await client.post("/v1/route/multi-stop", json=PAYLOAD_A)
            
            print(f"Status: {res.status_code}")
            
            if res.status_code == 200:
                data = res.json()
                print("\nValidating Frontend Contract Fields...")
                print(f"Success Flag: {data.get('success')}")
                print(f"Total Distance: {data.get('total_distance_m')} m")
                
                print("\nOptimized Stops Mapping:")
                for i, stop in enumerate(data.get("optimized_stops", [])):
                    print(f" Stop {i+1}: Lat {stop['latitude']}, Lon {stop['longitude']}")
                
                print("\nLegs Mapping:")
                for i, leg in enumerate(data.get("legs", [])):
                    start = leg['start_location']
                    end = leg['end_location']
                    print(f" Leg {i+1}: Lat {start['latitude']} -> Lat {end['latitude']} | Distance: {leg['distance_m']} m")
                    
                    geo = leg.get("geometry")
                    if geo:
                        print(f" Geometry: {geo['type']} with {len(geo['coordinates'])} points")
                    else:
                        print(" Geometry: None")
                        
                # Black-box assertion
                assert "path_nodes" not in str(data), "Internal path_nodes leaked!"
                assert "total_cost" not in str(data), "Internal total_cost leaked!"
                assert "source_node" not in str(data), "Internal source_node leaked!"
                
            else:
                print(f"Route failed (Expected if mock nodes are isolated). Error: {res.text}")

if __name__ == "__main__":
    asyncio.run(main())
