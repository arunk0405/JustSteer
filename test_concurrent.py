import asyncio
import httpx
from src.api.main import app
from httpx import AsyncClient, ASGITransport

async def test_concurrency():
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            payload_1 = {
                "origin": {"latitude": 13.0807843, "longitude": 80.2651669},
                "deliveries": [{"latitude": 13.0806923, "longitude": 80.2652332}],
                "timestamp": "2023-10-01T12:00:00Z"
            }
            
            payload_2 = {
                "origin": {"latitude": 13.0807843, "longitude": 80.2651669},
                "deliveries": [{"latitude": 13.0806923, "longitude": 80.2652332}],
                "timestamp": "2023-10-01T12:00:00Z"
            }
            
            results = await asyncio.gather(
                client.post("/v1/route/multi-stop", json=payload_1),
                client.post("/v1/route/multi-stop", json=payload_2)
            )
            
            for r in results:
                print(r.status_code)

if __name__ == "__main__":
    asyncio.run(test_concurrency())
