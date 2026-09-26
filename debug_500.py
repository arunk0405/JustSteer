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
    async with app.router.lifespan_context(app):
        # Setting raise_app_exceptions=True to see the traceback
        transport = ASGITransport(app=app, raise_app_exceptions=True)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            res = await client.post("/v1/route/multi-stop", json=PAYLOAD_A)
            print(res.status_code)
            print(res.text)

if __name__ == "__main__":
    asyncio.run(main())
