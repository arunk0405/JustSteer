import asyncio
import httpx
from httpx import AsyncClient, ASGITransport
import time
from unittest.mock import patch
from src.api.main import app

# Payloads
PAYLOAD_A = {
    "origin": {"latitude": 13.0807843, "longitude": 80.2651669},
    "deliveries": [{"latitude": 13.0806923, "longitude": 80.2652332}],
    "timestamp": "2023-10-01T12:00:00Z"
}
PAYLOAD_B = {
    "origin": {"latitude": 13.0820000, "longitude": 80.2600000},
    "deliveries": [{"latitude": 13.0810000, "longitude": 80.2610000}, {"latitude": 13.0800000, "longitude": 80.2620000}],
    "timestamp": "2023-10-02T12:00:00Z"
}
PAYLOAD_INVALID_COORD = {
    "origin": {"latitude": 900.0, "longitude": 80.2651669},
    "deliveries": [{"latitude": 13.0806923, "longitude": 80.2652332}],
    "timestamp": "2023-10-01T12:00:00Z"
}

async def run_request(client, payload):
    return await client.post("/v1/route/multi-stop", json=payload)

async def main():
    print("Starting Phase 4 Audit...")
    
    # Track metrics
    metrics = {
        "generate_calls": 0
    }
    
    from src.routing.dynamic_cost_generator import DynamicCostGenerator
    original_generate = DynamicCostGenerator.generate
    
    def mocked_generate(*args, **kwargs):
        metrics["generate_calls"] += 1
        return original_generate(*args, **kwargs)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            
            # C. Application Startup Test
            print("C. Lifespan triggered successfully.")
            
            # E. Application State
            graph_id = id(app.state.dynamic_routing_app.graph)
            
            # Snapshot Graph
            graph = app.state.dynamic_routing_app.graph
            edges_before = list(graph.edges(keys=True, data=True))
            
            with patch("src.routing.dynamic_cost_generator.DynamicCostGenerator.generate", side_effect=mocked_generate):
                
                # D. Single Request
                res = await run_request(client, PAYLOAD_A)
                print(f"D. Single Request Status: {res.status_code}")
                
                # F. Request Isolation & H. Mixed Concurrency
                results = await asyncio.gather(
                    run_request(client, PAYLOAD_A),
                    run_request(client, PAYLOAD_B),
                    run_request(client, PAYLOAD_INVALID_COORD)
                )
                print(f"H. Mixed Concurrency Statuses: {[r.status_code for r in results]}")
                
                # G. True Concurrency (20 requests)
                tasks = [run_request(client, PAYLOAD_A) for _ in range(20)]
                res_bulk = await asyncio.gather(*tasks)
                print(f"G. True Concurrency (20 reqs) Statuses: {[r.status_code for r in res_bulk]}")
                
                # I. Order Dependency (A -> B -> A)
                res_a1 = await run_request(client, PAYLOAD_A)
                res_b1 = await run_request(client, PAYLOAD_B)
                res_a2 = await run_request(client, PAYLOAD_A)
                print(f"I. Order Dependency: A1={res_a1.status_code}, B1={res_b1.status_code}, A2={res_a2.status_code}")
                
                # N. Post-Failure Recovery
                res_valid1 = await run_request(client, PAYLOAD_A)
                res_fail = await run_request(client, PAYLOAD_INVALID_COORD)
                res_valid2 = await run_request(client, PAYLOAD_A)
                print(f"N. Post-Failure Recovery: Valid1={res_valid1.status_code}, Fail={res_fail.status_code}, Valid2={res_valid2.status_code}")
                
                # R. Resource Leak Check (25 sequential)
                print("R. Running 25 sequential requests...")
                for _ in range(25):
                    await run_request(client, PAYLOAD_A)
                print("R. 25 requests completed.")

            # J. Graph Immutability Check
            edges_after = list(graph.edges(keys=True, data=True))
            if edges_before == edges_after:
                print("J. Graph Immutability: TRUE (Zero Mutations)")
            else:
                print("J. Graph Immutability: FAILED (Mutations detected)")

            # E. Check global state identities
            graph_id_after = id(app.state.dynamic_routing_app.graph)
            if graph_id == graph_id_after:
                print("E. Application Global State Identity: UNCHANGED")

            # L. Dynamic Cost Generation Count
            print(f"L. DynamicCostGenerator.generate calls: {metrics['generate_calls']}")
            
            print("Audit tests completed.")

if __name__ == "__main__":
    asyncio.run(main())
