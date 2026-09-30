import os
from fastapi.testclient import TestClient
import networkx as nx
import copy

def check_graph_immutability():
    from src.api.main import app
    
    with TestClient(app) as client:
        graph = app.state.dynamic_routing_app.graph
        
        nodes_before = list(graph.nodes(data=True))
        edges_before = list(graph.edges(keys=True, data=True))
        
        # Issue request
        payload = {
            "origin": {"latitude": 13.0807843, "longitude": 80.2651669},
            "deliveries": [{"latitude": 13.0806923, "longitude": 80.2652332}],
            "timestamp": "2023-10-01T12:00:00Z"
        }
        res = client.post("/v1/route/multi-stop", json=payload)
        print("Response:", res.status_code)
        
        nodes_after = list(graph.nodes(data=True))
        edges_after = list(graph.edges(keys=True, data=True))
        
        if nodes_before == nodes_after and edges_before == edges_after:
            print("GRAPH REMAINS IMMUTABLE")
        else:
            print("GRAPH MUTATED")

if __name__ == "__main__":
    check_graph_immutability()
