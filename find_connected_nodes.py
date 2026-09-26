import osmnx as ox
graph = ox.load_graphml("data/road_network/chennai_drive.graphml")
edges = list(graph.edges(data=True))
if edges:
    u, v, data = edges[0]
    node_u = graph.nodes[u]
    node_v = graph.nodes[v]
    print(f"Origin: {node_u['y']}, {node_u['x']}")
    print(f"Delivery: {node_v['y']}, {node_v['x']}")
