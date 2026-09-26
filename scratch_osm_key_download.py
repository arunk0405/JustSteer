import osmnx as ox
import networkx as nx

# Download tiny graph
place = "Piedmont, California, USA"
graph = ox.graph_from_place(place, network_type="drive")
for u, v, k in graph.edges(keys=True):
    print(f"Key type: {type(k)} value: {k}")
    break
