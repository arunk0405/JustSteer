import networkx as nx
import os
import pickle

graph_path = "data/osm/chennai_drivable_graph.pkl"
if os.path.exists(graph_path):
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    print("Graph loaded.")
    for u, v, k in G.edges(keys=True):
        print(f"Key type: {type(k)} value: {k}")
        break
else:
    print("Graph not found.")
