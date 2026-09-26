import networkx as nx
import osmnx as ox

G = nx.MultiDiGraph()
G.add_node(1, y=10.0, x=10.0)
G.add_node(2, y=20.0, x=20.0)
G.add_edge(1, 2, key=0, length=10)
ox.save_graphml(G, "test.graphml")

G2 = ox.load_graphml("test.graphml")
for u, v, k in G2.edges(keys=True):
    print(f"Node u type: {type(u)}")
    print(f"Key type: {type(k)}")
    break
