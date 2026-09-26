import threading
import time
import networkx as nx
from datetime import datetime
from src.routing.edge_cost import EdgeId
from src.routing.dynamic_routing_app import DynamicRoutingApp
from src.routing.feature_resolver import FeatureResolver, RoutingContext, WeatherContext, ResolvedEdgeFeatures
from src.routing.prediction_orchestrator import PredictionOrchestrator, EdgeCostPredictionResult
from src.routing.contracts import TrafficPrediction, SafetyPrediction, CombinedEdgeCost
from src.routing.candidate_selector import CandidateEdgeSelector

class SlowFakePredictionOrchestrator(PredictionOrchestrator):
    def __init__(self, costs):
        self.costs = costs

    def predict_edge(self, edge_id, graph, traffic_features, safety_features):
        time.sleep(0.1) # Simulate slow ML inference to ensure threads overlap
        cost = self.costs.get(edge_id, 1.0)
        return EdgeCostPredictionResult(
            edge_id=edge_id,
            traffic_prediction=TrafficPrediction(predicted_travel_time=10.0),
            safety_prediction=SafetyPrediction(risk_score=20.0),
            travel_time_minutes=10.0,
            combined_cost=CombinedEdgeCost(
                edge_id=str(edge_id),
                predicted_travel_time=10.0,
                raw_safety_risk=20.0,
                normalized_travel_time=0.5,
                normalized_safety_risk=0.2,
                travel_time_weight=0.7,
                safety_weight=0.3,
                combined_cost=cost
            )
        )

class FakeFeatureResolver(FeatureResolver):
    def __init__(self):
        super().__init__(nx.MultiDiGraph())
    def resolve(self, edge_id, context):
        return ResolvedEdgeFeatures(edge_id, {"mock": 1}, {"mock": 2})

G = nx.MultiDiGraph()
G.add_nodes_from([(1,dict(x=0.,y=0.)), (2,dict(x=0.,y=.1)), (3,dict(x=.1,y=.1))])
G.add_edge(1, 2, key=0, length=100.)
G.add_edge(2, 3, key=0, length=100.)
G.add_edge(1, 3, key=0, length=100.)
G.add_edge(3, 2, key=0, length=100.)

weather = WeatherContext(30., 0., 10., 10000., 60., 0.1, 0)
ctx = RoutingContext(datetime(2023, 1, 1), weather, 1.0, None)
selector = CandidateEdgeSelector(G, 500000.)

costs_A = {EdgeId(1, 2, 0): 0.1, EdgeId(2, 3, 0): 0.1} # Route 1->2->3 is best (cost 0.2)
costs_B = {EdgeId(1, 3, 0): 0.1, EdgeId(3, 2, 0): 0.1} # Route 1->3->2 is best (cost 0.2)

app_A = DynamicRoutingApp(G, FakeFeatureResolver(), SlowFakePredictionOrchestrator(costs_A), selector)
app_B = DynamicRoutingApp(G, FakeFeatureResolver(), SlowFakePredictionOrchestrator(costs_B), selector)

res_A = None
res_B = None

def run_A():
    global res_A
    res_A = app_A.route(1, (2, 3), ctx)

def run_B():
    global res_B
    res_B = app_B.route(1, (2, 3), ctx)

t1 = threading.Thread(target=run_A)
t2 = threading.Thread(target=run_B)

t1.start()
t2.start()

t1.join()
t2.join()

print("A sequence:", res_A.delivery_sequence.sequence)
print("B sequence:", res_B.delivery_sequence.sequence)
