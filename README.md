# JustSteer: Intelligent Delivery Route Optimization System

## Introduction

JustSteer is a delivery route optimization system designed to help identify efficient routes between multiple delivery locations while considering both travel time and road safety.

The system combines geospatial data, graph-based pathfinding, machine learning predictions, and multi-stop route optimization to generate routes through a road network.

Unlike a conventional shortest-distance routing approach, JustSteer uses a configurable cost model that considers estimated travel time and safety risk when evaluating road segments.

The goal is to support more informed delivery route planning by balancing efficiency with road conditions and risk-related factors.

## What the System Does

JustSteer provides an end-to-end route planning workflow:

- Converts delivery addresses into geographic coordinates.
- Maps geographic coordinates to nearby nodes in a road network.
- Uses a graph representation of the road network for route calculation.
- Predicts road-related traffic and safety characteristics using machine learning models.
- Calculates dynamic costs for road segments.
- Finds routes using the A* pathfinding algorithm.
- Optimizes the visiting order for multiple delivery locations.
- Returns route geometry and relevant route information for visualization.
- Displays the planned route on an interactive map.

## System Architecture

JustSteer is organized around the following major stages.

### Stage 1: Address and Location Processing

This stage converts delivery addresses into geographic locations that can be used by the routing engine.

Main components:

- Nominatim geocoding integration
- Address search and candidate selection
- Coordinate validation and normalization
- Geocoding cache and rate limiting
- Geographic coordinate processing

Outcome:

A set of confirmed delivery coordinates represented as latitude and longitude.

Geocoding results are treated as location candidates. Ambiguous results can be reviewed and selected before being used for route planning.

### Stage 2: Road Network Processing

This stage maps delivery coordinates onto the road network used by the routing engine.

Main components:

- OpenStreetMap road network data
- NetworkX graph representation
- Geographic node lookup using a BallTree spatial index
- Road segment attributes and geometry
- Optimized graph cache for faster loading

The road network is represented as a directed graph, allowing the routing engine to account for road direction and available connections.

Outcome:

Delivery locations are associated with nearby road network nodes, and the routing engine can access the graph required for pathfinding.

### Stage 3: Machine Learning Predictions

JustSteer uses machine learning models to estimate road-related characteristics that contribute to route evaluation.

Main components:

- Traffic velocity prediction model
- Road and environmental feature processing
- Safety risk prediction model
- Feature scaling and model inference
- Prediction orchestration

The traffic model estimates road speed using contextual and road-network features. The safety model estimates a risk score using its trained features.

These predictions are used as inputs to route evaluation rather than directly selecting the final route.

Outcome:

Estimated traffic and safety-related values that can be associated with road segments during route calculation.

**Important:** The current traffic model should not be described as using live traffic sensors on every road. The safety model was trained using synthetic road-safety data, so its predictions are experimental rather than validated real-world accident probabilities.

### Stage 4: Dynamic Route Cost Calculation

This stage combines the road network and machine learning predictions into a cost used by the routing algorithm.

The cost model considers:

- Road segment length
- Estimated travel time
- Predicted safety risk
- Configurable weighting between route efficiency and risk

The system evaluates road segments using a combined cost rather than relying only on geographic distance.

Outcome:

A dynamically evaluated road network in which different road segments can have different costs depending on their predicted characteristics.

### Stage 5: A* Pathfinding

JustSteer uses the A* search algorithm to find a path between two road network nodes.

A* evaluates candidate paths using:

- The accumulated cost from the starting node
- A heuristic estimate of the remaining cost
- The dynamic cost of traversing road segments

The algorithm explores the graph and reconstructs the selected path after reaching the destination.

Outcome:

A sequence of road network nodes and edges representing the selected route between two locations.

### Stage 6: Multi-Stop Route Optimization

For delivery requests containing multiple stops, JustSteer evaluates different visiting orders to identify a lower-cost delivery sequence.

Main components:

- Delivery stop permutation generation
- Route evaluation between consecutive stops
- Total route cost calculation
- Best-order selection

The current approach is suitable for a relatively small number of delivery stops. It is a permutation-based optimizer, not a complete large-scale vehicle routing solution.

Outcome:

An optimized delivery stop order and the corresponding route through the road network.

### Stage 7: Route Response and Map Visualization

The final route is prepared for the frontend and displayed on an interactive map.

Main components:

- FastAPI backend
- Route geometry generation
- Route response serialization
- React and TypeScript frontend
- Leaflet map visualization
- OpenStreetMap map tiles

Outcome:

A visual delivery route with the planned stops and road geometry.

## High-Level Workflow

1. The user enters delivery locations.
2. Address search converts addresses into geographic coordinates.
3. Coordinates are mapped to nearby road network nodes.
4. The routing engine accesses the road graph.
5. Machine learning models estimate traffic and safety-related characteristics.
6. The dynamic cost model evaluates road segments.
7. A* calculates routes between the required locations.
8. The multi-stop optimizer evaluates delivery stop orders.
9. The backend generates the final route response.
10. The frontend displays the route on the interactive map.

## Core Technology Stack

| Technology | Purpose |
|---|---|
| Python | Routing and backend development |
| FastAPI | Backend API framework |
| Pydantic | Request and response validation |
| NetworkX | Road network graph representation |
| OpenStreetMap | Road network data |
| Nominatim | Address geocoding |
| BallTree | Efficient geographic nearest-node lookup |
| scikit-learn | Traffic prediction and feature scaling |
| Ridge Regression | Traffic velocity prediction |
| XGBoost | Safety risk prediction |
| A* | Graph-based pathfinding |
| React | Frontend application |
| TypeScript | Frontend development |
| Vite | Frontend development and build tooling |
| Leaflet | Interactive map visualization |
| MLflow | Machine learning experiment tracking |

## Algorithms Used

### 1. Geospatial Nearest-Node Lookup

A BallTree spatial index is used to find nearby road network nodes for geographic coordinates.

This avoids repeatedly comparing a delivery location against every node in the road graph.

### 2. Machine Learning Inference

The traffic and safety models process road-related features and produce predictions used by the routing cost model.

The models support route evaluation; they do not independently determine the final delivery route.

### 3. Dynamic Edge Cost

Each road segment is evaluated using a cost that combines travel-time and safety-related factors.

This allows the routing engine to compare routes based on more than distance alone.

### 4. A* Search Algorithm

A* searches the road graph for a path with a low total cost, using accumulated path cost and a heuristic estimate of the remaining cost.

### 5. Permutation-Based Multi-Stop Optimization

For a small number of delivery stops, the system evaluates possible visiting orders and selects the order with the best route cost.

The number of possible orders grows factorially, so this approach is not intended for large delivery fleets or hundreds of stops.

## Road Network Representation

The road network is represented using a directed NetworkX graph.

Nodes represent geographic road locations, while edges represent road connections between nodes.

Road edges can contain attributes such as:

- Road length
- Road type
- Speed limit, when available
- Lane information, when available
- Road geometry

The original road network is stored in GraphML format. An optimized graph cache is used to reduce the overhead of loading and processing the network during application startup.

## Machine Learning Overview

JustSteer uses machine learning to provide additional context for route evaluation.

### Traffic Prediction

The traffic velocity model estimates road speed from contextual and road-network features.

The predicted speed can be used to estimate travel time for a road segment.

### Safety Risk Prediction

The safety model estimates a risk-related score using road features.

The current model is trained on synthetic road-safety data. Its output should therefore be interpreted as an experimental risk estimate, not as a calibrated probability of an actual accident.

### Experiment Tracking

MLflow is used to track model training experiments, including relevant parameters, metrics, and artifacts.

## Repository Structure

```text
JustSteer/
├── frontend/                     # React + Vite frontend
├── src/                          # Backend application and routing logic
│   ├── api/                      # FastAPI endpoints
│   ├── geocoding/                # Address search and geocoding
│   └── ...                       # Routing, graph and ML modules
├── data/
│   └── road_network/             # Road network source and cache
├── models/                       # Trained ML models and metadata
├── scripts/                      # Setup, compilation and verification scripts
├── docs/                         # Documentation and system evidence
└── README.md
```

The exact module organization may evolve as the project develops.

## Quick Start

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd JustSteer
```

### 2. Set Up the Backend

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the backend dependencies using the requirements file provided in the repository.

### 3. Set Up the Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Prepare the Road Network

Ensure the road network source and optimized graph cache are available in the expected project locations.

If the cache needs to be regenerated, use the graph compilation script provided in the `scripts/` directory.

### 5. Configure and Run

Configure the backend environment variables and start the FastAPI application using the repository's documented development command.

The frontend must be configured to communicate with the backend API.

**Note:** Confirm the exact backend requirements-file location, startup command, environment variables, and graph preparation steps in the repository before publishing this Quick Start as a verified setup guide.

## Project Highlights

- Combines geospatial processing, graph algorithms, and machine learning.
- Uses a road network rather than straight-line distance for route calculation.
- Evaluates routes using both travel-time and safety-related costs.
- Supports multi-stop delivery route planning.
- Provides address search and interactive map visualization.
- Uses an optimized graph cache to reduce road network loading overhead.
- Separates route calculation from frontend map rendering.

## Current Limitations

- Traffic estimates are not equivalent to live traffic sensor data.
- The safety model uses synthetic training data and requires real-world validation.
- Nearest-node mapping may not always identify the most suitable road access point.
- Permutation-based multi-stop optimization becomes expensive as the number of stops increases.
- The system is a route planning prototype, not a complete fleet management or dispatch platform.

## Future Improvements

- Integrate reliable real-time traffic data.
- Validate traffic predictions against real observations.
- Train and evaluate the safety model using suitable real-world datasets.
- Improve road snapping using road-segment projection and access constraints.
- Introduce scalable multi-stop optimization techniques.
- Benchmark route quality against shortest-distance and fastest-route baselines.
- Add route comparison and improved explainability.
- Improve deployment monitoring and production performance validation.

## License

Add the license model selected for the project, such as MIT, Apache-2.0, or a proprietary license.
