export interface Coordinate {
  latitude: number;
  longitude: number;
}

export interface RouteSummary {
  total_distance_m: number;
  estimated_duration_s: number;
  stop_count: number;
}

export interface RouteLeg {
  start_location: Coordinate;
  end_location: Coordinate;
  distance_m: number;
  estimated_duration_s: number;
  geometry: any; // GeoJSON LineString
}

export interface ExplainabilityMetadata {
  candidate_edges_evaluated: number;
  a_star_expanded_nodes: number;
  evaluated_permutations: number;
  traffic: {
    average_predicted_travel_time_s: number;
  };
  safety: {
    average_risk: number;
  };
  weights: {
    travel_time: number;
    safety: number;
  };
  models: {
    traffic: string;
    safety: string;
  };
}

export interface MultiStopRouteResponse {
  success: boolean;
  summary: RouteSummary;
  optimized_stops: Coordinate[];
  legs: RouteLeg[];
  explainability?: ExplainabilityMetadata;
}

export interface MultiStopRouteRequest {
  origin: Coordinate;
  deliveries: Coordinate[];
  timestamp?: string;
}
