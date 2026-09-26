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

export interface MultiStopRouteResponse {
  success: boolean;
  summary: RouteSummary;
  optimized_stops: Coordinate[];
  legs: RouteLeg[];
}

export interface MultiStopRouteRequest {
  origin: Coordinate;
  deliveries: Coordinate[];
  timestamp?: string;
}
