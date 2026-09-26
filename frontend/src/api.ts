import { MultiStopRouteRequest, MultiStopRouteResponse } from "./types";

const API_BASE_URL = "http://127.0.0.1:8000";

export async function fetchOptimizedRoute(request: MultiStopRouteRequest): Promise<MultiStopRouteResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/route/multi-stop`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    let errorMsg = "Unable to connect to routing service.";
    try {
      const errorData = await response.json();
      errorMsg = errorData.detail || errorMsg;
    } catch (e) {
      // ignore parsing error
    }
    throw new Error(errorMsg);
  }

  return response.json();
}
