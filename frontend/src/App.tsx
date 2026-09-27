import { useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents, GeoJSON } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Coordinate, MultiStopRouteResponse } from './types';
import { fetchOptimizedRoute } from './api';

// Fix for default leaflet icons in React
import iconUrl from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

const DefaultIcon = L.icon({
  iconUrl,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});
L.Marker.prototype.options.icon = DefaultIcon;

const originIcon = L.icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

function MapEvents({ onMapClick }: { onMapClick: (lat: number, lng: number) => void }) {
  useMapEvents({
    click(e) {
      onMapClick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

export default function App() {
  const [origin, setOrigin] = useState<Coordinate | null>(null);
  const [deliveries, setDeliveries] = useState<Coordinate[]>([]);
  const [routeResult, setRouteResult] = useState<MultiStopRouteResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleMapClick = (latitude: number, longitude: number) => {
    // If no origin, set origin. Otherwise add to deliveries (max 8)
    if (!origin) {
      setOrigin({ latitude, longitude });
    } else if (deliveries.length < 8) {
      setDeliveries([...deliveries, { latitude, longitude }]);
    }
  };

  const handleClear = () => {
    setOrigin(null);
    setDeliveries([]);
    setRouteResult(null);
    setError(null);
  };

  const handleRoute = async () => {
    if (!origin || deliveries.length === 0) return;
    
    setLoading(true);
    setError(null);
    setRouteResult(null);

    try {
      const res = await fetchOptimizedRoute({ origin, deliveries });
      setRouteResult(res);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const formatDuration = (sec: number) => {
    const mins = Math.floor(sec / 60);
    return `${mins} min`;
  };

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', fontFamily: 'sans-serif' }}>
      
      {/* Sidebar */}
      <div style={{ width: '350px', padding: '20px', display: 'flex', flexDirection: 'column', gap: '15px', backgroundColor: '#f8f9fa', overflowY: 'auto' }}>
        <h2>JustSteer Routing</h2>
        
        <p style={{ fontSize: '14px', color: '#555' }}>
          1. Click the map to set the <strong>Origin (Green)</strong>.<br />
          2. Click to add <strong>Destinations (Blue)</strong> (max 8).<br />
          3. Click 'Calculate Optimized Route'.
        </p>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button 
            onClick={handleRoute}
            disabled={!origin || deliveries.length === 0 || loading}
            style={{ flex: 1, padding: '10px', backgroundColor: '#007bff', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
          >
            {loading ? "Calculating..." : "Calculate Route"}
          </button>
          <button 
            onClick={handleClear}
            style={{ padding: '10px', backgroundColor: '#dc3545', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
          >
            Clear
          </button>
        </div>

        {error && (
          <div style={{ padding: '10px', backgroundColor: '#f8d7da', color: '#721c24', borderRadius: '4px' }}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {routeResult && (
          <div style={{ padding: '15px', backgroundColor: '#d4edda', color: '#155724', borderRadius: '4px' }}>
            <h3>Route Summary</h3>
            <p><strong>Distance:</strong> {(routeResult.summary.total_distance_m / 1000).toFixed(2)} km</p>
            <p><strong>ETA (Duration):</strong> {formatDuration(routeResult.summary.estimated_duration_s)}</p>
            <p><strong>Stops:</strong> {routeResult.summary.stop_count}</p>
            
            <h4 style={{marginTop: '15px'}}>Optimized Order:</h4>
            <ul style={{ paddingLeft: '20px', margin: '5px 0' }}>
              <li>Origin</li>
              {routeResult.optimized_stops.map((_, idx) => (
                <li key={idx}>Stop {idx + 1}</li>
              ))}
            </ul>
          </div>
        )}

        {routeResult && routeResult.explainability && (
          <div style={{ padding: '15px', backgroundColor: '#e2e3e5', color: '#383d41', borderRadius: '4px', fontSize: '14px' }}>
            <h3 style={{ marginTop: 0 }}>How the route was calculated</h3>
            
            <div style={{ marginBottom: '10px' }}>
              <div>✓ Candidate roads: {routeResult.explainability.candidate_edges_evaluated.toLocaleString()}</div>
              <div>✓ ML predictions generated</div>
              <div>✓ A* nodes explored: {routeResult.explainability.a_star_expanded_nodes.toLocaleString()}</div>
              <div>✓ Stop sequences evaluated: {routeResult.explainability.evaluated_permutations}</div>
            </div>
            
            <div style={{ marginBottom: '10px' }}>
              <strong>Traffic / Travel Time</strong><br/>
              Avg. predicted time: {routeResult.explainability.traffic.average_predicted_travel_time_s.toFixed(1)} s
            </div>
            
            <div style={{ marginBottom: '10px' }}>
              <strong>Safety</strong><br/>
              Avg. risk: {routeResult.explainability.safety.average_risk.toFixed(2)}
            </div>
            
            <div>
              <strong>Cost weighting</strong><br/>
              Travel Time: {(routeResult.explainability.weights.travel_time * 100).toFixed(0)} %<br/>
              Safety: {(routeResult.explainability.weights.safety * 100).toFixed(0)} %
            </div>
          </div>
        )}
      </div>

      {/* Map Area */}
      <div style={{ flex: 1 }}>
        <MapContainer 
          center={[13.0827, 80.2707]} // Chennai default
          zoom={13} 
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="http://osm.org/copyright">OSM</a>'
          />
          <MapEvents onMapClick={handleMapClick} />

          {origin && (
            <Marker position={[origin.latitude, origin.longitude]} icon={originIcon}>
              <Popup>Origin</Popup>
            </Marker>
          )}

          {deliveries.map((del, idx) => (
            <Marker key={idx} position={[del.latitude, del.longitude]}>
              <Popup>Destination {idx + 1}</Popup>
            </Marker>
          ))}

          {routeResult && routeResult.legs.map((leg, idx) => (
            leg.geometry && (
              <GeoJSON 
                key={`leg-${idx}`} 
                data={leg.geometry} 
                style={{
                  color: '#FF0000',
                  weight: 4,
                  opacity: 0.7
                }} 
              />
            )
          ))}
        </MapContainer>
      </div>
    </div>
  );
}
