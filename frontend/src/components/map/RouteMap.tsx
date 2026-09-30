import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import styles from './RouteMap.module.css';
import type { Coordinate, MultiStopRouteResponse } from '../../types';

// Custom icons using standard Leaflet DivIcon
const createCustomIcon = (text: string, color: string) => {
  return L.divIcon({
    className: styles.customMarker,
    html: `<div style="background-color: ${color};" class="${styles.markerContent}">${text}</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });
};

const OriginIcon = createCustomIcon('★', 'var(--color-success)');

interface MapClickHandlerProps {
  onMapClick?: (latlng: L.LatLng) => void;
}

const MapClickHandler: React.FC<MapClickHandlerProps> = ({ onMapClick }) => {
  useMapEvents({
    click(e) {
      if (onMapClick) {
        onMapClick(e.latlng);
      }
    },
  });
  return null;
};

interface MapBoundsFitterProps {
  origin: Coordinate | null;
  deliveries: Coordinate[];
  routeResult: MultiStopRouteResponse | null;
}

const MapBoundsFitter: React.FC<MapBoundsFitterProps> = ({ origin, deliveries, routeResult }) => {
  const map = useMap();

  useEffect(() => {
    const bounds = L.latLngBounds([]);
    
    if (routeResult?.legs) {
      routeResult.legs.forEach(leg => {
        if (leg.geometry?.coordinates) {
          leg.geometry.coordinates.forEach((coord: [number, number]) => {
            // Backend currently returns [lat, lng] instead of GeoJSON standard [lng, lat].
            bounds.extend([coord[0], coord[1]]);
          });
        }
      });
    } else {
      if (origin) bounds.extend([origin.latitude, origin.longitude]);
      deliveries.forEach(d => {
        if (d) bounds.extend([d.latitude, d.longitude]);
      });
    }

    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [origin, deliveries, routeResult, map]);

  return null;
};

export interface RouteMapProps {
  origin: Coordinate | null;
  deliveries: Array<{ id: string; coordinate: Coordinate | null }>;
  routeResult: MultiStopRouteResponse | null;
  onMapClick?: (coord: Coordinate) => void;
  onOriginDragEnd?: (coord: Coordinate) => void;
  onDeliveryDragEnd?: (id: string, coord: Coordinate) => void;
}

export const RouteMap: React.FC<RouteMapProps> = ({
  origin,
  deliveries,
  routeResult,
  onMapClick,
  onOriginDragEnd,
  onDeliveryDragEnd
}) => {
  // Chennai bounds for default view
  const defaultCenter: L.LatLngTuple = [13.0827, 80.2707];
  
  const handleMapClick = (latlng: L.LatLng) => {
    if (onMapClick) {
      onMapClick({ latitude: latlng.lat, longitude: latlng.lng });
    }
  };

  return (
    <div className={styles.mapWrapper}>
      <MapContainer 
        center={defaultCenter} 
        zoom={12} 
        className={styles.mapContainer}
        zoomControl={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        <MapClickHandler onMapClick={handleMapClick} />
        <MapBoundsFitter origin={origin} deliveries={deliveries.map(d => d.coordinate).filter((c): c is Coordinate => c !== null)} routeResult={routeResult} />

        {/* Origin Marker */}
        {origin && (
          <Marker 
            position={[origin.latitude, origin.longitude]} 
            icon={OriginIcon}
            draggable={true}
            eventHandlers={{
              dragend: (e) => {
                const marker = e.target;
                const position = marker.getLatLng();
                if (onOriginDragEnd) {
                  onOriginDragEnd({ latitude: position.lat, longitude: position.lng });
                }
              },
            }}
          >
            <Popup>Start Location</Popup>
          </Marker>
        )}

        {/* Delivery Markers */}
        {deliveries.map((delivery, index) => {
          if (!delivery.coordinate) return null;
          
          let displayNum = index + 1;
          let isOptimized = false;
          
          // If we have a routeResult with optimized_stops, find the order
          if (routeResult?.optimized_stops) {
            const optIndex = routeResult.optimized_stops.findIndex(
              opt => Math.abs(opt.latitude - delivery.coordinate!.latitude) < 0.0001 &&
                     Math.abs(opt.longitude - delivery.coordinate!.longitude) < 0.0001
            );
            if (optIndex !== -1) {
              displayNum = optIndex + 1;
              isOptimized = true;
            }
          }

          const DeliveryIcon = createCustomIcon(
            displayNum.toString(),
            isOptimized ? 'var(--color-primary-dark)' : 'var(--color-primary)'
          );

          return (
            <Marker
              key={delivery.id}
              position={[delivery.coordinate.latitude, delivery.coordinate.longitude]}
              icon={DeliveryIcon}
              draggable={true}
              eventHandlers={{
                dragend: (e) => {
                  const marker = e.target;
                  const position = marker.getLatLng();
                  if (onDeliveryDragEnd) {
                    onDeliveryDragEnd(delivery.id, { latitude: position.lat, longitude: position.lng });
                  }
                },
              }}
            >
              <Popup>Delivery {displayNum}</Popup>
            </Marker>
          );
        })}

        {/* Route Polylines */}
        {routeResult?.legs && routeResult.legs.map((leg, idx) => {
          if (!leg.geometry?.coordinates) return null;
          // Backend currently returns [lat, lng]
          const positions: L.LatLngTuple[] = leg.geometry.coordinates.map((coord: [number, number]) => [coord[0], coord[1]]);
          
          return (
            <Polyline 
              key={`leg-${idx}`} 
              positions={positions} 
              color="var(--color-primary)" 
              weight={5} 
              opacity={0.8}
            />
          );
        })}
      </MapContainer>
      
      {!origin && (
        <div className={styles.mapInstruction}>
          Click on the map to select your starting location.
        </div>
      )}
    </div>
  );
};
