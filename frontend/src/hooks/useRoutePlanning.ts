import { useState, useCallback } from 'react';
import type { MultiStopRouteRequest, Coordinate, MultiStopRouteResponse } from '../types';
import { fetchOptimizedRoute } from '../api';

export interface CoordinateInputState {
  latitude: string;
  longitude: string;
}

export interface DeliveryInput {
  id: string;
  coordinate: CoordinateInputState;
}

export interface ValidationErrors {
  origin?: string;
  deliveries?: Record<string, string>;
  general?: string;
}

export type RouteExecutionState = 'idle' | 'loading' | 'success' | 'error';

export function useRoutePlanning() {
  const [origin, setOrigin] = useState<CoordinateInputState>({ latitude: '', longitude: '' });
  const [deliveries, setDeliveries] = useState<DeliveryInput[]>([]);
  const [timestamp, setTimestamp] = useState<string>('');
  
  const [errors, setErrors] = useState<ValidationErrors>({});
  
  const [executionState, setExecutionState] = useState<RouteExecutionState>('idle');
  const [routeResult, setRouteResult] = useState<MultiStopRouteResponse | null>(null);
  const [routeError, setRouteError] = useState<string | null>(null);

  const generateId = () => Math.random().toString(36).substring(2, 9);

  const handleOriginChange = (field: keyof CoordinateInputState, value: string) => {
    setOrigin(prev => ({ ...prev, [field]: value }));
    clearErrorsAndResult();
  };

  const addDelivery = (initialCoord?: CoordinateInputState) => {
    if (deliveries.length >= 8) return;
    setDeliveries(prev => [...prev, { 
      id: generateId(), 
      coordinate: initialCoord || { latitude: '', longitude: '' } 
    }]);
    clearErrorsAndResult();
  };

  const updateDelivery = (id: string, field: keyof CoordinateInputState, value: string) => {
    setDeliveries(prev => 
      prev.map(d => d.id === id ? { ...d, coordinate: { ...d.coordinate, [field]: value } } : d)
    );
    clearErrorsAndResult();
  };

  const removeDelivery = (id: string) => {
    setDeliveries(prev => prev.filter(d => d.id !== id));
    clearErrorsAndResult();
  };

  const handleTimestampChange = (val: string) => {
    setTimestamp(val);
    clearErrorsAndResult();
  };

  const clearErrorsAndResult = useCallback(() => {
    setErrors({});
    if (executionState !== 'idle') {
      setExecutionState('idle');
      setRouteResult(null);
      setRouteError(null);
    }
  }, [executionState]);

  const resetRoute = () => {
    setOrigin({ latitude: '', longitude: '' });
    setDeliveries([]);
    setTimestamp('');
    setErrors({});
    setExecutionState('idle');
    setRouteResult(null);
    setRouteError(null);
  };

  // Map Interactions
  const handleMapClick = (coord: Coordinate) => {
    const latStr = coord.latitude.toFixed(6);
    const lngStr = coord.longitude.toFixed(6);
    
    if (!origin.latitude && !origin.longitude) {
      setOrigin({ latitude: latStr, longitude: lngStr });
      clearErrorsAndResult();
    } else if (deliveries.length < 8) {
      addDelivery({ latitude: latStr, longitude: lngStr });
    }
  };

  const handleOriginDrag = (coord: Coordinate) => {
    setOrigin({ latitude: coord.latitude.toFixed(6), longitude: coord.longitude.toFixed(6) });
    clearErrorsAndResult();
  };

  const handleDeliveryDrag = (id: string, coord: Coordinate) => {
    updateDelivery(id, 'latitude', coord.latitude.toFixed(6));
    updateDelivery(id, 'longitude', coord.longitude.toFixed(6));
  };

  // Validation
  const validateCoordinate = (latStr: string, lngStr: string, isOptional: boolean = false): { valid: boolean; coord?: Coordinate; error?: string } => {
    const latTrimmed = latStr.trim();
    const lngTrimmed = lngStr.trim();
    
    if (isOptional && !latTrimmed && !lngTrimmed) return { valid: true };
    
    if (!latTrimmed || !lngTrimmed) {
      return { valid: false, error: 'Both latitude and longitude are required.' };
    }

    const lat = Number(latTrimmed);
    const lng = Number(lngTrimmed);

    if (isNaN(lat) || isNaN(lng)) {
      return { valid: false, error: 'Coordinates must be valid numbers.' };
    }

    if (lat < -90 || lat > 90) {
      return { valid: false, error: 'Latitude must be between -90 and 90.' };
    }

    if (lng < -180 || lng > 180) {
      return { valid: false, error: 'Longitude must be between -180 and 180.' };
    }

    return { valid: true, coord: { latitude: lat, longitude: lng } };
  };

  const executeRoutePlanning = async () => {
    const newErrors: ValidationErrors = { deliveries: {} };
    let hasError = false;

    // Validate Origin
    const originCheck = validateCoordinate(origin.latitude, origin.longitude);
    if (!originCheck.valid) {
      newErrors.origin = originCheck.error;
      hasError = true;
    }

    // Validate Deliveries
    if (deliveries.length === 0) {
      newErrors.general = 'Add at least one delivery location.';
      hasError = true;
    } else if (deliveries.length > 8) {
      newErrors.general = 'Maximum 8 delivery locations allowed.';
      hasError = true;
    }

    const parsedDeliveries: Coordinate[] = [];
    deliveries.forEach(d => {
      const check = validateCoordinate(d.coordinate.latitude, d.coordinate.longitude);
      if (!check.valid) {
        newErrors.deliveries![d.id] = check.error!;
        hasError = true;
      } else if (check.coord) {
        parsedDeliveries.push(check.coord);
      }
    });

    if (hasError) {
      setErrors(newErrors);
      setExecutionState('error');
      setRouteError('Please correct the validation errors before generating.');
      return;
    }

    // Success validation
    setErrors({});
    const request: MultiStopRouteRequest = {
      origin: originCheck.coord!,
      deliveries: parsedDeliveries,
    };
    if (timestamp.trim()) {
      request.timestamp = timestamp.trim();
    }
    
    // Execute API Call
    setExecutionState('loading');
    setRouteError(null);
    setRouteResult(null);
    
    try {
      const response = await fetchOptimizedRoute(request);
      setRouteResult(response);
      setExecutionState('success');
    } catch (err: any) {
      setExecutionState('error');
      if (err.message) {
        setRouteError(err.message);
      } else {
        setRouteError('Routing service encountered an unexpected error. Please try again.');
      }
    }
  };

  // Selectors for map
  const parsedOrigin = validateCoordinate(origin.latitude, origin.longitude).coord || null;
  const parsedDeliveries = deliveries.map(d => ({
    id: d.id,
    coordinate: validateCoordinate(d.coordinate.latitude, d.coordinate.longitude).coord || null
  }));

  return {
    origin,
    deliveries,
    timestamp,
    errors,
    
    executionState,
    routeResult,
    routeError,
    
    parsedOrigin,
    parsedDeliveries,
    
    handleOriginChange,
    addDelivery,
    updateDelivery,
    removeDelivery,
    handleTimestampChange,
    resetRoute,
    
    handleMapClick,
    handleOriginDrag,
    handleDeliveryDrag,
    
    executeRoutePlanning,
    isAtMaxDeliveries: deliveries.length >= 8
  };
}
