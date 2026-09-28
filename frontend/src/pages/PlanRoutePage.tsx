import React from 'react';
import { WorkspaceLayout } from '../components/layout/WorkspaceLayout';
import { RouteMap } from '../components/map/RouteMap';
import { Button } from '../components/ui/Button';
import { IconButton } from '../components/ui/IconButton';
import { StatusMessage } from '../components/ui/StatusMessage';
import { useRoutePlanning } from '../hooks/useRoutePlanning';
import styles from './PlanRoutePage.module.css';

export const PlanRoutePage: React.FC = () => {
  const {
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
    isAtMaxDeliveries
  } = useRoutePlanning();

  const handleUseMyLocation = () => {
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          handleOriginChange('latitude', position.coords.latitude.toString());
          handleOriginChange('longitude', position.coords.longitude.toString());
        },
        () => {
          alert('Unable to access your location. Please enter coordinates manually.');
        },
        { timeout: 5000 }
      );
    } else {
      alert('Geolocation is not supported by your browser.');
    }
  };

  const sidebar = (
    <div className={styles.sidebarContainer}>
      <div className={styles.header}>
        <h2 className={styles.title}>Plan Your Route</h2>
        <p className={styles.subtitle}>Enter your starting location and delivery stops.</p>
      </div>
      
      <div className={styles.formContent}>
        {/* SECTION 1 - ORIGIN */}
        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <div className={styles.sectionNumber}>1</div>
            <div>
              <h3 className={styles.sectionTitle}>Start Location</h3>
              <p className={styles.sectionDesc}>Where should the route begin?</p>
            </div>
          </div>
          
          <div className={styles.inputGroup}>
            <div className={styles.coordinateRow}>
              <div className={styles.field}>
                <label className={styles.label}>Latitude</label>
                <input 
                  type="text"
                  className={styles.input}
                  placeholder="e.g. 13.0807843"
                  value={origin.latitude}
                  onChange={(e) => handleOriginChange('latitude', e.target.value)}
                />
              </div>
              <div className={styles.field}>
                <label className={styles.label}>Longitude</label>
                <input 
                  type="text"
                  className={styles.input}
                  placeholder="e.g. 80.2651669"
                  value={origin.longitude}
                  onChange={(e) => handleOriginChange('longitude', e.target.value)}
                />
              </div>
            </div>
            {errors.origin && <StatusMessage variant="error" className={styles.inlineError}>{errors.origin}</StatusMessage>}
            <button className={styles.textButton} onClick={handleUseMyLocation} type="button">
              Use My Location
            </button>
          </div>
        </section>

        {/* SECTION 2 - DELIVERIES */}
        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <div className={styles.sectionNumber}>2</div>
            <div>
              <h3 className={styles.sectionTitle}>Delivery Locations</h3>
              <p className={styles.sectionDesc}>Add up to 8 destinations.</p>
            </div>
          </div>
          
          <div className={styles.deliveriesList}>
            {deliveries.map((delivery, index) => (
              <div key={delivery.id} className={styles.deliveryCard}>
                <div className={styles.deliveryCardHeader}>
                  <span className={styles.deliveryCardTitle}>Stop {index + 1}</span>
                  <IconButton 
                    icon={<span>✕</span>} 
                    aria-label={`Remove delivery stop ${index + 1}`} 
                    size="sm" 
                    variant="ghost"
                    onClick={() => removeDelivery(delivery.id)}
                    title="Remove stop"
                  />
                </div>
                <div className={styles.coordinateRow}>
                  <div className={styles.field}>
                    <label className={styles.label}>Latitude</label>
                    <input 
                      type="text"
                      className={styles.input}
                      placeholder="Latitude"
                      value={delivery.coordinate.latitude}
                      onChange={(e) => updateDelivery(delivery.id, 'latitude', e.target.value)}
                    />
                  </div>
                  <div className={styles.field}>
                    <label className={styles.label}>Longitude</label>
                    <input 
                      type="text"
                      className={styles.input}
                      placeholder="Longitude"
                      value={delivery.coordinate.longitude}
                      onChange={(e) => updateDelivery(delivery.id, 'longitude', e.target.value)}
                    />
                  </div>
                </div>
                {errors.deliveries?.[delivery.id] && (
                  <StatusMessage variant="error" className={styles.inlineError}>
                    {errors.deliveries[delivery.id]}
                  </StatusMessage>
                )}
              </div>
            ))}
          </div>

          {deliveries.length === 0 && (
            <div className={styles.emptyState}>No delivery stops added yet.</div>
          )}

          <div className={styles.addStopWrapper}>
            <Button 
              variant="secondary" 
              onClick={() => addDelivery()} 
              disabled={isAtMaxDeliveries}
              className={styles.addStopButton}
            >
              + Add Another Stop
            </Button>
            {isAtMaxDeliveries && (
              <div className={styles.maxStopsWarning}>Maximum of 8 delivery stops reached.</div>
            )}
          </div>
        </section>

        {/* SECTION 3 - TIME */}
        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <div className={styles.sectionNumber}>3</div>
            <div>
              <h3 className={styles.sectionTitle}>Route Time</h3>
              <p className={styles.sectionDesc}>Optional: provide the time conditions for routing.</p>
            </div>
          </div>
          <div className={styles.field}>
            <label className={styles.label}>Departure Time</label>
            <input 
              type="datetime-local" 
              className={styles.input} 
              value={timestamp}
              onChange={(e) => handleTimestampChange(e.target.value)}
            />
          </div>
        </section>

        {/* VALIDATION / ERRORS */}
        {errors.general && (
          <StatusMessage variant="error" className={styles.generalError}>
            {errors.general}
          </StatusMessage>
        )}

        {/* LOADING / API ERROR */}
        {executionState === 'loading' && (
          <StatusMessage variant="info" className={styles.generalError}>
            Optimizing route...
          </StatusMessage>
        )}
        
        {executionState === 'error' && routeError && (
          <StatusMessage variant="error" className={styles.generalError}>
            {routeError === 'Unprocessable Entity' 
              ? 'Unable to generate this route. Please check your locations and try again.'
              : routeError}
          </StatusMessage>
        )}

        {/* SUCCESSFUL RESPONSE */}
        {executionState === 'success' && routeResult && (
          <div className={styles.previewContainer}>
            <StatusMessage variant="success" className={styles.readyMessage}>
              Route Generated ✓
            </StatusMessage>
            <div className={styles.previewPayload}>
              <h4>Route Summary</h4>
              <p>Distance: {(routeResult.summary.total_distance_m / 1000).toFixed(2)} km</p>
              <p>Estimated Time: {Math.round(routeResult.summary.estimated_duration_s / 60)} mins</p>
              <p>Stops: {routeResult.summary.stop_count}</p>
              
              <h4 style={{marginTop: '12px'}}>Optimized Stop Order</h4>
              <ul style={{paddingLeft: '20px', margin: '8px 0 0 0'}}>
                <li>● Start</li>
                {routeResult.optimized_stops.map((_, idx) => (
                  <li key={idx}>↓ Stop {idx + 1}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>

      <div className={styles.footerActions}>
        <Button 
          variant="primary" 
          size="lg" 
          className={styles.generateButton} 
          onClick={executeRoutePlanning}
          disabled={executionState === 'loading'}
        >
          {executionState === 'loading' ? 'Optimizing Route...' : 'Generate Optimized Route →'}
        </Button>
        <button className={styles.textButton} onClick={resetRoute} type="button">
          Clear Route
        </button>
      </div>
    </div>
  );

  return (
    <WorkspaceLayout 
      sidebar={sidebar}
      map={
        <div style={{ position: 'relative', width: '100%', height: '100%' }}>
          <RouteMap 
            origin={parsedOrigin}
            deliveries={parsedDeliveries}
            routeResult={routeResult}
            onMapClick={handleMapClick}
            onOriginDragEnd={handleOriginDrag}
            onDeliveryDragEnd={handleDeliveryDrag}
          />
          {executionState === 'loading' && (
            <div className={styles.loadingOverlay}>
              <div className={styles.loadingCard}>
                <div className={styles.spinner}></div>
                <h3>Optimizing route...</h3>
                <p>ML predictions</p>
                <p>Route optimization</p>
                <p>Preparing final route</p>
              </div>
            </div>
          )}
        </div>
      }
    />
  );
};
