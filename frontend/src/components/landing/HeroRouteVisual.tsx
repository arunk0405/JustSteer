import React from 'react';
import styles from './HeroRouteVisual.module.css';

export const HeroRouteVisual: React.FC = () => {
  return (
    <div className={styles.visualContainer} aria-hidden="true">
      {/* Background map grid effect */}
      <div className={styles.mapBackground}></div>

      {/* SVG Route Line */}
      <svg className={styles.routeSvg} viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
        <path 
          d="M 80,320 C 120,300 160,350 200,280 C 240,210 180,180 250,140 C 320,100 350,150 340,80" 
          fill="none" 
          stroke="var(--color-primary)" 
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray="8 8"
          className={styles.routePath}
        />
      </svg>

      {/* Markers */}
      <div className={`${styles.marker} ${styles.originMarker}`} style={{ left: '80px', top: '320px' }}>
        <div className={styles.markerInner}></div>
      </div>
      
      <div className={`${styles.marker} ${styles.deliveryMarker}`} style={{ left: '200px', top: '280px' }}>
        <div className={styles.markerInner}>1</div>
      </div>
      
      <div className={`${styles.marker} ${styles.deliveryMarker}`} style={{ left: '250px', top: '140px' }}>
        <div className={styles.markerInner}>2</div>
      </div>
      
      <div className={`${styles.marker} ${styles.deliveryMarker}`} style={{ left: '340px', top: '80px' }}>
        <div className={styles.markerInner}>3</div>
      </div>

      {/* Info Card Overlay */}
      <div className={styles.infoCard}>
        <div className={styles.infoCardHeader}>OPTIMIZED ROUTE</div>
        <div className={styles.infoCardStats}>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Stops</span>
            <span className={styles.statValue}>3</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Distance</span>
            <span className={styles.statValue}>12.5 km</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Time</span>
            <span className={styles.statValue}>~32 min</span>
          </div>
        </div>
      </div>
    </div>
  );
};
