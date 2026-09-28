import React from 'react';
import styles from './MapPlaceholder.module.css';

export const MapPlaceholder: React.FC = () => {
  return (
    <div className={styles.container}>
      <div className={styles.content}>
        <div className={styles.icon}>🗺️</div>
        <h3 className={styles.title}>Interactive Map</h3>
        <p className={styles.subtitle}>
          The interactive route map will appear here during planning.
        </p>
      </div>
    </div>
  );
};
