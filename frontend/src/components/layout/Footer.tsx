import React from 'react';
import { Link } from 'react-router-dom';
import styles from './Footer.module.css';

export const Footer: React.FC = () => {
  return (
    <footer className={styles.footer}>
      <div className={styles.container}>
        <div className={styles.brandCol}>
          <div className={styles.logo}>
            <div className={styles.logoIcon}>◉</div>
            <span className={styles.logoText}>JustSteer</span>
          </div>
          <p className={styles.tagline}>
            ML-powered multi-stop route optimization
          </p>
        </div>
        
        <div className={styles.linksCol}>
          <Link to="/" className={styles.link}>Home</Link>
          <Link to="/plan" className={styles.link}>Plan Route</Link>
          <Link to="/how-it-works" className={styles.link}>How It Works</Link>
        </div>
      </div>
      
      <div className={styles.bottomBar}>
        <div className={styles.container}>
          <span className={styles.disclaimer}>
            Academic / Demonstration Project
          </span>
        </div>
      </div>
    </footer>
  );
};
