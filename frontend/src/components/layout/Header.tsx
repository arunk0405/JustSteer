import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import styles from './Header.module.css';
import { Button } from '../ui/Button';

export const Header: React.FC = () => {
  const location = useLocation();

  const isCurrentPath = (path: string) => {
    return location.pathname === path;
  };

  return (
    <header className={styles.header}>
      <div className={styles.container}>
        <div className={styles.brand}>
          <Link to="/" className={styles.logo}>
            <div className={styles.logoIcon}>◉</div>
            <span className={styles.logoText}>JustSteer</span>
          </Link>
        </div>

        <nav className={styles.nav}>
          <Link 
            to="/" 
            className={`${styles.navLink} ${isCurrentPath('/') ? styles.active : ''}`}
          >
            Home
          </Link>
          <Link 
            to="/how-it-works" 
            className={`${styles.navLink} ${isCurrentPath('/how-it-works') ? styles.active : ''}`}
          >
            How It Works
          </Link>
        </nav>

        <div className={styles.actions}>
          <Link to="/plan" tabIndex={-1}>
            <Button variant="primary" size="sm">
              Plan Route
            </Button>
          </Link>
        </div>
      </div>
    </header>
  );
};
