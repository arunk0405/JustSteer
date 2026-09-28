import React from 'react';
import styles from './StatusMessage.module.css';

export interface StatusMessageProps {
  children: React.ReactNode;
  variant?: 'success' | 'error' | 'warning' | 'info';
  className?: string;
}

export const StatusMessage: React.FC<StatusMessageProps> = ({ 
  children, 
  variant = 'info', 
  className = '' 
}) => {
  return (
    <div className={`${styles.statusMessage} ${styles[variant]} ${className}`.trim()}>
      {children}
    </div>
  );
};
