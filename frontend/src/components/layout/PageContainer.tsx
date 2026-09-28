import React from 'react';
import styles from './PageContainer.module.css';

export interface PageContainerProps {
  children: React.ReactNode;
  className?: string;
}

export const PageContainer: React.FC<PageContainerProps> = ({ children, className = '' }) => {
  return (
    <div className={`${styles.container} ${className}`.trim()}>
      {children}
    </div>
  );
};
