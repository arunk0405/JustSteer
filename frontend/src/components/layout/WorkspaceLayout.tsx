import React from 'react';
import styles from './WorkspaceLayout.module.css';

export interface WorkspaceLayoutProps {
  sidebar: React.ReactNode;
  map: React.ReactNode;
}

export const WorkspaceLayout: React.FC<WorkspaceLayoutProps> = ({ sidebar, map }) => {
  return (
    <div className={styles.workspace}>
      <aside className={styles.sidebar}>
        {sidebar}
      </aside>
      <main className={styles.mapArea}>
        {map}
      </main>
    </div>
  );
};
