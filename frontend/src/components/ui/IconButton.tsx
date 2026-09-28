import React from 'react';
import styles from './IconButton.module.css';

export interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  icon: React.ReactNode;
  'aria-label': string; // Enforce accessibility
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
}

export const IconButton: React.FC<IconButtonProps> = ({
  icon,
  variant = 'ghost',
  size = 'md',
  className = '',
  disabled,
  ...props
}) => {
  const baseClass = styles.iconButton;
  const variantClass = styles[variant];
  const sizeClass = styles[size];

  return (
    <button
      className={`${baseClass} ${variantClass} ${sizeClass} ${className}`.trim()}
      disabled={disabled}
      {...props}
    >
      {icon}
    </button>
  );
};
