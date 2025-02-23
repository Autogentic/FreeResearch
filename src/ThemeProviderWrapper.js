// ./src/ThemeProviderWrapper.js

import React, { useState } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import ErrorBoundary from './components/ErrorBoundary';
import App from './App';

function ThemeProviderWrapper() {
  const [mode, setMode] = useState('light');

  // Create the MUI theme with dynamic light/dark mode
  const theme = createTheme({
    palette: {
      mode,
      primary: { main: '#1976d2' },
      background: { default: mode === 'light' ? '#eef2f5' : '#121212' },
    },
    typography: { fontFamily: 'Roboto, sans-serif' },
  });

  const toggleTheme = () => {
    setMode(prev => (prev === 'light' ? 'dark' : 'light'));
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ErrorBoundary>
        <App toggleTheme={toggleTheme} mode={mode} />
      </ErrorBoundary>
    </ThemeProvider>
  );
}

export default ThemeProviderWrapper;

