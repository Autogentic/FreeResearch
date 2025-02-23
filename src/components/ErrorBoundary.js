// ./src/components/ErrorBoundary.js

import React from 'react';
import { Alert, Button, Box } from '@mui/material';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <Box m={2}>
          <Alert severity="error">
            Something went wrong. Please try reloading the application.
          </Alert>
          <Box mt={2} display="flex" justifyContent="center">
            <Button variant="contained" color="primary" onClick={this.handleReload}>
              Reload App
            </Button>
          </Box>
        </Box>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;

