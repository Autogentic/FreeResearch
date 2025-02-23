// ./src/components/IterationLoadingIndicator.js

import React from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';

function IterationLoadingIndicator({ iteration }) {
  return (
    <Box display="flex" alignItems="center" mt={2}>
      <CircularProgress size={24} style={{ marginRight: '0.5rem' }} />
      <Typography variant="body1">Iteration {iteration} in progress...</Typography>
    </Box>
  );
}

export default IterationLoadingIndicator;

