// ./src/components/ResearchStatus.js

import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';

function ResearchStatus({ subject, isLoading, logs, finalReport }) {
  // Determine status
  let status;
  if (finalReport) {
    status = 'Completed';
  } else if (isLoading) {
    status = 'In Progress';
  } else {
    status = 'Idle';
  }

  // Choose chip color
  let chipColor;
  if (status === 'Completed') {
    chipColor = 'success';
  } else if (status === 'In Progress') {
    chipColor = 'warning';
  } else {
    chipColor = 'default';
  }

  // Last update
  const lastUpdate = logs && logs.length > 0 ? logs[logs.length - 1].timestamp : 'N/A';

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Research Information
        </Typography>
        <Box mb={1}>
          <Typography variant="body1">
            <strong>Subject:</strong> {subject}
          </Typography>
        </Box>
        <Box mb={1} display="flex" alignItems="center" gap={1}>
          <Typography variant="body1">
            <strong>Status:</strong>
          </Typography>
          <Chip label={status} color={chipColor} size="small" />
        </Box>
        <Box>
          <Typography variant="body2" color="textSecondary">
            <strong>Last Update:</strong> {lastUpdate}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
}

export default ResearchStatus;

