// ./src/components/ResourceMonitorDisplay.js

import React, { useState, useEffect } from 'react';
import { Box, Card, CardContent, Typography, LinearProgress } from '@mui/material';
import axios from 'axios';

function ResourceMonitorDisplay() {
  const [resources, setResources] = useState(null);

  useEffect(() => {
    const intervalId = setInterval(() => {
      axios
        .get('/api/resources')
        .then((res) => setResources(res.data))
        .catch((err) => console.error('Error fetching resources:', err));
    }, 5000);

    return () => clearInterval(intervalId);
  }, []);

  if (!resources) {
    return <LinearProgress />;
  }

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6">Resource Monitor</Typography>
        <Typography variant="body2">
          <strong>CPU Usage:</strong> {resources.cpu_usage}%
        </Typography>
        <Typography variant="body2">
          <strong>Memory Usage:</strong> {resources.memory_usage}%
        </Typography>
        <Typography variant="body2">
          <strong>API Calls:</strong> {resources.api_calls}
        </Typography>
        <Typography variant="body2">
          <strong>Active Tasks:</strong> {resources.active_tasks}
        </Typography>
      </CardContent>
    </Card>
  );
}

export default ResourceMonitorDisplay;

