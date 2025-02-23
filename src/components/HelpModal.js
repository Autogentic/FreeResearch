// ./src/components/HelpModal.js

import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button
} from '@mui/material';

function HelpModal({ open, onClose }) {
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>FreeResearch Help</DialogTitle>
      <DialogContent dividers>
        <DialogContentText component="div">
          <p>
            Welcome to FreeResearch! This application performs multi-agent research, and this
            dashboard provides full real-time visibility.
          </p>
          <strong>Key Features:</strong>
          <ul>
            <li><strong>Research Form:</strong> Enter a subject to initiate research.</li>
            <li>
              <strong>Live Dashboard:</strong> View research progress, logs, fetched website links, a
              dynamic knowledge graph, agent conversation, iteration progress, and resource usage.
            </li>
            <li>
              <strong>Pause/Resume:</strong> Control the research flow by pausing/resuming polling.
            </li>
            <li>
              <strong>Export Logs:</strong> Download (filtered) logs in CSV format.
            </li>
            <li>
              <strong>Theme Toggle:</strong> Switch between light and dark modes.
            </li>
            <li>
              <strong>Dashboard Settings:</strong> Customize which panels are visible; settings are
              saved automatically.
            </li>
            <li>
              <strong>Resource Monitor:</strong> See current system resource metrics.
            </li>
          </ul>
          <strong>Usage Tips:</strong>
          <ul>
            <li>Use "Dashboard Settings" to tailor your view.</li>
            <li>Pause research when you need to examine logs or adjust settings.</li>
            <li>Use "Export Logs" to download log data for troubleshooting or reporting.</li>
          </ul>
          <p>
            For further assistance, contact support@freeresearch.com.
          </p>
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="primary">Got it</Button>
      </DialogActions>
    </Dialog>
  );
}

export default HelpModal;

