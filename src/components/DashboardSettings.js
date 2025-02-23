// ./src/components/DashboardSettings.js

import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  FormControlLabel,
  Switch
} from '@mui/material';

function DashboardSettings({ open, onClose, settings, onChange }) {
  const handleToggle = (name) => (event) => {
    const updated = { ...settings, [name]: event.target.checked };
    onChange(updated);
    localStorage.setItem('dashboardSettings', JSON.stringify(updated));
  };

  return (
    <Dialog open={open} onClose={onClose}>
      <DialogTitle>Dashboard Settings</DialogTitle>
      <DialogContent>
        <FormControlLabel
          control={<Switch checked={settings.showFetchedLinks} onChange={handleToggle('showFetchedLinks')} />}
          label="Show Fetched Links"
        />
        <FormControlLabel
          control={<Switch checked={settings.showKnowledgeGraph} onChange={handleToggle('showKnowledgeGraph')} />}
          label="Show Knowledge Graph"
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="primary">Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default DashboardSettings;
