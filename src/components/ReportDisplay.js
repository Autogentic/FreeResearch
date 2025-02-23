import React, { useState } from 'react';
import { Box, Typography, Paper, Button, Snackbar, Alert } from '@mui/material';
import { useTheme } from '@mui/material/styles';

function ReportDisplay({ report }) {
  const [downloadSuccess, setDownloadSuccess] = useState(false);
  const theme = useTheme();

  const downloadReport = () => {
    const element = document.createElement('a');
    const file = new Blob([report], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = 'final_report.txt';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);

    setDownloadSuccess(true);
    setTimeout(() => setDownloadSuccess(false), 3000);
  };

  return (
    <Box mt={3}>
      <Typography variant="h5" gutterBottom>
        Final Research Report
      </Typography>
      <Paper
        variant="outlined"
        sx={{
          maxHeight: '500px',
          overflowY: 'auto',
          p: 2,
          bgcolor: theme.palette.background.paper,
          mb: 2,
        }}
      >
        <Typography 
          variant="body1" 
          sx={{ whiteSpace: 'pre-wrap', color: theme.palette.text.primary }}
        >
          {report}
        </Typography>
      </Paper>
      <Button variant="contained" color="primary" onClick={downloadReport}>
        Download Report
      </Button>
      <Snackbar
        open={downloadSuccess}
        autoHideDuration={3000}
        onClose={() => setDownloadSuccess(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setDownloadSuccess(false)}
          severity="success"
          sx={{ width: '100%' }}
        >
          Download initiated!
        </Alert>
      </Snackbar>
    </Box>
  );
}

export default ReportDisplay;

