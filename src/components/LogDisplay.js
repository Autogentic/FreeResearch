// ./src/components/LogDisplay.js

import React, { useState } from 'react';
import { Box, Typography, Paper, Pagination, TextField, Button } from '@mui/material';
import { useTheme } from '@mui/material/styles';

function LogDisplay({ logs }) {
  const theme = useTheme();
  const logsPerPage = 10;
  const [page, setPage] = useState(1);
  const [filterText, setFilterText] = useState('');

  // Filter logs by sender or content
  const filteredLogs = logs
    ? logs.filter((log) => {
        const text = filterText.toLowerCase();
        return (
          log.sender.toLowerCase().includes(text) ||
          log.content.toLowerCase().includes(text)
        );
      })
    : [];

  const pageCount = Math.ceil(filteredLogs.length / logsPerPage);
  const paginatedLogs = filteredLogs.slice((page - 1) * logsPerPage, page * logsPerPage);

  const handlePageChange = (event, value) => {
    setPage(value);
  };

  const handleFilterChange = (event) => {
    setFilterText(event.target.value);
    setPage(1); // Reset to page 1 whenever filter changes
  };

  // Export logs as CSV
  const exportLogs = () => {
    let csvContent = 'data:text/csv;charset=utf-8,Timestamp,Sender,Content\n';
    filteredLogs.forEach((log) => {
      const timestamp = `"${log.timestamp}"`;
      const sender = `"${log.sender}"`;
      // Escape any double-quotes in content
      const content = `"${log.content.replace(/"/g, '""')}"`;
      csvContent += `${timestamp},${sender},${content}\n`;
    });
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'research_logs.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <Box mt={2} mb={2}>
      <Box display="flex" justifyContent="space-between" alignItems="center">
        <Typography variant="h6">
          Research Logs {logs && logs.length ? `(${logs.length})` : ''}
        </Typography>
        <Button variant="outlined" size="small" onClick={exportLogs}>
          Export Logs
        </Button>
      </Box>

      <TextField
        label="Filter logs"
        variant="outlined"
        size="small"
        value={filterText}
        onChange={handleFilterChange}
        fullWidth
        sx={{ my: 1 }}
      />

      <Paper
        variant="outlined"
        sx={{
          maxHeight: 300,
          overflowY: 'auto',
          p: 1,
          // Use theme-based background color
          backgroundColor: 'background.paper',
          // KEY ADDITIONS to prevent stretching:
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {paginatedLogs && paginatedLogs.length > 0 ? (
          paginatedLogs.map((log, index) => (
            <Box key={index} mb={1}>
              <Typography variant="caption" color="textSecondary">
                {log.timestamp} - {log.sender}
              </Typography>
              <Typography
                variant="body2"
                // Alternatively, you can set wrapping here:
                // sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
              >
                {log.content}
              </Typography>
            </Box>
          ))
        ) : (
          <Typography variant="body2" color="textSecondary">
            No logs available.
          </Typography>
        )}
      </Paper>

      {pageCount > 1 && (
        <Box display="flex" justifyContent="center" mt={1}>
          <Pagination
            count={pageCount}
            page={page}
            onChange={handlePageChange}
            color="primary"
          />
        </Box>
      )}
    </Box>
  );
}

export default LogDisplay;

