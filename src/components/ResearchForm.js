// ./src/components/ResearchForm.js

import React, { useState } from 'react';
import { TextField, Button, Box, Typography } from '@mui/material';

function ResearchForm({ onSubmit }) {
  const [subject, setSubject] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (subject.trim() !== '') {
      onSubmit(subject.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <Box display="flex" flexDirection="column" gap="1rem">
        <Typography variant="h6">Enter Research Subject:</Typography>
        <TextField
          label="Research Subject"
          variant="outlined"
          value={subject}
          onChange={(e) => setSubject(e.target.value)}
          multiline
          rows={4}  // Adjust the number of rows as needed
          required
        />
        <Button type="submit" variant="contained" color="primary">
          Start Research
        </Button>
      </Box>
    </form>
  );
}

export default ResearchForm;

