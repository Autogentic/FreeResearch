// ./src/components/AgentConversationDisplay.js

import React from 'react';
import { Box, Typography, Paper, List, ListItem, ListItemText } from '@mui/material';

function AgentConversationDisplay({ conversation }) {
  return (
    <Box mt={2}>
      <Typography variant="h6">Agent Conversation</Typography>
      <Paper
        variant="outlined"
        style={{
          maxHeight: '300px',
          overflowY: 'auto',
          padding: '1rem',
          backgroundColor: '#fff',
        }}
      >
        {conversation && conversation.length > 0 ? (
          <List>
            {conversation.map((msg, index) => (
              <ListItem key={index}>
                <ListItemText primary={`${msg.sender}: ${msg.content}`} />
              </ListItem>
            ))}
          </List>
        ) : (
          <Typography variant="body2" color="textSecondary">
            No conversation data available.
          </Typography>
        )}
      </Paper>
    </Box>
  );
}

export default AgentConversationDisplay;

