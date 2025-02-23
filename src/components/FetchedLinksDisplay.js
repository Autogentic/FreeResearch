// ./src/components/FetchedLinksDisplay.js

import React from 'react';
import { Box, Typography, List, ListItem, ListItemText, Link } from '@mui/material';

function FetchedLinksDisplay({ links }) {
  return (
    <Box mt={2}>
      <Typography variant="h6">Fetched Websites</Typography>
      {links && links.length > 0 ? (
        <List>
          {links.map((link, index) => (
            <ListItem key={index}>
              <ListItemText
                primary={
                  <Link href={link.url} target="_blank" rel="noopener">
                    {link.title}
                  </Link>
                }
              />
            </ListItem>
          ))}
        </List>
      ) : (
        <Typography variant="body2" color="textSecondary">
          No websites fetched.
        </Typography>
      )}
    </Box>
  );
}

export default FetchedLinksDisplay;

