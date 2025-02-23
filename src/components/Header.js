import React from 'react';
import { AppBar, Toolbar, Typography, IconButton, Switch, FormControlLabel, Box } from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

function Header({ toggleTheme, mode, onHelpOpen }) {
  return (
    <AppBar position="static">
      <Toolbar>
        {/* Even Larger Logo Image */}
        <img 
          src="/FreeResearch_logo.png" 
          alt="FreeResearch Logo" 
          style={{ width: 80, height: 80, marginRight: 16 }} 
        />
        <Typography variant="h6" sx={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}>
          FreeResearch
          <Box 
            sx={{ 
              backgroundColor: 'primary.main', 
              color: 'white', 
              borderRadius: '12px', 
              padding: '2px 8px', 
              marginLeft: 2,
              fontSize: '0.75rem',
              fontWeight: 'bold'
            }}
          >
            Beta
          </Box>
        </Typography>
        <FormControlLabel
          control={<Switch checked={mode === 'dark'} onChange={toggleTheme} color="default" />}
          label="Dark Mode"
        />
        <IconButton color="inherit" onClick={onHelpOpen}>
          <HelpOutlineIcon />
        </IconButton>
      </Toolbar>
    </AppBar>
  );
}

export default Header;

