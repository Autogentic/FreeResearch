// ./src/index.js

import React from 'react';
import ReactDOM from 'react-dom/client';
import ThemeProviderWrapper from './ThemeProviderWrapper';
import './App.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ThemeProviderWrapper />
  </React.StrictMode>
);

