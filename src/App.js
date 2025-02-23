// ./src/App.js

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ResearchForm from './components/ResearchForm';
import LogDisplay from './components/LogDisplay';
import ReportDisplay from './components/ReportDisplay';
import ResearchStatus from './components/ResearchStatus';
import FetchedLinksDisplay from './components/FetchedLinksDisplay';
import KnowledgeGraphDisplay from './components/KnowledgeGraphDisplay';
import AgentConversationDisplay from './components/AgentConversationDisplay';
import IterationLoadingIndicator from './components/IterationLoadingIndicator';
import ResourceMonitorDisplay from './components/ResourceMonitorDisplay';
import DashboardSettings from './components/DashboardSettings';
import DashboardPanel from './components/DashboardPanel';
import HelpModal from './components/HelpModal';
import Header from './components/Header';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Snackbar,
  Alert,
  Button,
  ButtonGroup
} from '@mui/material';

function App({ toggleTheme, mode }) {
  // Primary research state
  const [subject, setSubject] = useState('');
  const [researchStarted, setResearchStarted] = useState(false);
  const [logs, setLogs] = useState([]);
  const [finalReport, setFinalReport] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [fetchError, setFetchError] = useState(null);

  // Additional dashboard data
  const [fetchedLinks, setFetchedLinks] = useState([]);
  const [knowledgeGraph, setKnowledgeGraph] = useState(null);
  const [agentConversation, setAgentConversation] = useState([]);
  const [iteration, setIteration] = useState(0);

  // Dashboard settings (persisted to localStorage)
  // Modified: Removed agent conversation, iteration indicator, and resource monitor toggles,
  // and set them to false by default.
  const defaultSettings = {
    showFetchedLinks: true,
    showKnowledgeGraph: true,
    showAgentConversation: false,
    showIterationIndicator: false,
    showResourceMonitor: false,
  };
  const [dashboardSettings, setDashboardSettings] = useState(() => {
    const saved = localStorage.getItem('dashboardSettings');
    return saved ? JSON.parse(saved) : defaultSettings;
  });
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Pause/Resume state and notifications
  const [isPaused, setIsPaused] = useState(false);
  const [notification, setNotification] = useState('');

  // Help modal state
  const [helpOpen, setHelpOpen] = useState(false);

  // Intervals + request abortion
  const [logIntervalId, setLogIntervalId] = useState(null);
  const [reportIntervalId, setReportIntervalId] = useState(null);
  const [linksIntervalId, setLinksIntervalId] = useState(null);
  const [graphIntervalId, setGraphIntervalId] = useState(null);
  const [conversationIntervalId, setConversationIntervalId] = useState(null);
  const [abortController, setAbortController] = useState(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (logIntervalId) clearInterval(logIntervalId);
      if (reportIntervalId) clearInterval(reportIntervalId);
      if (linksIntervalId) clearInterval(linksIntervalId);
      if (graphIntervalId) clearInterval(graphIntervalId);
      if (conversationIntervalId) clearInterval(conversationIntervalId);
      if (abortController) abortController.abort();
    };
  }, [
    logIntervalId,
    reportIntervalId,
    linksIntervalId,
    graphIntervalId,
    conversationIntervalId,
    abortController
  ]);

  // Persist dashboard settings to localStorage on change
  useEffect(() => {
    localStorage.setItem('dashboardSettings', JSON.stringify(dashboardSettings));
  }, [dashboardSettings]);

  // Start polling intervals
  const startPolling = () => {
    // Logs
    const logId = setInterval(async () => {
      try {
        const res = await axios.get('/api/logs');
        setLogs(res.data);
      } catch (error) {
        console.error('Error fetching logs:', error);
      }
    }, 3000);
    setLogIntervalId(logId);

    // Final report
    const repId = setInterval(async () => {
      try {
        const res = await axios.get('/api/report');
        if (res.data && res.data.report) {
          setFinalReport(res.data.report);
          setIsLoading(false);
          clearInterval(logId);
          clearInterval(repId);
        }
      } catch (error) {
        console.error('Error fetching report:', error);
      }
    }, 5000);
    setReportIntervalId(repId);

    // Fetched links
    const linksId = setInterval(async () => {
      try {
        const res = await axios.get('/api/links');
        setFetchedLinks(res.data);
        setIteration((prev) => prev + 1);
      } catch (error) {
        console.error('Error fetching links:', error);
      }
    }, 5000);
    setLinksIntervalId(linksId);

    // Knowledge graph
    const graphId = setInterval(async () => {
      try {
        const res = await axios.get('/api/knowledge-graph');
        setKnowledgeGraph(res.data);
      } catch (error) {
        console.error('Error fetching knowledge graph:', error);
      }
    }, 6000);
    setGraphIntervalId(graphId);

    // Agent conversation
    const convId = setInterval(async () => {
      try {
        const res = await axios.get('/api/agent-conversation');
        setAgentConversation(res.data);
      } catch (error) {
        console.error('Error fetching agent conversation:', error);
      }
    }, 4000);
    setConversationIntervalId(convId);
  };

  // Start research
  const startResearch = async (subjectInput) => {
    setSubject(subjectInput);
    setResearchStarted(true);
    setIsLoading(true);
    setLogs([]);
    setFinalReport(null);
    setFetchedLinks([]);
    setKnowledgeGraph(null);
    setAgentConversation([]);
    setFetchError(null);
    setIteration(1);
    setIsPaused(false);

    const controller = new AbortController();
    setAbortController(controller);

    try {
      await axios.post('/api/research', { subject: subjectInput }, { signal: controller.signal });
      startPolling();
    } catch (error) {
      if (axios.isCancel(error)) {
        console.log('Research request cancelled:', error.message);
        setFetchError('Research cancelled by user.');
      } else {
        console.error('Error starting research:', error);
        setFetchError('Error starting research. Please try again.');
      }
      setIsLoading(false);
      setResearchStarted(false);
    }
  };

  // Pause research
  const pauseResearch = () => {
    if (logIntervalId) {
      clearInterval(logIntervalId);
      setLogIntervalId(null);
    }
    if (reportIntervalId) {
      clearInterval(reportIntervalId);
      setReportIntervalId(null);
    }
    if (linksIntervalId) {
      clearInterval(linksIntervalId);
      setLinksIntervalId(null);
    }
    if (graphIntervalId) {
      clearInterval(graphIntervalId);
      setGraphIntervalId(null);
    }
    if (conversationIntervalId) {
      clearInterval(conversationIntervalId);
      setConversationIntervalId(null);
    }
    setIsPaused(true);
    setNotification('Research paused.');
  };

  // Resume research
  const resumeResearch = () => {
    startPolling();
    setIsPaused(false);
    setNotification('Research resumed.');
  };

  // Cancel research
  const cancelResearch = () => {
    if (abortController) {
      abortController.abort();
    }
    if (logIntervalId) {
      clearInterval(logIntervalId);
      setLogIntervalId(null);
    }
    if (reportIntervalId) {
      clearInterval(reportIntervalId);
      setReportIntervalId(null);
    }
    if (linksIntervalId) {
      clearInterval(linksIntervalId);
      setLinksIntervalId(null);
    }
    if (graphIntervalId) {
      clearInterval(graphIntervalId);
      setGraphIntervalId(null);
    }
    if (conversationIntervalId) {
      clearInterval(conversationIntervalId);
      setConversationIntervalId(null);
    }
    setIsLoading(false);
    setResearchStarted(false);
    setFetchError('Research cancelled by user.');
  };

  return (
    <>
      <Header toggleTheme={toggleTheme} mode={mode} onHelpOpen={() => setHelpOpen(true)} />
      
      {/* 
        Using a Box with display="flex" for two columns.
        We explicitly set each column to 50% width.
      */}
      <Box sx={{ display: 'flex', marginTop: 2 }}>
        {/* LEFT COLUMN (50%) */}
        <Box sx={{ width: '50%', marginLeft: 2 }}>
          <Paper elevation={3} sx={{ padding: 2 }}>
            {!researchStarted && <ResearchForm onSubmit={startResearch} />}

            {researchStarted && (
              <Box mt={3}>
                <ResearchStatus
                  subject={subject}
                  isLoading={isLoading}
                  logs={logs}
                  finalReport={finalReport}
                />

                <Typography variant="h5" gutterBottom>
                  Research Progress
                </Typography>

                {isLoading && (
                  <Box display="flex" alignItems="center" mb={2}>
                    <CircularProgress size={24} sx={{ marginRight: 1 }} />
                    <Typography>Research is ongoing. Please wait...</Typography>
                  </Box>
                )}

                <LogDisplay logs={logs} />

                {finalReport && <ReportDisplay report={finalReport} />}

                <Box mt={2}>
                  <ButtonGroup variant="outlined">
                    <Button color="secondary" onClick={cancelResearch}>
                      Cancel Research
                    </Button>

                    {!isPaused ? (
                      <Button color="secondary" onClick={pauseResearch}>
                        Pause Research
                      </Button>
                    ) : (
                      <Button color="secondary" onClick={resumeResearch}>
                        Resume Research
                      </Button>
                    )}
                  </ButtonGroup>

                  <Button
                    variant="outlined"
                    sx={{ ml: 2 }}
                    onClick={() => setSettingsOpen(true)}
                  >
                    Dashboard Settings
                  </Button>
                </Box>

                {/* Additional dashboard panels */}
                {dashboardSettings.showResourceMonitor && (
                  <DashboardPanel title="Resource Monitor">
                    <ResourceMonitorDisplay />
                  </DashboardPanel>
                )}

                {dashboardSettings.showFetchedLinks && (
                  <DashboardPanel title="Fetched Websites">
                    <FetchedLinksDisplay links={fetchedLinks} />
                  </DashboardPanel>
                )}

                {dashboardSettings.showIterationIndicator && (
                  <DashboardPanel title="Iteration Progress">
                    <IterationLoadingIndicator iteration={iteration} />
                  </DashboardPanel>
                )}

                {dashboardSettings.showAgentConversation && (
                  <DashboardPanel title="Agent Conversation">
                    <AgentConversationDisplay conversation={agentConversation} />
                  </DashboardPanel>
                )}

                <DashboardSettings
                  open={settingsOpen}
                  onClose={() => setSettingsOpen(false)}
                  settings={dashboardSettings}
                  onChange={setDashboardSettings}
                />
              </Box>
            )}
          </Paper>
        </Box>

        {/* RIGHT COLUMN (50%) */}
        <Box sx={{ width: '50%', marginLeft: 2, marginRight: 2 }}>
          {researchStarted && dashboardSettings.showKnowledgeGraph && (
            <DashboardPanel title="Knowledge Graph">
              <KnowledgeGraphDisplay />
            </DashboardPanel>
          )}
        </Box>
      </Box>

      <HelpModal open={helpOpen} onClose={() => setHelpOpen(false)} />

      <Snackbar
        open={Boolean(fetchError)}
        autoHideDuration={6000}
        onClose={() => setFetchError(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setFetchError(null)}
          severity="error"
          sx={{ width: '100%' }}
        >
          {fetchError}
        </Alert>
      </Snackbar>

      <Snackbar
        open={Boolean(notification)}
        autoHideDuration={3000}
        onClose={() => setNotification('')}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setNotification('')}
          severity="info"
          sx={{ width: '100%' }}
        >
          {notification}
        </Alert>
      </Snackbar>
    </>
  );
}

export default App;
