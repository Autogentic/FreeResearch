import React, { useEffect, useState, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import {
  Box,
  Typography,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button
} from '@mui/material';

/**
 * Lays out the graph so that:
 *   1. Query nodes (IDs containing "_query_") appear in a row at the top.
 *   2. For each query node, we run a BFS queue with items: { nodeId, depth, parentId }.
 *   3. GREEN (non-Jina) nodes:
 *      - Horizontal: x = queryNode.fx + (depth - 1) * greenOffsetX
 *      - Vertical:   y = queryNode.fy + baseYSpacing + ((depth - 1) + rowIndex) * rowSpacing
 *   4. YELLOW (Jina) nodes:
 *      - If depth=1 => 100 px right of the query’s x
 *      - If parent is Jina => same x as parent (so references stack vertically)
 *      - Otherwise => same BFS-based offset as green plus 100
 *      - Vertical: y = [ same as green formula ] + rowSpacing/2
 *        => This places Jina nodes “in between” the green rows
 *   5. A global visited set ensures each node is only placed once if multiple queries link to it.
 *   6. BFS depth increments by 1 for grandchildren, so they appear further down the diagram.
 */
function layoutGraph(graph) {
  // Spacing constants
  const greenOffsetX = 50;         // horizontal shift per BFS depth for green nodes
  const queryToJinaOffset = 100;   // offset for a Jina node at depth=1
  const baseYSpacing = 80;         // vertical offset from the query node to the first row
  const rowSpacing = 60;           // vertical spacing between siblings

  const adjacency = {};
  const nodesById = {};

  // Build node lookup
  graph.nodes.forEach((n) => {
    nodesById[n.id] = n;
  });

  // Build adjacency list
  graph.links.forEach((l) => {
    const src = typeof l.source === 'object' ? l.source.id : l.source;
    const tgt = typeof l.target === 'object' ? l.target.id : l.target;
    if (!adjacency[src]) adjacency[src] = [];
    adjacency[src].push(tgt);
  });

  // Identify query nodes
  const queryNodes = graph.nodes.filter((n) => n.id.includes('_query_'));

  // Place query nodes horizontally at the top
  queryNodes.forEach((queryNode, qIndex) => {
    queryNode.fx = 100 + qIndex * 300; // horizontal offset for queries
    queryNode.fy = 50;                // vertical position
  });

  // Global visited set so a node is only placed once
  const globalVisited = new Set();

  // Helper to detect if node is a Jina (yellow) node
  const isJinaNode = (nodeId) => nodeId.includes('_jina_');

  // BFS from each query node
  queryNodes.forEach((queryNode) => {
    const queue = [];
    // Start with the query's children at depth=1
    const children = adjacency[queryNode.id] || [];
    children.forEach((childId) => {
      if (!childId.includes('_query_') && !globalVisited.has(childId)) {
        queue.push({ nodeId: childId, depth: 1, parentId: queryNode.id });
      }
    });

    // For BFS, track row indices separately for green vs. yellow at each depth
    const layerRowIndexGreen = {};
    const layerRowIndexJina = {};

    while (queue.length > 0) {
      const { nodeId, depth, parentId } = queue.shift();
      if (globalVisited.has(nodeId)) continue;

      const node = nodesById[nodeId];
      if (!node) continue;

      // Mark visited
      globalVisited.add(nodeId);

      // Identify if node is Jina
      const nodeIsJina = isJinaNode(nodeId);
      // Identify if parent is Jina
      const parentIsJina = isJinaNode(parentId);
      const parentNode = nodesById[parentId];

      // Ensure row counters exist for this depth
      if (nodeIsJina && !layerRowIndexJina[depth]) {
        layerRowIndexJina[depth] = 0;
      }
      if (!nodeIsJina && !layerRowIndexGreen[depth]) {
        layerRowIndexGreen[depth] = 0;
      }

      // Determine row index
      let rowIndex;
      if (nodeIsJina) {
        rowIndex = layerRowIndexJina[depth];
        layerRowIndexJina[depth]++;
      } else {
        rowIndex = layerRowIndexGreen[depth];
        layerRowIndexGreen[depth]++;
      }

      // Horizontal placement
      if (!nodeIsJina) {
        // GREEN node
        // BFS depth => shift from query's x
        node.fx = queryNode.fx + (depth - 1) * greenOffsetX;
      } else {
        // YELLOW (Jina) node
        if (depth === 1) {
          // Direct child of query => place it 100px right of query
          node.fx = queryNode.fx + queryToJinaOffset;
        } else if (parentIsJina && parentNode) {
          // Child of Jina => same x as parent to stack references vertically
          node.fx = parentNode.fx;
        } else {
          // BFS depth>1, parent is green => offset from the green lane by +100
          node.fx = queryNode.fx + (depth - 1) * greenOffsetX + queryToJinaOffset;
        }
      }

      // Vertical placement
      // - We incorporate BFS depth and rowIndex
      // - For Jina, we add rowSpacing/2 so it sits between green rows
      const depthRowBase = (depth - 1) + rowIndex; 
      node.fy = queryNode.fy
        + baseYSpacing
        + depthRowBase * rowSpacing
        + (nodeIsJina ? rowSpacing / 2 : 0);

      // Enqueue grandchildren at depth+1
      const grandChildren = adjacency[nodeId] || [];
      grandChildren.forEach((gcId) => {
        if (!gcId.includes('_query_') && !globalVisited.has(gcId)) {
          queue.push({ nodeId: gcId, depth: depth + 1, parentId: nodeId });
        }
      });
    }
  });
}

/**
 * Node color:
 *   - Jina => Yellow (#f1c40f)
 *   - Query => Blue (#1f77b4)
 *   - Everything else => Green (#2ca02c)
 */
function getNodeColor(node) {
  if (node.id && node.id.includes('_jina_')) return '#f1c40f';  // Yellow for Jina
  if (node.id && node.id.includes('_query_')) return '#1f77b4'; // Blue for queries
  return '#2ca02c'; // Green for everything else
}

/**
 * GraphCanvas: Renders a ForceGraph2D in a fixed-size container.
 * Supports clicking a node to display more info in a modal.
 */
function GraphCanvas({ graphData, title }) {
  const containerRef = useRef(null);
  const [width, setWidth] = useState(400);
  const [selectedNode, setSelectedNode] = useState(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const handleResize = () => {
      const rect = containerRef.current.getBoundingClientRect();
      setWidth(rect.width);
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  const handleNodeClick = (node) => {
    setSelectedNode(node);
  };

  const handleCloseModal = () => {
    setSelectedNode(null);
  };

  return (
    <Box
      ref={containerRef}
      sx={{
        width: '100%',
        maxHeight: 600,
        overflow: 'auto',
        position: 'relative',
        mt: 1,
      }}
    >
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>

      <ForceGraph2D
        graphData={graphData}
        width={width}
        height={600}
        backgroundColor="#f0f0f0"
        dagMode={null}
        cooldownTicks={0}
        onNodeClick={handleNodeClick}
        nodeCanvasObjectMode={() => 'replace'}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const fontSize = 10 / globalScale;
          const titleText = node.title || node.id || '';
          const snippetText = node.snippet
            ? ` - ${node.snippet.substring(0, 30)}${node.snippet.length > 30 ? '...' : ''}`
            : '';
          const label = titleText + snippetText;
          ctx.font = `${fontSize}px Sans-Serif`;

          const textWidth = ctx.measureText(label).width;
          const radius = 5;

          // Draw node circle
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
          ctx.fillStyle = getNodeColor(node);
          ctx.fill();

          // Label background
          ctx.fillStyle = 'rgba(255,255,255,0.8)';
          ctx.fillRect(
            node.x + radius + 3,
            node.y - fontSize,
            textWidth + 6,
            fontSize + 6
          );

          // Label text
          ctx.fillStyle = 'black';
          ctx.fillText(label, node.x + radius + 6, node.y + 2);
        }}
        linkDirectionalArrowLength={3}
        linkDirectionalArrowRelPos={1}
      />

      {/* Modal to display additional node details */}
      {selectedNode && (
        <Dialog open={Boolean(selectedNode)} onClose={handleCloseModal} maxWidth="sm" fullWidth>
          <DialogTitle>{selectedNode.title || selectedNode.id}</DialogTitle>
          <DialogContent dividers>
            {selectedNode.snippet && (
              <DialogContentText>
                <strong>Snippet:</strong> {selectedNode.snippet}
              </DialogContentText>
            )}
            {selectedNode.summary && (
              <DialogContentText>
                <strong>Summary:</strong> {selectedNode.summary}
              </DialogContentText>
            )}
            {selectedNode.entities && (
              <DialogContentText>
                <strong>Entities:</strong> {JSON.stringify(selectedNode.entities)}
              </DialogContentText>
            )}
            {selectedNode.timestamp && (
              <DialogContentText>
                <strong>Timestamp:</strong> {selectedNode.timestamp}
              </DialogContentText>
            )}
            {selectedNode.content && (
              <DialogContentText>
                <strong>Content:</strong> {selectedNode.content}
              </DialogContentText>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseModal} color="primary">Close</Button>
          </DialogActions>
        </Dialog>
      )}
    </Box>
  );
}

/**
 * KnowledgeGraphDisplay: fetches the data from the given endpoint,
 * applies layoutGraph to both the session and persistent sub-graphs,
 * and renders them with GraphCanvas. Polls every 5s for updated data.
 */
export default function KnowledgeGraphDisplay({ endpointUrl = '/api/knowledge-graph' }) {
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchGraph = async () => {
      try {
        const response = await fetch(endpointUrl);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        const mergedData = {
          session: {
            nodes: data.session?.nodes || [],
            links: data.session?.links || []
          },
          persistent: {
            nodes: data.persistent?.nodes || [],
            links: data.persistent?.links || []
          }
        };

        // Layout each sub-graph
        layoutGraph(mergedData.session);
        layoutGraph(mergedData.persistent);

        setGraphData(mergedData);
      } catch (error) {
        console.error('Error fetching knowledge graph data:', error);
        setGraphData({
          session: { nodes: [], links: [] },
          persistent: { nodes: [], links: [] }
        });
      } finally {
        setLoading(false);
      }
    };

    fetchGraph();
    const intervalId = setInterval(fetchGraph, 5000);
    return () => clearInterval(intervalId);
  }, [endpointUrl]);

  if (loading) {
    return (
      <Box mt={2} display="flex" justifyContent="center">
        <CircularProgress />
      </Box>
    );
  }

  const { session, persistent } = graphData;

  return (
    <Box mt={0}>
      <Typography variant="h5" gutterBottom>
        Knowledge Graph Overview
      </Typography>

      {session.nodes.length > 0 ? (
        <>
          <GraphCanvas graphData={session} title="Session Knowledge Graph" />
          <Box sx={{ mt: 2 }} />
        </>
      ) : (
        <Typography variant="body2" color="textSecondary">
          No session graph data available.
        </Typography>
      )}

      {persistent.nodes.length > 0 ? (
        <GraphCanvas graphData={persistent} title="Persistent Knowledge Graph" />
      ) : (
        <Typography variant="body2" color="textSecondary">
          No persistent graph data available.
        </Typography>
      )}
    </Box>
  );
}

