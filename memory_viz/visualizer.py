"""
3D graph visualizer using Plotly.
Generates interactive 3D force-directed graph visualization.
"""

import plotly.graph_objects as go
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from graph_builder import MemoryGraph


def compute_3d_layout(graph: nx.Graph, iterations: int = 50, seed: int = 42) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute 3D force-directed layout using NetworkX spring layout.

    Args:
        graph: NetworkX graph
        iterations: Number of iterations for layout algorithm
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node IDs to (x, y, z) coordinates
    """
    # Use spring_layout with 3D
    pos = nx.spring_layout(
        graph,
        dim=3,
        iterations=iterations,
        seed=seed,
        k=2/np.sqrt(graph.number_of_nodes())  # Optimal distance between nodes
    )

    return pos


def compute_temporal_layout(graph: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute 3D layout arranged by creation date (temporal).
    X-axis represents time (oldest to newest).
    Y and Z use spring layout for clustering.

    Returns:
        Dictionary mapping node IDs to (x, y, z) coordinates
    """
    # Parse creation dates
    nodes_with_dates = []
    for node in graph.nodes():
        created_at = graph.nodes[node].get('created_at', '2025-01-01')
        try:
            date = datetime.fromisoformat(created_at[:10])
            nodes_with_dates.append((node, date))
        except:
            # Fallback to epoch start
            nodes_with_dates.append((node, datetime(2025, 1, 1)))

    # Sort by date
    nodes_with_dates.sort(key=lambda x: x[1])

    # Find date range
    min_date = nodes_with_dates[0][1]
    max_date = nodes_with_dates[-1][1]
    date_range = (max_date - min_date).days or 1

    # Compute 2D spring layout for Y and Z
    pos_2d = nx.spring_layout(graph, dim=2, iterations=30, seed=42)

    # Combine temporal X with spring Y, Z
    pos = {}
    for i, (node, date) in enumerate(nodes_with_dates):
        # X is temporal (normalized to -1 to 1)
        days_from_start = (date - min_date).days
        x = -1 + 2 * (days_from_start / date_range)

        # Y and Z from spring layout
        y, z = pos_2d[node]

        pos[node] = (x, y, z)

    return pos


def compute_cluster_layout(graph: nx.Graph, communities: Dict[int, List[str]]) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute 3D layout with clusters arranged in space.
    Each community gets a region, nodes within cluster use spring layout.

    Args:
        graph: NetworkX graph
        communities: Dict mapping community_id to list of node IDs

    Returns:
        Dictionary mapping node IDs to (x, y, z) coordinates
    """
    pos = {}
    n_communities = len(communities)

    # Arrange community centers in a circle
    for comm_id, nodes in communities.items():
        # Community center position (arranged in circle)
        angle = 2 * np.pi * comm_id / n_communities
        radius = 3.0
        center_x = radius * np.cos(angle)
        center_y = radius * np.sin(angle)
        center_z = 0

        # Create subgraph for this community
        subgraph = graph.subgraph(nodes)

        # Spring layout within cluster (2D)
        if len(nodes) > 1:
            sub_pos = nx.spring_layout(subgraph, dim=2, iterations=20, scale=1.0, seed=42)
        else:
            sub_pos = {nodes[0]: (0, 0)}

        # Offset by community center
        for node in nodes:
            if node in sub_pos:
                local_x, local_y = sub_pos[node]
                pos[node] = (
                    center_x + local_x,
                    center_y + local_y,
                    center_z
                )
            else:
                pos[node] = (center_x, center_y, center_z)

    return pos


class MemoryVisualizer:
    """Generate 3D Plotly visualization of memory graph."""

    def __init__(self, memory_graph: MemoryGraph):
        self.memory_graph = memory_graph
        self.layout = None

    def create_visualization(
        self,
        connection_threshold: int = 0,
        show_high_access_glow: bool = True,
        high_access_threshold: int = 20,
        importance_weight: float = 0.6,
        connection_weight: float = 0.4,
        layout_mode: str = 'force',
        color_mode: str = 'category',
        highlight_path: Optional[List[str]] = None,
        show_particles: bool = False
    ) -> go.Figure:
        """
        Create interactive 3D visualization.

        Args:
            connection_threshold: Minimum connections for edge to show
            show_high_access_glow: Highlight high-access nodes
            high_access_threshold: Access count threshold for glow
            importance_weight: Weight for importance in node sizing
            connection_weight: Weight for connections in node sizing
            layout_mode: 'force', 'temporal', or 'cluster'
            color_mode: 'category' or 'community'
            highlight_path: List of node IDs to highlight as path
            show_particles: Show animated particles on edges

        Returns:
            Plotly Figure object
        """
        # Compute layout based on mode
        if layout_mode == 'temporal':
            self.layout = compute_temporal_layout(self.memory_graph.graph)
        elif layout_mode == 'cluster':
            communities = self.memory_graph.detect_communities()
            self.layout = compute_cluster_layout(self.memory_graph.graph, communities)
        else:
            self.layout = compute_3d_layout(self.memory_graph.graph)

        # Create figure
        fig = go.Figure()

        # Add edges
        edge_trace = self._create_edge_trace(connection_threshold, highlight_path)
        if edge_trace:
            if isinstance(edge_trace, list):
                for trace in edge_trace:
                    fig.add_trace(trace)
            else:
                fig.add_trace(edge_trace)

        # Add particle effects if enabled
        if show_particles:
            particle_traces = self._create_particle_traces(connection_threshold)
            for trace in particle_traces:
                fig.add_trace(trace)

        # Add nodes
        node_trace = self._create_node_trace(
            importance_weight,
            connection_weight,
            show_high_access_glow,
            high_access_threshold,
            color_mode,
            highlight_path
        )
        fig.add_trace(node_trace)

        # Update layout with dark theme
        fig.update_layout(
            title="Memory Graph Visualization",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            paper_bgcolor='#0a0e27',
            plot_bgcolor='#0a0e27',
            font=dict(color='#e5e7eb', family='Inter, sans-serif'),
            scene=dict(
                xaxis=dict(
                    showbackground=False,
                    showgrid=False,
                    showticklabels=False,
                    title='',
                    zeroline=False
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=False,
                    showticklabels=False,
                    title='',
                    zeroline=False
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=False,
                    showticklabels=False,
                    title='',
                    zeroline=False
                ),
                bgcolor='#0a0e27',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=800
        )

        return fig

    def _create_edge_trace(self, connection_threshold: int = 0, highlight_path: Optional[List[str]] = None):
        """Create edge trace for links between nodes, with optional path highlighting."""
        # Filter edges by connection threshold
        if connection_threshold > 0:
            edges = self.memory_graph.filter_by_connection_threshold(connection_threshold)
        else:
            edges = list(self.memory_graph.graph.edges())

        if not edges:
            return None

        # Separate path edges from regular edges
        path_edges = set()
        if highlight_path and len(highlight_path) > 1:
            for i in range(len(highlight_path) - 1):
                path_edges.add((highlight_path[i], highlight_path[i+1]))
                path_edges.add((highlight_path[i+1], highlight_path[i]))  # Undirected

        # Regular edges
        regular_edge_x = []
        regular_edge_y = []
        regular_edge_z = []

        # Path edges
        path_edge_x = []
        path_edge_y = []
        path_edge_z = []

        for edge in edges:
            x0, y0, z0 = self.layout[edge[0]]
            x1, y1, z1 = self.layout[edge[1]]

            if edge in path_edges:
                path_edge_x.extend([x0, x1, None])
                path_edge_y.extend([y0, y1, None])
                path_edge_z.extend([z0, z1, None])
            else:
                regular_edge_x.extend([x0, x1, None])
                regular_edge_y.extend([y0, y1, None])
                regular_edge_z.extend([z0, z1, None])

        traces = []

        # Add regular edges
        if regular_edge_x:
            regular_trace = go.Scatter3d(
                x=regular_edge_x,
                y=regular_edge_y,
                z=regular_edge_z,
                mode='lines',
                line=dict(
                    color='#2d3748',
                    width=1
                ),
                hoverinfo='none',
                showlegend=False
            )
            traces.append(regular_trace)

        # Add path edges (highlighted)
        if path_edge_x:
            path_trace = go.Scatter3d(
                x=path_edge_x,
                y=path_edge_y,
                z=path_edge_z,
                mode='lines',
                line=dict(
                    color='#fbbf24',  # Gold for path
                    width=4
                ),
                hoverinfo='none',
                showlegend=False
            )
            traces.append(path_trace)

        return traces if len(traces) > 1 else (traces[0] if traces else None)

    def _create_node_trace(
        self,
        importance_weight: float,
        connection_weight: float,
        show_high_access_glow: bool,
        high_access_threshold: int,
        color_mode: str = 'category',
        highlight_path: Optional[List[str]] = None
    ) -> go.Scatter3d:
        """Create node trace with colors, sizes, and hover info."""
        node_x = []
        node_y = []
        node_z = []
        node_colors = []
        node_sizes = []
        node_text = []
        node_customdata = []

        # Get high-access nodes for glow effect
        high_access_nodes = set(self.memory_graph.get_high_access_nodes(high_access_threshold))

        # Get community assignments if using community coloring
        node_to_community = {}
        community_colors = {}
        if color_mode == 'community':
            communities = self.memory_graph.detect_communities()
            community_colors = self.memory_graph.get_community_colors()
            for comm_id, nodes in communities.items():
                for node in nodes:
                    node_to_community[node] = comm_id

        # Path nodes for highlighting
        path_nodes = set(highlight_path) if highlight_path else set()

        for node in self.memory_graph.graph.nodes():
            pos = self.layout[node]
            node_x.append(pos[0])
            node_y.append(pos[1])
            node_z.append(pos[2])

            # Get node data
            node_data = self.memory_graph.graph.nodes[node]

            # Color based on mode
            if color_mode == 'community' and node in node_to_community:
                color = community_colors[node_to_community[node]]
            else:
                color = node_data['color']

            # Highlight path nodes
            if node in path_nodes:
                color = '#fbbf24'  # Gold for path nodes

            # Add glow for high-access nodes
            elif show_high_access_glow and node in high_access_nodes:
                # Brighten color for glow effect
                color = self._brighten_color(color, factor=1.3)

            node_colors.append(color)

            # Size
            size = self.memory_graph.get_node_size(
                node,
                importance_weight,
                connection_weight
            )
            node_sizes.append(size)

            # Hover text
            connections = self.memory_graph.graph.degree(node)
            hover_text = (
                f"<b>{node_data['category']}</b><br>"
                f"Importance: {node_data['importance']:.2f}<br>"
                f"Connections: {connections}<br>"
                f"Access count: {node_data['access_count']}<br>"
                f"Created: {node_data['created_at'][:10]}<br>"
                f"<br>{node_data['preview']}"
            )
            node_text.append(hover_text)

            # Custom data for click events (store node ID)
            node_customdata.append(node)

        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(
                    color='#1a1f3a',
                    width=0.5
                ),
                opacity=0.9
            ),
            text=node_text,
            customdata=node_customdata,
            hoverinfo='text',
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        )

        return node_trace

    def _brighten_color(self, hex_color: str, factor: float = 1.3) -> str:
        """Brighten a hex color by a factor."""
        # Remove '#' if present
        hex_color = hex_color.lstrip('#')

        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        # Brighten
        r = min(int(r * factor), 255)
        g = min(int(g * factor), 255)
        b = min(int(b * factor), 255)

        # Convert back to hex
        return f'#{r:02x}{g:02x}{b:02x}'

    def _create_particle_traces(self, connection_threshold: int = 0, num_particles: int = 50) -> List[go.Scatter3d]:
        """Create animated particle traces along edges."""
        # Filter edges
        if connection_threshold > 0:
            edges = self.memory_graph.filter_by_connection_threshold(connection_threshold)
        else:
            edges = list(self.memory_graph.graph.edges())

        if not edges or len(edges) == 0:
            return []

        # Sample random edges for particles
        sampled_edges = np.random.choice(len(edges), min(num_particles, len(edges)), replace=False)

        particle_traces = []

        for idx in sampled_edges:
            edge = edges[idx]
            x0, y0, z0 = self.layout[edge[0]]
            x1, y1, z1 = self.layout[edge[1]]

            # Create particle at random position along edge
            t = np.random.random()
            px = x0 + t * (x1 - x0)
            py = y0 + t * (y1 - y0)
            pz = z0 + t * (z1 - z0)

            particle = go.Scatter3d(
                x=[px],
                y=[py],
                z=[pz],
                mode='markers',
                marker=dict(
                    size=3,
                    color='#00d9ff',
                    opacity=0.8
                ),
                hoverinfo='none',
                showlegend=False
            )
            particle_traces.append(particle)

        return particle_traces

    def export_png(self, filename: str = "memory_graph.png", width: int = 1920, height: int = 1080):
        """
        Export visualization as PNG using kaleido.

        Args:
            filename: Output filename
            width: Image width in pixels
            height: Image height in pixels
        """
        try:
            fig = self.create_visualization()
            fig.write_image(filename, width=width, height=height)
            print(f"Exported to {filename}")
        except Exception as e:
            print(f"Error exporting PNG: {e}")
            print("Make sure kaleido is installed: pip install kaleido")


if __name__ == "__main__":
    # Test visualizer
    from database import MemoryDatabase
    from graph_builder import MemoryGraph

    print("Loading database...")
    db = MemoryDatabase("memoryv012.db")  # Use local path when running from parent dir
    db.connect()
    memories = db.load_all_memories()
    db.close()

    print(f"Building graph from {len(memories)} memories...")
    graph = MemoryGraph(memories)

    print("Creating visualization...")
    viz = MemoryVisualizer(graph)
    fig = viz.create_visualization()

    print("Saving to HTML...")
    fig.write_html("memory_graph_test.html")
    print("Done! Open memory_graph_test.html in browser.")
