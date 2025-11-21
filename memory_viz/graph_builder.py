"""
Graph builder for memory visualization.
Constructs NetworkX graph from memory data.
"""

import networkx as nx
from networkx.algorithms import community
from typing import List, Dict, Tuple, Optional
from database import Memory
import colorsys


def generate_category_colors(categories: List[str]) -> Dict[str, str]:
    """Generate distinct colors for categories using HSV color space."""
    colors = {}
    n = len(categories)

    for i, category in enumerate(sorted(categories)):
        # Distribute hues evenly around color wheel
        hue = i / n
        # Use high saturation and medium-high value for vibrant colors
        saturation = 0.7
        value = 0.85

        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

        # Convert to hex color
        hex_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        colors[category] = hex_color

    return colors


class MemoryGraph:
    """Build and analyze memory graph."""

    def __init__(self, memories: List[Memory]):
        self.memories = memories
        self.graph = nx.Graph()
        self.category_colors = {}
        self._build_graph()

    def _build_graph(self):
        """Construct NetworkX graph from memories."""
        # Get all unique categories
        categories = list(set(m.category for m in self.memories))
        self.category_colors = generate_category_colors(categories)

        # Add nodes
        for memory in self.memories:
            self.graph.add_node(
                memory.id,
                content=memory.content,
                preview=memory.preview,
                category=memory.category,
                importance=memory.importance,
                access_count=memory.access_count,
                created_at=memory.created_at,
                tags=memory.tags,
                color=self.category_colors.get(memory.category, '#4a5568')
            )

        # Add edges from related_memories
        for memory in self.memories:
            for related_id in memory.related_memories:
                # Only add edge if both nodes exist
                if self.graph.has_node(related_id):
                    self.graph.add_edge(memory.id, related_id)

    def get_node_metrics(self) -> Dict[str, Dict]:
        """Calculate metrics for each node."""
        metrics = {}

        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness = nx.betweenness_centrality(self.graph)

        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]

            metrics[node] = {
                'degree': self.graph.degree(node),
                'degree_centrality': degree_centrality[node],
                'betweenness': betweenness[node],
                'importance': node_data['importance'],
                'access_count': node_data['access_count'],
                'category': node_data['category']
            }

        return metrics

    def get_node_size(self, node_id: str, importance_weight: float = 0.6,
                      connection_weight: float = 0.4) -> float:
        """
        Calculate node size based on importance and connection count.

        Args:
            node_id: Node identifier
            importance_weight: Weight for importance (0-1)
            connection_weight: Weight for connection count (0-1)

        Returns:
            Size value (5-25 range)
        """
        node_data = self.graph.nodes[node_id]
        importance = node_data['importance']
        degree = self.graph.degree(node_id)

        # Normalize degree (assuming max ~50 connections)
        normalized_degree = min(degree / 50.0, 1.0)

        # Weighted combination
        score = (importance * importance_weight) + (normalized_degree * connection_weight)

        # Map to size range 5-25
        min_size = 5
        max_size = 25
        size = min_size + (score * (max_size - min_size))

        return size

    def get_high_access_nodes(self, threshold: int = 20) -> List[str]:
        """Get nodes with access count above threshold."""
        return [
            node for node in self.graph.nodes()
            if self.graph.nodes[node]['access_count'] >= threshold
        ]

    def filter_by_category(self, category: str) -> 'MemoryGraph':
        """Create subgraph filtered by category."""
        filtered_nodes = [
            node for node in self.graph.nodes()
            if self.graph.nodes[node]['category'] == category
        ]

        subgraph = self.graph.subgraph(filtered_nodes).copy()

        # Create new MemoryGraph instance with filtered data
        filtered_memories = [
            m for m in self.memories
            if m.category == category
        ]

        new_graph = MemoryGraph.__new__(MemoryGraph)
        new_graph.memories = filtered_memories
        new_graph.graph = subgraph
        new_graph.category_colors = self.category_colors

        return new_graph

    def filter_by_importance(self, min_importance: float) -> 'MemoryGraph':
        """Create subgraph filtered by minimum importance."""
        filtered_nodes = [
            node for node in self.graph.nodes()
            if self.graph.nodes[node]['importance'] >= min_importance
        ]

        subgraph = self.graph.subgraph(filtered_nodes).copy()

        filtered_memories = [
            m for m in self.memories
            if m.importance >= min_importance
        ]

        new_graph = MemoryGraph.__new__(MemoryGraph)
        new_graph.memories = filtered_memories
        new_graph.graph = subgraph
        new_graph.category_colors = self.category_colors

        return new_graph

    def filter_by_connection_threshold(self, min_connections: int) -> List[tuple]:
        """
        Filter edges by minimum connection threshold.
        Returns list of (source, target) tuples.
        """
        return [
            (u, v) for u, v in self.graph.edges()
            if self.graph.degree(u) >= min_connections
            and self.graph.degree(v) >= min_connections
        ]

    def detect_communities(self) -> Dict[int, List[str]]:
        """
        Detect communities using greedy modularity optimization (Louvain-like).
        Returns dict mapping community_id to list of node IDs.
        """
        communities = community.greedy_modularity_communities(self.graph)

        # Convert to dict format
        community_dict = {}
        for i, comm in enumerate(communities):
            community_dict[i] = list(comm)

        return community_dict

    def get_community_colors(self) -> Dict[int, str]:
        """Generate distinct colors for communities."""
        communities = self.detect_communities()
        n = len(communities)

        colors = {}
        for i in range(n):
            hue = i / n
            saturation = 0.8
            value = 0.9

            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors[i] = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

        return colors

    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        Find shortest path between two memories.
        Returns list of node IDs in path, or None if no path exists.
        """
        try:
            path = nx.shortest_path(self.graph, source, target)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def find_all_paths(self, source: str, target: str, cutoff: int = 5) -> List[List[str]]:
        """
        Find all paths between two memories up to cutoff length.
        Returns list of paths (each path is a list of node IDs).
        """
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=cutoff))
            return paths
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            'density': nx.density(self.graph),
            'num_components': nx.number_connected_components(self.graph),
            'num_categories': len(set(self.graph.nodes[n]['category'] for n in self.graph.nodes()))
        }


if __name__ == "__main__":
    # Test graph builder
    from database import MemoryDatabase

    db = MemoryDatabase()
    db.connect()
    memories = db.load_all_memories()
    db.close()

    print(f"Building graph from {len(memories)} memories...")
    graph = MemoryGraph(memories)

    stats = graph.get_stats()
    print(f"\nGraph Stats:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Avg degree: {stats['avg_degree']:.2f}")
    print(f"  Density: {stats['density']:.4f}")
    print(f"  Components: {stats['num_components']}")
    print(f"  Categories: {stats['num_categories']}")

    print(f"\nCategory colors:")
    for category, color in sorted(graph.category_colors.items())[:10]:
        print(f"  {category}: {color}")

    # Test metrics
    metrics = graph.get_node_metrics()
    top_connected = sorted(metrics.items(), key=lambda x: x[1]['degree'], reverse=True)[:5]
    print(f"\nTop 5 connected nodes:")
    for node_id, m in top_connected:
        node_data = graph.graph.nodes[node_id]
        print(f"  {node_data['category']} (degree: {m['degree']}, importance: {m['importance']:.2f})")
