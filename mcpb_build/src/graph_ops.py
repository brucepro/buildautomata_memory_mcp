"""
Graph operations for BuildAutomata Memory System
Copyright 2025 Jurden Bruce
"""

import json
import logging
from typing import List, Dict, Any, Optional, Set
from collections import Counter

logger = logging.getLogger("buildautomata-memory.graph")

try:
    from .models import Memory
except ImportError:
    from models import Memory


class GraphOperations:
    """Handles memory graph operations"""

    def __init__(self, db_conn, db_lock, get_memory_by_id_func):
        self.db_conn = db_conn
        self.db_lock = db_lock
        self.get_memory_by_id = get_memory_by_id_func

    def enrich_related_memories(self, related_ids: List[str]) -> List[Dict[str, Any]]:
        """Enrich related memory IDs with previews"""
        if not related_ids or not self.db_conn:
            return []

        enriched = []
        try:
            with self.db_lock:
                for mem_id in related_ids[:10]:
                    cursor = self.db_conn.execute(
                        "SELECT id, content, category, importance, tags, created_at FROM memories WHERE id = ?",
                        (mem_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        content = row[1]
                        preview = content[:150] + "..." if len(content) > 150 else content
                        tags = json.loads(row[4]) if row[4] else []

                        enriched.append({
                            "id": row[0],
                            "content_preview": preview,
                            "category": row[2],
                            "importance": row[3],
                            "tags": tags,
                            "created_at": row[5]
                        })
        except Exception as e:
            logger.error(f"Failed to enrich related_memories: {e}")

        return enriched

    async def traverse_graph(
        self,
        start_memory_id: str,
        depth: int = 2,
        max_nodes: int = 50,
        min_importance: float = 0.0,
        category_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Traverse memory graph N hops from starting node"""
        if depth < 1 or depth > 5:
            return {"error": "depth must be between 1 and 5"}

        visited = set()
        nodes = {}
        edges = []

        async def traverse_level(memory_ids: List[str], current_depth: int):
            if current_depth > depth or len(nodes) >= max_nodes:
                return

            next_level = []
            for mem_id in memory_ids:
                if mem_id in visited or len(nodes) >= max_nodes:
                    continue

                visited.add(mem_id)
                mem_result = await self.get_memory_by_id(mem_id)
                if "error" in mem_result:
                    continue

                if mem_result["importance"] < min_importance:
                    continue
                if category_filter and mem_result["category"] != category_filter:
                    continue

                nodes[mem_id] = {
                    "id": mem_id,
                    "content_preview": mem_result["content"][:150] + "..." if len(mem_result["content"]) > 150 else mem_result["content"],
                    "category": mem_result["category"],
                    "importance": mem_result["importance"],
                    "tags": mem_result["tags"],
                    "created_at": mem_result["created_at"],
                    "depth": current_depth
                }

                if mem_result.get("related_memories"):
                    for related_id in mem_result["related_memories"]:
                        if isinstance(related_id, dict):
                            related_id = related_id.get("id")
                        if related_id:
                            edges.append({
                                "source": mem_id,
                                "target": related_id,
                                "type": "related"
                            })
                            next_level.append(related_id)

            if next_level and current_depth < depth:
                await traverse_level(next_level, current_depth + 1)

        await traverse_level([start_memory_id], 1)

        return {
            "start_node": start_memory_id,
            "depth": depth,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": list(nodes.values()),
            "edges": edges,
            "truncated": len(nodes) >= max_nodes
        }

    async def find_clusters(
        self,
        min_cluster_size: int = 3,
        min_importance: float = 0.5,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Identify densely connected regions in memory graph"""
        if not self.db_conn:
            return {"error": "Database not available"}

        try:
            with self.db_lock:
                cursor = self.db_conn.execute(
                    """
                    SELECT id, content, category, importance, tags, related_memories, created_at
                    FROM memories
                    WHERE importance >= ?
                    ORDER BY importance DESC
                    LIMIT 200
                    """,
                    (min_importance,)
                )
                rows = cursor.fetchall()

            # Build adjacency graph
            graph = {}
            memory_data = {}

            for row in rows:
                mem_id = row["id"]
                related = json.loads(row["related_memories"]) if row["related_memories"] else []

                related_ids = []
                for r in related:
                    if isinstance(r, dict):
                        related_ids.append(r.get("id"))
                    else:
                        related_ids.append(r)

                graph[mem_id] = set(r for r in related_ids if r)
                memory_data[mem_id] = {
                    "id": mem_id,
                    "content_preview": row["content"][:100] + "..." if len(row["content"]) > 100 else row["content"],
                    "category": row["category"],
                    "importance": row["importance"],
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "created_at": row["created_at"]
                }

            # Find connected components (DFS)
            visited = set()
            clusters = []

            def dfs(node, cluster):
                if node in visited or node not in graph:
                    return
                visited.add(node)
                cluster.add(node)
                for neighbor in graph.get(node, set()):
                    if neighbor not in visited:
                        dfs(neighbor, cluster)

            for node in graph:
                if node not in visited:
                    cluster = set()
                    dfs(node, cluster)
                    if len(cluster) >= min_cluster_size:
                        clusters.append(cluster)

            # Sort and enrich clusters
            cluster_results = []
            for cluster in sorted(clusters, key=len, reverse=True)[:limit]:
                cluster_mems = [memory_data[mem_id] for mem_id in cluster if mem_id in memory_data]
                avg_importance = sum(m["importance"] for m in cluster_mems) / len(cluster_mems)

                all_tags = []
                categories = {}
                for m in cluster_mems:
                    all_tags.extend(m["tags"])
                    cat = m["category"]
                    categories[cat] = categories.get(cat, 0) + 1

                common_tags = [tag for tag, count in Counter(all_tags).most_common(5)]
                dominant_category = max(categories.items(), key=lambda x: x[1])[0] if categories else None

                cluster_results.append({
                    "size": len(cluster),
                    "avg_importance": round(avg_importance, 3),
                    "dominant_category": dominant_category,
                    "common_tags": common_tags,
                    "memory_ids": list(cluster),
                    "sample_memories": cluster_mems[:5]
                })

            return {
                "total_memories_analyzed": len(rows),
                "clusters_found": len(cluster_results),
                "clusters": cluster_results
            }

        except Exception as e:
            logger.error(f"Cluster detection failed: {e}")
            return {"error": str(e)}

    async def get_graph_statistics(
        self,
        category: Optional[str] = None,
        min_importance: float = 0.0
    ) -> Dict[str, Any]:
        """Get graph connectivity statistics"""
        if not self.db_conn:
            return {"error": "Database not available"}

        try:
            query = "SELECT id, category, importance, related_memories FROM memories WHERE importance >= ?"
            params = [min_importance]

            if category:
                query += " AND category = ?"
                params.append(category)

            with self.db_lock:
                cursor = self.db_conn.execute(query, params)
                rows = cursor.fetchall()

            connection_counts = {}
            total_connections = 0
            isolated_nodes = []

            for row in rows:
                mem_id = row["id"]
                related = json.loads(row["related_memories"]) if row["related_memories"] else []

                related_ids = []
                for r in related:
                    if isinstance(r, dict):
                        related_ids.append(r.get("id"))
                    else:
                        related_ids.append(r)

                connection_count = len([r for r in related_ids if r])
                connection_counts[mem_id] = connection_count
                total_connections += connection_count

                if connection_count == 0:
                    isolated_nodes.append({
                        "id": mem_id,
                        "category": row["category"],
                        "importance": row["importance"]
                    })

            # Find hubs
            hubs = sorted(
                [(mem_id, count) for mem_id, count in connection_counts.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]

            hub_details = []
            for mem_id, count in hubs:
                mem_result = await self.get_memory_by_id(mem_id)
                if "error" not in mem_result:
                    hub_details.append({
                        "id": mem_id,
                        "connections": count,
                        "category": mem_result["category"],
                        "importance": mem_result["importance"],
                        "content_preview": mem_result["content"][:100] + "..." if len(mem_result["content"]) > 100 else mem_result["content"]
                    })

            avg_connections = total_connections / len(rows) if rows else 0

            return {
                "total_memories": len(rows),
                "total_connections": total_connections,
                "avg_connections_per_memory": round(avg_connections, 2),
                "isolated_nodes_count": len(isolated_nodes),
                "isolated_nodes": isolated_nodes[:20],
                "most_connected_hubs": hub_details,
                "connectivity_distribution": {
                    "no_connections": len([c for c in connection_counts.values() if c == 0]),
                    "1-3_connections": len([c for c in connection_counts.values() if 1 <= c <= 3]),
                    "4-10_connections": len([c for c in connection_counts.values() if 4 <= c <= 10]),
                    "10+_connections": len([c for c in connection_counts.values() if c > 10])
                }
            }

        except Exception as e:
            logger.error(f"Graph statistics failed: {e}")
            return {"error": str(e)}
