#!/usr/bin/env python3
"""
Batch Graph Builder for BuildAutomata Memory System

Retroactively builds graph connections by finding similar memories for ALL memories
in the database and updating their related_memories field.

Uses same semantic search logic as store_memory auto-linking, but operates on
existing memories without creating new versions.

Usage:
    python batch_build_graph.py --limit 6 --threshold 0.0 --dry-run
    python batch_build_graph.py --limit 6  # Actually update database
"""

import sqlite3
import json
import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
import hashlib

# Add parent directory to path to import MemoryStore
sys.path.insert(0, str(Path(__file__).parent))

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("WARNING: sentence-transformers not available, using fallback embeddings")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("WARNING: qdrant-client not available, using SQLite-only search")


def get_db_path():
    """Get database path from environment or use default"""
    username = os.getenv("BA_USERNAME", "buildautomata_ai_v012")
    agent_name = os.getenv("BA_AGENT_NAME", "claude_assistant")
    repo_dir = Path(__file__).parent / "memory_repos" / f"{username}_{agent_name}"
    return repo_dir / "memoryv012.db"


class BatchGraphBuilder:
    """Builds graph connections for existing memories using semantic search"""

    def __init__(self, db_path: str, similarity_limit: int = 6):
        self.db_path = db_path
        self.similarity_limit = similarity_limit
        self.encoder = None
        self.qdrant_client = None
        self.collection_name = None

        # Initialize encoder
        if EMBEDDINGS_AVAILABLE:
            print("Initializing sentence transformer (all-mpnet-base-v2)...")
            self.encoder = SentenceTransformer("all-mpnet-base-v2", device="cpu")
            print("Encoder loaded successfully")
        else:
            print("Using fallback hash-based embeddings")

        # Initialize Qdrant if available
        if QDRANT_AVAILABLE:
            try:
                qdrant_host = os.getenv("QDRANT_HOST", "localhost")
                qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
                username = os.getenv("BA_USERNAME", "buildautomata_ai_v012")
                agent_name = os.getenv("BA_AGENT_NAME", "claude_assistant")
                self.collection_name = f"{username}_{agent_name}_memories"

                self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
                print(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}, collection: {self.collection_name}")
            except Exception as e:
                print(f"Qdrant connection failed: {e}, falling back to SQLite-only")
                self.qdrant_client = None

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.encoder:
            return self.encoder.encode(text).tolist()
        else:
            # Fallback: hash-based embedding (768 dimensions like all-mpnet-base-v2)
            embedding = []
            hash_input = text.encode()
            while len(embedding) < 768:
                hash_obj = hashlib.sha256(hash_input)
                hash_bytes = hash_obj.digest()
                embedding.extend([float(b) / 255.0 for b in hash_bytes])
                hash_input = hash_bytes
            return embedding[:768]

    def search_similar(self, memory_id: str, content: str) -> List[str]:
        """Find similar memories using semantic search, excluding self"""
        similar_ids = []

        # Generate embedding for query
        embedding = self.generate_embedding(content)

        # Try Qdrant first if available
        if self.qdrant_client and self.collection_name:
            try:
                results = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=embedding,
                    limit=self.similarity_limit + 1  # +1 to account for self-match
                )

                for hit in results.points:
                    hit_id = hit.id
                    if hit_id != memory_id:  # Exclude self
                        similar_ids.append(hit_id)
                        if len(similar_ids) >= self.similarity_limit:
                            break

                return similar_ids
            except Exception as e:
                print(f"Qdrant search failed for {memory_id}: {e}, falling back to SQLite")

        # Fallback to SQLite FTS search (less accurate but works without Qdrant)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            # Extract key terms from content for FTS search
            words = content.split()[:20]  # Use first 20 words as search query
            query = " OR ".join(words)

            cursor = conn.execute("""
                SELECT m.id
                FROM memories_fts fts
                JOIN memories m ON fts.id = m.id
                WHERE fts.content MATCH ?
                AND m.id != ?
                ORDER BY rank
                LIMIT ?
            """, (query, memory_id, self.similarity_limit))

            for row in cursor:
                similar_ids.append(row['id'])

        except Exception as e:
            print(f"SQLite FTS search failed for {memory_id}: {e}")

        finally:
            conn.close()

        return similar_ids

    def build_graph(self, dry_run: bool = False):
        """Build graph connections for all memories"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            # Get all memories
            cursor = conn.execute("SELECT id, content, related_memories FROM memories ORDER BY created_at")
            memories = cursor.fetchall()

            total = len(memories)
            print(f"\nProcessing {total} memories...")
            print(f"Similarity limit: {self.similarity_limit} connections per memory")
            print(f"Dry run: {dry_run}\n")

            updated = 0
            skipped = 0

            for idx, row in enumerate(memories, 1):
                memory_id = row['id']
                content = row['content']
                existing_related = json.loads(row['related_memories']) if row['related_memories'] else []

                # Find similar memories
                similar_ids = self.search_similar(memory_id, content)

                if similar_ids:
                    # Check if update needed
                    if set(similar_ids) != set(existing_related):
                        if not dry_run:
                            # Update database
                            conn.execute("""
                                UPDATE memories
                                SET related_memories = ?
                                WHERE id = ?
                            """, (json.dumps(similar_ids), memory_id))

                            # Also update relationships table
                            # First delete existing relationships
                            conn.execute("DELETE FROM relationships WHERE source_id = ?", (memory_id,))

                            # Insert new relationships
                            from datetime import datetime
                            for related_id in similar_ids:
                                conn.execute("""
                                    INSERT OR IGNORE INTO relationships
                                    (source_id, target_id, strength, created_at)
                                    VALUES (?, ?, ?, ?)
                                """, (memory_id, related_id, 1.0, datetime.now()))

                            conn.commit()

                        updated += 1
                        action = "[DRY-RUN] Would update" if dry_run else "Updated"
                        print(f"[{idx}/{total}] {action} {memory_id[:8]}... "
                              f"({len(existing_related)} -> {len(similar_ids)} connections)")
                    else:
                        skipped += 1
                        if idx % 100 == 0:  # Progress indicator
                            print(f"[{idx}/{total}] Processed... ({updated} updated, {skipped} unchanged)")
                else:
                    skipped += 1
                    if idx % 100 == 0:
                        print(f"[{idx}/{total}] Processed... ({updated} updated, {skipped} unchanged)")

            print(f"\n{'=' * 60}")
            print(f"Complete! Updated: {updated}, Skipped: {skipped}, Total: {total}")
            print(f"{'=' * 60}")

            # Show statistics
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_memories,
                    COUNT(CASE WHEN related_memories IS NOT NULL THEN 1 END) as memories_with_links,
                    SUM(json_array_length(related_memories)) as total_connections
                FROM memories
            """)
            stats = cursor.fetchone()

            if stats:
                print(f"\nGraph Statistics:")
                print(f"  Total memories: {stats['total_memories']}")
                print(f"  Memories with links: {stats['memories_with_links']}")
                print(f"  Total connections: {stats['total_connections'] or 0}")
                if stats['total_memories'] > 0:
                    avg_connections = (stats['total_connections'] or 0) / stats['total_memories']
                    print(f"  Average connections per memory: {avg_connections:.2f}")

        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Batch build graph connections for memories")
    parser.add_argument("--limit", type=int, default=6,
                       help="Number of similar memories to link per memory (default: 6)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be updated without making changes")
    parser.add_argument("--db", type=str, default=None,
                       help="Database path (default: auto-detect from env)")

    args = parser.parse_args()

    db_path = args.db if args.db else get_db_path()

    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        print("Set BA_USERNAME and BA_AGENT_NAME environment variables or use --db flag")
        return 1

    print(f"Database: {db_path}")

    builder = BatchGraphBuilder(db_path, similarity_limit=args.limit)
    builder.build_graph(dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
