"""
Database loader for memory visualization.
Reads memories from memoryv012.db SQLite database.
"""

import sqlite3
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Memory:
    """Memory data structure."""
    id: str
    content: str
    category: str
    importance: float
    tags: List[str]
    created_at: str
    access_count: int
    related_memories: List[str]

    @property
    def preview(self) -> str:
        """Get short preview of content (first 150 chars)."""
        return self.content[:150].replace('\n', ' ') + ('...' if len(self.content) > 150 else '')


class MemoryDatabase:
    """Load and query memories from SQLite database."""

    def __init__(self, db_path: str = None):
        # Default to the standard memory repo location
        if db_path is None:
            import os
            if os.path.exists("../memory_repos/buildautomata_ai_v012_claude_assistant/memoryv012.db"):
                db_path = "../memory_repos/buildautomata_ai_v012_claude_assistant/memoryv012.db"
        self.db_path = db_path
        self.connection = None

    def connect(self):
        """Open database connection."""
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()

    def load_all_memories(self) -> List[Memory]:
        """Load all memories from database."""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT id, content, category, importance, tags,
                   created_at, access_count, related_memories
            FROM memories
        """)

        memories = []
        for row in cursor.fetchall():
            # Parse JSON fields
            tags = json.loads(row['tags']) if row['tags'] else []
            related = json.loads(row['related_memories']) if row['related_memories'] else []

            memory = Memory(
                id=row['id'],
                content=row['content'],
                category=row['category'],
                importance=float(row['importance']),
                tags=tags,
                created_at=row['created_at'],
                access_count=int(row['access_count']) if row['access_count'] else 0,
                related_memories=related
            )
            memories.append(memory)

        return memories

    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get specific memory by ID."""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT id, content, category, importance, tags,
                   created_at, access_count, related_memories
            FROM memories
            WHERE id = ?
        """, (memory_id,))

        row = cursor.fetchone()
        if not row:
            return None

        tags = json.loads(row['tags']) if row['tags'] else []
        related = json.loads(row['related_memories']) if row['related_memories'] else []

        return Memory(
            id=row['id'],
            content=row['content'],
            category=row['category'],
            importance=float(row['importance']),
            tags=tags,
            created_at=row['created_at'],
            access_count=int(row['access_count']) if row['access_count'] else 0,
            related_memories=related
        )

    def get_categories(self) -> List[str]:
        """Get all unique categories."""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT DISTINCT category
            FROM memories
            ORDER BY category
        """)

        return [row['category'] for row in cursor.fetchall()]

    def get_stats(self) -> Dict:
        """Get database statistics."""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()

        # Total memories
        cursor.execute("SELECT COUNT(*) as total FROM memories")
        total = cursor.fetchone()['total']

        # Total connections
        cursor.execute("""
            SELECT SUM(json_array_length(related_memories)) as total_connections
            FROM memories
            WHERE related_memories IS NOT NULL
        """)
        total_connections = cursor.fetchone()['total_connections'] or 0

        # Avg importance
        cursor.execute("SELECT AVG(importance) as avg_importance FROM memories")
        avg_importance = cursor.fetchone()['avg_importance'] or 0

        # Categories
        categories = self.get_categories()

        return {
            'total_memories': total,
            'total_connections': total_connections,
            'avg_connections': total_connections / total if total > 0 else 0,
            'avg_importance': avg_importance,
            'num_categories': len(categories),
            'categories': categories
        }


if __name__ == "__main__":
    # Test the database loader
    db = MemoryDatabase()
    db.connect()

    stats = db.get_stats()
    print(f"Database Stats:")
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Total connections: {stats['total_connections']}")
    print(f"  Avg connections: {stats['avg_connections']:.2f}")
    print(f"  Categories: {stats['num_categories']}")

    memories = db.load_all_memories()
    print(f"\nLoaded {len(memories)} memories")

    if memories:
        m = memories[0]
        print(f"\nExample memory:")
        print(f"  ID: {m.id}")
        print(f"  Category: {m.category}")
        print(f"  Importance: {m.importance}")
        print(f"  Connections: {len(m.related_memories)}")
        print(f"  Preview: {m.preview}")

    db.close()
