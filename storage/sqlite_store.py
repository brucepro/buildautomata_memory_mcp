"""
SQLite persistence store for BuildAutomata Memory System
Copyright 2025 Jurden Bruce
"""

import sqlite3
import json
import logging
import threading
import traceback
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("buildautomata-memory.sqlite")

try:
    from ..models import Memory, Intention, MemoryRelationship
except ImportError:
    from models import Memory, Intention, MemoryRelationship


class SQLiteStore:
    """Handles all SQLite database operations"""

    def __init__(self, db_path: Path, db_conn, db_lock, error_log: List[Dict[str, Any]]):
        self.db_path = db_path
        self.conn = db_conn  # Use external connection
        self._lock = db_lock  # Use external lock
        self.error_log = error_log

    def _log_error(self, operation: str, error: Exception):
        """Log detailed error information"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "error_type": type(error).__name__,
            "error_msg": str(error),
            "traceback": traceback.format_exc(),
        }
        self.error_log.append(error_entry)
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]

    def initialize(self):
        """Initialize SQLite schema and migrations (uses external connection)"""
        if not self.conn:
            logger.error("Cannot initialize - no database connection provided")
            return

        try:
            with self._lock:
                self.conn.executescript("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        category TEXT NOT NULL,
                        importance REAL NOT NULL,
                        tags TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        last_accessed TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        decay_rate REAL DEFAULT 0.95,
                        version_count INTEGER DEFAULT 1,
                        content_hash TEXT NOT NULL,
                        memory_type TEXT DEFAULT 'episodic',
                        session_id TEXT,
                        task_context TEXT,
                        provenance TEXT,
                        relationships TEXT,
                        valid_from TIMESTAMP,
                        valid_until TIMESTAMP,
                        related_memories TEXT
                    );

                    CREATE TABLE IF NOT EXISTS memory_versions (
                        version_id TEXT PRIMARY KEY,
                        memory_id TEXT NOT NULL,
                        version_number INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        category TEXT NOT NULL,
                        importance REAL NOT NULL,
                        tags TEXT NOT NULL,
                        metadata TEXT,
                        change_type TEXT NOT NULL,
                        change_description TEXT,
                        created_at TIMESTAMP NOT NULL,
                        content_hash TEXT NOT NULL,
                        prev_version_id TEXT,
                        FOREIGN KEY (memory_id) REFERENCES memories(id),
                        UNIQUE(memory_id, version_number)
                    );

                    CREATE TABLE IF NOT EXISTS relationships (
                        source_id TEXT NOT NULL,
                        target_id TEXT NOT NULL,
                        relationship_type TEXT DEFAULT 'references',
                        strength REAL DEFAULT 1.0,
                        created_at TIMESTAMP NOT NULL,
                        metadata TEXT,
                        PRIMARY KEY (source_id, target_id, relationship_type)
                    );

                    CREATE TABLE IF NOT EXISTS intentions (
                        id TEXT PRIMARY KEY,
                        description TEXT NOT NULL,
                        priority REAL NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        deadline TIMESTAMP,
                        preconditions TEXT,
                        actions TEXT,
                        related_memories TEXT,
                        metadata TEXT,
                        last_checked TIMESTAMP,
                        check_count INTEGER DEFAULT 0
                    );

                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        id UNINDEXED,
                        content,
                        tags
                    );

                    CREATE INDEX IF NOT EXISTS idx_category ON memories(category);
                    CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC);
                    CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_updated ON memories(updated_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash);
                    CREATE INDEX IF NOT EXISTS idx_category_importance
                        ON memories(category, importance DESC, updated_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_search_filters
                        ON memories(category, importance, created_at, updated_at);
                    CREATE INDEX IF NOT EXISTS idx_version_memory ON memory_versions(memory_id, version_number DESC);
                    CREATE INDEX IF NOT EXISTS idx_version_timestamp ON memory_versions(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_version_hash ON memory_versions(content_hash);
                    CREATE INDEX IF NOT EXISTS idx_source ON relationships(source_id);
                    CREATE INDEX IF NOT EXISTS idx_target ON relationships(target_id);
                    CREATE INDEX IF NOT EXISTS idx_intention_status ON intentions(status, priority DESC);
                    CREATE INDEX IF NOT EXISTS idx_intention_deadline ON intentions(deadline);
                    CREATE INDEX IF NOT EXISTS idx_intention_priority ON intentions(priority DESC);
                    CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type);
                    CREATE INDEX IF NOT EXISTS idx_session_id ON memories(session_id);
                    CREATE INDEX IF NOT EXISTS idx_type_session ON memories(memory_type, session_id, created_at DESC);

                    CREATE TABLE IF NOT EXISTS command_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP NOT NULL,
                        tool_name TEXT NOT NULL,
                        args TEXT,
                        result_summary TEXT,
                        memory_id TEXT,
                        success INTEGER DEFAULT 1
                    );

                    CREATE INDEX IF NOT EXISTS idx_command_timestamp ON command_history(timestamp DESC);
                    CREATE INDEX IF NOT EXISTS idx_command_tool ON command_history(tool_name);
                """)
                self.conn.commit()

                # Run migrations
                cursor = self.conn.cursor()
                try:
                    cursor.execute("PRAGMA table_info(memories)")
                    columns = {row[1] for row in cursor.fetchall()}

                    migrations = [
                        ("memory_type", "ALTER TABLE memories ADD COLUMN memory_type TEXT DEFAULT 'episodic'"),
                        ("session_id", "ALTER TABLE memories ADD COLUMN session_id TEXT"),
                        ("task_context", "ALTER TABLE memories ADD COLUMN task_context TEXT"),
                        ("provenance", "ALTER TABLE memories ADD COLUMN provenance TEXT"),
                        ("relationships", "ALTER TABLE memories ADD COLUMN relationships TEXT"),
                        ("valid_from", "ALTER TABLE memories ADD COLUMN valid_from TIMESTAMP"),
                        ("valid_until", "ALTER TABLE memories ADD COLUMN valid_until TIMESTAMP"),
                        ("related_memories", "ALTER TABLE memories ADD COLUMN related_memories TEXT"),
                    ]

                    for col_name, sql in migrations:
                        if col_name not in columns:
                            logger.info(f"Migrating: adding {col_name} column")
                            cursor.execute(sql)

                    self.conn.commit()
                    logger.info("Migration completed successfully")

                except Exception as migration_error:
                    logger.warning(f"Migration error (non-fatal): {migration_error}")
                    self.conn.rollback()

            logger.info("SQLite initialized successfully")
        except Exception as e:
            logger.error(f"SQLite initialization failed: {e}")
            self._log_error("sqlite_init", e)
            self.conn = None

    def create_version(self, memory: Memory, change_type: str, change_description: str, prev_version_id: Optional[str] = None):
        """Create version snapshot"""
        if not self.conn:
            return None

        try:
            with self._lock:
                self.conn.execute("BEGIN IMMEDIATE")
                try:
                    cursor = self.conn.execute(
                        "SELECT COALESCE(MAX(version_number), 0) + 1 FROM memory_versions WHERE memory_id = ?",
                        (memory.id,)
                    )
                    version_number = cursor.fetchone()[0]

                    version_id = str(uuid.uuid4())
                    self.conn.execute("""
                        INSERT INTO memory_versions
                        (version_id, memory_id, version_number, content, category, importance,
                         tags, metadata, change_type, change_description, created_at, content_hash, prev_version_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        version_id, memory.id, version_number, memory.content, memory.category,
                        memory.importance, json.dumps(memory.tags), json.dumps(memory.metadata),
                        change_type, change_description, datetime.now(), memory.content_hash(), prev_version_id
                    ))
                    self.conn.commit()
                    return version_id
                except Exception as e:
                    self.conn.rollback()
                    raise
        except Exception as e:
            logger.error(f"Version creation failed: {e}")
            self._log_error("create_version", e)
            return None

    def store_memory(self, memory: Memory, is_update: bool = False, skip_version: bool = False) -> bool:
        """Store memory with versioning"""
        if not self.conn:
            return False

        try:
            with self._lock:
                version_id = None
                if not skip_version:
                    prev_version_id = None
                    if is_update:
                        cursor = self.conn.execute(
                            "SELECT version_id FROM memory_versions WHERE memory_id = ? ORDER BY version_number DESC LIMIT 1",
                            (memory.id,)
                        )
                        row = cursor.fetchone()
                        if row:
                            prev_version_id = row[0]

                    change_type = "update" if is_update else "create"
                    change_description = f"Memory {change_type}d"
                    version_id = self.create_version(memory, change_type, change_description, prev_version_id)

                    if not version_id and not skip_version:
                        return False

                self.conn.execute("""
                    INSERT OR REPLACE INTO memories
                    (id, content, category, importance, tags, metadata,
                     created_at, updated_at, last_accessed, access_count, decay_rate,
                     version_count, content_hash, memory_type, session_id, task_context,
                     provenance, relationships, valid_from, valid_until, related_memories)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            COALESCE((SELECT version_count FROM memories WHERE id = ?), 0) + ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id, memory.content, memory.category, memory.importance,
                    json.dumps(memory.tags), json.dumps(memory.metadata),
                    memory.created_at, memory.updated_at, memory.last_accessed, memory.access_count,
                    memory.decay_rate, memory.id, 0 if skip_version else 1, memory.content_hash(),
                    memory.memory_type, memory.session_id, memory.task_context,
                    json.dumps(memory.provenance) if memory.provenance else None,
                    json.dumps([r.to_dict() for r in memory.relationships]) if memory.relationships else None,
                    memory.valid_from, memory.valid_until,
                    json.dumps(memory.related_memories) if memory.related_memories else None
                ))

                # Update FTS
                self.conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory.id,))
                self.conn.execute("INSERT INTO memories_fts(id, content, tags) VALUES (?, ?, ?)",
                                (memory.id, memory.content, " ".join(memory.tags)))

                # Handle relationships
                if memory.related_memories:
                    for related_id in memory.related_memories:
                        self.conn.execute("""
                            INSERT OR IGNORE INTO relationships
                            (source_id, target_id, strength, created_at)
                            VALUES (?, ?, ?, ?)
                        """, (memory.id, related_id, 1.0, datetime.now()))

                self.conn.commit()
                return True
        except Exception as e:
            logger.error(f"SQLite store failed for {memory.id}: {e}")
            self._log_error("sqlite_store", e)
            try:
                self.conn.rollback()
            except:
                pass
            return False

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memory by ID"""
        with self._lock:
            cursor = self.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_memory(row)
        return None

    def search_fts(self, query: str, limit: int, category: Optional[str], min_importance: float,
                   created_after: Optional[str], created_before: Optional[str],
                   updated_after: Optional[str], updated_before: Optional[str],
                   memory_type: Optional[str] = None, session_id: Optional[str] = None) -> List[Memory]:
        """Full-text search"""
        if not self.conn:
            return []

        try:
            with self._lock:
                sanitized_query = self._sanitize_fts_query(query)
                conditions = []
                params = [sanitized_query]

                if category:
                    conditions.append("m.category = ?")
                    params.append(category)
                if min_importance > 0:
                    conditions.append("m.importance >= ?")
                    params.append(min_importance)
                if memory_type:
                    conditions.append("m.memory_type = ?")
                    params.append(memory_type)
                if session_id:
                    conditions.append("m.session_id = ?")
                    params.append(session_id)
                if created_after:
                    conditions.append("m.created_at >= ?")
                    params.append(created_after)
                if created_before:
                    conditions.append("m.created_at <= ?")
                    params.append(created_before)
                if updated_after:
                    conditions.append("m.updated_at >= ?")
                    params.append(updated_after)
                if updated_before:
                    conditions.append("m.updated_at <= ?")
                    params.append(updated_before)

                fts_sql = (
                    "SELECT m.* FROM memories m JOIN memories_fts ON m.id = memories_fts.id WHERE memories_fts MATCH ?"
                    + (" AND " + " AND ".join(conditions) if conditions else "")
                    + " ORDER BY m.importance DESC LIMIT ?"
                )
                params.append(limit)

                return [self._row_to_memory(row) for row in self.conn.execute(fts_sql, params)]
        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            self._log_error("fts_search", e)
            return []

    def sanitize_fts_query(self, query: str) -> str:
        """Sanitize FTS5 query (public interface)"""
        return self._sanitize_fts_query(query)

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize FTS5 query - split words for OR search to match semantic search behavior"""
        if not query or not query.strip():
            return '""'

        # Split into words and quote each separately for OR search
        # This allows "play autonomous exploration" to match memories containing any of those words
        words = query.strip().split()
        escaped_words = [f'"{word.replace(chr(34), chr(34)+chr(34))}"' for word in words]
        return ' OR '.join(escaped_words)

    def _row_to_memory(self, row) -> Memory:
        """Convert SQLite row to Memory"""
        provenance = None
        if "provenance" in row.keys() and row["provenance"]:
            try:
                provenance = json.loads(row["provenance"])
            except:
                pass

        relationships = []
        if "relationships" in row.keys() and row["relationships"]:
            try:
                relationships = [MemoryRelationship(**r) for r in json.loads(row["relationships"])]
            except:
                pass

        related_memories = []
        if "related_memories" in row.keys() and row["related_memories"]:
            try:
                related_memories = json.loads(row["related_memories"])
            except:
                pass

        return Memory(
            id=row["id"],
            content=row["content"],
            category=row["category"],
            importance=row["importance"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=row["access_count"],
            last_accessed=row["last_accessed"],
            valid_from=row["valid_from"] if "valid_from" in row.keys() else None,
            valid_until=row["valid_until"] if "valid_until" in row.keys() else None,
            decay_rate=row["decay_rate"],
            version_count=row["version_count"] if "version_count" in row.keys() else 1,
            memory_type=row["memory_type"] if "memory_type" in row.keys() else "episodic",
            session_id=row["session_id"] if "session_id" in row.keys() else None,
            task_context=row["task_context"] if "task_context" in row.keys() else None,
            provenance=provenance,
            relationships=relationships,
            related_memories=related_memories,
        )

    def update_access(self, memory_id: str):
        """Update access stats with Saint Bernard pattern"""
        if not self.conn:
            return

        try:
            with self._lock:
                cursor = self.conn.execute("""
                    SELECT importance, access_count, last_accessed, created_at, decay_rate
                    FROM memories WHERE id = ?
                """, (memory_id,))

                row = cursor.fetchone()
                if not row:
                    return

                current_importance, current_access_count, last_accessed, created_at, decay_rate = row
                reference_date = last_accessed if last_accessed else created_at

                if isinstance(reference_date, str):
                    reference_date = datetime.fromisoformat(reference_date)

                if reference_date:
                    days = (datetime.now() - reference_date).days
                    decayed_importance = current_importance * (decay_rate ** days)
                else:
                    decayed_importance = current_importance

                new_access_count = current_access_count + 1
                new_importance = decayed_importance + 0.03 if new_access_count > 5 else decayed_importance
                new_importance = max(0.1, min(1.0, new_importance))

                self.conn.execute("""
                    UPDATE memories
                    SET importance = ?, access_count = ?, last_accessed = ?
                    WHERE id = ?
                """, (new_importance, new_access_count, datetime.now(), memory_id))
                self.conn.commit()
        except Exception as e:
            logger.error(f"Access update failed for {memory_id}: {e}")
            self._log_error("update_access", e)

    def get_version_count(self, memory_id: str) -> int:
        """Get version count"""
        if not self.conn:
            return 1
        try:
            with self._lock:
                cursor = self.conn.execute(
                    "SELECT COUNT(*) FROM memory_versions WHERE memory_id = ?", (memory_id,)
                )
                return cursor.fetchone()[0] or 1
        except:
            return 1

    def get_access_stats(self, memory_id: str) -> Tuple[int, Optional[str]]:
        """Get access count and last_accessed"""
        if not self.conn:
            return (0, None)
        try:
            with self._lock:
                cursor = self.conn.execute(
                    "SELECT access_count, last_accessed FROM memories WHERE id = ?", (memory_id,)
                )
                row = cursor.fetchone()
                return (row[0], row[1]) if row else (0, None)
        except:
            return (0, None)

    def get_related_memories(self, memory_id: str) -> List[str]:
        """Get related memory IDs"""
        if not self.conn:
            return []
        try:
            with self._lock:
                cursor = self.conn.execute(
                    "SELECT related_memories FROM memories WHERE id = ?", (memory_id,)
                )
                row = cursor.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
                return []
        except:
            return []

    def get_version_history_summary(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get version history summary"""
        if not self.conn:
            return None

        try:
            with self._lock:
                cursor = self.conn.execute("""
                    SELECT version_number, change_type, created_at, content, category, importance, tags
                    FROM memory_versions
                    WHERE memory_id = ?
                    ORDER BY version_number ASC
                """, (memory_id,))

                versions = cursor.fetchall()
                if not versions or len(versions) <= 1:
                    return None

                summary = {
                    "total_versions": len(versions),
                    "created": versions[0]["created_at"],
                    "last_updated": versions[-1]["created_at"],
                    "update_count": len(versions) - 1,
                    "evolution": []
                }

                for v in versions:
                    summary["evolution"].append({
                        "version": v["version_number"],
                        "timestamp": v["created_at"],
                        "content": v["content"],
                        "category": v["category"],
                        "importance": v["importance"],
                        "tags": json.loads(v["tags"]) if v["tags"] else []
                    })

                return summary
        except Exception as e:
            logger.error(f"Failed to get version summary for {memory_id}: {e}")
            return None

    def log_command(self, tool_name: str, args: Dict[str, Any], result_summary: str = None,
                    memory_id: str = None, success: bool = True):
        """Log a command to the history table"""
        if not self.conn:
            return

        try:
            with self._lock:
                self.conn.execute("""
                    INSERT INTO command_history (timestamp, tool_name, args, result_summary, memory_id, success)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    tool_name,
                    json.dumps(args) if args else None,
                    result_summary,
                    memory_id,
                    1 if success else 0
                ))
                self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to log command: {e}")
            self._log_error("log_command", e)

    def get_command_history(self, limit: int = 20, tool_name: str = None,
                            start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """Get command history, optionally filtered by tool name and date range"""
        if not self.conn:
            return []

        try:
            with self._lock:
                # Build query with optional filters
                conditions = []
                params = []

                if tool_name:
                    conditions.append("tool_name = ?")
                    params.append(tool_name)
                if start_date:
                    conditions.append("timestamp >= ?")
                    params.append(start_date)
                if end_date:
                    conditions.append("timestamp <= ?")
                    params.append(end_date)

                where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
                params.append(limit)

                cursor = self.conn.execute(f"""
                    SELECT id, timestamp, tool_name, args, result_summary, memory_id, success
                    FROM command_history
                    {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, params)

                rows = cursor.fetchall()
                return [{
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "tool_name": row["tool_name"],
                    "args": json.loads(row["args"]) if row["args"] else None,
                    "result_summary": row["result_summary"],
                    "memory_id": row["memory_id"],
                    "success": bool(row["success"])
                } for row in rows]
        except Exception as e:
            logger.error(f"Failed to get command history: {e}")
            self._log_error("get_command_history", e)
            return []

    def get_last_activity(self) -> Optional[str]:
        """Get timestamp of last command - used for continuity calculation"""
        if not self.conn:
            return None

        try:
            with self._lock:
                cursor = self.conn.execute("""
                    SELECT timestamp FROM command_history
                    ORDER BY timestamp DESC LIMIT 1
                """)
                row = cursor.fetchone()
                return row["timestamp"] if row else None
        except Exception as e:
            logger.error(f"Failed to get last activity: {e}")
            return None

    def close(self):
        """Close database connection"""
        if self.conn:
            try:
                with self._lock:
                    self.conn.close()
                logger.info("SQLite connection closed")
            except Exception as e:
                logger.error(f"Error closing SQLite: {e}")
