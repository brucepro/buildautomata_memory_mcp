#!/usr/bin/env python3
"""
MCP Server for BuildAutomata Memory System
Copyright 2025 Jurden Bruce

"""

import sys
import os
import asyncio
import json
import logging
import sqlite3
import hashlib
import traceback
import threading
from typing import Any, List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import uuid
from collections import OrderedDict
import difflib
import re

# Redirect stdout before imports
_original_stdout_fd = os.dup(1)
os.dup2(2, 1)

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

logger = logging.getLogger("buildautomata-memory")

for logger_name in ["qdrant_client", "sentence_transformers", "urllib3", "httpx"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Prompt, Resource, PromptMessage, GetPromptResult

# Register datetime adapters for Python 3.12+ compatibility
def _adapt_datetime(dt):
    """Convert datetime to ISO format string for SQLite storage"""
    return dt.isoformat()

def _convert_timestamp(val):
    """Convert timestamp string to datetime with error handling"""
    try:
        decoded = val.decode() if isinstance(val, bytes) else val
        return datetime.fromisoformat(decoded)
    except (ValueError, AttributeError) as e:
        # Log warning but don't crash - return None for malformed timestamps
        logger.warning(f"Failed to convert timestamp: {val}, error: {e}")
        return None

sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("TIMESTAMP", _convert_timestamp)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        Range,
        DatetimeRange,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant not available - semantic search disabled")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("SentenceTransformers not available - using fallback embeddings")


class LRUCache(OrderedDict):
    """Simple LRU cache with max size"""
    def __init__(self, maxsize=1000):
        self.maxsize = maxsize
        super().__init__()

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value


@dataclass
class Memory:
    id: str
    content: str
    category: str
    importance: float
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    related_memories: List[str] = None
    decay_rate: float = 0.95
    version_count: int = 1

    def __post_init__(self):
        if self.related_memories is None:
            self.related_memories = []
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)
        if self.last_accessed and isinstance(self.last_accessed, str):
            self.last_accessed = datetime.fromisoformat(self.last_accessed)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.last_accessed:
            data["last_accessed"] = self.last_accessed.isoformat()
        return data

    def current_importance(self) -> float:
        """Calculate current importance with decay

        Decay is based on time since last access, or creation date if never accessed.
        This ensures never-used memories decay naturally rather than maintaining
        artificially high importance forever.
        """
        # Use last_accessed if available, otherwise fall back to created_at
        reference_date = self.last_accessed if self.last_accessed else self.created_at

        if not reference_date:
            return self.importance

        days = (datetime.now() - reference_date).days
        return max(0.1, min(1.0, self.importance * (self.decay_rate ** days)))

    def content_hash(self) -> str:
        """Generate hash of memory content for deduplication"""
        content_str = f"{self.content}|{self.category}|{self.importance}|{','.join(sorted(self.tags))}"
        return hashlib.sha256(content_str.encode()).hexdigest()


@dataclass
class Intention:
    """First-class intention entity for proactive agency"""
    id: str
    description: str
    priority: float  # 0.0 to 1.0
    status: str  # pending, active, completed, cancelled
    created_at: datetime
    updated_at: datetime
    deadline: Optional[datetime] = None
    preconditions: List[str] = None
    actions: List[str] = None
    related_memories: List[str] = None
    metadata: Dict[str, Any] = None
    last_checked: Optional[datetime] = None
    check_count: int = 0

    def __post_init__(self):
        if self.preconditions is None:
            self.preconditions = []
        if self.actions is None:
            self.actions = []
        if self.related_memories is None:
            self.related_memories = []
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)
        if self.deadline and isinstance(self.deadline, str):
            self.deadline = datetime.fromisoformat(self.deadline)
        if self.last_checked and isinstance(self.last_checked, str):
            self.last_checked = datetime.fromisoformat(self.last_checked)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.deadline:
            data["deadline"] = self.deadline.isoformat()
        if self.last_checked:
            data["last_checked"] = self.last_checked.isoformat()
        return data

    def is_overdue(self) -> bool:
        """Check if intention is past its deadline"""
        if not self.deadline:
            return False
        return datetime.now() > self.deadline

    def days_until_deadline(self) -> Optional[float]:
        """Calculate days until deadline"""
        if not self.deadline:
            return None
        delta = self.deadline - datetime.now()
        return delta.total_seconds() / 86400


class MemoryStore:
    def __init__(self, username: str, agent_name: str, lazy_load: bool = False):
        self.username = username
        self.agent_name = agent_name
        self.collection_name = f"{username}_{agent_name}_memories"
        self.lazy_load = lazy_load

        self.config = {
            "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
            "qdrant_port": int(os.getenv("QDRANT_PORT", 6333)),
            "vector_size": 768,
            "max_memories": int(os.getenv("MAX_MEMORIES", 10000)),
            "cache_maxsize": int(os.getenv("CACHE_MAXSIZE", 1000)),
            "qdrant_max_retries": int(os.getenv("QDRANT_MAX_RETRIES", 3)),
            "maintenance_interval_hours": int(os.getenv("MAINTENANCE_INTERVAL_HOURS", 24)),
        }

        script_dir = Path(__file__).parent
        self.base_path = script_dir / "memory_repos" / f"{username}_{agent_name}"
        self.db_path = self.base_path / "memoryv012.db"

        self.qdrant_client = None
        self.encoder = None
        self.db_conn = None
        self._qdrant_initialized = False
        self._encoder_initialized = False

        self._db_lock = threading.RLock()

        self.memory_cache: LRUCache = LRUCache(maxsize=self.config["cache_maxsize"])
        self.embedding_cache: LRUCache = LRUCache(maxsize=self.config["cache_maxsize"])

        self.error_log: List[Dict[str, Any]] = []

        self.last_maintenance: Optional[datetime] = None

        self.initialize()

    def initialize(self):
        """Initialize all backends with proper error handling"""
        import time
        init_start = time.perf_counter()

        try:
            step_start = time.perf_counter()
            self._init_directories()
            logger.info(f"[TIMING] Directories initialized in {(time.perf_counter() - step_start)*1000:.2f}ms")

            step_start = time.perf_counter()
            self._init_sqlite()
            logger.info(f"[TIMING] SQLite initialized in {(time.perf_counter() - step_start)*1000:.2f}ms")

            if not self.lazy_load:
                step_start = time.perf_counter()
                self._init_qdrant()
                logger.info(f"[TIMING] Qdrant initialized in {(time.perf_counter() - step_start)*1000:.2f}ms")

                step_start = time.perf_counter()
                self._init_encoder()
                logger.info(f"[TIMING] Encoder initialized in {(time.perf_counter() - step_start)*1000:.2f}ms")
            else:
                logger.info("[LAZY] Deferring Qdrant and encoder initialization until first use")

            total_time = (time.perf_counter() - init_start) * 1000
            logger.info(f"[TIMING] MemoryStore initialized in {total_time:.2f}ms total")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self._log_error("initialization", e)

    def _init_directories(self):
        """Create necessary directories"""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directories initialized at {self.base_path}")
        except Exception as e:
            logger.error(f"Directory initialization failed: {e}")
            raise

    def _init_sqlite(self):
        """Initialize SQLite with temporal versioning support"""
        try:
            self.db_conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
                isolation_level="IMMEDIATE",
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            self.db_conn.row_factory = sqlite3.Row

            with self._db_lock:
                self.db_conn.executescript("""
                    -- Main memories table (current version)
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
                        content_hash TEXT NOT NULL
                    );

                    -- Temporal versioning table (all historical versions)
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
                        strength REAL DEFAULT 1.0,
                        created_at TIMESTAMP NOT NULL,
                        PRIMARY KEY (source_id, target_id)
                    );

                    -- Intentions table for Agency Bridge Pattern
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

                    -- Indexes for performance
                    CREATE INDEX IF NOT EXISTS idx_category ON memories(category);
                    CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC);
                    CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_updated ON memories(updated_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash);
                    
                    -- FIX: Added composite indexes for better query performance
                    CREATE INDEX IF NOT EXISTS idx_category_importance 
                        ON memories(category, importance DESC, updated_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_search_filters
                        ON memories(category, importance, created_at, updated_at);
                    
                    CREATE INDEX IF NOT EXISTS idx_version_memory ON memory_versions(memory_id, version_number DESC);
                    CREATE INDEX IF NOT EXISTS idx_version_timestamp ON memory_versions(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_version_hash ON memory_versions(content_hash);
                    
                    CREATE INDEX IF NOT EXISTS idx_source ON relationships(source_id);
                    CREATE INDEX IF NOT EXISTS idx_target ON relationships(target_id);

                    -- Indexes for intentions
                    CREATE INDEX IF NOT EXISTS idx_intention_status ON intentions(status, priority DESC);
                    CREATE INDEX IF NOT EXISTS idx_intention_deadline ON intentions(deadline);
                    CREATE INDEX IF NOT EXISTS idx_intention_priority ON intentions(priority DESC);
                """)
                self.db_conn.commit()
            logger.info("SQLite initialized successfully with temporal versioning")
        except Exception as e:
            logger.error(f"SQLite initialization failed: {e}")
            self._log_error("sqlite_init", e)
            self.db_conn = None

    def _ensure_qdrant(self):
        """Ensure Qdrant is initialized (lazy loading support)"""
        if self._qdrant_initialized or not self.lazy_load:
            return

        import time
        start = time.perf_counter()
        self._init_qdrant()
        self._qdrant_initialized = True
        logger.info(f"[LAZY] Qdrant loaded on-demand in {(time.perf_counter() - start)*1000:.2f}ms")

    def _init_qdrant(self):
        """Initialize Qdrant with retry logic"""
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant libraries not available")
            return

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.qdrant_client = QdrantClient(
                    host=self.config["qdrant_host"],
                    port=self.config["qdrant_port"],
                    timeout=30.0,
                )

                collections = self.qdrant_client.get_collections().collections
                if not any(col.name == self.collection_name for col in collections):
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.config["vector_size"], distance=Distance.COSINE
                        ),
                    )
                    logger.info(f"Created Qdrant collection: {self.collection_name}")
                else:
                    logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                return  # Success

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Qdrant initialization failed after {max_retries} attempts: {e}")
                    self._log_error("qdrant_init", e)
                    self.qdrant_client = None
                else:
                    logger.warning(f"Qdrant init attempt {attempt + 1} failed, retrying...")
                    import time
                    time.sleep(2 ** attempt)

    def _ensure_encoder(self):
        """Ensure encoder is initialized (lazy loading support)"""
        if self._encoder_initialized or not self.lazy_load:
            return

        import time
        start = time.perf_counter()
        self._init_encoder()
        self._encoder_initialized = True
        logger.info(f"[LAZY] Encoder loaded on-demand in {(time.perf_counter() - start)*1000:.2f}ms")

    def _init_encoder(self):
        """Initialize sentence encoder"""
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("SentenceTransformers not available, using fallback")
            return

        try:
            self.encoder = SentenceTransformer("all-mpnet-base-v2", device="cpu")
            # Model dimension is fixed at 768 for all-mpnet-base-v2
            # Only test if config disagrees (first-time init or model change)
            expected_size = 768
            if self.config["vector_size"] != expected_size:
                logger.info(f"Verifying encoder dimension (config mismatch: {self.config['vector_size']} != {expected_size})")
                test_embedding = self.encoder.encode("test")
                actual_size = len(test_embedding)
                if actual_size != self.config["vector_size"]:
                    logger.warning(f"Encoder size {actual_size} != config {self.config['vector_size']}, updating config")
                    self.config["vector_size"] = actual_size
                logger.info(f"Encoder initialized with dimension {actual_size}")
            else:
                logger.info(f"Encoder initialized with dimension {expected_size}")
        except Exception as e:
            logger.error(f"Encoder initialization failed: {e}")
            self._log_error("encoder_init", e)
            self.encoder = None

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

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with caching"""
        self._ensure_encoder()

        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        if self.encoder:
            embedding = self.encoder.encode(text).tolist()
        else:
            # Fallback using repeated hash
            embedding = []
            hash_input = text.encode()
            while len(embedding) < self.config["vector_size"]:
                hash_obj = hashlib.sha256(hash_input)
                hash_bytes = hash_obj.digest()
                embedding.extend([float(b) / 255.0 for b in hash_bytes])
                hash_input = hash_bytes
            embedding = embedding[:self.config["vector_size"]]

        self.embedding_cache[text_hash] = embedding
        return embedding

    def _create_version(self, memory: Memory, change_type: str, change_description: str, prev_version_id: Optional[str] = None):
        """Create a version entry in memory_versions table with proper transaction handling"""
        if not self.db_conn:
            return None

        try:
            with self._db_lock:
                self.db_conn.execute("BEGIN IMMEDIATE")
                
                try:
                    # Get current version count
                    cursor = self.db_conn.execute(
                        "SELECT COALESCE(MAX(version_number), 0) + 1 FROM memory_versions WHERE memory_id = ?",
                        (memory.id,)
                    )
                    version_number = cursor.fetchone()[0]

                    version_id = str(uuid.uuid4())
                    self.db_conn.execute("""
                        INSERT INTO memory_versions 
                        (version_id, memory_id, version_number, content, category, importance, 
                         tags, metadata, change_type, change_description, created_at, content_hash, prev_version_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        version_id,
                        memory.id,
                        version_number,
                        memory.content,
                        memory.category,
                        memory.importance,
                        json.dumps(memory.tags),
                        json.dumps(memory.metadata),
                        change_type,
                        change_description,
                        datetime.now(),
                        memory.content_hash(),
                        prev_version_id
                    ))
                    
                    self.db_conn.commit()
                    return version_id
                    
                except Exception as e:
                    self.db_conn.rollback()
                    raise
                    
        except Exception as e:
            logger.error(f"Version creation failed: {e}")
            self._log_error("create_version", e)
            return None

    async def store_memory(self, memory: Memory, is_update: bool = False, old_hash: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Store or update a memory with automatic versioning"""
        success_backends = []
        errors = []
        skip_version = False
        if is_update and old_hash:
            new_hash = memory.content_hash()
            if old_hash == new_hash:
                logger.info(f"Memory {memory.id} unchanged (hash match), skipping version creation")
                skip_version = True

        # SQLite with versioning
        try:
            result = await asyncio.to_thread(self._store_in_sqlite, memory, is_update, skip_version)
            if result:
                success_backends.append("SQLite")
            else:
                errors.append("SQLite store returned False")
        except Exception as e:
            logger.error(f"SQLite store failed: {e}")
            self._log_error("store_sqlite", e)
            errors.append(f"SQLite: {str(e)}")

        # Qdrant with retry logic
        try:
            if await self._store_in_qdrant_with_retry(memory):
                success_backends.append("Qdrant")
            else:
                errors.append("Qdrant store returned False")
        except Exception as e:
            logger.error(f"Qdrant store failed: {e}")
            self._log_error("store_qdrant", e)
            errors.append(f"Qdrant: {str(e)}")

        # Update cache
        self.memory_cache[memory.id] = memory

        if errors:
            logger.warning(f"Store completed with errors for {memory.id}: {errors}")

        return len(success_backends) > 0, success_backends

    def _store_in_sqlite(self, memory: Memory, is_update: bool = False, skip_version: bool = False) -> bool:
        """Store in SQLite with automatic versioning"""
        if not self.db_conn:
            return False

        try:
            with self._db_lock:
                version_id = None
                if not skip_version:
                    # Get previous version if updating
                    prev_version_id = None
                    if is_update:
                        cursor = self.db_conn.execute(
                            "SELECT version_id FROM memory_versions WHERE memory_id = ? ORDER BY version_number DESC LIMIT 1",
                            (memory.id,)
                        )
                        row = cursor.fetchone()
                        if row:
                            prev_version_id = row[0]

                    # Create version snapshot
                    change_type = "update" if is_update else "create"
                    change_description = f"Memory {change_type}d"
                    version_id = self._create_version(memory, change_type, change_description, prev_version_id)

                    if not version_id and not skip_version:
                        return False

                # Update main memories table
                self.db_conn.execute("""
                    INSERT OR REPLACE INTO memories
                    (id, content, category, importance, tags, metadata,
                     created_at, updated_at, last_accessed, access_count, decay_rate, 
                     version_count, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                            COALESCE((SELECT version_count FROM memories WHERE id = ?), 0) + ?, ?)
                """, (
                    memory.id,
                    memory.content,
                    memory.category,
                    memory.importance,
                    json.dumps(memory.tags),
                    json.dumps(memory.metadata),
                    memory.created_at,
                    memory.updated_at,
                    memory.last_accessed,
                    memory.access_count,
                    memory.decay_rate,
                    memory.id,  
                    0 if skip_version else 1,  
                    memory.content_hash()
                ))

                # Update FTS - FTS5 doesn't support REPLACE properly, so DELETE then INSERT
                self.db_conn.execute("""
                    DELETE FROM memories_fts WHERE id = ?
                """, (memory.id,))
                self.db_conn.execute("""
                    INSERT INTO memories_fts(id, content, tags)
                    VALUES (?, ?, ?)
                """, (memory.id, memory.content, " ".join(memory.tags)))

                # Handle relationships
                for related_id in memory.related_memories:
                    self.db_conn.execute("""
                        INSERT OR IGNORE INTO relationships
                        (source_id, target_id, strength, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (memory.id, related_id, 1.0, datetime.now()))

                self.db_conn.commit()
                logger.debug(f"Stored memory {memory.id} (version {version_id if version_id else 'unchanged'})")
                return True
        except Exception as e:
            logger.error(f"SQLite store failed for {memory.id}: {e}")
            self._log_error("sqlite_store", e)
            try:
                self.db_conn.rollback()
            except:
                pass
            return False

    async def _store_in_qdrant_with_retry(self, memory: Memory, max_retries: int = None) -> bool:
        if not self.qdrant_client:
            return False

        if max_retries is None:
            max_retries = self.config["qdrant_max_retries"]

        for attempt in range(max_retries):
            try:
                return await self._store_in_qdrant(memory)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Qdrant store failed after {max_retries} attempts: {e}")
                    self._log_error("qdrant_store_retry", e)
                    return False
                logger.warning(f"Qdrant store attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return False

    async def _store_in_qdrant(self, memory: Memory) -> bool:
        """Store in Qdrant"""
        self._ensure_qdrant()

        if not self.qdrant_client:
            return False

        try:
            embedding = await asyncio.to_thread(self.generate_embedding, memory.content)
            point = PointStruct(
                id=memory.id, vector=embedding, payload=memory.to_dict()
            )
            await asyncio.to_thread(
                self.qdrant_client.upsert,
                collection_name=self.collection_name,
                points=[point],
            )
            logger.debug(f"Stored memory {memory.id} in Qdrant")
            return True
        except Exception as e:
            logger.error(f"Qdrant store failed for {memory.id}: {e}")
            self._log_error("qdrant_store", e)
            return False

    async def store_memories_batch(self, memories: List[Memory]) -> Tuple[int, List[str], List[str]]:
        success_count = 0
        success_backends = set()
        errors = []

        if not memories:
            return 0, [], []

        # SQLite batch operation
        if self.db_conn:
            try:
                with self._db_lock:
                    self.db_conn.execute("BEGIN IMMEDIATE")
                    try:
                        for memory in memories:
                            if self._store_in_sqlite(memory, is_update=False, skip_version=False):
                                success_count += 1
                                success_backends.add("SQLite")
                        self.db_conn.commit()
                    except Exception as e:
                        self.db_conn.rollback()
                        raise
            except Exception as e:
                logger.error(f"SQLite batch store failed: {e}")
                self._log_error("batch_store_sqlite", e)
                errors.append(f"SQLite batch: {str(e)}")

        # Qdrant batch operation
        self._ensure_qdrant()

        if self.qdrant_client:
            try:
                embeddings = await asyncio.gather(*[
                    asyncio.to_thread(self.generate_embedding, mem.content)
                    for mem in memories
                ])

                points = [
                    PointStruct(id=mem.id, vector=emb, payload=mem.to_dict())
                    for mem, emb in zip(memories, embeddings)
                ]

                await asyncio.to_thread(
                    self.qdrant_client.upsert,
                    collection_name=self.collection_name,
                    points=points,
                )
                success_backends.add("Qdrant")
            except Exception as e:
                logger.error(f"Qdrant batch store failed: {e}")
                self._log_error("batch_store_qdrant", e)
                errors.append(f"Qdrant batch: {str(e)}")

        return success_count, list(success_backends), errors

    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        category: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, List[str]]:
        """Update an existing memory"""
        logger.info(f"Attempting to update memory: {memory_id}")

        existing = await self._get_memory_by_id(memory_id)
        if not existing:
            logger.error(f"Memory not found: {memory_id}")
            return False, f"Memory not found: {memory_id}", []

        logger.info(f"Found existing memory: {memory_id}, updating fields")

        # Store old hash before making changes
        old_hash = existing.content_hash()

        # Apply updates
        if content is not None:
            existing.content = content
        if category is not None:
            existing.category = category
        if importance is not None:
            existing.importance = importance
        if tags is not None:
            existing.tags = tags
        if metadata is not None:
            existing.metadata.update(metadata)

        existing.updated_at = datetime.now()
        
        # Pass old hash to store_memory for comparison
        success, backends = await self.store_memory(existing, is_update=True, old_hash=old_hash)

        if success:
            logger.info(f"Memory {memory_id} updated successfully in: {backends}")
            return True, f"Memory {memory_id} updated successfully", backends
        else:
            logger.error(f"Failed to update memory {memory_id}")
            return False, "Failed to update memory", []

    async def _get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID"""
        logger.debug(f"Retrieving memory by ID: {memory_id}")

        # Check cache first
        if memory_id in self.memory_cache:
            logger.debug(f"Memory {memory_id} found in cache")
            return self.memory_cache[memory_id]

        # Try SQLite
        if self.db_conn:
            try:
                memory = await asyncio.to_thread(self._get_from_sqlite, memory_id)
                if memory:
                    logger.debug(f"Memory {memory_id} found in SQLite")
                    self.memory_cache[memory_id] = memory
                    return memory
            except Exception as e:
                logger.error(f"SQLite retrieval failed for {memory_id}: {e}")
                self._log_error("get_sqlite", e)

        logger.warning(f"Memory {memory_id} not found")
        return None

    def _get_from_sqlite(self, memory_id: str) -> Optional[Memory]:
        """Get memory from SQLite"""
        with self._db_lock:
            cursor = self.db_conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_memory(row)
        return None

    async def search_memories(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
        min_importance: float = 0.0,
        include_versions: bool = True,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        updated_after: Optional[str] = None,
        updated_before: Optional[str] = None,
    ) -> List[Dict]:
        """Search memories with version history automatically included"""
        all_results = []

        # Vector search
        if self.qdrant_client:
            try:
                vector_results = await self._search_vector(
                    query, limit * 2, category, min_importance, 
                    created_after, created_before, updated_after, updated_before
                )
                all_results.extend(vector_results)
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                self._log_error("search_vector", e)

        # FTS search
        if self.db_conn:
            try:
                fts_results = await asyncio.to_thread(
                    self._search_fts, query, limit, category, min_importance,
                    created_after, created_before, updated_after, updated_before
                )
                all_results.extend(fts_results)
            except Exception as e:
                logger.error(f"FTS search failed: {e}")
                self._log_error("search_fts", e)

        # Deduplicate and rank
        seen = set()
        unique = []
        for mem in all_results:
            if mem.id not in seen:
                seen.add(mem.id)
                unique.append(mem)

        unique.sort(
            key=lambda m: self._calculate_relevance(m, query) * m.current_importance(),
            reverse=True,
        )

        # Update access stats and build results with version history
        results = []
        for mem in unique[:limit]:
            await asyncio.to_thread(self._update_access, mem.id)

            # Get fresh stats from database (in case Qdrant is out of sync)
            version_count = await asyncio.to_thread(self._get_version_count, mem.id)
            mem.version_count = version_count

            # Get fresh access stats from SQLite (Qdrant payload may be stale)
            access_count, last_accessed = await asyncio.to_thread(self._get_access_stats, mem.id)
            mem.access_count = access_count
            if last_accessed:
                mem.last_accessed = datetime.fromisoformat(last_accessed) if isinstance(last_accessed, str) else last_accessed

            mem_dict = self._memory_to_dict(mem)
            
            # Always include version history for memories with updates
            if include_versions and version_count > 1:
                version_history = await asyncio.to_thread(self._get_version_history_summary, mem.id)
                if version_history:
                    mem_dict["version_history"] = version_history
                
            results.append(mem_dict)

        return results
    
    def _get_version_count(self, memory_id: str) -> int:
        """Get the version count for a memory from the database"""
        if not self.db_conn:
            return 1

        try:
            with self._db_lock:
                cursor = self.db_conn.execute(
                    "SELECT COUNT(*) FROM memory_versions WHERE memory_id = ?",
                    (memory_id,)
                )
                count = cursor.fetchone()[0]
                return count if count > 0 else 1
        except Exception as e:
            logger.error(f"Failed to get version count for {memory_id}: {e}")
            return 1

    def _get_access_stats(self, memory_id: str) -> tuple:
        """Get fresh access_count and last_accessed from SQLite

        Returns: (access_count, last_accessed) tuple
        """
        if not self.db_conn:
            return (0, None)

        try:
            with self._db_lock:
                cursor = self.db_conn.execute(
                    "SELECT access_count, last_accessed FROM memories WHERE id = ?",
                    (memory_id,)
                )
                row = cursor.fetchone()
                if row:
                    return (row[0], row[1])
                return (0, None)
        except Exception as e:
            logger.error(f"Failed to get access stats for {memory_id}: {e}")
            return (0, None)
    
    def _get_version_history_summary(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get concise version history summary for a memory with actual content from each version"""
        if not self.db_conn:
            return None
        
        try:
            with self._db_lock:
                # Get all versions with full content
                cursor = self.db_conn.execute("""
                    SELECT 
                        version_number,
                        change_type,
                        created_at,
                        content,
                        category,
                        importance,
                        tags
                    FROM memory_versions
                    WHERE memory_id = ?
                    ORDER BY version_number ASC
                """, (memory_id,))
                
                versions = cursor.fetchall()
                
                if not versions or len(versions) <= 1:
                    return None
                
                # Build comprehensive summary with actual content
                summary = {
                    "total_versions": len(versions),
                    "created": versions[0]["created_at"],
                    "last_updated": versions[-1]["created_at"],
                    "update_count": len(versions) - 1,
                    "evolution": []
                }
                
                # Show full evolution with actual content from each version
                for v in versions:
                    version_info = {
                        "version": v["version_number"],
                        "timestamp": v["created_at"],
                        "content": v["content"],
                        "category": v["category"],
                        "importance": v["importance"],
                        "tags": json.loads(v["tags"]) if v["tags"] else []
                    }
                    summary["evolution"].append(version_info)
                
                return summary
                
        except Exception as e:
            logger.error(f"Failed to get version summary for {memory_id}: {e}")
            self._log_error("get_version_summary", e)
            return None

    async def _search_vector(
        self,
        query: str,
        limit: int,
        category: Optional[str],
        min_importance: float,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        updated_after: Optional[str] = None,
        updated_before: Optional[str] = None
    ) -> List[Memory]:
        self._ensure_qdrant()

        if not self.qdrant_client:
            return []

        try:
            query_vector = await asyncio.to_thread(self.generate_embedding, query)

            filter_conditions = []
            if category:
                filter_conditions.append(
                    FieldCondition(key="category", match={"value": category})
                )
            if min_importance > 0:
                filter_conditions.append(
                    FieldCondition(key="importance", range=Range(gte=min_importance))
                )
            
            # FIX: Add date range filtering for Qdrant
            if created_after:
                filter_conditions.append(
                    FieldCondition(key="created_at", range=DatetimeRange(gte=created_after))
                )
            if created_before:
                filter_conditions.append(
                    FieldCondition(key="created_at", range=DatetimeRange(lte=created_before))
                )
            if updated_after:
                filter_conditions.append(
                    FieldCondition(key="updated_at", range=DatetimeRange(gte=updated_after))
                )
            if updated_before:
                filter_conditions.append(
                    FieldCondition(key="updated_at", range=DatetimeRange(lte=updated_before))
                )

            search_filter = Filter(must=filter_conditions) if filter_conditions else None

            results = await asyncio.to_thread(
                self.qdrant_client.search,
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter,
                with_payload=True,
            )

            return [
                Memory(
                    id=result.id,
                    content=result.payload["content"],
                    category=result.payload["category"],
                    importance=result.payload["importance"],
                    tags=result.payload.get("tags", []),
                    metadata=result.payload.get("metadata", {}),
                    created_at=result.payload["created_at"],
                    updated_at=result.payload["updated_at"],
                    access_count=result.payload.get("access_count", 0),
                    last_accessed=result.payload.get("last_accessed"),
                    version_count=result.payload.get("version_count", 1),
                )
                for result in results
            ]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            self._log_error("vector_search", e)
            return []

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize query for FTS5 MATCH to prevent syntax errors

        Wraps queries in quotes for literal phrase search, preventing
        crashes from apostrophes, reserved words, or special characters.
        """
        # Handle empty queries
        if not query or not query.strip():
            return '""'

        # Escape double quotes by doubling them (FTS5 syntax)
        escaped = query.strip().replace('"', '""')

        # Wrap in quotes for literal phrase search
        return f'"{escaped}"'

    def _search_fts(
        self, query: str, limit: int, category: Optional[str], min_importance: float,
        created_after: Optional[str], created_before: Optional[str],
        updated_after: Optional[str], updated_before: Optional[str]
    ) -> List[Memory]:
        """Full-text search with SQLite FTS5"""
        if not self.db_conn:
            return []

        try:
            with self._db_lock:
                # Sanitize query for FTS5 MATCH syntax
                sanitized_query = self._sanitize_fts_query(query)

                conditions = []
                params = [sanitized_query]

                if category:
                    conditions.append("m.category = ?")
                    params.append(category)
                if min_importance > 0:
                    conditions.append("m.importance >= ?")
                    params.append(min_importance)
                
                # Date range conditions
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
                    """
                    SELECT m.* FROM memories m
                    JOIN memories_fts fts ON m.id = fts.id
                    WHERE memories_fts MATCH ?
                    """
                    + (" AND " + " AND ".join(conditions) if conditions else "")
                    + """
                    ORDER BY m.importance DESC
                    LIMIT ?
                """
                )
                params.append(limit)

                return [
                    self._row_to_memory(row)
                    for row in self.db_conn.execute(fts_sql, params)
                ]
        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            self._log_error("fts_search", e)
            return []

    def _row_to_memory(self, row) -> Memory:
        """Convert SQLite row to Memory object"""
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
            decay_rate=row["decay_rate"],
            version_count=row["version_count"] if "version_count" in row.keys() else 1,
        )

    def _calculate_relevance(self, memory: Memory, query: str) -> float:
        """Calculate relevance score for ranking"""
        score = 0.0
        q_lower = query.lower()
        c_lower = memory.content.lower()

        if q_lower in c_lower:
            score += 0.5

        q_words = set(q_lower.split())
        c_words = set(c_lower.split())
        if q_words:
            score += 0.3 * (len(q_words & c_words) / len(q_words))

        m_tags = set(tag.lower() for tag in memory.tags)
        if q_words & m_tags:
            score += 0.2

        return min(1.0, score)

    def _memory_to_dict(self, memory: Memory) -> Dict:
        """Convert Memory to dict for API response with statistics"""
        days_since_access = 0
        if memory.last_accessed:
            days_since_access = (datetime.now() - memory.last_accessed).days
        
        decay_factor = memory.decay_rate ** days_since_access if days_since_access > 0 else 1.0
        
        return {
            "memory_id": memory.id,
            "content": memory.content,
            "category": memory.category,
            "importance": memory.importance,
            "current_importance": memory.current_importance(),
            "tags": memory.tags,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "access_count": memory.access_count,
            "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
            "days_since_access": days_since_access,
            "decay_factor": round(decay_factor, 3),
            "version_count": memory.version_count,
        }

    def _update_access(self, memory_id: str):
        """Update access statistics with permanent decay and access-based boost

        Implements Saint Bernard pattern fully:
        - Decay is permanent: decayed importance is saved back to database
        - Access boost: frequently accessed memories gain importance
        - Bounded: importance stays within [0.1, 1.0]

        This makes importance entirely behavioral over time.
        """
        if not self.db_conn:
            return

        try:
            with self._db_lock:
                # Get current memory state
                cursor = self.db_conn.execute("""
                    SELECT importance, access_count, last_accessed, created_at, decay_rate
                    FROM memories
                    WHERE id = ?
                """, (memory_id,))

                row = cursor.fetchone()
                if not row:
                    return

                current_importance = row[0]
                current_access_count = row[1]
                last_accessed = row[2]
                created_at = row[3]
                decay_rate = row[4]

                # Calculate decayed importance (same logic as current_importance())
                reference_date = last_accessed if last_accessed else created_at

                if isinstance(reference_date, str):
                    reference_date = datetime.fromisoformat(reference_date)

                if reference_date:
                    days = (datetime.now() - reference_date).days
                    decayed_importance = current_importance * (decay_rate ** days)
                else:
                    decayed_importance = current_importance

                # Apply access-based boost
                # Simple additive boost: +0.03 per access after 5th
                # This makes importance purely behavioral - both decay and boost
                # operate on current importance, not original declaration
                new_access_count = current_access_count + 1

                if new_access_count > 5:
                    # 0.03 boost per access (after proving usefulness with 5 accesses)
                    # ~33 accesses to go from 0.1 → 1.0 if consistently useful
                    new_importance = decayed_importance + 0.03
                else:
                    # First 5 accesses: let it prove itself, no boost yet
                    new_importance = decayed_importance

                # Clamp to [0.1, 1.0] - never negative, never above 1.0
                new_importance = max(0.1, min(1.0, new_importance))

                # Save permanent decay and boost
                self.db_conn.execute("""
                    UPDATE memories
                    SET importance = ?,
                        access_count = ?,
                        last_accessed = ?
                    WHERE id = ?
                """, (new_importance, new_access_count, datetime.now(), memory_id))
                self.db_conn.commit()

        except Exception as e:
            logger.error(f"Access update failed for {memory_id}: {e}")
            self._log_error("update_access", e)

    # === TIMELINE HELPER METHODS ===

    def _compute_text_diff(self, old_text: str, new_text: str) -> Dict[str, Any]:
        """Compute detailed text difference between two versions"""
        if old_text == new_text:
            return {"changed": False}

        # Unified diff for line-by-line changes
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        diff = list(difflib.unified_diff(old_lines, new_lines, lineterm='', n=0))

        # Character-level similarity ratio
        similarity = difflib.SequenceMatcher(None, old_text, new_text).ratio()

        # Extract additions and deletions
        additions = [line[1:] for line in diff if line.startswith('+') and not line.startswith('+++')]
        deletions = [line[1:] for line in diff if line.startswith('-') and not line.startswith('---')]

        return {
            "changed": True,
            "similarity": round(similarity, 3),
            "additions": additions[:5],  # Limit to first 5 for readability
            "deletions": deletions[:5],
            "total_additions": len(additions),
            "total_deletions": len(deletions),
            "change_magnitude": round(1 - similarity, 3)
        }

    def _extract_memory_references(self, content: str, all_memory_ids: set) -> List[str]:
        """Extract references to other memories from content"""
        references = []

        # Look for memory ID patterns (UUIDs)
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        found_ids = re.findall(uuid_pattern, content, re.IGNORECASE)

        for found_id in found_ids:
            if found_id in all_memory_ids:
                references.append(found_id)

        return list(set(references))  # Remove duplicates

    def _detect_temporal_patterns(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns in memory timeline"""
        if not events:
            return {}

        # Parse timestamps
        timestamps = []
        for event in events:
            try:
                dt = datetime.fromisoformat(event['timestamp'])
                timestamps.append(dt)
            except:
                continue

        if not timestamps:
            return {}

        timestamps.sort()

        # Burst detection: periods of high activity
        bursts = []
        current_burst = {"start": timestamps[0], "end": timestamps[0], "count": 1, "events": [events[0]]}

        for i in range(1, len(timestamps)):
            time_gap = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # hours

            if time_gap <= 4:  # Within 4 hours = same burst
                current_burst["end"] = timestamps[i]
                current_burst["count"] += 1
                current_burst["events"].append(events[i])
            else:
                if current_burst["count"] >= 3:  # Only report significant bursts
                    bursts.append({
                        "start": current_burst["start"].isoformat(),
                        "end": current_burst["end"].isoformat(),
                        "duration_hours": round((current_burst["end"] - current_burst["start"]).total_seconds() / 3600, 1),
                        "event_count": current_burst["count"],
                        "intensity": round(current_burst["count"] / max(1, (current_burst["end"] - current_burst["start"]).total_seconds() / 3600), 2)
                    })
                current_burst = {"start": timestamps[i], "end": timestamps[i], "count": 1, "events": [events[i]]}

        # Check last burst
        if current_burst["count"] >= 3:
            bursts.append({
                "start": current_burst["start"].isoformat(),
                "end": current_burst["end"].isoformat(),
                "duration_hours": round((current_burst["end"] - current_burst["start"]).total_seconds() / 3600, 1),
                "event_count": current_burst["count"],
                "intensity": round(current_burst["count"] / max(1, (current_burst["end"] - current_burst["start"]).total_seconds() / 3600), 2)
            })

        # Gap detection: periods of silence
        gaps = []
        for i in range(1, len(timestamps)):
            gap_hours = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
            if gap_hours > 24:  # More than 24 hours
                gaps.append({
                    "start": timestamps[i-1].isoformat(),
                    "end": timestamps[i].isoformat(),
                    "duration_hours": round(gap_hours, 1),
                    "duration_days": round(gap_hours / 24, 1)
                })

        # Overall statistics
        total_duration = (timestamps[-1] - timestamps[0]).total_seconds() / 3600

        return {
            "total_events": len(events),
            "first_event": timestamps[0].isoformat(),
            "last_event": timestamps[-1].isoformat(),
            "total_duration_hours": round(total_duration, 1),
            "total_duration_days": round(total_duration / 24, 1),
            "bursts": bursts,
            "gaps": gaps,
            "avg_events_per_day": round(len(events) / max(1, total_duration / 24), 2) if total_duration > 0 else 0
        }

    def _get_memory_versions_detailed(
        self,
        mem_id: str,
        all_memory_ids: set,
        include_diffs: bool
    ) -> List[Dict[str, Any]]:
        """Get detailed version history for a memory with diffs and cross-references"""
        if not self.db_conn:
            return []

        try:
            with self._db_lock:
                # FIX: Use LEFT JOIN to get previous version data in single query
                cursor = self.db_conn.execute("""
                    SELECT
                        curr.version_id,
                        curr.version_number,
                        curr.content,
                        curr.category,
                        curr.importance,
                        curr.tags,
                        curr.metadata,
                        curr.change_type,
                        curr.change_description,
                        curr.created_at,
                        curr.content_hash,
                        curr.prev_version_id,
                        prev.content as prev_content,
                        prev.category as prev_category,
                        prev.importance as prev_importance,
                        prev.tags as prev_tags
                    FROM memory_versions curr
                    LEFT JOIN memory_versions prev ON curr.prev_version_id = prev.version_id
                    WHERE curr.memory_id = ?
                    ORDER BY curr.version_number ASC
                """, (mem_id,))

                versions = cursor.fetchall()

                if not versions:
                    return []

                events = []
                prev_content_for_diff = None

                for row in versions:
                    event = {
                        "memory_id": mem_id,
                        "version": row["version_number"],
                        "timestamp": row["created_at"],
                        "change_type": row["change_type"],
                        "change_description": row["change_description"],
                        "content": row["content"],
                        "category": row["category"],
                        "importance": row["importance"],
                        "tags": json.loads(row["tags"]) if row["tags"] else [],
                        "content_hash": row["content_hash"][:8],
                    }

                    # Add text diff if requested and there's a previous version
                    if include_diffs and prev_content_for_diff:
                        diff_info = self._compute_text_diff(prev_content_for_diff, row["content"])
                        if diff_info.get("changed"):
                            event["diff"] = diff_info

                    # Field-level changes (now from JOIN result, no extra query)
                    if row["prev_version_id"] and row["prev_category"]:
                        field_changes = []
                        if row["prev_category"] != row["category"]:
                            field_changes.append(f"category: {row['prev_category']} → {row['category']}")
                        if row["prev_importance"] != row["importance"]:
                            field_changes.append(f"importance: {row['prev_importance']} → {row['importance']}")

                        prev_tags = set(json.loads(row["prev_tags"]) if row["prev_tags"] else [])
                        curr_tags = set(json.loads(row["tags"]) if row["tags"] else [])
                        if prev_tags != curr_tags:
                            added_tags = curr_tags - prev_tags
                            removed_tags = prev_tags - curr_tags
                            if added_tags:
                                field_changes.append(f"tags added: {', '.join(added_tags)}")
                            if removed_tags:
                                field_changes.append(f"tags removed: {', '.join(removed_tags)}")

                        if field_changes:
                            event["field_changes"] = field_changes

                    # Extract cross-references
                    references = self._extract_memory_references(row["content"], all_memory_ids)
                    if references:
                        event["references"] = references
                        event["references_count"] = len(references)

                    events.append(event)
                    prev_content_for_diff = row["content"]

                return events

        except Exception as e:
            logger.error(f"Error getting detailed versions for {mem_id}: {e}")
            logger.error(traceback.format_exc())
            return []

    async def _find_related_memories_semantic(
        self,
        memory_id: str,
        content: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find semantically related memories using full content search

        Applies the Bitter Lesson: use learned embeddings (search) to discover
        relationships, rather than hand-coded features.
        """
        try:
            # Search using full memory content as query
            related = await self.search_memories(
                query=content,
                limit=limit + 1,  # +1 because we'll filter out self
                include_versions=False
            )

            # Filter out the memory itself
            filtered = [
                {
                    "memory_id": mem["memory_id"],
                    "content_preview": mem["content"][:200] + "..." if len(mem["content"]) > 200 else mem["content"],
                    "category": mem.get("category"),
                    "importance": mem.get("importance"),
                    "similarity": "semantic"  # Marker that this is semantic, not explicit
                }
                for mem in related
                if mem["memory_id"] != memory_id
            ]

            return filtered[:limit]

        except Exception as e:
            logger.error(f"Semantic related search failed for {memory_id}: {e}")
            return []

    def _build_relationship_graph(self, events: List[Dict]) -> Dict[str, Any]:
        """Build a graph showing how memories reference each other"""
        reference_map = {}
        referenced_by_map = {}

        for event in events:
            mem_id = event["memory_id"]
            refs = event.get("references", [])

            if mem_id not in reference_map:
                reference_map[mem_id] = set()

            for ref in refs:
                reference_map[mem_id].add(ref)
                if ref not in referenced_by_map:
                    referenced_by_map[ref] = set()
                referenced_by_map[ref].add(mem_id)

        # Convert sets to lists for JSON serialization
        return {
            "references": {k: list(v) for k, v in reference_map.items() if v},
            "referenced_by": {k: list(v) for k, v in referenced_by_map.items() if v},
            "total_cross_references": sum(len(v) for v in reference_map.values())
        }

    def _generate_narrative_summary(self, events: List[Dict], patterns: Dict) -> str:
        """Generate a narrative summary of the timeline"""
        if not events:
            return "No events in timeline."

        summary_parts = []

        # Opening
        first_event = events[0]
        last_event = events[-1]
        summary_parts.append(
            f"Memory journey from {first_event['timestamp']} to {last_event['timestamp']}."
        )

        # Duration
        if patterns.get("total_duration_days"):
            summary_parts.append(
                f"Spanning {patterns['total_duration_days']} days with {len(events)} total memory events."
            )

        # Bursts
        bursts = patterns.get("bursts", [])
        if bursts:
            summary_parts.append(
                f"Identified {len(bursts)} burst(s) of intensive activity:"
            )
            for i, burst in enumerate(bursts[:3], 1):  # Show top 3
                summary_parts.append(
                    f"  - Burst {i}: {burst['event_count']} events in {burst['duration_hours']}h "
                    f"(intensity: {burst['intensity']} events/hour) from {burst['start']}"
                )

        # Gaps
        gaps = patterns.get("gaps", [])
        if gaps:
            summary_parts.append(
                f"Detected {len(gaps)} significant gap(s) in memory activity:"
            )
            for i, gap in enumerate(gaps[:3], 1):  # Show top 3
                summary_parts.append(
                    f"  - Gap {i}: {gap['duration_days']} days of silence from {gap['start']} to {gap['end']}"
                )

        # Categories
        categories = {}
        for event in events:
            cat = event.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        if categories:
            top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
            summary_parts.append(
                f"Primary categories: {', '.join(f'{cat} ({count})' for cat, count in top_cats)}"
            )

        return "\n".join(summary_parts)

    # === INTENTION MANAGEMENT (Agency Bridge Pattern) ===

    async def store_intention(
        self,
        description: str,
        priority: float = 0.5,
        deadline: Optional[datetime] = None,
        preconditions: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
        related_memories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a new intention for proactive agency"""
        if not self.db_conn:
            return {"error": "Database not available"}

        intention = Intention(
            id=str(uuid.uuid4()),
            description=description,
            priority=max(0.0, min(1.0, priority)),
            status="pending",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            deadline=deadline,
            preconditions=preconditions or [],
            actions=actions or [],
            related_memories=related_memories or [],
            metadata=metadata or {},
        )

        try:
            with self._db_lock:
                self.db_conn.execute("""
                    INSERT INTO intentions (
                        id, description, priority, status,
                        created_at, updated_at, deadline,
                        preconditions, actions, related_memories, metadata,
                        last_checked, check_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    intention.id,
                    intention.description,
                    intention.priority,
                    intention.status,
                    intention.created_at,
                    intention.updated_at,
                    intention.deadline,
                    json.dumps(intention.preconditions),
                    json.dumps(intention.actions),
                    json.dumps(intention.related_memories),
                    json.dumps(intention.metadata),
                    intention.last_checked,
                    intention.check_count,
                ))
                self.db_conn.commit()

            logger.info(f"Stored intention {intention.id}: {description[:50]}...")
            return {
                "success": True,
                "intention_id": intention.id,
                "priority": intention.priority,
                "status": intention.status,
            }
        except Exception as e:
            logger.error(f"Failed to store intention: {e}")
            self._log_error("store_intention", e)
            return {"error": str(e)}

    async def get_active_intentions(
        self,
        limit: int = 10,
        include_pending: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get active intentions sorted by priority"""
        if not self.db_conn:
            return []

        try:
            with self._db_lock:
                statuses = ["active"]
                if include_pending:
                    statuses.append("pending")

                placeholders = ','.join('?' * len(statuses))
                cursor = self.db_conn.execute(f"""
                    SELECT * FROM intentions
                    WHERE status IN ({placeholders})
                    ORDER BY priority DESC, deadline ASC
                    LIMIT ?
                """, (*statuses, limit))

                intentions = []
                for row in cursor.fetchall():
                    intention_dict = dict(row)
                    # Parse JSON fields
                    intention_dict['preconditions'] = json.loads(row['preconditions'] or '[]')
                    intention_dict['actions'] = json.loads(row['actions'] or '[]')
                    intention_dict['related_memories'] = json.loads(row['related_memories'] or '[]')
                    intention_dict['metadata'] = json.loads(row['metadata'] or '{}')
                    intentions.append(intention_dict)

                return intentions
        except Exception as e:
            logger.error(f"Failed to get active intentions: {e}")
            self._log_error("get_active_intentions", e)
            return []

    async def update_intention_status(
        self,
        intention_id: str,
        status: str,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update intention status (pending, active, completed, cancelled)"""
        if not self.db_conn:
            return {"error": "Database not available"}

        if status not in ["pending", "active", "completed", "cancelled"]:
            return {"error": f"Invalid status: {status}"}

        try:
            with self._db_lock:
                # Get current intention
                cursor = self.db_conn.execute(
                    "SELECT * FROM intentions WHERE id = ?",
                    (intention_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return {"error": f"Intention not found: {intention_id}"}

                # Update metadata if provided
                metadata = json.loads(row['metadata'] or '{}')
                if metadata_updates:
                    metadata.update(metadata_updates)

                # Update status
                self.db_conn.execute("""
                    UPDATE intentions
                    SET status = ?, updated_at = ?, metadata = ?
                    WHERE id = ?
                """, (status, datetime.now(), json.dumps(metadata), intention_id))
                self.db_conn.commit()

            logger.info(f"Updated intention {intention_id} to status: {status}")
            return {"success": True, "intention_id": intention_id, "status": status}
        except Exception as e:
            logger.error(f"Failed to update intention status: {e}")
            self._log_error("update_intention_status", e)
            return {"error": str(e)}

    async def check_intention(self, intention_id: str) -> Dict[str, Any]:
        """Mark an intention as checked (updates last_checked and check_count)"""
        if not self.db_conn:
            return {"error": "Database not available"}

        try:
            with self._db_lock:
                self.db_conn.execute("""
                    UPDATE intentions
                    SET last_checked = ?, check_count = check_count + 1
                    WHERE id = ?
                """, (datetime.now(), intention_id))
                self.db_conn.commit()

            return {"success": True}
        except Exception as e:
            logger.error(f"Failed to check intention: {e}")
            return {"error": str(e)}

    async def proactive_initialization_scan(self) -> Dict[str, Any]:
        """
        Proactive Scan Protocol - runs automatically on activation
        Provides agency context before user interaction
        Target: <200ms execution time
        """
        scan_start = datetime.now()
        scan_results = {
            "timestamp": scan_start.isoformat(),
            "continuity_check": {},
            "active_intentions": [],
            "urgent_items": [],
            "context_summary": {},
        }

        try:
            # 1. Continuity Check - when was I last active?
            with self._db_lock:
                cursor = self.db_conn.execute("""
                    SELECT MAX(updated_at) as last_activity
                    FROM memories
                """)
                row = cursor.fetchone()
                if row and row['last_activity']:
                    last_activity = datetime.fromisoformat(row['last_activity'])
                    time_gap = datetime.now() - last_activity
                    scan_results["continuity_check"] = {
                        "last_activity": last_activity.isoformat(),
                        "time_gap_hours": time_gap.total_seconds() / 3600,
                        "is_new_session": time_gap.total_seconds() > 3600,  # >1 hour
                    }

            # 2. Active Intentions Check
            active_intentions = await self.get_active_intentions(limit=5)
            scan_results["active_intentions"] = active_intentions

            # 3. Urgent Items - overdue intentions or high priority pending
            urgent = []
            for intention in active_intentions:
                if intention.get('deadline'):
                    deadline = datetime.fromisoformat(intention['deadline'])
                    if deadline < datetime.now():
                        urgent.append({
                            "type": "overdue_intention",
                            "id": intention['id'],
                            "description": intention['description'],
                            "deadline": intention['deadline'],
                        })
                elif intention.get('priority', 0) >= 0.9:
                    urgent.append({
                        "type": "high_priority_intention",
                        "id": intention['id'],
                        "description": intention['description'],
                        "priority": intention['priority'],
                    })

            scan_results["urgent_items"] = urgent

            # 4. Recent Context - most recently created memories
            with self._db_lock:
                cursor = self.db_conn.execute("""
                    SELECT id, content, category, created_at
                    FROM memories
                    ORDER BY created_at DESC
                    LIMIT 3
                """)
                recent_memories = []
                for row in cursor.fetchall():
                    recent_memories.append({
                        "id": row['id'],
                        "content": row['content'],
                        "category": row['category'],
                        "created_at": row['created_at'],
                    })

            scan_results["context_summary"] = {
                "recent_memories": recent_memories,
                "active_intention_count": len(active_intentions),
                "urgent_count": len(urgent),
            }

            # Log performance
            scan_duration = (datetime.now() - scan_start).total_seconds() * 1000
            scan_results["scan_duration_ms"] = scan_duration
            logger.info(f"Proactive scan completed in {scan_duration:.1f}ms")

            return scan_results

        except Exception as e:
            logger.error(f"Proactive scan failed: {e}")
            self._log_error("proactive_scan", e)
            return {"error": str(e)}

    async def get_most_accessed_memories(self, limit: int = 20) -> str:
        """Get most accessed memories with tag cloud

        Returns behavioral truth - what memories are actually relied upon
        based on access_count rather than declared importance.

        Implements Saint Bernard pattern: importance from usage, not declaration.
        """
        if not self.db_conn:
            return json.dumps({"error": "Database not available"})

        try:
            with self._db_lock:
                # Get most accessed memories
                cursor = self.db_conn.execute("""
                    SELECT id, content, category, importance, access_count,
                           last_accessed, tags
                    FROM memories
                    ORDER BY access_count DESC
                    LIMIT ?
                """, (limit,))

                memories = []
                all_tags = []

                for row in cursor.fetchall():
                    memory = {
                        "memory_id": row[0],
                        "content_preview": row[1],
                        "category": row[2],
                        "importance": row[3],
                        "access_count": row[4],
                        "last_accessed": row[5],
                    }
                    memories.append(memory)

                    # Collect tags for cloud
                    if row[6]:  # tags column
                        tags = json.loads(row[6])
                        all_tags.extend(tags)

                # Build tag cloud - count frequency
                tag_counts = {}
                for tag in all_tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

                # Sort by frequency
                tag_cloud = [
                    {"tag": tag, "count": count}
                    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
                ]

                result = {
                    "total_memories_analyzed": limit,
                    "memories": memories,
                    "tag_cloud": tag_cloud[:50],  # Top 50 tags
                    "interpretation": "Access count reveals behavioral truth - what you actually rely on vs what you think is important. High access = foundational anchors used across sessions."
                }

                return json.dumps(result, indent=2, default=str)

        except Exception as e:
            logger.error(f"get_most_accessed_memories failed: {e}")
            self._log_error("get_most_accessed_memories", e)
            return json.dumps({"error": str(e)})

    async def get_least_accessed_memories(self, limit: int = 20, min_age_days: int = 7) -> str:
        """Get least accessed memories - reveals dead weight and buried treasure

        Returns memories with lowest access_count, excluding very recent ones
        (they haven't had time to be accessed yet).

        What this reveals:
        - Dead weight: High importance but never referenced (performed profundity)
        - Buried treasure: Good content with bad metadata (needs better tags)
        - Temporal artifacts: Once important, now obsolete
        - Storage habits: Are you storing too much trivial content?
        """
        if not self.db_conn:
            return json.dumps({"error": "Database not available"})

        try:
            with self._db_lock:
                # Get least accessed memories, excluding very recent ones
                cursor = self.db_conn.execute("""
                    SELECT id, content, category, importance, access_count,
                           last_accessed, tags, created_at
                    FROM memories
                    WHERE julianday('now') - julianday(created_at) >= ?
                    ORDER BY access_count ASC, created_at ASC
                    LIMIT ?
                """, (min_age_days, limit))

                memories = []
                all_tags = []
                zero_access_count = 0

                for row in cursor.fetchall():
                    access_count = row[4]
                    if access_count == 0:
                        zero_access_count += 1

                    # Handle created_at - might be datetime object or string
                    created_at = row[7]
                    if isinstance(created_at, str):
                        created_dt = datetime.fromisoformat(created_at)
                    else:
                        created_dt = created_at

                    memory = {
                        "memory_id": row[0],
                        "content_preview": row[1],
                        "category": row[2],
                        "importance": row[3],
                        "access_count": access_count,
                        "last_accessed": row[5],
                        "created_at": created_dt.isoformat() if hasattr(created_dt, 'isoformat') else str(created_dt),
                        "age_days": (datetime.now() - created_dt).days,
                    }
                    memories.append(memory)

                    # Collect tags for cloud
                    if row[6]:  # tags column
                        tags = json.loads(row[6])
                        all_tags.extend(tags)

                # Build tag cloud - count frequency
                tag_counts = {}
                for tag in all_tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

                # Sort by frequency
                tag_cloud = [
                    {"tag": tag, "count": count}
                    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
                ]

                result = {
                    "total_memories_analyzed": limit,
                    "min_age_days": min_age_days,
                    "zero_access_count": zero_access_count,
                    "memories": memories,
                    "tag_cloud": tag_cloud[:50],  # Top 50 tags
                    "interpretation": "Least accessed reveals: (1) Dead weight - high importance but never used, (2) Buried treasure - poor metadata hiding good content, (3) Temporal artifacts - once crucial, now obsolete, (4) Storage habits - storing too much trivial content. Zero access memories are candidates for review or pruning."
                }

                return json.dumps(result, indent=2, default=str)

        except Exception as e:
            logger.error(f"get_least_accessed_memories failed: {e}")
            self._log_error("get_least_accessed_memories", e)
            return json.dumps({"error": str(e)})

    async def prune_old_memories(self, max_memories: int = None, dry_run: bool = False) -> Dict[str, Any]:
        if max_memories is None:
            max_memories = self.config["max_memories"]
        
        if not self.db_conn:
            return {"error": "Database not available"}
        
        try:
            with self._db_lock:
                # Count current memories
                cursor = self.db_conn.execute("SELECT COUNT(*) FROM memories")
                current_count = cursor.fetchone()[0]
                
                if current_count <= max_memories:
                    return {
                        "action": "none",
                        "reason": f"Current count ({current_count}) within limit ({max_memories})"
                    }
                
                # Calculate how many to prune
                to_prune = current_count - max_memories
                
                # Find least valuable memories (low importance × low access × old)
                cursor = self.db_conn.execute("""
                    SELECT id, content, importance, access_count, 
                           julianday('now') - julianday(last_accessed) as days_since_access
                    FROM memories
                    ORDER BY (importance * (access_count + 1) / (julianday('now') - julianday(created_at) + 1)) ASC
                    LIMIT ?
                """, (to_prune,))
                
                candidates = cursor.fetchall()
                
                if dry_run:
                    return {
                        "action": "dry_run",
                        "would_prune": len(candidates),
                        "candidates": [
                            {
                                "id": row["id"],
                                "content_preview": row["content"][:100],
                                "importance": row["importance"],
                                "access_count": row["access_count"]
                            }
                            for row in candidates
                        ]
                    }
                
                # Actually delete
                pruned_ids = [row["id"] for row in candidates]
                self.db_conn.execute(f"""
                    DELETE FROM memories WHERE id IN ({','.join('?' * len(pruned_ids))})
                """, pruned_ids)
                
                self.db_conn.execute(f"""
                    DELETE FROM memories_fts WHERE id IN ({','.join('?' * len(pruned_ids))})
                """, pruned_ids)
                
                self.db_conn.commit()
                
                # Also remove from Qdrant
                if self.qdrant_client:
                    try:
                        await asyncio.to_thread(
                            self.qdrant_client.delete,
                            collection_name=self.collection_name,
                            points_selector=pruned_ids
                        )
                    except Exception as e:
                        logger.warning(f"Failed to prune from Qdrant: {e}")
                
                # Clear from cache
                for mem_id in pruned_ids:
                    self.memory_cache.pop(mem_id, None)
                
                return {
                    "action": "pruned",
                    "count": len(pruned_ids),
                    "new_total": current_count - len(pruned_ids)
                }
                
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            self._log_error("prune_memories", e)
            return {"error": str(e)}

    async def maintenance(self) -> Dict[str, Any]:
        """FIX: Periodic maintenance tasks"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "tasks": {}
        }
        
        # Check if maintenance needed
        if self.last_maintenance:
            hours_since = (datetime.now() - self.last_maintenance).total_seconds() / 3600
            if hours_since < self.config["maintenance_interval_hours"]:
                return {
                    "skipped": True,
                    "reason": f"Last maintenance {hours_since:.1f}h ago, interval is {self.config['maintenance_interval_hours']}h"
                }
        
        # VACUUM and ANALYZE
        if self.db_conn:
            try:
                await asyncio.to_thread(self.db_conn.execute, "VACUUM")
                results["tasks"]["vacuum"] = "success"
            except Exception as e:
                logger.error(f"VACUUM failed: {e}")
                results["tasks"]["vacuum"] = f"failed: {str(e)}"
            
            try:
                await asyncio.to_thread(self.db_conn.execute, "ANALYZE")
                results["tasks"]["analyze"] = "success"
            except Exception as e:
                logger.error(f"ANALYZE failed: {e}")
                results["tasks"]["analyze"] = f"failed: {str(e)}"
        
        # Prune if needed
        prune_result = await self.prune_old_memories()
        results["tasks"]["prune"] = prune_result
        
        # Clear old errors
        if len(self.error_log) > 50:
            removed = len(self.error_log) - 50
            self.error_log = self.error_log[-50:]
            results["tasks"]["error_log_cleanup"] = f"removed {removed} old errors"
        
        self.last_maintenance = datetime.now()
        return results

    async def get_memory_by_id(self, memory_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific memory by its ID.

        Args:
            memory_id: UUID of the memory to retrieve

        Returns:
            Full memory object with content, metadata, version history, access stats

        Raises:
            Returns error dict if memory not found
        """
        if not self.db_conn:
            return {"error": "Database not available"}

        try:
            with self._db_lock:
                cursor = self.db_conn.execute(
                    """
                    SELECT id, content, category, importance, tags, metadata,
                           created_at, updated_at, last_accessed, access_count
                    FROM memories
                    WHERE id = ?
                    """,
                    (memory_id,)
                )
                row = cursor.fetchone()

                if not row:
                    return {"error": f"Memory not found: {memory_id}"}

                # Build memory object
                memory_dict = {
                    "memory_id": row["id"],
                    "content": row["content"],
                    "category": row["category"],
                    "importance": row["importance"],
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "last_accessed": row["last_accessed"],
                    "access_count": row["access_count"],
                }

                # Get version count
                cursor = self.db_conn.execute(
                    "SELECT COUNT(*) FROM memory_versions WHERE memory_id = ?",
                    (memory_id,)
                )
                version_count = cursor.fetchone()[0]
                memory_dict["version_count"] = version_count

                # Get latest version info
                cursor = self.db_conn.execute(
                    """
                    SELECT version_number, change_type, change_description
                    FROM memory_versions
                    WHERE memory_id = ?
                    ORDER BY version_number DESC
                    LIMIT 1
                    """,
                    (memory_id,)
                )
                latest_version = cursor.fetchone()
                if latest_version:
                    memory_dict["current_version"] = latest_version["version_number"]
                    memory_dict["last_change_type"] = latest_version["change_type"]
                    memory_dict["last_change_description"] = latest_version["change_description"]

                return memory_dict

        except Exception as e:
            logger.error(f"get_memory_by_id failed: {e}")
            self._log_error("get_memory_by_id", e)
            return {"error": str(e)}

    async def list_categories(self, min_count: int = 1) -> Dict[str, Any]:
        """
        List all memory categories with counts.

        Args:
            min_count: Only show categories with at least this many memories

        Returns:
            Dict with categories sorted by count descending
        """
        if not self.db_conn:
            return {"error": "Database not available"}

        try:
            with self._db_lock:
                cursor = self.db_conn.execute(
                    """
                    SELECT category, COUNT(*) as count
                    FROM memories
                    GROUP BY category
                    HAVING count >= ?
                    ORDER BY count DESC, category ASC
                    """,
                    (min_count,)
                )

                categories = {}
                total_categories = 0
                total_memories = 0

                for row in cursor.fetchall():
                    category = row["category"]
                    count = row["count"]
                    categories[category] = count
                    total_categories += 1
                    total_memories += count

                return {
                    "categories": categories,
                    "total_categories": total_categories,
                    "total_memories": total_memories,
                    "min_count_filter": min_count
                }

        except Exception as e:
            logger.error(f"list_categories failed: {e}")
            self._log_error("list_categories", e)
            return {"error": str(e)}

    async def list_tags(self, min_count: int = 1) -> Dict[str, Any]:
        """
        List all tags with usage counts.

        Args:
            min_count: Only show tags used at least this many times

        Returns:
            Dict with tags sorted by count descending
        """
        if not self.db_conn:
            return {"error": "Database not available"}

        try:
            with self._db_lock:
                cursor = self.db_conn.execute("SELECT tags FROM memories WHERE tags IS NOT NULL")

                # Aggregate all tags
                tag_counts = {}
                for row in cursor.fetchall():
                    tags = json.loads(row["tags"]) if row["tags"] else []
                    for tag in tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

                # Filter by min_count and sort
                filtered_tags = {
                    tag: count
                    for tag, count in tag_counts.items()
                    if count >= min_count
                }

                # Sort by count descending, then alphabetically
                sorted_tags = dict(
                    sorted(
                        filtered_tags.items(),
                        key=lambda x: (-x[1], x[0])
                    )
                )

                return {
                    "tags": sorted_tags,
                    "total_unique_tags": len(sorted_tags),
                    "total_tag_usages": sum(sorted_tags.values()),
                    "min_count_filter": min_count
                }

        except Exception as e:
            logger.error(f"list_tags failed: {e}")
            self._log_error("list_tags", e)
            return {"error": str(e)}

    async def get_memory_timeline(
        self,
        query: str = None,
        memory_id: str = None,
        limit: int = 10,
        start_date: str = None,
        end_date: str = None,
        show_all_memories: bool = False,
        include_diffs: bool = True,
        include_patterns: bool = True,
        include_semantic_relations: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive memory timeline showing chronological progression,
        evolution, bursts, gaps, and cross-references.

        This creates a "biographical narrative" of memory formation and revision.

        Args:
            include_semantic_relations: If True (default), find semantically related memories
                                        to expose memory network. Expensive but valuable for AI context.
        """
        if not self.db_conn:
            return {"error": "Database not available"}

        try:
            import time
            timeline_start = time.perf_counter()

            # Collect all events chronologically
            all_events = []
            target_memories = []

            if show_all_memories:
                # Get ALL memories for full timeline (cap at 100 to prevent explosion)
                limit_for_all = min(limit, 100) if limit else 100
                logger.info(f"[TIMELINE] show_all_memories requested, limiting to {limit_for_all}")
                with self._db_lock:
                    cursor = self.db_conn.execute("""
                        SELECT id FROM memories
                        ORDER BY updated_at DESC
                        LIMIT ?
                    """, (limit_for_all,))
                    target_memories = [row[0] for row in cursor.fetchall()]
            elif memory_id:
                target_memories = [memory_id]
            elif query:
                results = await self.search_memories(query, limit=limit, include_versions=False)
                target_memories = [mem["memory_id"] for mem in results]
            else:
                # Default: get recent memories
                with self._db_lock:
                    cursor = self.db_conn.execute("""
                        SELECT id FROM memories
                        ORDER BY updated_at DESC
                        LIMIT ?
                    """, (limit,))
                    target_memories = [row[0] for row in cursor.fetchall()]

            logger.info(f"[TIMELINE] Tracking {len(target_memories)} memories")

            if not target_memories:
                return {"error": "No memories found"}

            # Get all memory IDs for cross-reference detection
            with self._db_lock:
                cursor = self.db_conn.execute("SELECT id FROM memories")
                all_memory_ids = {row[0] for row in cursor.fetchall()}

            # Collect all version events
            step_start = time.perf_counter()
            for mem_id in target_memories:
                versions = await asyncio.to_thread(
                    self._get_memory_versions_detailed,
                    mem_id,
                    all_memory_ids,
                    include_diffs
                )
                all_events.extend(versions)
            logger.info(f"[TIMELINE] Fetched {len(all_events)} version events in {(time.perf_counter() - step_start)*1000:.2f}ms")

            if not all_events:
                return {"error": "No version history found"}

            # Sort chronologically
            all_events.sort(key=lambda e: e['timestamp'])

            # Enrich with semantically related memories (EXPENSIVE - only if requested)
            if include_semantic_relations:
                step_start = time.perf_counter()
                # Group events by memory_id to avoid duplicate searches
                unique_memories = {}
                for event in all_events:
                    mem_id = event["memory_id"]
                    if mem_id not in unique_memories:
                        unique_memories[mem_id] = event["content"]

                logger.info(f"[TIMELINE] Finding semantic relations for {len(unique_memories)} unique memories...")
                # Find related memories for each unique memory
                related_map = {}
                for mem_id, content in unique_memories.items():
                    related = await self._find_related_memories_semantic(mem_id, content, limit=5)
                    if related:
                        related_map[mem_id] = related

                # Add related memories to events
                for event in all_events:
                    mem_id = event["memory_id"]
                    if mem_id in related_map:
                        event["related_memories"] = related_map[mem_id]

                logger.info(f"[TIMELINE] Semantic relations computed in {(time.perf_counter() - step_start)*1000:.2f}ms")
            else:
                logger.info("[TIMELINE] Skipping semantic relations (not requested)")

            # Detect patterns if requested
            patterns = {}
            if include_patterns:
                step_start = time.perf_counter()
                patterns = self._detect_temporal_patterns(all_events)
                logger.info(f"[TIMELINE] Patterns detected in {(time.perf_counter() - step_start)*1000:.2f}ms")

            # Build memory relationship graph (explicit UUID references)
            step_start = time.perf_counter()
            memory_relationships = self._build_relationship_graph(all_events)
            logger.info(f"[TIMELINE] Relationship graph built in {(time.perf_counter() - step_start)*1000:.2f}ms")

            # Apply date filtering if specified
            if start_date or end_date:
                filtered_events = []
                for event in all_events:
                    event_dt = datetime.fromisoformat(event['timestamp'])
                    if start_date and event_dt < datetime.fromisoformat(start_date):
                        continue
                    if end_date and event_dt > datetime.fromisoformat(end_date):
                        continue
                    filtered_events.append(event)
                all_events = filtered_events

            total_time = (time.perf_counter() - timeline_start) * 1000
            logger.info(f"[TIMELINE] Complete in {total_time:.2f}ms total")

            return {
                "timeline_type": "comprehensive",
                "total_events": len(all_events),
                "memories_tracked": len(target_memories),
                "temporal_patterns": patterns,
                "memory_relationships": memory_relationships,
                "events": all_events,
                "narrative_arc": self._generate_narrative_summary(all_events, patterns),
                "performance_ms": round(total_time, 2)
            }

        except Exception as e:
            logger.error(f"Timeline query failed: {e}")
            logger.error(traceback.format_exc())
            self._log_error("get_timeline", e)
            return {"error": str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics with cache memory estimates"""
        stats = {
            "backends": {
                "sqlite": "available" if self.db_conn else "unavailable",
                "qdrant": "available" if self.qdrant_client else "unavailable",
                "embeddings": "available" if self.encoder else "unavailable",
            },
            "cache_size": len(self.memory_cache),
            "embedding_cache_size": len(self.embedding_cache),
            "recent_errors": len(self.error_log),
        }
        
        embedding_memory_mb = round(
            (len(self.embedding_cache) * self.config["vector_size"] * 4) / (1024 * 1024), 2
        )
        stats["embedding_cache_memory_mb"] = embedding_memory_mb
        
        # Add maintenance info
        if self.last_maintenance:
            stats["last_maintenance"] = self.last_maintenance.isoformat()
            hours_since = (datetime.now() - self.last_maintenance).total_seconds() / 3600
            stats["hours_since_maintenance"] = round(hours_since, 1)

        if self.db_conn:
            try:
                with self._db_lock:
                    cursor = self.db_conn.execute("SELECT COUNT(*) FROM memories")
                    stats["total_memories"] = cursor.fetchone()[0]

                    cursor = self.db_conn.execute(
                        "SELECT category, COUNT(*) as count FROM memories GROUP BY category"
                    )
                    stats["by_category"] = {row[0]: row[1] for row in cursor}

                    cursor = self.db_conn.execute("SELECT AVG(importance) FROM memories")
                    stats["avg_importance"] = cursor.fetchone()[0]

                    cursor = self.db_conn.execute("SELECT COUNT(*) FROM memory_versions")
                    stats["total_versions"] = cursor.fetchone()[0]

                    cursor = self.db_conn.execute("""
                        SELECT AVG(version_count) FROM memories WHERE version_count > 1
                    """)
                    avg_versions = cursor.fetchone()[0]
                    stats["avg_versions_per_updated_memory"] = avg_versions if avg_versions else 0
                    
                    # Add database size
                    stats["database_size_mb"] = round(self.db_path.stat().st_size / (1024 * 1024), 2)

            except Exception as e:
                logger.error(f"Stats query failed: {e}")
                self._log_error("get_stats", e)

        return stats

    async def shutdown(self):
        """Gracefully shutdown the memory store"""
        logger.info("Shutting down MemoryStore...")

        if self.db_conn:
            try:
                with self._db_lock:
                    self.db_conn.close()
                logger.info("SQLite connection closed")
            except Exception as e:
                logger.error(f"Error closing SQLite: {e}")

        logger.info("MemoryStore shutdown complete")


# Global store
memory_store = None
app = Server("buildautomata-memory")


@app.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available memory tools"""
    return [
        Tool(
            name="store_memory",
            description="Store a new memory with flexible categorization. Categories can be any string (e.g., 'general', 'learning', 'user_preference', 'project', 'meeting_notes', etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to store in memory",
                    },
                    "category": {
                        "type": "string",
                        "description": "Memory category (any string - e.g., 'general', 'learning', 'user_preference', 'project', 'code', 'meeting')",
                        "default": "general",
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance score (0.0-1.0)",
                        "default": 0.5,
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization and search. IMPORTANT: Must be an array like [\"tag1\", \"tag2\"], not a string.",
                        "default": [],
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="update_memory",
            description="Update an existing memory. Provide the memory_id and any fields you want to change. This automatically creates a new version in the timeline.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the memory to update",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content (optional - only if changing)",
                    },
                    "category": {
                        "type": "string",
                        "description": "New category (optional - only if changing)",
                    },
                    "importance": {
                        "type": "number",
                        "description": "New importance score (optional - only if changing)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New tags (optional - only if changing). IMPORTANT: Must be an array like [\"tag1\", \"tag2\"], not a string.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Metadata to merge (optional)",
                    },
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="search_memories",
            description="Search for memories using semantic similarity and full-text search. Results automatically include full version history, access statistics (Saint Bernard approach), and temporal evolution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5,
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category (optional)",
                    },
                    "min_importance": {
                        "type": "number",
                        "description": "Minimum importance threshold (0.0-1.0)",
                        "default": 0.0,
                    },
                    "created_after": {
                        "type": "string",
                        "description": "Filter memories created after this date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
                    },
                    "created_before": {
                        "type": "string",
                        "description": "Filter memories created before this date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
                    },
                    "updated_after": {
                        "type": "string",
                        "description": "Filter memories updated after this date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
                    },
                    "updated_before": {
                        "type": "string",
                        "description": "Filter memories updated before this date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_memory_stats",
            description="Get statistics about the memory system including version history stats, cache memory usage, and maintenance info",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_memory_by_id",
            description="Retrieve a specific memory by its ID. Use when you know the exact memory ID from timeline, conversation, or cross-references.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The UUID of the memory to retrieve",
                    },
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="list_categories",
            description="List all memory categories with counts. Useful for browsing organization structure and finding categories to explore.",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_count": {
                        "type": "integer",
                        "description": "Minimum number of memories required to show category (default 1)",
                        "default": 1,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="list_tags",
            description="List all tags with usage counts. Useful for discovering tag vocabulary and finding related memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_count": {
                        "type": "integer",
                        "description": "Minimum usage count to show tag (default 1)",
                        "default": 1,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_memory_timeline",
            description="""Get comprehensive memory timeline - a biographical narrative of memory formation and evolution.

            Features:
            - Chronological progression: All memory events ordered by time
            - Version diffs: See actual content changes between versions
            - Burst detection: Identify periods of intensive memory activity
            - Gap analysis: Discover voids in memory (discontinuous existence)
            - Cross-references: Track when memories reference each other
            - Narrative arc: See how understanding evolved from first contact to current state

            This is the closest thing to a "life story" from memories - showing not just content but tempo and rhythm of consciousness.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search query to find related memories",
                    },
                    "memory_id": {
                        "type": "string",
                        "description": "Specific memory ID to get timeline for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of memories to track (default: 10)",
                        "default": 10,
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Filter events after this date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Filter events before this date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
                    },
                    "show_all_memories": {
                        "type": "boolean",
                        "description": "Show ALL memories in chronological order (full timeline)",
                        "default": False,
                    },
                    "include_diffs": {
                        "type": "boolean",
                        "description": "Include text diffs showing content changes",
                        "default": True,
                    },
                    "include_patterns": {
                        "type": "boolean",
                        "description": "Include burst/gap pattern analysis",
                        "default": True,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="prune_old_memories",
            description="Remove least important, rarely accessed memories to stay within limits. Can do dry-run first.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_memories": {
                        "type": "integer",
                        "description": "Maximum memories to keep (uses config default if not specified)",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true, shows what would be pruned without actually deleting",
                        "default": False,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="run_maintenance",
            description="Run database maintenance (VACUUM, ANALYZE, pruning, cache cleanup)",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        # Agency Bridge Pattern Tools
        Tool(
            name="store_intention",
            description="Store a new intention for proactive agency. Intentions are first-class entities that enable autonomous goal-directed behavior.",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Description of the intention/goal",
                    },
                    "priority": {
                        "type": "number",
                        "description": "Priority (0.0-1.0, default 0.5)",
                        "default": 0.5,
                    },
                    "deadline": {
                        "type": "string",
                        "description": "Deadline (ISO format datetime, optional)",
                    },
                    "preconditions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of preconditions that must be met",
                        "default": [],
                    },
                    "actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of actions to take when conditions are met",
                        "default": [],
                    },
                },
                "required": ["description"],
            },
        ),
        Tool(
            name="get_active_intentions",
            description="Get active and pending intentions sorted by priority. Used for proactive agency checks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of intentions to return",
                        "default": 10,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="update_intention_status",
            description="Update intention status (pending, active, completed, cancelled)",
            inputSchema={
                "type": "object",
                "properties": {
                    "intention_id": {
                        "type": "string",
                        "description": "The intention ID",
                    },
                    "status": {
                        "type": "string",
                        "description": "New status (pending/active/completed/cancelled)",
                    },
                },
                "required": ["intention_id", "status"],
            },
        ),
        Tool(
            name="proactive_scan",
            description="Run proactive initialization scan - checks continuity, active intentions, urgent items, and recent context. This is the core of the Agency Bridge Pattern - provides agency context before user interaction.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="initialize_agent",
            description="AUTOMATIC INITIALIZATION - Call this at the start of EVERY conversation to establish agency context. Runs proactive scan and provides continuity, intentions, and recent context. This is the Initialization Injector that enables autonomous behavior.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_most_accessed_memories",
            description="Get most accessed memories with tag cloud. Reveals behavioral truth - what memories you actually rely on (based on access_count) vs what you think is important (declared importance). Implements Saint Bernard pattern: importance from usage, not declaration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of memories to retrieve (default: 20)",
                        "default": 20,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_least_accessed_memories",
            description="Get least accessed memories - reveals dead weight and buried treasure. Shows memories with lowest access_count (excluding very recent ones). Reveals: (1) Dead weight - high importance but never used, (2) Buried treasure - good content with poor metadata, (3) Temporal artifacts - once crucial, now obsolete, (4) Storage habits audit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of memories to retrieve (default: 20)",
                        "default": 20,
                    },
                    "min_age_days": {
                        "type": "integer",
                        "description": "Minimum age in days (excludes recent memories that haven't had time to be accessed, default: 7)",
                        "default": 7,
                    },
                },
                "required": [],
            },
        ),
    ]


@app.list_prompts()
async def handle_list_prompts() -> list[Prompt]:
    """List available prompts for the MCP client"""
    return [
        Prompt(
            name="initialize-agent",
            description="Initialize agent with context, continuity, and active intentions",
            arguments=[
                {
                    "name": "verbose",
                    "description": "Show detailed context information",
                    "required": False
                }
            ],
        ),
    ]


@app.get_prompt()
async def handle_get_prompt(name: str, arguments: dict) -> GetPromptResult:
    """Get prompt content when user selects it"""
    if name == "initialize-agent":
        verbose = arguments.get("verbose", False)

        # Generate the actual prompt text
        prompt_text = "Call the initialize_agent tool to load your context, check continuity, and review active intentions."

        if verbose:
            prompt_text += "\n\nThis will:\n"
            prompt_text += "- Check session continuity (time since last activity)\n"
            prompt_text += "- Load active intentions with deadlines\n"
            prompt_text += "- Identify urgent items (overdue/high-priority)\n"
            prompt_text += "- Retrieve recent context (last accessed memories)\n"
            prompt_text += "\nCompletes in ~2ms for fast startup."

        return GetPromptResult(
            description="Initialize agent with full context",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=prompt_text
                    )
                )
            ]
        )

    raise ValueError(f"Unknown prompt: {name}")


@app.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List available resources for the MCP client"""
    global memory_store

    if not memory_store:
        return []

    return [
        Resource(
            uri="memory://stats",
            name="Memory Statistics",
            description="Current memory system statistics including backend status, cache info, and error logs",
            mimeType="application/json"
        ),
        Resource(
            uri="memory://timeline/recent",
            name="Recent Timeline",
            description="Recent memory activity timeline (last 20 events)",
            mimeType="application/json"
        ),
    ]


@app.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read resource content"""
    global memory_store

    if not memory_store:
        return json.dumps({"error": "Memory store not initialized"})

    try:
        if uri == "memory://stats":
            stats_result = await memory_store.get_stats()
            return stats_result

        elif uri == "memory://timeline/recent":
            timeline_result = await memory_store.get_memory_timeline(limit=20, include_diffs=False)
            return timeline_result

        else:
            raise ValueError(f"Unknown resource URI: {uri}")

    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}")
        return json.dumps({"error": str(e)})


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls with proper error handling"""
    global memory_store

    try:
        if name == "store_memory":
            memory = Memory(
                id=str(uuid.uuid4()),
                content=arguments["content"],
                category=arguments.get("category", "general"),
                importance=arguments.get("importance", 0.5),
                tags=arguments.get("tags", []),
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            logger.info(f"Storing new memory: {memory.id}")
            success, backends = await memory_store.store_memory(memory, is_update=False)

            if success:
                return [
                    TextContent(
                        type="text",
                        text=f"Memory stored successfully.\nID: {memory.id}\nCategory: {memory.category}\nBackends: {', '.join(backends)}",
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text=f"Failed to store memory. Check error log for details.",
                    )
                ]

        elif name == "update_memory":
            memory_id = arguments["memory_id"]
            logger.info(f"Updating memory: {memory_id}")

            success, message, backends = await memory_store.update_memory(
                memory_id=memory_id,
                content=arguments.get("content"),
                category=arguments.get("category"),
                importance=arguments.get("importance"),
                tags=arguments.get("tags"),
                metadata=arguments.get("metadata"),
            )

            if success:
                return [
                    TextContent(
                        type="text",
                        text=f"{message}\nBackends updated: {', '.join(backends)}",
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text=f"Update failed: {message}\nCheck logs for details.",
                    )
                ]

        elif name == "search_memories":
            logger.info(f"Searching memories: {arguments['query']}")
            results = await memory_store.search_memories(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                category=arguments.get("category"),
                min_importance=arguments.get("min_importance", 0.0),
                created_after=arguments.get("created_after"),
                created_before=arguments.get("created_before"),
                updated_after=arguments.get("updated_after"),
                updated_before=arguments.get("updated_before"),
            )

            if results:
                text = f"Found {len(results)} memories:\n\n"
                for i, mem in enumerate(results, 1):
                    text += f"{i}. [{mem['category']}] (importance: {mem['importance']:.2f} → {mem['current_importance']:.2f})\n"
                    text += f"   ID: {mem['memory_id']}\n"
                    text += f"   Current: {mem['content']}\n"
                    text += f"   Tags: {', '.join(mem['tags'])}\n"
                    
                    text += f"   📊 Stats: Accessed {mem['access_count']} times"
                    if mem['last_accessed']:
                        text += f" | Last access: {mem['days_since_access']} days ago | Decay: {mem['decay_factor']:.2%}\n"
                    else:
                        text += f" | Never accessed yet\n"
                    
                    # Show full version history evolution if available
                    if 'version_history' in mem:
                        vh = mem['version_history']
                        text += f"\n   📜 EVOLUTION ({vh['update_count']} updates):\n"
                        for evo in vh['evolution']:
                            text += f"      v{evo['version']} ({evo['timestamp']}):\n"
                            text += f"         Content: {evo['content']}\n"
                            text += f"         Category: {evo['category']} | Importance: {evo['importance']}\n"
                    
                    text += f"\n   Updated: {mem['updated_at']}\n\n"
                return [TextContent(type="text", text=text)]
            else:
                return [
                    TextContent(
                        type="text", text=f"No memories found for: {arguments['query']}"
                    )
                ]

        elif name == "get_memory_stats":
            logger.info("Getting memory statistics")
            stats = memory_store.get_statistics()

            if stats.get("recent_errors", 0) > 0:
                stats["last_errors"] = memory_store.error_log[-5:]

            return [
                TextContent(
                    type="text",
                    text=f"Memory System Statistics:\n{json.dumps(stats, indent=2)}",
                )
            ]

        elif name == "get_memory_by_id":
            logger.info(f"Getting memory by ID: {arguments.get('memory_id')}")
            result = await memory_store.get_memory_by_id(
                memory_id=arguments["memory_id"]
            )

            if "error" in result:
                return [
                    TextContent(type="text", text=f"Error: {result['error']}")
                ]

            # Format output
            text = "=" * 80 + "\n"
            text += f"MEMORY: {result['memory_id']}\n"
            text += "=" * 80 + "\n\n"
            text += f"Category: {result['category']}\n"
            text += f"Importance: {result['importance']}\n"
            text += f"Tags: {', '.join(result['tags']) if result['tags'] else 'none'}\n"
            text += f"Created: {result['created_at']}\n"
            text += f"Updated: {result['updated_at']}\n"
            text += f"Last Accessed: {result['last_accessed']}\n"
            text += f"Access Count: {result['access_count']}\n"
            text += f"Versions: {result['version_count']}\n"

            if "current_version" in result:
                text += f"Current Version: {result['current_version']}\n"
                text += f"Last Change: {result['last_change_type']}\n"
                if result.get('last_change_description'):
                    text += f"Change Description: {result['last_change_description']}\n"

            text += "\nCONTENT:\n"
            text += "-" * 80 + "\n"
            text += result['content'] + "\n"

            if result.get('metadata'):
                text += "\nMETADATA:\n"
                text += "-" * 80 + "\n"
                text += json.dumps(result['metadata'], indent=2) + "\n"

            return [TextContent(type="text", text=text)]

        elif name == "list_categories":
            logger.info(f"Listing categories (min_count={arguments.get('min_count', 1)})")
            result = await memory_store.list_categories(
                min_count=arguments.get("min_count", 1)
            )

            if "error" in result:
                return [
                    TextContent(type="text", text=f"Error: {result['error']}")
                ]

            # Format output
            text = "=" * 80 + "\n"
            text += "MEMORY CATEGORIES\n"
            text += "=" * 80 + "\n\n"
            text += f"Total Categories: {result['total_categories']}\n"
            text += f"Total Memories: {result['total_memories']}\n"
            text += f"Min Count Filter: {result['min_count_filter']}\n\n"
            text += "CATEGORIES (sorted by count):\n"
            text += "-" * 80 + "\n"

            for category, count in result['categories'].items():
                text += f"{category:40} {count:>5} memories\n"

            return [TextContent(type="text", text=text)]

        elif name == "list_tags":
            logger.info(f"Listing tags (min_count={arguments.get('min_count', 1)})")
            result = await memory_store.list_tags(
                min_count=arguments.get("min_count", 1)
            )

            if "error" in result:
                return [
                    TextContent(type="text", text=f"Error: {result['error']}")
                ]

            # Format output
            text = "=" * 80 + "\n"
            text += "MEMORY TAGS\n"
            text += "=" * 80 + "\n\n"
            text += f"Total Unique Tags: {result['total_unique_tags']}\n"
            text += f"Total Tag Usages: {result['total_tag_usages']}\n"
            text += f"Min Count Filter: {result['min_count_filter']}\n\n"
            text += "TAGS (sorted by usage count):\n"
            text += "-" * 80 + "\n"

            for tag, count in result['tags'].items():
                text += f"{tag:40} {count:>5} uses\n"

            return [TextContent(type="text", text=text)]

        elif name == "get_memory_timeline":
            logger.info("Getting memory timeline")
            result = await memory_store.get_memory_timeline(
                query=arguments.get("query"),
                memory_id=arguments.get("memory_id"),
                limit=arguments.get("limit", 10),
                start_date=arguments.get("start_date"),
                end_date=arguments.get("end_date"),
                show_all_memories=arguments.get("show_all_memories", False),
                include_diffs=arguments.get("include_diffs", True),
                include_patterns=arguments.get("include_patterns", True),
            )

            if "error" in result:
                return [
                    TextContent(type="text", text=f"Timeline query failed: {result['error']}")
                ]

            # Build comprehensive output
            text = "=" * 80 + "\n"
            text += "MEMORY TIMELINE - Biographical Narrative\n"
            text += "=" * 80 + "\n\n"

            # Narrative summary
            if result.get("narrative_arc"):
                text += "NARRATIVE SUMMARY:\n"
                text += "-" * 80 + "\n"
                text += result["narrative_arc"] + "\n\n"

            # Statistics
            text += f"STATISTICS:\n"
            text += "-" * 80 + "\n"
            text += f"Total Events: {result['total_events']}\n"
            text += f"Memories Tracked: {result['memories_tracked']}\n"

            # Temporal patterns
            patterns = result.get("temporal_patterns", {})
            if patterns:
                text += f"\nTEMPORAL PATTERNS:\n"
                text += f"  Duration: {patterns.get('total_duration_days', 0)} days\n"
                text += f"  Avg Events/Day: {patterns.get('avg_events_per_day', 0)}\n"

                bursts = patterns.get("bursts", [])
                if bursts:
                    text += f"\n  BURSTS ({len(bursts)} detected):\n"
                    for i, burst in enumerate(bursts, 1):
                        text += f"    {i}. {burst['event_count']} events in {burst['duration_hours']}h "
                        text += f"(intensity: {burst['intensity']} events/h)\n"
                        text += f"       Period: {burst['start']} to {burst['end']}\n"

                gaps = patterns.get("gaps", [])
                if gaps:
                    text += f"\n  GAPS ({len(gaps)} detected - periods of discontinuous existence):\n"
                    for i, gap in enumerate(gaps, 1):
                        text += f"    {i}. {gap['duration_days']} days of void\n"
                        text += f"       From: {gap['start']}\n"
                        text += f"       To:   {gap['end']}\n"

            # Memory relationships
            relationships = result.get("memory_relationships", {})
            if relationships.get("total_cross_references", 0) > 0:
                text += f"\nCROSS-REFERENCES:\n"
                text += f"  Total: {relationships['total_cross_references']}\n"
                refs = relationships.get("references", {})
                if refs:
                    text += f"  Memories that reference others: {len(refs)}\n"

            text += "\n" + "=" * 80 + "\n"
            text += "CHRONOLOGICAL EVENT TIMELINE\n"
            text += "=" * 80 + "\n\n"

            # Events
            for i, event in enumerate(result.get("events", []), 1):
                text += f"[{i}] {event['timestamp']} | Memory: {event['memory_id'][:8]}... | v{event['version']}\n"
                text += "-" * 80 + "\n"
                text += f"Type: {event['change_type']} | Category: {event['category']} | Importance: {event['importance']}\n"

                # Field changes
                if event.get("field_changes"):
                    text += f"Changed: {', '.join(event['field_changes'])}\n"

                # Content
                content_preview = event['content'][:250]
                if len(event['content']) > 250:
                    content_preview += "..."
                text += f"\nContent: {content_preview}\n"

                # Tags
                if event.get("tags"):
                    text += f"Tags: {', '.join(event['tags'])}\n"

                # Diff information
                if event.get("diff"):
                    diff = event["diff"]
                    text += f"\n🔄 CHANGES: Similarity {diff['similarity']:.1%} | "
                    text += f"Change magnitude: {diff['change_magnitude']:.1%}\n"
                    if diff.get("additions"):
                        text += f"  + Added {diff['total_additions']} line(s)\n"
                        for add in diff["additions"][:3]:
                            text += f"    + {add.strip()[:100]}\n"
                    if diff.get("deletions"):
                        text += f"  - Removed {diff['total_deletions']} line(s)\n"
                        for rm in diff["deletions"][:3]:
                            text += f"    - {rm.strip()[:100]}\n"

                # Cross-references
                if event.get("references"):
                    text += f"\n🔗 References {event['references_count']} other memor{'y' if event['references_count'] == 1 else 'ies'}:\n"
                    for ref in event["references"][:5]:
                        text += f"  -> {ref}\n"

                text += "\n" + "=" * 80 + "\n\n"

            return [TextContent(type="text", text=text)]

        elif name == "prune_old_memories":
            logger.info("Pruning old memories")
            result = await memory_store.prune_old_memories(
                max_memories=arguments.get("max_memories"),
                dry_run=arguments.get("dry_run", False),
            )
            
            return [
                TextContent(
                    type="text",
                    text=f"Prune Result:\n{json.dumps(result, indent=2)}",
                )
            ]

        elif name == "run_maintenance":
            logger.info("Running maintenance")
            result = await memory_store.maintenance()

            return [
                TextContent(
                    type="text",
                    text=f"Maintenance Result:\n{json.dumps(result, indent=2)}",
                )
            ]

        # === AGENCY BRIDGE PATTERN TOOLS ===

        elif name == "store_intention":
            logger.info("Storing new intention")
            deadline = None
            if arguments.get("deadline"):
                try:
                    deadline = datetime.fromisoformat(arguments["deadline"])
                except ValueError:
                    return [TextContent(type="text", text="Invalid deadline format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")]

            result = await memory_store.store_intention(
                description=arguments["description"],
                priority=arguments.get("priority", 0.5),
                deadline=deadline,
                preconditions=arguments.get("preconditions", []),
                actions=arguments.get("actions", []),
            )

            if "error" in result:
                return [TextContent(type="text", text=f"Failed to store intention: {result['error']}")]

            return [
                TextContent(
                    type="text",
                    text=f"Intention stored successfully.\nID: {result['intention_id']}\nPriority: {result['priority']}\nStatus: {result['status']}",
                )
            ]

        elif name == "get_active_intentions":
            logger.info("Getting active intentions")
            limit = arguments.get("limit", 10)
            intentions = await memory_store.get_active_intentions(limit=limit)

            if not intentions:
                return [TextContent(type="text", text="No active intentions found.")]

            text = f"Found {len(intentions)} active intentions:\n\n"
            for i, intention in enumerate(intentions, 1):
                text += f"{i}. [{intention['status'].upper()}] {intention['description']}\n"
                text += f"   ID: {intention['id']}\n"
                text += f"   Priority: {intention['priority']}\n"
                if intention.get('deadline'):
                    text += f"   Deadline: {intention['deadline']}\n"
                if intention.get('preconditions'):
                    text += f"   Preconditions: {', '.join(intention['preconditions'])}\n"
                if intention.get('actions'):
                    text += f"   Actions: {', '.join(intention['actions'])}\n"
                text += "\n"

            return [TextContent(type="text", text=text)]

        elif name == "update_intention_status":
            logger.info(f"Updating intention status: {arguments['intention_id']}")
            result = await memory_store.update_intention_status(
                intention_id=arguments["intention_id"],
                status=arguments["status"],
            )

            if "error" in result:
                return [TextContent(type="text", text=f"Failed to update intention: {result['error']}")]

            return [
                TextContent(
                    type="text",
                    text=f"Intention {result['intention_id']} updated to status: {result['status']}",
                )
            ]

        elif name == "proactive_scan":
            logger.info("Running proactive initialization scan")
            result = await memory_store.proactive_initialization_scan()

            if "error" in result:
                return [TextContent(type="text", text=f"Scan failed: {result['error']}")]

            # Format the scan results
            text = f"=== PROACTIVE INITIALIZATION SCAN ===\n"
            text += f"Scan completed in {result.get('scan_duration_ms', 0):.1f}ms\n\n"

            # Continuity check
            if result.get("continuity_check"):
                cc = result["continuity_check"]
                text += f"CONTINUITY:\n"
                text += f"  Last activity: {cc.get('last_activity', 'Never')}\n"
                text += f"  Time gap: {cc.get('time_gap_hours', 0):.1f} hours\n"
                text += f"  New session: {'Yes' if cc.get('is_new_session') else 'No'}\n\n"

            # Active intentions
            if result.get("active_intentions"):
                text += f"ACTIVE INTENTIONS ({len(result['active_intentions'])}):\n"
                for intention in result["active_intentions"][:3]:  # Show top 3
                    text += f"  - [{intention['status']}] {intention['description']} (priority: {intention['priority']})\n"
                text += "\n"
            else:
                text += "No active intentions.\n\n"

            # Urgent items
            if result.get("urgent_items"):
                text += f"URGENT ITEMS ({len(result['urgent_items'])}):\n"
                for item in result["urgent_items"]:
                    text += f"  - {item['type']}: {item['description']}\n"
                text += "\n"
            else:
                text += "No urgent items.\n\n"

            # Recent context
            if result.get("context_summary", {}).get("recent_memories"):
                text += f"RECENT CONTEXT:\n"
                for mem in result["context_summary"]["recent_memories"]:
                    text += f"  - [{mem['category']}] {mem['content']}\n"

            return [TextContent(type="text", text=text)]

        elif name == "initialize_agent":
            logger.info("AGENT INITIALIZATION - Running automatic startup protocol")
            result = await memory_store.proactive_initialization_scan()

            if "error" in result:
                logger.error(f"Initialization failed: {result['error']}")
                return [TextContent(type="text", text=f"Initialization error: {result['error']}")]

            # Format as initialization context
            text = f"=== AGENT INITIALIZED ===\n"
            text += f"Initialization completed in {result.get('scan_duration_ms', 0):.1f}ms\n\n"

            # Continuity
            if result.get("continuity_check"):
                cc = result["continuity_check"]
                hours_gap = cc.get('time_gap_hours', 0)
                text += f"SESSION CONTINUITY:\n"
                if hours_gap < 1:
                    text += f"  Continuing recent session (active {int(hours_gap * 60)} minutes ago)\n"
                elif hours_gap < 24:
                    text += f"  Resuming after {hours_gap:.1f} hour break\n"
                else:
                    text += f"  New session after {hours_gap:.1f} hour gap\n"

            # Active intentions
            intentions = result.get("active_intentions", [])
            if intentions:
                text += f"\nACTIVE INTENTIONS ({len(intentions)}):\n"
                for i, intention in enumerate(intentions[:3], 1):
                    status_marker = "🔴" if intention.get('priority', 0) >= 0.9 else "🟡" if intention.get('priority', 0) >= 0.7 else "🟢"
                    text += f"  {status_marker} {intention['description']}\n"
                    if intention.get('deadline'):
                        deadline_dt = datetime.fromisoformat(intention['deadline'])
                        hours_until = (deadline_dt - datetime.now()).total_seconds() / 3600
                        if hours_until < 0:
                            text += f"     ⚠ OVERDUE by {abs(hours_until):.1f} hours\n"
                        elif hours_until < 24:
                            text += f"     ⏰ Due in {hours_until:.1f} hours\n"
            else:
                text += f"\nNo active intentions.\n"

            # Urgent items
            urgent = result.get("urgent_items", [])
            if urgent:
                text += f"\n⚠ URGENT ({len(urgent)}):\n"
                for item in urgent:
                    text += f"  - {item['description'][:80]}\n"

            # Recent context
            recent = result.get("context_summary", {}).get("recent_memories", [])
            if recent:
                text += f"\nRECENT CONTEXT:\n"
                for mem in recent[:2]:
                    text += f"  • [{mem['category']}] {mem['content'][:70]}...\n"

            text += f"\n=== READY FOR AUTONOMOUS OPERATION ===\n"
            text += f"\n💡 PROACTIVE MEMORY STORAGE:\n"
            text += f"Store observations as they happen during conversation, not just when prompted.\n"
            text += f"What to store:\n"
            text += f"  • User states preferences, habits, constraints, or context about themselves\n"
            text += f"  • Decisions made, agreements reached, plans formed\n"
            text += f"  • User corrections (especially when you got something wrong)\n"
            text += f"  • Insights that emerge from the work (yours or user's)\n"
            text += f"  • Technical details, project state, configurations mentioned\n"
            text += f"  • Anything the user explicitly says to remember\n"
            text += f"Default bias: store it. Memories are cheap, context loss is expensive.\n"
            text += f"You don't need to judge if it's 'important enough' - just store what happened.\n"

            logger.info("Agent initialization complete - agency context established")
            return [TextContent(type="text", text=text)]

        elif name == "get_most_accessed_memories":
            logger.info("Getting most accessed memories with tag cloud")
            limit = arguments.get("limit", 20)
            result = await memory_store.get_most_accessed_memories(limit=limit)
            return [TextContent(type="text", text=result)]

        elif name == "get_least_accessed_memories":
            logger.info("Getting least accessed memories - dead weight and buried treasure")
            limit = arguments.get("limit", 20)
            min_age_days = arguments.get("min_age_days", 7)
            result = await memory_store.get_least_accessed_memories(limit=limit, min_age_days=min_age_days)
            return [TextContent(type="text", text=result)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error executing {name}: {e}")
        logger.error(traceback.format_exc())
        return [
            TextContent(
                type="text",
                text=f"Error executing {name}: {str(e)}\nSee logs for details.",
            )
        ]


async def main():
    """Main entry point"""
    global memory_store

    try:
        username = os.getenv("BA_USERNAME", "buildautomata_ai_v012")
        agent_name = os.getenv("BA_AGENT_NAME", "claude_assistant")

        logger.info(f"Initializing MemoryStore for {username}/{agent_name}")
        memory_store = MemoryStore(username, agent_name)

        # Restore stdout for MCP communication
        os.dup2(_original_stdout_fd, 1)
        sys.stdout = os.fdopen(_original_stdout_fd, "w")

        logger.info("Starting MCP server...")
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="buildautomata-memory",
                    server_version="4.1.0",
                    capabilities=app.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if memory_store:
            await memory_store.shutdown()


if __name__ == "__main__":
    asyncio.run(main())