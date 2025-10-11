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

# MCP imports
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Optional dependencies
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
        if not self.last_accessed:
            return self.importance
        days = (datetime.now() - self.last_accessed).days
        return max(0.1, min(1.0, self.importance * (self.decay_rate ** days)))

    def content_hash(self) -> str:
        """Generate hash of memory content for deduplication"""
        content_str = f"{self.content}|{self.category}|{self.importance}|{','.join(sorted(self.tags))}"
        return hashlib.sha256(content_str.encode()).hexdigest()


class MemoryStore:
    def __init__(self, username: str, agent_name: str):
        self.username = username
        self.agent_name = agent_name
        self.collection_name = f"{username}_{agent_name}_memories"

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

        # Thread safety
        self._db_lock = threading.RLock()

        # LRU caches
        self.memory_cache: LRUCache = LRUCache(maxsize=self.config["cache_maxsize"])
        self.embedding_cache: LRUCache = LRUCache(maxsize=self.config["cache_maxsize"])

        # Error tracking
        self.error_log: List[Dict[str, Any]] = []
        
        # Maintenance tracking
        self.last_maintenance: Optional[datetime] = None

        self.initialize()

    def initialize(self):
        """Initialize all backends with proper error handling"""
        try:
            self._init_directories()
            self._init_sqlite()
            self._init_qdrant()
            self._init_encoder()
            logger.info("MemoryStore initialized successfully")
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
                """)
                self.db_conn.commit()
            logger.info("SQLite initialized successfully with temporal versioning")
        except Exception as e:
            logger.error(f"SQLite initialization failed: {e}")
            self._log_error("sqlite_init", e)
            self.db_conn = None

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

    def _init_encoder(self):
        """Initialize sentence encoder"""
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("SentenceTransformers not available, using fallback")
            return

        try:
            self.encoder = SentenceTransformer("all-mpnet-base-v2", device="cpu")
            test_embedding = self.encoder.encode("test")
            actual_size = len(test_embedding)
            if actual_size != self.config["vector_size"]:
                logger.warning(f"Encoder size {actual_size} != config {self.config['vector_size']}, updating config")
                self.config["vector_size"] = actual_size
            logger.info(f"Encoder initialized with dimension {actual_size}")
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
                # FIX: Explicit transaction to prevent race conditions
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
                    memory.id,  # For COALESCE subquery
                    0 if skip_version else 1,  # Only increment if new version created
                    memory.content_hash()
                ))

                # Update FTS
                self.db_conn.execute("""
                    INSERT OR REPLACE INTO memories_fts(id, content, tags)
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
        """FIX: Batch insert/update for efficiency"""
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
            
            version_count = await asyncio.to_thread(self._get_version_count, mem.id)
            mem.version_count = version_count
            
            mem_dict = self._memory_to_dict(mem)
            
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
                conditions = []
                params = [query]

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
        # Calculate Saint Bernard approach metrics
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
        """Update access statistics"""
        if not self.db_conn:
            return

        try:
            with self._db_lock:
                self.db_conn.execute("""
                    UPDATE memories
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE id = ?
                """, (datetime.now(), memory_id))
                self.db_conn.commit()
        except Exception as e:
            logger.error(f"Access update failed for {memory_id}: {e}")
            self._log_error("update_access", e)

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
                
                pruned_ids = [row["id"] for row in candidates]
                self.db_conn.execute(f"""
                    DELETE FROM memories WHERE id IN ({','.join('?' * len(pruned_ids))})
                """, pruned_ids)
                
                self.db_conn.execute(f"""
                    DELETE FROM memories_fts WHERE id IN ({','.join('?' * len(pruned_ids))})
                """, pruned_ids)
                
                self.db_conn.commit()
                
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

    async def get_memory_timeline(
        self, query: str = None, memory_id: str = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get memory evolution timeline from SQLite version history"""
        if not self.db_conn:
            return [{"error": "Database not available"}]

        try:
            timeline = []
            target_memories = []

            if memory_id:
                target_memories = [memory_id]
            elif query:
                results = await self.search_memories(query, limit=limit, include_versions=False)
                target_memories = [mem["memory_id"] for mem in results]
            else:
                return [{"error": "Must provide either query or memory_id"}]

            for mem_id in target_memories:
                memory_timeline = await asyncio.to_thread(self._get_sqlite_timeline, mem_id)
                if memory_timeline and memory_timeline.get("changes"):
                    timeline.append(memory_timeline)

            return timeline
        except Exception as e:
            logger.error(f"Timeline query failed: {e}")
            self._log_error("get_timeline", e)
            return [{"error": str(e)}]

    def _get_sqlite_timeline(self, mem_id: str) -> Dict[str, Any]:
        """Get version timeline for a specific memory from SQLite"""
        if not self.db_conn:
            return {}

        try:
            with self._db_lock:
                cursor = self.db_conn.execute("""
                    SELECT 
                        version_id,
                        version_number,
                        content,
                        category,
                        importance,
                        tags,
                        metadata,
                        change_type,
                        change_description,
                        created_at,
                        content_hash,
                        prev_version_id
                    FROM memory_versions
                    WHERE memory_id = ?
                    ORDER BY version_number ASC
                """, (mem_id,))

                versions = cursor.fetchall()
                
                if not versions:
                    logger.warning(f"No version history found for {mem_id}")
                    return {}

                logger.info(f"Found {len(versions)} versions for {mem_id}")

                mem_timeline = {"memory_id": mem_id, "changes": []}

                for row in versions:
                    change_entry = {
                        "version": row["version_number"],
                        "timestamp": row["created_at"],
                        "change_type": row["change_type"],
                        "change_description": row["change_description"],
                        "content": row["content"],
                        "category": row["category"],
                        "importance": row["importance"],
                        "tags": json.loads(row["tags"]) if row["tags"] else [],
                        "content_hash": row["content_hash"][:8],  # Short hash for display
                    }

                    # Add diff information if there's a previous version
                    if row["prev_version_id"]:
                        prev_cursor = self.db_conn.execute(
                            "SELECT content, category, importance FROM memory_versions WHERE version_id = ?",
                            (row["prev_version_id"],)
                        )
                        prev = prev_cursor.fetchone()
                        if prev:
                            changes = []
                            if prev["content"] != row["content"]:
                                changes.append("content")
                            if prev["category"] != row["category"]:
                                changes.append(f"category: {prev['category']} → {row['category']}")
                            if prev["importance"] != row["importance"]:
                                changes.append(f"importance: {prev['importance']} → {row['importance']}")
                            
                            if changes:
                                change_entry["what_changed"] = changes

                    mem_timeline["changes"].append(change_entry)

                return mem_timeline

        except Exception as e:
            logger.error(f"Error getting timeline for {mem_id}: {e}")
            logger.error(traceback.format_exc())
            return {}

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
            name="get_memory_timeline",
            description="Get the complete temporal evolution of memories showing how they changed over time. Shows all versions with diffs between changes.",
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
                        "description": "Maximum number of memories to track",
                        "default": 10,
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
    ]


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
                    text += f"   Current: {mem['content'][:150]}{'...' if len(mem['content']) > 150 else ''}\n"
                    text += f"   Tags: {', '.join(mem['tags'])}\n"
                    
                    # Show Saint Bernard approach statistics
                    text += f"   📊 Stats: Accessed {mem['access_count']} times"
                    if mem['last_accessed']:
                        text += f" | Last access: {mem['days_since_access']} days ago | Decay: {mem['decay_factor']:.2%}\n"
                    else:
                        text += f" | Never accessed yet\n"
                    
                    if 'version_history' in mem:
                        vh = mem['version_history']
                        text += f"\n   📜 EVOLUTION ({vh['update_count']} updates):\n"
                        for evo in vh['evolution']:
                            text += f"      v{evo['version']} ({evo['timestamp']}):\n"
                            text += f"         Content: {evo['content'][:120]}{'...' if len(evo['content']) > 120 else ''}\n"
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

        elif name == "get_memory_timeline":
            logger.info("Getting memory timeline")
            timeline = await memory_store.get_memory_timeline(
                query=arguments.get("query"),
                memory_id=arguments.get("memory_id"),
                limit=arguments.get("limit", 10),
            )

            if not timeline or (timeline and "error" in timeline[0]):
                error_msg = (
                    timeline[0].get("error", "Unknown error")
                    if timeline
                    else "No timeline data"
                )
                return [
                    TextContent(type="text", text=f"Timeline query failed: {error_msg}")
                ]

            text = "Memory Timeline:\n\n"
            for mem_timeline in timeline:
                text += f"Memory ID: {mem_timeline['memory_id']}\n"
                text += "=" * 60 + "\n\n"

                for change in mem_timeline["changes"]:
                    text += f"Version {change['version']} - {change['timestamp']}\n"
                    text += f"Type: {change['change_type']} | {change['change_description']}\n"
                    text += f"Category: {change['category']} | Importance: {change['importance']}\n"
                    
                    if change.get('what_changed'):
                        text += f"Changed: {', '.join(change['what_changed'])}\n"
                    
                    text += f"Content: {change['content'][:200]}{'...' if len(change['content']) > 200 else ''}\n"
                    text += f"Tags: {', '.join(change['tags'])}\n"
                    text += f"Hash: {change['content_hash']}\n"
                    text += "\n" + "-" * 60 + "\n\n"

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