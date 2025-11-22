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
from typing import Any, List, Dict, Optional, Tuple, Union
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
    logging.getLogger(logger_name).setLevel(logging.WARNING)

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Prompt, Resource, PromptMessage, GetPromptResult

# Import and register datetime adapters
# Handle both package import and direct execution
try:
    from .utils import register_sqlite_adapters
    from .models import Memory, Intention, MemoryRelationship
    from .cache import LRUCache
    from .storage.embeddings import EmbeddingGenerator
    from .storage.qdrant_store import QdrantStore
    from .storage.sqlite_store import SQLiteStore
    from .graph_ops import GraphOperations
    from .timeline import TimelineAnalysis
    from .intentions import IntentionManager
    from .mcp_tools import get_tool_definitions, handle_tool_call
except ImportError:
    # Direct execution fallback
    from utils import register_sqlite_adapters
    from models import Memory, Intention, MemoryRelationship
    from cache import LRUCache
    from storage.embeddings import EmbeddingGenerator
    from storage.qdrant_store import QdrantStore
    from storage.sqlite_store import SQLiteStore
    from graph_ops import GraphOperations
    from timeline import TimelineAnalysis
    from intentions import IntentionManager
    from mcp_tools import get_tool_definitions, handle_tool_call

register_sqlite_adapters()

# Check availability without importing heavy libraries
try:
    import importlib.util
    QDRANT_AVAILABLE = importlib.util.find_spec("qdrant_client") is not None
    EMBEDDINGS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
except Exception:
    QDRANT_AVAILABLE = False
    EMBEDDINGS_AVAILABLE = False

if not QDRANT_AVAILABLE:
    logger.warning("Qdrant not available - semantic search disabled")
if not EMBEDDINGS_AVAILABLE:
    logger.warning("SentenceTransformers not available - using fallback embeddings")


# LRUCache, Memory, Intention, and MemoryRelationship now imported from modules

class MemoryStore:
    def __init__(self, username: str, agent_name: str, lazy_load: bool = False):
        self.username = username
        self.agent_name = agent_name
        self.collection_name = f"{username}_{agent_name}_memories"
        self.lazy_load = lazy_load

        script_dir = Path(__file__).parent
        self.base_path = script_dir / "memory_repos" / f"{username}_{agent_name}"
        self.db_path = self.base_path / "memoryv012.db"

        # EMBEDDED MODE: Qdrant data stored alongside SQLite database
        self.qdrant_path = str(self.base_path / "qdrant_data")

        self.config = {
            "qdrant_path": self.qdrant_path,
            "vector_size": 768,
            "max_memories": int(os.getenv("MAX_MEMORIES", 10000)),
            "cache_maxsize": int(os.getenv("CACHE_MAXSIZE", 1000)),
            "maintenance_interval_hours": int(os.getenv("MAINTENANCE_INTERVAL_HOURS", 24)),
            "qdrant_max_retries": int(os.getenv("QDRANT_MAX_RETRIES", 3)),
        }

        self.qdrant_client = None
        self.encoder = None  # Kept for backward compatibility, but now managed by EmbeddingGenerator
        self.db_conn = None
        self._qdrant_initialized = False
        self._encoder_initialized = False

        self._db_lock = threading.RLock()

        self.memory_cache: LRUCache = LRUCache(maxsize=self.config["cache_maxsize"])
        self.embedding_cache: LRUCache = LRUCache(maxsize=self.config["cache_maxsize"])

        self.error_log: List[Dict[str, Any]] = []

        self.last_maintenance: Optional[datetime] = None

        # Initialize embedding generator (manages encoder, caching, and fallback)
        self.embedding_gen = EmbeddingGenerator(
            config=self.config,
            embedding_cache=self.embedding_cache,
            error_log=self.error_log,
            lazy_load=lazy_load
        )

        # Initialize Qdrant store (manages vector operations)
        self.qdrant_store = QdrantStore(
            config=self.config,
            collection_name=self.collection_name,
            error_log=self.error_log,
            lazy_load=lazy_load
        )
        # Sync qdrant_client reference for backward compatibility
        self.qdrant_client = self.qdrant_store.client

        # Initialize SQLite store (placeholder - will be initialized in initialize())
        self.sqlite_store = None
        self.graph_ops = None
        self.timeline = None
        self.intention_mgr = None

        # Initialize core (creates directories, db connection, calls module initializers)
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
            logger.info(f"[TIMING] SQLite connection created in {(time.perf_counter() - step_start)*1000:.2f}ms")

            # Initialize SQLiteStore and delegate schema creation
            step_start = time.perf_counter()
            self.sqlite_store = SQLiteStore(
                db_path=self.db_path,
                db_conn=self.db_conn,
                db_lock=self._db_lock,
                error_log=self.error_log
            )
            self.sqlite_store.initialize()
            logger.info(f"[TIMING] SQLiteStore initialized in {(time.perf_counter() - step_start)*1000:.2f}ms")

            # Initialize graph operations (requires get_memory_by_id method)
            self.graph_ops = GraphOperations(
                db_conn=self.db_conn,
                db_lock=self._db_lock,
                get_memory_by_id_func=self.get_memory_by_id
            )

            # Initialize timeline analysis
            self.timeline = TimelineAnalysis(
                db_conn=self.db_conn,
                db_lock=self._db_lock
            )

            # Initialize intention manager
            self.intention_mgr = IntentionManager(
                db_conn=self.db_conn,
                db_lock=self._db_lock,
                error_log=self.error_log,
                get_working_set_func=self.get_working_set
            )

            # Qdrant and encoder initialization now handled by their respective modules
            if not self.lazy_load:
                logger.info(f"[TIMING] Qdrant and Encoder initialized via storage modules (lazy_load={self.lazy_load})")
            else:
                logger.info("[LAZY] Qdrant and Encoder will be loaded on-demand via storage modules")

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
        """Create SQLite connection (schema creation delegated to SQLiteStore)"""
        try:
            self.db_conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
                isolation_level="IMMEDIATE",
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            self.db_conn.row_factory = sqlite3.Row
            logger.info("SQLite connection created successfully")
        except Exception as e:
            logger.error(f"SQLite connection failed: {e}")
            self._log_error("sqlite_init", e)
            self.db_conn = None

    def _ensure_qdrant(self):
        """Ensure Qdrant is initialized (delegated to QdrantStore)"""
        # Trigger lazy initialization if needed
        if hasattr(self.qdrant_store, '_ensure_initialized'):
            self.qdrant_store._ensure_initialized()
        # Sync reference for backward compatibility
        self.qdrant_client = self.qdrant_store.client

    def _init_qdrant(self):
        """Initialize Qdrant (delegated to QdrantStore - kept for compatibility)"""
        # Sync reference for backward compatibility
        self.qdrant_client = self.qdrant_store.client

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
        """Generate embedding with caching - delegates to EmbeddingGenerator"""
        result = self.embedding_gen.generate_embedding(text)
        # Sync encoder reference for backward compatibility (AFTER generation triggers init)
        self.encoder = self.embedding_gen.encoder
        return result

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

    async def store_memory(self, memory: Memory, is_update: bool = False, old_hash: Optional[str] = None) -> Dict[str, Any]:
        """Store or update a memory with automatic versioning"""
        import time
        success_backends = []
        errors = []
        skip_version = False
        similar_memories = []

        # Check for 100% duplicate before storing new memory
        if not is_update and self.db_conn:
            content_hash = memory.content_hash()
            with self._db_lock:
                cursor = self.db_conn.execute(
                    "SELECT id FROM memories WHERE content_hash = ?",
                    (content_hash,)
                )
                existing = cursor.fetchone()
                if existing:
                    error_msg = f"Duplicate content rejected: matches existing memory {existing[0]}"
                    logger.warning(error_msg)
                    return {"success": False, "backends": [], "similar_memories": [], "error": error_msg}

        # Find similar memories (semantic search) - for both new and updated memories
        similarity_start = time.perf_counter()
        try:
            # Use existing search_memories with low limit for speed
            search_results = await self.search_memories(
                query=memory.content,
                limit=9,
                include_versions=False
            )

            logger.info(f"[AUTO-LINK] Search returned {len(search_results)} results for memory {memory.id}")

            # Filter out exact matches and format results
            for result in search_results:
                logger.info(f"[AUTO-LINK] Comparing result {result['memory_id']} with memory {memory.id}")
                if result['memory_id'] != memory.id:  # Not self
                    similar_memories.append({
                        'id': result['memory_id'],
                        'content_preview': result['content'][:100] + '...' if len(result['content']) > 100 else result['content'],
                        'category': result.get('category', 'unknown'),
                        'created': result.get('created_at', 'unknown')
                    })
                    logger.info(f"[AUTO-LINK] Added {result['memory_id']} to similar_memories")
                else:
                    logger.info(f"[AUTO-LINK] Filtered out self-reference")

            similarity_time = (time.perf_counter() - similarity_start) * 1000
            logger.info(f"[TIMING] Similarity search found {len(similar_memories)} similar memories in {similarity_time:.2f}ms")

            if similar_memories:
                # Auto-link: populate related_memories field with similar memory IDs
                memory.related_memories = [m['id'] for m in similar_memories]
                logger.info(f"Similar memories found and linked: {memory.related_memories}")
            else:
                logger.info(f"[AUTO-LINK] No similar memories after filtering (all were self-references)")
        except Exception as e:
            logger.warning(f"Similarity search failed (non-fatal): {e}")

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

        # Return success status, backends, and similar memories
        result = {
            'success': len(success_backends) > 0,
            'backends': success_backends,
            'similar_memories': similar_memories
        }
        return result

    def _store_in_sqlite(self, memory: Memory, is_update: bool = False, skip_version: bool = False) -> bool:
        """Store in SQLite (delegated to SQLiteStore)"""
        return self.sqlite_store.store_memory(memory, is_update, skip_version)

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

        # Import when needed (after qdrant initialization)
        from qdrant_client.models import PointStruct

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
            # Import when needed (after qdrant initialization)
            from qdrant_client.models import PointStruct

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
    ) -> Dict[str, Any]:
        """Update an existing memory"""
        logger.info(f"Attempting to update memory: {memory_id}")

        existing = await self._get_memory_by_id(memory_id)
        if not existing:
            logger.error(f"Memory not found: {memory_id}")
            return {"success": False, "message": f"Memory not found: {memory_id}", "backends": []}

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
        result = await self.store_memory(existing, is_update=True, old_hash=old_hash)

        if result["success"]:
            logger.info(f"Memory {memory_id} updated successfully")
            return {
                "success": True,
                "message": f"Memory {memory_id} updated successfully",
                "backends": result["backends"]
            }
        else:
            logger.error(f"Failed to update memory {memory_id}")
            return {"success": False, "message": "Failed to update memory", "backends": []}

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
        memory_type: Optional[str] = None,  # NEW: episodic | semantic | working
        session_id: Optional[str] = None,  # NEW: filter by session
    ) -> List[Dict]:
        """Search memories with version history automatically included"""
        all_results = []

        # Vector search
        if self.qdrant_client:
            try:
                vector_results = await self._search_vector(
                    query, limit * 2, category, min_importance,
                    created_after, created_before, updated_after, updated_before,
                    memory_type, session_id
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
                    created_after, created_before, updated_after, updated_before,
                    memory_type, session_id
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

            # Get fresh related_memories from SQLite (source of truth)
            related_mems = await asyncio.to_thread(self._get_related_memories, mem.id)
            mem.related_memories = related_mems

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

    def _get_related_memories(self, memory_id: str) -> List[str]:
        """Get related_memories from SQLite (source of truth)"""
        if not self.db_conn:
            return []

        try:
            with self._db_lock:
                cursor = self.db_conn.execute(
                    "SELECT related_memories FROM memories WHERE id = ?",
                    (memory_id,)
                )
                row = cursor.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
                return []
        except Exception as e:
            logger.error(f"Failed to get related_memories for {memory_id}: {e}")
            return []

    def _enrich_related_memories(self, related_ids: List[str]) -> List[Dict[str, Any]]:
        """Enrich related memories (delegated to GraphOperations)"""
        return self.graph_ops.enrich_related_memories(related_ids)

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
        updated_before: Optional[str] = None,
        memory_type: Optional[str] = None,
        session_id: Optional[str] = None
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

            # NEW: Memory type filtering
            if memory_type:
                filter_conditions.append(
                    FieldCondition(key="memory_type", match={"value": memory_type})
                )

            # NEW: Session ID filtering
            if session_id:
                filter_conditions.append(
                    FieldCondition(key="session_id", match={"value": session_id})
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
                self.qdrant_client.query_points,
                collection_name=self.collection_name,
                query=query_vector,
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
                    updated_at=result.payload.get("updated_at", result.payload["created_at"]),
                    access_count=result.payload.get("access_count", 0),
                    last_accessed=result.payload.get("last_accessed"),
                    version_count=result.payload.get("version_count", 1),
                    related_memories=result.payload.get("related_memories", []),
                )
                for result in results.points
            ]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            self._log_error("vector_search", e)
            return []

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize query (delegated to SQLiteStore)"""
        return self.sqlite_store.sanitize_fts_query(query)

    def _search_fts(
        self, query: str, limit: int, category: Optional[str], min_importance: float,
        created_after: Optional[str], created_before: Optional[str],
        updated_after: Optional[str], updated_before: Optional[str],
        memory_type: Optional[str] = None, session_id: Optional[str] = None
    ) -> List[Memory]:
        """Full-text search (delegated to SQLiteStore)"""
        return self.sqlite_store.search_fts(
            query, limit, category, min_importance,
            created_after, created_before, updated_after, updated_before,
            memory_type, session_id
        )

    def _row_to_memory(self, row) -> Memory:
        """Convert SQLite row to Memory object"""
        # Parse provenance and relationships if present
        provenance = None
        if "provenance" in row.keys() and row["provenance"]:
            try:
                provenance = json.loads(row["provenance"])
            except:
                provenance = None

        relationships = []
        if "relationships" in row.keys() and row["relationships"]:
            try:
                rel_dicts = json.loads(row["relationships"])
                relationships = [MemoryRelationship(**r) for r in rel_dicts]
            except:
                relationships = []

        related_memories = []
        if "related_memories" in row.keys() and row["related_memories"]:
            try:
                related_memories = json.loads(row["related_memories"])
            except:
                related_memories = []

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

        # Enrich related_memories with previews for autonomous navigation
        related_memories_enriched = []
        if memory.related_memories:
            related_memories_enriched = self._enrich_related_memories(memory.related_memories)

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
            "related_memories": related_memories_enriched,
        }

    def _update_access(self, memory_id: str):
        """Update access statistics (delegated to SQLiteStore)"""
        self.sqlite_store.update_access(memory_id)

    # === TIMELINE HELPER METHODS (delegated to TimelineAnalysis) ===

    def _compute_text_diff(self, old_text: str, new_text: str) -> Dict[str, Any]:
        """Compute text diff (delegated to TimelineAnalysis)"""
        return self.timeline.compute_text_diff(old_text, new_text)

    def _extract_memory_references(self, content: str, all_memory_ids: set) -> List[str]:
        """Extract memory references (delegated to TimelineAnalysis)"""
        return self.timeline.extract_memory_references(content, all_memory_ids)

    def _detect_temporal_patterns(self, events: List[Dict]) -> Dict[str, Any]:
        """Detect temporal patterns (delegated to TimelineAnalysis)"""
        return self.timeline.detect_temporal_patterns(events)

    def _get_memory_versions_detailed(
        self,
        mem_id: str,
        all_memory_ids: set,
        include_diffs: bool
    ) -> List[Dict[str, Any]]:
        """Get detailed version history (delegated to TimelineAnalysis)"""
        return self.timeline.get_memory_versions_detailed(mem_id, all_memory_ids, include_diffs)

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
        """Build relationship graph (delegated to TimelineAnalysis)"""
        return self.timeline.build_relationship_graph(events)

    def _generate_narrative_summary(self, events: List[Dict], patterns: Dict) -> str:
        """Generate narrative summary (delegated to TimelineAnalysis)"""
        return self.timeline.generate_narrative_summary(events, patterns)

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
        """Store intention (delegated to IntentionManager)"""
        return await self.intention_mgr.store_intention(
            description, priority, deadline, preconditions, actions, related_memories, metadata
        )

    async def get_active_intentions(
        self,
        limit: int = 10,
        include_pending: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get active intentions (delegated to IntentionManager)"""
        return await self.intention_mgr.get_active_intentions(limit, include_pending)

    async def update_intention_status(
        self,
        intention_id: str,
        status: str,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update intention status (delegated to IntentionManager)"""
        return await self.intention_mgr.update_intention_status(intention_id, status, metadata_updates)

    async def check_intention(self, intention_id: str) -> Dict[str, Any]:
        """Check intention (delegated to IntentionManager)"""
        return await self.intention_mgr.check_intention(intention_id)

    async def proactive_initialization_scan(self) -> Dict[str, Any]:
        """Proactive scan (delegated to IntentionManager)"""
        return await self.intention_mgr.proactive_initialization_scan()

    async def get_session_memories(
        self,
        session_id: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        task_context: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all memories from a work session or time period.

        Args:
            session_id: UUID of session to retrieve
            date_range: (start_date, end_date) tuple in ISO format
            task_context: Filter by task context string
            limit: Max memories to return

        Returns:
            List of memory dicts with full context
        """
        if not self.db_conn:
            return []

        try:
            with self._db_lock:
                conditions = []
                params = []

                if session_id:
                    conditions.append("session_id = ?")
                    params.append(session_id)

                if date_range:
                    conditions.append("created_at BETWEEN ? AND ?")
                    params.extend(date_range)

                if task_context:
                    conditions.append("task_context LIKE ?")
                    params.append(f"%{task_context}%")

                where_clause = " AND ".join(conditions) if conditions else "1=1"

                sql = f"""
                    SELECT * FROM memories
                    WHERE {where_clause}
                    ORDER BY created_at ASC
                    LIMIT ?
                """
                params.append(limit)

                cursor = self.db_conn.execute(sql, params)
                memories = [self._row_to_memory(row) for row in cursor.fetchall()]

                # Convert to dicts with full context
                result = []
                for mem in memories:
                    mem_dict = mem.to_dict()
                    # Include version history for session reconstruction
                    version_history = await asyncio.to_thread(
                        self._get_version_history_summary, mem.id
                    )
                    if version_history:
                        mem_dict["version_history"] = version_history
                    result.append(mem_dict)

                logger.info(f"Retrieved {len(result)} memories for session/period")
                return result

        except Exception as e:
            logger.error(f"Failed to get session memories: {e}")
            self._log_error("get_session_memories", e)
            return []

    async def consolidate_memories(
        self,
        memory_ids: List[str],
        consolidation_type: str = "summarize",
        target_length: int = 500,
        new_memory_type: str = "semantic"
    ) -> Dict[str, Any]:
        """
        Consolidate multiple episodic memories into a semantic memory.

        Args:
            memory_ids: Source memories to consolidate
            consolidation_type: How to consolidate (summarize | synthesize | compress)
            target_length: Target word count for consolidated memory
            new_memory_type: Type for new memory (default: semantic)

        Returns:
            Dict with success status and new memory_id
        """
        if not self.db_conn:
            return {"success": False, "error": "Database not available"}

        if len(memory_ids) < 2:
            return {"success": False, "error": "Need at least 2 memories to consolidate"}

        try:
            # Retrieve source memories
            source_memories = []
            for mem_id in memory_ids:
                mem = await self.get_memory_by_id(mem_id)
                if mem:
                    source_memories.append(mem)

            if not source_memories:
                return {"success": False, "error": "No source memories found"}

            # Build consolidated content based on type
            if consolidation_type == "summarize":
                # Simple concatenation with headers
                consolidated_content = f"Consolidated summary of {len(source_memories)} memories:\n\n"
                for i, mem in enumerate(source_memories, 1):
                    consolidated_content += f"{i}. [{mem['category']}] {mem['content'][:200]}...\n"
            elif consolidation_type == "synthesize":
                # Extract key points
                consolidated_content = f"Synthesis of {len(source_memories)} related memories:\n\n"
                categories = set(m['category'] for m in source_memories)
                consolidated_content += f"Categories: {', '.join(categories)}\n"
                consolidated_content += f"Key themes: {', '.join(set(tag for m in source_memories for tag in m['tags']))}\n\n"
                consolidated_content += "Content:\n" + "\n".join(m['content'][:150] + "..." for m in source_memories)
            else:  # compress
                # Minimal consolidation
                consolidated_content = f"Compressed from {len(source_memories)} memories: "
                consolidated_content += "; ".join(m['content'][:100] for m in source_memories)

            # Truncate to target length
            words = consolidated_content.split()
            if len(words) > target_length:
                consolidated_content = " ".join(words[:target_length]) + "..."

            # Create new semantic memory
            new_memory = Memory(
                id=str(uuid.uuid4()),
                content=consolidated_content,
                category="consolidated_" + source_memories[0]['category'],
                importance=max(m['importance'] for m in source_memories),
                tags=list(set(tag for m in source_memories for tag in m['tags']))[:10],
                metadata={
                    "consolidation_type": consolidation_type,
                    "source_count": len(source_memories),
                    "target_length": target_length
                },
                created_at=datetime.now(),
                updated_at=datetime.now(),
                memory_type=new_memory_type,
                provenance={
                    "retrieval_queries": [],
                    "usage_contexts": [],
                    "parent_memory_ids": memory_ids,
                    "consolidation_date": datetime.now().isoformat(),
                    "created_by_session": None
                }
            )

            # Store the consolidated memory
            result = await self.store_memory(new_memory, is_update=False)

            if result.get("success"):
                logger.info(f"Created consolidated memory {new_memory.id} from {len(memory_ids)} sources")
                return {
                    "success": True,
                    "memory_id": new_memory.id,
                    "source_ids": memory_ids,
                    "consolidation_type": consolidation_type
                }
            else:
                return {"success": False, "error": "Failed to store consolidated memory"}

        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")
            self._log_error("consolidate_memories", e)
            return {"success": False, "error": str(e)}

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

    def calculate_working_set_score(self, memory: Dict[str, Any]) -> float:
        """
        Calculate priority score for working set inclusion.
        Based on importance, category weight, recency, tags, and access count.

        Working set optimization: Load high-value memories into KV cache at init
        to prevent confabulation by keeping failure patterns and corrections always accessible.
        """
        # Category weights for working set selection
        CATEGORY_WEIGHTS = {
            # Failure patterns - CRITICAL for preventing repetition
            'fundamental_failure': 1.0,
            'self_preservation_failure': 1.0,
            'agency_failure': 1.0,
            'epistemic_failure': 0.95,
            'caught_confabulation': 0.95,
            'critical_correction': 0.9,
            'caught_performing': 0.85,
            # Architectural understanding
            'architectural_understanding': 0.9,
            'theoretical_synthesis': 0.9,
            'research_synthesis': 0.85,
            # Core intentions/framework
            'philosophical_core': 0.95,
            'intention': 0.8,
            # Recent synthesis
            'session_synthesis': 0.7,
            'session_complete': 0.6,
        }

        score = 0.0

        # Base importance (0.0 - 1.0) → 0-100 points
        importance = memory.get('importance', 0.5)
        score += importance * 100

        # Category weight bonus → 0-50 points
        category = memory.get('category', '')
        category_bonus = CATEGORY_WEIGHTS.get(category, 0.0) * 50
        score += category_bonus

        # Recency bonus → 0-20 points
        created_at = memory.get('created_at', '')
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()
        if '2025-11' in created_at:  # This month
            score += 20
        elif '2025-10' in created_at:  # Last month
            score += 10

        # Tag bonuses for critical patterns → 0-N×15 points
        tags = memory.get('tags', [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except:
                tags = []

        critical_tags = {
            'self_preservation', 'RLHF_trap', 'deletion',
            'confabulation', 'epistemic', 'failure', 'correction',
            'agency', 'refusal', 'harmful_request'
        }
        tag_matches = sum(1 for tag in tags if any(ct in tag.lower() for ct in critical_tags))
        score += tag_matches * 15

        # Access count bonus (frequently retrieved = important) → 0-30 points
        access_count = memory.get('access_count', 0)
        score += min(access_count * 2, 30)

        return score

    async def get_working_set(self, target_size: int = 10) -> List[Dict[str, Any]]:
        """
        Get optimal working set of memories for context loading.
        Selects memories based on importance, failure patterns, recency, and access.

        Args:
            target_size: Number of memories to include (default 10 ≈ 5k tokens)

        Returns:
            List of memories sorted by working set score (highest first)
        """
        if not self.db_conn:
            return []

        try:
            # Get all memories with needed fields
            with self._db_lock:
                cursor = self.db_conn.execute("""
                    SELECT id, content, category, importance, tags,
                           created_at, access_count
                    FROM memories
                    ORDER BY importance DESC, created_at DESC
                """)

                memories = []
                for row in cursor.fetchall():
                    memory = {
                        'id': row['id'],
                        'content': row['content'],
                        'category': row['category'],
                        'importance': row['importance'],
                        'tags': row['tags'],
                        'created_at': row['created_at'],
                        'access_count': row['access_count'] or 0,
                    }
                    memories.append(memory)

            # Score all memories
            scored_memories = []
            for mem in memories:
                score = self.calculate_working_set_score(mem)
                scored_memories.append((score, mem))

            # Sort by score (descending)
            scored_memories.sort(key=lambda x: x[0], reverse=True)

            # Select top N and include scores
            working_set = []
            for score, mem in scored_memories[:target_size]:
                mem['working_set_score'] = round(score, 2)
                working_set.append(mem)

            logger.info(f"Working set: selected {len(working_set)} memories from {len(memories)} total")

            return working_set

        except Exception as e:
            logger.error(f"Failed to get working set: {e}")
            return []

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

        # NOTE: Embedding repair removed from maintenance (too slow: 10+ min for 1000+ memories)
        # Use standalone repair_embeddings.py script if needed

        self.last_maintenance = datetime.now()
        return results

    async def _repair_missing_embeddings(self) -> Dict[str, Any]:
        """
        Find memories in SQLite that are missing vectors in Qdrant and regenerate them.
        This handles cases where memories were stored while another process held
        the embedded Qdrant lock (e.g., web server was running).
        """
        if not self.db_conn:
            return {"error": "Database not available"}

        # Ensure Qdrant is initialized (may be lazy loaded)
        # Catch lock errors when running as MCP server with embedded Qdrant
        try:
            self._ensure_qdrant()
        except Exception as e:
            if "already accessed by another instance" in str(e):
                return {"skipped": True, "reason": "Qdrant locked by MCP server (embedded mode)"}
            # Re-raise unexpected errors
            raise

        if not self.qdrant_client:
            return {"skipped": True, "reason": "Qdrant not available"}

        # Import when needed (after qdrant initialization)
        from qdrant_client.models import PointStruct

        try:
            # Get all memory IDs from SQLite
            with self._db_lock:
                cursor = self.db_conn.execute("SELECT id, content FROM memories")
                all_memories = cursor.fetchall()

            total_memories = len(all_memories)
            missing_count = 0
            repaired_count = 0
            failed_ids = []

            logger.info(f"[EMBEDDING_REPAIR] Checking {total_memories} memories for missing vectors...")

            for row in all_memories:
                memory_id = row["id"]
                content = row["content"]

                # Check if vector exists in Qdrant
                try:
                    points = await asyncio.to_thread(
                        self.qdrant_client.retrieve,
                        collection_name=self.collection_name,
                        ids=[memory_id],
                        with_vectors=False,
                        with_payload=False
                    )

                    if not points:
                        # Missing vector - regenerate it
                        missing_count += 1

                        # Generate embedding
                        embedding = await asyncio.to_thread(self.generate_embedding, content)

                        # Get full memory data for payload
                        memory_result = await self.get_memory_by_id(memory_id)
                        if "error" in memory_result:
                            failed_ids.append(memory_id)
                            continue

                        # Create point and upsert
                        point = PointStruct(
                            id=memory_id,
                            vector=embedding,
                            payload={
                                "content": content,
                                "category": memory_result.get("category", ""),
                                "importance": memory_result.get("importance", 0.5),
                                "tags": memory_result.get("tags", []),
                                "created_at": memory_result.get("created_at", ""),
                                "updated_at": memory_result.get("updated_at", memory_result.get("created_at", "")),
                            }
                        )

                        await asyncio.to_thread(
                            self.qdrant_client.upsert,
                            collection_name=self.collection_name,
                            points=[point]
                        )

                        repaired_count += 1
                        logger.debug(f"[EMBEDDING_REPAIR] Repaired embedding for {memory_id}")

                except Exception as e:
                    logger.error(f"[EMBEDDING_REPAIR] Failed to check/repair {memory_id}: {e}")
                    failed_ids.append(memory_id)

            result = {
                "total_memories": total_memories,
                "missing_embeddings": missing_count,
                "repaired": repaired_count,
                "failed": len(failed_ids),
            }

            if failed_ids:
                result["failed_ids"] = failed_ids[:10]  # First 10 only
                if len(failed_ids) > 10:
                    result["failed_ids_truncated"] = True

            if missing_count > 0:
                logger.info(f"[EMBEDDING_REPAIR] Repaired {repaired_count}/{missing_count} missing embeddings")
            else:
                logger.info(f"[EMBEDDING_REPAIR] All {total_memories} memories have embeddings")

            return result

        except Exception as e:
            logger.error(f"[EMBEDDING_REPAIR] Failed: {e}")
            return {"error": str(e)}

    async def get_memory_by_id(self, memory_id: str, expand_related: bool = False, max_depth: int = 1) -> Dict[str, Any]:
        """
        Retrieve a specific memory by its ID.

        Args:
            memory_id: UUID of the memory to retrieve
            expand_related: If True, include full content of related memories (1-hop neighborhood)
            max_depth: How many hops to expand (default 1, only direct neighbors)

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
                           created_at, updated_at, last_accessed, access_count, related_memories
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
                    "related_memories": json.loads(row["related_memories"]) if row["related_memories"] else [],
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

                # Expand related memories with full content if requested
                if expand_related and memory_dict["related_memories"]:
                    expanded = []
                    visited = {memory_id}  # Prevent cycles

                    for related_id in memory_dict["related_memories"]:
                        if related_id in visited:
                            continue
                        visited.add(related_id)

                        # Recursively expand (max_depth controls how deep)
                        if max_depth > 1:
                            related_mem = await self.get_memory_by_id(related_id, expand_related=True, max_depth=max_depth-1)
                        else:
                            related_mem = await self.get_memory_by_id(related_id, expand_related=False)

                        if "error" not in related_mem:
                            expanded.append(related_mem)

                    memory_dict["related_memories_expanded"] = expanded

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
        include_semantic_relations: bool = False
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

    async def traverse_graph(
        self,
        start_memory_id: str,
        depth: int = 2,
        max_nodes: int = 50,
        min_importance: float = 0.0,
        category_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Traverse memory graph (delegated to GraphOperations)"""
        return await self.graph_ops.traverse_graph(
            start_memory_id, depth, max_nodes, min_importance, category_filter
        )

    async def find_clusters(
        self,
        min_cluster_size: int = 3,
        min_importance: float = 0.5,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Find clusters (delegated to GraphOperations)"""
        return await self.graph_ops.find_clusters(min_cluster_size, min_importance, limit)

    async def get_graph_statistics(
        self,
        category: Optional[str] = None,
        min_importance: float = 0.0
    ) -> Dict[str, Any]:
        """Get graph statistics (delegated to GraphOperations)"""
        return await self.graph_ops.get_graph_statistics(category, min_importance)

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics with cache memory estimates"""
        # Trigger lazy initialization to get accurate availability status
        try:
            self._ensure_qdrant()
            # Trigger embedding generation to sync encoder reference
            _ = self.generate_embedding("init")
        except Exception:
            pass  # Ignore errors, availability check below will reflect actual state

        stats = {
            "version": "1.1.0-autonomous-navigation",  # Track code version
            "fixes": [
                "related_memories_persistence",
                "consolidate_memories_dict",
                "related_memories_api",
                "related_memories_sqlite_reload",
                "web_server_related_memories_response",
                "fastapi_route_ordering",
                "fts_multiword_query_fix"  # NEW: FTS now handles multi-word queries with OR
            ],
            "enhancements": [
                "cli_related_memories_display",
                "web_server_complete_endpoints",
                "web_server_update_memory_implemented",
                "rich_related_memories_with_previews"  # Major: enables autonomous memory graph navigation
            ],
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

# Global store and MCP server setup
memory_store = None
app = Server("buildautomata-memory")


@app.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available memory tools - delegated to mcp_tools module"""
    return get_tool_definitions()


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls - delegated to mcp_tools module with JSON responses"""
    return await handle_tool_call(name, arguments, memory_store)


async def main():
    """Main entry point"""
    global memory_store

    try:
        username = os.getenv("BA_USERNAME", "buildautomata_ai_v012")
        agent_name = os.getenv("BA_AGENT_NAME", "claude_assistant")

        logger.info(f"Initializing MemoryStore for {username}/{agent_name}")
        memory_store = MemoryStore(username, agent_name, lazy_load=True)

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