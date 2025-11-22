"""
Qdrant vector store for BuildAutomata Memory System
Copyright 2025 Jurden Bruce
"""

import asyncio
import logging
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("buildautomata-memory.qdrant")

# Check availability without importing the heavy library
try:
    import importlib.util
    QDRANT_AVAILABLE = importlib.util.find_spec("qdrant_client") is not None
except Exception:
    QDRANT_AVAILABLE = False

if not QDRANT_AVAILABLE:
    logger.warning("Qdrant not available - vector search disabled")


class QdrantStore:
    """Handles Qdrant vector database operations"""

    def __init__(self, config: Dict[str, Any], collection_name: str, error_log: List[Dict[str, Any]], lazy_load: bool = False):
        """
        Initialize Qdrant store with embedded mode

        Args:
            config: Configuration dict with qdrant_path, vector_size
            collection_name: Name of the Qdrant collection
            error_log: Shared error log list
            lazy_load: If True, delay initialization until first use
        """
        self.config = config
        self.collection_name = collection_name
        self.error_log = error_log
        self.lazy_load = lazy_load
        self.client = None
        self._initialized = False

        if not lazy_load:
            self._init_client()
            self._initialized = True

    def _ensure_initialized(self):
        """Ensure client is initialized (lazy loading support)"""
        if self._initialized or not self.lazy_load:
            return

        import time
        start = time.perf_counter()
        self._init_client()
        self._initialized = True
        logger.info(f"[LAZY] Qdrant loaded on-demand in {(time.perf_counter() - start)*1000:.2f}ms")

    def _init_client(self):
        """
        Initialize Qdrant client

        DEFAULT: Embedded mode (no server required, stores locally)
        OPTIONAL: External server mode via environment variables:
            - USE_EXTERNAL_QDRANT=true
            - QDRANT_URL=http://localhost:6333 (or custom URL)
        """
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant libraries not available")
            return

        # Import only when actually needed (lazy loading)
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        import os

        # Check for external Qdrant configuration
        use_external = os.getenv("USE_EXTERNAL_QDRANT", "").lower() in ("true", "1", "yes")
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        try:
            if use_external:
                # EXTERNAL MODE - connect to Qdrant server
                self.client = QdrantClient(url=qdrant_url)
                logger.info(f"Qdrant external mode: connected to {qdrant_url}")
            else:
                # EMBEDDED MODE (DEFAULT) - works like SQLite, no server process needed
                qdrant_path = self.config.get("qdrant_path", "./qdrant_data")
                self.client = QdrantClient(path=qdrant_path)
                logger.info(f"Qdrant embedded mode (default): initialized at {qdrant_path}")

            # Create collection if it doesn't exist
            collections = self.client.get_collections().collections
            if not any(col.name == self.collection_name for col in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.config["vector_size"], distance=Distance.COSINE
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Qdrant embedded initialization failed: {e}")
            self._log_error("qdrant_init", e)
            self.client = None

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

    async def store_point(self, point_id: str, vector: List[float], payload: Dict[str, Any], max_retries: int = None) -> bool:
        """Store a single point with retry logic"""
        self._ensure_initialized()

        if not self.client:
            return False

        # Import models when needed (after initialization)
        from qdrant_client.models import PointStruct

        if max_retries is None:
            max_retries = self.config.get("qdrant_max_retries", 3)

        for attempt in range(max_retries):
            try:
                point = PointStruct(id=point_id, vector=vector, payload=payload)
                await asyncio.to_thread(
                    self.client.upsert,
                    collection_name=self.collection_name,
                    points=[point],
                )
                logger.debug(f"Stored point {point_id} in Qdrant")
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Qdrant store failed after {max_retries} attempts for {point_id}: {e}")
                    self._log_error("qdrant_store_retry", e)
                    return False
                logger.warning(f"Qdrant store attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return False

    async def store_points_batch(self, points: List[Dict[str, Any]]) -> bool:
        """Store multiple points in a batch

        Args:
            points: List of dicts with 'id', 'vector', 'payload' keys
        """
        self._ensure_initialized()

        if not self.client or not points:
            return False

        # Import models when needed (after initialization)
        from qdrant_client.models import PointStruct

        try:
            qdrant_points = [
                PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                for p in points
            ]

            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.collection_name,
                points=qdrant_points,
            )
            logger.debug(f"Stored {len(points)} points in Qdrant batch")
            return True
        except Exception as e:
            logger.error(f"Qdrant batch store failed: {e}")
            self._log_error("batch_store_qdrant", e)
            return False

    async def search(
        self,
        query_vector: List[float],
        limit: int,
        filter_conditions: Optional[List] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors

        Args:
            query_vector: Query embedding
            limit: Maximum results
            filter_conditions: Optional list of FieldCondition objects for filtering

        Returns:
            List of dicts with 'id' and 'payload' keys
        """
        self._ensure_initialized()

        if not self.client:
            return []

        # Import models when needed (after initialization)
        from qdrant_client.models import Filter

        try:
            search_filter = Filter(must=filter_conditions) if filter_conditions else None

            results = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=search_filter,
                with_payload=True,
            )

            return [
                {"id": result.id, "payload": result.payload}
                for result in results.points
            ]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            self._log_error("vector_search", e)
            return []

    async def delete_points(self, point_ids: List[str]) -> bool:
        """Delete points from the collection"""
        self._ensure_initialized()

        if not self.client or not point_ids:
            return False

        try:
            await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            logger.debug(f"Deleted {len(point_ids)} points from Qdrant")
            return True
        except Exception as e:
            logger.error(f"Qdrant delete failed: {e}")
            self._log_error("qdrant_delete", e)
            return False

    def is_available(self) -> bool:
        """Check if Qdrant client is available"""
        return self.client is not None
