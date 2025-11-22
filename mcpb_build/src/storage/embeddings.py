"""
Embedding generation for BuildAutomata Memory System
Copyright 2025 Jurden Bruce
"""

import hashlib
import logging
import traceback
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger("buildautomata-memory.embeddings")

# Check availability without importing the heavy library
try:
    import importlib.util
    EMBEDDINGS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
except Exception:
    EMBEDDINGS_AVAILABLE = False

if not EMBEDDINGS_AVAILABLE:
    logger.warning("SentenceTransformers not available - using fallback embeddings")


class EmbeddingGenerator:
    """Handles text embedding generation with caching and fallback"""

    def __init__(self, config: Dict[str, Any], embedding_cache, error_log: List[Dict[str, Any]], lazy_load: bool = False):
        """
        Initialize embedding generator

        Args:
            config: Configuration dict with 'vector_size' key
            embedding_cache: LRUCache for caching embeddings
            error_log: Shared error log list
            lazy_load: If True, delay encoder initialization until first use
        """
        self.config = config
        self.embedding_cache = embedding_cache
        self.error_log = error_log
        self.lazy_load = lazy_load
        self.encoder = None
        self._encoder_initialized = False

        if not lazy_load:
            self._init_encoder()
            self._encoder_initialized = True

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
            # Import only when actually needed (lazy loading)
            from sentence_transformers import SentenceTransformer

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

    def is_available(self) -> bool:
        """Check if real embeddings (not fallback) are available"""
        return self.encoder is not None
