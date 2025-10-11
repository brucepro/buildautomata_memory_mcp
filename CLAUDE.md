# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BuildAutomata Memory MCP is a Model Context Protocol (MCP) server that provides persistent memory storage for AI agents. It implements a sophisticated memory system with temporal versioning, semantic search via vector embeddings, and SQLite-based persistence.

**Key Features:**
- Temporal versioning with complete change history
- Semantic search using Qdrant vector database and sentence transformers
- LRU caching for embeddings and memory lookups
- SQLite backend with full-text search (FTS5)
- Automatic memory pruning and maintenance
- Thread-safe operations with proper locking

## Architecture

### Core Components

**MemoryStore** (buildautomata_memory_mcp.py:136)
- Main class managing all memory operations
- Initializes and coordinates SQLite, Qdrant, and embedding backends
- Handles thread safety with `threading.RLock`
- Maintains LRU caches for memories and embeddings

**Memory** (buildautomata_memory_mcp.py:91)
- Dataclass representing a memory item
- Fields: id, content, category, importance, tags, metadata, timestamps, access_count
- Supports temporal decay of importance based on access patterns

**LRUCache** (buildautomata_memory_mcp.py:70)
- OrderedDict-based LRU cache with configurable max size
- Used for both memory objects and embedding vectors

### Storage Architecture

**Triple-Backend System:**
1. **SQLite** - Primary persistent storage
   - `memories` table: Current version of each memory
   - `memory_versions` table: Complete temporal history
   - FTS5 virtual table for full-text search
   - Located at: `memory_repos/{username}_{agent_name}/memoryv012.db`

2. **Qdrant** (optional) - Vector search backend
   - Stores embeddings for semantic similarity search
   - Configurable via QDRANT_HOST and QDRANT_PORT
   - Collection per user/agent: `{username}_{agent_name}_memories`

3. **SentenceTransformers** (optional) - Embedding generation
   - Default model: all-MiniLM-L6-v2 (768-dimensional)
   - Falls back to simple hash-based embeddings if unavailable

### MCP Tools Exposed

The server exposes 7 tools (buildautomata_memory_mcp.py:1410):
- `store_memory` - Create new memory with category, importance, tags
- `update_memory` - Update existing memory (creates new version)
- `search_memories` - Semantic + full-text search with filters
- `get_memory_timeline` - View complete version history
- `get_memory_stats` - System statistics and cache info
- `prune_old_memories` - Remove least important memories
- `run_maintenance` - Database maintenance (VACUUM, ANALYZE)

### Key Algorithms

**Importance Decay:**
- Memories decay in importance over time based on access patterns
- Formula uses `decay_rate` (default 0.95) and time since last access
- Prevents stale memories from dominating search results

**Version Tracking:**
- Every update creates a new entry in `memory_versions` table
- Tracks change_type (created/content_changed/metadata_changed/etc.)
- Stores diffs and change descriptions
- Content hash (SHA-256) used for deduplication

**Hybrid Search:**
- Combines vector similarity (Qdrant) with full-text search (SQLite FTS5)
- Results merged and ranked by relevance
- Automatic fallback to FTS-only if Qdrant unavailable

## Running the Server

### Environment Variables

Required:
- `BA_USERNAME` - User identifier (default: "buildautomata_ai_v012")
- `BA_AGENT_NAME` - Agent identifier (default: "claude_assistant")

Optional:
- `QDRANT_HOST` - Qdrant server host (default: "localhost")
- `QDRANT_PORT` - Qdrant server port (default: 6333)
- `MAX_MEMORIES` - Maximum memories to retain (default: 10000)
- `CACHE_MAXSIZE` - LRU cache size (default: 1000)
- `QDRANT_MAX_RETRIES` - Retry attempts for Qdrant (default: 3)
- `MAINTENANCE_INTERVAL_HOURS` - Auto-maintenance interval (default: 24)

### Starting the Server

```bash
# Direct execution
python buildautomata_memory_mcp.py

# With custom configuration
BA_USERNAME="myuser" BA_AGENT_NAME="myagent" python buildautomata_memory_mcp.py
```

The server communicates via stdio using the MCP protocol.

### Testing Tools

Use MCP inspector or direct stdio communication to test:

```bash
# Example: Store a memory (via MCP client)
{
  "tool": "store_memory",
  "arguments": {
    "content": "User prefers dark mode",
    "category": "user_preference",
    "importance": 0.8,
    "tags": ["ui", "settings"]
  }
}
```

## Development Notes

### Dependencies

**Required:**
- `mcp` - Model Context Protocol SDK
- Standard library: sqlite3, asyncio, threading, logging, hashlib

**Optional (graceful degradation):**
- `qdrant-client` - Vector database client
- `sentence-transformers` - Embedding generation
- Falls back to SQLite-only mode if missing

### Thread Safety

- All database operations protected by `_db_lock` (RLock)
- SQLite connection created with `check_same_thread=False`
- Timeout set to 30 seconds to prevent deadlocks
- Isolation level: IMMEDIATE for write consistency

### Logging

- All logs sent to stderr (stdout reserved for MCP protocol)
- stdout redirected to stderr before imports (line 24)
- Restored before MCP communication starts (line 1787)
- Set third-party loggers (qdrant, sentence_transformers) to CRITICAL level

### Database Schema

**memories table** - Current state
- Indexed on: id (PK), category, importance, created_at, updated_at
- Full-text search on content via FTS5 virtual table

**memory_versions table** - Historical state
- Composite index on (memory_id, version_number)
- Timestamp index for temporal queries
- Stores complete snapshot of each version

### Error Handling

- Errors logged to `self.error_log` list
- `_log_error()` method tracks failures with timestamps
- Graceful degradation when optional backends unavailable
- Tools return error messages to MCP clients without crashing

## File Locations

- **Server code:** `buildautomata_memory_mcp.py` (1815 lines)
- **Database:** `memory_repos/{username}_{agent_name}/memoryv012.db`
- **Qdrant collection:** `{username}_{agent_name}_memories` (in Qdrant server)

## Common Operations

**Adding new MCP tools:**
1. Add tool definition in `handle_list_tools()` (line 1410)
2. Add handler case in `handle_call_tool()` (line 1575)
3. Implement method in MemoryStore class if needed

**Modifying search behavior:**
- Vector search: `_search_vector()` (line 878)
- SQLite search: `_search_sqlite()`
- Hybrid search: `search_memories()` (line 733)

**Changing database schema:**
- Modify `_init_sqlite()` (line 196)
- Consider migration path for existing databases
- Update version number in db filename (currently "memoryv012.db")

**Adjusting memory lifecycle:**
- Pruning logic: `prune_old_memories()` (line 1093)
- Decay calculation: `current_importance()` in Memory class
- Maintenance: `maintenance()` (line 1180)
