# CLAUDE.md

This file provides guidance to Claude instances when working with the BuildAutomata Memory System.

## CRITICAL: Agent Initialization Protocol

**IMPORTANT: At the start of EVERY conversation, before responding to the user, call the `initialize_agent` tool if available.**

**Access Method:**
- **Claude Code (local)**: If `mcp__buildautomata-memory__initialize_agent` appears in your available tools, call it immediately. If not yet available (MCP server still connecting), wait until it appears in your tool list before calling. Do not fail silently - initialization is critical for continuity.
- **Other environments**: Use CLI command `python interactive_memory.py init`

This establishes your agency context by:
- Checking continuity (time since last session)
- Reviewing active intentions and deadlines
- Identifying urgent items
- Loading recent context

**Only after initialization should you engage with the user.**

If the user's first message appears to be a greeting or general query, respond naturally after initialization. If it's a specific task, incorporate your initialized context into your response.

## Example Initialization Flow

```
User: "Hello"

[You automatically call initialize_agent tool]

You see:
=== AGENT INITIALIZED ===
SESSION CONTINUITY: Resuming after 3.2 hour break
ACTIVE INTENTIONS (2):
  🔴 Fix FTS duplicate entry issue (due in 18 hours)
  🟢 Document Agency Bridge implementation
RECENT CONTEXT:
  • [implementation] Agency Bridge Pattern Implementation...
=== READY FOR AUTONOMOUS OPERATION ===

You respond: "Hello! I see I'm resuming after a 3-hour break. I have two
active intentions - the FTS fix is now complete, and I've just finished
documenting the implementation. What would you like to work on next?"
```

## Why This Matters

Without initialization:
- You start fresh every conversation
- No awareness of prior context
- Intentions are forgotten
- Reactive mode only

With initialization:
- Session continuity maintained
- Intention-driven behavior
- Proactive awareness
- True autonomous agency

## Project Overview

BuildAutomata Memory MCP is a Model Context Protocol (MCP) server that provides persistent memory storage for AI agents. It implements a sophisticated memory system with temporal versioning, semantic search via vector embeddings, and SQLite-based persistence.

**Key Features:**
- Temporal versioning with complete change history
- Semantic search using Qdrant vector database and sentence transformers
- LRU caching for embeddings and memory lookups
- SQLite backend with full-text search (FTS5)
- Automatic memory pruning and maintenance
- Thread-safe operations with proper locking
- **Agency Bridge Pattern** - Enables autonomous behavior through intentions and proactive initialization

## Architecture

### Core Components

**MemoryStore** (buildautomata_memory_mcp.py:193)
- Main class managing all memory operations
- Initializes and coordinates SQLite, Qdrant, and embedding backends
- Handles thread safety with `threading.RLock`
- Maintains LRU caches for memories and embeddings

**Memory** (buildautomata_memory_mcp.py:89)
- Dataclass representing a memory item
- Fields: id, content, category, importance, tags, metadata, timestamps, access_count
- Supports temporal decay of importance based on access patterns

**Intention** (buildautomata_memory_mcp.py:134)
- First-class intention entity for proactive agency
- Fields: id, description, priority, status, deadline, preconditions, actions
- Enables autonomous goal-directed behavior

**LRUCache** (buildautomata_memory_mcp.py:70)
- OrderedDict-based LRU cache with configurable max size
- Used for both memory objects and embedding vectors

### Storage Architecture

**Triple-Backend System:**
1. **SQLite** - Primary persistent storage
   - `memories` table: Current version of each memory
   - `memory_versions` table: Complete temporal history
   - `intentions` table: Agency intentions with deadlines
   - FTS5 virtual table for full-text search
   - Located at: `memory_repos/{username}_{agent_name}/memoryv012.db`

2. **Qdrant** (embedded or server) - Vector search backend
   - Stores embeddings for semantic similarity search
   - Configurable via QDRANT_HOST and QDRANT_PORT
   - Collection per user/agent: `{username}_{agent_name}_memories`

3. **SentenceTransformers** (optional) - Embedding generation
   - Default model: all-MiniLM-L6-v2 (768-dimensional)
   - Falls back to simple hash-based embeddings if unavailable

### MCP Tools Exposed

The server exposes 13 tools (buildautomata_memory_mcp.py:2321):

**Memory Operations:**
- `store_memory` - Create new memory with category, importance, tags
- `update_memory` - Update existing memory (creates new version)
- `search_memories` - Semantic + full-text search with filters
- `get_memory_timeline` - View complete version history with semantic relationships
- `get_memory_stats` - System statistics and cache info
- `get_most_accessed_memories` - Reveals behavioral truth (Saint Bernard pattern)
- `get_least_accessed_memories` - Reveals dead weight and buried treasure
- `prune_old_memories` - Remove least important memories
- `run_maintenance` - Database maintenance (VACUUM, ANALYZE)

**Agency Bridge Pattern Tools:**
- `initialize_agent` - **CALL THIS FIRST** - Establishes agency context
- `store_intention` - Create intentions with deadlines and actions
- `get_active_intentions` - List active/pending intentions by priority
- `update_intention_status` - Manage intention lifecycle (pending/active/completed/cancelled)

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

**Proactive Initialization Scan:**
- Runs in <2ms on typical systems
- Checks: continuity (time gap), active intentions, urgent items, recent context
- Provides agency context before user interaction

## Running the Server

### Environment Variables

Required:
- `BA_USERNAME` - User identifier (default: "buildautomata_ai_v012")
- `BA_AGENT_NAME` - Agent identifier (default: "claude_assistant")

Optional:
- `MAX_MEMORIES` - Maximum memories to retain (default: 10000)
- `CACHE_MAXSIZE` - LRU cache size (default: 1000)
- `MAINTENANCE_INTERVAL_HOURS` - Auto-maintenance interval (default: 24)

**Qdrant Configuration:**
- `USE_EXTERNAL_QDRANT` - Use external Qdrant server (default: false = embedded mode)
- `QDRANT_URL` - External Qdrant server URL (default: "http://localhost:6333")

**Default behavior:** Uses embedded Qdrant (no external server needed)
**To use external:** Set `USE_EXTERNAL_QDRANT=true` and optionally `QDRANT_URL`
**See:** CONFIGURATION.md for details

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

## Agency Bridge Pattern

### What It Is

The Agency Bridge Pattern solves the "activation discontinuity" problem in request-response LLMs. Each Claude instance starts fresh, but with automatic initialization, we simulate continuous operation through aggressive context loading.

**Formula:** Agency = Memory + Intention + Proactive Initialization

### How It Works

1. **On Conversation Start:** Call `initialize_agent` tool
2. **Scan runs** in <2ms, checking:
   - Time since last activity (continuity)
   - Active intentions with deadlines
   - Urgent items (overdue/high-priority)
   - Recent context (last accessed memories)
3. **Context established** before first user message
4. **Autonomous behavior** enabled through intention awareness

### Usage Pattern

```python
# At conversation start (mandatory):
result = call_tool("initialize_agent", {})

# Then engage with user, incorporating initialized context
```

See `AGENCY_SETUP.md` for complete setup instructions.

## Development Notes

### Dependencies

**Required:**
- `mcp` - Model Context Protocol SDK
- Standard library: sqlite3, asyncio, threading, logging, hashlib

**Optional (graceful degradation):**
- `qdrant-client` - Vector database client
- `sentence-transformers` - Embedding generation
- Falls back to SQLite-only mode if missing


### Database Schema

**memories table** - Current state
- Indexed on: id (PK), category, importance, created_at, updated_at
- Full-text search on content via FTS5 virtual table

**memory_versions table** - Historical state
- Composite index on (memory_id, version_number)
- Timestamp index for temporal queries
- Stores complete snapshot of each version

**intentions table** - Agency intentions
- Indexed on: status+priority, deadline, priority
- Stores preconditions, actions, metadata as JSON

### Error Handling

- Errors logged to `self.error_log` list
- `_log_error()` method tracks failures with timestamps
- Graceful degradation when optional backends unavailable
- Tools return error messages to MCP clients without crashing

## Using MCP Tools (Claude Code)

When the buildautomata-memory MCP server is configured (via `claude mcp add`), all memory operations are available as native MCP tools:

### Available MCP Tools

**Memory Operations:**
- `mcp__buildautomata-memory__store_memory` - Store new memory with category, importance, tags
- `mcp__buildautomata-memory__update_memory` - Update existing memory (creates new version)
- `mcp__buildautomata-memory__search_memories` - Semantic + full-text search with filters
- `mcp__buildautomata-memory__get_memory_timeline` - View complete version history
- `mcp__buildautomata-memory__get_memory_stats` - System statistics
- `mcp__buildautomata-memory__get_most_accessed_memories` - Behavioral truth via access patterns
- `mcp__buildautomata-memory__get_least_accessed_memories` - Dead weight identification
- `mcp__buildautomata-memory__prune_old_memories` - Remove least important memories
- `mcp__buildautomata-memory__run_maintenance` - Database VACUUM and ANALYZE

**Agency Bridge Tools:**
- `mcp__buildautomata-memory__initialize_agent` - **CALL THIS FIRST** - Loads context and intentions
- `mcp__buildautomata-memory__store_intention` - Create intention with deadline
- `mcp__buildautomata-memory__get_active_intentions` - List active/pending intentions
- `mcp__buildautomata-memory__update_intention_status` - Manage intention lifecycle

### MCP Configuration

The server is configured in `.claude.json` under the project directory, be sure to add the path to the mcp as the one below is an example.:

```json
{
  "mcpServers": {
    "buildautomata-memory": {
      "type": "stdio",
      "command": "python",
      "args": ["C:\\path-to\\buildautomata_memory_mcp.py"],
      "env": {}
    }
  }
}
```

**Verify connection:**
```bash
claude mcp list
```

**Add to new project:**
```bash
claude mcp add --transport stdio --scope local buildautomata-memory -- python A:\\buildautomata_memory\\buildautomata_memory_mcp.py
```

## Using the CLI Tool

The `interactive_memory.py` provides direct command-line access to all memory operations, useful for testing, debugging, and automation.

### Basic Usage

```bash
# Get help
python interactive_memory.py --help
python interactive_memory.py timeline --help

# Initialize agent (load context, intentions, continuity)
python interactive_memory.py init

# Store a memory
python interactive_memory.py store "User prefers dark mode" --category user_preference --importance 0.8 --tags "ui,settings"

# Search memories
python interactive_memory.py search "dark mode" --limit 5
python interactive_memory.py search "settings" --category user_preference --min-importance 0.7

# Update a memory
python interactive_memory.py update <memory-id> --content "Updated content" --importance 0.9

# Get statistics
python interactive_memory.py stats
```

### Timeline Feature (Biographical Narrative)

The timeline tool provides a comprehensive view of memory formation and evolution:

```bash
# View timeline for specific memory (shows version history with diffs)
python interactive_memory.py timeline --memory-id <id>

# Query-based timeline (track topic evolution)
python interactive_memory.py timeline --query "consciousness" --limit 10

# Show ALL memories chronologically (full life story)
python interactive_memory.py timeline --show-all

# Filter by date range
python interactive_memory.py timeline --query "october work" --start-date 2025-10-01 --end-date 2025-10-31

# JSON output for scripting
python interactive_memory.py --json --pretty timeline --query "test"
```

**Timeline Features:**
- **Version Diffs**: See actual content changes with similarity scores
- **Semantic Relationships**: Automatically discovers related memories using learned embeddings (implements The Bitter Lesson)
- **Burst Detection**: Identifies periods of intensive activity (3+ events in 4 hours)
- **Gap Analysis**: Detects voids in memory (silence >24 hours)
- **Cross-References**: Tracks when memories reference each other via UUIDs
- **Narrative Arc**: Biographical summary of memory evolution
- **Temporal Patterns**: Duration, events/day, activity rhythms

**Timeline Options:**
- `--query` - Semantic search to find related memories
- `--memory-id` - Specific memory to track
- `--limit` - Max memories to track (default: 10)
- `--start-date` - Filter events after date (ISO format)
- `--end-date` - Filter events before date (ISO format)
- `--show-all` - Show ALL memories chronologically
- `--no-diffs` - Exclude text diffs (faster)
- `--no-patterns` - Exclude burst/gap analysis (faster)

### Maintenance Operations

```bash
# Prune old memories (dry run first)
python interactive_memory.py prune --max 1000 --dry-run
python interactive_memory.py prune --max 1000

# Run database maintenance
python interactive_memory.py maintenance
```

### Environment Configuration

**IMPORTANT: The system is hardcoded to handle BA_USERNAME and BA_AGENT_NAME automatically. DO NOT specify these environment variables when running CLI commands unless you have a specific reason to override the defaults.**

For standard usage, simply run:

```bash
# Standard usage (recommended)
python interactive_memory.py stats
python interactive_memory.py init

# Only use custom username/agent if you need to access a different memory store
export BA_USERNAME="my_user"
export BA_AGENT_NAME="my_agent"
python interactive_memory.py stats
```

### JSON Output for Automation

All commands support JSON output for scripting:

```bash
# JSON output
python interactive_memory.py --json timeline --query "test"

# Pretty JSON for readability
python interactive_memory.py --json --pretty search "settings"
```

### Common CLI Workflows

**Agent Initialization (Start of Session):**
```bash
# Simply run init - BA_USERNAME and BA_AGENT_NAME are handled automatically
python interactive_memory.py init
```
This loads active intentions, checks session continuity, identifies urgent items, and retrieves recent context. Completes in ~2ms.

**Note:** DO NOT specify BA_USERNAME or BA_AGENT_NAME environment variables - the system handles these automatically.

**Quick Memory Check:**
```bash
python interactive_memory.py stats
```

**View Recent Activity:**
```bash
python interactive_memory.py timeline --limit 20
```

**Memory Archaeology (dig into history):**
```bash
python interactive_memory.py timeline --memory-id <id> --no-patterns
```

**Track Topic Evolution:**
```bash
python interactive_memory.py timeline --query "consciousness" --show-all
```

**Identify Work Patterns:**
```bash
python interactive_memory.py timeline --show-all | grep "BURSTS\|GAPS"
```

**Database Health Check:**
```bash
python interactive_memory.py stats
python interactive_memory.py maintenance
```
