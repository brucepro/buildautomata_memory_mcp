# BuildAutomata Memory Web Server

## Overview
FastAPI-based web server exposing the full BuildAutomata Memory system via HTTP/HTTPS, including all 5 new memory improvements.

## Files
- **memory_web_server_fixed.py** - Web server with all memory features
- **start_tunnel.bat** - Cloudflare tunnel launcher for remote access

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install fastapi uvicorn
```

### 2. Start the Server
```bash
python memory_web_server_fixed.py
```

Server will run on `http://localhost:8766`

### 3. Setup Remote Access (Optional)

**Option A: Cloudflare Tunnel (Recommended)**
1. Download cloudflared.exe from https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/
2. Create `bin/` folder and place `cloudflared.exe` inside
3. Run `start_tunnel.bat`
4. Copy the HTTPS URL displayed (e.g., `https://xxx-yyy-zzz.trycloudflare.com`)

**Option B: Localtunnel**
```bash
npm install -g localtunnel
lt --port 8766
```

## API Endpoints

### Web UI
- **/** - Interactive web interface for managing memories

### Core Memory Operations
- **POST /api/memories** - Store new memory
- **GET /api/memories/search** - Search memories
- **POST /api/memories/session** - Get session memories
- **POST /api/memories/consolidate** - Consolidate memories

### Statistics & Timeline
- **GET /api/stats** - Get memory statistics
- **GET /api/timeline** - Get memory timeline
- **GET /api/init** - Initialize agent context

### Intentions
- **POST /api/intentions** - Create intention
- **GET /api/intentions/active** - Get active intentions

## Tool Endpoints (for Claude Code)

All features available at `/tool/{tool_name}`:

### store_memory
```json
POST /tool/store_memory
{
  "content": "Memory content",
  "category": "general",
  "importance": 0.7,
  "tags": ["tag1", "tag2"],
  "memory_type": "episodic",
  "session_id": "session-2025-11-12",
  "task_context": "Working on memory improvements"
}
```

### search_memories
```json
POST /tool/search_memories
{
  "query": "search term",
  "limit": 10,
  "category": "research",
  "min_importance": 0.5,
  "memory_type": "episodic",
  "session_id": "session-2025-11-12"
}
```

### get_session_memories
```json
POST /tool/get_session_memories
{
  "session_id": "session-2025-11-12",
  "start_date": "2025-11-10T00:00:00",
  "end_date": "2025-11-12T23:59:59",
  "task_context": "memory improvements",
  "limit": 100
}
```

### consolidate_memories
```json
POST /tool/consolidate_memories
{
  "memory_ids": ["id1", "id2", "id3"],
  "consolidation_type": "synthesize",
  "target_length": 500,
  "new_memory_type": "semantic"
}
```

### Other Tools
- **GET /tool/get_stats** - Memory statistics
- **POST /tool/get_timeline** - Memory timeline
- **GET /tool/get_active_intentions** - Active intentions
- **GET /tool/initialize_agent** - Initialize agent context

## New Features (5 Memory Improvements)

### 1. Memory Type Classification
- `memory_type`: "episodic" | "semantic" | "working"
- Episodic: Specific events/experiences
- Semantic: Timeless knowledge/facts
- Working: Temporary task-related info

### 2. Session Clustering
- `session_id`: Group memories by work session
- `task_context`: Describe what was being worked on
- Retrieve all memories from a session

### 3. Provenance Tracking
- Automatic tracking of when/why memory accessed
- Session and task context preserved
- Enables "show me everything from Nov 10-11"

### 4. Memory Consolidation
- Merge multiple episodic memories into semantic
- Three modes: summarize, synthesize, compress
- Preserves provenance chain

### 5. Typed Relationships
- Explicit relationships between memories
- Types: builds_on, contradicts, implements, analyzes, references
- Makes connections explicit

## Usage Examples

### Store an Episodic Memory with Session Context
```python
import requests

response = requests.post("http://localhost:8766/tool/store_memory", json={
    "content": "Fixed migration bug in schema creation order",
    "category": "implementation",
    "importance": 0.8,
    "tags": ["bugfix", "migration", "Nov_12_2025"],
    "memory_type": "episodic",
    "session_id": "memory-improvements-nov12",
    "task_context": "Implementing 5 research-backed memory improvements"
})
```

### Search by Session
```python
response = requests.post("http://localhost:8766/tool/search_memories", json={
    "query": "migration",
    "memory_type": "episodic",
    "session_id": "memory-improvements-nov12",
    "limit": 20
})
```

### Get All Memories from a Session
```python
response = requests.post("http://localhost:8766/tool/get_session_memories", json={
    "session_id": "memory-improvements-nov12",
    "limit": 100
})
```

### Consolidate Multiple Memories
```python
response = requests.post("http://localhost:8766/tool/consolidate_memories", json={
    "memory_ids": ["id1", "id2", "id3", "id4"],
    "consolidation_type": "synthesize",
    "target_length": 600,
    "new_memory_type": "semantic"
})
```

## Configuration

The server is hardcoded to use:
- **Username**: `buildautomata_ai_v012`
- **Agent Name**: `claude_assistant`

This matches the Claude Code instance memory account.

## Ports
- Local server: `8766`
- Tunnel exposes: `8766` → `https://random-url.trycloudflare.com`

## API Documentation
Interactive API docs available at:
- Swagger UI: `http://localhost:8766/docs`
- ReDoc: `http://localhost:8766/redoc`
- Tools list: `http://localhost:8766/tools/list`

## Important: Embedded Qdrant Locking

**When the web server is running, it holds an exclusive lock on the embedded Qdrant vector database.**

This means:
- Other processes (CLI, MCP server, other scripts) **cannot write** to the vector store while the web server is active
- Memories stored via other interfaces will save to SQLite but **fail to create embeddings**
- This is expected behavior with embedded Qdrant (single-process access only)

**Solutions:**
1. **Use the web server's API** - All memory operations should go through the web server when it's running
2. **Stop the web server** - If you need to use CLI or MCP directly, stop the web server first
3. **Use external Qdrant** - For concurrent access, set `USE_EXTERNAL_QDRANT=true` and run a Qdrant server
4. **Run maintenance** - After web server stops, run maintenance to repair any memories missing embeddings

**Repair missing embeddings:**
```bash
# Via CLI when web server is stopped
python interactive_memory.py maintenance

# Or via MCP tool
run_maintenance  # Will detect and repair missing embeddings
```

See `CONFIGURATION.md` for details on embedded vs external Qdrant modes.

## Security Notes
- Server allows CORS from all origins (development mode)
- Tunnel URLs are random and temporary
- No authentication required (trusted local network)
- For production: add authentication, restrict CORS, use permanent tunnels
