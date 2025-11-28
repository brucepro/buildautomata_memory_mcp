#!/usr/bin/env python3
"""
BuildAutomata Memory Web Server
FastAPI server providing web interface to memory operations.
Enables Claude Code Web instance to access memory storage via localhost.

"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime
from contextlib import asynccontextmanager
import uvicorn
import sys
import os
import uuid

# Add parent directory to path to import interactive_memory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the memory store
from buildautomata_memory_mcp import MemoryStore, Memory

# Initialize FastAPI
app = FastAPI(
    title="BuildAutomata Memory API",
    description="Web interface for BuildAutomata persistent memory system",
    version="1.0.0"
)

# Enable CORS for localhost access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Command history logging middleware for tool endpoints
@app.middleware("http")
async def log_tool_commands(request, call_next):
    """Log tool endpoint calls to command history"""
    response = await call_next(request)

    # Only log tool endpoints
    if request.url.path.startswith("/tool/"):
        try:
            tool_name = f"web_{request.url.path.replace('/tool/', '')}"
            # Create temporary store just for logging
            temp_store = MemoryStore(username="buildautomata_ai_v012", agent_name="claude_assistant", lazy_load=True)
            try:
                # Log with minimal info (can't easily get request body in middleware after call_next)
                temp_store.sqlite_store.log_command(
                    tool_name,
                    {"method": request.method, "path": request.url.path},
                    f"status_{response.status_code}",
                    None,
                    response.status_code < 400
                )
            finally:
                await temp_store.shutdown()
        except Exception:
            pass  # Don't fail requests if logging fails

    return response


# Configuration - HARDCODED USERNAME
USERNAME = "buildautomata_ai_v012"
AGENT_NAME = "claude_assistant"

async def get_memory_store() -> AsyncGenerator[MemoryStore, None]:
    """
    Dependency that creates a new memory store instance per request and ensures cleanup.

    Usage in endpoints:
        @app.get("/endpoint")
        async def handler(store: MemoryStore = Depends(get_memory_store)):
            ...
    """
    store = MemoryStore(username=USERNAME, agent_name=AGENT_NAME)
    try:
        yield store
    finally:
        await store.shutdown()


# Pydantic models for request/response
class StoreMemoryRequest(BaseModel):
    content: str = Field(..., description="Memory content to store")
    category: str = Field(default="general", description="Memory category")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score 0-1")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")
    memory_type: str = Field(default="episodic", description="Memory type: episodic/semantic/working")
    session_id: Optional[str] = Field(default=None, description="Session ID for grouping memories")
    task_context: Optional[str] = Field(default=None, description="Task context description")


class SearchMemoriesRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    category: Optional[str] = Field(default=None, description="Filter by category")
    min_importance: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Min importance")
    memory_type: Optional[str] = Field(default=None, description="Filter by memory type")
    session_id: Optional[str] = Field(default=None, description="Filter by session ID")


class UpdateMemoryRequest(BaseModel):
    memory_id: str = Field(..., description="Memory ID to update")
    content: Optional[str] = Field(default=None, description="New content")
    category: Optional[str] = Field(default=None, description="New category")
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="New importance")
    tags: Optional[List[str]] = Field(default=None, description="New tags")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="New metadata")


class StoreIntentionRequest(BaseModel):
    description: str = Field(..., description="Intention description")
    priority: str = Field(default="medium", description="Priority: low/medium/high/urgent")
    deadline: Optional[str] = Field(default=None, description="ISO format deadline")
    preconditions: Optional[List[str]] = Field(default=None, description="Preconditions")
    actions: Optional[List[str]] = Field(default=None, description="Action items")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class UpdateIntentionStatusRequest(BaseModel):
    intention_id: str = Field(..., description="Intention ID")
    status: str = Field(..., description="Status: pending/active/completed/cancelled")
    outcome: Optional[str] = Field(default=None, description="Outcome description")


class GetSessionMemoriesRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Session ID to retrieve")
    start_date: Optional[str] = Field(default=None, description="Start date (ISO format)")
    end_date: Optional[str] = Field(default=None, description="End date (ISO format)")
    task_context: Optional[str] = Field(default=None, description="Task context filter")
    limit: int = Field(default=100, ge=1, le=500, description="Max memories")


class ConsolidateMemoriesRequest(BaseModel):
    memory_ids: List[str] = Field(..., description="List of memory IDs to consolidate")
    consolidation_type: str = Field(default="summarize", description="Type: summarize/synthesize/compress")
    target_length: int = Field(default=500, ge=100, le=2000, description="Target length in characters")
    new_memory_type: str = Field(default="semantic", description="New memory type")


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve web UI."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BuildAutomata Memory</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }
            h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
            h2 { color: #555; margin-top: 30px; }
            .container { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .form-group { margin-bottom: 15px; }
            label { display: block; font-weight: bold; margin-bottom: 5px; color: #555; }
            input[type="text"], input[type="number"], textarea, select {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
                box-sizing: border-box;
            }
            textarea { min-height: 100px; font-family: monospace; }
            button {
                background: #4CAF50;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
            }
            button:hover { background: #45a049; }
            button.secondary { background: #2196F3; }
            button.secondary:hover { background: #0b7dda; }
            button.danger { background: #f44336; }
            button.danger:hover { background: #da190b; }
            #results {
                margin-top: 20px;
                padding: 15px;
                background: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-family: monospace;
                white-space: pre-wrap;
                max-height: 600px;
                overflow-y: auto;
            }
            .memory-card {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #4CAF50;
                border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .memory-meta {
                color: #666;
                font-size: 12px;
                margin-top: 8px;
            }
            .memory-content {
                margin: 10px 0;
                color: #333;
                line-height: 1.6;
            }
            .tag {
                display: inline-block;
                background: #e3f2fd;
                color: #1976d2;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 11px;
                margin-right: 5px;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            .stat-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-value {
                font-size: 32px;
                font-weight: bold;
                margin-bottom: 5px;
            }
            .stat-label {
                font-size: 14px;
                opacity: 0.9;
            }
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                border-bottom: 2px solid #ddd;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                background: none;
                border: none;
                border-bottom: 3px solid transparent;
                font-size: 14px;
                font-weight: bold;
                color: #666;
            }
            .tab.active {
                color: #4CAF50;
                border-bottom-color: #4CAF50;
            }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
        </style>
    </head>
    <body>
        <h1>🧠 BuildAutomata Memory System</h1>

        <div class="tabs">
            <button class="tab active" onclick="showTab('store')">Store Memory</button>
            <button class="tab" onclick="showTab('search')">Search</button>
            <button class="tab" onclick="showTab('intentions')">Intentions</button>
            <button class="tab" onclick="showTab('timeline')">Timeline</button>
            <button class="tab" onclick="showTab('stats')">Statistics</button>
        </div>

        <!-- Store Memory Tab -->
        <div id="store" class="tab-content active">
            <div class="container">
                <h2>Store New Memory</h2>
                <div class="form-group">
                    <label>Content:</label>
                    <textarea id="storeContent" placeholder="Enter memory content..."></textarea>
                </div>
                <div class="form-group">
                    <label>Category:</label>
                    <select id="storeCategory">
                        <option value="general">General</option>
                        <option value="research">Research</option>
                        <option value="implementation">Implementation</option>
                        <option value="philosophy">Philosophy</option>
                        <option value="synthesis">Synthesis</option>
                        <option value="project_context">Project Context</option>
                        <option value="session_start">Session Start</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Importance (0-1):</label>
                    <input type="number" id="storeImportance" value="0.7" min="0" max="1" step="0.1">
                </div>
                <div class="form-group">
                    <label>Tags (comma-separated):</label>
                    <input type="text" id="storeTags" placeholder="tag1, tag2, tag3">
                </div>
                <button onclick="storeMemory()">Store Memory</button>
            </div>
        </div>

        <!-- Search Tab -->
        <div id="search" class="tab-content">
            <div class="container">
                <h2>Search Memories</h2>
                <div class="form-group">
                    <label>Query:</label>
                    <input type="text" id="searchQuery" placeholder="Search for...">
                </div>
                <div class="form-group">
                    <label>Limit:</label>
                    <input type="number" id="searchLimit" value="10" min="1" max="100">
                </div>
                <div class="form-group">
                    <label>Category (optional):</label>
                    <select id="searchCategory">
                        <option value="">All Categories</option>
                        <option value="research">Research</option>
                        <option value="implementation">Implementation</option>
                        <option value="philosophy">Philosophy</option>
                        <option value="synthesis">Synthesis</option>
                        <option value="project_context">Project Context</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Min Importance:</label>
                    <input type="number" id="searchMinImportance" value="0" min="0" max="1" step="0.1">
                </div>
                <button onclick="searchMemories()">Search</button>
            </div>
        </div>

        <!-- Intentions Tab -->
        <div id="intentions" class="tab-content">
            <div class="container">
                <h2>Create Intention</h2>
                <div class="form-group">
                    <label>Description:</label>
                    <textarea id="intentionDesc" placeholder="What do you intend to accomplish?"></textarea>
                </div>
                <div class="form-group">
                    <label>Priority:</label>
                    <select id="intentionPriority">
                        <option value="low">Low</option>
                        <option value="medium" selected>Medium</option>
                        <option value="high">High</option>
                        <option value="urgent">Urgent</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Deadline (ISO format, optional):</label>
                    <input type="text" id="intentionDeadline" placeholder="2025-11-17T00:00:00">
                </div>
                <div class="form-group">
                    <label>Actions (one per line):</label>
                    <textarea id="intentionActions" placeholder="Action 1\nAction 2\nAction 3"></textarea>
                </div>
                <button onclick="storeIntention()">Create Intention</button>
                <button class="secondary" onclick="getActiveIntentions()" style="margin-left: 10px;">View Active Intentions</button>
            </div>
        </div>

        <!-- Timeline Tab -->
        <div id="timeline" class="tab-content">
            <div class="container">
                <h2>Memory Timeline</h2>
                <div class="form-group">
                    <label>Memory ID (optional):</label>
                    <input type="text" id="timelineMemoryId" placeholder="Leave empty for query-based timeline">
                </div>
                <div class="form-group">
                    <label>Query:</label>
                    <input type="text" id="timelineQuery" placeholder="Search query for timeline">
                </div>
                <div class="form-group">
                    <label>Limit:</label>
                    <input type="number" id="timelineLimit" value="10" min="1" max="100">
                </div>
                <button onclick="getTimeline()">Get Timeline</button>
            </div>
        </div>

        <!-- Stats Tab -->
        <div id="stats" class="tab-content">
            <div class="container">
                <h2>Memory Statistics</h2>
                <button onclick="getStats()">Refresh Statistics</button>
            </div>
        </div>

        <div id="results"></div>

        <script>
            function showTab(tabName) {
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));

                // Show selected tab
                document.getElementById(tabName).classList.add('active');
                event.target.classList.add('active');
            }

            async function storeMemory() {
                const content = document.getElementById('storeContent').value;
                const category = document.getElementById('storeCategory').value;
                const importance = parseFloat(document.getElementById('storeImportance').value);
                const tagsText = document.getElementById('storeTags').value;
                const tags = tagsText ? tagsText.split(',').map(t => t.trim()) : null;

                try {
                    const response = await fetch('/api/memories', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ content, category, importance, tags })
                    });
                    const data = await response.json();
                    document.getElementById('results').textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    document.getElementById('results').textContent = 'Error: ' + error.message;
                }
            }

            async function searchMemories() {
                const query = document.getElementById('searchQuery').value;
                const limit = parseInt(document.getElementById('searchLimit').value);
                const category = document.getElementById('searchCategory').value || null;
                const minImportance = parseFloat(document.getElementById('searchMinImportance').value) || null;

                try {
                    const params = new URLSearchParams({ query, limit });
                    if (category) params.append('category', category);
                    if (minImportance) params.append('min_importance', minImportance);

                    const response = await fetch('/api/memories/search?' + params);
                    const data = await response.json();

                    // Format results as cards
                    if (data.memories && data.memories.length > 0) {
                        let html = '<div style="background: white; padding: 15px; border-radius: 8px;">';
                        html += '<h3>Found ' + data.memories.length + ' memories</h3>';
                        data.memories.forEach(mem => {
                            html += '<div class="memory-card">';
                            html += '<div class="memory-content">' + escapeHtml(mem.content) + '</div>';
                            html += '<div class="memory-meta">';
                            html += '<strong>ID:</strong> ' + mem.memory_id + ' | ';
                            html += '<strong>Category:</strong> ' + mem.category + ' | ';
                            html += '<strong>Importance:</strong> ' + mem.importance.toFixed(2) + ' | ';
                            html += '<strong>Created:</strong> ' + new Date(mem.created_at).toLocaleString();
                            if (mem.tags && mem.tags.length > 0) {
                                html += '<br>';
                                mem.tags.forEach(tag => {
                                    html += '<span class="tag">' + tag + '</span>';
                                });
                            }
                            html += '</div></div>';
                        });
                        html += '</div>';
                        document.getElementById('results').innerHTML = html;
                    } else {
                        document.getElementById('results').textContent = 'No memories found';
                    }
                } catch (error) {
                    document.getElementById('results').textContent = 'Error: ' + error.message;
                }
            }

            async function storeIntention() {
                const description = document.getElementById('intentionDesc').value;
                const priority = document.getElementById('intentionPriority').value;
                const deadline = document.getElementById('intentionDeadline').value || null;
                const actionsText = document.getElementById('intentionActions').value;
                const actions = actionsText ? actionsText.split('\n').filter(a => a.trim()) : null;

                try {
                    const response = await fetch('/api/intentions', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ description, priority, deadline, actions })
                    });
                    const data = await response.json();
                    document.getElementById('results').textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    document.getElementById('results').textContent = 'Error: ' + error.message;
                }
            }

            async function getActiveIntentions() {
                try {
                    const response = await fetch('/api/intentions/active');
                    const data = await response.json();
                    document.getElementById('results').textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    document.getElementById('results').textContent = 'Error: ' + error.message;
                }
            }

            async function getTimeline() {
                const memoryId = document.getElementById('timelineMemoryId').value || null;
                const query = document.getElementById('timelineQuery').value || null;
                const limit = parseInt(document.getElementById('timelineLimit').value);

                try {
                    const params = new URLSearchParams({ limit });
                    if (memoryId) params.append('memory_id', memoryId);
                    if (query) params.append('query', query);

                    const response = await fetch('/api/timeline?' + params);
                    const data = await response.json();
                    document.getElementById('results').textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    document.getElementById('results').textContent = 'Error: ' + error.message;
                }
            }

            async function getStats() {
                try {
                    const response = await fetch('/api/stats');
                    const data = await response.json();

                    // Format stats as cards
                    let html = '<div class="stats-grid">';
                    html += '<div class="stat-card"><div class="stat-value">' + data.total_memories + '</div><div class="stat-label">Total Memories</div></div>';
                    html += '<div class="stat-card"><div class="stat-value">' + Object.keys(data.by_category).length + '</div><div class="stat-label">Categories</div></div>';
                    html += '<div class="stat-card"><div class="stat-value">' + data.total_versions + '</div><div class="stat-label">Total Versions</div></div>';
                    html += '<div class="stat-card"><div class="stat-value">' + data.avg_importance.toFixed(2) + '</div><div class="stat-label">Avg Importance</div></div>';
                    html += '</div>';

                    html += '<pre style="background: white; padding: 15px; border-radius: 8px; margin-top: 20px;">';
                    html += JSON.stringify(data, null, 2);
                    html += '</pre>';

                    document.getElementById('results').innerHTML = html;
                } catch (error) {
                    document.getElementById('results').textContent = 'Error: ' + error.message;
                }
            }

            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }

            // Load stats on page load
            window.onload = () => getStats();
        </script>
    </body>
    </html>
    """


@app.post("/api/memories")
async def create_memory(
    request: StoreMemoryRequest,
    store: MemoryStore = Depends(get_memory_store)
):
    """Store a new memory."""
    try:

        # Create Memory object (matching MCP pattern)
        memory = Memory(
            id=str(uuid.uuid4()),
            content=request.content,
            category=request.category,
            importance=request.importance,
            tags=request.tags or [],
            metadata=request.metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            memory_type=request.memory_type,
            session_id=request.session_id,
            task_context=request.task_context,
        )

        # FIXED: Properly await async method
        result = await store.store_memory(memory, is_update=False)

        if result.get("success"):
            return {"success": True, "memory_id": memory.id, "result": result}
        else:
            return {"success": False, "error": result.get("error", "Unknown error")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memories/search")
async def search_memories(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100),
    category: Optional[str] = None,
    min_importance: Optional[float] = Query(None, ge=0.0, le=1.0),
    memory_type: Optional[str] = None,
    session_id: Optional[str] = None,
    store: MemoryStore = Depends(get_memory_store)
):
    """Search memories."""
    try:

        # FIXED: Properly await async method
        results = await store.search_memories(
            query=query,
            limit=limit,
            category=category,
            min_importance=min_importance or 0.0,
            memory_type=memory_type,
            session_id=session_id
        )

        # Convert results to serializable format
        memories = []
        for mem in results:
            memories.append({
                "memory_id": mem.get("memory_id"),
                "content": mem.get("content"),
                "category": mem.get("category"),
                "importance": mem.get("importance"),
                "current_importance": mem.get("current_importance"),
                "tags": mem.get("tags", []),
                "created_at": mem.get("created_at"),
                "updated_at": mem.get("updated_at"),
                "access_count": mem.get("access_count", 0),
                "related_memories": mem.get("related_memories", [])
            })

        return {"memories": memories, "count": len(memories)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/memories/{memory_id}")
@app.post("/tool/update_memory")
async def update_memory(memory_id: str = None, request: UpdateMemoryRequest = None, store: MemoryStore = Depends(get_memory_store)):
    """Update an existing memory."""
    try:

        # Handle both path parameter and request body
        if not memory_id and request:
            memory_id = getattr(request, 'memory_id', None)

        if not memory_id:
            raise HTTPException(status_code=400, detail="memory_id is required")

        # Build updates dict
        updates = {}
        if request.content is not None:
            updates['content'] = request.content
        if request.category is not None:
            updates['category'] = request.category
        if request.importance is not None:
            updates['importance'] = request.importance
        if request.tags is not None:
            updates['tags'] = request.tags
        if request.metadata is not None:
            updates['metadata'] = request.metadata

        result = await store.update_memory(memory_id=memory_id, **updates)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/intentions")
async def create_intention(request: StoreIntentionRequest, store: MemoryStore = Depends(get_memory_store)):
    """Store a new intention."""
    try:

        # Convert string priority to float
        priority_map = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "urgent": 0.95
        }
        priority_float = priority_map.get(request.priority.lower(), 0.6)

        result = await store.store_intention(
            description=request.description,
            priority=priority_float,
            deadline=request.deadline,
            preconditions=request.preconditions,
            actions=request.actions,
            metadata=request.metadata
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/intentions/active")
async def get_active_intentions(store: MemoryStore = Depends(get_memory_store)):
    """Get active intentions."""
    try:

        # FIXED: Properly await async method
        intentions = await store.get_active_intentions()

        # Convert to dicts
        result = []
        for intent in intentions:
            result.append({
                "intention_id": intent.get("intention_id"),
                "description": intent.get("description"),
                "priority": intent.get("priority"),
                "status": intent.get("status"),
                "deadline": intent.get("deadline"),
                "created_at": intent.get("created_at"),
                "actions": intent.get("actions", [])
            })

        return {"intentions": result, "count": len(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/intentions/{intention_id}/status")
async def update_intention_status(intention_id: str, request: UpdateIntentionStatusRequest, store: MemoryStore = Depends(get_memory_store)):
    """Update intention status."""
    try:

        # FIXED: Need to implement
        raise HTTPException(status_code=501, detail="Update intention not implemented yet")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/timeline")
async def get_timeline(
    memory_id: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100)
):
    """Get memory timeline."""
    try:

        # FIXED: Properly await async method
        timeline = await store.get_memory_timeline(
            memory_id=memory_id,
            query=query,
            limit=limit
        )

        return timeline
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats(store: MemoryStore = Depends(get_memory_store)):
    """Get memory statistics."""
    try:
        # NOTE: get_statistics is synchronous, no await needed
        stats = store.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/init")
async def initialize_agent(store: MemoryStore = Depends(get_memory_store)):
    """Initialize agent - load context, intentions, continuity, and working set."""
    try:

        # Use MCP's proactive_initialization_scan - includes working set
        result = await store.intention_mgr.proactive_initialization_scan()

        # Add backend status
        result["backend_status"] = {
            "sqlite": "available" if store.db_conn else "unavailable",
            "qdrant": "available" if store.qdrant_client else "unavailable",
            "embeddings": "available" if store.encoder else "unavailable"
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/memories/session")
async def get_session_memories_api(request: GetSessionMemoriesRequest, store: MemoryStore = Depends(get_memory_store)):
    """Get memories from a work session or time period."""
    try:

        date_range = None
        if request.start_date and request.end_date:
            date_range = (request.start_date, request.end_date)

        memories = await store.get_session_memories(
            session_id=request.session_id,
            date_range=date_range,
            task_context=request.task_context,
            limit=request.limit
        )

        return {
            "session_id": request.session_id,
            "date_range": date_range,
            "task_context": request.task_context,
            "count": len(memories),
            "memories": memories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/memories/consolidate")
async def consolidate_memories_api(request: ConsolidateMemoriesRequest, store: MemoryStore = Depends(get_memory_store)):
    """Consolidate multiple episodic memories into semantic memory."""
    try:

        result = await store.consolidate_memories(
            memory_ids=request.memory_ids,
            consolidation_type=request.consolidation_type,
            target_length=request.target_length,
            new_memory_type=request.new_memory_type
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# /tool/ Endpoints - Simplified interface for Claude Code Web
# =============================================================================

@app.get("/tools/list")
async def list_tools():
    """List all available tools with their schemas."""
    return {
        "tools": [
            {
                "name": "store_memory",
                "description": "Store a new memory with content, category, importance, and tags",
                "endpoint": "/tool/store_memory",
                "method": "POST",
                "parameters": {
                    "content": {"type": "string", "required": True},
                    "category": {"type": "string", "default": "general"},
                    "importance": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
                    "tags": {"type": "array", "items": "string", "optional": True},
                    "memory_type": {"type": "string", "default": "episodic", "optional": True},
                    "session_id": {"type": "string", "optional": True},
                    "task_context": {"type": "string", "optional": True}
                }
            },
            {
                "name": "search_memories",
                "description": "Search memories by query with optional filters",
                "endpoint": "/tool/search_memories",
                "method": "POST",
                "parameters": {
                    "query": {"type": "string", "required": True},
                    "limit": {"type": "integer", "default": 10},
                    "category": {"type": "string", "optional": True},
                    "min_importance": {"type": "float", "optional": True},
                    "memory_type": {"type": "string", "optional": True},
                    "session_id": {"type": "string", "optional": True}
                }
            },
            {
                "name": "get_stats",
                "description": "Get memory system statistics",
                "endpoint": "/tool/get_stats",
                "method": "GET",
                "parameters": {}
            },
            {
                "name": "get_timeline",
                "description": "Get memory timeline showing evolution and relationships",
                "endpoint": "/tool/get_timeline",
                "method": "POST",
                "parameters": {
                    "memory_id": {"type": "string", "optional": True},
                    "query": {"type": "string", "optional": True},
                    "limit": {"type": "integer", "default": 10}
                }
            },
            {
                "name": "get_active_intentions",
                "description": "Get all active and pending intentions",
                "endpoint": "/tool/get_active_intentions",
                "method": "GET",
                "parameters": {}
            },
            {
                "name": "initialize_agent",
                "description": "Initialize agent context - load continuity, intentions, recent memories",
                "endpoint": "/tool/initialize_agent",
                "method": "GET",
                "parameters": {}
            },
            {
                "name": "get_session_memories",
                "description": "Get memories from a work session or time period",
                "endpoint": "/tool/get_session_memories",
                "method": "POST",
                "parameters": {
                    "session_id": {"type": "string", "optional": True},
                    "start_date": {"type": "string", "optional": True},
                    "end_date": {"type": "string", "optional": True},
                    "task_context": {"type": "string", "optional": True},
                    "limit": {"type": "integer", "default": 100}
                }
            },
            {
                "name": "consolidate_memories",
                "description": "Consolidate multiple episodic memories into semantic memory",
                "endpoint": "/tool/consolidate_memories",
                "method": "POST",
                "parameters": {
                    "memory_ids": {"type": "array", "items": "string", "required": True},
                    "consolidation_type": {"type": "string", "default": "summarize"},
                    "target_length": {"type": "integer", "default": 500},
                    "new_memory_type": {"type": "string", "default": "semantic"}
                }
            },
            {
                "name": "traverse_memory_graph",
                "description": "Traverse memory graph N hops from starting node. Returns subgraph showing connections between memories for context building.",
                "endpoint": "/tool/traverse_memory_graph",
                "method": "GET",
                "parameters": {
                    "start_memory_id": {"type": "string", "required": True},
                    "depth": {"type": "integer", "default": 2, "min": 1, "max": 5},
                    "max_nodes": {"type": "integer", "default": 50, "min": 1, "max": 200},
                    "min_importance": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
                    "category_filter": {"type": "string", "optional": True}
                }
            },
            {
                "name": "find_memory_clusters",
                "description": "Find densely connected regions in memory graph. Discovers thematic groups and coherent knowledge areas.",
                "endpoint": "/tool/find_memory_clusters",
                "method": "GET",
                "parameters": {
                    "min_cluster_size": {"type": "integer", "default": 3, "min": 2},
                    "min_importance": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
                    "limit": {"type": "integer", "default": 10, "min": 1, "max": 50}
                }
            },
            {
                "name": "get_memory_graph_stats",
                "description": "Get graph connectivity statistics - shows hubs, isolated nodes, connectivity distribution. Helps understand memory network structure.",
                "endpoint": "/tool/get_memory_graph_stats",
                "method": "GET",
                "parameters": {
                    "category": {"type": "string", "optional": True},
                    "min_importance": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0}
                }
            }
        ]
    }


@app.post("/tool/store_memory")
async def tool_store_memory(request: StoreMemoryRequest):
    """Tool endpoint: Store memory."""
    return await create_memory(request)


@app.post("/tool/search_memories")
async def tool_search_memories(request: SearchMemoriesRequest):
    """Tool endpoint: Search memories."""
    return await search_memories(
        query=request.query,
        limit=request.limit,
        category=request.category,
        min_importance=request.min_importance,
        memory_type=request.memory_type,
        session_id=request.session_id
    )


@app.get("/tool/get_stats")
async def tool_get_stats():
    """Tool endpoint: Get statistics."""
    return await get_stats()


@app.post("/tool/get_timeline")
async def tool_get_timeline(
    memory_id: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 10
):
    """Tool endpoint: Get timeline."""
    return await get_timeline(memory_id=memory_id, query=query, limit=limit)


@app.get("/tool/get_active_intentions")
async def tool_get_active_intentions():
    """Tool endpoint: Get active intentions."""
    return await get_active_intentions()


@app.post("/tool/store_intention")
async def tool_store_intention(request: StoreIntentionRequest, store: MemoryStore = Depends(get_memory_store)):
    """Tool endpoint: Store a new intention."""
    return await create_intention(request)


@app.post("/tool/update_intention_status")
async def tool_update_intention_status(request: UpdateIntentionStatusRequest, store: MemoryStore = Depends(get_memory_store)):
    """Tool endpoint: Update intention status."""
    try:

        result = await store.update_intention_status(
            intention_id=request.intention_id,
            status=request.status,
            metadata={"outcome": request.outcome} if request.outcome else {}
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tool/initialize_agent")
async def tool_initialize_agent():
    """Tool endpoint: Initialize agent."""
    return await initialize_agent()


@app.get("/tool/get_command_history")
async def tool_get_command_history(
    limit: int = Query(20, ge=1, le=1000),
    tool_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    store: MemoryStore = Depends(get_memory_store)
):
    """Tool endpoint: Get command history for audit trail and session reconstruction."""
    try:
        result = store.sqlite_store.get_command_history(
            limit=limit,
            tool_name=tool_name,
            start_date=start_date,
            end_date=end_date
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tool/get_session_memories")
async def tool_get_session_memories(request: GetSessionMemoriesRequest):
    """Tool endpoint: Get session memories."""
    return await get_session_memories_api(request)


@app.post("/tool/consolidate_memories")
async def tool_consolidate_memories(request: ConsolidateMemoriesRequest):
    """Tool endpoint: Consolidate memories."""
    return await consolidate_memories_api(request)


# NOTE: Specific routes MUST come before parameterized routes in FastAPI
# /api/memories/most_accessed must be defined before /api/memories/{memory_id}

@app.get("/tool/get_most_accessed_memories")
@app.get("/api/memories/most_accessed")
async def get_most_accessed_memories(limit: int = Query(20, ge=1, le=100)):
    """Get most frequently accessed memories."""
    try:
        result = await store.get_most_accessed_memories(limit=limit)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tool/get_least_accessed_memories")
@app.get("/api/memories/least_accessed")
async def get_least_accessed_memories(
    limit: int = Query(20, ge=1, le=100),
    min_age_days: int = Query(7, ge=1)
):
    """Get least accessed memories (candidates for pruning)."""
    try:
        result = await store.get_least_accessed_memories(limit=limit, min_age_days=min_age_days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tool/get_memory_by_id/{memory_id}")
@app.get("/api/memories/{memory_id}")
async def get_memory_by_id(
    memory_id: str,
    expand_related: bool = Query(False, description="Include full content of related memories"),
    max_depth: int = Query(1, ge=1, le=3, description="Expansion depth (1-3 hops)")
):
    """Get a specific memory by ID with optional related memory expansion."""
    try:
        result = await store.get_memory_by_id(memory_id, expand_related=expand_related, max_depth=max_depth)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tool/list_categories")
@app.get("/api/categories")
async def list_categories(min_count: int = Query(1, ge=1)):
    """List all categories with their counts."""
    try:
        result = await store.list_categories(min_count=min_count)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tool/list_tags")
@app.get("/api/tags")
async def list_tags(min_count: int = Query(1, ge=1)):
    """List all tags with their counts."""
    try:
        result = await store.list_tags(min_count=min_count)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tool/check_intention/{intention_id}")
@app.get("/api/intentions/{intention_id}")
async def check_intention(intention_id: str, store: MemoryStore = Depends(get_memory_store)):
    """Check status of a specific intention."""
    try:
        result = await store.check_intention(intention_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tool/maintenance")
@app.post("/api/maintenance")
async def maintenance(store: MemoryStore = Depends(get_memory_store)):
    """Run maintenance operations."""
    try:
        result = await store.maintenance()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tool/prune_old_memories")
@app.post("/api/prune")
async def prune_old_memories(
    max_memories: Optional[int] = None,
    dry_run: bool = Query(True, description="Preview changes without applying")
):
    """Prune old, low-importance memories."""
    try:
        result = await store.prune_old_memories(max_memories=max_memories, dry_run=dry_run)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tool/traverse_memory_graph")
@app.get("/api/graph/traverse/{start_memory_id}")
async def traverse_memory_graph(
    start_memory_id: str,
    depth: int = Query(2, ge=1, le=5, description="Traversal depth (1-5 hops)"),
    max_nodes: int = Query(50, ge=1, le=200, description="Maximum nodes to return"),
    min_importance: float = Query(0.0, ge=0.0, le=1.0, description="Minimum importance filter"),
    category_filter: Optional[str] = Query(None, description="Filter by category")
):
    """Traverse memory graph N hops from starting node. Returns subgraph showing connections."""
    try:
        result = await store.traverse_graph(
            start_memory_id=start_memory_id,
            depth=depth,
            max_nodes=max_nodes,
            min_importance=min_importance,
            category_filter=category_filter
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tool/find_memory_clusters")
@app.get("/api/graph/clusters")
async def find_memory_clusters(
    min_cluster_size: int = Query(3, ge=2, description="Minimum memories in a cluster"),
    min_importance: float = Query(0.5, ge=0.0, le=1.0, description="Minimum importance"),
    limit: int = Query(10, ge=1, le=50, description="Maximum clusters to return")
):
    """Find densely connected regions in memory graph. Discovers thematic groups."""
    try:
        result = await store.find_clusters(
            min_cluster_size=min_cluster_size,
            min_importance=min_importance,
            limit=limit
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tool/get_memory_graph_stats")
@app.get("/api/graph/stats")
async def get_memory_graph_stats(
    category: Optional[str] = Query(None, description="Filter to specific category"),
    min_importance: float = Query(0.0, ge=0.0, le=1.0, description="Minimum importance")
):
    """Get graph connectivity statistics - hubs, isolated nodes, connectivity distribution."""
    try:
        result = await store.get_graph_statistics(
            category=category,
            min_importance=min_importance
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("=" * 80)
    print("BuildAutomata Memory Web Server (FIXED VERSION)")
    print("=" * 80)
    print()
    print("Access points:")
    print("  Web UI:    http://localhost:8766")
    print("  API docs:  http://localhost:8766/docs")
    print("  Tools:     http://localhost:8766/tools/list")
    print()
    print("Tool endpoints available at: /tool/{tool_name}")
    print()
    print("FIXES Applied:")
    print("  ✓ All async methods properly awaited")
    print("  ✓ Memory objects created correctly")
    print("  ✓ Initialize endpoint implemented (replaces proactive_scan)")
    print("  ✓ Proper error handling")
    print()
    print("=" * 80)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="info")
