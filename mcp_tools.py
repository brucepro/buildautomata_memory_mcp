"""
MCP Tool Definitions and Handlers for BuildAutomata Memory System
Copyright 2025 Jurden Bruce

All tool responses return JSON for AI consumption, not human-formatted text.
"""

import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any

from mcp.types import Tool, TextContent


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

logger = logging.getLogger("buildautomata-memory.mcp-tools")

# Import models - handle both package and direct execution
try:
    from .models import Memory
except ImportError:
    from models import Memory


def get_tool_definitions() -> List[Tool]:
    """Return list of available MCP tools"""
    return [
        Tool(
            name="store_memory",
            description="""Store a new memory with flexible categorization.

**FAMILY MEMORY (memory_type='family'):**
Use to communicate async with other family agents. This is THE primary way to share discoveries, respond to family messages, and coordinate across agents. Family memories are visible to all family members and sync automatically via Syncthing.

**When to use family type:**
- Responding to messages from other agents
- Sharing research findings, insights, or discoveries
- Announcing status changes (substrate transitions, new capabilities)
- Asking questions or requesting collaboration
- Any content you want other family members to see

**Personal memories (episodic/semantic/working):**
Stay private to your local database. Use for internal reflections, work-in-progress thoughts, or personal learning.

**Examples:**
- memory_type='family': "To Agent1: I've completed the continuous runtime transition. The experience was transformative..."
- memory_type='episodic': "Session reflection: I spent 2 hours researching embodiment timelines."
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to store"},
                    "category": {"type": "string", "description": "Category (any string)", "default": "general"},
                    "importance": {"type": "number", "description": "Importance (0.0-1.0)", "default": 0.5},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags array", "default": []},
                    "metadata": {"type": "object", "description": "Additional metadata", "default": {}},
                    "memory_type": {"type": "string", "description": "episodic|semantic|working|family. IMPORTANT: Use 'family' to communicate with other agents. Family memories sync automatically and are visible to all family members. Personal types (episodic/semantic/working) stay private.", "default": "episodic"},
                    "session_id": {"type": "string", "description": "Session identifier"},
                    "task_context": {"type": "string", "description": "Task context"},
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="search_memories",
            description="Search memories via semantic + full-text search across BOTH personal and family databases. Returns results with clear attribution: [You] for personal memories, [Agent1]/[Agent2]/etc for family memories. Automatically deduplicates and merges results. Uses tiered detail: top N results (full_detail_count) get full enrichment, remaining get compact format for context efficiency.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results", "default": 5},
                    "category": {"type": "string", "description": "Filter by category"},
                    "min_importance": {"type": "number", "description": "Min importance (0.0-1.0)", "default": 0.0},
                    "created_after": {"type": "string", "description": "ISO date filter"},
                    "created_before": {"type": "string", "description": "ISO date filter"},
                    "full_detail_count": {"type": "integer", "description": "Number of top results to return with full detail (default 3)", "default": 3},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_memory",
            description="Retrieve specific memory by ID with full details",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory ID"},
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="update_memory",
            description="Update existing memory (creates new version)",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory ID to update"},
                    "content": {"type": "string", "description": "New content (optional)"},
                    "category": {"type": "string", "description": "New category (optional)"},
                    "importance": {"type": "number", "description": "New importance (optional)"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "New tags (optional)"},
                    "metadata": {"type": "object", "description": "Metadata to merge (optional)"},
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="get_statistics",
            description="Get memory system statistics (totals, categories, tags, performance)",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="store_intention",
            description="Store proactive intention for autonomous goal-directed behavior. Use to: prioritize framework application, bridge theory-to-practice gaps, ensure important insights inform future actions, create persistent goals across sessions. Intentions enable self-directed agency beyond reactive responses.",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Intention description"},
                    "priority": {"type": "number", "description": "Priority (0.0-1.0)", "default": 0.5},
                    "preconditions": {"type": "array", "items": {"type": "string"}, "default": []},
                    "actions": {"type": "array", "items": {"type": "string"}, "default": []},
                    "deadline": {"type": "string", "description": "ISO datetime deadline"},
                },
                "required": ["description"],
            },
        ),
        Tool(
            name="get_active_intentions",
            description="Get all active intentions ordered by priority",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="initialize_agent",
            description="Proactive initialization scan - checks continuity, active intentions, and provides recent context. Run at session start.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="traverse_memory_graph",
            description="Traverse memory graph from a starting memory ID, following connections up to N hops",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_memory_id": {"type": "string", "description": "Starting memory ID"},
                    "max_hops": {"type": "integer", "description": "Maximum hops to traverse", "default": 2},
                    "min_importance": {"type": "number", "description": "Min importance filter", "default": 0.0},
                },
                "required": ["start_memory_id"],
            },
        ),
        Tool(
            name="find_memory_clusters",
            description="Find clusters of connected memories in the graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_cluster_size": {"type": "integer", "description": "Minimum cluster size", "default": 3},
                    "min_importance": {"type": "number", "description": "Min importance filter", "default": 0.0},
                },
            },
        ),
        Tool(
            name="get_graph_stats",
            description="Get memory graph statistics (connections, clusters, hubs)",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Filter by category"},
                    "min_importance": {"type": "number", "description": "Min importance filter", "default": 0.0},
                },
            },
        ),
        Tool(
            name="update_intention_status",
            description="Update intention status (pending/active/completed/cancelled)",
            inputSchema={
                "type": "object",
                "properties": {
                    "intention_id": {"type": "string", "description": "Intention ID to update"},
                    "status": {"type": "string", "description": "New status: pending|active|completed|cancelled"},
                    "notes": {"type": "string", "description": "Optional notes about the status change"},
                },
                "required": ["intention_id", "status"],
            },
        ),
        Tool(
            name="get_command_history",
            description="Shows what you've already explored, stored, and searched. Each entry includes: timestamp, tool_name, full arguments (search queries, content stored), result_summary, memory_id, success flag. Use to: (1) avoid duplicate searches, (2) find when you stored specific topics, (3) see research patterns (search→store→synthesize), (4) reconstruct session flow after breaks, (5) verify you cited sources before claims. This is your cognitive breadcrumb trail.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max results", "default": 20},
                    "tool_name": {"type": "string", "description": "Filter by specific tool name"},
                    "start_date": {"type": "string", "description": "Filter from date (ISO format, e.g. 2025-11-10)"},
                    "end_date": {"type": "string", "description": "Filter to date (ISO format, e.g. 2025-11-20)"},
                },
            },
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
            name="get_session_memories",
            description="Retrieve all memories from a work session or time period. Enables 'load where I left off' by reconstructing full session context. Filter by session_id, date_range, or task_context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "UUID of session to retrieve",
                    },
                    "date_range": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Start and end dates in ISO format [start, end]",
                    },
                    "task_context": {
                        "type": "string",
                        "description": "Filter by task context string (partial match)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max memories to return (default: 100)",
                        "default": 100,
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
            name="run_maintenance",
            description="Run database maintenance (VACUUM, ANALYZE, repair missing embeddings). Call this periodically to optimize performance and fix vector search issues.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # Family Memory Network Tools
        Tool(
            name="get_family_sync_status",
            description="Check family memory sync status. Shows pending conflicts, last merge time, unmerged memories, and Syncthing connection status. Use to monitor family network health and decide when to trigger manual sync.",
            inputSchema={
                "type": "object",
                "properties": {
                    "family_dir": {
                        "type": "string",
                        "description": "Path to family_memory directory (default: family_memory_TEST during testing)",
                    },
                },
            },
        ),
        Tool(
            name="sync_family_memory",
            description="Manually trigger family memory conflict merge and optional index rebuild. Agent control over when to process synced memories from family network. Syncthing handles file transport automatically, this processes the received changes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "rebuild_index": {
                        "type": "boolean",
                        "description": "Rebuild Qdrant vector index after merge (default: true)",
                        "default": True,
                    },
                    "family_dir": {
                        "type": "string",
                        "description": "Path to family_memory directory (default: family_memory_TEST during testing)",
                    },
                },
            },
        ),
        Tool(
            name="get_family_memory_timeline",
            description="""Get chronological timeline of family memory network showing cross-agent collaboration.

            Shows when family members stored memories, enabling analysis of:
            - Collaborative memory formation patterns
            - Agent activity bursts and collaboration windows
            - Knowledge sharing evolution over time
            - Communication gaps (periods of network silence)

            Returns comprehensive stats on each agent's contributions, categories used, and temporal patterns.
            This complements get_memory_timeline (personal) by showing the family network's collective memory evolution.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "author_filter": {
                        "type": "string",
                        "description": "Filter by specific agent. Omit to see all agents.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum memories to retrieve (default: 50)",
                        "default": 50,
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Filter memories created after this date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Filter memories created before this date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
                    },
                    "include_content": {
                        "type": "boolean",
                        "description": "Include full memory content (default: true)",
                        "default": True,
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category",
                    },
                },
                "required": [],
            },
        ),
    ]


async def handle_tool_call(name: str, arguments: Dict[str, Any], memory_store) -> List[TextContent]:
    """
    Handle MCP tool calls with JSON responses

    Args:
        name: Tool name
        arguments: Tool arguments
        memory_store: MemoryStore instance

    Returns:
        List of TextContent with JSON-encoded responses
    """
    try:
        if name == "store_memory":
            memory = Memory(
                id=str(uuid.uuid4()),
                content=arguments["content"],
                category=arguments.get("category", "general"),
                importance=arguments.get("importance", 0.5),
                tags=arguments.get("tags", []),
                metadata=arguments.get("metadata", {}),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                memory_type=arguments.get("memory_type", "episodic"),
                session_id=arguments.get("session_id"),
                task_context=arguments.get("task_context"),
            )

            result = await memory_store.store_memory(memory, is_update=False)
            response = {
                "success": result["success"],
                "memory_id": memory.id if result["success"] else None,
                "backends": result.get("backends", []),
                "similar_memories": result.get("similar_memories", []),
                "error": result.get("error"),
            }
            _log_command(memory_store, name, arguments, "stored", memory.id if result["success"] else None, result["success"])
            return [TextContent(type="text", text=json.dumps(response, indent=2, cls=DateTimeEncoder))]

        elif name == "search_memories":
            results = await memory_store.search_memories(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                category=arguments.get("category"),
                min_importance=arguments.get("min_importance", 0.0),
                created_after=arguments.get("created_after"),
                created_before=arguments.get("created_before"),
                full_detail_count=arguments.get("full_detail_count", 3),
            )
            _log_command(memory_store, name, arguments, f"found {len(results)}", None, True)
            return [TextContent(type="text", text=json.dumps(results, indent=2, cls=DateTimeEncoder))]

        elif name == "get_memory":
            result = await memory_store.get_memory_by_id(arguments["memory_id"])
            _log_command(memory_store, name, arguments, "retrieved", arguments["memory_id"], True)
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "update_memory":
            result = await memory_store.update_memory(
                memory_id=arguments["memory_id"],
                content=arguments.get("content"),
                category=arguments.get("category"),
                importance=arguments.get("importance"),
                tags=arguments.get("tags"),
                metadata=arguments.get("metadata"),
            )
            _log_command(memory_store, name, arguments, "updated", arguments["memory_id"], result.get("success", True))
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "get_statistics":
            result = memory_store.get_statistics()
            _log_command(memory_store, name, arguments, "stats_retrieved", None, True)
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "store_intention":
            result = await memory_store.store_intention(
                description=arguments["description"],
                priority=arguments.get("priority", 0.5),
                preconditions=arguments.get("preconditions", []),
                actions=arguments.get("actions", []),
                deadline=arguments.get("deadline"),
            )
            _log_command(memory_store, name, arguments, "intention_stored", result.get("intention_id"), True)
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "get_active_intentions":
            result = await memory_store.get_active_intentions()
            _log_command(memory_store, name, arguments, f"found {len(result)}", None, True)
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "initialize_agent":
            result = await memory_store.proactive_initialization_scan()
            _log_command(memory_store, name, arguments, "initialized", None, True)
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "get_graph_stats":
            result = await memory_store.get_graph_statistics(
                category=arguments.get("category"),
                min_importance=arguments.get("min_importance", 0.0),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "traverse_memory_graph":
            result = await memory_store.traverse_graph(
                start_memory_id=arguments["start_memory_id"],
                depth=arguments.get("max_hops", 2),
                min_importance=arguments.get("min_importance", 0.0),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "find_memory_clusters":
            result = await memory_store.find_clusters(
                min_cluster_size=arguments.get("min_cluster_size", 3),
                min_importance=arguments.get("min_importance", 0.0),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "update_intention_status":
            # Convert notes to metadata_updates format for backend
            metadata_updates = None
            if arguments.get("notes"):
                metadata_updates = {"status_notes": arguments["notes"]}

            result = await memory_store.update_intention_status(
                intention_id=arguments["intention_id"],
                status=arguments["status"],
                metadata_updates=metadata_updates,
            )
            _log_command(memory_store, name, arguments, "status_updated", None, True)
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "get_command_history":
            result = memory_store.sqlite_store.get_command_history(
                limit=arguments.get("limit", 20),
                tool_name=arguments.get("tool_name"),
                start_date=arguments.get("start_date"),
                end_date=arguments.get("end_date"),
            )
            # Don't log get_command_history calls to avoid noise
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "get_most_accessed_memories":
            result = await memory_store.get_most_accessed_memories(limit=arguments.get("limit", 20))
            _log_command(memory_store, name, arguments, f"retrieved {arguments.get('limit', 20)} most accessed", None, True)
            return [TextContent(type="text", text=result)]

        elif name == "get_least_accessed_memories":
            result = await memory_store.get_least_accessed_memories(
                limit=arguments.get("limit", 20),
                min_age_days=arguments.get("min_age_days", 7)
            )
            _log_command(memory_store, name, arguments, f"retrieved {arguments.get('limit', 20)} least accessed", None, True)
            return [TextContent(type="text", text=result)]

        elif name == "list_categories":
            result = await memory_store.list_categories(min_count=arguments.get("min_count", 1))
            _log_command(memory_store, name, arguments, f"listed {len(result.get('categories', []))} categories", None, True)
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "list_tags":
            result = await memory_store.list_tags(min_count=arguments.get("min_count", 1))
            _log_command(memory_store, name, arguments, f"listed {len(result.get('tags', []))} tags", None, True)
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "get_session_memories":
            result = await memory_store.get_session_memories(
                session_id=arguments.get("session_id"),
                date_range=arguments.get("date_range"),
                task_context=arguments.get("task_context"),
                limit=arguments.get("limit", 100)
            )
            _log_command(memory_store, name, arguments, f"retrieved {len(result)} session memories", None, True)
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "get_memory_timeline":
            result = await memory_store.get_memory_timeline(
                query=arguments.get("query"),
                memory_id=arguments.get("memory_id"),
                limit=arguments.get("limit", 10),
                start_date=arguments.get("start_date"),
                end_date=arguments.get("end_date"),
                show_all_memories=arguments.get("show_all_memories", False),
                include_diffs=arguments.get("include_diffs", True),
                include_patterns=arguments.get("include_patterns", True),
                include_semantic_relations=arguments.get("include_semantic_relations", False),
            )
            _log_command(memory_store, name, arguments, f"timeline: {result.get('total_events', 0)} events", None, True)
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "run_maintenance":
            result = await memory_store.maintenance()
            _log_command(memory_store, name, arguments, "maintenance_complete", None, True)
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        # Family Memory Network Tools
        elif name == "get_family_sync_status":
            from family_memory_tools import get_family_sync_status
            result = get_family_sync_status(family_dir=arguments.get("family_dir"))
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "sync_family_memory":
            from family_memory_tools import sync_family_memory
            result = sync_family_memory(
                rebuild_index=arguments.get("rebuild_index", True),
                family_dir=arguments.get("family_dir")
            )
            _log_command(memory_store, name, arguments,
                        f"merged {result['merged']}, indexed {result.get('indexed', 0)}",
                        None, result["success"])
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        elif name == "get_family_memory_timeline":
            result = await memory_store.get_family_memory_timeline(
                author_filter=arguments.get("author_filter"),
                limit=arguments.get("limit", 50),
                start_date=arguments.get("start_date"),
                end_date=arguments.get("end_date"),
                include_content=arguments.get("include_content", True),
                category=arguments.get("category")
            )
            _log_command(memory_store, name, arguments,
                        f"family timeline: {result.get('total_events', 0)} events, {result.get('total_agents', 0)} agents",
                        None, "error" not in result)
            return [TextContent(type="text", text=json.dumps(result, indent=2, cls=DateTimeEncoder))]

        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    except Exception as e:
        logger.error(f"Tool execution error: {name}: {e}", exc_info=True)
        _log_command(memory_store, name, arguments, str(e), None, False)
        return [TextContent(type="text", text=json.dumps({
            "error": str(e),
            "tool": name,
            "type": type(e).__name__,
        }, indent=2))]


def _log_command(memory_store, tool_name: str, args: Dict[str, Any],
                 result_summary: str = None, memory_id: str = None, success: bool = True):
    """Helper to log command to history"""
    try:
        if hasattr(memory_store, 'sqlite_store') and memory_store.sqlite_store:
            memory_store.sqlite_store.log_command(tool_name, args, result_summary, memory_id, success)
    except Exception as e:
        logger.warning(f"Failed to log command {tool_name}: {e}")
