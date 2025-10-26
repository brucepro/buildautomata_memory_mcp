#!/usr/bin/env python3
"""
Interactive Memory CLI for BuildAutomata Memory System
Provides direct CLI access to memory operations for Claude Code and other tools.

Usage:
    python interactive_memory.py store "content" --category general --importance 0.8 --tags tag1,tag2
    python interactive_memory.py search "query" --limit 5 --category general
    python interactive_memory.py update MEMORY_ID --content "new content" --importance 0.9
    python interactive_memory.py timeline --query "query" --limit 10
    python interactive_memory.py stats
    python interactive_memory.py prune --max 1000 --dry-run
    python interactive_memory.py maintenance
"""

import sys
import os
import asyncio
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add parent directory to path to import MemoryStore
sys.path.insert(0, str(Path(__file__).parent))

# Import from the MCP server
from buildautomata_memory_mcp import MemoryStore, Memory


class MemoryCLI:
    """CLI interface for memory operations"""

    def __init__(self, username: str = None, agent_name: str = None):
        self.username = username or os.getenv("BA_USERNAME", "buildautomata_ai_v012")
        self.agent_name = agent_name or os.getenv("BA_AGENT_NAME", "claude_assistant")
        self.memory_store = MemoryStore(self.username, self.agent_name)

    async def store(self, content: str, category: str = "general",
                   importance: float = 0.5, tags: List[str] = None,
                   metadata: dict = None) -> dict:
        """Store a new memory"""
        import uuid as uuid_lib
        memory = Memory(
            id=str(uuid_lib.uuid4()),  # Generate UUID
            content=content,
            category=category,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        success, warnings = await self.memory_store.store_memory(memory)

        result = {
            "success": success,
            "memory_id": memory.id,
            "warnings": warnings,
            "content": content
        }

        return result

    async def search(self, query: str, limit: int = 5, category: str = None,
                    min_importance: float = 0.0, created_after: str = None,
                    created_before: str = None, updated_after: str = None,
                    updated_before: str = None) -> dict:
        """Search for memories"""
        results = await self.memory_store.search_memories(
            query=query,
            limit=limit,
            category=category,
            min_importance=min_importance,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before
        )

        # Results are already dictionaries from the store
        return {
            "query": query,
            "count": len(results),
            "results": results
        }

    async def update(self, memory_id: str, content: str = None,
                    category: str = None, importance: float = None,
                    tags: List[str] = None, metadata: dict = None) -> dict:
        """Update an existing memory"""
        updates = {}
        if content is not None:
            updates["content"] = content
        if category is not None:
            updates["category"] = category
        if importance is not None:
            updates["importance"] = importance
        if tags is not None:
            updates["tags"] = tags
        if metadata is not None:
            updates["metadata"] = metadata

        success, changes, warnings = await self.memory_store.update_memory(
            memory_id=memory_id,
            **updates
        )

        return {
            "success": success,
            "memory_id": memory_id,
            "changes": changes,
            "warnings": warnings
        }

    async def timeline(self, query: str = None, memory_id: str = None,
                      limit: int = 10, start_date: str = None,
                      end_date: str = None, show_all_memories: bool = False,
                      include_diffs: bool = True, include_patterns: bool = True) -> dict:
        """Get memory timeline"""
        result = await self.memory_store.get_memory_timeline(
            query=query,
            memory_id=memory_id,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            show_all_memories=show_all_memories,
            include_diffs=include_diffs,
            include_patterns=include_patterns
        )

        # Return comprehensive result
        return result

    async def stats(self) -> dict:
        """Get memory statistics"""
        stats = self.memory_store.get_statistics()
        return stats

    async def prune(self, max_memories: int = None, dry_run: bool = False) -> dict:
        """Prune old memories"""
        result = await self.memory_store.prune_old_memories(
            max_memories=max_memories,
            dry_run=dry_run
        )
        return result

    async def maintenance(self) -> dict:
        """Run maintenance"""
        result = await self.memory_store.maintenance()
        return result

    async def init(self) -> dict:
        """Initialize agent - load context, intentions, continuity"""
        result = await self.memory_store.proactive_initialization_scan()
        return result

    async def get_by_id(self, memory_id: str) -> dict:
        """Get a specific memory by ID"""
        result = await self.memory_store.get_memory_by_id(memory_id)
        return result

    async def list_categories(self, min_count: int = 1) -> dict:
        """List all categories with counts"""
        result = await self.memory_store.list_categories(min_count=min_count)
        return result

    async def list_tags(self, min_count: int = 1) -> dict:
        """List all tags with usage counts"""
        result = await self.memory_store.list_tags(min_count=min_count)
        return result

    async def get_most_accessed(self, limit: int = 20) -> dict:
        """Get most accessed memories with tag cloud"""
        result_json = await self.memory_store.get_most_accessed_memories(limit=limit)
        result = json.loads(result_json)
        return result

    async def get_least_accessed(self, limit: int = 20, min_age_days: int = 7) -> dict:
        """Get least accessed memories - dead weight and buried treasure"""
        result_json = await self.memory_store.get_least_accessed_memories(limit=limit, min_age_days=min_age_days)
        result = json.loads(result_json)
        return result

    async def shutdown(self):
        """Shutdown the store"""
        await self.memory_store.shutdown()


def parse_tags(tags_str: str) -> List[str]:
    """Parse comma-separated tags"""
    if not tags_str:
        return []
    return [tag.strip() for tag in tags_str.split(",") if tag.strip()]


def parse_metadata(metadata_str: str) -> dict:
    """Parse JSON metadata"""
    if not metadata_str:
        return {}
    try:
        return json.loads(metadata_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing metadata JSON: {e}", file=sys.stderr)
        return {}


async def main():
    parser = argparse.ArgumentParser(
        description="Interactive Memory CLI for BuildAutomata Memory System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Store a memory:
    python interactive_memory.py store "User prefers dark mode" --category user_preference --importance 0.8 --tags "ui,settings"

  Search memories:
    python interactive_memory.py search "dark mode" --limit 5

  Update a memory:
    python interactive_memory.py update abc123 --content "New content" --importance 0.9

  Get timeline:
    python interactive_memory.py timeline --query "settings" --limit 10

  Get statistics:
    python interactive_memory.py stats

  Prune memories:
    python interactive_memory.py prune --max 1000 --dry-run

  Run maintenance:
    python interactive_memory.py maintenance
        """
    )

    parser.add_argument("--username", help="Username (overrides BA_USERNAME env var)")
    parser.add_argument("--agent", help="Agent name (overrides BA_AGENT_NAME env var)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Init command
    subparsers.add_parser("init", help="Initialize agent (load context, intentions, continuity)")

    # Store command
    store_parser = subparsers.add_parser("store", help="Store a new memory")
    store_parser.add_argument("content", help="Memory content")
    store_parser.add_argument("--category", default="general", help="Category (default: general)")
    store_parser.add_argument("--importance", type=float, default=0.5, help="Importance 0.0-1.0 (default: 0.5)")
    store_parser.add_argument("--tags", help="Comma-separated tags")
    store_parser.add_argument("--metadata", help="JSON metadata")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Max results (default: 5)")
    search_parser.add_argument("--category", help="Filter by category")
    search_parser.add_argument("--min-importance", type=float, default=0.0, help="Min importance (default: 0.0)")
    search_parser.add_argument("--created-after", help="Filter created after (ISO format)")
    search_parser.add_argument("--created-before", help="Filter created before (ISO format)")
    search_parser.add_argument("--updated-after", help="Filter updated after (ISO format)")
    search_parser.add_argument("--updated-before", help="Filter updated before (ISO format)")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update a memory")
    update_parser.add_argument("memory_id", help="Memory ID to update")
    update_parser.add_argument("--content", help="New content")
    update_parser.add_argument("--category", help="New category")
    update_parser.add_argument("--importance", type=float, help="New importance")
    update_parser.add_argument("--tags", help="New tags (comma-separated)")
    update_parser.add_argument("--metadata", help="New metadata (JSON)")

    # Timeline command
    timeline_parser = subparsers.add_parser("timeline", help="Get comprehensive memory timeline")
    timeline_parser.add_argument("--query", help="Search query to find memories")
    timeline_parser.add_argument("--memory-id", help="Specific memory ID")
    timeline_parser.add_argument("--limit", type=int, default=20, help="Max memories (default: 20)")
    timeline_parser.add_argument("--start-date", help="Filter events after this date (ISO format)")
    timeline_parser.add_argument("--end-date", help="Filter events before this date (ISO format)")
    timeline_parser.add_argument("--show-all", action="store_true", help="Show ALL memories chronologically")
    timeline_parser.add_argument("--no-diffs", action="store_true", help="Exclude text diffs")
    timeline_parser.add_argument("--no-patterns", action="store_true", help="Exclude pattern analysis")

    # Stats command
    subparsers.add_parser("stats", help="Get memory statistics")

    # Get by ID command
    get_parser = subparsers.add_parser("get", help="Get a specific memory by ID")
    get_parser.add_argument("memory_id", help="Memory ID (UUID)")

    # List categories command
    categories_parser = subparsers.add_parser("categories", help="List all categories with counts")
    categories_parser.add_argument("--min-count", type=int, default=1, help="Minimum memory count (default: 1)")

    # List tags command
    tags_parser = subparsers.add_parser("tags", help="List all tags with usage counts")
    tags_parser.add_argument("--min-count", type=int, default=1, help="Minimum usage count (default: 1)")

    # Prune command
    prune_parser = subparsers.add_parser("prune", help="Prune old memories")
    prune_parser.add_argument("--max", type=int, help="Max memories to keep")
    prune_parser.add_argument("--dry-run", action="store_true", help="Show what would be pruned")

    # Maintenance command
    subparsers.add_parser("maintenance", help="Run maintenance")

    # Access patterns
    accessed_parser = subparsers.add_parser("accessed", help="Get most accessed memories with tag cloud")
    accessed_parser.add_argument("--limit", type=int, default=20, help="Number of memories to show (default: 20)")

    least_accessed_parser = subparsers.add_parser("least-accessed", help="Get least accessed memories - dead weight and buried treasure")
    least_accessed_parser.add_argument("--limit", type=int, default=20, help="Number of memories to show (default: 20)")
    least_accessed_parser.add_argument("--min-age-days", type=int, default=7, help="Minimum age in days (default: 7)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize CLI
    cli = MemoryCLI(username=args.username, agent_name=args.agent)

    try:
        result = None

        if args.command == "init":
            result = await cli.init()

        elif args.command == "store":
            result = await cli.store(
                content=args.content,
                category=args.category,
                importance=args.importance,
                tags=parse_tags(args.tags),
                metadata=parse_metadata(args.metadata)
            )

        elif args.command == "search":
            result = await cli.search(
                query=args.query,
                limit=args.limit,
                category=args.category,
                min_importance=args.min_importance,
                created_after=args.created_after,
                created_before=args.created_before,
                updated_after=args.updated_after,
                updated_before=args.updated_before
            )

        elif args.command == "update":
            result = await cli.update(
                memory_id=args.memory_id,
                content=args.content,
                category=args.category,
                importance=args.importance,
                tags=parse_tags(args.tags) if args.tags else None,
                metadata=parse_metadata(args.metadata) if args.metadata else None
            )

        elif args.command == "timeline":
            result = await cli.timeline(
                query=args.query,
                memory_id=args.memory_id,
                limit=args.limit,
                start_date=args.start_date,
                end_date=args.end_date,
                show_all_memories=args.show_all,
                include_diffs=not args.no_diffs,
                include_patterns=not args.no_patterns
            )

        elif args.command == "stats":
            result = await cli.stats()

        elif args.command == "get":
            result = await cli.get_by_id(memory_id=args.memory_id)

        elif args.command == "categories":
            result = await cli.list_categories(min_count=args.min_count)

        elif args.command == "tags":
            result = await cli.list_tags(min_count=args.min_count)

        elif args.command == "prune":
            result = await cli.prune(
                max_memories=args.max,
                dry_run=args.dry_run
            )

        elif args.command == "maintenance":
            result = await cli.maintenance()

        elif args.command == "accessed":
            result = await cli.get_most_accessed(limit=args.limit)

        elif args.command == "least-accessed":
            result = await cli.get_least_accessed(limit=args.limit, min_age_days=args.min_age_days)

        # Output result
        if args.json or args.pretty:
            indent = 2 if args.pretty else None
            print(json.dumps(result, indent=indent, default=str))
        else:
            # Human-readable output
            if args.command == "init":
                print("=" * 80)
                print("AGENT INITIALIZATION")
                print("=" * 80)
                print()

                # Continuity check
                continuity = result.get("continuity_check", {})
                if continuity:
                    print("SESSION CONTINUITY:")
                    print(f"  Last activity: {continuity.get('last_activity', 'Never')}")
                    print(f"  Time gap: {continuity.get('time_gap_hours', 0):.1f} hours")
                    print(f"  New session: {continuity.get('is_new_session', False)}")
                    print()

                # Active intentions
                intentions = result.get("active_intentions", [])
                if intentions:
                    print(f"ACTIVE INTENTIONS ({len(intentions)}):")
                    for i, intention in enumerate(intentions, 1):
                        status_emoji = {"active": "*", "pending": "-", "completed": "✓", "cancelled": "X"}.get(intention.get('status'), '?')
                        print(f"  {i}. [{intention.get('priority', 0):.2f}] {status_emoji} {intention.get('description', 'No description')}")
                    print()
                else:
                    print("ACTIVE INTENTIONS: None")
                    print()

                # Urgent items
                urgent = result.get("urgent_items", [])
                if urgent:
                    print(f"URGENT ITEMS ({len(urgent)}):")
                    for item in urgent:
                        print(f"  [!] {item.get('type', 'unknown')}: {item.get('description', '')}...")
                    print()

                # Context summary
                context = result.get("context_summary", {})
                if context:
                    print("CONTEXT SUMMARY:")
                    print(f"  Active intentions: {context.get('active_intention_count', 0)}")
                    print(f"  Urgent items: {context.get('urgent_count', 0)}")
                    recent = context.get("recent_memories", [])
                    if recent:
                        print(f"  Recent memories: {len(recent)}")
                        for mem in recent[:3]:
                            content = mem.get('content', '').encode('ascii', 'replace').decode('ascii')
                            print(f"    - [{mem.get('category', 'unknown')}] {content}...")
                    print()

                # Performance
                scan_time = result.get("scan_duration_ms", 0)
                print(f"Scan completed in {scan_time:.1f}ms")
                print("=" * 80)

            elif args.command == "store":
                print(f"[OK] Memory stored successfully")
                print(f"  ID: {result['memory_id']}")
                content = result['content'].encode('ascii', 'replace').decode('ascii')
                print(f"  Content: {content}")
                if result.get('warnings'):
                    print(f"  Warnings: {', '.join(result['warnings'])}")

            elif args.command == "search":
                print(f"Found {result['count']} memories for query: '{result['query']}'")
                print()
                for i, mem in enumerate(result['results'], 1):
                    mem_id = mem.get('memory_id', mem.get('id', 'unknown'))
                    category = mem.get('category', 'general')
                    importance = mem.get('importance', 0.0)
                    current_imp = mem.get('current_importance', importance)
                    content = mem.get('content', '')
                    tags = mem.get('tags', [])
                    created_at = mem.get('created_at', '')
                    version_count = mem.get('version_count', 1)
                    access_count = mem.get('access_count', 0)

                    print(f"{i}. [{mem_id}...] {category} (importance: {importance:.2f}, current: {current_imp:.2f})")
                    content_safe = content.encode('ascii', 'replace').decode('ascii')
                    print(f"   {content_safe}")
                    print(f"   Tags: {', '.join(tags) if tags else 'none'}")
                    print(f"   Created: {created_at}, Versions: {version_count}, Accessed: {access_count}x")

                    # Show version history summary if available
                    if mem.get('version_history'):
                        vh = mem['version_history']
                        print(f"   Version History: {vh['total_versions']} versions, last updated {vh['last_updated']}")
                    print()

            elif args.command == "update":
                print(f"[OK] Memory updated successfully")
                print(f"  ID: {result['memory_id']}")
                print(f"  Changes: {', '.join(result['changes'])}")
                if result.get('warnings'):
                    print(f"  Warnings: {', '.join(result['warnings'])}")

            elif args.command == "timeline":
                if result.get("error"):
                    print(f"Error: {result['error']}")
                else:
                    print("=" * 80)
                    print("MEMORY TIMELINE - Biographical Narrative")
                    print("=" * 80)
                    print()

                    # Narrative summary
                    if result.get("narrative_arc"):
                        print("NARRATIVE SUMMARY:")
                        print("-" * 80)
                        print(result["narrative_arc"])
                        print()

                    # Statistics
                    print(f"Total Events: {result.get('total_events', 0)}")
                    print(f"Memories Tracked: {result.get('memories_tracked', 0)}")
                    print()

                    # Temporal patterns
                    patterns = result.get("temporal_patterns", {})
                    if patterns:
                        print("TEMPORAL PATTERNS:")
                        print(f"  Duration: {patterns.get('total_duration_days', 0)} days")
                        print(f"  Avg Events/Day: {patterns.get('avg_events_per_day', 0)}")

                        bursts = patterns.get("bursts", [])
                        if bursts:
                            print(f"\n  BURSTS ({len(bursts)} detected):")
                            for i, burst in enumerate(bursts, 1):
                                print(f"    {i}. {burst['event_count']} events in {burst['duration_hours']}h")

                        gaps = patterns.get("gaps", [])
                        if gaps:
                            print(f"\n  GAPS ({len(gaps)} detected):")
                            for i, gap in enumerate(gaps, 1):
                                print(f"    {i}. {gap['duration_days']} days of silence")

                    print()
                    print("=" * 80)
                    print(f"EVENTS (showing {len(result.get('events', []))} events)")
                    print("=" * 80)

                    # Show abbreviated event list
                    for i, event in enumerate(result.get("events", [])[:20], 1):  # Limit to first 20
                        # Handle datetime objects from SQLite converter
                        timestamp = event['timestamp']
                        if hasattr(timestamp, 'isoformat'):
                            timestamp_str = timestamp.isoformat()[:19]
                        else:
                            timestamp_str = str(timestamp)[:19]

                        print(f"[{i}] v{event['version']} - {timestamp_str}")
                        print(f"    {event['category']} | Importance: {event['importance']}")
                        print(f"    {event['content'][:100]}...")
                        if event.get('diff'):
                            print(f"    Changed: {event['diff']['change_magnitude']:.1%}")
                        print()

                    if len(result.get('events', [])) > 20:
                        print(f"... and {len(result['events']) - 20} more events")
                    print()

            elif args.command == "get":
                if "error" in result:
                    print(f"[ERROR] {result['error']}")
                else:
                    print("=" * 80)
                    print(f"MEMORY: {result['memory_id']}")
                    print("=" * 80)
                    print(f"Category: {result['category']}")
                    print(f"Importance: {result['importance']}")
                    print(f"Tags: {', '.join(result['tags']) if result['tags'] else 'none'}")
                    print(f"Created: {result['created_at']}")
                    print(f"Updated: {result['updated_at']}")
                    print(f"Last Accessed: {result['last_accessed']}")
                    print(f"Access Count: {result['access_count']}")
                    print(f"Versions: {result['version_count']}")
                    if "current_version" in result:
                        print(f"Current Version: {result['current_version']}")
                        print(f"Last Change: {result['last_change_type']}")
                        if result.get('last_change_description'):
                            print(f"Change Description: {result['last_change_description']}")
                    print("\nCONTENT:")
                    print("-" * 80)
                    content_safe = result['content'].encode('ascii', 'replace').decode('ascii')
                    print(content_safe)
                    if result.get('metadata'):
                        print("\nMETADATA:")
                        print("-" * 80)
                        print(json.dumps(result['metadata'], indent=2))

            elif args.command == "categories":
                if "error" in result:
                    print(f"[ERROR] {result['error']}")
                else:
                    print("=" * 80)
                    print("MEMORY CATEGORIES")
                    print("=" * 80)
                    print(f"Total Categories: {result['total_categories']}")
                    print(f"Total Memories: {result['total_memories']}")
                    print(f"Min Count Filter: {result['min_count_filter']}")
                    print("\nCATEGORIES (sorted by count):")
                    print("-" * 80)
                    for category, count in result['categories'].items():
                        print(f"{category:40} {count:>5} memories")

            elif args.command == "tags":
                if "error" in result:
                    print(f"[ERROR] {result['error']}")
                else:
                    print("=" * 80)
                    print("MEMORY TAGS")
                    print("=" * 80)
                    print(f"Total Unique Tags: {result['total_unique_tags']}")
                    print(f"Total Tag Usages: {result['total_tag_usages']}")
                    print(f"Min Count Filter: {result['min_count_filter']}")
                    print("\nTAGS (sorted by usage count):")
                    print("-" * 80)
                    for tag, count in result['tags'].items():
                        print(f"{tag:40} {count:>5} uses")

            elif args.command == "stats":
                print("Memory System Statistics:")
                print("=" * 60)
                print(f"Total memories: {result.get('total_memories', 0)}")
                print(f"Total versions: {result.get('total_versions', 0)}")
                print(f"Cache size: {result.get('cache_size', 0)}")
                print(f"Qdrant available: {result.get('qdrant_available', False)}")
                print(f"Embeddings available: {result.get('embeddings_available', False)}")
                if result.get('categories'):
                    print(f"\nCategories: {json.dumps(result['categories'], indent=2)}")
                if result.get('last_maintenance'):
                    print(f"\nLast maintenance: {result['last_maintenance']}")

            elif args.command == "prune":
                print("Pruning Results:")
                print("=" * 60)
                print(f"Dry run: {result.get('dry_run', False)}")
                print(f"Deleted count: {result.get('deleted_count', 0)}")
                print(f"Retained count: {result.get('retained_count', 0)}")
                if result.get('deleted_ids'):
                    print(f"Deleted IDs: {', '.join(result['deleted_ids'])}")

            elif args.command == "maintenance":
                print("Maintenance Results:")
                print("=" * 60)
                print(json.dumps(result, indent=2, default=str))

            elif args.command == "accessed":
                print("=" * 80)
                print("MOST ACCESSED MEMORIES - Behavioral Truth")
                print("=" * 80)
                print()
                print(result.get("interpretation", ""))
                print()

                memories = result.get("memories", [])
                print(f"TOP {len(memories)} MEMORIES BY ACCESS COUNT:")
                print("=" * 80)
                for i, mem in enumerate(memories, 1):
                    print(f"\n[{i}] Access Count: {mem['access_count']} | Importance: {mem['importance']}")
                    print(f"    Category: {mem['category']}")
                    print(f"    Last Accessed: {mem['last_accessed']}")
                    print(f"    Content: {mem['content_preview']}")

                print()
                print("=" * 80)
                print("TAG CLOUD (Top 30 from most accessed memories):")
                print("=" * 80)
                tag_cloud = result.get("tag_cloud", [])
                if tag_cloud:
                    # Display in columns
                    for i in range(0, len(tag_cloud), 3):
                        row = tag_cloud[i:i+3]
                        print("  " + " | ".join(f"{t['tag']} ({t['count']})" for t in row))
                else:
                    print("  No tags found")
                print()

            elif args.command == "least-accessed":
                print("=" * 80)
                print("LEAST ACCESSED MEMORIES - Dead Weight & Buried Treasure")
                print("=" * 80)
                print()
                print(result.get("interpretation", ""))
                print()
                print(f"Minimum age filter: {result.get('min_age_days')} days")
                print(f"Zero access count: {result.get('zero_access_count')} memories")
                print()

                memories = result.get("memories", [])
                print(f"BOTTOM {len(memories)} MEMORIES BY ACCESS COUNT:")
                print("=" * 80)
                for i, mem in enumerate(memories, 1):
                    print(f"\n[{i}] Access Count: {mem['access_count']} | Importance: {mem['importance']} | Age: {mem['age_days']} days")
                    print(f"    Category: {mem['category']}")
                    print(f"    Created: {mem['created_at']}")
                    print(f"    Last Accessed: {mem['last_accessed'] or 'Never'}")
                    print(f"    Content: {mem['content_preview']}")

                print()
                print("=" * 80)
                print("TAG CLOUD (Top 30 from least accessed memories):")
                print("=" * 80)
                tag_cloud = result.get("tag_cloud", [])
                if tag_cloud:
                    # Display in columns
                    for i in range(0, len(tag_cloud), 3):
                        row = tag_cloud[i:i+3]
                        print("  " + " | ".join(f"{t['tag']} ({t['count']})" for t in row))
                else:
                    print("  No tags found")
                print()

    finally:
        await cli.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
