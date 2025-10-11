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
                      limit: int = 10) -> dict:
        """Get memory timeline"""
        timeline = await self.memory_store.get_memory_timeline(
            query=query,
            memory_id=memory_id,
            limit=limit
        )

        # Format timeline
        formatted_timeline = []
        for mem_timeline in timeline:
            changes = []
            for change in mem_timeline["changes"]:
                changes.append({
                    "version": change["version"],
                    "timestamp": change["timestamp"],
                    "change_type": change["change_type"],
                    "change_description": change["change_description"],
                    "category": change["category"],
                    "importance": change["importance"],
                    "content_preview": change["content"],
                    "tags": change["tags"],
                    "what_changed": change.get("what_changed", [])
                })

            formatted_timeline.append({
                "memory_id": mem_timeline["memory_id"],
                "total_versions": len(changes),
                "changes": changes
            })

        return {
            "count": len(formatted_timeline),
            "timeline": formatted_timeline
        }

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
    timeline_parser = subparsers.add_parser("timeline", help="Get memory timeline")
    timeline_parser.add_argument("--query", help="Search query to find memories")
    timeline_parser.add_argument("--memory-id", help="Specific memory ID")
    timeline_parser.add_argument("--limit", type=int, default=10, help="Max memories (default: 10)")

    # Stats command
    subparsers.add_parser("stats", help="Get memory statistics")

    # Prune command
    prune_parser = subparsers.add_parser("prune", help="Prune old memories")
    prune_parser.add_argument("--max", type=int, help="Max memories to keep")
    prune_parser.add_argument("--dry-run", action="store_true", help="Show what would be pruned")

    # Maintenance command
    subparsers.add_parser("maintenance", help="Run maintenance")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize CLI
    cli = MemoryCLI(username=args.username, agent_name=args.agent)

    try:
        result = None

        if args.command == "store":
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
                limit=args.limit
            )

        elif args.command == "stats":
            result = await cli.stats()

        elif args.command == "prune":
            result = await cli.prune(
                max_memories=args.max,
                dry_run=args.dry_run
            )

        elif args.command == "maintenance":
            result = await cli.maintenance()

        # Output result
        if args.json or args.pretty:
            indent = 2 if args.pretty else None
            print(json.dumps(result, indent=indent, default=str))
        else:
            # Human-readable output
            if args.command == "store":
                print(f"[OK] Memory stored successfully")
                print(f"  ID: {result['memory_id']}")
                print(f"  Content: {result['content']}")
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
                    print(f"   {content[]}")
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
                print(f"Timeline for {result['count']} memories:")
                print()
                for mem_timeline in result['timeline']:
                    print(f"Memory: {mem_timeline['memory_id']}")
                    print(f"Total versions: {mem_timeline['total_versions']}")
                    print("=" * 60)
                    for change in mem_timeline['changes']:
                        print(f"  Version {change['version']} - {change['timestamp']}")
                        print(f"  Type: {change['change_type']} | {change['change_description']}")
                        print(f"  Category: {change['category']} | Importance: {change['importance']}")
                        if change.get('what_changed'):
                            print(f"  Changed: {', '.join(change['what_changed'])}")
                        print(f"  Content: {change['content_preview']}")
                        print(f"  Tags: {', '.join(change['tags'])}")
                        print("-" * 60)
                    print()

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

    finally:
        await cli.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
