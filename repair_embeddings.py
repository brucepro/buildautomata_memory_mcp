#!/usr/bin/env python3
"""
Repair Missing Embeddings - Standalone Utility

Finds memories in SQLite that are missing vectors in Qdrant and regenerates them.
This handles cases where memories were stored while another process held the
embedded Qdrant lock (e.g., MCP server was running).

Run this separately when needed, not as part of regular maintenance.

Usage:
    python repair_embeddings.py [--dry-run]
"""

import os
import sys
import argparse
import asyncio
from datetime import datetime
from typing import Dict, Any

# Set BA_USERNAME and BA_AGENT_NAME defaults
os.environ.setdefault("BA_USERNAME", "buildautomata_ai_v012")
os.environ.setdefault("BA_AGENT_NAME", "claude_assistant")

from buildautomata_memory_mcp import MemoryStore


async def repair_missing_embeddings(store: MemoryStore, dry_run: bool = False) -> Dict[str, Any]:
    """
    Find memories in SQLite that are missing vectors in Qdrant and regenerate them.

    Args:
        store: MemoryStore instance
        dry_run: If True, only report what would be repaired without making changes

    Returns:
        Dict with repair statistics
    """
    if not store.db_conn:
        return {"error": "Database not available"}

    # Ensure Qdrant is initialized (may be lazy loaded)
    try:
        store._ensure_qdrant()
    except Exception as e:
        if "already accessed by another instance" in str(e):
            return {"error": "Qdrant locked by another instance (MCP server running?)",
                    "suggestion": "Stop the MCP server before running repair"}
        raise

    if not store.qdrant_client:
        return {"error": "Qdrant not available"}

    # Import when needed (after qdrant initialization)
    from qdrant_client.models import PointStruct

    try:
        # Get all memory IDs from SQLite
        with store._db_lock:
            cursor = store.db_conn.execute("SELECT id, content FROM memories")
            all_memories = cursor.fetchall()

        total_memories = len(all_memories)
        missing_count = 0
        repaired_count = 0
        failed_ids = []

        print(f"[EMBEDDING_REPAIR] Checking {total_memories} memories for missing vectors...")
        if dry_run:
            print("[DRY RUN MODE] No changes will be made")

        for idx, row in enumerate(all_memories, 1):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{total_memories} ({idx*100//total_memories}%)")

            memory_id = row["id"]
            content = row["content"]

            # Check if vector exists in Qdrant
            try:
                points = await asyncio.to_thread(
                    store.qdrant_client.retrieve,
                    collection_name=store.collection_name,
                    ids=[memory_id],
                    with_vectors=False,
                    with_payload=False
                )

                if not points:
                    # Missing vector
                    missing_count += 1
                    print(f"\n  [MISSING] {memory_id[:8]}... - {content[:60]}...")

                    if not dry_run:
                        # Generate embedding
                        embedding = await asyncio.to_thread(store.generate_embedding, content)

                        # Get full memory data for payload
                        memory_result = await store.get_memory_by_id(memory_id)
                        if "error" in memory_result:
                            failed_ids.append(memory_id)
                            print(f"    [FAILED] Could not retrieve memory data")
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
                            store.qdrant_client.upsert,
                            collection_name=store.collection_name,
                            points=[point]
                        )

                        repaired_count += 1
                        print(f"    [REPAIRED] Vector regenerated and stored")

            except Exception as e:
                print(f"  [ERROR] {memory_id[:8]}... - {str(e)}")
                failed_ids.append(memory_id)

        print(f"\n[EMBEDDING_REPAIR] Complete")
        print(f"  Total memories: {total_memories}")
        print(f"  Missing vectors: {missing_count}")

        if not dry_run:
            print(f"  Successfully repaired: {repaired_count}")
            print(f"  Failed: {len(failed_ids)}")
            if failed_ids:
                print(f"  Failed IDs: {', '.join(id[:8] + '...' for id in failed_ids[:5])}")
        else:
            print(f"  Would repair: {missing_count}")

        return {
            "total_memories": total_memories,
            "missing_count": missing_count,
            "repaired_count": repaired_count if not dry_run else 0,
            "failed_count": len(failed_ids),
            "failed_ids": failed_ids,
            "dry_run": dry_run
        }

    except Exception as e:
        return {"error": str(e)}


async def main():
    parser = argparse.ArgumentParser(
        description="Repair missing embeddings in Qdrant vector database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be repaired without making changes"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EMBEDDING REPAIR UTILITY")
    print("=" * 60)
    print(f"Username: {os.getenv('BA_USERNAME')}")
    print(f"Agent: {os.getenv('BA_AGENT_NAME')}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'REPAIR'}")
    print("=" * 60)
    print()

    # Initialize store
    print("Initializing memory store...")
    store = MemoryStore()

    # Run repair
    result = await repair_missing_embeddings(store, dry_run=args.dry_run)

    print()
    print("=" * 60)
    print("RESULT:")
    for key, value in result.items():
        if key != "failed_ids":
            print(f"  {key}: {value}")
    print("=" * 60)

    return 0 if "error" not in result else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
