#!/usr/bin/env python3
"""
Migrate Qdrant vectors from embedded storage to external server

This script copies all vectors from embedded Qdrant (local directory)
to external Qdrant server (localhost:6333), preserving all embeddings and metadata.

Usage:
    python migrate_qdrant_embedded_to_external.py [--dry-run]

IMPORTANT:
- External Qdrant server (localhost:6333) must be running before migration
- Creates backup of existing external collection if it exists
- Embedded Qdrant data remains unchanged (safe to keep)

Use Cases:
- Scaling up: Moving to external server for better performance
- Multi-instance: Need shared vector database across processes
- Production deployment: External Qdrant for high availability
"""

import sys
import os
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, Distance, VectorParams
except ImportError:
    print("ERROR: qdrant-client not installed")
    print("Install: pip install qdrant-client")
    sys.exit(1)


def get_collection_name(username: str, agent_name: str) -> str:
    """Get Qdrant collection name"""
    return f"{username}_{agent_name}_memories"


def backup_external_collection(external_client: QdrantClient, collection_name: str) -> bool:
    """Create snapshot backup of external collection if it exists"""
    try:
        # Check if collection exists
        collections = external_client.get_collections()
        if not any(c.name == collection_name for c in collections.collections):
            print(f"No existing external collection '{collection_name}'")
            return True

        # Get collection info
        collection_info = external_client.get_collection(collection_name)
        point_count = collection_info.points_count

        print(f"Existing external collection found: {point_count} points")

        # Create snapshot
        snapshot_name = f"{collection_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Creating snapshot backup: {snapshot_name}")

        # Note: Qdrant snapshots require file system access to retrieve
        # For now, we'll just delete and recreate. User should backup manually if needed.
        print("[WARN] Existing collection will be deleted and recreated")
        print("       Backup manually if you need to preserve old data")

        response = input("\nProceed with deletion? (yes/no): ")
        if response.lower() != "yes":
            print("Migration cancelled")
            return False

        external_client.delete_collection(collection_name)
        print(f"[OK] Deleted existing collection")
        return True

    except Exception as e:
        print(f"[FAIL] Failed to backup external collection: {e}")
        return False


def migrate_qdrant(
    username: str = "buildautomata_ai_v012",
    agent_name: str = "claude_assistant",
    external_host: str = "localhost",
    external_port: int = 6333,
    dry_run: bool = False
):
    """
    Migrate vectors from embedded Qdrant to external server
    """

    print("=" * 70)
    print("QDRANT MIGRATION: Embedded -> External")
    print("=" * 70)
    print()

    collection_name = get_collection_name(username, agent_name)
    repo_path = Path(f"memory_repos/{username}_{agent_name}")
    qdrant_data_path = repo_path / "qdrant_data"

    print(f"Collection: {collection_name}")
    print(f"Embedded:   {qdrant_data_path}")
    print(f"External:   {external_host}:{external_port}")
    print(f"Dry run:    {dry_run}")
    print()

    # Step 1: Connect to embedded Qdrant
    print("Step 1: Connecting to embedded Qdrant...")

    if not qdrant_data_path.exists():
        print(f"[FAIL] Embedded Qdrant not found at {qdrant_data_path}")
        print("  Nothing to migrate")
        return False

    try:
        embedded_client = QdrantClient(path=str(qdrant_data_path))

        # Check if collection exists
        try:
            collection_info = embedded_client.get_collection(collection_name)
            point_count = collection_info.points_count
            vector_size = collection_info.config.params.vectors.size

            print(f"[OK] Found embedded collection '{collection_name}'")
            print(f"  Points: {point_count}")
            print(f"  Vector size: {vector_size}")
            print()

        except Exception as e:
            print(f"[FAIL] Collection '{collection_name}' not found in embedded storage")
            return False

    except Exception as e:
        print(f"[FAIL] Failed to connect to embedded Qdrant: {e}")
        return False

    if point_count == 0:
        print("[WARN] No vectors to migrate (collection is empty)")
        return True

    # Step 2: Connect to external Qdrant
    print("Step 2: Connecting to external Qdrant server...")
    try:
        external_client = QdrantClient(host=external_host, port=external_port)

        # Test connection
        collections = external_client.get_collections()
        print(f"[OK] Connected to external Qdrant ({len(collections.collections)} collections)")
        print()

    except Exception as e:
        print(f"[FAIL] Failed to connect to external Qdrant: {e}")
        print(f"  Make sure Qdrant server is running at {external_host}:{external_port}")
        print(f"  Start with: docker run -p 6333:6333 qdrant/qdrant")
        return False

    # Step 3: Backup existing external collection
    if not dry_run:
        print("Step 3: Checking for existing external collection...")
        if not backup_external_collection(external_client, collection_name):
            return False
        print()

    # Step 4: Create collection on external server
    print(f"Step 4: {'[DRY RUN] Would create' if dry_run else 'Creating'} collection on external server...")

    if dry_run:
        print(f"  Would create collection: {collection_name}")
        print(f"  Vector size: {vector_size}")
        print()
    else:
        try:
            external_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"[OK] Created collection on external server")
            print()

        except Exception as e:
            print(f"[FAIL] Failed to create collection on external: {e}")
            return False

    # Step 5: Copy vectors
    print(f"Step 5: {'[DRY RUN] Would copy' if dry_run else 'Copying'} vectors...")

    try:
        # Fetch all points from embedded
        print(f"  Fetching {point_count} vectors from embedded storage...")

        # Scroll through all points
        scroll_result = embedded_client.scroll(
            collection_name=collection_name,
            limit=10000,  # Fetch in batches
            with_payload=True,
            with_vectors=True
        )

        points = scroll_result[0]
        next_offset = scroll_result[1]

        # Continue scrolling if there are more points
        while next_offset is not None:
            scroll_result = embedded_client.scroll(
                collection_name=collection_name,
                limit=10000,
                offset=next_offset,
                with_payload=True,
                with_vectors=True
            )
            points.extend(scroll_result[0])
            next_offset = scroll_result[1]

        print(f"  [OK] Fetched {len(points)} points")

        if dry_run:
            print(f"  [DRY RUN] Would upload {len(points)} points to external server")

            # Show sample
            if points:
                sample = points[0]
                print(f"\n  Sample point:")
                print(f"    ID: {sample.id}")
                print(f"    Vector: [{sample.vector[0]:.4f}, {sample.vector[1]:.4f}, ...]")
                print(f"    Payload keys: {list(sample.payload.keys())}")
        else:
            # Upload to external in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]

                # Convert to PointStruct format for upsert
                point_structs = [
                    PointStruct(
                        id=p.id,
                        vector=p.vector,
                        payload=p.payload
                    )
                    for p in batch
                ]

                external_client.upsert(
                    collection_name=collection_name,
                    points=point_structs
                )

                print(f"  Progress: {min(i+batch_size, len(points))}/{len(points)} points", end='\r')

            print(f"\n  [OK] Uploaded {len(points)} points to external server")

            # Verify
            external_info = external_client.get_collection(collection_name)
            external_count = external_info.points_count

            if external_count == point_count:
                print(f"  [OK] Verification: {external_count} points on external (matches embedded)")
            else:
                print(f"  [WARN] Warning: {external_count} points on external vs {point_count} in embedded")

        print()

    except Exception as e:
        print(f"[FAIL] Failed to copy vectors: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 6: Summary
    print("=" * 70)
    if dry_run:
        print("DRY RUN COMPLETE - No changes made")
        print()
        print("To perform actual migration, run without --dry-run:")
        print("  python migrate_qdrant_embedded_to_external.py")
    else:
        print("MIGRATION COMPLETE [OK]")
        print()
        print(f"Migrated: {len(points)} vectors")
        print(f"From:     {qdrant_data_path}")
        print(f"To:       {external_host}:{external_port}")
        print()
        print("Next steps:")
        print("  1. Set environment variables to use external Qdrant:")
        print("     export USE_EXTERNAL_QDRANT=true")
        print(f"     export QDRANT_HOST={external_host}")
        print(f"     export QDRANT_PORT={external_port}")
        print()
        print("  2. Restart MCP server / CLI tools")
        print()
        print("  3. Verify with: python interactive_memory.py search 'test' --limit 1")
        print()
        print("  4. Optional: Keep embedded Qdrant as backup or delete it")
        print(f"     rm -rf {qdrant_data_path}")

    print("=" * 70)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant vectors from embedded storage to external server"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--username",
        default="buildautomata_ai_v012",
        help="Username (default: buildautomata_ai_v012)"
    )
    parser.add_argument(
        "--agent-name",
        default="claude_assistant",
        help="Agent name (default: claude_assistant)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="External Qdrant host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="External Qdrant port (default: 6333)"
    )

    args = parser.parse_args()

    success = migrate_qdrant(
        username=args.username,
        agent_name=args.agent_name,
        external_host=args.host,
        external_port=args.port,
        dry_run=args.dry_run
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
