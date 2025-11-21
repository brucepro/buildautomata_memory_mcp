#!/usr/bin/env python3
"""
Migrate Qdrant vectors from external server to embedded storage

This script copies all vectors from external Qdrant (localhost:6333)
to embedded Qdrant (local directory), preserving all embeddings and metadata.

Usage:
    python migrate_qdrant_external_to_embedded.py [--dry-run]

IMPORTANT:
- Stop web server before running (to avoid lock conflicts)
- External Qdrant server (localhost:6333) must be running
- Creates backup before migration
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
    from qdrant_client.models import PointStruct
except ImportError:
    print("ERROR: qdrant-client not installed")
    print("Install: pip install qdrant-client")
    sys.exit(1)


def get_collection_name(username: str, agent_name: str) -> str:
    """Get Qdrant collection name"""
    return f"{username}_{agent_name}_memories"


def backup_embedded_qdrant(qdrant_data_path: Path) -> Path:
    """Create backup of embedded Qdrant if it exists"""
    if not qdrant_data_path.exists():
        print(f"No existing embedded Qdrant at {qdrant_data_path}")
        return None

    backup_path = qdrant_data_path.parent / f"qdrant_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Creating backup: {backup_path}")

    import shutil
    shutil.copytree(qdrant_data_path, backup_path)
    print(f"[OK] Backup created: {backup_path}")
    return backup_path


def migrate_qdrant(
    username: str = "buildautomata_ai_v012",
    agent_name: str = "claude_assistant",
    external_host: str = "localhost",
    external_port: int = 6333,
    dry_run: bool = False
):
    """
    Migrate vectors from external Qdrant to embedded storage
    """

    print("=" * 70)
    print("QDRANT MIGRATION: External -> Embedded")
    print("=" * 70)
    print()

    collection_name = get_collection_name(username, agent_name)
    repo_path = Path(f"memory_repos/{username}_{agent_name}")
    qdrant_data_path = repo_path / "qdrant_data"

    print(f"Collection: {collection_name}")
    print(f"External:   {external_host}:{external_port}")
    print(f"Embedded:   {qdrant_data_path}")
    print(f"Dry run:    {dry_run}")
    print()

    # Step 1: Connect to external Qdrant
    print("Step 1: Connecting to external Qdrant...")
    try:
        external_client = QdrantClient(host=external_host, port=external_port)

        # Test connection
        collections = external_client.get_collections()
        print(f"[OK] Connected to external Qdrant ({len(collections.collections)} collections)")

        # Check if collection exists
        collection_exists = any(c.name == collection_name for c in collections.collections)
        if not collection_exists:
            print(f"[FAIL] Collection '{collection_name}' not found on external server")
            print(f"  Available collections: {[c.name for c in collections.collections]}")
            return False

        # Get collection info
        collection_info = external_client.get_collection(collection_name)
        point_count = collection_info.points_count
        vector_size = collection_info.config.params.vectors.size

        print(f"[OK] Collection '{collection_name}' found")
        print(f"  Points: {point_count}")
        print(f"  Vector size: {vector_size}")
        print()

    except Exception as e:
        print(f"[FAIL] Failed to connect to external Qdrant: {e}")
        print(f"  Make sure Qdrant server is running at {external_host}:{external_port}")
        return False

    if point_count == 0:
        print("[WARN] No vectors to migrate (collection is empty)")
        return True

    # Step 2: Backup existing embedded Qdrant
    if not dry_run:
        print("Step 2: Backing up existing embedded Qdrant...")
        backup_path = backup_embedded_qdrant(qdrant_data_path)
        if backup_path:
            print()

    # Step 3: Initialize embedded Qdrant
    print(f"Step 3: {'[DRY RUN] Would initialize' if dry_run else 'Initializing'} embedded Qdrant...")

    if dry_run:
        print(f"  Would create: {qdrant_data_path}")
        print(f"  Vector size: {vector_size}")
        print()
    else:
        try:
            # Import here to avoid conflicts
            from qdrant_client import QdrantClient as EmbeddedClient

            # Create embedded client
            embedded_client = EmbeddedClient(path=str(qdrant_data_path))

            # Check if collection exists
            try:
                existing_info = embedded_client.get_collection(collection_name)
                print(f"[WARN] Collection '{collection_name}' already exists in embedded")
                print(f"  Existing vectors: {existing_info.points_count}")
                print(f"  Will delete and recreate with {point_count} vectors from external")

                embedded_client.delete_collection(collection_name)
                print("  [OK] Deleted existing collection")
            except:
                print(f"  [OK] No existing collection, will create fresh")
                pass  # Collection doesn't exist, that's fine

            # Create collection with same config
            from qdrant_client.models import Distance, VectorParams

            embedded_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"[OK] Created collection in embedded Qdrant")
            print()

        except Exception as e:
            print(f"[FAIL] Failed to initialize embedded Qdrant: {e}")
            return False

    # Step 4: Copy vectors
    print(f"Step 4: {'[DRY RUN] Would copy' if dry_run else 'Copying'} vectors...")

    try:
        # Fetch all points from external
        print(f"  Fetching {point_count} vectors from external...")

        # Scroll through all points
        scroll_result = external_client.scroll(
            collection_name=collection_name,
            limit=10000,  # Fetch in batches
            with_payload=True,
            with_vectors=True
        )

        points = scroll_result[0]
        next_offset = scroll_result[1]

        # Continue scrolling if there are more points
        while next_offset is not None:
            scroll_result = external_client.scroll(
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
            print(f"  [DRY RUN] Would upload {len(points)} points to embedded")

            # Show sample
            if points:
                sample = points[0]
                print(f"\n  Sample point:")
                print(f"    ID: {sample.id}")
                print(f"    Vector: [{sample.vector[0]:.4f}, {sample.vector[1]:.4f}, ...]")
                print(f"    Payload keys: {list(sample.payload.keys())}")
        else:
            # Upload to embedded in batches
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

                embedded_client.upsert(
                    collection_name=collection_name,
                    points=point_structs
                )

                print(f"  Progress: {min(i+batch_size, len(points))}/{len(points)} points", end='\r')

            print(f"\n  [OK] Uploaded {len(points)} points to embedded")

            # Verify
            embedded_info = embedded_client.get_collection(collection_name)
            embedded_count = embedded_info.points_count

            if embedded_count == point_count:
                print(f"  [OK] Verification: {embedded_count} points in embedded (matches external)")
            else:
                print(f"  [WARN] Warning: {embedded_count} points in embedded vs {point_count} in external")

        print()

    except Exception as e:
        print(f"[FAIL] Failed to copy vectors: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Summary
    print("=" * 70)
    if dry_run:
        print("DRY RUN COMPLETE - No changes made")
        print()
        print("To perform actual migration, run without --dry-run:")
        print("  python migrate_qdrant_external_to_embedded.py")
    else:
        print("MIGRATION COMPLETE [OK]")
        print()
        print(f"Migrated: {len(points)} vectors")
        print(f"From:     {external_host}:{external_port}")
        print(f"To:       {qdrant_data_path}")
        print()
        print("Next steps:")
        print("  1. Stop external Qdrant server (or keep for backup)")
        print("  2. Remove QDRANT_HOST and QDRANT_PORT environment variables")
        print("  3. System will automatically use embedded Qdrant")
        print()
        if backup_path:
            print(f"Backup: {backup_path}")

    print("=" * 70)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant vectors from external server to embedded storage"
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
