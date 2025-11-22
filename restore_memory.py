#!/usr/bin/env python3
"""
Restore BuildAutomata Memory System
Imports SQLite database and Qdrant collection from backup zip archive

WARNING: This will DELETE all existing data!
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
from pathlib import Path
from typing import Dict, Any

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("Warning: Qdrant not available - will restore SQLite only")


def get_config() -> Dict[str, Any]:
    """Get configuration from environment variables"""
    username = os.getenv("BA_USERNAME", "buildautomata_ai_v012")
    agent_name = os.getenv("BA_AGENT_NAME", "claude_assistant")

    # Detect Qdrant mode: embedded (default) or external
    use_external = os.getenv("USE_EXTERNAL_QDRANT", "").lower() in ("true", "1", "yes")

    repo_dir = Path(__file__).parent / "memory_repos" / f"{username}_{agent_name}"

    config = {
        "username": username,
        "agent_name": agent_name,
        "db_dir": repo_dir,
        "backup_dir": Path(__file__).parent / "backups",
        "collection_name": f"{username}_{agent_name}_memories",
        "use_external_qdrant": use_external
    }

    if use_external:
        # External Qdrant mode
        config["qdrant_host"] = os.getenv("QDRANT_HOST", "localhost")
        config["qdrant_port"] = int(os.getenv("QDRANT_PORT", "6333"))
        config["qdrant_path"] = None
    else:
        # Embedded Qdrant mode (default)
        config["qdrant_host"] = None
        config["qdrant_port"] = None
        config["qdrant_path"] = str(repo_dir / "qdrant_data")

    return config


def validate_backup(backup_path: Path) -> Dict[str, Any]:
    """Validate backup archive and read manifest"""
    print(f"Validating backup: {backup_path.name}...")

    if not backup_path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")

    if not zipfile.is_zipfile(backup_path):
        raise ValueError(f"Not a valid zip file: {backup_path}")

    # Read manifest
    with zipfile.ZipFile(backup_path, 'r') as zipf:
        files = zipf.namelist()

        if 'manifest.json' not in files:
            raise ValueError("Backup missing manifest.json")

        with zipf.open('manifest.json') as f:
            manifest = json.load(f)

        # Check for required files
        if 'memoryv012.db' not in files:
            raise ValueError("Backup missing SQLite database")

        has_qdrant = 'qdrant_vectors.json' in files

        print(f"  OK - Valid backup from {manifest.get('created_at', 'unknown')}")
        print(f"  SQLite: {manifest['stats']['sqlite'].get('total_memories', '?')} memories")
        print(f"  Qdrant: {manifest['stats']['qdrant'].get('total_vectors', '?')} vectors")

        return manifest


def confirm_destructive_operation(config: Dict[str, Any]) -> bool:
    """Ask user to confirm destructive restore operation"""
    print("\n" + "="*60)
    print("WARNING  WARNING: DESTRUCTIVE OPERATION WARNING")
    print("="*60)
    print("This will DELETE ALL existing data:")
    print(f"  - SQLite database: {config['db_dir'] / 'memoryv012.db'}")
    print(f"  - Qdrant collection: {config['collection_name']}")
    print("="*60)

    response = input("\nType 'DELETE ALL DATA' to confirm: ")
    return response == "DELETE ALL DATA"


def delete_existing_data(config: Dict[str, Any]):
    """Delete existing SQLite database and Qdrant collection"""
    print("\nDeleting existing data...")

    # Delete SQLite database
    db_path = config["db_dir"] / "memoryv012.db"
    if db_path.exists():
        db_path.unlink()
        print(f"  OK - Deleted SQLite database")
    else:
        print(f"  INFO - SQLite database not found (clean state)")

    # Delete Qdrant collection
    if QDRANT_AVAILABLE:
        try:
            if config["use_external_qdrant"]:
                client = QdrantClient(host=config["qdrant_host"], port=config["qdrant_port"])
            else:
                client = QdrantClient(path=config["qdrant_path"])

            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            if config["collection_name"] in collection_names:
                client.delete_collection(config["collection_name"])
                print(f"  OK - Deleted Qdrant collection")
            else:
                print(f"  INFO - Qdrant collection not found (clean state)")

        except Exception as e:
            print(f"  WARNING - Could not delete Qdrant collection: {e}")
    else:
        print(f"  INFO - Qdrant not available, skipping collection delete")


def restore_sqlite(backup_path: Path, config: Dict[str, Any]):
    """Restore SQLite database from backup"""
    print("\nRestoring SQLite database...")

    # Ensure target directory exists
    config["db_dir"].mkdir(parents=True, exist_ok=True)

    # Extract database
    with zipfile.ZipFile(backup_path, 'r') as zipf:
        zipf.extract('memoryv012.db', config["db_dir"])

    db_path = config["db_dir"] / "memoryv012.db"

    # Verify
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM memories")
    memory_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM memory_versions")
    version_count = cursor.fetchone()[0]

    conn.close()

    print(f"  OK - Restored {memory_count} memories, {version_count} versions")

    return {"memories": memory_count, "versions": version_count}


def restore_qdrant(backup_path: Path, config: Dict[str, Any], manifest: Dict[str, Any]):
    """Restore Qdrant collection from backup"""
    if not QDRANT_AVAILABLE:
        print("\nSkipping Qdrant restore (not available)")
        return {"vectors": 0, "skipped": True}

    # Check if backup has Qdrant data
    with zipfile.ZipFile(backup_path, 'r') as zipf:
        if 'qdrant_vectors.json' not in zipf.namelist():
            print("\nSkipping Qdrant restore (not in backup)")
            return {"vectors": 0, "not_in_backup": True}

    mode = "external" if config["use_external_qdrant"] else "embedded"
    print(f"\nRestoring Qdrant collection ({mode} mode)...")

    try:
        if config["use_external_qdrant"]:
            client = QdrantClient(host=config["qdrant_host"], port=config["qdrant_port"])
            print(f"  Connected to external Qdrant at {config['qdrant_host']}:{config['qdrant_port']}")
        else:
            client = QdrantClient(path=config["qdrant_path"])
            print(f"  Using embedded Qdrant at {config['qdrant_path']}")

        # Extract vector data
        temp_qdrant = Path(__file__).parent / "temp_qdrant.json"
        with zipfile.ZipFile(backup_path, 'r') as zipf:
            zipf.extract('qdrant_vectors.json', Path(__file__).parent)
            temp_qdrant = Path(__file__).parent / 'qdrant_vectors.json'

        # Load vector data
        with open(temp_qdrant, 'r') as f:
            qdrant_data = json.load(f)

        vector_size = qdrant_data["vector_size"]
        distance_str = qdrant_data["distance"]
        points_data = qdrant_data["points"]

        # Map distance string to enum
        distance_map = {
            "Distance.COSINE": Distance.COSINE,
            "Distance.EUCLID": Distance.EUCLID,
            "Distance.DOT": Distance.DOT
        }
        distance = distance_map.get(distance_str, Distance.COSINE)

        print(f"  Creating collection (vector_size={vector_size}, distance={distance_str})...")

        # Create collection
        client.create_collection(
            collection_name=config["collection_name"],
            vectors_config=VectorParams(size=vector_size, distance=distance)
        )

        # Upload points in batches
        batch_size = 100
        total_points = len(points_data)

        print(f"  Uploading {total_points} vectors...")

        for i in range(0, total_points, batch_size):
            batch = points_data[i:i + batch_size]

            from qdrant_client.models import PointStruct
            points = [
                PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point["payload"]
                )
                for point in batch
            ]

            client.upsert(
                collection_name=config["collection_name"],
                points=points
            )

            print(f"  Progress: {min(i + batch_size, total_points)}/{total_points} vectors", end="\r")

        print(f"\n  OK - Restored {total_points} vectors")

        # Cleanup temp file
        temp_qdrant.unlink()

        return {"vectors": total_points}

    except Exception as e:
        print(f"  [FAILED] Qdrant restore failed: {e}")
        import traceback
        traceback.print_exc()
        return {"vectors": 0, "error": str(e)}


def restore_backup(backup_path: Path, config: Dict[str, Any], dry_run: bool = False):
    """Restore from backup archive"""

    # Validate backup
    manifest = validate_backup(backup_path)

    if dry_run:
        print("\n[DRY RUN] Would restore:")
        print(f"  SQLite: {manifest['stats']['sqlite'].get('total_memories', '?')} memories")
        print(f"  Qdrant: {manifest['stats']['qdrant'].get('total_vectors', '?')} vectors")
        print("\n[DRY RUN] No changes made")
        return

    # Confirm destructive operation
    if not confirm_destructive_operation(config):
        print("\n[FAILED] Restore cancelled")
        return 1

    print("\nProceeding with restore...")

    # Delete existing data
    delete_existing_data(config)

    # Restore SQLite
    sqlite_stats = restore_sqlite(backup_path, config)

    # Restore Qdrant
    qdrant_stats = restore_qdrant(backup_path, config, manifest)

    print("\n" + "="*60)
    print("OK - Restore completed successfully!")
    print("="*60)
    print(f"Restored from: {backup_path.name}")
    print(f"Memories: {sqlite_stats.get('memories', 0)}")
    print(f"Versions: {sqlite_stats.get('versions', 0)}")
    print(f"Vectors: {qdrant_stats.get('vectors', 0)}")
    print("="*60)

    return 0


def list_backups(config: Dict[str, Any]):
    """List available backups for restore"""
    from backup_memory import list_backups as backup_list
    backup_list(config)


def main():
    parser = argparse.ArgumentParser(
        description="Restore BuildAutomata Memory System from backup",
        epilog="WARNING: This will DELETE all existing data!"
    )
    parser.add_argument("backup", nargs="?", help="Backup file to restore (e.g., backup_20251024_183000.zip)")
    parser.add_argument("--dry-run", action="store_true", help="Validate backup without restoring")
    parser.add_argument("--list", "-l", action="store_true", help="List available backups")
    parser.add_argument("--username", help="Override BA_USERNAME")
    parser.add_argument("--agent", help="Override BA_AGENT_NAME")

    args = parser.parse_args()

    # Override config if specified
    if args.username:
        os.environ["BA_USERNAME"] = args.username
    if args.agent:
        os.environ["BA_AGENT_NAME"] = args.agent

    config = get_config()

    if args.list:
        list_backups(config)
        return 0

    if not args.backup:
        print("Error: No backup specified")
        print("\nUse --list to see available backups")
        parser.print_help()
        return 1

    # Resolve backup path
    backup_path = Path(args.backup)
    if not backup_path.is_absolute():
        # Try in backups directory
        backup_path = config["backup_dir"] / args.backup

    print("="*60)
    print("BuildAutomata Memory System - Restore")
    print("="*60)
    print(f"Username: {config['username']}")
    print(f"Agent: {config['agent_name']}")
    print(f"Target database: {config['db_dir']}")
    print(f"Target collection: {config['collection_name']}")
    print("="*60)

    try:
        return restore_backup(backup_path, config, dry_run=args.dry_run)
    except Exception as e:
        print(f"\n[FAILED] Restore failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
