#!/usr/bin/env python3
"""
Backup BuildAutomata Memory System
Exports SQLite database and Qdrant collection to timestamped zip archive
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("Warning: Qdrant not available - will backup SQLite only")


def get_config() -> Dict[str, Any]:
    """Get configuration from environment variables"""
    username = os.getenv("BA_USERNAME", "buildautomata_ai_v012")
    agent_name = os.getenv("BA_AGENT_NAME", "claude_assistant")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    return {
        "username": username,
        "agent_name": agent_name,
        "qdrant_host": qdrant_host,
        "qdrant_port": qdrant_port,
        "db_dir": Path(__file__).parent / "memory_repos" / f"{username}_{agent_name}",
        "backup_dir": Path(__file__).parent / "backups",
        "collection_name": f"{username}_{agent_name}_memories"
    }


def export_sqlite(db_path: Path, temp_dir: Path) -> Dict[str, Any]:
    """Export SQLite database"""
    print(f"Exporting SQLite database from {db_path}...")

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Copy database file
    db_backup = temp_dir / "memoryv012.db"
    shutil.copy2(db_path, db_backup)

    # Get stats
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    stats = {}
    cursor.execute("SELECT COUNT(*) FROM memories")
    stats["total_memories"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM memory_versions")
    stats["total_versions"] = cursor.fetchone()[0]

    try:
        cursor.execute("SELECT COUNT(*) FROM intentions")
        stats["total_intentions"] = cursor.fetchone()[0]
    except:
        stats["total_intentions"] = 0

    conn.close()

    print(f"  OK - Exported {stats['total_memories']} memories, {stats['total_versions']} versions")

    return stats


def export_qdrant(config: Dict[str, Any], temp_dir: Path) -> Dict[str, Any]:
    """Export Qdrant collection to JSON"""
    if not QDRANT_AVAILABLE:
        print("Skipping Qdrant export (not available)")
        return {"total_vectors": 0, "skipped": True}

    print(f"Exporting Qdrant collection '{config['collection_name']}'...")

    try:
        client = QdrantClient(host=config["qdrant_host"], port=config["qdrant_port"])

        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if config["collection_name"] not in collection_names:
            print(f"  WARNING - Collection not found, skipping Qdrant export")
            return {"total_vectors": 0, "collection_not_found": True}

        # Get collection info
        collection_info = client.get_collection(config["collection_name"])
        total_vectors = collection_info.points_count

        print(f"  Exporting {total_vectors} vectors...")

        # Scroll through all points
        all_points = []
        offset = None
        batch_size = 100

        while True:
            results = client.scroll(
                collection_name=config["collection_name"],
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )

            points, offset = results

            if not points:
                break

            for point in points:
                all_points.append({
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                })

            if offset is None:
                break

            print(f"  Progress: {len(all_points)}/{total_vectors} vectors", end="\r")

        print(f"\n  OK - Exported {len(all_points)} vectors")

        # Write to JSON
        qdrant_file = temp_dir / "qdrant_vectors.json"
        with open(qdrant_file, 'w', encoding='utf-8') as f:
            json.dump({
                "collection_name": config["collection_name"],
                "vector_size": collection_info.config.params.vectors.size,
                "distance": str(collection_info.config.params.vectors.distance),
                "points": all_points
            }, f, indent=2)

        return {"total_vectors": len(all_points)}

    except Exception as e:
        print(f"  WARNING - Qdrant export failed: {e}")
        return {"total_vectors": 0, "error": str(e)}


def create_backup(config: Dict[str, Any], description: str = None) -> Path:
    """Create backup archive"""

    # Create backup directory
    config["backup_dir"].mkdir(parents=True, exist_ok=True)

    # Create temp directory for export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = config["backup_dir"] / f"temp_{timestamp}"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Export SQLite
        db_path = config["db_dir"] / "memoryv012.db"
        sqlite_stats = export_sqlite(db_path, temp_dir)

        # Export Qdrant
        qdrant_stats = export_qdrant(config, temp_dir)

        # Create manifest
        manifest = {
            "timestamp": timestamp,
            "created_at": datetime.now().isoformat(),
            "description": description or "Memory system backup",
            "config": {
                "username": config["username"],
                "agent_name": config["agent_name"],
                "collection_name": config["collection_name"]
            },
            "stats": {
                "sqlite": sqlite_stats,
                "qdrant": qdrant_stats
            }
        }

        manifest_file = temp_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        # Create zip archive
        backup_filename = f"backup_{timestamp}.zip"
        backup_path = config["backup_dir"] / backup_filename

        print(f"\nCreating archive {backup_filename}...")

        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in temp_dir.iterdir():
                zipf.write(file, file.name)
                print(f"  Added {file.name}")

        # Cleanup temp directory
        shutil.rmtree(temp_dir)

        # Get final size
        size_mb = backup_path.stat().st_size / (1024 * 1024)

        print(f"\n[SUCCESS] Backup created successfully!")
        print(f"  Location: {backup_path}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Memories: {sqlite_stats.get('total_memories', 0)}")
        print(f"  Versions: {sqlite_stats.get('total_versions', 0)}")
        print(f"  Vectors: {qdrant_stats.get('total_vectors', 0)}")

        return backup_path

    except Exception as e:
        # Cleanup on error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise e


def list_backups(config: Dict[str, Any]):
    """List available backups"""
    backup_dir = config["backup_dir"]

    if not backup_dir.exists():
        print("No backups directory found")
        return

    backups = sorted(backup_dir.glob("backup_*.zip"), reverse=True)

    if not backups:
        print("No backups found")
        return

    print(f"\nAvailable backups in {backup_dir}:\n")

    for backup in backups:
        size_mb = backup.stat().st_size / (1024 * 1024)

        # Try to read manifest
        try:
            with zipfile.ZipFile(backup, 'r') as zipf:
                with zipf.open('manifest.json') as f:
                    manifest = json.load(f)
                    created = manifest.get('created_at', 'Unknown')
                    desc = manifest.get('description', 'No description')
                    stats = manifest.get('stats', {})
                    sqlite_stats = stats.get('sqlite', {})
                    qdrant_stats = stats.get('qdrant', {})

                    print(f"[BACKUP] {backup.name}")
                    print(f"   Created: {created}")
                    print(f"   Size: {size_mb:.2f} MB")
                    print(f"   Memories: {sqlite_stats.get('total_memories', '?')}, "
                          f"Versions: {sqlite_stats.get('total_versions', '?')}, "
                          f"Vectors: {qdrant_stats.get('total_vectors', '?')}")
                    print(f"   Description: {desc}")
                    print()
        except:
            print(f"📦 {backup.name}")
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   (Manifest not readable)")
            print()


def main():
    parser = argparse.ArgumentParser(description="Backup BuildAutomata Memory System")
    parser.add_argument("--description", "-d", help="Backup description")
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

    print("="*60)
    print("BuildAutomata Memory System - Backup")
    print("="*60)
    print(f"Username: {config['username']}")
    print(f"Agent: {config['agent_name']}")
    print(f"Database: {config['db_dir']}")
    print(f"Backup directory: {config['backup_dir']}")
    print("="*60)
    print()

    try:
        backup_path = create_backup(config, args.description)
        return 0
    except Exception as e:
        print(f"\n[FAILED] Backup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
