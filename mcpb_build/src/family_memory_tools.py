"""
Family Memory Network Tool Implementations
Handles hybrid sync: Syncthing for file transport, agent control for processing
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import subprocess
import sys

# Import merge and rebuild scripts
try:
    from merge_family_conflicts import merge_syncthing_conflicts
    from rebuild_family_qdrant import rebuild_qdrant_index
except ImportError:
    pass

# DEFAULT_FAMILY_DIR computed based on memory_repos structure
def get_default_family_dir() -> Path:
    """Get family_share folder path based on current instance"""
    import os
    # Use BA_INSTANCE_NAME (Scout, Alpha, etc.) for agent-specific directory
    instance_name = os.getenv("BA_INSTANCE_NAME", "Scout")
    username = os.getenv("MEMORY_USERNAME", "buildautomata_ai")
    version = os.getenv("MEMORY_VERSION", "v012")
    return Path(f"A:/buildautomata_memory/memory_repos/{username}_{version}_{instance_name}/family_share")


def get_family_sync_status(family_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Check family memory sync status

    Returns:
        - pending_conflicts: Number of unmerged conflict files
        - last_merge: Timestamp of last merge operation
        - last_index_rebuild: Timestamp of last Qdrant rebuild
        - unmerged_memories: Estimated count of unmerged memories
        - database_exists: Whether family_shared.db exists
        - syncthing_running: Whether Syncthing process detected
    """

    family_path = Path(family_dir) if family_dir else get_default_family_dir()
    db_path = family_path / "family_shared.db"
    log_dir = family_path / "logs"
    qdrant_dir = family_path / "family_shared_qdrant"

    result = {
        "family_directory": str(family_path),
        "database_exists": db_path.exists(),
        "pending_conflicts": 0,
        "last_merge": None,
        "last_index_rebuild": None,
        "unmerged_memories": 0,
        "syncthing_running": False,
        "database_size_mb": 0.0,
    }

    # Check for conflict files
    if family_path.exists():
        conflict_files = list(family_path.glob("family_shared.sync-conflict-*"))
        result["pending_conflicts"] = len(conflict_files)

        # Estimate unmerged memories
        for conflict_file in conflict_files:
            try:
                conn = sqlite3.connect(conflict_file)
                cursor = conn.execute("SELECT COUNT(*) FROM family_memories")
                count = cursor.fetchone()[0]
                result["unmerged_memories"] += count
                conn.close()
            except:
                pass

    # Check last merge time
    merge_log = log_dir / "merge_operations.log"
    if merge_log.exists():
        try:
            # Read last line of merge log
            with open(merge_log, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if '"timestamp"' in line:
                        # Extract timestamp from JSON
                        start = line.find('"timestamp": "') + 14
                        end = line.find('"', start)
                        result["last_merge"] = line[start:end]
                        break
        except:
            pass

    # Check last index rebuild
    metadata_file = qdrant_dir / ".index_metadata"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                result["last_index_rebuild"] = metadata.get("rebuild_time")
        except:
            pass

    # Check database size
    if db_path.exists():
        result["database_size_mb"] = round(db_path.stat().st_size / (1024 * 1024), 2)

    # Check if Syncthing running (simple process check)
    try:
        if sys.platform == "win32":
            output = subprocess.check_output("tasklist", shell=True, text=True)
            result["syncthing_running"] = "syncthing.exe" in output.lower()
        else:
            output = subprocess.check_output(["ps", "aux"], text=True)
            result["syncthing_running"] = "syncthing" in output.lower()
    except:
        result["syncthing_running"] = "unknown"

    return result


def sync_family_memory(rebuild_index: bool = True, family_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Manually trigger family memory conflict merge and optional index rebuild

    Args:
        rebuild_index: Whether to rebuild Qdrant index after merge
        family_dir: Path to family_memory directory

    Returns:
        - merged: Number of memories merged
        - conflicts_processed: Number of conflict files processed
        - errors: Number of errors encountered
        - indexed: Number of memories indexed (if rebuild_index=True)
        - success: Overall success status
    """

    family_path = Path(family_dir) if family_dir else get_default_family_dir()

    result = {
        "family_directory": str(family_path),
        "merged": 0,
        "conflicts_processed": 0,
        "errors": 0,
        "indexed": None,
        "success": False,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # Run merge script
        # Update FAMILY_DIR, MAIN_DB, LOG_DIR in merge module
        import merge_family_conflicts
        merge_family_conflicts.FAMILY_DIR = family_path
        merge_family_conflicts.MAIN_DB = family_path / "family_shared.db"
        merge_family_conflicts.LOG_DIR = family_path / "logs"
        merge_family_conflicts.ARCHIVE_DIR = family_path / "conflicts_archived"
        merge_family_conflicts.LOCK_FILE = family_path / ".merge_lock"

        merged, conflicts, errors = merge_family_conflicts.merge_syncthing_conflicts(verbose=False)

        result["merged"] = merged
        result["conflicts_processed"] = conflicts
        result["errors"] = errors

        # Optionally rebuild index
        if rebuild_index:
            import rebuild_family_qdrant
            rebuild_family_qdrant.FAMILY_DIR = family_path
            rebuild_family_qdrant.MAIN_DB = family_path / "family_shared.db"
            rebuild_family_qdrant.QDRANT_DIR = family_path / "family_shared_qdrant"

            indexed, skipped = rebuild_family_qdrant.rebuild_qdrant_index(force=False, verbose=False)
            result["indexed"] = indexed

        result["success"] = errors == 0

    except Exception as e:
        result["error"] = str(e)
        result["success"] = False

    return result


def search_family_memories(
    query: str,
    limit: int = 10,
    include_personal: bool = True,
    author_filter: Optional[str] = None,
    family_dir: Optional[str] = None,
    memory_store = None,
) -> Dict[str, Any]:
    """
    Search family memory network with attribution

    Args:
        query: Search query
        limit: Max results
        include_personal: Include personal memories in results
        author_filter: Filter by specific author instance
        family_dir: Path to family_memory directory
        memory_store: MemoryStore instance for personal search

    Returns:
        Combined search results with attribution
    """

    family_path = Path(family_dir) if family_dir else get_default_family_dir()
    db_path = family_path / "family_shared.db"

    results = []

    # Search family database (FTS for now, semantic search requires Qdrant setup)
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)

            # FTS search on family memories
            sql = """
                SELECT m.memory_id, m.author_instance, m.content, m.category,
                       m.created_at, m.importance, m.tags
                FROM family_memories_fts fts
                JOIN family_memories m ON m.rowid = fts.rowid
                WHERE family_memories_fts MATCH ?
            """

            params = [query]

            if author_filter:
                sql += " AND m.author_instance = ?"
                params.append(author_filter)

            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)

            for row in cursor:
                results.append({
                    "memory_id": row[0],
                    "author_instance": row[1],
                    "content": row[2],
                    "category": row[3],
                    "created_at": row[4],
                    "importance": row[5],
                    "tags": json.loads(row[6]) if row[6] else [],
                    "display_prefix": f"[{row[1]}]",
                    "is_family_memory": True,
                    "is_external": True,
                })

            conn.close()

        except Exception as e:
            return {
                "error": f"Family search failed: {e}",
                "results": [],
                "total": 0,
            }

    # TODO: Include personal memories if requested (requires memory_store integration)
    # For now, just return family results

    return {
        "results": results,
        "total": len(results),
        "query": query,
        "sources": {
            "family": len(results),
            "personal": 0,  # TODO
        },
    }


def share_memory_to_family(
    memory_id: str,
    family_dir: Optional[str] = None,
    memory_store = None,
) -> Dict[str, Any]:
    """
    Share specific personal memory to family network

    Args:
        memory_id: Personal memory ID to share
        family_dir: Path to family_memory directory
        memory_store: MemoryStore instance to read personal memory

    Returns:
        Success status and shared memory details
    """

    family_path = Path(family_dir) if family_dir else get_default_family_dir()
    db_path = family_path / "family_shared.db"

    if not db_path.exists():
        return {
            "success": False,
            "error": "Family database not found. Run init_family_db.py first.",
        }

    if not memory_store:
        return {
            "success": False,
            "error": "Memory store not available. Cannot read personal memory.",
        }

    # TODO: Implement memory retrieval from personal store and copy to family DB
    # This requires integration with MemoryStore.get_memory()

    return {
        "success": False,
        "error": "share_memory_to_family not yet implemented - requires MemoryStore integration",
        "memory_id": memory_id,
    }
