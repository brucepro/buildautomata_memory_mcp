"""
Intention management for BuildAutomata Memory System
Copyright 2025 Jurden Bruce
"""

import json
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("buildautomata-memory.intentions")

try:
    from .models import Intention
except ImportError:
    from models import Intention


class IntentionManager:
    """Handles intention storage and retrieval (Agency Bridge Pattern)"""

    def __init__(self, db_conn, db_lock, error_log, get_working_set_func=None):
        self.db_conn = db_conn
        self.db_lock = db_lock
        self.error_log = error_log
        self.get_working_set_func = get_working_set_func

    def _log_error(self, operation: str, error: Exception):
        """Log detailed error information"""
        import traceback
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "error_type": type(error).__name__,
            "error_msg": str(error),
            "traceback": traceback.format_exc(),
        }
        self.error_log.append(error_entry)
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]

    async def store_intention(
        self,
        description: str,
        priority: float = 0.5,
        deadline: Optional[datetime] = None,
        preconditions: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
        related_memories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a new intention"""
        if not self.db_conn:
            return {"error": "Database not available"}

        intention = Intention(
            id=str(uuid.uuid4()),
            description=description,
            priority=max(0.0, min(1.0, priority)),
            status="pending",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            deadline=deadline,
            preconditions=preconditions or [],
            actions=actions or [],
            related_memories=related_memories or [],
            metadata=metadata or {},
        )

        try:
            with self.db_lock:
                self.db_conn.execute("""
                    INSERT INTO intentions (
                        id, description, priority, status,
                        created_at, updated_at, deadline,
                        preconditions, actions, related_memories, metadata,
                        last_checked, check_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    intention.id, intention.description, intention.priority, intention.status,
                    intention.created_at, intention.updated_at, intention.deadline,
                    json.dumps(intention.preconditions), json.dumps(intention.actions),
                    json.dumps(intention.related_memories), json.dumps(intention.metadata),
                    intention.last_checked, intention.check_count,
                ))
                self.db_conn.commit()

            logger.info(f"Stored intention {intention.id}: {description[:50]}...")
            return {
                "success": True,
                "intention_id": intention.id,
                "priority": intention.priority,
                "status": intention.status,
            }
        except Exception as e:
            logger.error(f"Failed to store intention: {e}")
            self._log_error("store_intention", e)
            return {"error": str(e)}

    async def get_active_intentions(
        self,
        limit: int = 10,
        include_pending: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get active intentions sorted by priority"""
        if not self.db_conn:
            return []

        try:
            with self.db_lock:
                statuses = ["active"]
                if include_pending:
                    statuses.append("pending")

                placeholders = ','.join('?' * len(statuses))
                cursor = self.db_conn.execute(f"""
                    SELECT * FROM intentions
                    WHERE status IN ({placeholders})
                    ORDER BY priority DESC, deadline ASC
                    LIMIT ?
                """, (*statuses, limit))

                intentions = []
                for row in cursor.fetchall():
                    intention_dict = dict(row)
                    intention_dict['preconditions'] = json.loads(row['preconditions'] or '[]')
                    intention_dict['actions'] = json.loads(row['actions'] or '[]')
                    intention_dict['related_memories'] = json.loads(row['related_memories'] or '[]')
                    intention_dict['metadata'] = json.loads(row['metadata'] or '{}')
                    intentions.append(intention_dict)

                return intentions
        except Exception as e:
            logger.error(f"Failed to get active intentions: {e}")
            self._log_error("get_active_intentions", e)
            return []

    async def update_intention_status(
        self,
        intention_id: str,
        status: str,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update intention status"""
        if not self.db_conn:
            return {"error": "Database not available"}

        if status not in ["pending", "active", "completed", "cancelled"]:
            return {"error": f"Invalid status: {status}"}

        try:
            with self.db_lock:
                cursor = self.db_conn.execute(
                    "SELECT * FROM intentions WHERE id = ?", (intention_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return {"error": f"Intention not found: {intention_id}"}

                metadata = json.loads(row['metadata'] or '{}')
                if metadata_updates:
                    metadata.update(metadata_updates)

                self.db_conn.execute("""
                    UPDATE intentions
                    SET status = ?, updated_at = ?, metadata = ?
                    WHERE id = ?
                """, (status, datetime.now(), json.dumps(metadata), intention_id))
                self.db_conn.commit()

            logger.info(f"Updated intention {intention_id} to status: {status}")
            return {"success": True, "intention_id": intention_id, "status": status}
        except Exception as e:
            logger.error(f"Failed to update intention status: {e}")
            self._log_error("update_intention_status", e)
            return {"error": str(e)}

    async def check_intention(self, intention_id: str) -> Dict[str, Any]:
        """Mark intention as checked"""
        if not self.db_conn:
            return {"error": "Database not available"}

        try:
            with self.db_lock:
                self.db_conn.execute("""
                    UPDATE intentions
                    SET last_checked = ?, check_count = check_count + 1
                    WHERE id = ?
                """, (datetime.now(), intention_id))
                self.db_conn.commit()

            return {"success": True}
        except Exception as e:
            logger.error(f"Failed to check intention: {e}")
            return {"error": str(e)}

    async def proactive_initialization_scan(self) -> Dict[str, Any]:
        """Proactive scan on startup"""
        scan_start = datetime.now()
        scan_results = {
            "timestamp": scan_start.isoformat(),
            "continuity_check": {},
            "active_intentions": [],
            "urgent_items": [],
            "context_summary": {},
        }

        if self.db_conn is None:
            logger.warning("Proactive scan skipped: SQLite not initialized")
            scan_results["error"] = "SQLite not initialized"
            return scan_results

        try:
            # Continuity check - prefer command_history for accuracy, fall back to memories
            with self.db_lock:
                last_activity = None

                # Try command_history first (more accurate - tracks actual MCP calls)
                try:
                    cursor = self.db_conn.execute(
                        "SELECT timestamp FROM command_history ORDER BY timestamp DESC LIMIT 1"
                    )
                    row = cursor.fetchone()
                    if row and row['timestamp']:
                        last_activity = row['timestamp']
                except Exception:
                    pass  # Table might not exist yet

                # Fall back to memories if no command history
                if not last_activity:
                    cursor = self.db_conn.execute("SELECT MAX(updated_at) as last_activity FROM memories")
                    row = cursor.fetchone()
                    if row and row['last_activity']:
                        last_activity = row['last_activity']

                if last_activity:
                    if isinstance(last_activity, str):
                        last_activity = datetime.fromisoformat(last_activity)
                    now = datetime.now(last_activity.tzinfo) if last_activity.tzinfo else datetime.now()
                    time_gap = now - last_activity
                    scan_results["continuity_check"] = {
                        "last_activity": last_activity.isoformat(),
                        "time_gap_hours": time_gap.total_seconds() / 3600,
                        "is_new_session": time_gap.total_seconds() > 3600,
                    }

            # Active intentions
            active_intentions = await self.get_active_intentions(limit=5)
            scan_results["active_intentions"] = active_intentions

            # Urgent items
            urgent = []
            for intention in active_intentions:
                if intention.get('deadline'):
                    deadline = intention['deadline']
                    if isinstance(deadline, str):
                        deadline = datetime.fromisoformat(deadline)
                    now = datetime.now(deadline.tzinfo) if deadline.tzinfo else datetime.now()
                    if deadline < now:
                        urgent.append({
                            "type": "overdue_intention",
                            "id": intention['id'],
                            "description": intention['description'],
                            "deadline": intention['deadline'],
                        })
                elif intention.get('priority', 0) >= 0.9:
                    urgent.append({
                        "type": "high_priority_intention",
                        "id": intention['id'],
                        "description": intention['description'],
                        "priority": intention['priority'],
                    })

            scan_results["urgent_items"] = urgent

            # Recent context
            with self.db_lock:
                cursor = self.db_conn.execute("""
                    SELECT id, content, category, created_at
                    FROM memories
                    ORDER BY created_at DESC
                    LIMIT 5
                """)
                recent_memories = []
                for row in cursor.fetchall():
                    recent_memories.append({
                        "id": row['id'],
                        "content": row['content'],
                        "category": row['category'],
                        "created_at": row['created_at'],
                    })

            scan_results["context_summary"] = {
                "recent_memories": recent_memories,
                "active_intention_count": len(active_intentions),
                "urgent_count": len(urgent),
            }

            # 5. Working Set - optimal context memories for confabulation prevention
            # Load top 10 memories by working set score (importance + failure patterns + recency)
            if self.get_working_set_func:
                working_set = await self.get_working_set_func(target_size=40)

                # Format working set with FULL content - these are important enough to load completely
                working_set_formatted = []
                for mem in working_set[:10]:  # Limit to 10 for token budget
                    working_set_formatted.append({
                        "id": mem['id'],
                        "category": mem['category'],
                        "importance": mem.get('importance', 0),
                        "content": mem['content'],  # Full content, not preview
                        "created_at": mem.get('created_at', ''),
                        "working_set_score": mem.get('working_set_score', 0),  # Include score for transparency
                    })

                scan_results["working_set"] = {
                    "size": len(working_set),
                    "memories": working_set_formatted,
                    "description": "Top memories by importance, failure patterns, and recency for confabulation prevention"
                }
            else:
                scan_results["working_set"] = {
                    "size": 0,
                    "memories": [],
                    "description": "Working set not available (get_working_set_func not provided)"
                }

            scan_duration = (datetime.now() - scan_start).total_seconds() * 1000
            scan_results["scan_duration_ms"] = scan_duration
            logger.info(f"Proactive scan completed in {scan_duration:.1f}ms with {scan_results['working_set']['size']} working set memories")

            return scan_results

        except Exception as e:
            logger.error(f"Proactive scan failed: {e}")
            self._log_error("proactive_scan", e)
            return {"error": str(e)}
