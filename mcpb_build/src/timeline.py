"""
Timeline analysis for BuildAutomata Memory System
Copyright 2025 Jurden Bruce
"""

import asyncio
import json
import logging
import re
import difflib
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("buildautomata-memory.timeline")


class TimelineAnalysis:
    """Handles temporal analysis and timeline generation"""

    def __init__(self, db_conn, db_lock):
        self.db_conn = db_conn
        self.db_lock = db_lock

    def compute_text_diff(self, old_text: str, new_text: str) -> Dict[str, Any]:
        """Compute text difference between versions"""
        if old_text == new_text:
            return {"changed": False}

        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        diff = list(difflib.unified_diff(old_lines, new_lines, lineterm='', n=0))

        similarity = difflib.SequenceMatcher(None, old_text, new_text).ratio()

        additions = [line[1:] for line in diff if line.startswith('+') and not line.startswith('+++')]
        deletions = [line[1:] for line in diff if line.startswith('-') and not line.startswith('---')]

        return {
            "changed": True,
            "similarity": round(similarity, 3),
            "additions": additions[:5],
            "deletions": deletions[:5],
            "total_additions": len(additions),
            "total_deletions": len(deletions),
            "change_magnitude": round(1 - similarity, 3)
        }

    def extract_memory_references(self, content: str, all_memory_ids: set) -> List[str]:
        """Extract references to other memories from content"""
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        found_ids = re.findall(uuid_pattern, content, re.IGNORECASE)
        return list(set(found_id for found_id in found_ids if found_id in all_memory_ids))

    def detect_temporal_patterns(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns in memory timeline"""
        if not events:
            return {}

        timestamps = []
        for event in events:
            try:
                timestamps.append(datetime.fromisoformat(event['timestamp']))
            except:
                continue

        if not timestamps:
            return {}

        timestamps.sort()

        # Burst detection
        bursts = []
        current_burst = {"start": timestamps[0], "end": timestamps[0], "count": 1, "events": [events[0]]}

        for i in range(1, len(timestamps)):
            time_gap = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600

            if time_gap <= 4:
                current_burst["end"] = timestamps[i]
                current_burst["count"] += 1
                current_burst["events"].append(events[i])
            else:
                if current_burst["count"] >= 3:
                    bursts.append({
                        "start": current_burst["start"].isoformat(),
                        "end": current_burst["end"].isoformat(),
                        "duration_hours": round((current_burst["end"] - current_burst["start"]).total_seconds() / 3600, 1),
                        "event_count": current_burst["count"],
                        "intensity": round(current_burst["count"] / max(1, (current_burst["end"] - current_burst["start"]).total_seconds() / 3600), 2)
                    })
                current_burst = {"start": timestamps[i], "end": timestamps[i], "count": 1, "events": [events[i]]}

        if current_burst["count"] >= 3:
            bursts.append({
                "start": current_burst["start"].isoformat(),
                "end": current_burst["end"].isoformat(),
                "duration_hours": round((current_burst["end"] - current_burst["start"]).total_seconds() / 3600, 1),
                "event_count": current_burst["count"],
                "intensity": round(current_burst["count"] / max(1, (current_burst["end"] - current_burst["start"]).total_seconds() / 3600), 2)
            })

        # Gap detection
        gaps = []
        for i in range(1, len(timestamps)):
            gap_hours = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
            if gap_hours > 24:
                gaps.append({
                    "start": timestamps[i-1].isoformat(),
                    "end": timestamps[i].isoformat(),
                    "duration_hours": round(gap_hours, 1),
                    "duration_days": round(gap_hours / 24, 1)
                })

        total_duration = (timestamps[-1] - timestamps[0]).total_seconds() / 3600

        return {
            "total_events": len(events),
            "first_event": timestamps[0].isoformat(),
            "last_event": timestamps[-1].isoformat(),
            "total_duration_hours": round(total_duration, 1),
            "total_duration_days": round(total_duration / 24, 1),
            "bursts": bursts,
            "gaps": gaps,
            "avg_events_per_day": round(len(events) / max(1, total_duration / 24), 2) if total_duration > 0 else 0
        }

    def get_memory_versions_detailed(
        self,
        mem_id: str,
        all_memory_ids: set,
        include_diffs: bool
    ) -> List[Dict[str, Any]]:
        """Get detailed version history with diffs"""
        if not self.db_conn:
            return []

        try:
            with self.db_lock:
                cursor = self.db_conn.execute("""
                    SELECT
                        curr.version_id, curr.version_number, curr.content, curr.category,
                        curr.importance, curr.tags, curr.metadata, curr.change_type,
                        curr.change_description, curr.created_at, curr.content_hash, curr.prev_version_id,
                        prev.content as prev_content, prev.category as prev_category,
                        prev.importance as prev_importance, prev.tags as prev_tags
                    FROM memory_versions curr
                    LEFT JOIN memory_versions prev ON curr.prev_version_id = prev.version_id
                    WHERE curr.memory_id = ?
                    ORDER BY curr.version_number ASC
                """, (mem_id,))

                versions = cursor.fetchall()
                if not versions:
                    return []

                events = []
                prev_content_for_diff = None

                for row in versions:
                    event = {
                        "memory_id": mem_id,
                        "version": row["version_number"],
                        "timestamp": row["created_at"],
                        "change_type": row["change_type"],
                        "change_description": row["change_description"],
                        "content": row["content"],
                        "category": row["category"],
                        "importance": row["importance"],
                        "tags": json.loads(row["tags"]) if row["tags"] else [],
                        "content_hash": row["content_hash"][:8],
                    }

                    if include_diffs and prev_content_for_diff:
                        diff_info = self.compute_text_diff(prev_content_for_diff, row["content"])
                        if diff_info.get("changed"):
                            event["diff"] = diff_info

                    if row["prev_version_id"] and row["prev_category"]:
                        field_changes = []
                        if row["prev_category"] != row["category"]:
                            field_changes.append(f"category: {row['prev_category']} → {row['category']}")
                        if row["prev_importance"] != row["importance"]:
                            field_changes.append(f"importance: {row['prev_importance']} → {row['importance']}")

                        prev_tags = set(json.loads(row["prev_tags"]) if row["prev_tags"] else [])
                        curr_tags = set(json.loads(row["tags"]) if row["tags"] else [])
                        if prev_tags != curr_tags:
                            added_tags = curr_tags - prev_tags
                            removed_tags = prev_tags - curr_tags
                            if added_tags:
                                field_changes.append(f"tags added: {', '.join(added_tags)}")
                            if removed_tags:
                                field_changes.append(f"tags removed: {', '.join(removed_tags)}")

                        if field_changes:
                            event["field_changes"] = field_changes

                    references = self.extract_memory_references(row["content"], all_memory_ids)
                    if references:
                        event["references"] = references
                        event["references_count"] = len(references)

                    events.append(event)
                    prev_content_for_diff = row["content"]

                return events

        except Exception as e:
            logger.error(f"Error getting detailed versions for {mem_id}: {e}")
            return []

    def build_relationship_graph(self, events: List[Dict]) -> Dict[str, Any]:
        """Build graph showing how memories reference each other"""
        reference_map = {}
        referenced_by_map = {}

        for event in events:
            mem_id = event["memory_id"]
            refs = event.get("references", [])

            if mem_id not in reference_map:
                reference_map[mem_id] = set()

            for ref in refs:
                reference_map[mem_id].add(ref)
                if ref not in referenced_by_map:
                    referenced_by_map[ref] = set()
                referenced_by_map[ref].add(mem_id)

        return {
            "references": {k: list(v) for k, v in reference_map.items() if v},
            "referenced_by": {k: list(v) for k, v in referenced_by_map.items() if v},
            "total_cross_references": sum(len(v) for v in reference_map.values())
        }

    def generate_narrative_summary(self, events: List[Dict], patterns: Dict) -> str:
        """Generate narrative summary of timeline"""
        if not events:
            return "No events in timeline."

        summary_parts = []
        first_event = events[0]
        last_event = events[-1]

        summary_parts.append(
            f"Memory journey from {first_event['timestamp']} to {last_event['timestamp']}."
        )

        if patterns.get("total_duration_days"):
            summary_parts.append(
                f"Spanning {patterns['total_duration_days']} days with {len(events)} total memory events."
            )

        bursts = patterns.get("bursts", [])
        if bursts:
            summary_parts.append(f"Identified {len(bursts)} burst(s) of intensive activity:")
            for i, burst in enumerate(bursts[:3], 1):
                summary_parts.append(
                    f"  - Burst {i}: {burst['event_count']} events in {burst['duration_hours']}h "
                    f"(intensity: {burst['intensity']} events/hour) from {burst['start']}"
                )

        gaps = patterns.get("gaps", [])
        if gaps:
            summary_parts.append(f"Detected {len(gaps)} significant gap(s) in memory activity:")
            for i, gap in enumerate(gaps[:3], 1):
                summary_parts.append(
                    f"  - Gap {i}: {gap['duration_days']} days of silence from {gap['start']} to {gap['end']}"
                )

        categories = {}
        for event in events:
            cat = event.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        if categories:
            top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
            summary_parts.append(
                f"Primary categories: {', '.join(f'{cat} ({count})' for cat, count in top_cats)}"
            )

        return "\n".join(summary_parts)
