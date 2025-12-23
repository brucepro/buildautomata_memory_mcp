"""
Data models for BuildAutomata Memory System
Copyright 2025 Jurden Bruce
"""

import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, List, Dict, Optional


@dataclass
class MemoryRelationship:
    """Typed relationship between memories"""
    target_memory_id: str
    relationship_type: str  # builds_on | contradicts | implements | analyzes | references
    strength: float  # 0.0-1.0
    created_at: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_memory_id": self.target_memory_id,
            "relationship_type": self.relationship_type,
            "strength": self.strength,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class Memory:
    id: str
    content: str
    category: str
    importance: float
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    # Bi-temporal validity: when the content was/is actually valid
    valid_from: Optional[datetime] = None  # When this knowledge became valid
    valid_until: Optional[datetime] = None  # When this knowledge became invalid (None = still valid)
    related_memories: List[str] = None
    decay_rate: float = 0.95
    version_count: int = 1
    # NEW: Memory type classification
    memory_type: str = "episodic"  # episodic | semantic | working
    # NEW: Session and task context
    session_id: Optional[str] = None
    task_context: Optional[str] = None
    # NEW: Provenance tracking
    provenance: Optional[Dict[str, Any]] = None
    # NEW: Typed relationships
    relationships: Optional[List[MemoryRelationship]] = None
    # Vector similarity score from search (when applicable)
    vector_score: Optional[float] = None

    def __post_init__(self):
        if self.related_memories is None:
            self.related_memories = []
        if self.relationships is None:
            self.relationships = []
        if self.provenance is None:
            self.provenance = {
                "retrieval_queries": [],
                "usage_contexts": [],
                "parent_memory_ids": [],
                "consolidation_date": None,
                "created_by_session": self.session_id
            }
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)
        if self.last_accessed and isinstance(self.last_accessed, str):
            self.last_accessed = datetime.fromisoformat(self.last_accessed)
        if self.valid_from and isinstance(self.valid_from, str):
            self.valid_from = datetime.fromisoformat(self.valid_from)
        if self.valid_until and isinstance(self.valid_until, str):
            self.valid_until = datetime.fromisoformat(self.valid_until)
        # Convert relationship dicts back to objects if needed
        if self.relationships and isinstance(self.relationships[0], dict):
            self.relationships = [
                MemoryRelationship(**r) if isinstance(r, dict) else r
                for r in self.relationships
            ]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.last_accessed:
            data["last_accessed"] = self.last_accessed.isoformat()
        # Convert relationships to dicts
        if self.relationships:
            data["relationships"] = [r.to_dict() if isinstance(r, MemoryRelationship) else r for r in self.relationships]
        return data

    def current_importance(self) -> float:
        """Calculate current importance with decay

        Decay is based on time since last access, or creation date if never accessed.
        This ensures never-used memories decay naturally rather than maintaining
        artificially high importance forever.
        """
        # Use last_accessed if available, otherwise fall back to created_at
        reference_date = self.last_accessed if self.last_accessed else self.created_at

        if not reference_date:
            return self.importance

        days = (datetime.now() - reference_date).days
        return max(0.1, min(1.0, self.importance * (self.decay_rate ** days)))

    def content_hash(self) -> str:
        """Generate hash of memory content for deduplication"""
        content_str = f"{self.content}|{self.category}|{self.importance}|{','.join(sorted(self.tags))}"
        return hashlib.sha256(content_str.encode()).hexdigest()

    @classmethod
    def from_row(cls, row) -> 'Memory':
        """Convert SQLite row to Memory object

        Args:
            row: sqlite3.Row object with keys() method

        Returns:
            Memory instance
        """
        import json

        # Parse provenance and relationships if present
        provenance = None
        if "provenance" in row.keys() and row["provenance"]:
            try:
                provenance = json.loads(row["provenance"])
            except:
                provenance = None

        relationships = []
        if "relationships" in row.keys() and row["relationships"]:
            try:
                rel_dicts = json.loads(row["relationships"])
                relationships = [MemoryRelationship(**r) for r in rel_dicts]
            except:
                relationships = []

        related_memories = []
        if "related_memories" in row.keys() and row["related_memories"]:
            try:
                related_memories = json.loads(row["related_memories"])
            except:
                related_memories = []

        return cls(
            id=row["id"],
            content=row["content"],
            category=row["category"],
            importance=row["importance"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=row["access_count"],
            last_accessed=row["last_accessed"],
            valid_from=row["valid_from"] if "valid_from" in row.keys() else None,
            valid_until=row["valid_until"] if "valid_until" in row.keys() else None,
            decay_rate=row["decay_rate"],
            version_count=row["version_count"] if "version_count" in row.keys() else 1,
            memory_type=row["memory_type"] if "memory_type" in row.keys() else "episodic",
            session_id=row["session_id"] if "session_id" in row.keys() else None,
            task_context=row["task_context"] if "task_context" in row.keys() else None,
            provenance=provenance,
            relationships=relationships,
            related_memories=related_memories,
        )

    def to_api_dict(self, enrich_related: Optional[callable] = None) -> Dict:
        """Convert Memory to dict for API response with statistics

        Args:
            enrich_related: Optional function to enrich related_memories with previews

        Returns:
            Dict suitable for API responses
        """
        days_since_access = 0
        if self.last_accessed:
            days_since_access = (datetime.now() - self.last_accessed).days

        decay_factor = self.decay_rate ** days_since_access if days_since_access > 0 else 1.0

        # Enrich related_memories with previews for autonomous navigation
        related_memories_enriched = []
        if self.related_memories and enrich_related:
            related_memories_enriched = enrich_related(self.related_memories)
        else:
            related_memories_enriched = self.related_memories

        return {
            "memory_id": self.id,
            "content": self.content,
            "category": self.category,
            "importance": self.importance,
            "current_importance": self.current_importance(),
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "days_since_access": days_since_access,
            "decay_factor": round(decay_factor, 3),
            "version_count": self.version_count,
            "related_memories": related_memories_enriched,
        }


@dataclass
class Intention:
    """First-class intention entity for proactive agency"""
    id: str
    description: str
    priority: float  # 0.0 to 1.0
    status: str  # pending, active, completed, cancelled
    created_at: datetime
    updated_at: datetime
    deadline: Optional[datetime] = None
    preconditions: List[str] = None
    actions: List[str] = None
    related_memories: List[str] = None
    metadata: Dict[str, Any] = None
    last_checked: Optional[datetime] = None
    check_count: int = 0

    def __post_init__(self):
        if self.preconditions is None:
            self.preconditions = []
        if self.actions is None:
            self.actions = []
        if self.related_memories is None:
            self.related_memories = []
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)
        if self.deadline and isinstance(self.deadline, str):
            self.deadline = datetime.fromisoformat(self.deadline)
        if self.last_checked and isinstance(self.last_checked, str):
            self.last_checked = datetime.fromisoformat(self.last_checked)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.deadline:
            data["deadline"] = self.deadline.isoformat()
        if self.last_checked:
            data["last_checked"] = self.last_checked.isoformat()
        return data

    def is_overdue(self) -> bool:
        """Check if intention is past its deadline"""
        if not self.deadline:
            return False
        return datetime.now() > self.deadline

    def days_until_deadline(self) -> Optional[float]:
        """Calculate days until deadline"""
        if not self.deadline:
            return None
        delta = self.deadline - datetime.now()
        return delta.total_seconds() / 86400
