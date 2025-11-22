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
