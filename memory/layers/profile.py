"""
Profile Store
=============

Persistent identity facts that NEVER decay.

Features:
    - Static facts (almost never change)
    - Dynamic state (changes with context)
    - Learned preferences (updated from patterns)
    - Relationships (people Friday knows about)
    - Projects (all known projects)
    - Version history with rollback

Brain Inspiration:
    Core identity and self-knowledge that forms the
    foundation of personality and behavior.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory.config import ProfileConfig, get_memory_config

LOGGER = logging.getLogger(__name__)


class VoiceConfirmationRequired(Exception):
    """Raised when a profile change requires voice confirmation"""

    pass


@dataclass
class Relationship:
    """A relationship Friday knows about"""

    name: str
    relation: str  # character, family, friend, colleague
    project: Optional[str]  # Project context if applicable
    notes: str = ""
    added_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "relation": self.relation,
            "project": self.project,
            "notes": self.notes,
            "added_at": self.added_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        return cls(
            name=data["name"],
            relation=data["relation"],
            project=data.get("project"),
            notes=data.get("notes", ""),
            added_at=(
                datetime.fromisoformat(data["added_at"])
                if data.get("added_at")
                else datetime.now()
            ),
        )


@dataclass
class Project:
    """A project Friday knows about"""

    name: str
    slug: str  # URL-safe identifier
    status: str  # active, paused, completed, archived
    description: str = ""
    deadline: Optional[datetime] = None
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "slug": self.slug,
            "status": self.status,
            "description": self.description,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        return cls(
            name=data["name"],
            slug=data["slug"],
            status=data["status"],
            description=data.get("description", ""),
            deadline=(
                datetime.fromisoformat(data["deadline"])
                if data.get("deadline")
                else None
            ),
            notes=data.get("notes", ""),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now()
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at")
                else datetime.now()
            ),
        )


@dataclass
class UserProfile:
    """
    The complete user profile.

    Contains:
        - static: Facts that almost never change (name, profession)
        - dynamic: Facts that change with context (current project)
        - preferences: Learned preferences (communication style)
        - relationships: People Friday knows about
        - projects: All known projects
    """

    # Static identity (almost never changes)
    static: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "Poorna",
            "role": "Telugu screenwriter",
            "languages": ["Telugu", "English"],
            "address_as": "Boss",
            "communication_style": "Direct, no flattery, concise",
        }
    )

    # Dynamic state (changes with context)
    dynamic: Dict[str, Any] = field(
        default_factory=lambda: {
            "current_project": None,
            "current_room": "general",
            "recent_mood": "neutral",
            "active_deadlines": [],
            "last_interaction": None,
        }
    )

    # Learned preferences (updated from patterns)
    preferences: Dict[str, Any] = field(
        default_factory=lambda: {
            "response_length": "concise",
            "telugu_usage": "natural, for emotions",
            "feedback_style": "direct with opinions",
            "working_hours": "flexible",
        }
    )

    # Relationships (people Friday knows about)
    relationships: Dict[str, Relationship] = field(default_factory=dict)

    # Projects (all known projects)
    projects: Dict[str, Project] = field(default_factory=dict)

    # Metadata
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to dictionary"""
        return {
            "static": self.static,
            "dynamic": self.dynamic,
            "preferences": self.preferences,
            "relationships": {k: v.to_dict() for k, v in self.relationships.items()},
            "projects": {k: v.to_dict() for k, v in self.projects.items()},
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Deserialize profile from dictionary"""
        profile = cls()
        profile.static = data.get("static", profile.static)
        profile.dynamic = data.get("dynamic", profile.dynamic)
        profile.preferences = data.get("preferences", profile.preferences)

        # Relationships
        profile.relationships = {
            k: Relationship.from_dict(v)
            for k, v in data.get("relationships", {}).items()
        }

        # Projects
        profile.projects = {
            k: Project.from_dict(v) for k, v in data.get("projects", {}).items()
        }

        profile.version = data.get("version", 1)
        profile.created_at = (
            datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now()
        )
        profile.updated_at = (
            datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else datetime.now()
        )

        return profile


class ProfileStore:
    """
    Profile storage with version history.

    Manages persistent identity facts that NEVER decay.

    Features:
        - JSON file storage with version control
        - Voice confirmation for static changes
        - Automatic backup before changes
        - Rollback capability

    Usage:
        store = ProfileStore()
        await store.initialize()

        # Get profile
        profile = store.profile

        # Update dynamic state
        store.update_dynamic("current_project", "gusagusalu")

        # Update static (requires voice confirmation)
        store.update_static("name", "New Name", voice_confirmed=True)

        # Learn preference
        store.learn_preference("response_length", "verbose", confidence=0.9)
    """

    def __init__(self, config: Optional[ProfileConfig] = None):
        self.config = config or get_memory_config().profile
        self._profile_path = Path(self.config.profile_path)
        self._history_path = Path(self.config.history_path)
        self._profile: Optional[UserProfile] = None
        self._audit_log: List[Dict[str, Any]] = []

    async def initialize(self) -> None:
        """Initialize profile store"""
        self._profile_path.parent.mkdir(parents=True, exist_ok=True)
        self._history_path.mkdir(parents=True, exist_ok=True)

        # Load or create profile
        if self._profile_path.exists():
            self._load()
        else:
            self._profile = UserProfile()
            self._save()

        LOGGER.info("Profile store initialized: %s", self._profile_path)

    def _load(self) -> None:
        """Load profile from file"""
        try:
            with open(self._profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._profile = UserProfile.from_dict(data)
            LOGGER.info("Loaded profile v%d", self._profile.version)
        except Exception as e:
            LOGGER.error("Failed to load profile: %s", e)
            # Try to recover from history
            self._recover_from_history()

    def _save(self) -> None:
        """Save profile to file"""
        # Backup before save
        self._backup()

        # Update metadata
        self._profile.updated_at = datetime.now()

        # Save
        with open(self._profile_path, "w", encoding="utf-8") as f:
            json.dump(self._profile.to_dict(), f, ensure_ascii=False, indent=2)

        LOGGER.debug("Saved profile v%d", self._profile.version)

    def _backup(self) -> None:
        """Create backup in history"""
        if not self._profile_path.exists():
            return

        # Generate history filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        version = self._profile.version if self._profile else 0
        history_file = self._history_path / f"v{version}_{timestamp}.json"

        # Copy current profile to history
        shutil.copy2(self._profile_path, history_file)

        # Cleanup old history (keep max_history_versions)
        self._cleanup_history()

        LOGGER.debug("Backed up profile to %s", history_file.name)

    def _cleanup_history(self) -> None:
        """Remove old history files beyond retention limit"""
        history_files = sorted(self._history_path.glob("*.json"), reverse=True)

        if len(history_files) > self.config.max_history_versions:
            for old_file in history_files[self.config.max_history_versions :]:
                old_file.unlink()
                LOGGER.debug("Removed old history: %s", old_file.name)

    def _recover_from_history(self) -> None:
        """Recover profile from most recent history"""
        history_files = sorted(self._history_path.glob("*.json"), reverse=True)

        for history_file in history_files:
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._profile = UserProfile.from_dict(data)
                LOGGER.warning("Recovered profile from %s", history_file.name)
                self._save()
                return
            except Exception as e:
                LOGGER.warning("Failed to recover from %s: %s", history_file.name, e)
                continue

        # No valid history, create new profile
        LOGGER.warning("No valid history found, creating new profile")
        self._profile = UserProfile()
        self._save()

    def _audit(
        self, operation: str, key: str, old_value: Any, new_value: Any, **kwargs
    ) -> None:
        """Log an audit entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "key": key,
            "old_value": old_value,
            "new_value": new_value,
            **kwargs,
        }
        self._audit_log.append(entry)

        # Keep last 1000 entries
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]

        LOGGER.debug("Audit: %s %s", operation, key)

    # =========================================================================
    # Profile Access
    # =========================================================================

    @property
    def profile(self) -> UserProfile:
        """Get current profile"""
        return self._profile

    def get_static(self, key: str, default: Any = None) -> Any:
        """Get a static fact"""
        return self._profile.static.get(key, default)

    def get_dynamic(self, key: str, default: Any = None) -> Any:
        """Get a dynamic state value"""
        return self._profile.dynamic.get(key, default)

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a preference"""
        return self._profile.preferences.get(key, default)

    def get_relationship(self, name: str) -> Optional[Relationship]:
        """Get a relationship by name"""
        return self._profile.relationships.get(name.lower())

    def get_project(self, slug: str) -> Optional[Project]:
        """Get a project by slug"""
        return self._profile.projects.get(slug.lower())

    # =========================================================================
    # Static Updates (Require Voice Confirmation)
    # =========================================================================

    def update_static(
        self,
        key: str,
        value: Any,
        voice_confirmed: bool = False,
    ) -> None:
        """
        Update a static fact.

        Static facts require voice confirmation to prevent
        accidental identity drift.

        Args:
            key: The fact key
            value: New value
            voice_confirmed: Whether user confirmed via voice

        Raises:
            VoiceConfirmationRequired: If voice confirmation not provided
        """
        if not voice_confirmed:
            raise VoiceConfirmationRequired(
                f"Changing static profile '{key}' requires voice confirmation. "
                f"Say: 'Friday, update my profile: {key} is {value}'"
            )

        old_value = self._profile.static.get(key)

        if old_value != value:
            self._profile.static[key] = value
            self._profile.version += 1
            self._audit("static_update", key, old_value, value, voice_confirmed=True)
            self._save()

            LOGGER.info("Updated static '%s': %s -> %s", key, old_value, value)

    # =========================================================================
    # Dynamic Updates (Auto-update)
    # =========================================================================

    def update_dynamic(self, key: str, value: Any) -> None:
        """
        Update a dynamic state value.

        Dynamic state auto-updates without confirmation.
        """
        old_value = self._profile.dynamic.get(key)

        if old_value != value:
            self._profile.dynamic[key] = value
            self._profile.dynamic["last_updated"] = datetime.now().isoformat()
            self._audit("dynamic_update", key, old_value, value)
            self._save()

            LOGGER.debug("Updated dynamic '%s': %s -> %s", key, old_value, value)

    def set_current_project(self, project_slug: Optional[str]) -> None:
        """Set the current active project"""
        self.update_dynamic("current_project", project_slug)

    def set_current_room(self, room: str) -> None:
        """Set the current room/context"""
        self.update_dynamic("current_room", room)

    def set_mood(self, mood: str) -> None:
        """Set detected mood"""
        self.update_dynamic("recent_mood", mood)

    def record_interaction(self) -> None:
        """Record that an interaction occurred"""
        self.update_dynamic("last_interaction", datetime.now().isoformat())

    # =========================================================================
    # Preference Learning
    # =========================================================================

    def learn_preference(
        self,
        key: str,
        value: Any,
        confidence: float,
        min_confidence: float = 0.8,
    ) -> bool:
        """
        Learn a preference from observed patterns.

        Only updates if confidence exceeds threshold.

        Args:
            key: Preference key
            value: Observed preference value
            confidence: Confidence level (0-1)
            min_confidence: Minimum confidence to update

        Returns:
            Whether the preference was updated
        """
        if confidence < min_confidence:
            LOGGER.debug(
                "Preference '%s' confidence too low: %.2f < %.2f",
                key,
                confidence,
                min_confidence,
            )
            return False

        old_value = self._profile.preferences.get(key)

        if old_value != value:
            self._profile.preferences[key] = value
            self._audit(
                "preference_learned", key, old_value, value, confidence=confidence
            )
            self._save()

            LOGGER.info(
                "Learned preference '%s': %s -> %s (confidence: %.2f)",
                key,
                old_value,
                value,
                confidence,
            )
            return True

        return False

    # =========================================================================
    # Relationship Management
    # =========================================================================

    def add_relationship(
        self,
        name: str,
        relation: str,
        project: Optional[str] = None,
        notes: str = "",
    ) -> Relationship:
        """Add a new relationship"""
        key = name.lower()
        rel = Relationship(
            name=name,
            relation=relation,
            project=project,
            notes=notes,
        )

        old = self._profile.relationships.get(key)
        self._profile.relationships[key] = rel
        self._audit(
            "relationship_add", key, old.to_dict() if old else None, rel.to_dict()
        )
        self._save()

        LOGGER.info("Added relationship: %s (%s)", name, relation)
        return rel

    def update_relationship(self, name: str, **kwargs) -> Optional[Relationship]:
        """Update a relationship"""
        key = name.lower()
        rel = self._profile.relationships.get(key)

        if not rel:
            return None

        old_data = rel.to_dict()

        for k, v in kwargs.items():
            if hasattr(rel, k):
                setattr(rel, k, v)

        self._audit("relationship_update", key, old_data, rel.to_dict())
        self._save()

        return rel

    def remove_relationship(self, name: str) -> bool:
        """Remove a relationship"""
        key = name.lower()

        if key in self._profile.relationships:
            old = self._profile.relationships.pop(key)
            self._audit("relationship_remove", key, old.to_dict(), None)
            self._save()
            LOGGER.info("Removed relationship: %s", name)
            return True

        return False

    def list_relationships(
        self,
        relation: Optional[str] = None,
        project: Optional[str] = None,
    ) -> List[Relationship]:
        """List relationships with optional filters"""
        results = list(self._profile.relationships.values())

        if relation:
            results = [r for r in results if r.relation == relation]

        if project:
            results = [r for r in results if r.project == project]

        return results

    # =========================================================================
    # Project Management
    # =========================================================================

    def add_project(
        self,
        name: str,
        slug: str,
        status: str = "active",
        description: str = "",
        deadline: Optional[datetime] = None,
        notes: str = "",
    ) -> Project:
        """Add a new project"""
        key = slug.lower()
        proj = Project(
            name=name,
            slug=key,
            status=status,
            description=description,
            deadline=deadline,
            notes=notes,
        )

        old = self._profile.projects.get(key)
        self._profile.projects[key] = proj
        self._audit("project_add", key, old.to_dict() if old else None, proj.to_dict())
        self._save()

        LOGGER.info("Added project: %s (%s)", name, status)
        return proj

    def update_project(self, slug: str, **kwargs) -> Optional[Project]:
        """Update a project"""
        key = slug.lower()
        proj = self._profile.projects.get(key)

        if not proj:
            return None

        old_data = proj.to_dict()

        for k, v in kwargs.items():
            if hasattr(proj, k):
                setattr(proj, k, v)

        proj.updated_at = datetime.now()
        self._audit("project_update", key, old_data, proj.to_dict())
        self._save()

        return proj

    def set_project_status(self, slug: str, status: str) -> Optional[Project]:
        """Set project status"""
        return self.update_project(slug, status=status)

    def list_projects(self, status: Optional[str] = None) -> List[Project]:
        """List projects with optional status filter"""
        results = list(self._profile.projects.values())

        if status:
            results = [p for p in results if p.status == status]

        return sorted(results, key=lambda p: p.updated_at, reverse=True)

    def get_active_projects(self) -> List[Project]:
        """Get all active projects"""
        return self.list_projects(status="active")

    # =========================================================================
    # History & Rollback
    # =========================================================================

    def list_history(self, limit: int = 20) -> List[Path]:
        """List history versions"""
        history_files = sorted(self._history_path.glob("*.json"), reverse=True)
        return history_files[:limit]

    def rollback_to_version(self, version_file: Path) -> bool:
        """Rollback to a specific version"""
        if not version_file.exists():
            LOGGER.error("Version file not found: %s", version_file)
            return False

        try:
            # Backup current before rollback
            self._backup()

            # Load old version
            with open(version_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._profile = UserProfile.from_dict(data)
            self._profile.version += 1
            self._save()

            LOGGER.warning("Rolled back to %s", version_file.name)
            return True

        except Exception as e:
            LOGGER.error("Rollback failed: %s", e)
            return False

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries"""
        return self._audit_log[-limit:]

    # =========================================================================
    # Export & Summary
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get profile summary for context building"""
        return {
            "name": self.get_static("name"),
            "role": self.get_static("role"),
            "address_as": self.get_static("address_as"),
            "communication_style": self.get_static("communication_style"),
            "languages": self.get_static("languages"),
            "current_project": self.get_dynamic("current_project"),
            "current_room": self.get_dynamic("current_room"),
            "preferences": self._profile.preferences,
            "active_projects": [p.name for p in self.get_active_projects()],
        }

    def export_for_prompt(self) -> str:
        """Export profile as text for system prompt"""
        lines = [
            f"User: {self.get_static('name')} ({self.get_static('role')})",
            f"Address as: {self.get_static('address_as')}",
            f"Communication: {self.get_static('communication_style')}",
            f"Languages: {', '.join(self.get_static('languages', []))}",
        ]

        current_project = self.get_dynamic("current_project")
        if current_project:
            proj = self.get_project(current_project)
            if proj:
                lines.append(f"Current project: {proj.name} ({proj.status})")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ProfileStore(v{self._profile.version if self._profile else 0})"
