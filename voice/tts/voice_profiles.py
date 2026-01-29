"""
Voice Profile Management for Friday AI
======================================

Manages TTS voice cloning profiles with database persistence.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from voice.config import get_voice_config


LOGGER = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
VOICE_SAMPLES_DIR = REPO_ROOT / "voice" / "data" / "voice_samples"


@dataclass
class VoiceProfileInfo:
    """Voice profile information"""

    name: str
    description: str
    language: str
    reference_audio_paths: List[str]
    tts_engine: str
    is_active: bool
    is_default: bool

    @property
    def has_audio(self) -> bool:
        """Check if profile has valid reference audio"""
        return len(self.reference_audio_paths) > 0 and all(
            Path(p).exists() for p in self.reference_audio_paths
        )


class VoiceProfileManager:
    """
    Manages voice profiles for TTS voice cloning.

    Supports both file-based and database-backed profiles.

    Usage:
        manager = VoiceProfileManager()

        # Create a new profile
        manager.create_profile(
            name="friday_telugu",
            reference_audio=["voice_sample_1.wav", "voice_sample_2.wav"],
            language="te",
        )

        # Get profile for TTS
        profile = manager.get_profile("friday_telugu")
        tts.synthesize(text, speaker_wav=profile.reference_audio_paths)
    """

    def __init__(self, use_database: bool = False):
        self._use_database = use_database
        self._profiles: dict[str, VoiceProfileInfo] = {}

        # Load default profiles
        self._load_default_profiles()

    def _load_default_profiles(self) -> None:
        """Load built-in profiles"""
        # Default Friday Telugu profile
        friday_te_samples = list(VOICE_SAMPLES_DIR.glob("friday_te_*.wav"))
        if friday_te_samples:
            self._profiles["friday_telugu"] = VoiceProfileInfo(
                name="friday_telugu",
                description="Friday AI Telugu voice - warm, helpful assistant",
                language="te",
                reference_audio_paths=[str(p) for p in friday_te_samples],
                tts_engine="xtts_v2",
                is_active=True,
                is_default=True,
            )
            LOGGER.debug(
                "Loaded friday_telugu profile with %d samples", len(friday_te_samples)
            )

        # Default Friday English profile
        friday_en_samples = list(VOICE_SAMPLES_DIR.glob("friday_en_*.wav"))
        if friday_en_samples:
            self._profiles["friday_english"] = VoiceProfileInfo(
                name="friday_english",
                description="Friday AI English voice",
                language="en",
                reference_audio_paths=[str(p) for p in friday_en_samples],
                tts_engine="xtts_v2",
                is_active=True,
                is_default=False,
            )
            LOGGER.debug(
                "Loaded friday_english profile with %d samples", len(friday_en_samples)
            )

    def create_profile(
        self,
        name: str,
        reference_audio: List[str],
        language: str = "te",
        description: str = "",
        make_default: bool = False,
    ) -> VoiceProfileInfo:
        """
        Create a new voice profile.

        Args:
            name: Profile name (unique identifier)
            reference_audio: List of paths to reference audio files
            language: Primary language code
            description: Profile description
            make_default: Set as default profile

        Returns:
            Created VoiceProfileInfo
        """
        # Validate audio files
        valid_paths = []
        for path in reference_audio:
            p = Path(path)
            if p.exists():
                valid_paths.append(str(p))
            else:
                LOGGER.warning("Reference audio not found: %s", path)

        if not valid_paths:
            raise ValueError("No valid reference audio files provided")

        # If making default, unset current default
        if make_default:
            for profile in self._profiles.values():
                profile.is_default = False

        profile = VoiceProfileInfo(
            name=name,
            description=description or f"Voice profile: {name}",
            language=language,
            reference_audio_paths=valid_paths,
            tts_engine="xtts_v2",
            is_active=True,
            is_default=make_default,
        )

        self._profiles[name] = profile

        if self._use_database:
            self._save_to_database(profile)

        LOGGER.info(
            "Created voice profile '%s' with %d audio files", name, len(valid_paths)
        )
        return profile

    def get_profile(self, name: str) -> Optional[VoiceProfileInfo]:
        """Get a voice profile by name"""
        return self._profiles.get(name)

    def get_default_profile(self) -> Optional[VoiceProfileInfo]:
        """Get the default voice profile"""
        for profile in self._profiles.values():
            if profile.is_default:
                return profile
        # Return first available profile if no default
        if self._profiles:
            return next(iter(self._profiles.values()))
        return None

    def list_profiles(self) -> List[VoiceProfileInfo]:
        """List all voice profiles"""
        return list(self._profiles.values())

    def delete_profile(self, name: str) -> bool:
        """Delete a voice profile"""
        if name in self._profiles:
            del self._profiles[name]
            if self._use_database:
                self._delete_from_database(name)
            LOGGER.info("Deleted voice profile '%s'", name)
            return True
        return False

    def add_reference_audio(
        self,
        profile_name: str,
        audio_path: str,
    ) -> bool:
        """Add reference audio to an existing profile"""
        profile = self._profiles.get(profile_name)
        if not profile:
            LOGGER.error("Profile '%s' not found", profile_name)
            return False

        path = Path(audio_path)
        if not path.exists():
            LOGGER.error("Audio file not found: %s", audio_path)
            return False

        profile.reference_audio_paths.append(str(path))

        if self._use_database:
            self._save_to_database(profile)

        LOGGER.info("Added reference audio to '%s': %s", profile_name, path.name)
        return True

    def _save_to_database(self, profile: VoiceProfileInfo) -> None:
        """Save profile to database"""
        try:
            from sqlalchemy.orm import Session
            from sqlalchemy import create_engine
            from db.voice_schema import VoiceProfile

            db_url = os.environ.get(
                "DATABASE_URL", "postgresql://friday:friday@localhost:5432/friday"
            )
            engine = create_engine(db_url)

            with Session(engine) as session:
                # Check if exists
                existing = (
                    session.query(VoiceProfile).filter_by(name=profile.name).first()
                )

                if existing:
                    existing.description = profile.description
                    existing.language = profile.language
                    existing.reference_audio_paths = profile.reference_audio_paths
                    existing.tts_engine = profile.tts_engine
                    existing.is_active = profile.is_active
                    existing.is_default = profile.is_default
                else:
                    db_profile = VoiceProfile(
                        name=profile.name,
                        description=profile.description,
                        language=profile.language,
                        reference_audio_paths=profile.reference_audio_paths,
                        tts_engine=profile.tts_engine,
                        is_active=profile.is_active,
                        is_default=profile.is_default,
                    )
                    session.add(db_profile)

                session.commit()

        except Exception as e:
            LOGGER.error("Failed to save profile to database: %s", e)

    def _delete_from_database(self, name: str) -> None:
        """Delete profile from database"""
        try:
            from sqlalchemy.orm import Session
            from sqlalchemy import create_engine
            from db.voice_schema import VoiceProfile

            db_url = os.environ.get(
                "DATABASE_URL", "postgresql://friday:friday@localhost:5432/friday"
            )
            engine = create_engine(db_url)

            with Session(engine) as session:
                session.query(VoiceProfile).filter_by(name=name).delete()
                session.commit()

        except Exception as e:
            LOGGER.error("Failed to delete profile from database: %s", e)

    def load_from_database(self) -> int:
        """Load all profiles from database"""
        try:
            from sqlalchemy.orm import Session
            from sqlalchemy import create_engine
            from db.voice_schema import VoiceProfile

            db_url = os.environ.get(
                "DATABASE_URL", "postgresql://friday:friday@localhost:5432/friday"
            )
            engine = create_engine(db_url)

            with Session(engine) as session:
                db_profiles = (
                    session.query(VoiceProfile).filter_by(is_active=True).all()
                )

                for db_profile in db_profiles:
                    self._profiles[db_profile.name] = VoiceProfileInfo(
                        name=db_profile.name,
                        description=db_profile.description or "",
                        language=db_profile.language,
                        reference_audio_paths=db_profile.reference_audio_paths or [],
                        tts_engine=db_profile.tts_engine,
                        is_active=db_profile.is_active,
                        is_default=db_profile.is_default,
                    )

                LOGGER.info("Loaded %d profiles from database", len(db_profiles))
                return len(db_profiles)

        except Exception as e:
            LOGGER.error("Failed to load profiles from database: %s", e)
            return 0
