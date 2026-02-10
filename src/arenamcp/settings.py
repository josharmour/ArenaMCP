"""Persistent settings for ArenaMCP standalone coach.

Settings are stored in ~/.arenamcp/settings.json and persist between sessions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Settings file location
SETTINGS_DIR = Path.home() / ".arenamcp"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"

# Default settings
DEFAULTS = {
    "voice": "am_adam",  # American Male - Adam
    "voice_speed": 1.0,
    "muted": False,
    "backend": "proxy",
    "auto_speak": True,
    "voice_mode": "ptt",
    "device_index": None,
}


class Settings:
    """Persistent settings manager.

    Example:
        settings = Settings()
        voice = settings.get("voice")
        settings.set("voice", "af_heart")
        settings.save()
    """

    def __init__(self) -> None:
        """Initialize settings, loading from disk if available."""
        self._data: dict[str, Any] = DEFAULTS.copy()
        self._load()

    def _load(self) -> None:
        """Load settings from disk."""
        if not SETTINGS_FILE.exists():
            return

        try:
            with open(SETTINGS_FILE, "r") as f:
                loaded = json.load(f)
                # Merge with defaults (new settings get defaults)
                for key, value in loaded.items():
                    self._data[key] = value
            logger.debug(f"Loaded settings from {SETTINGS_FILE}")
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")

    def save(self) -> None:
        """Save settings to disk."""
        try:
            SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(SETTINGS_FILE, "w") as f:
                json.dump(self._data, f, indent=2)
            logger.debug(f"Saved settings to {SETTINGS_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save settings: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value.

        Args:
            key: Setting key
            default: Default value if key not found

        Returns:
            Setting value or default
        """
        return self._data.get(key, default if default is not None else DEFAULTS.get(key))

    def set(self, key: str, value: Any, save: bool = True) -> None:
        """Set a setting value.

        Args:
            key: Setting key
            value: Value to set
            save: If True (default), immediately save to disk
        """
        self._data[key] = value
        if save:
            self.save()

    def reset(self) -> None:
        """Reset all settings to defaults."""
        self._data = DEFAULTS.copy()
        self.save()


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
