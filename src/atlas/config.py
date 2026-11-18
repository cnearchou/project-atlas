"""Configuration loading.

Spec §0.1 and §A.1: the settings file holds every tunable number of the specification (mandate,
scenario thresholds, cleaning limits, model settings), so that the code itself contains no hidden
constants. API keys are read from environment variables, which are loaded from a local .env file
that is never committed to GitHub (Technical Reference, Part 2.1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The project root is two levels above this file: src/atlas/config.py -> project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = PROJECT_ROOT / "config" / "settings.yaml"


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from a .env file into os.environ.

    A minimal reader is used so that python-dotenv is optional. Existing environment variables are
    never overwritten, so values set in the shell take priority.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    """Typed access to config/settings.yaml plus helpers for paths and secrets."""

    raw: dict[str, Any] = field(default_factory=dict)
    root: Path = PROJECT_ROOT

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, name: str) -> Path:
        """Return an absolute path for one of the storage layers in Spec §0.1."""
        p = self.root / self.raw["paths"][name]
        p.mkdir(parents=True, exist_ok=True) if p.suffix == "" else p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def secret(name: str, required: bool = False) -> str | None:
        """Return an API key from the environment (loaded from .env).

        Keys are never written into code or logs. A missing required key raises an error that names
        the variable, so the user can add it to .env.
        """
        value = os.environ.get(name)
        if required and not value:
            raise RuntimeError(f"Environment variable {name} is not set. Add it to the .env file.")
        return value


def load_settings(path: str | Path | None = None, root: str | Path | None = None) -> Settings:
    """Load the YAML settings and the .env file."""
    root = Path(root) if root else PROJECT_ROOT
    _load_dotenv(root / ".env")
    with open(path or DEFAULT_SETTINGS) as fh:
        raw = yaml.safe_load(fh)
    return Settings(raw=raw, root=root)
