"""Named parameter presets: <config dir>/presets/<name>.json.

Each preset uses the same flat {ParamName: value} format as
saved_parameters.json, so a preset file and the quick-save slot are
interchangeable (and both go through parameters_migration for key
filtering and type coercion).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rope.qt import paths
from rope.qt.parameters_migration import load as load_params, save as save_params

_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_name(name: str) -> str:
    """Make a user-typed preset name safe to use as a filename."""
    cleaned = _INVALID.sub("_", name).strip().strip(".")
    return cleaned[:80]


def preset_path(name: str) -> Path:
    return paths.presets_dir() / f"{sanitize_name(name)}.json"


def list_presets() -> list[str]:
    return sorted(
        (p.stem for p in paths.presets_dir().glob("*.json")),
        key=str.casefold,
    )


def load_preset(name: str) -> dict[str, Any]:
    return load_params(preset_path(name))


def save_preset(name: str, values: dict[str, Any]) -> str:
    """Save and return the sanitized name actually used."""
    safe = sanitize_name(name)
    if not safe:
        raise ValueError("Preset name is empty")
    save_params(values, preset_path(safe))
    return safe


def delete_preset(name: str) -> None:
    preset_path(name).unlink(missing_ok=True)
