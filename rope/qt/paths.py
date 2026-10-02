"""Central path resolution for Rope's user files.

Everything that used to be a bare relative path (``data.json``,
``saved_parameters.json``, ``./rope/media/...``) resolved against the
process CWD, so launching Rope from anywhere but the repo root silently
lost settings and icons. All of those now resolve through here.

Config directory precedence:
    1. ``set_config_dir()`` (the ``--config-dir`` CLI flag)
    2. ``ROPE_HOME`` environment variable
    3. the repo root (backwards compatible with existing installs)

Layout inside the config directory::

    data.json               window/folder settings (rope.qt.settings)
    saved_parameters.json   quick-save slot (Ctrl+S / Load Params)
    presets/<name>.json     named parameter presets
    user.qss                optional stylesheet appended after rope.qss
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROPE_PKG = REPO_ROOT / "rope"

_config_dir_override: Path | None = None


def set_config_dir(path: str | os.PathLike | None) -> None:
    global _config_dir_override
    _config_dir_override = Path(path).expanduser().resolve() if path else None


def config_dir() -> Path:
    if _config_dir_override is not None:
        d = _config_dir_override
    elif os.environ.get("ROPE_HOME"):
        d = Path(os.environ["ROPE_HOME"]).expanduser().resolve()
    else:
        d = REPO_ROOT
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_json() -> Path:
    return config_dir() / "data.json"


def saved_parameters_json() -> Path:
    return config_dir() / "saved_parameters.json"


def presets_dir() -> Path:
    d = config_dir() / "presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_stylesheet() -> Path:
    return config_dir() / "user.qss"


def resolve_asset(path: str | os.PathLike | None) -> Path | None:
    """Resolve a bundled asset path such as ``./rope/media/play.png``.

    Absolute paths and paths that exist relative to the CWD are returned
    as-is; otherwise the path is tried relative to the repo root.
    """
    if not path:
        return None
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p
