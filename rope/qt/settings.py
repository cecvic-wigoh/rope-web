"""Persistent app settings — reads/writes data.json in the config dir
(see rope.qt.paths: --config-dir, $ROPE_HOME, or the repo root).

Same on-disk schema as the legacy Tk GUI (rope/GUI.py:112-118):

    {
        "source videos":          str | None,
        "source faces":           str | None,
        "saved videos":           str | None,
        "merged_embeddings_file": str | None,
        "dock_win_geom":          [width, height, x, y],
        "splitter_main_sizes":    [left, center, right],   # new in Qt port
        "splitter_left_sizes":    [videos, faces],         # new in Qt port
    }

New keys are additive — the Tk version ignores them, and the Qt version
tolerates their absence.

User-customizable keys (edit data.json while Rope is closed):

    "shortcuts":      {action: key sequence}, overrides DEFAULT_SHORTCUTS;
                      an empty string disables that shortcut
    "nudge_frames":   frames skipped by the nudge_back/nudge_forward keys
    "ui_font_family": UI font family (None = platform default)
    "ui_font_size":   UI font point size (None = stylesheet default)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rope.qt import paths

# Action name -> default key sequence. Users override any subset via the
# "shortcuts" dict in data.json; MainWindow maps action names to handlers.
DEFAULT_SHORTCUTS: dict[str, str] = {
    "play_pause": "Space",
    "timeline_start": "Q",
    "nudge_back": "A",
    "nudge_forward": "D",
    "frame_back": "Left",
    "frame_forward": "Right",
    "seek_start": "Home",
    "seek_end": "End",
    "add_marker": "M",
    "delete_marker": "Shift+M",
    "prev_marker": "Shift+,",
    "next_marker": "Shift+.",
    "save_params": "Ctrl+S",
    "save_preset_as": "Ctrl+Shift+S",
    "toggle_hud": "F3",
}


@dataclass
class Settings:
    source_videos: str | None = None
    source_faces: str | None = None
    saved_videos: str | None = None
    merged_embeddings_file: str | None = None
    # Models folder location. None = use the default './models' next to
    # the repo root (the legacy hardcoded path). Selectable from the
    # Settings tab so users can put weights on a separate drive without
    # symlinks.
    models_folder: str | None = None
    dock_win_geom: list[int] = field(default_factory=lambda: [1600, 950, 100, 80])
    splitter_main_sizes: list[int] = field(default_factory=lambda: [340, 880, 380])
    splitter_left_sizes: list[int] = field(default_factory=lambda: [500, 400])
    # Vertical split inside the center pane: [top half (video + chrome +
    # found-faces), embeddings]. Drag handle lives between Found Faces
    # and Embeddings.
    splitter_center_sizes: list[int] = field(default_factory=lambda: [700, 180])
    # Collapsed state for each parameters-pane section, keyed by title.
    # Missing entries default to expanded (False).
    params_collapsed: dict[str, bool] = field(default_factory=dict)
    # Per-model backend preference. Keys are Models attribute names
    # (e.g. "swapper_model", "retinaface_model"); values are "trt" or
    # "onnx". A missing entry means "auto" — prefer TRT if the engine
    # file is on disk, fall back to ONNX. Set via the Settings tab's
    # Backend toggle and applied at the next lazy model load.
    model_backends: dict[str, str] = field(default_factory=dict)
    shortcuts: dict[str, str] = field(default_factory=dict)
    nudge_frames: int = 30
    ui_font_family: str | None = None
    ui_font_size: int | None = None
    # Name of the preset last loaded/saved, re-selected in the preset box.
    last_preset: str | None = None

    def shortcut_map(self) -> dict[str, str]:
        """DEFAULT_SHORTCUTS with the user's overrides applied."""
        merged = dict(DEFAULT_SHORTCUTS)
        merged.update({k: v for k, v in self.shortcuts.items() if k in merged})
        return merged

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Settings":
        p = Path(path) if path is not None else paths.data_json()
        if not p.is_file():
            return cls()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        pc_raw = raw.get("params_collapsed", {})
        params_collapsed = {
            str(k): bool(v) for k, v in pc_raw.items()
        } if isinstance(pc_raw, dict) else {}
        mb_raw = raw.get("model_backends", {})
        model_backends = {
            str(k): str(v) for k, v in mb_raw.items()
            if v in ("trt", "onnx")
        } if isinstance(mb_raw, dict) else {}
        return cls(
            source_videos=raw.get("source videos"),
            source_faces=raw.get("source faces"),
            saved_videos=raw.get("saved videos"),
            merged_embeddings_file=raw.get("merged_embeddings_file"),
            models_folder=raw.get("models_folder"),
            dock_win_geom=list(raw.get("dock_win_geom", [1600, 950, 100, 80])),
            splitter_main_sizes=list(raw.get("splitter_main_sizes", [340, 880, 380])),
            splitter_left_sizes=list(raw.get("splitter_left_sizes", [500, 400])),
            splitter_center_sizes=list(raw.get("splitter_center_sizes", [700, 180])),
            params_collapsed=params_collapsed,
            model_backends=model_backends,
            shortcuts={
                str(k): str(v) for k, v in raw.get("shortcuts", {}).items()
            } if isinstance(raw.get("shortcuts"), dict) else {},
            nudge_frames=_int_or(raw.get("nudge_frames"), 30, minimum=1),
            ui_font_family=raw.get("ui_font_family") or None,
            ui_font_size=_int_or(raw.get("ui_font_size"), None, minimum=6),
            last_preset=raw.get("last_preset") or None,
        )

    def save(self, path: Path | str | None = None) -> None:
        out: dict[str, Any] = {
            "source videos": self.source_videos,
            "source faces": self.source_faces,
            "saved videos": self.saved_videos,
            "merged_embeddings_file": self.merged_embeddings_file,
            "models_folder": self.models_folder,
            "dock_win_geom": list(self.dock_win_geom),
            "splitter_main_sizes": list(self.splitter_main_sizes),
            "splitter_left_sizes": list(self.splitter_left_sizes),
            "splitter_center_sizes": list(self.splitter_center_sizes),
            "params_collapsed": dict(self.params_collapsed),
            "model_backends": dict(self.model_backends),
            "shortcuts": dict(self.shortcuts),
            "nudge_frames": int(self.nudge_frames),
            "ui_font_family": self.ui_font_family,
            "ui_font_size": self.ui_font_size,
            "last_preset": self.last_preset,
        }
        target = Path(path) if path is not None else paths.data_json()
        # Write-then-rename so a crash mid-write can't truncate data.json.
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        tmp.replace(target)


def _int_or(value: Any, default: Any, *, minimum: int) -> Any:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def shorten_path(path: str | None, max_len: int = 28) -> str:
    """Mirror of rope/GUI.py:create_path_string — trim long paths for display."""
    if not path:
        return ""
    if len(path) <= max_len:
        return path
    import os
    last_folder = os.path.basename(os.path.normpath(path))
    if len(last_folder) > max_len:
        return path[:3] + "..." + path[-max_len + 6:]
    return path[: max_len - len(last_folder)] + ".../" + path[-len(last_folder):]
