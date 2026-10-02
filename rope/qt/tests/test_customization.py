"""Customization features: config dir, presets, shortcuts, CLI, status bar.

Run from the repo root (no GPU needed; Qt runs offscreen):

    python -m rope.qt.tests.test_customization
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"  FAIL  {msg}")
        sys.exit(1)
    print(f"  ok  {msg}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rope_custom_")).resolve()

    from rope.qt import paths, presets
    from rope.qt.settings import DEFAULT_SHORTCUTS, Settings

    # --- paths -----------------------------------------------------------
    paths.set_config_dir(tmp)
    check(paths.data_json() == tmp / "data.json", "data.json resolves into --config-dir")
    check(paths.presets_dir().is_dir(), "presets/ created on demand")
    icon = paths.resolve_asset("./rope/media/play_off.png")
    check(icon is not None and icon.is_file(), "bundled icons resolve independent of CWD")

    # --- settings: shortcuts + new keys round-trip ------------------------
    (tmp / "data.json").write_text(json.dumps({
        "shortcuts": {"play_pause": "P", "bogus_action": "X", "toggle_hud": ""},
        "nudge_frames": "12",
        "ui_font_size": 11,
    }), encoding="utf-8")
    s = Settings.load()
    sm = s.shortcut_map()
    check(sm["play_pause"] == "P", "shortcut override applied")
    check("bogus_action" not in sm, "unknown shortcut actions ignored")
    check(sm["toggle_hud"] == "", "empty string disables a shortcut")
    check(sm["seek_end"] == DEFAULT_SHORTCUTS["seek_end"], "unspecified shortcuts keep defaults")
    check(s.nudge_frames == 12 and s.ui_font_size == 11, "nudge_frames / ui_font_size parsed")
    s.last_preset = "Night"
    s.save()
    check(Settings.load().last_preset == "Night", "settings save/load round-trip")
    check(not (tmp / "data.json.tmp").exists(), "atomic save leaves no temp file")

    # --- presets ---------------------------------------------------------
    name = presets.save_preset('My/Pre:set', {"ThresholdSlider": 70, "NotAParam": 1})
    check(name == "My_Pre_set", "preset names sanitized for the filesystem")
    check(presets.list_presets() == ["My_Pre_set"], "list_presets finds saved preset")
    loaded = presets.load_preset(name)
    check(loaded == {"ThresholdSlider": 70.0}, "preset load filters + coerces values")
    presets.delete_preset(name)
    check(presets.list_presets() == [], "delete_preset removes the file")

    # --- CLI ---------------------------------------------------------------
    from rope.qt.app import parse_args
    a = parse_args(["--config-dir", "/x", "--preset", "Night", "--no-backend", "-platform", "offscreen"])
    check(a.config_dir == "/x" and a.preset == "Night" and a.no_backend, "CLI args parsed (Qt flags tolerated)")

    # --- UI: preset row, shortcuts, status label ----------------------------
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    from rope.qt.coordinator import Coordinator
    from rope.qt.main_window import MainWindow

    presets.save_preset("Strong", {"ThresholdSlider": 81})
    win = MainWindow()
    win._coordinator = Coordinator(models=None, vm=None)
    pane = win._params_pane
    combo = pane._preset_combo
    items = [combo.itemText(i) for i in range(combo.count())]
    check("Strong" in items, "preset dropdown lists presets from disk")

    win._on_preset_selected("Strong")
    check(abs(pane.values["ThresholdSlider"] - 81) < 1e-6, "selecting a preset applies its values")
    check(Settings.load().last_preset == "Strong", "last preset remembered")

    keys = {sc.key().toString() for sc in win._shortcuts}
    check("P" in keys and "Space" not in keys, "custom play shortcut bound, default replaced")
    check("F3" not in keys, "disabled shortcut not bound")

    win._tooltip_label.setText("LINE ONE:\nline two")
    check(win._tooltip_label.isVisibleTo(win), "status/help label is visible")
    check("\n" not in win._tooltip_label.text(), "status text flattened to one line")

    win.close()
    app.processEvents()
    print("all customization checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
