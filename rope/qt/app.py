import argparse
import os
import sys
from pathlib import Path

# `import rope` triggers rope/__init__.py → rope/_native_dlls.py, which
# (a) registers TRT runtime DLL directories and (b) sets
# ORT_LOG_SEVERITY_LEVEL=3 BEFORE ORT is imported anywhere. We keep the
# belt-and-braces set_default_logger_severity(3) call below for the
# case where ORT was imported by some other side-effect import first.

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from rope.qt import paths
from rope.qt.bus import bus
from rope.qt.coordinator import Coordinator
from rope.qt.main_window import MainWindow
from rope.qt.settings import Settings

# Fallback UI fonts when "Segoe UI" (the stylesheet's Windows default)
# isn't installed — first match wins.
_PLATFORM_FONTS = {
    "darwin": ".AppleSystemUIFont",
    "linux": "Noto Sans",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Rope",
        description="Rope face-swap GUI.",
    )
    parser.add_argument(
        "--config-dir", metavar="DIR",
        help="Where data.json, saved_parameters.json, presets/ and user.qss "
             "live (default: $ROPE_HOME, else the repo root).",
    )
    parser.add_argument("--models-dir", metavar="DIR",
                        help="Models folder for this run (overrides the saved setting).")
    parser.add_argument("--output-dir", metavar="DIR",
                        help="Output folder for recordings/images for this run.")
    parser.add_argument("--preset", metavar="NAME_OR_FILE",
                        help="Parameter preset name (from presets/) or a JSON "
                             "file path to apply at startup.")
    parser.add_argument("--stylesheet", metavar="QSS",
                        help="Extra Qt stylesheet appended after rope.qss "
                             "(default: <config dir>/user.qss if present).")
    parser.add_argument("--no-backend", action="store_true",
                        help="Start the UI without loading models (UI testing).")
    parser.add_argument("--print-config", action="store_true",
                        help="Print resolved config paths and exit.")
    # Qt consumes its own flags (-platform, -style ...); ignore unknowns.
    args, _unknown = parser.parse_known_args(argv)
    return args


def _load_stylesheet(app: QApplication, extra: str | None = None) -> None:
    css = ""
    qss_path = Path(__file__).with_name("rope.qss")
    if qss_path.is_file():
        css = qss_path.read_text(encoding="utf-8")
    user_qss = Path(extra) if extra else paths.user_stylesheet()
    if user_qss.is_file():
        css += "\n/* ---- user stylesheet: %s ---- */\n" % user_qss
        css += user_qss.read_text(encoding="utf-8")
    elif extra:
        print(f"[app] stylesheet not found: {extra}")
    app.setStyleSheet(css)


def _apply_font(app: QApplication, settings: Settings) -> None:
    """Honour ui_font_family / ui_font_size from data.json, else swap the
    Windows-only "Segoe UI" for a native font on macOS/Linux."""
    from PySide6.QtGui import QFontDatabase

    family = settings.ui_font_family
    if not family and "Segoe UI" not in QFontDatabase.families():
        for prefix, candidate in _PLATFORM_FONTS.items():
            if sys.platform.startswith(prefix):
                family = candidate
                break
    if not family and not settings.ui_font_size:
        return
    size = settings.ui_font_size or 9
    # The stylesheet's `font:` rules beat QApplication.setFont, so
    # override them with a trailing rule as well.
    fam = f' font-family: "{family}";' if family else ""
    # Buttons and tabs keep their 1pt-larger size from rope.qss.
    css = (f"\nQWidget {{ font-size: {size}pt;{fam} }}\n"
           f"QPushButton, QTabBar::tab {{ font-size: {size + 1}pt;{fam} }}\n")
    app.setStyleSheet(app.styleSheet() + css)
    font = QFont(family) if family else app.font()
    font.setPointSize(size)
    app.setFont(font)


def _resolve_preset(spec: str) -> dict:
    from rope.qt import presets
    from rope.qt.parameters_migration import load as load_params

    p = Path(spec).expanduser()
    if p.suffix.lower() == ".json" and p.is_file():
        return load_params(p)
    return presets.load_preset(spec)


def run(skip_backend: bool = False, argv: list[str] | None = None) -> int:
    """Launch the Qt version of Rope.

    skip_backend=True instantiates only the window/Bus/Coordinator skeleton
    without loading Models/VideoManager. Useful for smoke testing where
    loading ONNX models is slow and unnecessary. Also via --no-backend.
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.config_dir:
        paths.set_config_dir(args.config_dir)
    if args.print_config:
        print(f"config dir:        {paths.config_dir()}")
        print(f"settings:          {paths.data_json()}")
        print(f"quick-save params: {paths.saved_parameters_json()}")
        print(f"presets:           {paths.presets_dir()}")
        print(f"user stylesheet:   {paths.user_stylesheet()}")
        return 0
    skip_backend = skip_backend or args.no_backend

    if sys.platform == "darwin" and QApplication.instance() is None:
        # The preview shaders are "#version 330 core"; macOS only hands out
        # a core profile (max 4.1) when asked, otherwise legacy GL 2.1.
        from PySide6.QtGui import QSurfaceFormat
        fmt = QSurfaceFormat()
        fmt.setVersion(4, 1)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication.instance() or QApplication(sys.argv)
    _load_stylesheet(app, args.stylesheet)
    _apply_font(app, Settings.load())

    if skip_backend:
        models, vm = None, None
    else:
        # Belt-and-braces — env var handles fresh ORT init, this handles
        # the case where ORT was loaded earlier by another import.
        try:
            import onnxruntime
            onnxruntime.set_default_logger_severity(3)
        except (ImportError, AttributeError):
            pass
        import rope.Models as Models
        import rope.VideoManager as VM
        models = Models.Models()
        vm = VM.VideoManager(models)

    window = MainWindow()
    # Apply the persisted Settings.models_folder before any model is
    # lazily loaded. set_models_folder is a no-op when the path matches
    # the default, so first-run startup pays nothing.
    # CLI overrides are per-run: applied to the in-memory settings, and
    # only persisted if the user later changes something that saves.
    if args.models_dir:
        window.settings.models_folder = str(Path(args.models_dir).expanduser().resolve())
        window._params_pane.set_models_folder(window.settings.models_folder)
    if args.output_dir:
        out = Path(args.output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        window.settings.saved_videos = str(out)
        window._params_pane.set_output_folder(window.settings.saved_videos)
    saved_folder = getattr(window.settings, "models_folder", None)
    if saved_folder and models is not None:
        models.set_models_folder(saved_folder)
    coordinator = Coordinator(models=models, vm=vm)

    window.show()
    # Keep a reference so the coordinator isn't GC'd
    window._coordinator = coordinator
    # Seed the VM's recording output folder from persisted settings so the
    # Record button works out of the box. MainWindow.set_output_folder()
    # only updates the params-pane display; the VM's saved_video_path is
    # set via bus.saved_video_path -> coordinator._on_saved_video_path,
    # which only fires when the user picks a folder. Without this, a fresh
    # session leaves vm.saved_video_path == [] and recording would crash in
    # os.path.join at record time.
    if vm is not None:
        saved_videos = getattr(window.settings, "saved_videos", None)
        if saved_videos:
            bus.saved_video_path.emit(saved_videos)
    # Hand the Models instance to the Settings tab so it can render the
    # Loaded column. Skipped under skip_backend (smoke test) since
    # models is None there.
    if models is not None and hasattr(window, "_params_pane"):
        # Apply any persisted per-model backend overrides before the
        # first lazy load. unload=False because nothing is loaded yet,
        # and we don't want a stray vram_updated emit on startup.
        prefs = getattr(window.settings, "model_backends", None) or {}
        for attr_name, backend in prefs.items():
            try:
                models.set_backend_preference(attr_name, backend, unload=False)
            except Exception:
                pass
        window._params_pane.set_models_instance(models)
        # Re-apply the loaded saved_parameters now that `models` is
        # reachable via window._coordinator. _load_saved_parameters ran
        # during MainWindow.__init__ — before the coordinator was
        # assigned — so its emit's _on_params_changed call saw
        # _get_models() return None and skipped the Models-side
        # propagation (set_model_session_mode, update_load_expectation).
        # Re-calling here ensures ModelSessionsTextSel and similar
        # Models-affecting params from saved_parameters.json take effect.
        try:
            window._on_params_changed(dict(window._params_pane.values))
        except Exception as e:
            print(f'[app] late re-apply of saved params failed: {e}')
        # Model preload is no longer automatic at startup — it now happens
        # only when the user clicks the "Preload Models" button (left of
        # Enable Audio in the toggle row; MainWindow._on_preload_models),
        # so the app starts instantly and engines build against whatever
        # backend / thread count the user has actually selected.

    if args.preset:
        values = _resolve_preset(args.preset)
        if values:
            window._params_pane.apply_values(values, emit=True)
            from rope.qt import presets
            window._params_pane.set_presets(presets.list_presets(), args.preset)
            window._tooltip_label.setText(f"Applied preset '{args.preset}'")
        else:
            window._tooltip_label.setText(f"Preset not found or empty: {args.preset}")

    # Show the startup splash in the preview. Staged last (after the
    # late saved-params re-apply above) so any startup frame request
    # can't clobber it; the first real media frame replaces it later.
    window.show_splash()

    return app.exec()
