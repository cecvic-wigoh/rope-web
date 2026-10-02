"""Rope web app — FaceFusion-style browser UI over the headless engine.

    python webapp/app.py --models-dir models --port 7860

Access is password-protected by default: set ROPE_WEB_USER /
ROPE_WEB_PASSWORD, or a random password is printed at startup.
ROPE_WEB_FAKE=1 swaps the engine for a CPU stub (UI development only).
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

REQUIRED_MODELS = ["det_10g.onnx", "w600k_r50.onnx", "inswapper_128.fp16.onnx"]


def _install_builtin_presets() -> None:
    """Copy webapp/presets/*.json into the presets folder (never
    overwriting a preset the user saved under the same name)."""
    import shutil
    from rope.qt import paths

    for src in sorted((Path(__file__).parent / "presets").glob("*.json")):
        dst = paths.presets_dir() / src.name
        if not dst.exists():
            shutil.copyfile(src, dst)


def _default_backend() -> str:
    try:
        import importlib.util
        return "tensorrt" if importlib.util.find_spec("tensorrt_libs") else "cuda"
    except Exception:
        return "cuda"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rope web app")
    ap.add_argument("--models-dir", default=os.environ.get("ROPE_MODELS", str(REPO_ROOT / "models")))
    ap.add_argument("--output-dir", default=os.environ.get("ROPE_OUTPUT_DIR", str(REPO_ROOT / "outputs")))
    ap.add_argument("--inputs-dir", default=os.environ.get("ROPE_INPUTS_DIR", str(REPO_ROOT / "inputs")),
                    help="Server-side folder of target media, selectable in the UI (for large videos).")
    ap.add_argument("--config-dir", default=os.environ.get("ROPE_HOME"),
                    help="Where presets/ live (shared with the desktop app).")
    ap.add_argument("--backend", choices=["cuda", "tensorrt"], default=os.environ.get("ROPE_BACKEND", _default_backend()),
                    help="tensorrt (default when installed): ~1.6x faster; first start compiles engines "
                         "for a few minutes, cached afterwards. cuda: no compile step.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--share", action="store_true", help="Also create a public gradio.live link.")
    ap.add_argument("--no-auth", action="store_true", help="Disable the login (not recommended).")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.config_dir:
        from rope.qt import paths
        paths.set_config_dir(args.config_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.inputs_dir).mkdir(parents=True, exist_ok=True)

    from webapp.ui import layout, media_utils, state

    media_utils.install()
    _install_builtin_presets()
    state.init_state(args)
    state.capture_stdout()
    missing = [m for m in REQUIRED_MODELS if not (Path(args.models_dir) / m).is_file()]
    if missing and os.environ.get("ROPE_WEB_FAKE") != "1":
        state.log(f"Models missing in {args.models_dir}: {', '.join(missing)} — "
                  "run: python webapp/download_models.py")
    else:
        state.log("Ready. Add a source face and a target image or video.")

    auth = None
    if not args.no_auth:
        user = os.environ.get("ROPE_WEB_USER", "rope")
        password = os.environ.get("ROPE_WEB_PASSWORD") or secrets.token_urlsafe(9)
        if not os.environ.get("ROPE_WEB_PASSWORD"):
            sys.__stdout__.write(f"\n  Login  user: {user}   password: {password}\n"
                                 "  (set ROPE_WEB_PASSWORD to choose your own)\n\n")
            sys.__stdout__.flush()
        auth = (user, password)

    ui = layout.render()
    ui.queue(max_size=32)
    ui.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
        allowed_paths=[str(Path(args.output_dir).resolve()), str(Path(args.inputs_dir).resolve()),
                       str(media_utils.CACHE_DIR.resolve())],
        max_file_size="4gb",
        show_error=True,
        show_api=False,
    )


if __name__ == "__main__":
    main()
