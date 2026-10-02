"""Rope web app — browser UI for the headless engine (rope/engine.py).

Built for GPU notebooks such as Lightning AI Studios, where there's no
desktop for the Qt GUI:

    python webapp/app.py --models-dir models --port 7860

Then open the port (Lightning: "Port viewer" plugin → 7860).

Access is password-protected by default. Set ROPE_WEB_USER /
ROPE_WEB_PASSWORD, or a random password is generated and printed at
startup. ROPE_WEB_FAKE=1 swaps the engine for a CPU stub (UI testing only).
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from rope.qt import presets  # noqa: E402
from rope.qt.parameters import (  # noqa: E402
    PARAMETER_BY_NAME,
    SECTIONS,
    default_values,
)

# Parameter panel = the desktop Parameters tab + the Settings-tab knobs
# that matter for offline rendering.
WEB_SECTIONS: list[tuple[str, list[str]]] = list(SECTIONS) + [
    ("Detection", ["DetectTypeTextSel", "DetectInputSizeTextSel", "DetectScoreSlider"]),
    ("Performance", ["ThreadsSlider", "ModelSessionsTextSel"]),
    ("Output", ["VideoQualSlider"]),
]
REQUIRED_MODELS = ["det_10g.onnx", "w600k_r50.onnx", "inswapper_128.fp16.onnx"]

ARGS: argparse.Namespace
_engine = None
_engine_lock = threading.Lock()
_cancel = threading.Event()


# ----- engine -----------------------------------------------------------------

class _FakeEngine:
    """CPU stand-in so the UI can be exercised without a GPU."""

    def __init__(self):
        from rope import engine as E
        self._E = E
        self.lock = threading.RLock()
        self.params = default_values()

    def set_parameters(self, values):
        self.params = {**default_values(), **values}

    def scan_faces(self, path, samples=12):
        rgb = self._E.read_frame(path, 0)
        h, w = rgb.shape[:2]
        crop = rgb[: min(h, w), : min(h, w)]
        thumb = np.ascontiguousarray(crop[:: max(1, crop.shape[0] // 112), :: max(1, crop.shape[1] // 112)][:112, :112])
        rng = np.random.default_rng(0)
        return [self._E.Face(rng.normal(size=512).astype(np.float32), thumb) for _ in range(2)]

    def source_embedding(self, paths, merge_mode=None):
        return np.ones(512, dtype=np.float32)

    def set_assignments(self, pairs):
        self.pairs = pairs

    def swap_frame(self, rgb, frame_number=0):
        return rgb[:, ::-1].copy()

    def render_image(self, in_path, out_path):
        import cv2
        cv2.imwrite(out_path, cv2.cvtColor(self.swap_frame(self._E.read_image_rgb(in_path)), cv2.COLOR_RGB2BGR))
        return out_path

    def render_video(self, in_path, out_path, *, start_frame=0, end_frame=None, progress=None, cancel=None, threads=None):
        import shutil
        n = max(1, (end_frame or 10) - start_frame)
        for i in range(n):
            time.sleep(0.02)
            if progress:
                progress(i + 1, n)
            if cancel is not None and cancel.is_set():
                raise self._E.Cancelled()
        shutil.copy(in_path, out_path)
        return out_path


def get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            if os.environ.get("ROPE_WEB_FAKE") == "1":
                _engine = _FakeEngine()
            else:
                from rope.engine import RopeEngine
                _engine = RopeEngine(models_dir=ARGS.models_dir, backend=ARGS.backend)
        return _engine


def missing_models() -> list[str]:
    folder = Path(ARGS.models_dir)
    return [m for m in REQUIRED_MODELS if not (folder / m).is_file()]


# ----- parameter panel --------------------------------------------------------

def _info(name: str) -> str | None:
    text = PARAMETER_BY_NAME[name].info_text or ""
    # InfoText is "TITLE:\nbody" — keep the body, trimmed.
    body = text.split("\n", 1)[-1].strip().replace("\n", " ")
    return (body[:220] + "…") if len(body) > 220 else (body or None)


def build_param_controls() -> tuple[list[str], list[gr.components.Component]]:
    names, comps = [], []
    for title, section_names in WEB_SECTIONS:
        with gr.Accordion(title, open=title in ("Similarity", "Swapper", "Restorer")):
            for name in section_names:
                p = PARAMETER_BY_NAME.get(name)
                if p is None or p.kind == "button":
                    continue
                if p.kind == "slider":
                    c = gr.Slider(p.min, p.max, value=p.default, step=p.inc, label=p.label, info=_info(name))
                elif p.kind == "switch":
                    c = gr.Checkbox(value=p.default, label=p.label, info=_info(name))
                elif p.kind == "select":
                    c = gr.Radio(choices=p.modes, value=p.default, label=p.label, info=_info(name))
                else:
                    c = gr.Textbox(value=p.default, label=p.label)
                names.append(name)
                comps.append(c)
    return names, comps


def values_from(names, vals) -> dict:
    return dict(zip(names, vals))


# ----- handlers ---------------------------------------------------------------

def _file_path(f) -> str | None:
    if f is None:
        return None
    return f if isinstance(f, str) else getattr(f, "name", None) or f.get("path")


def on_target_upload(target, state):
    from rope.engine import is_image, video_info

    state = dict(state or {})
    path = _file_path(target)
    state.update(target_path=path, target_faces=[])
    if not path:
        return state, gr.update(maximum=1, value=0), gr.update(value=None), "Upload a target video or image."
    if is_image(path):
        return state, gr.update(maximum=1, value=0, interactive=False), gr.update(value=None), "Image target loaded."
    info = video_info(path)
    last = max(info["frames"] - 1, 1)
    state["video_info"] = info
    msg = f"Video: {info['width']}x{info['height']}, {info['fps']:.2f} fps, {info['frames']} frames."
    return state, gr.update(maximum=last, value=0, interactive=True), gr.update(value=None), msg


def on_scan(state, *vals, names=None):
    state = dict(state or {})
    path = state.get("target_path")
    if not path:
        raise gr.Error("Upload a target first.")
    eng = get_engine()
    with eng.lock:
        eng.set_parameters(values_from(names, vals))
        faces = eng.scan_faces(path)
    state["target_faces"] = faces
    labels = [f"Face {i + 1}" for i in range(len(faces))]
    gallery = [(f.thumbnail, lbl) for f, lbl in zip(faces, labels)]
    msg = f"Found {len(faces)} distinct face(s)." if faces else "No faces found — try a lower Detect Score."
    return state, gallery, gr.update(choices=labels, value=labels), msg


def _apply_faces(eng, state, sources, selected):
    src_paths = [_file_path(s) for s in (sources or []) if _file_path(s)]
    if not src_paths:
        raise gr.Error("Upload at least one source face image.")
    faces = state.get("target_faces") or []
    if not faces:
        raise gr.Error("Click 'Find faces' on the target first.")
    idx = [int(s.split()[-1]) - 1 for s in (selected or [])]
    if not idx:
        raise gr.Error("Select at least one target face to swap.")
    source_emb = eng.source_embedding(src_paths)
    eng.set_assignments([(faces[i].embedding, source_emb) for i in idx if 0 <= i < len(faces)])


def on_preview(state, sources, selected, frame_no, *vals, names=None):
    from rope.engine import read_frame

    state = state or {}
    path = state.get("target_path")
    if not path:
        raise gr.Error("Upload a target first.")
    eng = get_engine()
    with eng.lock:
        eng.set_parameters(values_from(names, vals))
        _apply_faces(eng, state, sources, selected)
        original = read_frame(path, int(frame_no))
        t0 = time.time()
        swapped = eng.swap_frame(original, int(frame_no))
        dt = time.time() - t0
    return original, swapped, f"Preview of frame {int(frame_no)} rendered in {dt:.2f}s."


def on_render(state, sources, selected, consent, start, end, *vals, names=None, progress=gr.Progress()):
    from rope.engine import Cancelled, is_image

    if not consent:
        raise gr.Error("Confirm you have consent from the people whose faces are used.")
    state = state or {}
    path = state.get("target_path")
    if not path:
        raise gr.Error("Upload a target first.")
    out_dir = Path(ARGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    eng = get_engine()
    _cancel.clear()
    with eng.lock:
        eng.set_parameters(values_from(names, vals))
        _apply_faces(eng, state, sources, selected)
        if is_image(path):
            out = str(out_dir / f"{stem}_swap_{int(time.time())}.png")
            eng.render_image(path, out)
            return None, out, out, f"Saved {out}"
        out = str(out_dir / f"{stem}_swap_{int(time.time())}.mp4")
        t0 = time.time()

        def report(done, total):
            progress(done / max(total, 1), desc=f"Frame {done}/{total}")

        try:
            eng.render_video(path, out, start_frame=int(start), end_frame=int(end) or None,
                             progress=report, cancel=_cancel)
        except Cancelled:
            return None, None, None, "Render cancelled."
    return out, None, out, f"Saved {out} in {time.time() - t0:.0f}s."


def on_cancel():
    _cancel.set()
    return "Cancelling after the current frame…"


def on_preset_load(name):
    if not name:
        raise gr.Error("Pick a preset.")
    loaded = presets.load_preset(name)
    vals = {**default_values(), **loaded}
    return [gr.update(value=vals[n]) for n in PARAM_NAMES] + [f"Loaded preset '{name}' ({len(loaded)} values)."]


def on_preset_save(name, *vals):
    if not name or not name.strip():
        raise gr.Error("Type a preset name.")
    saved = presets.save_preset(name, values_from(PARAM_NAMES, vals))
    return gr.update(choices=presets.list_presets(), value=saved), f"Saved preset '{saved}'."


def on_defaults():
    d = default_values()
    return [gr.update(value=d[n]) for n in PARAM_NAMES] + ["Parameters reset to defaults."]


# ----- layout -----------------------------------------------------------------

PARAM_NAMES: list[str] = []


def build_ui() -> gr.Blocks:
    global PARAM_NAMES
    with gr.Blocks(title="Rope") as demo:
        gr.Markdown("# Rope\nFace swapping on GPU. Use only with consent from everyone whose face appears.")
        missing = missing_models()
        if missing and os.environ.get("ROPE_WEB_FAKE") != "1":
            gr.Markdown(
                f"**Models missing in `{ARGS.models_dir}`:** {', '.join(missing)}. "
                "Run `python webapp/download_models.py` first."
            )
        state = gr.State({})
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    sources = gr.File(label="1. Source face image(s)", file_count="multiple",
                                      file_types=["image"])
                    target = gr.File(label="2. Target video or image",
                                     file_types=["video", "image"])
                with gr.Row():
                    scan_btn = gr.Button("3. Find faces in target")
                faces_gallery = gr.Gallery(label="Faces found in target", columns=8, height=170,
                                           allow_preview=False)
                selected = gr.CheckboxGroup(label="Faces to swap", choices=[])
                with gr.Row():
                    frame_no = gr.Slider(0, 1, value=0, step=1, label="Preview frame")
                    preview_btn = gr.Button("Preview frame", variant="secondary")
                with gr.Row():
                    before = gr.Image(label="Original", interactive=False)
                    after = gr.Image(label="Swapped", interactive=False)
                with gr.Accordion("4. Render", open=True):
                    with gr.Row():
                        start = gr.Number(value=0, precision=0, label="Start frame")
                        end = gr.Number(value=0, precision=0, label="End frame (0 = to the end)")
                    consent = gr.Checkbox(label="I have consent from the people whose faces are used")
                    with gr.Row():
                        render_btn = gr.Button("Render", variant="primary")
                        cancel_btn = gr.Button("Cancel")
                    out_video = gr.Video(label="Result", interactive=False)
                    out_image = gr.Image(label="Result image", interactive=False)
                    out_file = gr.File(label="Download")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=2):
                with gr.Group():
                    with gr.Row():
                        preset_dd = gr.Dropdown(choices=presets.list_presets(), label="Preset",
                                                allow_custom_value=False)
                        preset_load = gr.Button("Load")
                    with gr.Row():
                        preset_name = gr.Textbox(label="Save current as", placeholder="preset name")
                        preset_save = gr.Button("Save")
                    defaults_btn = gr.Button("Reset to defaults")
                PARAM_NAMES, param_comps = build_param_controls()

        def bind(fn):
            # Handlers receive param values positionally after fixed inputs.
            def wrapped(*a, **kw):
                return fn(*a, names=PARAM_NAMES, **kw)
            wrapped.__name__ = fn.__name__
            return wrapped

        target.change(on_target_upload, [target, state], [state, frame_no, faces_gallery, status])
        scan_btn.click(bind(on_scan), [state, *param_comps], [state, faces_gallery, selected, status])
        preview_btn.click(bind(on_preview), [state, sources, selected, frame_no, *param_comps],
                          [before, after, status])
        render_btn.click(_render_entry, [state, sources, selected, consent, start, end, *param_comps],
                         [out_video, out_image, out_file, status])
        cancel_btn.click(on_cancel, None, status, queue=False)
        preset_load.click(on_preset_load, preset_dd, [*param_comps, status])
        preset_save.click(on_preset_save, [preset_name, *param_comps], [preset_dd, status])
        defaults_btn.click(on_defaults, None, [*param_comps, status])
    return demo


def _render_entry(state, sources, selected, consent, start, end, *vals, progress=gr.Progress()):
    return on_render(state, sources, selected, consent, start, end, *vals,
                     names=PARAM_NAMES, progress=progress)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rope web app")
    ap.add_argument("--models-dir", default=os.environ.get("ROPE_MODELS", str(REPO_ROOT / "models")))
    ap.add_argument("--output-dir", default=os.environ.get("ROPE_OUTPUT_DIR", str(REPO_ROOT / "outputs")))
    ap.add_argument("--config-dir", default=os.environ.get("ROPE_HOME"),
                    help="Where presets/ live (shared with the desktop app).")
    ap.add_argument("--backend", choices=["onnx", "trt"], default=os.environ.get("ROPE_BACKEND", "onnx"),
                    help="onnx = CUDA EP (fast startup); trt = TensorRT (faster frames, slow first build).")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--share", action="store_true", help="Also create a public gradio.live link.")
    ap.add_argument("--no-auth", action="store_true", help="Disable the login (not recommended).")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    global ARGS
    ARGS = parse_args(argv)
    if ARGS.config_dir:
        from rope.qt import paths
        paths.set_config_dir(ARGS.config_dir)

    auth = None
    if not ARGS.no_auth:
        user = os.environ.get("ROPE_WEB_USER", "rope")
        password = os.environ.get("ROPE_WEB_PASSWORD")
        if not password:
            password = secrets.token_urlsafe(9)
            print(f"\n  Login  user: {user}   password: {password}\n"
                  "  (set ROPE_WEB_PASSWORD to choose your own)\n", flush=True)
        auth = (user, password)

    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=8)
    demo.launch(
        server_name=ARGS.host,
        server_port=ARGS.port,
        share=ARGS.share,
        auth=auth,
        allowed_paths=[ARGS.output_dir],
        max_file_size="4gb",
        show_error=True,
    )


if __name__ == "__main__":
    main()
