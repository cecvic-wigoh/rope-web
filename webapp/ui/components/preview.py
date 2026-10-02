"""Live preview + preview options (right column).

Re-renders the selected frame through the GPU pipeline whenever any
option, source, target or face selection changes.
"""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from rope.engine import cosine_similarity_pct, is_image, read_frame
from webapp.ui import core, state
from webapp.ui.components import GPU, on_state, refresh


def render() -> None:
    core.register("preview_image", gr.Image(
        label="PREVIEW", interactive=False, elem_classes=["preview-image"]))
    core.register("preview_frame_slider", gr.Slider(
        label="PREVIEW FRAME", minimum=0, maximum=100, step=1, value=0, visible=False))
    with gr.Row():
        core.register("preview_mode_dropdown", gr.Dropdown(
            label="PREVIEW MODE", choices=state.PREVIEW_MODES, value=state.STATE["preview_mode"]))
        core.register("preview_resolution_dropdown", gr.Dropdown(
            label="PREVIEW RESOLUTION", choices=state.PREVIEW_RESOLUTIONS,
            value=state.STATE["preview_resolution"]))


def restrict(rgb: np.ndarray, resolution: str) -> np.ndarray:
    max_w, max_h = (int(v) for v in resolution.split("x"))
    h, w = rgb.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2), interpolation=cv2.INTER_AREA)
    return rgb


def _orientation(rgb: np.ndarray) -> str:
    h, w = rgb.shape[:2]
    return "is-landscape" if w > h else "is-portrait"


def _face_by_face(eng, original: np.ndarray, swapped: np.ndarray) -> np.ndarray:
    faces = eng.analyze(original)
    if not faces:
        return np.zeros((512, 1024, 3), np.uint8)
    face = faces[0]
    ref = state.get_item("reference_embedding")
    if ref is not None:
        face = max(faces, key=lambda f: cosine_similarity_pct(f.embedding, ref))
    return np.hstack((face.crop(original, 512), face.crop(swapped, 512)))


def update_preview():
    target = state.get_item("target_path")
    if not target:
        return gr.Image(value=None)
    frame_no = 0 if is_image(target) else int(state.get_item("preview_frame") or 0)
    original = restrict(read_frame(target, frame_no), state.get_item("preview_resolution"))
    if not state.get_item("source_paths"):
        return gr.Image(value=original, elem_classes=["preview-image", _orientation(original)])
    eng = state.get_engine()
    with eng.lock:
        if not state.apply_to_engine(eng):
            return gr.Image(value=original, elem_classes=["preview-image", _orientation(original)])
        swapped = eng.swap_frame(original, frame_no)
        mode = state.get_item("preview_mode")
        if mode == "frame-by-frame":
            result = np.hstack((original, swapped))
        elif mode == "face-by-face":
            result = _face_by_face(eng, original, swapped)
        else:
            result = swapped
    return gr.Image(value=result, elem_classes=["preview-image", _orientation(result)])


def update_frame_slider():
    target = state.get_item("target_path")
    if target and not is_image(target):
        total = int(state.get_item("video_info", {}).get("frames", 1))
        return gr.Slider(value=0, maximum=max(total - 1, 1), visible=True)
    return gr.Slider(value=0, visible=False)


def listen() -> None:
    from webapp.ui.components import face_selector

    slider = core.get_component("preview_frame_slider")
    gallery = core.get_component("reference_face_gallery")
    # Slider: refresh preview while dragging (once per settle) and the
    # face gallery on release.
    ev = on_state(slider, "preview_frame", "change", int)
    ev.then(update_preview, outputs=core.get_component("preview_image"), show_progress="hidden",
            trigger_mode="always_last", **GPU)
    on_state(slider, "preview_frame", "release", int).then(
        face_selector.update_gallery, outputs=gallery, show_progress="hidden", **GPU)
    refresh(on_state(core.get_component("preview_mode_dropdown"), "preview_mode"))
    refresh(on_state(core.get_component("preview_resolution_dropdown"), "preview_resolution"))
