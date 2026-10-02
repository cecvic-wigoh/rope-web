"""Face selector: 'reference' (click a face to swap only that person) or
'many' (swap every face). Right column."""

from __future__ import annotations

import gradio as gr

from rope.engine import is_image, read_frame
from webapp.ui import core, state
from webapp.ui.components import bind_param, param_slider, refresh

_GALLERY_FACES: list = []


def render() -> None:
    core.register("face_selector_mode_dropdown", gr.Dropdown(
        label="FACE SELECTOR MODE", choices=state.FACE_SELECTOR_MODES,
        value=state.STATE["face_selector_mode"]))
    core.register("reference_face_gallery", gr.Gallery(
        label="REFERENCE FACE", object_fit="cover", allow_preview=False, columns=7, height="auto",
        elem_classes=["face-gallery"], visible=state.STATE["face_selector_mode"] == "reference"))
    param_slider("ThresholdSlider", "Reference face similarity",
                 visible=state.STATE["face_selector_mode"] == "reference")


def update_gallery():
    _GALLERY_FACES.clear()
    target = state.get_item("target_path")
    if not target or state.get_item("face_selector_mode") != "reference":
        return gr.Gallery(value=None)
    frame_no = 0 if is_image(target) else int(state.get_item("preview_frame") or 0)
    rgb = read_frame(target, frame_no)
    eng = state.get_engine()
    with eng.lock:
        eng.set_parameters(state.effective_params())
        faces = eng.analyze(rgb)
    _GALLERY_FACES.extend(faces)
    if faces and state.get_item("reference_embedding") is None:
        state.set_item("reference_embedding", faces[0].embedding)
        state.set_item("reference_face_position", 0)
    return gr.Gallery(value=[f.crop(rgb, 128) for f in faces] or None)


def select_reference(event: gr.SelectData):
    if 0 <= event.index < len(_GALLERY_FACES):
        state.set_item("reference_embedding", _GALLERY_FACES[event.index].embedding)
        state.set_item("reference_face_position", event.index)
        state.log(f"Reference face #{event.index + 1} selected")


def update_mode(mode: str):
    state.set_item("face_selector_mode", mode)
    show = mode == "reference"
    return gr.Gallery(visible=show), gr.Slider(visible=show)


def listen() -> None:
    mode = core.get_component("face_selector_mode_dropdown")
    gallery = core.get_component("reference_face_gallery")
    refresh(mode.change(update_mode, inputs=mode, outputs=[gallery, core.get_component("ThresholdSlider")],
                        queue=False), gallery=True)
    refresh(gallery.select(select_reference, queue=False))
    bind_param("ThresholdSlider")
