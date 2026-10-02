"""Trim frame, face masker and face detector (right column)."""

from __future__ import annotations

import gradio as gr
from gradio_rangeslider import RangeSlider

from rope.engine import is_image
from webapp.ui import core, state
from webapp.ui.components import bind_param, on_state, param_dropdown, param_slider, refresh

MASK_GROUPS: dict[str, gr.Group] = {}
MASK_PARAMS = ["BorderTopSlider", "BorderSidesSlider", "BorderBottomSlider", "BorderBlurSlider",
               "OccluderSlider", "FaceParserSlider", "MouthParserSlider",
               "DFLXSegSizeSlider", "DFLXSegBlurSlider", "DiffSlider", "BlendSlider"]


# ----- trim frame -------------------------------------------------------------

def render_trim() -> None:
    core.register("trim_frame_slider", RangeSlider(
        label="TRIM FRAME", minimum=0, maximum=100, step=1, value=(0, 100), visible=False))


def update_trim():
    target = state.get_item("target_path")
    if target and not is_image(target):
        total = int(state.get_item("video_info", {}).get("frames", 1))
        return RangeSlider(value=(0, total), maximum=total, visible=True)
    return RangeSlider(visible=False)


def _set_trim(value) -> None:
    start, end = (int(v) for v in value)
    total = int(state.get_item("video_info", {}).get("frames", end))
    state.set_item("trim_start", start)
    state.set_item("trim_end", end if end < total else None)


# ----- face masker ------------------------------------------------------------

def render_masker() -> None:
    types = state.STATE["mask_types"]
    core.register("face_mask_types_checkbox_group", gr.CheckboxGroup(
        label="FACE MASK TYPES", choices=state.MASK_TYPES, value=types))
    with gr.Group(visible="box" in types) as MASK_GROUPS["box"]:
        with gr.Row():
            param_slider("BorderTopSlider", "Mask padding top")
            param_slider("BorderBottomSlider", "Mask padding bottom")
        with gr.Row():
            param_slider("BorderSidesSlider", "Mask padding sides")
            param_slider("BorderBlurSlider", "Mask blur")
    with gr.Group(visible="occlusion" in types) as MASK_GROUPS["occlusion"]:
        param_slider("OccluderSlider", "Occlusion size")
    with gr.Group(visible="region" in types) as MASK_GROUPS["region"]:
        with gr.Row():
            param_slider("FaceParserSlider", "Background region")
            param_slider("MouthParserSlider", "Mouth region")
    with gr.Group(visible="xseg" in types) as MASK_GROUPS["xseg"]:
        with gr.Row():
            param_slider("DFLXSegSizeSlider", "XSeg size")
            param_slider("DFLXSegBlurSlider", "XSeg blur")
    with gr.Group(visible="diff" in types) as MASK_GROUPS["diff"]:
        param_slider("DiffSlider", "Difference amount")
    param_slider("BlendSlider", "Overall mask blend")


def update_mask_types(types: list[str]):
    types = list(types or [])
    state.set_item("mask_types", types)
    return [gr.Group(visible=t in types) for t in MASK_GROUPS]


# ----- face detector ----------------------------------------------------------

def render_detector() -> None:
    with gr.Row():
        param_dropdown("DetectTypeTextSel", "Face detector model",
                       choices=["Retinaface", "SCRDF"])
        sizes = ["640"] if state.STATE["backend"] == "cuda" else ["320", "416", "480", "640"]
        param_dropdown("DetectInputSizeTextSel", "Face detector size", choices=sizes)
    core.register("face_detector_angles_radio", gr.Radio(
        label="FACE DETECTOR ANGLE", choices=state.DETECTOR_ANGLES, value=state.STATE["detector_angle"]))
    param_slider("DetectScoreSlider", "Face detector score")


# ----- events -----------------------------------------------------------------

def listen() -> None:
    trim = core.get_component("trim_frame_slider")
    trim.release(_set_trim, inputs=trim, queue=False)

    types = core.get_component("face_mask_types_checkbox_group")
    refresh(types.change(update_mask_types, inputs=types, outputs=list(MASK_GROUPS.values()), queue=False))
    for name in MASK_PARAMS:
        bind_param(name)

    for name in ("DetectTypeTextSel", "DetectInputSizeTextSel", "DetectScoreSlider"):
        bind_param(name, gallery=True)
    refresh(on_state(core.get_component("face_detector_angles_radio"), "detector_angle"), gallery=True)
