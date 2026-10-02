"""Processors checkbox + one option group per processor (left column)."""

from __future__ import annotations

import gradio as gr

from webapp.ui import core, state
from webapp.ui.components import (
    bind_param, param_checkbox, param_dropdown, param_slider, refresh, section,
)

GROUPS: dict[str, gr.Group] = {}

SWAPPER_PARAMS = ["SwapperTypeTextSel", "MergeTextSel", "StrengthSlider", "LikenessSlider",
                  "EmbExtrapSlider", "HighFidelitySwitch", "HighFidelityAlphaSlider",
                  "HighFidelityModeTextSel"]
ENHANCER_PARAMS = ["RestorerTypeTextSel", "RestorerDetTypeTextSel", "RestorerSlider"]
COLOR_PARAMS = ["ColorMatchSwitch", "ColorRedSlider", "ColorGreenSlider", "ColorBlueSlider",
                "ColorGammaSlider", "ColorContrastSlider", "ColorSaturationSlider"]
ADJUSTER_PARAMS = ["KPSXSlider", "KPSYSlider", "KPSScaleSlider", "FaceScaleSlider"]
ALL_PARAMS = SWAPPER_PARAMS + ENHANCER_PARAMS + COLOR_PARAMS + ADJUSTER_PARAMS


def render() -> None:
    core.register("processors_checkbox_group", gr.CheckboxGroup(
        label="PROCESSORS",
        choices=state.PROCESSORS,
        value=state.STATE["processors"],
    ))

    procs = state.STATE["processors"]
    with gr.Group(visible="face_swapper" in procs) as GROUPS["face_swapper"]:
        with gr.Row():
            param_dropdown("SwapperTypeTextSel", "Face swapper resolution")
            param_dropdown("MergeTextSel", "Source merge")
        param_slider("StrengthSlider", "Face swapper weight")
        with gr.Row():
            param_slider("LikenessSlider", "Likeness")
            param_slider("EmbExtrapSlider", "Distinctiveness")
        param_checkbox("HighFidelitySwitch", "High fidelity (2-pass identity correction)")
        with gr.Row():
            param_slider("HighFidelityAlphaSlider", "Correction strength")
            param_dropdown("HighFidelityModeTextSel", "Correction mode")

    with gr.Group(visible="face_enhancer" in procs) as GROUPS["face_enhancer"]:
        with gr.Row():
            param_dropdown("RestorerTypeTextSel", "Face enhancer model")
            param_dropdown("RestorerDetTypeTextSel", "Alignment")
        param_slider("RestorerSlider", "Face enhancer blend")

    with gr.Group(visible="color_corrector" in procs) as GROUPS["color_corrector"]:
        param_checkbox("ColorMatchSwitch", "Match target color (LAB)")
        with gr.Row():
            param_slider("ColorRedSlider", "Red")
            param_slider("ColorGreenSlider", "Green")
            param_slider("ColorBlueSlider", "Blue")
        with gr.Row():
            param_slider("ColorGammaSlider", "Gamma")
            param_slider("ColorContrastSlider", "Contrast")
            param_slider("ColorSaturationSlider", "Saturation")

    with gr.Group(visible="face_adjuster" in procs) as GROUPS["face_adjuster"]:
        with gr.Row():
            param_slider("KPSXSlider", "Landmark X")
            param_slider("KPSYSlider", "Landmark Y")
        with gr.Row():
            param_slider("KPSScaleSlider", "Landmark scale")
            param_slider("FaceScaleSlider", "Face scale")


def update_processors(processors: list[str]):
    if "face_swapper" not in processors:
        processors = ["face_swapper"] + list(processors)  # the swapper is the pipeline
    state.set_item("processors", processors)
    return [gr.CheckboxGroup(value=processors)] + [
        gr.Group(visible=name in processors) for name in GROUPS
    ]


def listen() -> None:
    box = core.get_component("processors_checkbox_group")
    event = box.change(update_processors, inputs=box, outputs=[box, *GROUPS.values()], queue=False)
    refresh(event)
    for name in ALL_PARAMS:
        bind_param(name)
