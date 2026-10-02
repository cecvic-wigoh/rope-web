"""About, execution, output options and presets (left column)."""

from __future__ import annotations

import gradio as gr

from rope.qt import presets
from rope.qt.parameters import PARAMETER_BY_NAME, default_values
from webapp.ui import core, state
from webapp.ui.components import (
    GPU, bind_param, on_state, param_dropdown, param_slider, refresh, section,
)


# ----- about ------------------------------------------------------------------

def render_about() -> None:
    gr.Markdown(
        "# Rope\n"
        "GPU face swapping. Only use media of people who have given consent.",
        elem_classes=["about"],
    )


# ----- execution --------------------------------------------------------------

def render_execution() -> None:
    core.register("execution_provider_radio", gr.Radio(
        label="EXECUTION PROVIDER",
        choices=["cuda", "tensorrt"],
        value=state.STATE["backend"],
    ))
    with gr.Row():
        param_slider("ThreadsSlider", "Execution thread count")
        param_dropdown("ModelSessionsTextSel", "Model sessions")


def _switch_backend(backend: str) -> str:
    if backend == state.STATE["backend"]:
        return backend
    if state.STATE["processing"]:
        gr.Warning("Wait for the current render to finish.")
        return state.STATE["backend"]
    state.set_item("backend", backend)
    state.reset_engine()
    state.log(f"Execution provider set to {backend}; models reload on next use.")
    return backend


# ----- output options ---------------------------------------------------------

def render_output_options() -> None:
    param_slider("VideoQualSlider", "Output video quality (CRF)")
    with gr.Row():
        core.register("output_video_encoder_dropdown", gr.Dropdown(
            label="OUTPUT VIDEO ENCODER", choices=state.VIDEO_ENCODERS,
            value=state.STATE["video_encoder"]))
        core.register("output_video_preset_dropdown", gr.Dropdown(
            label="OUTPUT VIDEO PRESET", choices=state.VIDEO_PRESETS,
            value=state.STATE["video_preset"]))


# ----- presets ----------------------------------------------------------------

def render_presets() -> None:
    with gr.Row():
        core.register("preset_dropdown", gr.Dropdown(
            label="PRESET", choices=presets.list_presets(), value=None, scale=3))
        core.register("preset_load_button", gr.Button("Load", size="sm", scale=1))
    with gr.Row():
        core.register("preset_name_textbox", gr.Textbox(
            label="SAVE AS", placeholder="preset name", max_lines=1, scale=3))
        core.register("preset_save_button", gr.Button("Save", size="sm", scale=1))
    core.register("preset_reset_button", gr.Button("Reset all to defaults", size="sm"))


# Controls whose value is restored when a preset loads / defaults reset.
def _restorable() -> list[str]:
    return [n for n in state.STATE["params"] if core.get_component(n) is not None
            and PARAMETER_BY_NAME.get(n) is not None and PARAMETER_BY_NAME[n].kind != "button"]


def _ui_extras_from(params: dict) -> dict:
    procs = ["face_swapper"]
    if params.get("RestorerSwitch"):
        procs.append("face_enhancer")
    if params.get("ColorSwitch") or params.get("ColorMatchSwitch"):
        procs.append("color_corrector")
    if params.get("FaceAdjSwitch"):
        procs.append("face_adjuster")
    masks = ["box"] + [m for m, sw in state.MASK_SWITCHES.items() if params.get(sw)]
    if params.get("OrientAutoSwitch"):
        angle = "auto"
    elif params.get("OrientSwitch"):
        angle = str(int(params.get("OrientSlider", 0)))
    else:
        angle = "0"
    return {"processors": procs, "mask_types": masks, "detector_angle": angle}


def _apply_params(params: dict, message: str):
    from webapp.ui.components import processors as proc_mod

    merged = {**default_values(), **params}
    merged["ThreadsSlider"] = params.get("ThreadsSlider", state.STATE["params"]["ThreadsSlider"])
    state.STATE["params"].update(merged)
    extras = _ui_extras_from(merged)
    for key, value in extras.items():
        state.set_item(key, value)
    names = _restorable()
    updates = [gr.update(value=state.STATE["params"][n]) for n in names]
    updates += [
        gr.update(value=extras["processors"]),
        gr.update(value=extras["mask_types"]),
        gr.update(value=extras["detector_angle"]),
    ]
    updates += [gr.Group(visible=n in extras["processors"]) for n in proc_mod.GROUPS]
    state.log(message)
    gr.Info(message)
    return updates


def _preset_outputs():
    from webapp.ui.components import processors as proc_mod

    return ([core.get_component(n) for n in _restorable()]
            + [core.get_component("processors_checkbox_group"),
               core.get_component("face_mask_types_checkbox_group"),
               core.get_component("face_detector_angles_radio")]
            + list(proc_mod.GROUPS.values()))


def load_preset(name: str | None):
    if not name:
        raise gr.Error("Pick a preset first.")
    return _apply_params(presets.load_preset(name), f"Loaded preset '{name}'.")


def reset_defaults():
    return _apply_params(default_values(), "Reset all options to defaults.")


def save_preset(name: str):
    if not name or not name.strip():
        raise gr.Error("Type a preset name.")
    saved = presets.save_preset(name, state.effective_params())
    gr.Info(f"Saved preset '{saved}'.")
    state.log(f"Saved preset '{saved}' to {presets.preset_path(saved)}")
    return gr.Dropdown(choices=presets.list_presets(), value=saved), ""


# ----- events -----------------------------------------------------------------

def listen() -> None:
    radio = core.get_component("execution_provider_radio")
    radio.change(_switch_backend, inputs=radio, outputs=radio, **GPU)
    bind_param("ThreadsSlider", preview=False)
    bind_param("ModelSessionsTextSel", preview=False)
    bind_param("VideoQualSlider", preview=False)
    on_state(core.get_component("output_video_encoder_dropdown"), "video_encoder")
    on_state(core.get_component("output_video_preset_dropdown"), "video_preset")

    outputs = _preset_outputs()
    refresh(core.get_component("preset_load_button").click(
        load_preset, inputs=core.get_component("preset_dropdown"), outputs=outputs, queue=False))
    refresh(core.get_component("preset_reset_button").click(
        reset_defaults, outputs=outputs, queue=False))
    core.get_component("preset_save_button").click(
        save_preset, inputs=core.get_component("preset_name_textbox"),
        outputs=[core.get_component("preset_dropdown"), core.get_component("preset_name_textbox")],
        queue=False)
