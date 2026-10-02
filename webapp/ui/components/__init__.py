"""UI components. Each module has render() (build widgets, register them)
and listen() (wire events once every component exists)."""

from __future__ import annotations

from typing import Any, Callable

import gradio as gr

from rope.qt.parameters import PARAMETER_BY_NAME
from webapp.ui import core, state

GPU = dict(concurrency_limit=1, concurrency_id="gpu")


def info(name: str, limit: int = 160) -> str | None:
    """First sentence-ish of a Rope parameter's InfoText."""
    text = PARAMETER_BY_NAME[name].info_text or ""
    body = text.split("\n", 1)[-1].strip().replace("\n", " ")
    if not body:
        return None
    return body if len(body) <= limit else body[:limit].rsplit(" ", 1)[0] + "…"


def param_slider(name: str, label: str | None = None, **kw) -> gr.Slider:
    p = PARAMETER_BY_NAME[name]
    comp = gr.Slider(
        minimum=p.min, maximum=p.max, step=p.inc,
        value=state.STATE["params"].get(name, p.default),
        label=(label or p.label).upper(), **kw,
    )
    return core.register(name, comp)


def param_dropdown(name: str, label: str | None = None, choices: list[str] | None = None, **kw) -> gr.Dropdown:
    p = PARAMETER_BY_NAME[name]
    comp = gr.Dropdown(
        choices=choices or p.modes, value=state.STATE["params"].get(name, p.default),
        label=(label or p.label).upper(), **kw,
    )
    return core.register(name, comp)


def param_checkbox(name: str, label: str | None = None, **kw) -> gr.Checkbox:
    p = PARAMETER_BY_NAME[name]
    comp = gr.Checkbox(
        value=bool(state.STATE["params"].get(name, p.default)),
        label=(label or p.label).upper(), **kw,
    )
    return core.register(name, comp)


def bind_param(name: str, *, preview: bool = True, gallery: bool = False) -> None:
    """Write a param control's value to state, then refresh the preview."""
    comp = core.get_component(name)
    if comp is None:
        return
    trigger = comp.release if isinstance(comp, gr.Slider) else comp.change

    def setter(value: Any) -> None:
        state.set_param(name, value)

    event = trigger(setter, inputs=comp, queue=False)
    if preview or gallery:
        refresh(event, gallery=gallery)


def refresh(event, *, gallery: bool = False) -> None:
    """Chain a preview (and optionally face gallery) refresh after `event`."""
    from webapp.ui.components import face_selector, preview

    if gallery and core.get_component("reference_face_gallery") is not None:
        event = event.then(face_selector.update_gallery, outputs=core.get_component("reference_face_gallery"),
                           show_progress="hidden", **GPU)
    event.then(preview.update_preview, outputs=core.get_component("preview_image"),
               show_progress="hidden", **GPU)


def section(title: str) -> None:
    """Kept for API compatibility; FaceFusion-style layout has no headers."""


def on_state(comp, key: str, trigger: str = "change", cast: Callable[[Any], Any] = lambda v: v):
    """Bind a non-param control to a UI state key; returns the event."""
    def setter(value: Any) -> None:
        state.set_item(key, cast(value))
    return getattr(comp, trigger)(setter, inputs=comp, queue=False)
