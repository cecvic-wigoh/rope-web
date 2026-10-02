"""Default layout: options | source, target, output, run | live preview + face tools.

Mirrors the column structure of FaceFusion's default web layout.
"""

from __future__ import annotations

import gradio as gr

from webapp.ui import core
from webapp.ui.components import face_selector, face_tools, media, preview, processors, runner, settings


def render() -> gr.Blocks:
    with gr.Blocks(theme=core.theme(), css=core.css(), title="Rope", fill_width=True) as ui:
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Blocks():
                    settings.render_about()
                with gr.Blocks():
                    processors.render()
                with gr.Blocks():
                    settings.render_execution()
                with gr.Blocks():
                    settings.render_output_options()
                with gr.Blocks():
                    settings.render_presets()
            with gr.Column(scale=4):
                with gr.Blocks():
                    media.render_source()
                with gr.Blocks():
                    media.render_target()
                with gr.Blocks():
                    media.render_output()
                with gr.Blocks():
                    runner.render_terminal()
                with gr.Blocks():
                    runner.render_runner()
            with gr.Column(scale=7):
                with gr.Blocks():
                    preview.render()
                with gr.Blocks():
                    face_tools.render_trim()
                with gr.Blocks():
                    face_selector.render()
                with gr.Blocks():
                    face_tools.render_masker()
                with gr.Blocks():
                    face_tools.render_detector()
        listen()
    return ui


def listen() -> None:
    processors.listen()
    settings.listen()
    media.listen()
    runner.listen()
    preview.listen()
    face_selector.listen()
    face_tools.listen()
