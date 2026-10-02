"""Start / Stop / Clear and the terminal (middle column)."""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from rope.engine import Cancelled, is_image
from webapp.ui import core, state


def render_terminal() -> None:
    core.register("terminal_textbox", gr.Textbox(
        label="TERMINAL", value=state.read_logs, lines=8, max_lines=8,
        interactive=False, autoscroll=True, show_copy_button=True,
        elem_classes=["terminal"]))
    core.register("terminal_timer", gr.Timer(0.5))


def render_runner() -> None:
    core.register("consent_checkbox", gr.Checkbox(
        label="I have consent from the people whose faces are used", value=False))
    with gr.Row():
        core.register("start_button", gr.Button("Start", variant="primary", size="sm"))
        core.register("stop_button", gr.Button("Stop", variant="primary", size="sm", visible=False))
        core.register("clear_button", gr.Button("Clear", size="sm"))


def start(consent: bool):
    if not consent:
        raise gr.Error("Confirm you have consent from the people whose faces are used.")
    if not state.get_item("source_paths"):
        raise gr.Error("Add a source face image.")
    if not state.get_item("target_path"):
        raise gr.Error("Add a target image or video.")
    # Reset here, not in run(): a Stop clicked while run() is still queued
    # must not be wiped out when it starts.
    state.CANCEL.clear()
    return gr.Button(visible=False), gr.Button(visible=True)


def run(consent: bool):
    idle = (gr.Button(visible=True), gr.Button(visible=False))
    if not consent or not state.get_item("source_paths") or not state.get_item("target_path"):
        return *idle, gr.update(), gr.update()
    target = state.get_item("target_path")
    out_dir = Path(state.get_item("output_path") or state.ARGS.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(target).stem
    if state.CANCEL.is_set():
        state.log("Processing stopped.")
        return *idle, gr.update(), gr.update()
    state.set_item("processing", True)
    eng = state.get_engine()
    t0 = time.time()
    try:
        with eng.lock:
            if not state.apply_to_engine(eng):
                raise gr.Error("No face found in the source image(s).")
            if is_image(target):
                out = str(out_dir / f"{stem}-rope-{int(time.time())}.png")
                state.log("Processing image…")
                eng.render_image(target, out)
                state.log(f"Saved {out} in {time.time() - t0:.1f}s")
                return *idle, gr.Image(value=out, visible=True), gr.Video(value=None, visible=False)

            out = str(out_dir / f"{stem}-rope-{int(time.time())}.mp4")
            start_f = int(state.get_item("trim_start") or 0)
            end_f = state.get_item("trim_end")

            def progress(done: int, total: int) -> None:
                pct = int(done * 100 / max(total, 1))
                fps = done / max(time.time() - t0, 1e-6)
                state.log(f"Processing: {pct}% ({done}/{total}) {fps:.1f} fps", replace_prefix="Processing:")

            eng.render_video(
                target, out, start_frame=start_f, end_frame=end_f, progress=progress,
                cancel=state.CANCEL, video_encoder=state.get_item("video_encoder"),
                video_preset=state.get_item("video_preset"),
            )
            state.log(f"Saved {out} in {time.time() - t0:.1f}s")
            return *idle, gr.Image(value=None, visible=False), gr.Video(value=out, visible=True)
    except Cancelled:
        state.log("Processing stopped.")
        return *idle, gr.update(), gr.update()
    finally:
        state.set_item("processing", False)


def stop():
    state.CANCEL.set()
    state.log("Stopping…")
    return gr.Button(visible=True), gr.Button(visible=False)


def clear():
    return gr.Image(value=None, visible=False), gr.Video(value=None, visible=True)


def listen() -> None:
    consent = core.get_component("consent_checkbox")
    start_b, stop_b = core.get_component("start_button"), core.get_component("stop_button")
    out_img, out_vid = core.get_component("output_image"), core.get_component("output_video")
    start_b.click(start, inputs=consent, outputs=[start_b, stop_b], queue=False).success(
        run, inputs=consent, outputs=[start_b, stop_b, out_img, out_vid],
        concurrency_limit=1, concurrency_id="gpu")
    stop_b.click(stop, outputs=[start_b, stop_b], queue=False)
    core.get_component("clear_button").click(clear, outputs=[out_img, out_vid], queue=False)

    term = core.get_component("terminal_textbox")
    core.get_component("terminal_timer").tick(state.read_logs, outputs=term, queue=False,
                                              show_progress="hidden")
