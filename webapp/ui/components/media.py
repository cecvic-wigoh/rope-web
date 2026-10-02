"""Source, target and output media (middle column)."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from rope.engine import is_image
from webapp.ui import core, state
from webapp.ui.components import GPU, refresh


def _paths(files) -> list[str]:
    if not files:
        return []
    files = files if isinstance(files, list) else [files]
    return [f if isinstance(f, str) else getattr(f, "name", None) or f.get("path") for f in files]


# ----- source -----------------------------------------------------------------

def render_source() -> None:
    core.register("source_file", gr.File(
        label="SOURCE", file_count="multiple", file_types=["image"]))
    core.register("source_image", gr.Image(
        show_label=False, visible=False, interactive=False, height=220))


def update_source(files):
    paths = [p for p in _paths(files) if p and is_image(p)]
    state.set_item("source_paths", paths)
    if paths:
        state.log(f"Source: {len(paths)} image(s)")
    return gr.Image(value=paths[0] if paths else None, visible=bool(paths))


# ----- target -----------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
BROWSER_PLAYABLE = {".mp4", ".webm"}


def list_inputs() -> list[str]:
    """Image/video files in the server-side inputs folder (for big videos:
    copy them onto the Studio instead of uploading through the browser)."""
    folder = Path(state.ARGS.inputs_dir)
    folder.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
             if p.is_file() and (is_image(str(p)) or p.suffix.lower() in VIDEO_EXTS)]
    return [p.name for p in files]


def render_target() -> None:
    core.register("target_file", gr.File(
        label="TARGET", file_types=["image", "video"]))
    with gr.Row():
        core.register("target_server_dropdown", gr.Dropdown(
            label="OR PICK FROM INPUTS FOLDER", choices=list_inputs(), value=None, scale=4))
        core.register("target_server_refresh", gr.Button("↻", size="sm", scale=0, min_width=48))
    core.register("target_image", gr.Image(show_label=False, visible=False, interactive=False))
    core.register("target_video", gr.Video(show_label=False, visible=False, interactive=False))


def update_target(file):
    paths = _paths(file)
    return set_target(paths[0] if paths else None)


def update_target_from_inputs(name: str | None):
    if not name:
        return gr.update(), gr.update()
    return set_target(str(Path(state.ARGS.inputs_dir) / name))


def set_target(path: str | None):
    state.set_item("target_path", path)
    state.set_item("reference_embedding", None)
    state.set_item("reference_face_position", 0)
    state.set_item("preview_frame", 0)
    state.set_item("trim_start", 0)
    state.set_item("trim_end", None)
    if path and is_image(path):
        state.log(f"Target image: {path.rsplit('/', 1)[-1]}")
        return gr.Image(value=path, visible=True), gr.Video(value=None, visible=False)
    if path:
        from rope.engine import video_info
        info = video_info(path)
        state.set_item("video_info", info)
        state.log(f"Target video: {path.rsplit('/', 1)[-1]} — {info['width']}x{info['height']}, "
                  f"{info['fps']:.2f} fps, {info['frames']} frames")
        if Path(path).suffix.lower() not in BROWSER_PLAYABLE:
            # Gradio would re-encode the whole file for the player; show a
            # still instead — processing reads the original either way.
            from rope.engine import read_frame
            return gr.Image(value=read_frame(path, 0), visible=True), gr.Video(value=None, visible=False)
        return gr.Image(value=None, visible=False), gr.Video(value=path, visible=True)
    return gr.Image(value=None, visible=False), gr.Video(value=None, visible=False)


# ----- output -----------------------------------------------------------------

def render_output() -> None:
    core.register("output_path_textbox", gr.Textbox(
        label="OUTPUT PATH", value=state.STATE["output_path"], max_lines=1))
    core.register("output_image", gr.Image(label="OUTPUT", visible=False, interactive=False))
    core.register("output_video", gr.Video(label="OUTPUT", interactive=False))


# ----- events -----------------------------------------------------------------

def listen() -> None:
    from webapp.ui.components import face_selector, face_tools, preview

    src = core.get_component("source_file")
    refresh(src.change(update_source, inputs=src, outputs=core.get_component("source_image"), queue=False))

    target_outputs = [core.get_component("target_image"), core.get_component("target_video")]

    def after_target(event) -> None:
        event = event.then(preview.update_frame_slider, outputs=core.get_component("preview_frame_slider"),
                           queue=False)
        event = event.then(face_tools.update_trim, outputs=core.get_component("trim_frame_slider"), queue=False)
        event = event.then(face_selector.update_gallery, outputs=core.get_component("reference_face_gallery"),
                           show_progress="hidden", **GPU)
        event.then(preview.update_preview, outputs=core.get_component("preview_image"),
                   show_progress="hidden", **GPU)

    tgt = core.get_component("target_file")
    after_target(tgt.change(update_target, inputs=tgt, outputs=target_outputs, queue=False))
    picker = core.get_component("target_server_dropdown")
    after_target(picker.change(update_target_from_inputs, inputs=picker, outputs=target_outputs, queue=False))
    core.get_component("target_server_refresh").click(
        lambda: gr.Dropdown(choices=list_inputs()), outputs=picker, queue=False)

    out = core.get_component("output_path_textbox")
    out.change(lambda v: state.set_item("output_path", v), inputs=out, queue=False)
