"""Live preview + preview options + live swapped playback (right column).

The preview re-renders the selected frame through the GPU pipeline
whenever any option, source, target or face selection changes. "Play
swapped" streams the face-swapped video from the current frame in ~1 s
H.264 chunks (with the original audio) while it is being processed.
"""

from __future__ import annotations

import queue
import tempfile
import threading
import time

import cv2
import gradio as gr
import numpy as np

from rope.engine import cosine_similarity_pct, is_image, read_frame
from webapp.ui import core, media_utils, state
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
    with gr.Row(visible=False) as row:
        core.register("live_play_button", gr.Button("▶ Play swapped", variant="primary", size="sm"))
        core.register("live_stop_button", gr.Button("■ Stop playback", size="sm"))
    core.register("live_controls_row", row)
    core.register("live_video", gr.Video(
        label="LIVE PLAYBACK", streaming=True, autoplay=True, interactive=False, visible=False))


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
    is_video = bool(target) and not is_image(target)
    total = int(state.get_item("video_info", {}).get("frames", 1)) if is_video else 1
    return (gr.Slider(value=0, maximum=max(total - 1, 1), visible=is_video),
            gr.Row(visible=is_video), gr.Video(value=None, visible=is_video))


# ----- live swapped playback ----------------------------------------------------

LIVE_CANCEL = threading.Event()


def _put(out: queue.Queue, item) -> bool:
    """Queue put that gives up once playback is cancelled (e.g. the tab
    closed), so the producer never blocks while holding the GPU lock."""
    while True:
        try:
            out.put(item, timeout=1.0)
            return True
        except queue.Full:
            if LIVE_CANCEL.is_set():
                return False


PREBUFFER_SEGMENTS = 2


def _live_producer(out: queue.Queue, target: str, start: int, end: int | None,
                   max_size: tuple[int, int]) -> None:
    """Swap frames on the GPU (this thread owns the GPU lock for the whole
    run) and feed them to one continuous segmenting encoder; completed
    ~1 s segments go to the streaming generator.

    The first ~0.5 s measures swap throughput; if the GPU can't keep up
    with the video's frame rate, every Nth frame is used for the rest of the
    run so playback stays real-time (audio is always complete)."""
    enc = None
    try:
        eng = state.get_engine()
        info = state.get_item("video_info") or {}
        fps = float(info.get("fps") or 25.0)
        try:
            src = media_utils.web_proxy(target)  # light 720p copy decodes fastest
        except Exception:
            src = target
        workdir = tempfile.mkdtemp(prefix="live-", dir=media_utils.CACHE_DIR)
        with eng.lock:
            if not state.apply_to_engine(eng):
                _put(out, RuntimeError("Add a source face image first."))
                return
            stride = [1]
            next_idx = start  # next source frame due in the output
            probe: list[tuple[int, np.ndarray]] = []
            probe_len = max(6, int(fps * 0.5))
            t0 = time.time()
            frames = eng.iter_swapped_frames(src, start, end, max_size=max_size,
                                             cancel=LIVE_CANCEL, stride=lambda: stride[0])
            for idx, frame in frames:
                if enc is None:
                    probe.append((idx, frame))
                    if len(probe) < probe_len:
                        continue
                    rate = len(probe) / max(time.time() - t0, 1e-6)
                    stride[0] = int(min(4, max(1, np.ceil(fps / (rate * 0.85)))))
                    h, w = frame.shape[:2]
                    enc = media_utils.SegmentEncoder(w, h, fps / stride[0], workdir, source=target,
                                                     start_seconds=start / fps)
                    state.log(f"Live: GPU swaps {rate:.1f} fps — playing "
                              f"{fps / stride[0]:.0f} of {fps:.0f} fps", replace_prefix="Live:")
                    pending = probe
                else:
                    pending = [(idx, frame)]
                for i, f in pending:
                    # Keep one frame per `stride` source frames; drops frames
                    # that were already in flight when the stride was chosen.
                    if i >= next_idx:
                        enc.write(f)
                        next_idx = i + stride[0]
                for path, _dur in enc.poll():
                    if not _put(out, path):
                        return
            if enc is None and probe and not LIVE_CANCEL.is_set():  # clip shorter than the probe
                h, w = probe[0][1].shape[:2]
                enc = media_utils.SegmentEncoder(w, h, fps, workdir, source=target,
                                                 start_seconds=start / fps)
                for _, f in probe:
                    enc.write(f)
        if enc is not None:
            if LIVE_CANCEL.is_set():
                enc.kill()
            else:
                for path, _dur in enc.close():
                    _put(out, path)
    except Exception as exc:  # surfaced by the generator
        if enc is not None:
            enc.kill()
        _put(out, exc)
    finally:
        _put(out, None)


def live_play():
    target = state.get_item("target_path")
    if not target or is_image(target):
        raise gr.Error("Live playback needs a target video.")
    if not state.get_item("source_paths"):
        raise gr.Error("Add a source face image first.")
    if state.get_item("processing"):
        raise gr.Error("Wait for the current render to finish.")
    LIVE_CANCEL.clear()
    start = int(state.get_item("preview_frame") or 0)
    end = state.get_item("trim_end")
    max_w, max_h = (int(v) for v in state.get_item("preview_resolution").split("x"))
    chunks: queue.Queue = queue.Queue(maxsize=4)
    state.log(f"Live: starting at frame {start}…", replace_prefix="Live:")
    threading.Thread(target=_live_producer, args=(chunks, target, start, end, (max_w, max_h)),
                     daemon=True).start()
    finished = False
    held: list[str] | None = []
    try:
        while True:
            item = chunks.get()
            if item is None:
                finished = True
                break
            if isinstance(item, Exception):
                raise gr.Error(str(item))
            # Hold only the first segments so playback starts with a cushion.
            if held is not None:
                held.append(item)
                if len(held) >= PREBUFFER_SEGMENTS:
                    yield from held
                    held = None
                continue
            yield item
        if held:
            yield from held
    finally:
        if not finished:
            LIVE_CANCEL.set()  # stopped, errored or tab closed: release the GPU
    state.log("Live: playback finished" if not LIVE_CANCEL.is_set() else "Live: playback stopped",
              replace_prefix="Live:")


def live_stop() -> None:
    LIVE_CANCEL.set()


def listen() -> None:
    from webapp.ui.components import face_selector

    live_video = core.get_component("live_video")
    play = core.get_component("live_play_button").click(
        live_play, outputs=live_video, show_progress="minimal", **GPU)
    core.get_component("live_stop_button").click(live_stop, queue=False, cancels=[play])

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
