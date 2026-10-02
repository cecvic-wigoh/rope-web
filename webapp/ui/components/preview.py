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
CHUNK_SECONDS = 1.0


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


def _live_producer(out: queue.Queue, target: str, start: int, end: int | None,
                   max_size: tuple[int, int]) -> None:
    """Swap + encode on dedicated threads (this one owns the GPU lock for the
    whole run) and hand finished .ts chunk paths to the streaming generator.

    Keeps playback real-time: if the GPU swaps slower than the video's frame
    rate, only every Nth frame is processed (N re-estimated each chunk) and
    each chunk is timed to the source span it covers, so audio stays in sync.
    """
    encode_q: queue.Queue = queue.Queue(maxsize=2)

    def encoder() -> None:
        while True:
            job = encode_q.get()
            if job is None:
                _put(out, None)
                return
            frames, rate, path, start_s, offset = job
            try:
                media_utils.encode_ts_chunk(frames, rate, path, source=target,
                                            start_seconds=start_s, offset_seconds=offset)
            except Exception as exc:
                _put(out, exc)
                continue
            _put(out, path)  # keeps draining after a cancel so the producer never blocks

    enc = threading.Thread(target=encoder, daemon=True)
    enc.start()
    try:
        eng = state.get_engine()
        info = state.get_item("video_info") or {}
        fps = float(info.get("fps") or 25.0)
        # Decode from the light web copy when available (same frame count/timing).
        try:
            src = media_utils.web_proxy(target)
        except Exception:
            src = target
        stride = [2 if fps > 20 else 1]
        workdir = tempfile.mkdtemp(prefix="live-", dir=media_utils.CACHE_DIR)
        with eng.lock:
            if not state.apply_to_engine(eng):
                _put(out, RuntimeError("Add a source face image first."))
                return
            buf: list[np.ndarray] = []
            first_idx = last_idx = start
            offset, n = 0.0, 0
            chunk_t0 = time.time()

            def flush(final: bool = False) -> None:
                nonlocal buf, offset, n, chunk_t0
                span = (last_idx - first_idx + stride[0]) / fps   # seconds of source covered
                rate = len(buf) / span
                encode_q.put((buf, rate, f"{workdir}/chunk{n:05d}.ts", first_idx / fps, offset))
                swap_fps = len(buf) / max(time.time() - chunk_t0, 1e-6)
                offset += span
                n += 1
                if not final:
                    # Next step N so fps/N stays ~15% under what the GPU sustains.
                    stride[0] = int(min(6, max(1, np.ceil(fps / max(swap_fps * 0.85, 1e-6)))))
                    state.log(f"Live: frame {last_idx} — GPU {swap_fps:.1f} fps, playing "
                              f"{fps / stride[0]:.0f} of {fps:.0f} fps", replace_prefix="Live:")
                buf = []
                chunk_t0 = time.time()

            for idx, frame in eng.iter_swapped_frames(src, start, end, max_size=max_size,
                                                      cancel=LIVE_CANCEL, stride=lambda: stride[0]):
                if not buf:
                    first_idx = idx
                last_idx = idx
                buf.append(frame)
                if (last_idx - first_idx + stride[0]) / fps >= CHUNK_SECONDS:
                    flush()
            if buf and not LIVE_CANCEL.is_set():
                flush(final=True)
    except Exception as exc:  # surfaced by the generator
        _put(out, exc)
    finally:
        encode_q.put(None)


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
    try:
        while True:
            item = chunks.get()
            if item is None:
                finished = True
                break
            if isinstance(item, Exception):
                raise gr.Error(str(item))
            yield item
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
