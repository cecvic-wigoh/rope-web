"""Global UI state, engine access and the terminal log buffer.

The web UI is single-user per GPU (like FaceFusion's), so state lives in
one process-wide dict. Every control writes here; the preview, gallery and
runner read from here.
"""

from __future__ import annotations

import io
import os
import sys
import threading
from typing import Any

import numpy as np

from rope.qt.parameters import default_values

# Processors: FaceFusion-style toggles over Rope's pipeline stages.
PROCESSORS = ["face_swapper", "face_enhancer", "color_corrector", "face_adjuster"]
PROCESSOR_SWITCHES = {
    "face_enhancer": "RestorerSwitch",
    "face_adjuster": "FaceAdjSwitch",
}
# Mask types -> Rope switches ("box" is the always-on border mask).
MASK_TYPES = ["box", "occlusion", "region", "xseg", "diff"]
MASK_SWITCHES = {
    "occlusion": "OccluderSwitch",
    "region": "FaceParserSwitch",
    "xseg": "DFLXSegSwitch",
    "diff": "DiffSwitch",
}
FACE_SELECTOR_MODES = ["reference", "many"]
PREVIEW_MODES = ["default", "frame-by-frame", "face-by-face"]
PREVIEW_RESOLUTIONS = ["512x512", "768x768", "1024x1024", "1280x1280", "1920x1920"]
DETECTOR_ANGLES = ["0", "90", "180", "270", "auto"]
VIDEO_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]
VIDEO_ENCODERS = ["libx264", "libx265"]

ARGS: Any = None
STATE: dict[str, Any] = {}
_engine = None
_engine_lock = threading.Lock()
CANCEL = threading.Event()


def init_state(args) -> None:
    global ARGS
    ARGS = args
    STATE.clear()
    STATE.update(
        source_paths=[],
        target_path=None,
        params=default_values(),
        processors=["face_swapper"],
        mask_types=["box"],
        face_selector_mode="reference",
        reference_frame_number=0,
        reference_face_position=0,
        reference_embedding=None,
        preview_mode=PREVIEW_MODES[0],
        preview_resolution=PREVIEW_RESOLUTIONS[2],
        preview_frame=0,
        trim_start=0,
        trim_end=None,
        output_path=args.output_dir,
        backend=args.backend,
        video_preset="medium",
        video_encoder="libx264",
        detector_angle="0",
        processing=False,
    )
    STATE["params"]["ThreadsSlider"] = 2


def get_item(key: str, default: Any = None) -> Any:
    return STATE.get(key, default)


def set_item(key: str, value: Any) -> None:
    STATE[key] = value


def set_param(name: str, value: Any) -> None:
    STATE["params"][name] = value


def effective_params() -> dict[str, Any]:
    """Rope parameters with processor / mask toggles folded in."""
    p = dict(STATE["params"])
    procs = STATE["processors"]
    for proc, switch in PROCESSOR_SWITCHES.items():
        p[switch] = proc in procs
    if "color_corrector" not in procs:
        p["ColorSwitch"] = False
        p["ColorMatchSwitch"] = False
    else:
        p["ColorSwitch"] = True
    for mask, switch in MASK_SWITCHES.items():
        p[switch] = mask in STATE["mask_types"]
    weight = float(p.get("StrengthSlider", 100))
    p["StrengthSwitch"] = abs(weight - 100) > 1e-6
    angle = STATE["detector_angle"]
    p["OrientAutoSwitch"] = angle == "auto"
    p["OrientSwitch"] = angle not in ("0", "auto")
    if p["OrientSwitch"]:
        p["OrientSlider"] = int(angle)
    return p


# ----- engine -----------------------------------------------------------------

def get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            if os.environ.get("ROPE_WEB_FAKE") == "1":
                from webapp.ui.fake_engine import FakeEngine
                _engine = FakeEngine()
            else:
                from rope.engine import RopeEngine
                backend = "trt" if STATE.get("backend") == "tensorrt" else "onnx"
                log(f"Loading engine ({STATE.get('backend')})…")
                _engine = RopeEngine(models_dir=ARGS.models_dir, backend=backend)
        return _engine


def reset_engine() -> None:
    global _engine
    with _engine_lock:
        _engine = None
        _source_cache.clear()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


_source_cache: dict[tuple, np.ndarray] = {}


def source_embedding(eng) -> np.ndarray | None:
    paths = tuple(STATE["source_paths"])
    if not paths:
        return None
    key = (paths, STATE["params"].get("MergeTextSel", "Mean"))
    if key not in _source_cache:
        _source_cache[key] = eng.source_embedding(list(paths), key[1])
    return _source_cache[key]


def apply_to_engine(eng) -> bool:
    """Push the current state into the engine. False = nothing to swap
    (no source face yet)."""
    eng.set_parameters(effective_params())
    try:
        src = source_embedding(eng)
    except ValueError:
        log("No face found in the source image(s).")
        return False
    if src is None:
        return False
    refs = []
    if STATE["face_selector_mode"] == "reference" and STATE["reference_embedding"] is not None:
        refs = [STATE["reference_embedding"]]
    eng.select_faces(src, mode="reference" if refs else "many", references=refs)
    return True


# ----- terminal log -----------------------------------------------------------

LOG_BUFFER = io.StringIO()
_log_lock = threading.Lock()
MAX_LOG_CHARS = 20000


def log(line: str, replace_prefix: str | None = None) -> None:
    """Append a line to the terminal. With replace_prefix, overwrite the
    last line if it starts with that prefix (progress updates)."""
    with _log_lock:
        text = LOG_BUFFER.getvalue()
        lines = text.splitlines()
        if replace_prefix and lines and lines[-1].startswith(replace_prefix):
            lines[-1] = line
        else:
            lines.append(line)
        text = "\n".join(lines)[-MAX_LOG_CHARS:]
        LOG_BUFFER.seek(0)
        LOG_BUFFER.truncate()
        LOG_BUFFER.write(text)


def read_logs() -> str:
    with _log_lock:
        return LOG_BUFFER.getvalue()


def clear_logs() -> None:
    with _log_lock:
        LOG_BUFFER.seek(0)
        LOG_BUFFER.truncate()


class _Tee(io.TextIOBase):
    """Mirror engine prints (model loading, warnings) into the terminal."""

    def __init__(self, stream):
        self._stream = stream
        self._partial = ""

    def write(self, s: str) -> int:
        self._stream.write(s)
        self._partial += s
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            if line.strip():
                log(line.rstrip())
        return len(s)

    def flush(self) -> None:
        self._stream.flush()


def capture_stdout() -> None:
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout)
