"""ffmpeg helpers for smooth in-browser playback.

- web_proxy(): small, fast-start H.264 copy of a video for the browser
  players. Phone videos are high bitrate and often keep their index (moov
  atom) at the end, so the browser must download most of the file before
  it can play or seek — painfully slow through a cloud proxy. Processing
  always uses the original file.
- encode_ts_chunk(): one H.264/AAC MPEG-TS segment for live streaming.
- install(): make Gradio's streaming Video work without system ffmpeg /
  ffprobe (Lightning Studios ship neither).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import threading
from pathlib import Path

import numpy as np

CACHE_DIR = Path(os.environ.get("ROPE_WEB_CACHE", Path(__file__).resolve().parents[2] / ".cache" / "web"))
PROXY_MAX_HEIGHT = 720
# Files already this light and web-friendly are served as-is.
PROXY_SKIP_BYTES = 25 * 1024 * 1024

CHUNK_DURATIONS: dict[str, float] = {}
_proxy_lock = threading.Lock()


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def install() -> None:
    """Expose the bundled ffmpeg on PATH and give Gradio chunk durations
    without ffprobe."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bin_dir = CACHE_DIR / "bin"
    bin_dir.mkdir(exist_ok=True)
    link = bin_dir / "ffmpeg"
    if not link.exists():
        link.symlink_to(ffmpeg_exe())
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"

    import gradio as gr

    original = gr.Video.get_video_duration_ffprobe

    def duration(filename: str):
        if filename in CHUNK_DURATIONS:
            return CHUNK_DURATIONS[filename]
        try:
            return original(filename)
        except Exception:
            return None

    gr.Video.get_video_duration_ffprobe = staticmethod(duration)


def _probe(path: str) -> dict:
    import cv2
    cap = cv2.VideoCapture(path)
    info = {"w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def _has_faststart(path: str) -> bool:
    """True if the mp4 'moov' index comes before 'mdat' (browser can start
    playback immediately)."""
    try:
        with open(path, "rb") as f:
            head = f.read(64 * 1024)
        moov, mdat = head.find(b"moov"), head.find(b"mdat")
        return moov != -1 and (mdat == -1 or moov < mdat)
    except OSError:
        return False


def web_proxy(path: str) -> str:
    """Return a browser-friendly version of `path` (cached)."""
    p = Path(path)
    st = p.stat()
    if (p.suffix.lower() == ".mp4" and st.st_size <= PROXY_SKIP_BYTES
            and _probe(path)["h"] <= PROXY_MAX_HEIGHT and _has_faststart(path)):
        return path
    key = hashlib.sha1(f"{p.resolve()}:{st.st_size}:{st.st_mtime}".encode()).hexdigest()[:16]
    out = CACHE_DIR / "proxy" / f"{p.stem[:40]}-{key}.mp4"
    with _proxy_lock:
        if out.exists():
            return str(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part.mp4")
        cmd = [
            ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", path,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", f"scale=-2:'min({PROXY_MAX_HEIGHT},ih)'",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
            "-g", "48", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"preview transcode failed: {result.stderr.strip()[-300:]}")
        tmp.replace(out)
    return str(out)


def encode_ts_chunk(frames: list[np.ndarray], fps: float, out_path: str, *, source: str | None,
                    start_seconds: float, offset_seconds: float) -> float:
    """Encode RGB frames (+ the matching slice of the source's audio) as
    one MPEG-TS segment. Returns its duration in seconds."""
    h, w = frames[0].shape[:2]
    duration = len(frames) / fps
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "pipe:0"]
    if source:
        cmd += ["-ss", f"{start_seconds:.6f}", "-t", f"{duration:.6f}", "-i", source,
                "-map", "0:v:0", "-map", "1:a:0?"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-crf", "23",
            "-pix_fmt", "yuv420p", "-g", str(len(frames)),
            "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "128k",
            "-output_ts_offset", f"{offset_seconds:.6f}", "-f", "mpegts", out_path]
    proc = subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames),
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"chunk encode failed: {proc.stderr.decode(errors='replace')[-300:]}")
    CHUNK_DURATIONS[out_path] = duration
    return duration
