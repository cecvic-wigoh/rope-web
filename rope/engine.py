"""Headless Rope engine — the swap pipeline without the Qt GUI.

Drives the same Models + VideoManager code the desktop app uses, so
results match the GUI for the same parameters. Used by the web app
(webapp/app.py) and usable from scripts or notebooks:

    from rope.engine import RopeEngine
    eng = RopeEngine(models_dir="models")
    src = eng.source_embedding(["me.jpg"])
    targets = eng.scan_faces("clip.mp4")
    eng.set_assignments([(t.embedding, src) for t in targets])
    eng.render_video("clip.mp4", "out.mp4")

Requires an NVIDIA GPU (the pipeline is CUDA-only).
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

# torch must be imported before onnxruntime: on Linux, onnxruntime-gpu
# then resolves cuDNN/cuBLAS from the libraries torch already loaded.
import torch  # noqa: F401  (import-order side effect)
import cv2
import numpy as np

try:
    import onnxruntime as _ort
    if hasattr(_ort, "preload_dlls"):
        _ort.preload_dlls()
except Exception:  # pragma: no cover - best effort
    pass

from rope.qt.parameters import PARAMETER_BY_NAME, default_values, seed_control_dict
from rope.qt.parameters_migration import _coerce

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Face:
    embedding: np.ndarray   # (512,) ArcFace embedding
    thumbnail: np.ndarray   # 112x112x3 RGB uint8 aligned crop
    kps: np.ndarray | None = None              # (5, 2) landmarks, image coords
    bbox: tuple[int, int, int, int] | None = None  # x1, y1, x2, y2 estimate

    def crop(self, rgb: np.ndarray, size: int = 128, margin: float = 0.25) -> np.ndarray:
        """Square crop of this face from `rgb` (the image it was found in)."""
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = self.bbox or (0, 0, w, h)
        mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(w, x2 + mx), min(h, y2 + my)
        region = rgb[y1:y2, x1:x2]
        if region.size == 0:
            region = rgb
        side = min(region.shape[:2])
        cy, cx = region.shape[0] // 2, region.shape[1] // 2
        square = region[cy - side // 2: cy - side // 2 + side, cx - side // 2: cx - side // 2 + side]
        return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def _bbox_from_kps(kps: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    """Approximate face box from the 5 landmarks (eyes, nose, mouth corners)."""
    center = kps.mean(axis=0)
    eye_dist = float(np.linalg.norm(kps[1] - kps[0]))
    eye_mouth = float(np.linalg.norm((kps[3] + kps[4]) / 2 - (kps[0] + kps[1]) / 2))
    half = 1.15 * max(eye_dist, eye_mouth, 4.0)
    cx, cy = float(center[0]), float(center[1]) - 0.1 * half
    return (max(0, int(cx - half)), max(0, int(cy - half * 1.15)),
            min(w, int(cx + half)), min(h, int(cy + half * 1.05)))


class Cancelled(Exception):
    pass


def cosine_similarity_pct(a: np.ndarray, b: np.ndarray) -> float:
    """Same scale as VideoManager.findCosineDistance / ThresholdSlider:
    100 = identical, 50 = orthogonal."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return 100.0 - (1.0 - float(np.dot(a, b)) / denom) * 50.0


def read_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


class RopeEngine:
    def __init__(self, models_dir: str | None = None, backend: str = "onnx"):
        """backend: 'onnx' (CUDA EP, starts fast) or 'trt' (TensorRT EP,
        faster per frame but builds engines on first use — minutes)."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Rope's swap pipeline needs an NVIDIA GPU with CUDA; "
                "torch.cuda.is_available() is False."
            )
        from rope.Models import Models
        from rope.VideoManager import VideoManager

        self.models = Models()
        if models_dir:
            self.models.set_models_folder(os.path.abspath(models_dir))
        if backend in ("onnx", "trt"):
            for attr in Models._BACKEND_PREF_ATTRS:
                self.models.set_backend_preference(attr, backend, unload=False)
        self.vm = VideoManager(self.models)
        self.vm.control = seed_control_dict()
        self.vm.control["SwapFacesButton"] = True
        self.vm.control["MaskViewButton"] = False
        self.vm.found_faces = []
        self.vm.markers = []
        self.set_parameters({})
        # One render/preview at a time: VideoManager state is shared.
        self.lock = threading.RLock()

    # ----- parameters ------------------------------------------------------

    @staticmethod
    def default_parameters() -> dict[str, Any]:
        return default_values()

    def set_parameters(self, values: dict[str, Any]) -> dict[str, Any]:
        """Apply values on top of the defaults (unknown keys dropped,
        types coerced like a saved_parameters.json load)."""
        params = default_values()
        for name, value in values.items():
            if name in PARAMETER_BY_NAME and name in params:
                params[name] = _coerce(name, value)
        for name, value in params.items():
            p = PARAMETER_BY_NAME.get(name)
            # Integer-step sliders must stay ints: swap_core uses several
            # (Border*, blur sizes) as slice indices / kernel sizes. Web
            # sliders and saved JSON deliver floats.
            if p is not None and p.kind == "slider" and float(p.inc).is_integer():
                params[name] = int(round(float(value)))
        self.vm.parameters = params
        mode = str(params.get("ModelSessionsTextSel", "Shared"))
        if hasattr(self.models, "set_model_session_mode"):
            self.models.set_model_session_mode(mode)
        return params

    @property
    def parameters(self) -> dict[str, Any]:
        return self.vm.parameters

    # ----- faces -----------------------------------------------------------

    def analyze(self, rgb: np.ndarray, max_num: int = 20) -> list[Face]:
        """Detect + embed every face in an RGB HxWx3 uint8 image."""
        p = self.vm.parameters
        img = torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.uint8)).to("cuda").permute(2, 0, 1)

        def detect(t):
            return self.models.run_detect(
                t,
                str(p.get("DetectTypeTextSel", "Retinaface")),
                max_num=max_num,
                score=float(p.get("DetectScoreSlider", 50)) / 100.0,
                input_size=int(p.get("DetectInputSizeTextSel", 640)),
            )

        pad = 0
        kpss = detect(img)
        if len(kpss) == 0:
            # Tightly cropped portraits (face filling the frame) defeat
            # RetinaFace/SCRFD; retry with a black margin around the image.
            pad = max(img.shape[1], img.shape[2]) // 4
            img = torch.nn.functional.pad(img, (pad, pad, pad, pad))
            kpss = detect(img)
        faces = []
        h, w = rgb.shape[:2]
        for kps in kpss:
            emb, crop = self.models.run_recognize(img, kps)
            pts = np.asarray(kps, dtype=np.float32).reshape(-1, 2) - pad
            faces.append(Face(
                embedding=np.asarray(emb, dtype=np.float32),
                thumbnail=crop.cpu().numpy().astype(np.uint8),
                kps=pts,
                bbox=_bbox_from_kps(pts, w, h),
            ))
        # Left-to-right, like FaceFusion's default face order.
        faces.sort(key=lambda f: f.bbox[0])
        return faces

    def source_embedding(self, image_paths: Iterable[str], merge_mode: str | None = None) -> np.ndarray:
        """Embedding of the largest-confidence face in each image, merged
        with MergeTextSel (Mean / Median / Sph / Geo / Qual)."""
        from rope.EmbeddingMerge import combine

        embs = []
        for path in image_paths:
            faces = self.analyze(read_image_rgb(path), max_num=1)
            if faces:
                embs.append(faces[0].embedding)
        if not embs:
            raise ValueError("No face detected in any source image")
        mode = merge_mode or str(self.vm.parameters.get("MergeTextSel", "Mean"))
        return np.asarray(combine(embs, mode), dtype=np.float32)

    def dedupe(self, faces: Iterable[Face], threshold: float | None = None) -> list[Face]:
        thr = float(self.vm.parameters["ThresholdSlider"]) if threshold is None else threshold
        unique: list[Face] = []
        for f in faces:
            if all(cosine_similarity_pct(f.embedding, u.embedding) < thr for u in unique):
                unique.append(f)
        return unique

    def scan_faces(self, path: str, samples: int = 12) -> list[Face]:
        """Distinct faces in an image, or across `samples` evenly spaced
        frames of a video."""
        if is_image(path):
            return self.dedupe(self.analyze(read_image_rgb(path)))
        found: list[Face] = []
        for _idx, rgb in sample_frames(path, samples):
            found.extend(self.analyze(rgb))
        return self.dedupe(found)

    def select_faces(self, source_emb: np.ndarray, mode: str = "many",
                     references: Sequence[np.ndarray] = (),
                     others: Sequence[np.ndarray] = ()) -> None:
        """mode 'many': swap every detected face with the source.
        mode 'reference': swap only faces matching `references`. `others`
        are embeddings of the other people in the target; a detected face
        closer to one of them than to a reference is left alone, which
        keeps similar-looking people from being swapped together."""
        if mode == "reference" and references:
            self.set_assignments([(r, source_emb) for r in references], decoys=others)
        else:
            self.set_assignments([(None, source_emb)])

    def set_assignments(self, pairs: Sequence[tuple[np.ndarray | None, np.ndarray]],
                        decoys: Sequence[np.ndarray] = ()) -> None:
        """pairs: (target face embedding, source embedding to put on it)."""
        slots = []
        for target_emb, source_emb in pairs:
            slots.append({
                "MatchAll": target_emb is None,
                "Embedding": None if target_emb is None else np.asarray(target_emb, dtype=np.float32),
                "SourceFaceAssignments": ["web"],
                "AssignedEmbedding": np.asarray(source_emb, dtype=np.float32),
                "Thumbnail": None,
                "HFCorrectionGap": None,
                "HFCorrectionGapSamples": 0,
                "HFRefinePending": False,
            })
        for emb in decoys:
            slots.append({
                "Decoy": True,
                "Embedding": np.asarray(emb, dtype=np.float32),
                "SourceFaceAssignments": [],
                "AssignedEmbedding": None,
            })
        self.vm.found_faces = slots
        sources = [s["AssignedEmbedding"] for s in slots if not s.get("Decoy")]
        if sources and hasattr(self.models, "set_session_mean_embedding"):
            self.models.set_session_mean_embedding(np.mean(np.stack(sources), axis=0))
        if hasattr(self.vm, "clear_latent_cache"):
            self.vm.clear_latent_cache()

    # ----- swapping --------------------------------------------------------

    def swap_frame(self, rgb: np.ndarray, frame_number: int = 0) -> np.ndarray:
        """Swap one RGB HxWx3 uint8 frame; returns RGB uint8 of the same size."""
        h, w = rgb.shape[:2]
        out = self.vm.swap_video(np.ascontiguousarray(rgb), frame_number, False)
        out = out.cpu().numpy()
        if out.shape[:2] != (h, w):
            # Fine-angle rotation can grow the canvas; keep the frame size fixed.
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
        return out

    def render_image(self, in_path: str, out_path: str) -> str:
        with self.lock:
            result = self.swap_frame(read_image_rgb(in_path))
        cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        return out_path

    def render_video(
        self,
        in_path: str,
        out_path: str,
        *,
        start_frame: int = 0,
        end_frame: int | None = None,
        progress: Callable[[int, int], None] | None = None,
        cancel: threading.Event | None = None,
        threads: int | None = None,
        video_encoder: str = "libx264",
        video_preset: str = "medium",
    ) -> str:
        """Swap every frame in [start_frame, end_frame) and encode an H.264
        MP4 with the source's audio track (if any)."""
        from rope.VideoManager import _find_ffmpeg

        ffmpeg = _find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found (pip install imageio-ffmpeg)")
        info = video_info(in_path)
        fps, total, w, h = info["fps"], info["frames"], info["width"], info["height"]
        end = total if end_frame is None or end_frame <= 0 else min(end_frame, total)
        start = max(0, min(start_frame, max(end - 1, 0)))
        n_frames = max(end - start, 0)
        crf = int(self.vm.parameters.get("VideoQualSlider", 18))
        workers = int(threads or self.vm.parameters.get("ThreadsSlider", 2))

        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}",
            "-i", "pipe:0",
            "-ss", f"{start / fps:.6f}", "-t", f"{n_frames / fps:.6f}", "-i", in_path,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", video_encoder, "-crf", str(crf), "-preset", video_preset, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            out_path,
        ]
        if video_encoder == "libx265":
            cmd[-1:-1] = ["-tag:v", "hvc1"]  # playable in browsers / QuickTime
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        cap = cv2.VideoCapture(in_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        done = 0
        try:
            with self.lock, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                pending: deque = deque()
                idx = start
                while True:
                    # Keep the pool fed, then write results strictly in order.
                    while idx < end and len(pending) < workers * 2:
                        ok, bgr = cap.read()
                        if not ok:
                            end = idx
                            break
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        pending.append(pool.submit(self.swap_frame, rgb, idx))
                        idx += 1
                    if not pending:
                        break
                    frame = pending.popleft().result()
                    enc.stdin.write(np.ascontiguousarray(frame).tobytes())
                    done += 1
                    if progress:
                        progress(done, n_frames)
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
        except BaseException:
            enc.kill()
            enc.wait()
            if os.path.exists(out_path):
                os.remove(out_path)
            raise
        finally:
            cap.release()
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="replace")
        if enc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()[-800:]}")
        return out_path


# ----- video helpers ----------------------------------------------------------

def video_info(path: str) -> dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    info = {
        "fps": fps if fps > 0 else 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def read_frame(path: str, index: int) -> np.ndarray:
    if is_image(path):
        return read_image_rgb(path)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {index} of {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sample_frames(path: str, samples: int):
    total = max(video_info(path)["frames"], 1)
    picks = sorted({int(i) for i in np.linspace(0, total - 1, num=max(1, samples))})
    cap = cv2.VideoCapture(path)
    try:
        for i in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = cap.read()
            if ok:
                yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
