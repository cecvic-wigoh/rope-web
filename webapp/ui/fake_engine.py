"""CPU stand-in for RopeEngine (ROPE_WEB_FAKE=1) so the UI can be
developed without a GPU. "Swapping" just tints and mirrors face boxes."""

from __future__ import annotations

import shutil
import threading
import time

import cv2
import numpy as np

from rope import engine as E
from rope.qt.parameters import default_values


class FakeEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.params = default_values()
        self.mode = "many"

    def set_parameters(self, values):
        self.params = {**default_values(), **values}

    def analyze(self, rgb, max_num=20):
        h, w = rgb.shape[:2]
        rng = np.random.default_rng(1)
        faces = []
        for i, (fx, fy) in enumerate([(0.3, 0.45), (0.7, 0.45)]):
            cx, cy, half = int(w * fx), int(h * fy), int(min(w, h) * 0.15)
            kps = np.array([[cx - half / 2, cy - half / 3], [cx + half / 2, cy - half / 3], [cx, cy],
                            [cx - half / 3, cy + half / 2], [cx + half / 3, cy + half / 2]], np.float32)
            faces.append(E.Face(rng.normal(size=512).astype(np.float32),
                                np.zeros((112, 112, 3), np.uint8), kps,
                                (cx - half, cy - half, cx + half, cy + half)))
        return faces

    def scan_faces(self, path, samples=12):
        return self.analyze(E.read_frame(path, 0))

    def source_embedding(self, paths, merge_mode=None):
        return np.ones(512, dtype=np.float32)

    def select_faces(self, source_emb, mode="many", references=(), others=()):
        self.mode = mode

    def swap_frame(self, rgb, frame_number=0):
        out = rgb.copy()
        faces = self.analyze(rgb)
        if self.mode != "many":
            faces = faces[:1]
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            out[y1:y2, x1:x2] = out[y1:y2, x1:x2][:, ::-1] // 2 + np.array([120, 20, 20], np.uint8)
        time.sleep(0.05)
        return out

    def render_image(self, in_path, out_path):
        cv2.imwrite(out_path, cv2.cvtColor(self.swap_frame(E.read_image_rgb(in_path)), cv2.COLOR_RGB2BGR))
        return out_path

    def render_video(self, in_path, out_path, *, start_frame=0, end_frame=None, progress=None,
                     cancel=None, threads=None, **_):
        n = max(1, (end_frame or 50) - start_frame)
        for i in range(n):
            time.sleep(0.03)
            if progress:
                progress(i + 1, n)
            if cancel is not None and cancel.is_set():
                raise E.Cancelled()
        shutil.copy(in_path, out_path)
        return out_path
