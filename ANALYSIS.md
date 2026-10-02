# Rope — codebase analysis

Rope ("Rope-Bronze") is a desktop face-swap application built around insightface's
`inswapper_128` model. It has a PySide6 GUI, GPU video decode/playback, a range of
masking, restoration and color tools, a live screen-capture mode, and video export.
It is about 15k lines of Python.

## Layout

| Path | Role |
|---|---|
| `Rope.py` → `rope/qt/app.py` | Entry point: builds `Models`, `VideoManager`, `MainWindow`, `Coordinator` |
| `rope/Models.py` | Every ONNX session (detectors, ArcFace, swappers, restorers, maskers) with TensorRT/CUDA providers |
| `rope/VideoManager.py` | Playback, recording, and the per-frame swap pipeline (`swap_video` → `swap_core`) |
| `rope/MediaPlayer.py` | Decode: NVDEC (PyNvVideoCodec / torchcodec) or PyAV CPU fallback; audio via sounddevice |
| `rope/qt/bus.py` | Global Qt signal bus between the GUI and the VideoManager |
| `rope/qt/coordinator.py` | Wires bus signals to VideoManager methods; pushes frames to the GUI thread |
| `rope/qt/main_window.py` | Most UI logic: face gallery, markers, record flow, folder pickers |
| `rope/qt/parameters.py` + `_default_data.py` | Parameter schema (sliders, switches, selects) and defaults |
| `rope/qt/panes/parameters_pane.py` | Builds the Parameters and Settings tabs from the schema |
| `rope/qt/widgets/preview.py` | OpenGL preview, with a CUDA→GL zero-copy path |

## Frame pipeline

1. **Decode.** `MediaPlayer` uses NVDEC or PyAV and feeds a queue of depth 8.
2. **Dispatch.** A pacer thread calls `VideoManager.process()`, which fans frames out to a thread pool (`ThreadsSlider`).
3. **Prepare.** `swap_video` applies the marker parameter snapshot, uploads to CUDA, upscales to at least 512 px, and handles orientation.
4. **Detect.** `Models.run_detect` runs RetinaFace (`det_10g`) or SCRFD, producing 5-point landmarks.
5. **Embed.** ArcFace (`w600k_r50`) produces an embedding. Cosine similarity matches it against the assigned source faces (`ThresholdSlider`).
6. **Swap.** `swap_core` takes an aligned crop and builds the latent from the source embedding and the inswapper emap. It then runs one of:
   - a polyphase 128 pass at 128/256/512 resolution;
   - native 512;
   - native 256.

   Optional steps: Strength iterations, a 2-pass High-Fidelity identity correction, and LAB color matching.
7. **Restore and mask.**
   - Restorers: GFPGAN, GPEN-256/512, or CodeFormer.
   - Masks: border, diff, occluder, DFL XSeg, and face-parser (including mouth).
8. **Blend.** The face is inverse-warped into the frame and composited; the frame is then un-rotated.
9. **Output.** The result goes to the preview or to the recorder.
   - The recorder pipes frames to `ffmpeg libx264` (or OpenCV `mp4v`).
   - Audio is muxed from the source afterwards.

## Platform reality

The swap path is **NVIDIA-only**:

- `Models.__init__` allocates `cuda:0` tensors.
- Every ORT session uses the `CUDAExecutionProvider` or `TensorrtExecutionProvider` with CUDA IO binding.
- `VideoManager` uses `torch.cuda` streams.

On macOS (Apple Silicon) or CPU-only machines the **UI runs**, after the macOS OpenGL fixes in this branch, but `Models()` construction fails. Porting would mean abstracting the device (`cuda`/`mps`/`cpu`), adding CoreML/CPU providers, and replacing IO binding with numpy I/O. That is substantial but mechanical work. `onnxruntime` on macOS already exposes `CoreMLExecutionProvider`.

## Changes in this branch (`customization-usability`)

**Customization**

- Config directory abstraction (`rope/qt/paths.py`): `--config-dir` / `$ROPE_HOME`. Settings, params and icons no longer depend on the launch directory. Models default to `<repo>/models` or `$ROPE_MODELS`.
- CLI flags: `--config-dir`, `--models-dir`, `--output-dir`, `--preset`, `--stylesheet`, `--no-backend`, `--print-config`.
- Named parameter presets (`rope/qt/presets.py`) with a Preset row in the Parameters tab (select / Save As… / Delete). The last-used preset is remembered.
- Rebindable keyboard shortcuts and a configurable nudge step (`data.json` → `shortcuts`, `nudge_frames`).
- UI font override (`ui_font_family`, `ui_font_size`) and a user stylesheet (`user.qss`).

**Usability**

- The bottom status bar is visible again. The ~40 status and error messages and every widget's hover help were previously routed to a hidden label.
- `data.json` is written atomically.
- `rope.sh` launcher for macOS/Linux; `Rope.bat` forwards CLI args.
- macOS: request a GL 4.1 core context, fall back from `glTexStorage2D`, and style tabs and combo boxes so native styling doesn't wash them out.

**Bug fixes**

- `StrengthSlider = 0` crashed `swap_core` (`UnboundLocalError`).
- Frames before the first marker used the *last* marker's parameters.
- The diff mask subtracted `uint8` tensors, which wrapped around.
- A failed audio mux deleted the only copy of the render. The silent video is now kept.
- Recording with no video loaded, an image loaded, or no output folder crashed. It now stops cleanly.
- After a reload, the ResNet50 priors were appended twice, causing a shape mismatch.
- `--no-backend` with a saved models folder crashed at startup. The leftover debug status-bar probe was also removed.

## Known issues not addressed here

These are documented for follow-up:

- **Rotation canvas.** Non-90° rotation uses `expand=True`, so the output canvas size changes from frame to frame. This breaks the fixed-size OpenCV writer.
- **Thread safety.** Orientation EMA state is mutated from several worker threads without a lock. Lazy singleton model init is also unlocked.
- **Recording finalization.** It can fail if stopped while no frames are in flight. End of video depends on a metadata frame-count estimate.
- **Persistence.** Markers aren't persisted, and there is no project/session file covering media, face assignments and markers together.
- **No headless batch mode.** `VideoManager` imports the Qt bus, so it needs a Qt event loop.
- **Dead code.** `MediaCache.store_face` doesn't exist, the `PARAMS` dict in `_default_data.py` is unused, and so are the `TensorRTEngine` constants.

## Tests

Tests need no GPU. They run as scripts from the repo root, with `QT_QPA_PLATFORM=offscreen` for the UI ones:

```
python -m rope.qt.tests.test_parameters
python -m rope.qt.tests.test_customization
python -m rope.qt.tests.smoke
python -m rope.qt.tests.phase_c_e2e
```
