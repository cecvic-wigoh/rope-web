# Rope web app (Lightning AI / any Linux + NVIDIA GPU)

A browser UI over Rope's swap pipeline. It uses the same models and parameters as the desktop app, without the Qt window, so it works on headless GPU machines like Lightning AI Studios.

## Lightning AI Studio

1. Create a Studio and switch the machine to a **GPU** (L4 or A10G work well; T4 works with fewer threads).
2. Open a terminal and run:
   ```bash
   git clone <your fork or this repo> rope && cd rope
   bash webapp/setup_lightning.sh        # deps + ~1.5 GB of models, one time
   export ROPE_WEB_PASSWORD='choose-a-password'
   python webapp/app.py
   ```
3. Expose port **7860**: in the Studio's right sidebar, add the **Port viewer** (or "Web Port") plugin, then open port 7860. Log in with user `rope` and your password.

To keep it running after you close the terminal, use `nohup python webapp/app.py > rope.log 2>&1 &`.

## Using it

1. Upload one or more **source** face images. They are merged into one identity, using the Merge Math setting.
2. Upload the **target** video or image, then click **Find faces in target**. The app samples frames across the video and lists each distinct face.
3. Tick the faces to swap, pick a frame, and click **Preview frame** to tune the settings on the right.
4. Choose an optional frame range, confirm consent, and click **Render**. The result is an H.264 MP4 with the original audio, saved to `outputs/` and downloadable from the page.

Settings on the right are the same parameters as the desktop app. **Presets** are saved as `presets/<name>.json` in the config dir, so they are interchangeable with the desktop app's presets.

## Options

| Flag / env | Default | |
|---|---|---|
| `--models-dir` / `ROPE_MODELS` | `models/` | Model weights |
| `--output-dir` / `ROPE_OUTPUT_DIR` | `outputs/` | Rendered files |
| `--config-dir` / `ROPE_HOME` | repo root | Presets location |
| `--backend` / `ROPE_BACKEND` | `onnx` | `trt` = TensorRT: faster per frame, but builds engines for several minutes on first use (needs `pip install tensorrt`) |
| `--port` / `PORT` | 7860 | |
| `ROPE_WEB_USER` / `ROPE_WEB_PASSWORD` | `rope` / random | If no password is set, a random one is printed at startup |
| `--share` | off | Also create a public gradio.live link |
| `--no-auth` | off | Disable the login (not recommended on a shared URL) |

**Performance.** The **Threads** setting (Performance section) controls how many frames are swapped in parallel. Start at 2–4 on 24 GB GPUs; lower it if you run out of VRAM. The restorer and 256/512 swapper modes cost the most time.

## Scripting without the UI

```python
from rope.engine import RopeEngine
eng = RopeEngine(models_dir="models")
eng.set_parameters({"RestorerSwitch": True, "ThresholdSlider": 60})
src = eng.source_embedding(["me.jpg"])
faces = eng.scan_faces("clip.mp4")
eng.set_assignments([(f.embedding, src) for f in faces])
eng.render_video("clip.mp4", "outputs/clip_swap.mp4")
```

## Testing without a GPU

`ROPE_WEB_FAKE=1 python webapp/app.py` swaps the engine for a CPU stub that mirrors frames, so you can work on the UI locally.
