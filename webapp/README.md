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

### Large videos

Browser uploads go through Lightning's proxy at roughly your upload speed; a 500 MB phone video can take many minutes. For large files, copy them into `rope-web/inputs/` on the Studio instead, then pick them under **TARGET → Or pick from inputs folder** (click ↻ to refresh the list). Ways to copy:
- drag the file into the `rope-web/inputs` folder in the Studio's file browser, or
- from your computer, run:
  ```bash
  lightning studio cp ./video.mp4 rope-web/inputs/ --name rope-web --teamspace <owner>/<teamspace>
  ```

## Using it

The interface follows FaceFusion's three-column layout:

| Column | What's there |
|---|---|
| **Left** | **Processors** (`face_swapper`, `face_enhancer`, `color_corrector`, `face_adjuster`) and the options for each enabled processor. Below them: execution provider (CUDA / TensorRT), thread count, output quality and encoder, and presets. |
| **Middle** | **Source** face image(s), **Target** image or video, output path, the **terminal** (live log and progress), and **Start / Stop / Clear**. |
| **Right** | **Live preview**, which re-renders on every change. Preview modes are `default`, `frame-by-frame` (before and after side by side) and `face-by-face`. Also here: preview frame slider, **trim frame** range, **face selector**, **face masker** and **face detector**. |

**Face selector modes**

- `reference`: click a face in the gallery to swap only that person.
- `many`: swap every face.

**Face mask types**

| Type | What it does |
|---|---|
| `box` | Feathered border |
| `occlusion` | Hands and objects |
| `region` | Face parser, including the mouth |
| `xseg` | DFL XSeg mask |
| `diff` | Keeps unchanged pixels |

Results are saved to the output path (default `outputs/`) and shown in the OUTPUT player with a download button. **Presets** are saved as `presets/<name>.json` in the config dir and are interchangeable with the desktop app's presets.

## Options

| Flag / env | Default | |
|---|---|---|
| `--models-dir` / `ROPE_MODELS` | `models/` | Model weights |
| `--output-dir` / `ROPE_OUTPUT_DIR` | `outputs/` | Rendered files |
| `--config-dir` / `ROPE_HOME` | repo root | Presets location |
| `--backend` / `ROPE_BACKEND` | `cuda` | `tensorrt`: faster per frame, but builds engines for several minutes on first use (needs `pip install tensorrt`). Also switchable in the UI. |
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
