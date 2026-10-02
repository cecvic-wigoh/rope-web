"""Download Rope's model weights (the "Bronze" release, ~1.5 GB zip).

    python webapp/download_models.py [--models-dir models]

Skips the download when the core models are already present.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/Hillobar/Rope/releases/download/Bronze/models.zip"
REQUIRED = ["det_10g.onnx", "w600k_r50.onnx", "inswapper_128.fp16.onnx"]
REPO_ROOT = Path(__file__).resolve().parents[1]


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(8 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    tmp.replace(dest)


def extract(zip_path: Path, models_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            # Flatten: the archive may wrap files in a "models/" folder.
            name = Path(member.filename).name
            if not name or name.startswith("."):
                continue
            with zf.open(member) as src, open(models_dir / name, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(REPO_ROOT / "models"))
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if all((models_dir / m).is_file() for m in REQUIRED):
        print(f"Models already present in {models_dir}")
        return 0
    zip_path = models_dir / "models.zip"
    if not zip_path.is_file():
        print(f"Downloading {URL}")
        download(URL, zip_path)
    print(f"Extracting into {models_dir}")
    extract(zip_path, models_dir)
    if not args.keep_zip:
        zip_path.unlink()
    missing = [m for m in REQUIRED if not (models_dir / m).is_file()]
    if missing:
        print(f"Still missing after extract: {missing}")
        return 1
    print("Done:", ", ".join(sorted(p.name for p in models_dir.glob("*.onnx"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
