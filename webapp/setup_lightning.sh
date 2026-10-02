#!/usr/bin/env bash
# One-time setup for a Lightning AI Studio (or any Linux + NVIDIA GPU box).
#   bash webapp/setup_lightning.sh
set -euo pipefail
cd "$(dirname "$0")/.."

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "No NVIDIA GPU visible — switch the Studio to a GPU machine first."; exit 1; }

if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "Installing CUDA build of torch/torchvision..."
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

pip install -r webapp/requirements.txt

# TensorRT (~1.6x faster swaps). Match the CUDA major onnxruntime-gpu was built for.
ORT_CAPI=$(python -c "import onnxruntime,os;print(os.path.join(os.path.dirname(onnxruntime.__file__),'capi'))")
CUDA_MAJOR=$(ldd "$ORT_CAPI/libonnxruntime_providers_tensorrt.so" 2>/dev/null | grep -oE 'libcudart\.so\.[0-9]+' | head -1 | grep -oE '[0-9]+$' || true)
if [ -n "$CUDA_MAJOR" ]; then
  pip install "tensorrt-cu${CUDA_MAJOR}==10.16.1.11" || echo "TensorRT install failed - the app will use CUDA instead."
fi
python webapp/download_models.py --models-dir models

python - <<'PY'
import torch, onnxruntime as ort
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
print("onnxruntime", ort.__version__, "| providers", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers(), "onnxruntime-gpu CUDA provider missing"
PY
echo
echo "Setup complete. Start the app with:  python webapp/app.py"
