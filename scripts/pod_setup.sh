#!/usr/bin/env bash
# One-shot setup on a fresh RunPod RTX 5090 pod (PyTorch template, CUDA 12.8+).
#   HF_TOKEN=hf_... bash scripts/pod_setup.sh
# Then: bash scripts/pod_start.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/workspace}
COMFY=$WORK/ComfyUI

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

apt-get update -qq && apt-get install -y -qq ffmpeg git >/dev/null

if [[ ! -d $COMFY ]]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY"
fi
pip install -q -r "$COMFY/requirements.txt"

if [[ ! -d $COMFY/custom_nodes/ComfyUI-LTXVideo ]]; then
  git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo "$COMFY/custom_nodes/ComfyUI-LTXVideo"
fi
pip install -q -r "$COMFY/custom_nodes/ComfyUI-LTXVideo/requirements.txt" || true
ln -sfn "$REPO/node_pack/ComfyUI-DollyGrip" "$COMFY/custom_nodes/ComfyUI-DollyGrip"

pip install -q comfy-api-proxy -r "$REPO/app/requirements.txt" "huggingface_hub[cli]"
if [[ -n ${HF_TOKEN:-} ]]; then hf auth login --token "$HF_TOKEN" >/dev/null; fi
COMFYUI_DIR=$COMFY bash "$REPO/scripts/download_models.sh"

echo "setup done. next: bash $REPO/scripts/pod_start.sh"
