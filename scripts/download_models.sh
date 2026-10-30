#!/usr/bin/env bash
# Download the open weights DollyGrip needs into a ComfyUI install.
# LTX-2.5 is gated: accept the terms at https://huggingface.co/Lightricks/LTX-2.5 and `hf auth login` first.
set -euo pipefail

COMFYUI_DIR=${COMFYUI_DIR:?set COMFYUI_DIR to your ComfyUI checkout}
M=$COMFYUI_DIR/models
command -v hf >/dev/null || pip install -q "huggingface_hub[cli]"

get() {  # repo  path-in-repo  dest-dir
  local repo=$1 file=$2 dest=$3
  mkdir -p "$dest"
  if [[ -f $dest/$(basename "$file") ]]; then echo "have $(basename "$file")"; return; fi
  hf download "$repo" "$file" --local-dir "$M/.hf/$repo"
  ln -sf "$M/.hf/$repo/$file" "$dest/$(basename "$file")"
}

LTX=Lightricks/LTX-2.5
# LTX_VARIANT=nvfp4 for Blackwell (RTX 5090, 18.7GB); int8 for Ada/Ampere (RTX 4090, 21.5GB).
# LTX_VARIANT=gguf: unsloth Q4_K_M (13.6GB) via ComfyUI-GGUF; stays resident next to the text encoder on 24GB.
case ${LTX_VARIANT:-int8} in
  gguf)  get unsloth/LTX-2.5-GGUF distilled/ltx-2.5-22b-distilled-Q4_K_M.gguf "$M/diffusion_models" ;;
  nvfp4) get $LTX diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors "$M/diffusion_models" ;;
  int8)  get $LTX diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors "$M/diffusion_models" ;;
esac
get $LTX text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors "$M/text_encoders"
get $LTX vae/ltx-2.5-video-vae-bf16.safetensors                                 "$M/vae"
get $LTX vae/ltx-2.5-audio-vae-bf16.safetensors                                 "$M/checkpoints"  # LTXVAudioVAELoader reads checkpoints/
get $LTX latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors "$M/latent_upscale_models"
get Cseti/LTX2.3-22B_IC-LoRA-Cameraman_v2 LTX2.3-22B_IC-LoRA-Cameraman_v2_14000.safetensors "$M/loras"

echo "done. models in $M"
