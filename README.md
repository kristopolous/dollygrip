# DollyGrip

**Named camera moves as real camera paths, on open weights.**

Pick "360 Orbit", "Crash Zoom" or "Dolly Zoom (Vertigo)", upload a start image, and get a video whose camera
does *exactly* that move, the same way every time.

Closed "cinema" apps implement camera presets as prompt text: the preset is a sentence handed to a video model,
so the trajectory is a suggestion. You can't set the speed, the travel distance or the easing, and two runs give
two different moves. DollyGrip treats a preset as a **parametric 3D camera trajectory**. It renders that
trajectory as a reference clip and drives [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) with a camera-motion
IC-LoRA, so the move is specified, not hoped for.

> A *dolly grip* is the crew member who pushes the camera dolly and hits the same marks on every take.

Built for Comfy's **#OpenSourceHiggsfield** challenge. Everything runs on a single consumer GPU (RTX 4090 / 5090) with open weights:
no partner nodes, no closed APIs.

---

## How it works

```
 preset / DSL / share code
          │
          ▼
 ┌──────────────────────────┐   rig parameters per frame (yaw, pitch, dolly, truck, pedestal,
 │  dollygrip.trajectory    │   pan, tilt, roll, zoom, vertigo) → camera pose + FOV per frame
 └──────────┬───────────────┘
            ▼
 ┌──────────────────────────┐   CPU ray-cast of a synthetic set with strong parallax cues
 │  dollygrip.render        │   (or the start image as a card) → guide clip, 8n+1 frames
 └──────────┬───────────────┘
            ▼
 ┌──────────────────────────┐   LTXAddVideoICLoRAGuide ← guide clip
 │  ComfyUI + LTX-2.5       │   LTXVImgToVideoConditionOnly ← start image
 │  + Cameraman IC-LoRA     │   two-stage sampling (half res → x2 latent upscale), native audio
 └──────────┬───────────────┘
            ▼
        MP4 + audio
```

Everything up to the ComfyUI box is deterministic and plain numpy. The same shot DSL, duration and fps produce a
bit-identical guide clip, which is what makes a move repeatable and shareable.

### The shot DSL

Every preset is a one-line program. You can edit a preset's string to make your own move.

```
dolly_in
orbit_360
0%-40%: static then 40%-100%: crash_zoom_in
0%-30%: static then 30%-62%: whip_pan_left then 62%-100%: static
dolly_in@0.5 then static then dolly_in@0.6        # unranged segments split the shot evenly
dutch_angle + dolly_in@0.4                        # "+" runs moves at the same time
0-48: dolly_in then 48-96: pan_left@0.6 ease_out  # frame, seconds (2s) or percent (40%) ranges
```

- **Moves:** `static`, `dolly_in/out`, `super_dolly_in`, `truck_left/right`, `pedestal_up/down`, `pan_left/right`,
  `tilt_up/down`, `roll_cw/ccw`, `whip_pan_left/right`, `dutch_angle`, `arc_left/right`, `orbit_360`, `bullet_time`,
  `crane_up/down`, `top_down`, `robo_arm`, `helicopter`, `fpv_drone`, `zoom_in/out`, `crash_zoom_in/out`,
  `dolly_zoom_in/out`, `handheld`.
- **Easings:** `linear`, `ease_in`, `ease_out`, `ease_in_out`, `expo_in`, `expo_out`, `whip`.
- **`@k`:** scales a move's travel.

Travel is fixed per move and speed comes from duration, so a 2 s crash zoom is faster than a 6 s one: one move,
stated travel, speed from time.

### Share codes

A move round-trips through a short string such as `dg1-eNqrVkpRslIqzs9NV...`. Paste it into the app to reproduce
someone else's exact camera path on your own image.

## Presets

| Preset | DSL |
|---|---|
| Dolly In | `dolly_in` |
| Dolly Out | `dolly_out` |
| 360 Orbit | `orbit_360` |
| Crash Zoom In | `0%-40%: static then 40%-100%: crash_zoom_in` |
| Crash Zoom Out | `0%-60%: crash_zoom_out then 60%-100%: static` |
| Dolly Zoom (Vertigo) | `dolly_zoom_in` |
| Bullet Time | `bullet_time` |
| FPV Drone | `fpv_drone` |
| Crane Up | `crane_up` |
| Whip Pan | `0%-30%: static then 30%-62%: whip_pan_left then 62%-100%: static` |

There are 14 more: Arc L/R, Crane Down, Super Dolly, Robo Arm, Helicopter, Overhead, Truck L/R, Pan, Tilt,
Dutch Angle, Handheld and Double Dolly. See `node_pack/ComfyUI-DollyGrip/dollygrip/presets.py`.

## Use it in ComfyUI

Install the node pack, then open **Templates → ComfyUI-DollyGrip → dollygrip_ltx25_i2v**. It opens in
**App Mode**: start image, move, prompt, duration, strength, guide scene and seed on one screen, a Run button, and
the video next to its camera path. Toggle to the node graph any time. Save it with a `.app.json` name to keep it in
the Apps sidebar. It's the full LTX-2.5
two-stage image-to-video graph, grouped as Camera → Models → Prompt → Stage 1 → Stage 2 → Output. You can also
drop **DollyGrip Camera** into your own LTX workflow:

| DollyGrip Camera output | wire to |
|---|---|
| `guide` | `LTXAddVideoICLoRAGuide.image` (with the Cameraman IC-LoRA loaded via `LTXICLoRALoaderModelOnly`) |
| `frames` | `EmptyLTXVLatentVideo.length`, `LTXVEmptyLatentAudio.frames_number` |
| `fps` / `fps_int` | `LTXVConditioning.frame_rate`, `CreateVideo.fps` / `LTXVEmptyLatentAudio.frame_rate` |
| `stage1_width/height`, `width/height` | half-res latent + start-image scale, full-res start-image scale |
| `prompt` | `CLIPTextEncode.text` (your prompt with the move's camera words appended) |
| `seed` | `RandomNoise.noise_seed` |
| `trajectory_json` | per-frame camera-to-world matrices + FOV, for other camera-control models |

The node is the single source of truth for the shot. Change the duration or the move there, and frame counts and
sizes stay consistent everywhere downstream.

## Make any workflow runnable: `resolve_workflow.py`

ComfyUI's "Setup required" panel tells you what's missing. This script fixes it:

```bash
python scripts/resolve_workflow.py my_workflow.json --comfy-dir ~/ComfyUI            # show the plan
python scripts/resolve_workflow.py my_workflow.json --comfy-dir ~/ComfyUI --apply \
    --sample-image example.png --install-to ~/ComfyUI/user/default/workflows
```

- **Node packs:** read from each node's `cnr_id`/`ver` and installed from the Comfy Registry (or `aux_id` → git clone),
  with their requirements.
- **Models:** downloaded from the workflow's embedded `properties.models` URLs. Bare filenames are looked up with
  `comfy build refs resolve` (Hugging Face + Civitai). Gated HF repos use your HF token.
- **Anything with no public source** gets patched into a `*.resolved.json`: a close sibling model in the same folder
  is swapped in (e.g. another distilled LoRA of the same base), unknown LoRAs are bypassed, and missing input media
  points at a sample image.

## Repository layout

```
node_pack/ComfyUI-DollyGrip/   ComfyUI custom node: "DollyGrip Camera"
  example_workflows/           drag-and-drop / template-browser workflow (LTX-2.5 + Cameraman)
  dollygrip/trajectory.py      rig, moves, easing, DSL → per-frame cameras
  dollygrip/render.py          guide-clip renderer (numpy)
  dollygrip/presets.py         named presets
  dollygrip/sharecode.py       share codes
app/                           web app (FastAPI + static UI), talks to ComfyUI via comfy-sdk
workflows/                     API-format workflows (*.api.json)
scripts/                       model download, workflow build, guide preview tools
tests/                         determinism / DSL / share-code tests (no GPU needed)
```

## Run it locally

**You need:** an NVIDIA GPU with 24 GB+ (developed on an RTX 4090 with the int8 weights; on an RTX 5090 set
`LTX_VARIANT=nvfp4` for the faster NVFP4 build), Python 3.10+,
a recent ComfyUI, and a Hugging Face account that has accepted the LTX-2.5 terms.

```bash
# 1. ComfyUI + custom nodes
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI && pip install -r requirements.txt
git clone https://github.com/Lightricks/ComfyUI-LTXVideo custom_nodes/ComfyUI-LTXVideo
ln -s /path/to/dollygrip/node_pack/ComfyUI-DollyGrip custom_nodes/ComfyUI-DollyGrip

# 2. Models (LTX-2.5 is gated: accept the terms on HF, then `hf auth login`)
COMFYUI_DIR=$PWD /path/to/dollygrip/scripts/download_models.sh

# 3. Start ComfyUI and the v2 API proxy
python main.py --listen 127.0.0.1 --port 8188 &
pip install comfy-api-proxy && comfy-api-proxy &       # serves the v2 API on :8189

# 4. Start the app
cd /path/to/dollygrip && pip install -r app/requirements.txt
COMFY_BASE_URL=http://127.0.0.1:8189 uvicorn app.server:app --port 8000
```

Then open http://localhost:8000.

**One-shot on a fresh Linux GPU box:** `HF_TOKEN=hf_... bash scripts/setup.sh && bash scripts/start.sh` installs
ComfyUI and the nodes, downloads the weights, builds and validates the workflow against the live server, and serves
the app on :8000.

## Hosted on the Comfy platform

The hosted demo runs on [Comfy API](https://comfy.org/platform/). It's the same ComfyUI install packaged with
`comfy build init` / `comfy build push`, released and deployed with `comfy deploy up --min 0` (scale-to-zero). The
app is unchanged: point it at the deployment.

```bash
COMFY_BASE_URL=https://<deployment>.run.comfy.app COMFY_API_KEY=comfyui-... uvicorn app.server:app
```

**No GPU?** Run `DOLLYGRIP_BACKEND=mock uvicorn app.server:app` to get the full UI with guide-clip previews. The
"generation" step returns the guide clip instead of an LTX render.

## Models and licenses

| Component | Source | License |
|---|---|---|
| LTX-2.5 22B distilled (NVFP4), Gemma-4 12B encoder, video/audio VAEs, x2 latent upscaler | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (gated) | LTX-2.x Community License (free commercial use under $10M annual revenue) |
| Depth Anything V2 Small (guide mode `reproject` only) | [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | Apache-2.0 |
| Cameraman IC-LoRA v2 (trained on LTX-2.3; most 2.3 IC-LoRAs run on 2.5) | [Cseti/LTX2.3-22B_IC-LoRA-Cameraman_v2](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-Cameraman_v2) | Author states research / non-commercial use |
| DollyGrip code + node pack | this repo | Apache-2.0 |

## Status

Built in 42 hours for the challenge (deadline Fri Oct 30, 9am PT).

- [x] Trajectory rig, 30+ moves, DSL, 24 presets, share codes
- [x] Guide-clip renderer (deterministic, CPU)
- [x] ComfyUI node
- [x] Web app: preset gallery, live camera-path preview + top-down plan, DSL editor, share codes, SSE job progress, takes (verified end to end on the mock backend)
- [x] Workflow validated against live ComfyUI 0.43 (all 42 nodes and inputs; RTX 4090)
- [ ] LTX-2.5 + Cameraman spike: does the rendered guide transfer?
- [ ] Hosted demo on Comfy API (Builds + Deployments)

## License

Apache-2.0. Model weights keep their own licenses (see above).
