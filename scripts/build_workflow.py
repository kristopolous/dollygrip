#!/usr/bin/env python3
"""Build workflows/dollygrip_i2v.api.json and validate it against a live ComfyUI.

    python scripts/build_workflow.py --server http://127.0.0.1:8188           # build + validate
    python scripts/build_workflow.py --server http://127.0.0.1:8188 --show LTXAddVideoICLoRAGuide
    python scripts/build_workflow.py --templates /path/to/ComfyUI             # dump LTX-2.5 template loaders

Graph (mirrors Cseti's Cameraman v2 workflow, trimmed to core ComfyUI +
ComfyUI-LTXVideo + DollyGrip):

  stage 1 @ half res: IC-LoRA model, start image (strength 0.5) + guide clip (strength 1.0)
  stage 2 @ full res: x2 latent upscale, base model, start image (strength 0.7)
  decode video + audio -> CreateVideo -> SaveVideo

Inputs named in WIDGETS_POSITIONAL are given positionally (as in the UI
export) and named from /object_info, because their names are not documented.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workflows" / "dollygrip_i2v.api.json"

# LTX_VARIANT=nvfp4 for Blackwell (RTX 5090), int8 for Ada/Ampere (RTX 4090/3090).
VARIANT = os.environ.get("LTX_VARIANT", "int8")
MODELS = {
    "unet": {"nvfp4": "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
             "int8": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"}[VARIANT],
    "text_encoder": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "video_vae": "ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "ltx-2.5-audio-vae-bf16.safetensors",
    "upscaler": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "cameraman": "LTX2.3-22B_IC-LoRA-Cameraman_v2_14000.safetensors",
}

STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
STAGE2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
NEGATIVE = "text, subtitles, logo, watermark, signature, cut, scene change"


class G:
    def __init__(self):
        self.nodes = {}
        self.positional = {}

    def add(self, cls, title=None, positional=None, **inputs):
        nid = str(len(self.nodes) + 1)
        node = {"class_type": cls, "inputs": inputs}
        if title:
            node["_meta"] = {"title": title}
        self.nodes[nid] = node
        if positional is not None:
            self.positional[nid] = positional
        return nid


def out(nid, idx=0):
    return [nid, idx]


def build() -> G:
    g = G()
    cam = g.add("DollyGripCamera", "DG Camera", preset="dolly_in", duration_s=5.0, fps=24, strength=1.0,
                width=1920, height=1088, guide_width=960, guide_height=544, scene="set", prompt="",
                add_camera_words=True, seed=42, dsl="", share_code="")
    # DollyGripCamera outputs
    GUIDE, FRAMES, FPS_INT, FPS, W, H, W1, H1, PROMPT, SEED = range(10)

    img = g.add("LoadImage", "DG Start Image", image="example.png")
    g.nodes[cam]["inputs"]["start_image"] = out(img)

    unet = g.add("UNETLoader", unet_name=MODELS["unet"], weight_dtype="default")
    model_ic = g.add("LTXICLoRALoaderModelOnly", positional=[MODELS["cameraman"], 1.0], model=out(unet))
    clip = g.add("CLIPLoader", clip_name=MODELS["text_encoder"], type="ltxv", device="default")
    vae = g.add("VAELoader", vae_name=MODELS["video_vae"])
    avae = g.add("LTXVAudioVAELoader", ckpt_name=MODELS["audio_vae"])

    pos = g.add("CLIPTextEncode", "DG Prompt", text=out(cam, PROMPT), clip=out(clip))
    neg = g.add("CLIPTextEncode", "DG Negative", text=NEGATIVE, clip=out(clip))
    cond = g.add("LTXVConditioning", positive=out(pos), negative=out(neg), frame_rate=out(cam, FPS))

    # ---- stage 1 (half res, IC-LoRA guide)
    img1 = g.add("ImageScale", image=out(img), upscale_method="lanczos", width=out(cam, W1), height=out(cam, H1),
                 crop="center")
    img1p = g.add("LTXVPreprocess", image=out(img1), img_compression=18)
    lat = g.add("EmptyLTXVLatentVideo", width=out(cam, W1), height=out(cam, H1), length=out(cam, FRAMES),
                batch_size=1)
    lat = g.add("LTXVImgToVideoConditionOnly", positional=[0.5, False], vae=out(vae), image=out(img1p),
                latent=out(lat))
    guide = g.add("LTXAddVideoICLoRAGuide", positional=[0, 1.0, 1, "disabled", False, 256, 64],
                  positive=out(cond, 0), negative=out(cond, 1), vae=out(vae), latent=out(lat),
                  image=out(cam, GUIDE))
    alat = g.add("LTXVEmptyLatentAudio", audio_vae=out(avae), frames_number=out(cam, FRAMES),
                 frame_rate=out(cam, FPS_INT), batch_size=1)
    av = g.add("LTXVConcatAVLatent", video_latent=out(guide, 2), audio_latent=out(alat))
    guider = g.add("CFGGuider", model=out(model_ic), positive=out(guide, 0), negative=out(guide, 1), cfg=1.0)
    s1 = g.add("SamplerCustomAdvanced",
               noise=out(g.add("RandomNoise", "DG Seed", noise_seed=out(cam, SEED))),
               guider=out(guider),
               sampler=out(g.add("KSamplerSelect", sampler_name="euler_ancestral_cfg_pp")),
               sigmas=out(g.add("ManualSigmas", sigmas=STAGE1_SIGMAS)),
               latent_image=out(av))
    sep = g.add("LTXVSeparateAVLatent", av_latent=out(s1, 0))
    crop = g.add("LTXVCropGuides", positive=out(guide, 0), negative=out(guide, 1), latent=out(sep, 0))

    # ---- stage 2 (x2 upscale, base model)
    upm = g.add("LatentUpscaleModelLoader", model_name=MODELS["upscaler"])
    up = g.add("LTXVLatentUpsampler", samples=out(crop, 2), upscale_model=out(upm), vae=out(vae))
    img2 = g.add("ImageScale", image=out(img), upscale_method="lanczos", width=out(cam, W), height=out(cam, H),
                 crop="center")
    img2p = g.add("LTXVPreprocess", image=out(img2), img_compression=18)
    up = g.add("LTXVImgToVideoConditionOnly", positional=[0.7, False], vae=out(vae), image=out(img2p),
               latent=out(up))
    av2 = g.add("LTXVConcatAVLatent", video_latent=out(up), audio_latent=out(sep, 1))
    guider2 = g.add("CFGGuider", model=out(unet), positive=out(crop, 0), negative=out(crop, 1), cfg=1.0)
    s2 = g.add("SamplerCustomAdvanced",
               noise=out(g.add("RandomNoise", "DG Seed 2", noise_seed=out(cam, SEED))),
               guider=out(guider2),
               sampler=out(g.add("KSamplerSelect", sampler_name="euler_cfg_pp")),
               sigmas=out(g.add("ManualSigmas", sigmas=STAGE2_SIGMAS)),
               latent_image=out(av2))
    sep2 = g.add("LTXVSeparateAVLatent", av_latent=out(s2, 0))
    frames = g.add("VAEDecode", samples=out(sep2, 0), vae=out(vae))
    audio = g.add("LTXVAudioVAEDecode", samples=out(sep2, 1), audio_vae=out(avae))
    video = g.add("CreateVideo", images=out(frames), fps=out(cam, FPS), audio=out(audio))
    g.add("SaveVideo", "DG Output", video=out(video), filename_prefix="dollygrip/shot", format="mp4", codec="h264")
    # Guide clip saved alongside, for side-by-side demos.
    gv = g.add("CreateVideo", images=out(cam, GUIDE), fps=out(cam, FPS))
    g.add("SaveVideo", "DG Guide Output", video=out(gv), filename_prefix="dollygrip/guide", format="mp4",
          codec="h264")
    return g


# ---------------------------------------------------------------- object_info

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def fetch(server, path):
    with urllib.request.urlopen(server.rstrip("/") + path, timeout=60) as r:
        return json.load(r)


def input_specs(info):
    req = info["input"].get("required", {})
    opt = info["input"].get("optional", {})
    return list(req.items()) + list(opt.items()), set(req)


def is_widget(spec):
    t = spec[0]
    return isinstance(t, list) or t in WIDGET_TYPES


def resolve(g: G, object_info: dict) -> list[str]:
    problems = []
    for nid, node in g.nodes.items():
        cls = node["class_type"]
        if cls not in object_info:
            problems.append(f"[{nid}] missing node class {cls}")
            continue
        specs, required = input_specs(object_info[cls])
        names = [n for n, _ in specs]
        if nid in g.positional:
            vals = list(g.positional[nid])
            for name, spec in specs:
                if not is_widget(spec) or not vals:
                    continue
                v = vals.pop(0)
                opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                if opts.get("control_after_generate") and vals:
                    vals.pop(0)
                node["inputs"].setdefault(name, v)
            if vals:
                problems.append(f"[{nid}] {cls}: {len(vals)} positional widget values left over: {vals}")
        for name in node["inputs"]:
            if name not in names:
                problems.append(f"[{nid}] {cls}: unknown input {name!r} (has: {', '.join(names)})")
        for name in required:
            if name not in node["inputs"]:
                problems.append(f"[{nid}] {cls}: missing required input {name!r}")
        for name, spec in specs:
            v = node["inputs"].get(name)
            if isinstance(spec[0], list) and isinstance(v, str) and v not in spec[0]:
                problems.append(f"[{nid}] {cls}.{name} = {v!r} not in options ({len(spec[0])} available)")
    return problems


def show(object_info, cls):
    specs, required = input_specs(object_info[cls])
    print(f"{cls}  ->  {object_info[cls].get('output')}  {object_info[cls].get('output_name')}")
    for name, spec in specs:
        t = spec[0] if not isinstance(spec[0], list) else f"COMBO[{len(spec[0])}] {spec[0][:6]}"
        print(f"   {'*' if name in required else ' '} {name}: {t} {spec[1] if len(spec) > 1 else ''}")


def dump_templates(comfy_dir):
    hits = []
    roots = [Path(comfy_dir)]
    try:
        import comfyui_workflow_templates as t  # noqa
        roots.append(Path(t.__file__).parent)
    except Exception:
        pass
    for root in roots:
        for p in root.rglob("*.json"):
            if "ltx" in p.name.lower() and ("2_5" in p.name or "2.5" in p.name or "25" in p.name):
                hits.append(p)
    for p in hits:
        print("==", p)
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print("  ", e)
            continue
        for n in d.get("nodes", []):
            if "Loader" in n.get("type", "") or "Load" in n.get("type", ""):
                print("  ", n["type"], json.dumps(n.get("widgets_values"))[:160])
    if not hits:
        print("no LTX-2.5 templates found under", roots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--show", nargs="*")
    ap.add_argument("--templates")
    args = ap.parse_args()

    if args.templates:
        dump_templates(args.templates)
        return
    object_info = fetch(args.server, "/object_info")
    if args.show:
        for cls in args.show:
            show(object_info, cls)
        return
    g = build()
    problems = resolve(g, object_info)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(g.nodes, indent=2) + "\n")
    print(f"wrote {OUT} ({len(g.nodes)} nodes)")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  ", p)
        sys.exit(1)
    print("validated against", args.server)


if __name__ == "__main__":
    main()
