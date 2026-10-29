#!/usr/bin/env python3
"""Resolve a ComfyUI workflow's missing node packs and models, programmatically.

    python scripts/resolve_workflow.py WORKFLOW.json --comfy-dir /path/to/ComfyUI            # plan only
    python scripts/resolve_workflow.py WORKFLOW.json --comfy-dir /path/to/ComfyUI --apply    # install + download

What it uses, all from the workflow file itself:
  - node packs: each node's properties.cnr_id / ver (Comfy Registry) or aux_id (GitHub owner/repo)
  - models:     properties.models [{name, url, directory}] embedded by modern ComfyUI, then any
                model-looking widget value, looked up with `comfy build refs resolve` (HF + Civitai)
  - media:      LoadImage/LoadVideo inputs that aren't in ComfyUI/input (reported only)

Missing node *types* are checked against a live server's /object_info when --server answers.
Gated Hugging Face files use HF_TOKEN or ~/.cache/huggingface/token. Restart ComfyUI after
installing packs.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

MODEL_EXT = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft")
MEDIA_NODES = {"LoadImage": "image", "LoadVideo": "file", "VHS_LoadVideo": "video", "LoadAudio": "audio"}
# Fallback model folder by loader type, when the workflow doesn't say.
LOADER_DIRS = {
    "CheckpointLoaderSimple": "checkpoints", "LTXVAudioVAELoader": "checkpoints",
    "LoraLoader": "loras", "LoraLoaderModelOnly": "loras", "LTXICLoRALoaderModelOnly": "loras",
    "VAELoader": "vae", "VAELoaderKJ": "vae", "UNETLoader": "diffusion_models",
    "CLIPLoader": "text_encoders", "DualCLIPLoader": "text_encoders", "LTXAVTextEncoderLoader": "text_encoders",
    "LatentUpscaleModelLoader": "latent_upscale_models", "UpscaleModelLoader": "upscale_models",
}
REGISTRY = "https://api.comfy.org"


def all_nodes(wf: dict) -> list[dict]:
    if "nodes" not in wf:  # API format
        return [{"type": n["class_type"], "widgets_values": list(n.get("inputs", {}).values()), "properties": {}}
                for n in wf.values() if isinstance(n, dict) and "class_type" in n]
    nodes = list(wf["nodes"])
    for sg in wf.get("definitions", {}).get("subgraphs", []):
        nodes += sg.get("nodes", [])
    sub_ids = {sg["id"] for sg in wf.get("definitions", {}).get("subgraphs", [])}
    return [n for n in nodes if n.get("type") not in sub_ids]


def widget_strings(node: dict) -> list[str]:
    w = node.get("widgets_values")
    vals = w.values() if isinstance(w, dict) else (w or [])
    return [v for v in vals if isinstance(v, str)]


def http_json(url: str, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def hf_token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    p = Path.home() / ".cache" / "huggingface" / "token"
    return p.read_text().strip() if p.exists() else None


# ---------------------------------------------------------------- planning

def plan(wf: dict, comfy: Path, server: str | None) -> dict:
    nodes = all_nodes(wf)
    known = None
    if server:
        try:
            known = set(http_json(server.rstrip("/") + "/object_info", timeout=60))
        except Exception as e:
            print(f"(server {server} not reachable: {e}; checking packs by folder only)")
    installed_dirs = {p.name.lower() for p in (comfy / "custom_nodes").iterdir() if p.is_dir()}

    packs: dict[str, dict] = {}
    for n in nodes:
        props = n.get("properties", {}) or {}
        cnr, aux = props.get("cnr_id"), props.get("aux_id")
        if not cnr and not aux or cnr == "comfy-core":
            continue
        missing = (n["type"] not in known) if known is not None else (str(cnr or aux).split("/")[-1].lower() not in installed_dirs)
        if missing:
            key = cnr or aux
            packs.setdefault(key, {"cnr_id": cnr, "aux_id": aux, "version": props.get("ver"), "node_types": set()})
            packs[key]["node_types"].add(n["type"])
    unknown_types = sorted({n["type"] for n in nodes if known is not None and n["type"] not in known
                            and not (n.get("properties") or {}).get("cnr_id") and not (n.get("properties") or {}).get("aux_id")
                            and n["type"] not in ("Reroute", "Note", "MarkdownNote", "PrimitiveNode")})

    present = {p.name for p in (comfy / "models").rglob("*") if p.is_file() or p.is_symlink()}
    models: dict[str, dict] = {}
    for n in nodes:
        for m in (n.get("properties", {}) or {}).get("models", []) or []:
            models.setdefault(m["name"], {"name": m["name"], "url": m.get("url"), "directory": m.get("directory")})
        for v in widget_strings(n):
            name = Path(v).name
            if v.lower().endswith(MODEL_EXT):
                models.setdefault(name, {"name": name, "url": None, "directory": LOADER_DIRS.get(n["type"])})
                if not models[name]["directory"]:
                    models[name]["directory"] = LOADER_DIRS.get(n["type"])
    missing_models = [m for m in models.values() if m["name"] not in present]

    inputs = {p.name for p in (comfy / "input").rglob("*")} if (comfy / "input").exists() else set()
    media = sorted({v for n in nodes if n["type"] in MEDIA_NODES for v in widget_strings(n)[:1]
                    if v and Path(v).name not in inputs})
    return {"packs": packs, "unknown_types": unknown_types, "models": missing_models, "media": media}


def lookup_model(name: str, comfy_cli: str | None) -> dict | None:
    """Find a public source for a bare filename via `comfy build refs resolve`."""
    if not comfy_cli:
        return None
    try:
        out = subprocess.run([comfy_cli, "--skip-prompt", "build", "refs", "resolve", name],
                             capture_output=True, text=True, timeout=120).stdout
        cands = json.loads(out)["data"]["results"][0]["candidates"]
    except Exception:
        return None
    cands = sorted(cands, key=lambda c: (not c.get("verified"), -c.get("confidence", 0)))
    return cands[0] if cands else None


# ---------------------------------------------------------------- applying

def install_pack(p: dict, comfy: Path, python: str):
    dest_root = comfy / "custom_nodes"
    if p["cnr_id"]:
        q = f"?version={p['version']}" if p["version"] else ""
        info = http_json(f"{REGISTRY}/nodes/{p['cnr_id']}/install{q}")
        dest = dest_root / p["cnr_id"]
        print(f"  pack {p['cnr_id']} {info.get('version', p['version'])} <- {info['downloadUrl']}")
        with urllib.request.urlopen(info["downloadUrl"], timeout=300) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)
    else:
        dest = dest_root / p["aux_id"].split("/")[-1]
        print(f"  pack {p['aux_id']} <- git clone")
        subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{p['aux_id']}", str(dest)], check=True)
    req = dest / "requirements.txt"
    if req.exists():
        uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
        cmd = [uv, "pip", "install", "-q", "-p", python, "-r", str(req)] if Path(uv).exists() else \
              [python, "-m", "pip", "install", "-q", "-r", str(req)]
        subprocess.run(cmd, check=True)


def hf_sibling(url: str, name: str) -> str | None:
    """For a dead huggingface.co/<repo>/resolve/<rev>/<path> link, the closest-named file in that repo."""
    if "huggingface.co/" not in url or "/resolve/" not in url:
        return None
    repo, rest = url.split("huggingface.co/", 1)[1].split("/resolve/", 1)
    rev = rest.split("/", 1)[0]
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true")
    if tok := hf_token():
        req.add_header("Authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            files = [f["path"] for f in json.load(r) if f.get("type") == "file" and f["path"].endswith(MODEL_EXT)]
    except Exception:
        return None
    best = max(files, key=lambda f: _common_prefix(Path(f).name, name), default=None)
    if best and _common_prefix(Path(best).name, name) >= 16:
        return f"https://huggingface.co/{repo}/resolve/{rev}/{best}"
    return None


def download(url: str, dest: Path):
    if dest.exists():
        print("    already present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url)
    tok = hf_token()
    if tok and "huggingface.co" in url:
        req.add_header("Authorization", f"Bearer {tok}")
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(8 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e9:6.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    part.replace(dest)


LORA_LOADERS = {"LoraLoader", "LoraLoaderModelOnly", "LTXICLoRALoaderModelOnly"}


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def patch_workflow(wf: dict, unresolved: list[dict], available: dict[str, set[str]], media: list[str],
                   sample_image: str | None) -> list[str]:
    """Make the workflow runnable without files nobody can download.

    - a missing model with a close sibling in the same folder (e.g. another
      distilled LoRA of the same base) is swapped for it
    - an unresolvable LoRA is bypassed (its loader passes the model through)
    - missing media inputs are pointed at sample_image
    """
    notes = []
    swap = {}
    for m in unresolved:
        pool = available.get(m["directory"] or "", set())
        best = max(pool, key=lambda c: _common_prefix(c, m["name"]), default=None)
        if best and _common_prefix(best, m["name"]) >= 16:
            swap[m["name"]] = best
    gone = {m["name"] for m in unresolved} - set(swap)
    for n in all_nodes(wf):
        vals = n.get("widgets_values")
        if not isinstance(vals, list):
            continue
        for i, v in enumerate(vals):
            if not isinstance(v, str):
                continue
            name = Path(v).name
            if name in swap:
                vals[i] = swap[name]
                notes.append(f"{n['type']} #{n.get('id')}: {name} -> {swap[name]}")
            elif name in gone and n["type"] in LORA_LOADERS:
                n["mode"] = 4  # bypass
                notes.append(f"{n['type']} #{n.get('id')}: bypassed (no source for {name})")
            elif sample_image and n["type"] in MEDIA_NODES and i == 0 and v in media:
                vals[i] = sample_image
                notes.append(f"{n['type']} #{n.get('id')}: {v} -> {sample_image}")
    return notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workflow")
    ap.add_argument("--comfy-dir", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--python", default=sys.executable, help="python of the ComfyUI env (for pack requirements)")
    ap.add_argument("--comfy-cli", default=shutil.which("comfy"), help="comfy CLI for model lookups")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--sample-image", help="file in ComfyUI/input to use for missing media inputs")
    ap.add_argument("--install-to", help="also copy the resolved workflow here (e.g. ComfyUI/user/default/workflows)")
    args = ap.parse_args()

    comfy = Path(args.comfy_dir)
    wf = json.loads(Path(args.workflow).read_text())
    p = plan(wf, comfy, args.server)

    print(f"\nNode packs missing: {len(p['packs'])}")
    for k, v in p["packs"].items():
        print(f"  {k} {v['version'] or ''}  ({', '.join(sorted(v['node_types']))})")
    if p["unknown_types"]:
        print(f"  ! node types with no pack info: {', '.join(p['unknown_types'])}")

    print(f"\nModels missing: {len(p['models'])}")
    unresolved = []
    for m in p["models"]:
        if not m["url"]:
            c = lookup_model(m["name"], args.comfy_cli)
            if c:
                m["url"] = c["sourceUri"]
                m["directory"] = m["directory"] or c.get("type")
                m["via"] = f"lookup ({c.get('repo')}, verified={c.get('verified')})"
        src = m["url"] or "NOT FOUND (needs the workflow author's file)"
        print(f"  {m['directory'] or '?'}/{m['name']}  <- {src}  {m.get('via', '')}")
        if not m["url"] or not m["directory"]:
            unresolved.append(m["name"])

    if p["media"]:
        print(f"\nInputs to supply yourself: {', '.join(p['media'])}")

    if not args.apply:
        print("\n(plan only; re-run with --apply to install and download)")
        return
    print("\nApplying...")
    for v in p["packs"].values():
        install_pack(v, comfy, args.python)
    for m in p["models"]:
        if not (m["url"] and m["directory"]):
            continue
        print(f"  model {m['directory']}/{m['name']}")
        try:
            download(m["url"], comfy / "models" / m["directory"] / m["name"])
            continue
        except Exception as e:
            print(f"    {e}; looking for a replacement")
        alt = hf_sibling(m["url"], m["name"])
        if alt:
            print(f"    -> {alt.rsplit('/', 1)[-1]} (closest file in the same repo)")
            try:
                download(alt, comfy / "models" / m["directory"] / alt.rsplit("/", 1)[-1])
            except Exception as e:
                print(f"    {e}")
        unresolved.append(m["name"])  # patch step swaps in whatever sibling landed
    available: dict[str, set[str]] = {}
    for f in (comfy / "models").rglob("*"):
        if f.is_file() and f.suffix in MODEL_EXT:
            available.setdefault(f.relative_to(comfy / "models").parts[0], set()).add(f.name)
    notes = patch_workflow(wf, [m for m in p["models"] if m["name"] in unresolved], available,
                           p["media"], args.sample_image)
    if notes:
        out = Path(args.workflow).with_suffix(".resolved.json")
        out.write_text(json.dumps(wf, indent=1))
        print("\nPatched workflow -> " + str(out))
        for line in notes:
            print("  " + line)
        if args.install_to:
            dest = Path(args.install_to) / out.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(out, dest)
            print(f"  installed as {dest}")
    if p["packs"]:
        print("\nRestart ComfyUI to load the new node packs.")


if __name__ == "__main__":
    main()
