"""DollyGrip web app: preset gallery, camera preview, and generation jobs.

    COMFY_BASE_URL=http://127.0.0.1:8189 uvicorn app.server:app --port 8000   # real (comfy-api-proxy)
    DOLLYGRIP_BACKEND=mock uvicorn app.server:app --port 8000                 # no GPU: guide clip as result
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)  # .env beats the shell; the API key never leaves the server
sys.path.insert(0, str(ROOT / "node_pack" / "ComfyUI-DollyGrip"))

from dollygrip import presets, render, resolve_shot, sharecode, snap_frames, trajectory  # noqa: E402

OUTPUTS = Path(os.environ.get("DOLLYGRIP_OUTPUTS", ROOT / "outputs"))
WORKFLOW = Path(os.environ.get("DOLLYGRIP_WORKFLOW", ROOT / "workflows" / "dollygrip_i2v.api.json"))
BACKEND = os.environ.get("DOLLYGRIP_BACKEND", "comfy")
MAX_CONCURRENT = int(os.environ.get("DOLLYGRIP_MAX_JOBS", "1"))
for d in ("previews", "uploads", "jobs"):
    (OUTPUTS / d).mkdir(parents=True, exist_ok=True)

FPS = 24
PREVIEW_SIZE = (384, 224)
SIZES = {  # aspect -> {resolution: (w, h)}, guide (w, h); all divisible by 32
    "16:9": {"720p": (1280, 704), "1080p": (1920, 1088), "guide": (960, 544)},
    "9:16": {"720p": (704, 1280), "1080p": (1088, 1920), "guide": (544, 960)},
    "1:1": {"720p": (960, 960), "1080p": (1440, 1440), "guide": (768, 768)},
}

app = FastAPI(title="DollyGrip")


# ---------------------------------------------------------------- helpers

def encode_mp4(frames: np.ndarray, path: Path, fps: float):
    n, h, w, _ = frames.shape
    tmp = path.with_suffix(".tmp.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", str(tmp)],
        input=frames.tobytes(), check=True)
    tmp.replace(path)


def load_card(path: Path | None, max_side=512) -> np.ndarray | None:
    if not path:
        return None
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    return np.asarray(im, dtype=np.float32) / 255.0


def path_summary(dsl: str, n: int, strength: float) -> list[list[float]]:
    """Top-down camera path for the UI: [x, z, heading_x, heading_z, fov] per sampled frame."""
    cams = trajectory.build(dsl, n, FPS, strength).cameras
    step = max(1, n // 48)
    out = []
    for c in cams[::step] + [cams[-1]]:
        fwd = -c.rotation[:, 2]
        out.append([round(float(c.position[0]), 4), round(float(c.position[2]), 4),
                    round(float(fwd[0]), 4), round(float(fwd[2]), 4), round(c.fov_y, 2)])
    return out


class Shot(BaseModel):
    preset: str = "dolly_in"
    dsl: str = ""
    share_code: str = ""
    duration_s: float = 5.0
    strength: float = 1.0


def resolve(shot: Shot):
    try:
        dsl, hint, blur = resolve_shot(shot.preset, shot.dsl, shot.share_code)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e).strip('"'))
    duration = shot.duration_s
    if shot.share_code.strip():
        d = sharecode.decode(shot.share_code)
        shot.strength = d["strength"]
        duration = d["duration_s"] or duration
    duration = float(np.clip(duration, 1.0, 20.0))
    return dsl, hint, blur, duration


_preview_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def make_preview(dsl: str, blur: int, duration: float, strength: float) -> Path:
    n = snap_frames(duration, FPS)
    key = hashlib.sha1(json.dumps([dsl, blur, n, strength, PREVIEW_SIZE]).encode()).hexdigest()[:16]
    path = OUTPUTS / "previews" / f"{key}.mp4"
    with _locks_guard:
        lock = _preview_locks.setdefault(key, threading.Lock())
    with lock:
        if not path.exists():
            frames = render.render(dsl, n, FPS, strength, *PREVIEW_SIZE, motion_blur=blur)
            encode_mp4(frames, path, FPS)
    return path


# ---------------------------------------------------------------- API: presets & preview

@app.get("/api/presets")
def list_presets():
    return {"presets": [p.to_json() for p in presets.PRESETS],
            "moves": sorted(trajectory.MOVES), "easings": list(trajectory.EASINGS), "backend": BACKEND}


@app.post("/api/preview")
def preview(shot: Shot):
    dsl, hint, blur, duration = resolve(shot)
    path = make_preview(dsl, blur, duration, shot.strength)
    n = snap_frames(duration, FPS)
    return {
        "dsl": trajectory.format_dsl(trajectory.parse(dsl)),
        "prompt_hint": hint,
        "frames": n,
        "duration_s": duration,
        "strength": shot.strength,
        "share_code": sharecode.encode(dsl, shot.strength, duration),
        "video_url": f"/outputs/previews/{path.name}",
        "path": path_summary(dsl, n, shot.strength),
    }


@app.get("/api/sharecode/{code}")
def decode_code(code: str):
    try:
        return sharecode.decode(code)
    except Exception as e:
        raise HTTPException(400, f"bad share code: {e}")


# ---------------------------------------------------------------- jobs

@dataclass
class Job:
    id: str
    params: dict
    status: str = "queued"  # queued | running | succeeded | failed
    progress: float = 0.0
    message: str = "Queued"
    video_url: str | None = None
    guide_url: str | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    version: int = 0

    def update(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.version += 1

    def public(self):
        d = asdict(self)
        d.pop("version")
        return d


JOBS: dict[str, Job] = {}
_slots = threading.Semaphore(MAX_CONCURRENT)


def run_mock(job: Job, p: dict):
    steps = 20
    for i in range(steps):
        time.sleep(0.15)
        job.update(status="running", progress=(i + 1) / steps * 0.7, message="Sampling (mock)")
    job.update(message="Rendering guide as stand-in output", progress=0.8)
    w, h = SIZES[p["aspect"]]["guide"]
    card = load_card(Path(p["image_path"])) if p["image_path"] else None
    frames = render.render(p["dsl"], p["frames"], FPS, p["strength"], w, h, card=card, motion_blur=p["blur"])
    out = OUTPUTS / "jobs" / f"{job.id}.mp4"
    encode_mp4(frames, out, FPS)
    job.update(video_url=f"/outputs/jobs/{out.name}", guide_url=f"/outputs/jobs/{out.name}")


def run_comfy(job: Job, p: dict):
    from comfy_sdk import Comfy, ComfyError, JobFailed, Progress, QueueFull, StatusChange

    client = Comfy(client_info="dollygrip")
    wf = client.workflows.from_file(str(WORKFLOW))
    titles = {n.get("_meta", {}).get("title"): nid for nid, n in wf.json.items()}

    def node(title):
        if title not in titles:
            raise RuntimeError(f"workflow has no node titled {title!r}")
        return titles[title]

    cam = node("DG Camera")
    w, h = SIZES[p["aspect"]][p["resolution"]]
    gw = gh = 0  # node matches the guide to the stage-1 latent size
    for k, v in {
        "preset": p["preset"] if p["preset"] in presets.PRESET_IDS else "dolly_in",
        "dsl": p["dsl"] if p["custom"] else "", "share_code": "", "duration_s": p["duration_s"], "fps": FPS,
        "strength": p["strength"], "width": w, "height": h, "guide_width": gw, "guide_height": gh,
        "scene": p["scene"], "prompt": p["prompt"], "add_camera_words": True, "seed": p["seed"],
    }.items():
        wf.set_input(cam, k, v)
    wf.set_input(node("DG Start Image"), "image", client.assets.from_file(p["image_path"]))

    job.update(status="running", message="Submitting", progress=0.02)
    try:
        cj = client.submit(wf)
        for ev in cj.events():
            if isinstance(ev, Progress):
                job.update(progress=0.05 + 0.9 * float(ev.value or 0), message=ev.message or "Sampling")
            elif isinstance(ev, StatusChange):
                job.update(message=str(ev.status).capitalize())
                if ev.status in ("succeeded", "failed", "cancelled"):
                    break
        cj.result()
    except JobFailed as e:
        raise RuntimeError(f"ComfyUI job failed: {getattr(e, 'error', e)}")
    except QueueFull as e:
        raise RuntimeError(f"GPU queue is full, retry in {getattr(e, 'retry_after', '?')}s")
    except ComfyError as e:
        raise RuntimeError(f"ComfyUI error ({getattr(e, 'code', '?')}): {e}")

    for title, key in (("DG Output", "video_url"), ("DG Guide Output", "guide_url")):
        if title not in titles:
            continue
        outs = [o for o in cj.get_outputs(titles[title]) if o.type == "video"] or cj.get_outputs(titles[title])
        if outs:
            dest = OUTPUTS / "jobs" / f"{job.id}{'' if key == 'video_url' else '.guide'}.mp4"
            outs[0].to_file(str(dest))
            job.update(**{key: f"/outputs/jobs/{dest.name}"})
    if not job.video_url:
        raise RuntimeError("job finished without a video output")


def worker(job: Job):
    with _slots:
        try:
            (run_mock if BACKEND == "mock" else run_comfy)(job, job.params)
            job.update(status="succeeded", progress=1.0, message="Done")
        except Exception as e:  # surfaced to the UI as a typed message
            job.update(status="failed", error=f"{type(e).__name__}: {e}", message="Failed")
        (OUTPUTS / "jobs" / f"{job.id}.json").write_text(json.dumps(job.public(), indent=2))


@app.post("/api/jobs")
async def create_job(
    image: UploadFile | None = File(None),
    image_id: str = Form(""),
    prompt: str = Form(""),
    preset: str = Form("dolly_in"),
    dsl: str = Form(""),
    share_code: str = Form(""),
    duration_s: float = Form(5.0),
    strength: float = Form(1.0),
    resolution: str = Form("1080p"),
    aspect: str = Form("16:9"),
    scene: str = Form("set"),
    seed: int = Form(42),
):
    if aspect not in SIZES or resolution not in ("720p", "1080p"):
        raise HTTPException(400, "bad aspect/resolution")
    if scene not in ("set", "start_image_card", "reproject"):
        raise HTTPException(400, "bad scene")
    shot = Shot(preset=preset, dsl=dsl, share_code=share_code, duration_s=duration_s, strength=strength)
    shot_dsl, _, blur, duration = resolve(shot)

    image_path = ""
    if image is not None and image.filename:
        data = await image.read()
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(413, "image too large (20MB max)")
        try:
            im = Image.open(__import__("io").BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(400, "could not read image")
        im.thumbnail((1920, 1920))
        image_id = hashlib.sha1(data).hexdigest()[:16]
        image_path = str(OUTPUTS / "uploads" / f"{image_id}.png")
        im.save(image_path)
    elif image_id:
        if not image_id.isalnum():
            raise HTTPException(400, "bad image id")
        image_path = str(OUTPUTS / "uploads" / f"{image_id}.png")
        if not Path(image_path).exists():
            raise HTTPException(404, "image not found; upload again")
    if not image_path and BACKEND != "mock":
        raise HTTPException(400, "upload a start image")

    active = sum(j.status in ("queued", "running") for j in JOBS.values())
    if active >= 4:
        raise HTTPException(429, "queue is full, try again in a minute")

    custom = bool(dsl.strip() or share_code.strip())
    params = dict(prompt=prompt.strip()[:2000], preset="" if custom else presets.get(preset).id, custom=custom,
                  dsl=shot_dsl, duration_s=duration, frames=snap_frames(duration, FPS),
                  strength=shot.strength, resolution=resolution, aspect=aspect, scene=scene, seed=seed,
                  image_path=image_path, image_id=image_id, blur=blur,
                  share_code=sharecode.encode(shot_dsl, shot.strength, duration))
    job = Job(id=uuid.uuid4().hex[:12], params=params)
    JOBS[job.id] = job
    threading.Thread(target=worker, args=(job,), daemon=True).start()
    return job.public()


@app.get("/api/jobs")
def list_jobs():
    jobs = sorted(JOBS.values(), key=lambda j: -j.created)
    return {"jobs": [j.public() for j in jobs[:50]]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "no such job")
    return JOBS[job_id].public()


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "no such job")
    job = JOBS[job_id]

    async def stream():
        seen = -1
        while True:
            if job.version != seen:
                seen = job.version
                yield f"data: {json.dumps(job.public())}\n\n"
                if job.status in ("succeeded", "failed"):
                    return
            await asyncio.sleep(0.25)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/health")
def health():
    return {"ok": True, "backend": BACKEND, "workflow": WORKFLOW.exists(), "ffmpeg": bool(shutil.which("ffmpeg"))}


# ---------------------------------------------------------------- static

app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.on_event("startup")
def warm_previews():
    def warm():
        for p in presets.PRESETS:
            if p.mvp:
                try:
                    make_preview(p.dsl, p.motion_blur, 5.0, 1.0)
                except Exception as e:  # never block startup on a preview
                    print("preview warm failed", p.id, e)
    threading.Thread(target=warm, daemon=True).start()
