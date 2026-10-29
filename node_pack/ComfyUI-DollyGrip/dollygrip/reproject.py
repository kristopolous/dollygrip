"""Guide clip from the start image itself: depth -> 3D points -> re-rendered along the path.

The synthetic set works for moves along the view axis, but on arcs/orbits the
Cameraman IC-LoRA copies the guide's *content*. Re-rendering the user's own
image along the trajectory removes that conflict: whatever the model copies is
the right scene, with real parallax from monocular depth. Disoccluded areas
are filled from coarser pyramid levels; the video model repaints them.

Depth: Depth Anything V2 Small (Apache-2.0), loaded through transformers.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .trajectory import SUBJECT_DISTANCE, Camera, build_at

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
_depth = {}


def _device():
    try:
        import comfy.model_management as mm  # inside ComfyUI
        return mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def estimate_depth(img: torch.Tensor) -> torch.Tensor:
    """img (H,W,3) float [0,1] on device -> metric-ish depth (H,W), subject ~ SUBJECT_DISTANCE."""
    from transformers import AutoModelForDepthEstimation

    dev = img.device
    if "model" not in _depth:
        _depth["model"] = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).eval()
    model = _depth["model"].to(dev)
    h, w = img.shape[:2]
    # DA-V2 wants multiples of 14; ~518 on the short side.
    s = 518 / min(h, w)
    th, tw = max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14)
    x = img.permute(2, 0, 1)[None]
    x = F.interpolate(x, (th, tw), mode="bicubic", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    with torch.no_grad():
        disp = model(pixel_values=((x - mean) / std)).predicted_depth  # (1,th,tw) relative inverse depth
    model.to("cpu")
    disp = F.interpolate(disp[:, None], (h, w), mode="bilinear", align_corners=False)[0, 0].float()
    dn = (disp - disp.min()) / (disp.max() - disp.min()).clamp_min(1e-6)  # 0 far .. 1 near
    # Bounded inverse mapping anchored on the foreground (80th-percentile nearness),
    # which is where the subject usually is: it lands at the rig's subject distance,
    # so orbits/arcs pivot around it and pushes stop short of it.
    sample = dn[:: max(1, h // 256), :: max(1, w // 256)].flatten()
    dn_fg = torch.quantile(sample, 0.8)
    k = 0.12
    depth = SUBJECT_DISTANCE * (dn_fg + k) / (dn + k)
    return depth.clamp(0.5 * SUBJECT_DISTANCE, 40.0)


def _fill_holes(color: torch.Tensor, mask: torch.Tensor, levels: int = 7) -> torch.Tensor:
    """Push-pull fill. color (3,H,W), mask (1,H,W) in {0,1}."""
    pyr = [(color * mask, mask)]
    c, m = pyr[0]
    for _ in range(levels):
        c = F.avg_pool2d(c[None], 2, ceil_mode=True)[0]
        m = F.avg_pool2d(m[None], 2, ceil_mode=True)[0]
        pyr.append((c, m))
    c, m = pyr[-1]
    filled = c / m.clamp_min(1e-6)
    for c, m in reversed(pyr[:-1]):
        up = F.interpolate(filled[None], size=c.shape[-2:], mode="bilinear", align_corners=False)[0]
        mine = c / m.clamp_min(1e-6)
        filled = torch.where(m > 0.999, mine, mine * m + up * (1 - m))
    return filled


def _center_crop_resize(img: torch.Tensor, w: int, h: int) -> torch.Tensor:
    ih, iw = img.shape[:2]
    target = w / h
    if iw / ih > target:
        nw = round(ih * target)
        x0 = (iw - nw) // 2
        img = img[:, x0:x0 + nw]
    else:
        nh = round(iw / target)
        y0 = (ih - nh) // 2
        img = img[y0:y0 + nh]
    return F.interpolate(img.permute(2, 0, 1)[None], (h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)


def _splat(points, colors, cam: Camera, w: int, h: int):
    dev = points.device
    R = torch.as_tensor(cam.rotation, dtype=torch.float32, device=dev)
    pos = torch.as_tensor(cam.position, dtype=torch.float32, device=dev)
    p = (points - pos) @ R  # world -> camera (R is world-from-camera, orthonormal)
    z = -p[:, 2]
    f = (h / 2) / math.tan(math.radians(cam.fov_y) / 2)
    ok = z > 0.05
    u = torch.round(f * p[:, 0] / z.clamp_min(1e-6) + w / 2 - 0.5).long()
    v = torch.round(-f * p[:, 1] / z.clamp_min(1e-6) + h / 2 - 0.5).long()
    ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z, col = u[ok], v[ok], z[ok], colors[ok]
    idx = v * w + u
    zbuf = torch.full((h * w,), float("inf"), device=dev)
    zbuf.scatter_reduce_(0, idx, z, reduce="amin")
    front = z <= zbuf[idx] * 1.002
    out = torch.zeros((h * w, 3), device=dev)
    out[idx[front]] = col[front]
    mask = torch.zeros(h * w, device=dev)
    mask[idx[front]] = 1.0
    out = out.T.reshape(3, h, w)
    mask = mask.reshape(1, h, w)
    return _fill_holes(out, mask)


def render(dsl: str, n_frames: int, fps: float, strength: float, width: int, height: int,
           image: np.ndarray, motion_blur: int = 1, supersample: float = 1.5) -> np.ndarray:
    """image: float RGB [0,1] (H,W,3). Returns uint8 (n_frames, height, width, 3)."""
    dev = _device()
    img = torch.as_tensor(np.ascontiguousarray(image[..., :3]), dtype=torch.float32, device=dev)
    sw, sh = round(width * supersample), round(height * supersample)
    src = _center_crop_resize(img, sw, sh)
    depth = estimate_depth(src)

    k = max(int(motion_blur), 1)
    offsets = (np.arange(k) + 0.5) / k - 0.5 if k > 1 else np.zeros(1)
    times = np.clip((np.arange(n_frames)[:, None] + offsets[None]).ravel(), 0, n_frames - 1)
    cams = build_at(dsl, n_frames, fps, strength, np.concatenate([[0.0], times]))
    cam0, cams = cams[0], cams[1:]

    # Unproject with the frame-0 camera, so frame 0 reproduces the start image.
    f0 = (sh / 2) / math.tan(math.radians(cam0.fov_y) / 2)
    vs, us = torch.meshgrid(torch.arange(sh, device=dev), torch.arange(sw, device=dev), indexing="ij")
    x = (us.float() + 0.5 - sw / 2) / f0 * depth
    y = -(vs.float() + 0.5 - sh / 2) / f0 * depth
    pc = torch.stack([x, y, -depth], -1).reshape(-1, 3)
    R0 = torch.as_tensor(cam0.rotation, dtype=torch.float32, device=dev)
    P0 = torch.as_tensor(cam0.position, dtype=torch.float32, device=dev)
    points = pc @ R0.T + P0
    colors = src.reshape(-1, 3)

    out = np.empty((n_frames, height, width, 3), dtype=np.uint8)
    for i in range(n_frames):
        acc = sum(_splat(points, colors, c, width, height) for c in cams[i * k:(i + 1) * k]) / k
        out[i] = (acc.clamp(0, 1).permute(1, 2, 0) * 255 + 0.5).byte().cpu().numpy()
    return out
