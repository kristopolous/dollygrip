"""Render a camera trajectory as a reference ("guide") video.

The Cameraman IC-LoRA copies camera motion from a reference clip. We feed it
a clip we render ourselves: a synthetic set with strong parallax cues
(checker floor, striped walls with a distinct hue per side, props at many
depths), optionally with the user's start image standing in as the subject.
Pure numpy, CPU, deterministic.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .trajectory import Camera, build_at

FLOOR = -1.0
ROOM_LO = np.array([-14.0, FLOOR, -14.0])
ROOM_HI = np.array([14.0, 7.0, 14.0])
# The camera lives in the annulus between these radii; props stay out of it.
CAMERA_RING = (2.9, 6.6)
FOG = np.array([0.80, 0.82, 0.86])
LIGHT = np.array([0.45, 0.8, 0.35]) / np.linalg.norm([0.45, 0.8, 0.35])

# Wall hues: +x, -x, +z, -z. Distinct so a pan/orbit always reads.
WALL_COLORS = np.array([
    [0.80, 0.42, 0.30],
    [0.30, 0.55, 0.78],
    [0.85, 0.75, 0.38],
    [0.40, 0.70, 0.45],
])


def _props():
    rng = np.random.default_rng(7)
    boxes = []

    def add(r, ang, size, h):
        cx, cz = r * math.cos(ang), r * math.sin(ang)
        hue = rng.uniform(0, 1)
        col = 0.5 + 0.4 * np.cos(2 * math.pi * (hue + np.array([0, 1 / 3, 2 / 3])))
        boxes.append(([cx - size, FLOOR, cz - size], [cx + size, FLOOR + h, cz + size], col))

    # Near ring: short markers just around the subject (inside the camera ring).
    for i in range(6):
        ang = (i + 0.5) / 6 * 2 * math.pi
        if abs(math.sin(ang)) > 0.8 and math.sin(ang) > 0:  # not between camera and subject at start
            continue
        add(rng.uniform(1.6, 2.4), ang + rng.uniform(-0.2, 0.2), rng.uniform(0.12, 0.22), rng.uniform(0.3, 1.0))
    # Far ring: tall pillars beyond the camera ring for parallax.
    for i in range(16):
        ang = i / 16 * 2 * math.pi + rng.uniform(-0.12, 0.12)
        add(rng.uniform(CAMERA_RING[1] + 0.8, 12.0), ang, rng.uniform(0.3, 0.8), rng.uniform(1.0, 5.0))
    lo = np.array([b[0] for b in boxes])
    hi = np.array([b[1] for b in boxes])
    col = np.array([b[2] for b in boxes])
    return lo, hi, col


PROP_LO, PROP_HI, PROP_COL = (a.astype(np.float32) for a in _props())
SPHERE_C = np.array([0.0, 0.15, 0.0])
SPHERE_R = 0.5
PEDESTAL_LO = np.array([[-0.32, FLOOR, -0.32]])
PEDESTAL_HI = np.array([[0.32, -0.35, 0.32]])


def _rays(cam: Camera, w: int, h: int):
    t = math.tan(math.radians(cam.fov_y) / 2)
    xs = ((np.arange(w) + 0.5) / w * 2 - 1) * t * (w / h)
    ys = (1 - (np.arange(h) + 0.5) / h * 2) * t
    X, Y = np.meshgrid(xs, ys)
    d = np.stack([X, Y, -np.ones_like(X)], -1).reshape(-1, 3)
    d = d @ cam.rotation.T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return cam.position.astype(np.float32), d.astype(np.float32)


def _boxes(o, d, lo, hi):
    """Nearest hit of rays against K boxes from outside. Returns t (N,), idx (N,), normal (N,3)."""
    n = len(d)
    inv = [1.0 / np.where(np.abs(d[:, a]) < 1e-9, 1e-9, d[:, a]) for a in range(3)]
    t = np.full(n, np.inf, dtype=d.dtype)
    idx = np.zeros(n, dtype=np.int32)
    axis = np.zeros(n, dtype=np.int8)
    for k in range(len(lo)):
        tnear = np.full(n, -np.inf, dtype=d.dtype)
        tfar = np.full(n, np.inf, dtype=d.dtype)
        ax = np.zeros(n, dtype=np.int8)
        for a in range(3):
            t0 = (lo[k, a] - o[a]) * inv[a]
            t1 = (hi[k, a] - o[a]) * inv[a]
            tmn = np.minimum(t0, t1)
            ax = np.where(tmn > tnear, np.int8(a), ax)
            tnear = np.maximum(tnear, tmn)
            tfar = np.minimum(tfar, np.maximum(t0, t1))
        better = (tfar >= tnear) & (tnear > 1e-4) & (tnear < t)
        t = np.where(better, tnear, t)
        idx = np.where(better, k, idx)
        axis = np.where(better, ax, axis)
    normal = np.zeros((n, 3), dtype=d.dtype)
    rows = np.arange(n)
    normal[rows, axis] = -np.sign(d[rows, axis])
    return t, idx, normal


def _edge_lines(p, lo, hi, normal, width=0.025):
    """1 near box edges on the hit face, else 0 (draws crisp outlines)."""
    rel = np.minimum(np.abs(p - lo), np.abs(p - hi))
    rel = np.where(np.abs(normal) > 0.5, np.inf, rel)  # ignore the face's own axis
    return (rel.min(-1) < width).astype(np.float64)


def _shade_lambert(col, normal, amount=0.65):
    return col * (0.35 + amount * np.clip(normal @ LIGHT, 0, 1))[:, None]


def render_frame(cam: Camera, w: int, h: int, card: np.ndarray | None = None) -> np.ndarray:
    o, d = _rays(cam, w, h)
    n = len(d)
    color = np.zeros((n, 3))

    # Room interior: exit distance of the room box.
    inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
    tf = np.maximum((ROOM_LO - o) * inv, (ROOM_HI - o) * inv)
    axis = tf.argmin(1)
    t_best = tf.min(1)
    kind = np.zeros(n, dtype=np.int8)  # 0 room, 1 prop, 2 sphere, 3 pedestal, 4 card

    tp, pidx, pn = _boxes(o, d, PROP_LO, PROP_HI)
    m = tp < t_best
    t_best = np.where(m, tp, t_best)
    kind[m] = 1

    if card is None:
        oc = o - SPHERE_C
        b = d @ oc
        c = oc @ oc - SPHERE_R ** 2
        disc = b * b - c
        ts = np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
        ts = np.where(ts > 1e-4, ts, np.inf)
        m = ts < t_best
        t_best = np.where(m, ts, t_best)
        kind[m] = 2
        tq, _, qn = _boxes(o, d, PEDESTAL_LO, PEDESTAL_HI)
        m = tq < t_best
        t_best = np.where(m, tq, t_best)
        kind[m] = 3
    else:
        ch, cw = card.shape[:2]
        card_h = 2.6
        card_w = card_h * cw / ch
        tc = np.where(np.abs(d[:, 2]) > 1e-9, -o[2] / np.where(np.abs(d[:, 2]) > 1e-9, d[:, 2], 1), np.inf)
        pc = o + tc[:, None] * d
        inside = (tc > 1e-4) & (np.abs(pc[:, 0]) <= card_w / 2) & (pc[:, 1] >= FLOOR) & (pc[:, 1] <= FLOOR + card_h)
        tc = np.where(inside, tc, np.inf)
        m = tc < t_best
        t_best = np.where(m, tc, t_best)
        kind[m] = 4

    p = o + t_best[:, None] * d

    # Room surfaces.
    m = kind == 0
    if m.any():
        pm, ax, dm = p[m], axis[m], d[m]
        c = np.zeros((m.sum(), 3))
        floor = (ax == 1) & (dm[:, 1] < 0)
        ceil = (ax == 1) & (dm[:, 1] > 0)
        wall = ~(floor | ceil)
        # Floor: 1m checker with grid lines.
        fx, fz = pm[floor, 0], pm[floor, 2]
        chk = (np.floor(fx) + np.floor(fz)) % 2
        line = (np.minimum(fx % 1, 1 - fx % 1) < 0.02) | (np.minimum(fz % 1, 1 - fz % 1) < 0.02)
        c[floor] = np.where(chk[:, None] > 0, [0.82, 0.80, 0.76], [0.42, 0.40, 0.38])
        c[floor] *= np.where(line, 0.5, 1.0)[:, None]
        # Ceiling.
        cx, cz = pm[ceil, 0], pm[ceil, 2]
        cl = (np.minimum(cx % 2, 2 - cx % 2) < 0.04) | (np.minimum(cz % 2, 2 - cz % 2) < 0.04)
        c[ceil] = np.where(cl[:, None], [0.35, 0.35, 0.38], [0.62, 0.62, 0.66])
        # Walls: hue per side, vertical stripes, a horizon band, height ticks.
        pw, aw, dw = pm[wall], ax[wall], dm[wall]
        side = np.where(aw == 0, np.where(dw[:, 0] > 0, 0, 1), np.where(dw[:, 2] > 0, 2, 3))
        u = np.where(aw == 0, pw[:, 2], pw[:, 0])
        base = WALL_COLORS[side]
        stripe = (np.floor(u / 0.75) % 2)[:, None]
        cw_ = base * (0.75 + 0.25 * stripe)
        band = (pw[:, 1] > 1.2) & (pw[:, 1] < 1.5)
        cw_[band] = [0.95, 0.95, 0.92]
        tick = (np.minimum(pw[:, 1] % 1, 1 - pw[:, 1] % 1) < 0.02)
        cw_[tick] *= 0.6
        c[wall] = cw_
        color[m] = c

    m = kind == 1
    if m.any():
        idx = pidx[m]
        lo, hi = PROP_LO[idx], PROP_HI[idx]
        c = _shade_lambert(PROP_COL[idx], pn[m])
        c *= (1 - 0.65 * _edge_lines(p[m], lo, hi, pn[m]))[:, None]
        color[m] = c

    m = kind == 2
    if m.any():
        q = p[m] - SPHERE_C
        nrm = q / SPHERE_R
        lon = np.arctan2(q[:, 0], q[:, 2])
        lat = np.arcsin(np.clip(q[:, 1] / SPHERE_R, -1, 1))
        seg = ((np.floor(lon / (math.pi / 6)) + np.floor(lat / (math.pi / 8))) % 2)[:, None]
        base = np.where(seg > 0, [0.92, 0.30, 0.22], [0.96, 0.92, 0.85])
        color[m] = _shade_lambert(base, nrm, 0.7)

    m = kind == 3
    if m.any():
        c = _shade_lambert(np.full((m.sum(), 3), 0.25), qn[m])
        c *= (1 - 0.6 * _edge_lines(p[m], PEDESTAL_LO[0], PEDESTAL_HI[0], qn[m]))[:, None]
        color[m] = c

    m = kind == 4
    if m.any():
        pm = p[m]
        u = np.clip((pm[:, 0] + card_w / 2) / card_w, 0, 1)
        v = np.clip(1 - (pm[:, 1] - FLOOR) / card_h, 0, 1)
        back = d[m, 2] > 0  # seen from behind: mirrored and darker
        u = np.where(back, 1 - u, u)
        px = np.minimum((u * cw).astype(int), cw - 1)
        py = np.minimum((v * ch).astype(int), ch - 1)
        c = card[py, px, :3].astype(np.float64)
        c[back] *= 0.45
        color[m] = c

    f = 1 - np.exp(-np.minimum(t_best, 60) / 28.0)
    color = color * (1 - f[:, None]) + FOG * f[:, None]
    return color.reshape(h, w, 3)


def render(dsl: str, n_frames: int, fps: float, strength: float, width: int, height: int,
           card: np.ndarray | None = None, motion_blur: int = 1) -> np.ndarray:
    """Render the guide clip. Returns uint8 (n_frames, height, width, 3).

    card: optional float RGB image in [0,1] (H,W,3) placed as the subject.
    motion_blur: subframes per frame, spread over a 360-degree shutter.
    """
    k = max(int(motion_blur), 1)
    offsets = (np.arange(k) + 0.5) / k - 0.5 if k > 1 else np.zeros(1)
    times = (np.arange(n_frames)[:, None] + offsets[None]).ravel()
    times = np.clip(times, 0, n_frames - 1)
    cams = build_at(dsl, n_frames, fps, strength, times)
    out = np.empty((n_frames, height, width, 3), dtype=np.uint8)

    def one(i):
        acc = sum(render_frame(c, width, height, card) for c in cams[i * k:(i + 1) * k]) / k
        out[i] = np.clip(acc * 255 + 0.5, 0, 255).astype(np.uint8)

    # numpy releases the GIL on the big array ops, so threads scale well enough
    # and stay safe inside ComfyUI (no fork of a CUDA process).
    with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 16)) as pool:
        list(pool.map(one, range(n_frames)))
    return out
