"""Camera rig, moves, and the shot DSL.

A shot is a set of rig parameters as a function of frame. Every named move is
a delta on those parameters with an easing curve; a timeline sums the deltas
of its segments, so moves compose (sequentially with "then", simultaneously
with "+"). The result is fully deterministic: same DSL + frame count + fps
=> the same camera path, bit for bit.

World: y is up, the subject sits at the origin, the camera starts on +z at
SUBJECT_DISTANCE looking toward -z. Distances are expressed as fractions of
the subject distance, angles in degrees, zoom in stops (log2 of focal length).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

SUBJECT_DISTANCE = 4.0
TARGET_HEIGHT = 0.2
BASE_FOV_Y = 40.0  # degrees
MIN_DISTANCE_FRAC = 0.12
FOV_RANGE = (6.0, 110.0)

PARAMS = ("yaw", "pitch", "dolly", "truck", "pedestal", "pan", "tilt", "roll", "zoom", "vertigo")


# ---------------------------------------------------------------- easing

def _smoother(u):
    return u * u * u * (u * (u * 6 - 15) + 10)


def _whip(u):
    # Logistic squeezed into the middle third, normalized to hit 0 and 1.
    k = 14.0
    lo, hi = 1 / (1 + math.exp(k * 0.5)), 1 / (1 + math.exp(-k * 0.5))
    return (1 / (1 + np.exp(-k * (u - 0.5))) - lo) / (hi - lo)


EASINGS = {
    "linear": lambda u: u,
    "ease_in": lambda u: u ** 3,
    "ease_out": lambda u: 1 - (1 - u) ** 3,
    "ease_in_out": _smoother,
    "expo_in": lambda u: np.where(u <= 0, 0.0, 2.0 ** (10 * (u - 1)) - 2.0 ** -10) / (1 - 2.0 ** -10),
    "expo_out": lambda u: (1 - 2.0 ** (-10 * u)) / (1 - 2.0 ** -10),
    "whip": _whip,
}


def ease(name: str, u):
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    return np.clip(EASINGS[name](u), 0.0, 1.0)


# ---------------------------------------------------------------- moves

@dataclass(frozen=True)
class Move:
    """A primitive camera move.

    ramp:  param deltas reached at the end of the segment (eased).
    bump:  param offsets that rise and fall back within the segment (sin).
    shake: param amplitudes of deterministic handheld noise.
    pre:   offsets applied before the shot starts (only for the first
           segment) so e.g. a zoom-out can begin tight.
    """

    ramp: dict = field(default_factory=dict)
    easing: str = "ease_in_out"
    bump: dict = field(default_factory=dict)
    shake: dict = field(default_factory=dict)
    pre: dict = field(default_factory=dict)


MOVES: dict[str, Move] = {
    "static": Move(),
    # translation
    "dolly_in": Move({"dolly": 0.45}),
    "dolly_out": Move({"dolly": -0.55}, pre={"dolly": 0.35}),
    "super_dolly_in": Move({"dolly": 0.75}, "ease_in"),
    "truck_left": Move({"truck": -0.6}),
    "truck_right": Move({"truck": 0.6}),
    "pedestal_up": Move({"pedestal": 0.4}),
    "pedestal_down": Move({"pedestal": -0.35}, pre={"pedestal": 0.2}),
    # rotation in place
    "pan_left": Move({"pan": 35}),
    "pan_right": Move({"pan": -35}),
    "tilt_up": Move({"tilt": 22}),
    "tilt_down": Move({"tilt": -22}),
    "roll_cw": Move({"roll": -25}),
    "roll_ccw": Move({"roll": 25}),
    "whip_pan_left": Move({"pan": 110}, "whip"),
    "whip_pan_right": Move({"pan": -110}, "whip"),
    "dutch_angle": Move({"roll": 18}, "ease_out"),
    # orbits
    "arc_left": Move({"yaw": -60}),
    "arc_right": Move({"yaw": 60}),
    "orbit_360": Move({"yaw": 360}, "linear"),
    "bullet_time": Move({"yaw": 160}, "linear"),
    "crane_up": Move({"pitch": 40, "pedestal": 0.15}, "ease_out", pre={"pitch": -4}),
    "crane_down": Move({"pitch": -40, "pedestal": -0.15}, "ease_in_out", pre={"pitch": 42, "pedestal": 0.15}),
    "top_down": Move({"yaw": 45}, "linear", pre={"pitch": 82, "dolly": -0.2}),
    "robo_arm": Move({"yaw": 70, "pitch": 25, "dolly": 0.3}, "ease_in_out"),
    "helicopter": Move({"yaw": 90}, "linear", pre={"pitch": 35, "dolly": -0.6}),
    "fpv_drone": Move({"dolly": 1.05, "pitch": -22, "yaw": 35}, "ease_in",
                      bump={"roll": 16}, shake={"roll": 1.5, "tilt": 0.8},
                      pre={"dolly": -0.55, "pitch": 26}),
    # lens
    "zoom_in": Move({"zoom": 1.0}),
    "zoom_out": Move({"zoom": -1.0}, pre={"zoom": 1.0}),
    "crash_zoom_in": Move({"zoom": 2.2}, "expo_in"),
    "crash_zoom_out": Move({"zoom": -2.2}, "expo_out", pre={"zoom": 2.2}),
    "dolly_zoom_in": Move({"vertigo": 0.55}),
    "dolly_zoom_out": Move({"vertigo": -0.55}, pre={"vertigo": 0.55}),
    # texture
    "handheld": Move(shake={"pan": 1.2, "tilt": 0.9, "roll": 0.8, "truck": 0.01, "pedestal": 0.008}),
}


# ---------------------------------------------------------------- DSL

class DSLError(ValueError):
    pass


@dataclass
class MoveSpec:
    name: str
    strength: float = 1.0
    easing: str | None = None

    def __str__(self):
        s = self.name
        if self.strength != 1.0:
            s += f"@{self.strength:g}"
        if self.easing:
            s += f" {self.easing}"
        return s


@dataclass
class Segment:
    moves: list[MoveSpec]
    start: str | None = None  # "48", "2s", "35%"
    end: str | None = None

    def __str__(self):
        body = " + ".join(str(m) for m in self.moves)
        return f"{self.start}-{self.end}: {body}" if self.start is not None else body


_TIME = r"\d+(?:\.\d+)?(?:s|%)?"
_RANGE_RE = re.compile(rf"^\s*({_TIME})\s*-\s*({_TIME})\s*:\s*(.*)$")
_SPLIT_RE = re.compile(r"\s*(?:\bthen\b|;|\n)\s*")


def parse(text: str) -> list[Segment]:
    """Parse a shot DSL string.

        dolly_in
        0-48: dolly_in then 48-96: pan_left@0.6 ease_out
        0%-35%: static; 35%-60%: whip_pan_left; 60%-100%: static
        dolly_in + handheld@0.5
    """
    segments = []
    for raw in _SPLIT_RE.split(text.strip()):
        if not raw:
            continue
        start = end = None
        m = _RANGE_RE.match(raw)
        if m:
            start, end, raw = m.group(1), m.group(2), m.group(3)
        moves = []
        for part in raw.split("+"):
            tokens = part.split()
            if not tokens:
                raise DSLError(f"empty move in {raw!r}")
            name, _, strength = tokens[0].partition("@")
            name = resolve_move(name)
            try:
                strength = float(strength) if strength else 1.0
            except ValueError:
                raise DSLError(f"bad strength in {tokens[0]!r}") from None
            easing = None
            for tok in tokens[1:]:
                if tok not in EASINGS:
                    raise DSLError(f"unknown easing {tok!r} (know: {', '.join(EASINGS)})")
                easing = tok
            moves.append(MoveSpec(name, strength, easing))
        segments.append(Segment(moves, start, end))
    if not segments:
        raise DSLError("empty shot")
    if len({s.start is None for s in segments}) > 1:
        raise DSLError("either give every segment a range or none of them")
    return segments


def format_dsl(segments: list[Segment]) -> str:
    return " then ".join(str(s) for s in segments)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


MOVE_ALIASES = {
    "whip_pan": "whip_pan_left",
    "dolly_zoom": "dolly_zoom_in",
    "vertigo": "dolly_zoom_in",
    "orbit": "orbit_360",
    "360_orbit": "orbit_360",
    "jib_up": "crane_up",
    "jib_down": "crane_down",
    "fpv": "fpv_drone",
    "overhead": "top_down",
    "push_in": "dolly_in",
    "pull_out": "dolly_out",
}


def resolve_move(name: str) -> str:
    n = _norm(name)
    n = MOVE_ALIASES.get(n, n)
    if n not in MOVES:
        raise DSLError(f"unknown move {name!r}")
    return n


def _to_frame(t: str, n_frames: int, fps: float) -> float:
    if t.endswith("%"):
        return float(t[:-1]) / 100.0 * (n_frames - 1)
    if t.endswith("s"):
        return float(t[:-1]) * fps
    return float(t)


# ---------------------------------------------------------------- evaluation

def _noise(seed: int, t: np.ndarray) -> np.ndarray:
    """Deterministic smooth noise in [-1, 1]: a fixed sum of sines."""
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.4, 2.6, 5)
    phases = rng.uniform(0, 2 * math.pi, 5)
    amps = 1.0 / freqs
    v = sum(a * np.sin(2 * math.pi * f * t + p) for a, f, p in zip(amps, freqs, phases))
    return v / amps.sum()


def rig_params(segments: list[Segment], n_frames: int, fps: float, strength: float = 1.0,
               times=None) -> dict[str, np.ndarray]:
    """Evaluate a shot to rig parameters at each frame (or at fractional frame `times`)."""
    frames = np.arange(n_frames, dtype=np.float64) if times is None else np.asarray(times, dtype=np.float64)
    out = {p: np.zeros(len(frames)) for p in PARAMS}

    if segments[0].start is None:
        # Unranged: split the shot evenly between segments.
        bounds = np.linspace(0, n_frames - 1, len(segments) + 1)
        ranges = [(bounds[i], bounds[i + 1]) for i in range(len(segments))]
    else:
        ranges = [(_to_frame(s.start, n_frames, fps), _to_frame(s.end, n_frames, fps)) for s in segments]

    for i, (seg, (a, b)) in enumerate(zip(segments, ranges)):
        span = max(b - a, 1e-6)
        u = (frames - a) / span
        for spec in seg.moves:
            move = MOVES[spec.name]
            k = spec.strength * strength
            e = ease(spec.easing or move.easing, u)
            for p, v in move.ramp.items():
                out[p] += k * v * e
            inside = (u >= 0) & (u <= 1)
            for p, v in move.bump.items():
                out[p] += k * v * np.where(inside, np.sin(math.pi * np.clip(u, 0, 1)), 0.0)
            for j, (p, v) in enumerate(move.shake.items()):
                out[p] += k * v * np.where(inside, _noise(1000 * i + j, frames / fps), 0.0)
            if i == 0:
                for p, v in move.pre.items():
                    out[p] += k * v
    return out


@dataclass
class Camera:
    position: np.ndarray  # (3,)
    rotation: np.ndarray  # (3,3) world-from-camera; columns = right, up, back
    fov_y: float  # degrees

    def c2w(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.rotation
        m[:3, 3] = self.position
        return m


def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    x, y, z = axis / np.linalg.norm(axis)
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def camera_from_rig(r: dict[str, float]) -> Camera:
    D = SUBJECT_DISTANCE
    dist_free = max(1.0 - r["dolly"], MIN_DISTANCE_FRAC)
    dist = max(dist_free - r["vertigo"], MIN_DISTANCE_FRAC) * D
    yaw, pitch = math.radians(r["yaw"]), math.radians(np.clip(r["pitch"], -85, 89))
    target = np.array([0.0, TARGET_HEIGHT, 0.0])
    direction = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)])
    pos = target + dist * direction

    fwd = -direction
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, world_up)
    if np.linalg.norm(right) < 1e-6:  # straight down: keep heading from yaw
        right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    pos = pos + D * (r["truck"] * right + r["pedestal"] * world_up)

    # Pan about world up, tilt about the camera's right axis, roll about forward.
    R = np.stack([right, up, -fwd], axis=1)
    R = _rot(world_up, r["pan"]) @ R
    R = _rot(R[:, 0], r["tilt"]) @ R
    R = _rot(-R[:, 2], -r["roll"]) @ R

    # Zoom in stops; vertigo zooms to cancel its own dolly so the subject holds size.
    focal_scale = 2.0 ** r["zoom"] * (dist / (max(dist_free, MIN_DISTANCE_FRAC) * D))
    half = math.atan(math.tan(math.radians(BASE_FOV_Y) / 2) / focal_scale)
    fov = float(np.clip(math.degrees(2 * half), *FOV_RANGE))
    return Camera(pos, R, fov)


def snap_frames(duration_s: float, fps: float) -> int:
    """LTX needs 8n+1 frames."""
    n = max(round(duration_s * fps), 9)
    return 8 * round((n - 1) / 8) + 1


@dataclass
class Trajectory:
    dsl: str
    n_frames: int
    fps: float
    strength: float
    cameras: list[Camera]

    def to_json(self) -> dict:
        return {
            "dsl": self.dsl,
            "n_frames": self.n_frames,
            "fps": self.fps,
            "strength": self.strength,
            "convention": "c2w, OpenGL camera (x right, y up, looking -z), world y-up, subject at origin",
            "frames": [
                {"c2w": np.round(c.c2w(), 6).tolist(), "fov_y": round(c.fov_y, 4)} for c in self.cameras
            ],
        }


def build(dsl: str, n_frames: int, fps: float = 24.0, strength: float = 1.0) -> Trajectory:
    segments = parse(dsl)
    rig = rig_params(segments, n_frames, fps, strength)
    cams = [camera_from_rig({p: float(rig[p][i]) for p in PARAMS}) for i in range(n_frames)]
    return Trajectory(format_dsl(segments), n_frames, fps, strength, cams)


def build_at(dsl: str, n_frames: int, fps: float, strength: float, frame_times) -> list[Camera]:
    """Cameras at fractional frame times (for motion-blur subframes)."""
    rig = rig_params(parse(dsl), n_frames, fps, strength, times=frame_times)
    return [camera_from_rig({p: float(rig[p][i]) for p in PARAMS}) for i in range(len(frame_times))]
