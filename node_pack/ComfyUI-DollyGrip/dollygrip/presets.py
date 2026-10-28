"""Named camera presets.

A preset is a shot DSL string plus the words that help the video model read
the move. Because presets are just DSL, users can fork any of them by editing
the string, and a share code is the DSL itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .trajectory import _norm


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    description: str
    dsl: str
    prompt_hint: str
    motion_blur: int = 1  # subframes averaged in the guide render
    mvp: bool = False
    aliases: tuple[str, ...] = ()

    def to_json(self):
        return asdict(self)


PRESETS: list[Preset] = [
    Preset("dolly_in", "Dolly In", "Slow push toward the subject; background parallax grows.",
           "dolly_in", "slow smooth dolly in toward the subject", mvp=True, aliases=("push in",)),
    Preset("dolly_out", "Dolly Out", "Pull back from the subject to reveal the space.",
           "dolly_out", "slow smooth dolly out, pulling back to reveal the scene", mvp=True, aliases=("pull out",)),
    Preset("orbit_360", "360 Orbit", "Full circle around the subject at constant speed.",
           "orbit_360", "the camera orbits 360 degrees around the subject", mvp=True, aliases=("360 orbit", "orbit")),
    Preset("crash_zoom_in", "Crash Zoom In", "Violent accelerating lens zoom onto the subject.",
           "0%-40%: static then 40%-100%: crash_zoom_in", "sudden fast crash zoom in on the subject",
           motion_blur=3, mvp=True),
    Preset("crash_zoom_out", "Crash Zoom Out", "Starts tight, snaps wide.",
           "0%-60%: crash_zoom_out then 60%-100%: static", "sudden fast crash zoom out", motion_blur=3, mvp=True),
    Preset("dolly_zoom", "Dolly Zoom (Vertigo)", "Camera pushes in while the lens widens: subject holds, world stretches.",
           "dolly_zoom_in", "dolly zoom vertigo effect, background warps while the subject stays the same size",
           mvp=True, aliases=("vertigo", "dolly zoom in")),
    Preset("bullet_time", "Bullet Time", "Half-orbit around a frozen moment.",
           "bullet_time", "bullet time, frozen moment, time stands still as the camera sweeps around the subject",
           mvp=True),
    Preset("fpv_drone", "FPV Drone", "Swooping drone dive with banking.",
           "fpv_drone", "fpv drone shot diving and banking toward the subject", mvp=True, aliases=("fpv",)),
    Preset("crane_up", "Crane Up", "Camera rises and looks down on the subject.",
           "crane_up", "crane shot rising up above the subject", mvp=True, aliases=("jib up",)),
    Preset("whip_pan", "Whip Pan", "Hold, then a blurred snap pan, then hold.",
           "0%-30%: static then 30%-62%: whip_pan_left then 62%-100%: static",
           "fast whip pan with motion blur", motion_blur=5, mvp=True),
    # Beyond the MVP set.
    Preset("arc_left", "Arc Left", "Partial orbit to the left.", "arc_left", "the camera arcs around to the left"),
    Preset("arc_right", "Arc Right", "Partial orbit to the right.", "arc_right", "the camera arcs around to the right"),
    Preset("crane_down", "Crane Down", "Descend from high angle to eye level.", "crane_down",
           "crane shot descending to eye level", aliases=("jib down",)),
    Preset("super_dolly", "Super Dolly In", "Long accelerating push.", "super_dolly_in",
           "fast long dolly push in"),
    Preset("robo_arm", "Robo Arm", "Compound sweep: orbit, rise and push together.", "robo_arm",
           "robotic arm camera sweeping around and up toward the subject"),
    Preset("helicopter", "Helicopter", "Wide high slow circle.", "helicopter",
           "aerial helicopter shot slowly circling"),
    Preset("overhead", "Overhead", "Top-down view, slowly rotating.", "top_down",
           "top down overhead shot rotating slowly", aliases=("top down",)),
    Preset("truck_left", "Truck Left", "Slide sideways to the left.", "truck_left", "camera trucks left"),
    Preset("truck_right", "Truck Right", "Slide sideways to the right.", "truck_right", "camera trucks right"),
    Preset("pan_left", "Pan Left", "Pivot left from a fixed position.", "pan_left", "camera pans left"),
    Preset("tilt_up", "Tilt Up", "Pivot upward from a fixed position.", "tilt_up", "camera tilts up"),
    Preset("dutch_angle", "Dutch Angle", "Ease into a canted frame while pushing in.",
           "dutch_angle + dolly_in@0.4", "dutch angle, tilted frame"),
    Preset("handheld", "Handheld", "Documentary micro-shake with a slow drift in.",
           "handheld + dolly_in@0.25", "handheld camera, natural shake"),
    Preset("double_dolly", "Double Dolly", "Push in, settle, push again.",
           "dolly_in@0.5 then static then dolly_in@0.6", "two-stage dolly in"),
]

_BY_KEY: dict[str, Preset] = {}
for _p in PRESETS:
    for _k in (_p.id, _p.label, *_p.aliases):
        _BY_KEY[_norm(_k)] = _p


def get(name: str) -> Preset:
    try:
        return _BY_KEY[_norm(name)]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}") from None


PRESET_IDS = [p.id for p in PRESETS]
