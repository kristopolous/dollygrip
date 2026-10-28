from . import presets, render, sharecode, trajectory
from .trajectory import snap_frames

__all__ = ["presets", "render", "sharecode", "trajectory", "snap_frames", "resolve_shot"]


def resolve_shot(preset: str = "", dsl: str = "", share_code: str = "") -> tuple[str, str, int]:
    """Pick the shot source: share code > custom DSL > preset.

    Returns (dsl, prompt_hint, motion_blur).
    """
    if share_code.strip():
        return sharecode.decode(share_code)["dsl"], "", 1
    if dsl.strip():
        trajectory.parse(dsl)
        return dsl.strip(), "", 1
    p = presets.get(preset)
    return p.dsl, p.prompt_hint, p.motion_blur
