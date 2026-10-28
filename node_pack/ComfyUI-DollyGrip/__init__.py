"""ComfyUI-DollyGrip: named camera moves as real, deterministic camera paths.

The DollyGrip Camera node turns a preset / DSL / share code into a rendered
guide clip (IMAGE batch) for the LTX Cameraman IC-LoRA, plus the frame count,
fps, sizes and prompt the rest of the graph needs, so an app only has to set
this one node and the start image.
"""

import json

import numpy as np
import torch

from .dollygrip import presets, render, resolve_shot, sharecode, snap_frames, trajectory


def _snap32(x):
    return max(32, int(round(x / 32.0)) * 32)


class DollyGripCamera:
    CATEGORY = "DollyGrip"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "INT", "INT", "INT", "INT", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("guide", "frames", "fps_int", "fps", "width", "height", "stage1_width", "stage1_height",
                    "prompt", "seed", "trajectory_json", "share_code")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (presets.PRESET_IDS, {"default": "dolly_in"}),
                "duration_s": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 50}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "width": ("INT", {"default": 1920, "min": 256, "max": 3840, "step": 32}),
                "height": ("INT", {"default": 1088, "min": 256, "max": 2176, "step": 32}),
                "guide_width": ("INT", {"default": 960, "min": 256, "max": 1920, "step": 32}),
                "guide_height": ("INT", {"default": 544, "min": 256, "max": 1088, "step": 32}),
                "scene": (["set", "start_image_card"], {"default": "set"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "add_camera_words": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "dsl": ("STRING", {"multiline": True, "default": ""}),
                "share_code": ("STRING", {"default": ""}),
                "start_image": ("IMAGE",),
            },
        }

    def run(self, preset, duration_s, fps, strength, width, height, guide_width, guide_height, scene,
            prompt, add_camera_words, seed, dsl="", share_code="", start_image=None):
        shot, hint, blur = resolve_shot(preset, dsl, share_code)
        n = snap_frames(duration_s, fps)
        card = None
        if scene == "start_image_card" and start_image is not None:
            card = start_image[0].detach().cpu().numpy().astype(np.float32)
        frames = render.render(shot, n, float(fps), strength, _snap32(guide_width), _snap32(guide_height),
                               card=card, motion_blur=blur)
        guide = torch.from_numpy(frames).float() / 255.0
        full_prompt = prompt.strip()
        if add_camera_words and hint:
            full_prompt = f"{full_prompt}. {hint}." if full_prompt else f"{hint}."
        traj = trajectory.build(shot, n, float(fps), strength)
        w, h = _snap32(width), _snap32(height)
        return (guide, n, int(fps), float(fps), w, h, _snap32(w / 2), _snap32(h / 2),
                full_prompt, seed, json.dumps(traj.to_json()), sharecode.encode(shot, strength, duration_s))


NODE_CLASS_MAPPINGS = {"DollyGripCamera": DollyGripCamera}
NODE_DISPLAY_NAME_MAPPINGS = {"DollyGripCamera": "DollyGrip Camera"}
