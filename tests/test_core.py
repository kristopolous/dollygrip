import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "node_pack" / "ComfyUI-DollyGrip"))

from dollygrip import presets, render, resolve_shot, sharecode, snap_frames, trajectory  # noqa: E402


@pytest.mark.parametrize("p", presets.PRESETS, ids=lambda p: p.id)
def test_every_preset_parses_and_builds(p):
    traj = trajectory.build(p.dsl, 49)
    assert len(traj.cameras) == 49
    for c in traj.cameras:
        assert np.all(np.isfinite(c.position))
        assert np.allclose(c.rotation.T @ c.rotation, np.eye(3), atol=1e-9)
        assert trajectory.FOV_RANGE[0] <= c.fov_y <= trajectory.FOV_RANGE[1]


@pytest.mark.parametrize("p", presets.PRESETS, ids=lambda p: p.id)
def test_camera_stays_inside_the_set(p):
    lo, hi = render.ROOM_LO, render.ROOM_HI
    for c in trajectory.build(p.dsl, 49).cameras:
        assert np.all(c.position > lo) and np.all(c.position < hi), p.id
        r = np.hypot(c.position[0], c.position[2])
        assert r < render.CAMERA_RING[1], f"{p.id} leaves the camera ring (r={r:.2f})"


def test_guide_is_bit_identical_across_runs():
    a = render.render("crash_zoom_in + handheld@0.5", 17, 24, 1.0, 128, 96, motion_blur=2)
    b = render.render("crash_zoom_in + handheld@0.5", 17, 24, 1.0, 128, 96, motion_blur=2)
    assert a.shape == (17, 96, 128, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)


def test_static_is_static_and_moves_move():
    s = render.render("static", 9, 24, 1.0, 64, 48)
    assert np.array_equal(s[0], s[-1])
    m = render.render("dolly_in", 9, 24, 1.0, 64, 48)
    assert not np.array_equal(m[0], m[-1])


def test_first_frame_matches_across_presets_without_pre_offsets():
    first = trajectory.build("static", 9).cameras[0]
    for name in ("dolly_in", "orbit_360", "pan_left", "crash_zoom_in"):
        c = trajectory.build(name, 9).cameras[0]
        assert np.allclose(c.position, first.position) and np.isclose(c.fov_y, first.fov_y)


def test_orbit_returns_home():
    t = trajectory.build("orbit_360", 49).cameras
    assert np.allclose(t[0].position, t[-1].position, atol=1e-9)


def test_dolly_zoom_holds_subject_size():
    # Subject size on screen ~ focal / distance; vertigo keeps it constant.
    for c in trajectory.build("dolly_zoom_in", 25).cameras:
        dist = np.linalg.norm(c.position - [0, trajectory.TARGET_HEIGHT, 0])
        focal = 1 / np.tan(np.radians(c.fov_y) / 2)
        base = 1 / np.tan(np.radians(trajectory.BASE_FOV_Y) / 2) / trajectory.SUBJECT_DISTANCE
        assert np.isclose(focal / dist, base, rtol=1e-6)


def test_duration_changes_speed_not_travel():
    short = trajectory.build("dolly_in", snap_frames(2, 24)).cameras
    long = trajectory.build("dolly_in", snap_frames(6, 24)).cameras
    assert np.allclose(short[-1].position, long[-1].position)


def test_snap_frames_is_8n_plus_1():
    for d in (1, 2.5, 5, 7.3, 10):
        assert snap_frames(d, 24) % 8 == 1


def test_dsl_ranges_and_errors():
    segs = trajectory.parse("0-48: dolly_in then 48-96: pan_left@0.6 ease_out")
    assert [s.moves[0].name for s in segs] == ["dolly_in", "pan_left"]
    assert segs[1].moves[0].strength == 0.6 and segs[1].moves[0].easing == "ease_out"
    assert trajectory.parse("Vertigo")[0].moves[0].name == "dolly_zoom_in"
    for bad in ("", "warp_speed", "dolly_in wobble", "0-10: dolly_in then pan_left", "dolly_in@x"):
        with pytest.raises(trajectory.DSLError):
            trajectory.parse(bad)


def test_seconds_and_percent_ranges_agree():
    a = trajectory.build("0s-1s: dolly_in then 1s-2s: static", 49, 24)
    b = trajectory.build("0%-50%: dolly_in then 50%-100%: static", 49, 24)
    assert np.allclose(a.cameras[30].position, b.cameras[30].position)


def test_sharecode_roundtrip():
    dsl = "0%-30%: static then 30%-62%: whip_pan_left then 62%-100%: static"
    code = sharecode.encode(dsl, 0.8, 4.0)
    assert code.startswith("dg1-") and len(code) < 120
    out = sharecode.decode(code)
    assert out == {"dsl": dsl, "strength": 0.8, "duration_s": 4.0}
    a = trajectory.build(dsl, 49, strength=0.8).to_json()
    b = trajectory.build(out["dsl"], 49, strength=out["strength"]).to_json()
    assert a["frames"] == b["frames"]


def test_resolve_shot_priority():
    code = sharecode.encode("arc_left")
    assert resolve_shot("orbit_360", "dolly_in", code)[0] == "arc_left"
    assert resolve_shot("orbit_360", "dolly_in", "")[0] == "dolly_in"
    assert resolve_shot("360 Orbit")[0] == "orbit_360"


def test_image_card_scene():
    card = np.zeros((90, 160, 3), np.float32)
    card[..., 0] = 1.0  # pure red card
    f = render.render("static", 9, 24, 1.0, 96, 64, card=card)
    center = f[0, 32, 48].astype(int)
    assert center[0] > 150 and center[1] < 80
