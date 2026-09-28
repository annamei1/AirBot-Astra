"""
Harness configuration.

VLM endpoint comes from environment variables so no key is ever committed:

    export HARNESS_VLM_API_KEY=sk-...
    export HARNESS_VLM_MODEL=gpt-6            # any OpenAI-compatible multimodal model with tool calling
    export HARNESS_VLM_BASE_URL=...           # omit for api.openai.com; set for a gateway / DashScope
    export HARNESS_VLM_REASONING=medium       # "", low, medium, high, xhigh (dropped if endpoint rejects it)

Robot constants (ports, cameras, workspace) are re-exported from play.config, which reads config/play_config.json.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HARNESS_DIR = Path(__file__).resolve().parent
LOG_DIR = HARNESS_DIR / "logs"
FRAME_DIR = LOG_DIR / "frames"


def _load_dotenv(path: Path):
    """Load KEY=VALUE lines from harness/.env (gitignored) into os.environ; shell exports still win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key.startswith("export "):
            key = key[7:].strip()
        os.environ.setdefault(key, value)


_load_dotenv(HARNESS_DIR / ".env")

# ---- VLM ----
VLM_MODEL = os.environ.get("HARNESS_VLM_MODEL", "gpt-6-astra")
VLM_BASE_URL = os.environ.get("HARNESS_VLM_BASE_URL") or None
VLM_API_KEY = os.environ.get("HARNESS_VLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
VLM_REASONING = os.environ.get("HARNESS_VLM_REASONING", "medium")
VLM_API = os.environ.get("HARNESS_VLM_API", "responses")        # responses (reasoning + tools) | chat
# 60 s: the slowest of 210 replies on 2026-09-26 took 11 s, and a stalled request is retried sooner.
VLM_TIMEOUT = float(os.environ.get("HARNESS_VLM_TIMEOUT", "60"))
IMAGE_MAX_EDGE = int(os.environ.get("HARNESS_IMAGE_MAX_EDGE", "768"))

# ---- Rig geometry, measured rather than assumed ----
# How far the fingertips reach beyond the reported end-effector origin along the approach axis.
# Measured on 2026-09-16 by probe_contact: the arm met a surface the wrist camera put at -0.0807
# with the gripper reporting -0.0775. It is a lower bound, because the contact stop fires after the
# current has already risen, so the fingers had pressed in slightly before the arm noticed.
# NOT the same thing as play.config.TABLE_SURFACE_Z, which is a configured constant roughly 15 mm above
# the physical table.
FINGERTIP_OFFSET_M = 0.0032

# ---- Agent ----
# Model calls per episode. 40 was not enough for two objects: the 04:15 run put both in the bowl and
# was still scored a failure because the limit arrived before done() did.
MAX_STEPS = int(os.environ.get("HARNESS_MAX_STEPS", "60"))
MAX_PICK_RETRIES = 3

# ---- Robot (single source of truth: config/play_config.json via play.config) ----
from play.config import (  # noqa: E402
    HOVER_HEIGHT, GRASP_Z, TABLE_SURFACE_Z,
    WORKSPACE_BOUNDS_MIN, WORKSPACE_BOUNDS_MAX,
    SAM3_CHECKPOINT, SAM3_CONFIDENCE, CALIB_DIR, HEAD_ZOOM,
    load_head_camera_calibration, load_hand_eye_calibration,
)


# ---- Second arm: its own hand-eye, and where its base sits in the first arm's base ----
# Everything the harness measures lives in the RIGHT arm's base frame, so that frame is the world and
# the right arm's base-to-world is identity. These two files place the left arm in it; both come from
# calibration/play/.
LEFT_HAND_EYE_FILE = CALIB_DIR / "hand_eye_extrinsics_left.json"
BASE_TO_BASE_FILE = CALIB_DIR / "base_to_base_extrinsics.json"


def _load_transform(path):
    import json as _json
    import numpy as _np
    with open(path) as f:
        d = _json.load(f)
    T = _np.eye(4)
    T[:3, :3] = _np.array(d["rotation_matrix"])
    T[:3, 3] = _np.array(d["translation"])
    return T


def load_left_hand_eye_calibration():
    """→ (intrinsics dict, extrinsics dict) for the left wrist camera, same shape as the right one."""
    import json as _json
    with open(LEFT_HAND_EYE_FILE) as f:
        extrinsics = _json.load(f)
    intrinsics = extrinsics.get("camera_intrinsics")
    if intrinsics is None:
        raise ValueError(f"No camera_intrinsics in {LEFT_HAND_EYE_FILE}")
    return intrinsics, extrinsics


def load_base_to_base():
    """→ T_left_base -> right_base as a 4x4, or None if the arms have never been tied together.

    Measured at 8 mm on 2026-09-18, and that is the floor of the arms' own kinematics rather than of the
    calibration: two independent calibrations of the right hand-eye agreed to 0.7 mm and the residual did
    not move. Good enough to say roughly where the other arm's things are and whether the two will
    collide; NOT good enough to hand one arm a grasp point measured by the other. For that, let the arm
    that is going to act re-measure with its own wrist camera once it is there.
    """
    if not BASE_TO_BASE_FILE.exists():
        return None
    return _load_transform(BASE_TO_BASE_FILE)


def require_vlm_key():
    if not VLM_API_KEY:
        raise SystemExit("HARNESS_VLM_API_KEY (or OPENAI_API_KEY) is not set")
