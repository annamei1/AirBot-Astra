"""
AirBot Play configuration loader.
Reads config/play_config.json and exports module-level constants
so downstream code can continue using:
    from play.config import HOME_JOINT, HOVER_HEIGHT, ...
"""
import json
import os
import numpy as np
from pathlib import Path

# ==================== Load JSON config ====================
_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "play_config.json"

with open(_CONFIG_PATH) as _f:
    _cfg = json.load(_f)

# ==================== Arm ====================
ARM_PORT = _cfg["arm"]["port"]
HOME_JOINT = _cfg["arm"]["home_joint"]
ZERO_JOINT = _cfg["arm"]["zero_joint"]

# The second arm, if this rig has one. Its base is NOT the world frame — everything measured lives in
# the first arm's base, and base_to_base_extrinsics.json is what carries a pose across. Absent from an
# older config file, in which case the rig is single-arm and nothing below is used.
_second = _cfg.get("second_arm") or {}
SECOND_ARM_PORT = _second.get("port")
SECOND_WRIST_CAMERA_SERIAL = _second.get("wrist_serial")

# ==================== Cameras ====================
WRIST_CAMERA_SERIAL = _cfg["cameras"]["wrist_serial"]
HEAD_CAMERA_SERIAL = _cfg["cameras"]["head_serial"]
D405_DEPTH_RAW_TO_MM = _cfg["cameras"]["d405_depth_raw_to_mm"]
HEAD_ZOOM = _cfg["cameras"].get("head_zoom", 1.0)
WRIST_ZOOM = _cfg["cameras"].get("wrist_zoom", 1.0)
# Stream format, shared by all three cameras. The harness grabs a frame when it wants one and never
# reads a stream continuously, so frames per second buys it nothing and costs USB bandwidth — which is
# the scarce thing once three D405 share a machine. Defaults are what the SDK already used.
CAMERA_WIDTH = _cfg["cameras"].get("width", 640)
CAMERA_HEIGHT = _cfg["cameras"].get("height", 480)
CAMERA_FPS = _cfg["cameras"].get("fps", 30)

# ==================== Calibration ====================
CALIB_DIR = _PROJECT_ROOT / _cfg["calibration"]["dir"]
HAND_EYE_EXTRINSICS_FILE = CALIB_DIR / _cfg["calibration"]["hand_eye_extrinsics"]
HEAD_CAMERA_EXTRINSICS_FILE = CALIB_DIR / _cfg["calibration"]["head_camera_extrinsics"]

# ==================== Workspace ====================
WORKSPACE_BOUNDS_MIN = _cfg["workspace"]["bounds_min"]
WORKSPACE_BOUNDS_MAX = _cfg["workspace"]["bounds_max"]

# ==================== Grasp ====================
HOVER_HEIGHT = _cfg["grasp"]["hover_height"]
GRIPPER_RELEASE_WIDTH = _cfg["grasp"]["gripper_release_width"]
GRASP_Z = _cfg["grasp"].get("grasp_z", None)  # fixed grasp Z per layer, None = use depth
TABLE_SURFACE_Z = _cfg["grasp"].get("table_surface_z", GRASP_Z) 

# ==================== SAM3 ====================
# "$SAM3_HOME/..." and "~/..." are expanded; SAM3_HOME defaults to ~/sam3-main.
os.environ.setdefault("SAM3_HOME", os.path.expanduser("~/sam3-main"))
SAM3_CHECKPOINT = os.path.expanduser(os.path.expandvars(_cfg["sam3"]["checkpoint"]))
SAM3_CONFIDENCE = _cfg["sam3"]["confidence"]

# ==================== Timing ====================
MOVE_WAIT_SEC = _cfg["timing"]["move_wait_sec"]
GRIPPER_CLOSE_WAIT_SEC = _cfg["timing"]["gripper_close_wait_sec"]
GRIPPER_OPEN_WAIT_SEC = _cfg["timing"]["gripper_open_wait_sec"]


# ==================== Helpers ====================
def load_hand_eye_calibration():
    """Load hand-eye extrinsics and camera intrinsics."""
    with open(HAND_EYE_EXTRINSICS_FILE) as f:
        extrinsics = json.load(f)
    intrinsics = extrinsics.get('camera_intrinsics', None)
    if intrinsics is None:
        raise ValueError(f"No camera_intrinsics in {HAND_EYE_EXTRINSICS_FILE}")
    return intrinsics, extrinsics


def load_head_camera_calibration():
    """Load head camera extrinsics. Returns (intrinsics_dict, T_head2base 4x4 ndarray)."""
    with open(HEAD_CAMERA_EXTRINSICS_FILE) as f:
        data = json.load(f)
    intrinsics = data.get('head_camera_intrinsics', None)
    if intrinsics is None:
        raise ValueError(f"No head_camera_intrinsics in {HEAD_CAMERA_EXTRINSICS_FILE}")

    R = np.array(data['rotation_matrix'])
    t = np.array(data['translation'])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return intrinsics, T
