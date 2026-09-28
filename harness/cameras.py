"""
Camera views: one frame from one camera, carrying everything needed to turn its pixels into
metric points in the robot base frame — and to go back the other way.

A view is the only thing the geometry code knows about, so a fixed head camera, a wrist camera on a
moving arm, and (later) a second wrist camera on a second arm are all the same object. The head
camera's transform is static from calibration; a wrist camera's is the arm pose at the moment the
frame was taken, composed with the hand-eye extrinsics. Nothing downstream needs to care which.

`project` is what makes cameras cooperate: once anything has been located in the base frame by one
camera, every other camera can be asked where that point falls in its own image. That turns a
close-up into a geometric question ("segment whatever is at this pixel") instead of a naming
question ("segment the black block"), which is what the text encoder keeps failing at.

A camera mounted on the robot also sees the robot. A wrist camera looks past its own gripper
fingers, so every view may carry a `self_mask` of pixels that belong to the robot. Those pixels
never count as object, support surface or depth evidence, and a camera whose view of an object is
mostly blocked by them is not asked to measure it. First live run: a refine through the fingers
bled onto the table around a half-hidden brick and was wrongly accepted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

DEPTH_MIN_MM = 100.0
DEPTH_MAX_MM = 2000.0
# Robot pixels in a wrist view: anything closer to the camera than this is the robot itself (the
# gripper fingers hang ~5-15 cm in front of the lens), never the scene. Calibrate per mount with
# scripts/capture_frame.py — the wrist depth it saves shows the finger depths directly.
SELF_NEAR_MM = 160.0
SELF_DILATE_PX = 15


@dataclass
class View:
    name: str                          # "head", "wrist", later "left_wrist" / "right_wrist"
    rgb: np.ndarray
    depth: Optional[np.ndarray]        # millimetres, same shape as rgb, or None
    fx: float
    fy: float
    cx: float
    cy: float
    T_cam2base: np.ndarray             # 4x4
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))   # static base-frame correction
    step: int = 0
    t: float = field(default_factory=time.time)
    eye_xyz: Optional[List[float]] = None       # where this camera was, base frame
    self_mask: Optional[np.ndarray] = None      # bool, True where the pixel shows the robot itself
    note: str = ""

    # ---------------- shape ----------------

    @property
    def shape(self) -> Tuple[int, int]:
        h, w = self.rgb.shape[:2]
        return h, w

    def contains(self, u: float, v: float, margin: int = 0) -> bool:
        h, w = self.shape
        return margin <= u < w - margin and margin <= v < h - margin

    def scene(self, mask: np.ndarray) -> np.ndarray:
        """A mask with the robot's own pixels removed."""
        m = mask.astype(bool)
        if self.self_mask is not None and self.self_mask.shape == m.shape:
            m = m & ~self.self_mask
        return m

    def occluded_fraction(self, box_xyxy: Tuple[int, int, int, int]) -> float:
        """Fraction of a pixel box that shows the robot rather than the scene."""
        if self.self_mask is None:
            return 0.0
        x0, y0, x1, y1 = box_xyxy
        patch = self.self_mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        return float(patch.mean()) if patch.size else 0.0

    # ---------------- pixels → base frame ----------------

    def backproject(self, mask: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Scene pixels of a mask → Nx3 points in the base frame, plus the valid-depth fraction."""
        if self.depth is None:
            return None, 0.0
        m = self.scene(mask)
        if not m.any():
            return None, 0.0
        ys, xs = np.where(m)
        d = self.depth[ys, xs].astype(np.float32)
        ok = (d > DEPTH_MIN_MM) & (d < DEPTH_MAX_MM)
        if ok.sum() < 8:
            return None, float(ok.mean())
        z = d[ok] / 1000.0
        cam = np.stack([(xs[ok] - self.cx) * z / self.fx,
                        (ys[ok] - self.cy) * z / self.fy,
                        z, np.ones(ok.sum())])
        return (self.T_cam2base @ cam)[:3].T + self.offset, float(ok.mean())

    def pixel_to_base(self, u: float, v: float, depth_m: float) -> np.ndarray:
        cam = np.array([(u - self.cx) * depth_m / self.fx,
                        (v - self.cy) * depth_m / self.fy, depth_m, 1.0])
        return (self.T_cam2base @ cam)[:3] + self.offset

    # ---------------- base frame → pixels ----------------

    def project(self, xyz_base) -> Optional[Tuple[float, float, float]]:
        """Base-frame point → (u, v, range_m) in this view, or None if it is behind the camera."""
        p = np.linalg.inv(self.T_cam2base) @ np.append(np.asarray(xyz_base, float) - self.offset, 1.0)
        if p[2] <= 1e-6:
            return None
        return float(self.fx * p[0] / p[2] + self.cx), float(self.fy * p[1] / p[2] + self.cy), float(p[2])

    def px_per_m(self, range_m: float) -> float:
        return max(self.fx, self.fy) / max(range_m, 0.03)

    def box_around(self, xyz_base, extent_m: float, pad: float = 1.6) -> Optional[Tuple[int, int, int, int]]:
        """Pixel box covering a physical extent centred on a base-frame point, as seen from here."""
        pr = self.project(xyz_base)
        if pr is None:
            return None
        u, v, rng = pr
        half = 0.5 * pad * max(extent_m, 0.01) * self.px_per_m(rng)
        half = float(np.clip(half, 12.0, 0.45 * min(self.shape)))
        h, w = self.shape
        x0, y0 = int(np.clip(u - half, 0, w - 2)), int(np.clip(v - half, 0, h - 2))
        x1, y1 = int(np.clip(u + half, x0 + 2, w - 1)), int(np.clip(v + half, y0 + 2, h - 1))
        return x0, y0, x1, y1

    def describe(self) -> Dict[str, Any]:
        out = {"camera": self.name, "size": [self.shape[1], self.shape[0]],
               "has_depth": self.depth is not None,
               "eye_xyz": None if self.eye_xyz is None else [round(float(c), 4) for c in self.eye_xyz],
               "note": self.note}
        if self.self_mask is not None:
            out["robot_pixels_frac"] = round(float(self.self_mask.mean()), 3)
        return out


# ==================== building views from the robot ====================

SELF_GROW_ITERS = 60     # geodesic growth steps (1 px each) from near pixels into adjacent dropouts


def self_mask_from_depth(depth: Optional[np.ndarray], near_mm: float = SELF_NEAR_MM,
                         dilate_px: int = SELF_DILATE_PX, grow_iters: int = SELF_GROW_ITERS
                         ) -> Optional[np.ndarray]:
    """Pixels that are the robot itself, for a camera that looks past part of the robot.

    The gripper fingers sit closer than the sensor's working range, so about half of their pixels come
    back with no depth at all rather than a small one. Measured on this mount: 45-54% zeros over the
    finger areas, valid finger pixels clustered at 60-160 mm, and an empty 160-200 mm gap before the
    scene. So start from the pixels that read near and grow them geodesically — one pixel at a time,
    only through depth dropouts connected to them. That fills the fingers without jumping across the
    scene, so a dark object whose own depth drops out is not swallowed unless it touches the gripper.
    """
    if depth is None:
        return None
    near = ((depth > 0) & (depth < near_mm)).astype(np.uint8)
    near = cv2.morphologyEx(near, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if not near.any():
        return np.zeros(depth.shape, bool)
    allowed = ((near > 0) | (depth == 0)).astype(np.uint8)
    robot, k = near.copy(), np.ones((3, 3), np.uint8)
    for _ in range(grow_iters):
        nxt = cv2.dilate(robot, k) & allowed
        if np.array_equal(nxt, robot):
            break
        robot = nxt
    robot = cv2.morphologyEx(robot, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return cv2.dilate(robot, np.ones((dilate_px, dilate_px), np.uint8)).astype(bool)


@dataclass
class Wrist:
    """One camera that rides on an arm, and everything needed to put what it sees in the world frame.

    `T_base2world` is identity for the arm whose base IS the world frame, and the measured base-to-base
    transform for any other arm. It is the only thing that distinguishes a second arm's camera from the
    first one's: same maths, one extra multiply. Keeping it here means nothing downstream — find(),
    measure(), can_see() — has to know which arm it is looking through.
    """
    name: str
    calc: Any                                    # HandEyeCalculator: intrinsics + T_cam2gripper
    frame: Callable[[], Any]                     # () -> (rgb, depth) or rgb
    pose: Callable[[], Any]                      # () -> (R_gripper2base, t_gripper2base) or None
    T_base2world: np.ndarray = field(default_factory=lambda: np.eye(4))
    # Which arm this camera rides on. It decides whose picture rides along with a motion result:
    # the wrist that just moved. Not a preference — a camera bolted to the acting arm is the only one
    # that is somewhere new, and therefore the only close-up that can show what the motion did.
    arm: str = "right"
    note: str = ("camera on the moving wrist; close-up and finer per millimetre, but only sees what "
                 "the arm is over, and its own gripper fingers block part of the image")


class CameraRig:
    """Builds View objects from whatever cameras this robot has.

    Single-arm AirBot Play: a fixed head camera and one wrist camera. A bimanual robot adds a second
    wrist camera with its own hand-eye extrinsics and its own arm pose — same code, one more entry.
    """

    def __init__(self, env, head_calc, handeye_calc=None, wrist_name: str = "wrist",
                 wrists: Optional[List[Wrist]] = None):
        self.env = env
        self.head_calc = head_calc
        self.handeye_calc = handeye_calc
        self.wrist_name = wrist_name
        self.wrists: List[Wrist] = list(wrists or [])
        if handeye_calc is not None and hasattr(env, "get_handeye_camera_frame"):
            self.wrists.insert(0, Wrist(name=wrist_name, calc=handeye_calc,
                                        frame=env.get_handeye_camera_frame,
                                        # calibrated against the SDK's frame, not the tool point
                                        pose=(lambda: env.get_arm_pose(tool=False)) if hasattr(env, "get_arm_pose") else (lambda: None)))

    def names(self) -> List[str]:
        return ["head"] + [w.name for w in self.wrists]

    def wrist_of(self, arm: str) -> Optional[str]:
        """The camera riding on this arm, if it has one."""
        for w in self.wrists:
            if w.arm == arm:
                return w.name
        return None

    def default_image_names(self, acting_arm: str = "right") -> List[str]:
        """Which cameras' PICTURES ride along with a motion result nobody asked to look at.

        Not which cameras are used. Every camera is captured every time; find and measure can work
        through any of them, and can_see reports what each one can see of every known object. This
        decides only which ones spend pixels unasked — and a pixel is not free: an image costs about
        1.5k tokens, which is a tenth of a second of latency on that call and on every call after it,
        because it stays in the history.

        Two earn it unconditionally. The head is the only camera that sees the whole scene. The wrist
        of the arm that just moved is the only one that shows, at close range, what the motion did —
        it is somewhere new, so its picture cannot be the one the model already has.

        WHICH wrist that is depends on which arm just moved, and getting this wrong is expensive in a
        way that is hard to see. This used to return a fixed list, chosen when the second arm had no
        motion tools and so never moved. Once it could move, every left-arm action still came back
        with the head view and the RIGHT arm's wrist view — a close-up of an arm parked in mid-air,
        while the camera actually looking at the fingers doing the work was left out. The model spent
        ten calls asking for it by hand, and still released a grasp it could not see.

        Any other wrist is captured just the same, so can_see reports what it sees of every known
        object and measure can use it. It is not pictured unless asked for, because it has not moved
        and its picture is the one the model already has. That is the difference between a third
        camera being more information and being more noise: always reported, pictured on request.
        """
        return ["head"] + [n for n in [self.wrist_of(acting_arm)] if n]

    def grab(self, name: str, step: int = 0) -> Optional[View]:
        if name == "head":
            return self._head(step)
        for w in self.wrists:
            if w.name == name:
                return self._wrist(w, step)
        return None

    def _head(self, step: int) -> Optional[View]:
        rgb, depth = self.env.get_head_camera_frame()
        if rgb is None:
            return None
        T = np.asarray(self.env.get_head_camera_transform(), float)
        hc = self.head_calc
        return View(name="head", rgb=rgb, depth=depth, fx=hc.fx, fy=hc.fy, cx=hc.cx, cy=hc.cy,
                    T_cam2base=T, offset=np.asarray(hc.offset, float), step=step, eye_xyz=list(T[:3, 3]),
                    note="fixed camera above the table; sees the whole workspace at ~0.4 m")

    def _wrist(self, w: Wrist, step: int) -> Optional[View]:
        frame = w.frame()
        if frame is None:
            return None
        rgb, depth = frame if isinstance(frame, tuple) else (frame, None)
        if rgb is None:
            return None
        pose = w.pose()
        if pose is None:
            return None
        R_g2b, t_g2b = pose
        T_g2b = np.eye(4)
        T_g2b[:3, :3] = np.asarray(R_g2b, float)
        T_g2b[:3, 3] = np.asarray(t_g2b, float).flatten()
        # base2world is identity for the arm that defines the world frame; for the other arm it is the
        # measured transform between the two bases, so this camera's points come out in the same frame
        # as everything else without anything downstream knowing.
        T = np.asarray(w.T_base2world, float) @ T_g2b @ w.calc.T_cam2gripper
        hc = w.calc
        return View(name=w.name, rgb=rgb, depth=depth, fx=hc.fx, fy=hc.fy, cx=hc.cx, cy=hc.cy,
                    T_cam2base=T, step=step, eye_xyz=list(T[:3, 3]), self_mask=self_mask_from_depth(depth),
                    note=w.note)
