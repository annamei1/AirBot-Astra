"""
Perception over every camera the robot has, as one service.

Tools exposed to the VLM:
    look(camera)          → a fresh frame from one camera or all of them, known object ids drawn on
    find(label, ...)      → segment in a chosen camera by text, or by a box/point read off the image;
                            register each instance as an id with a metric pose in the base frame
    measure(u, v, camera) → any pixel → 3D point, local surface tilt, relief above its surroundings
    refine(object_id)     → re-measure a known object from whichever camera currently sees it best

Three ideas hold this together.

**One world, many eyes.** Every camera contributes frames carrying their own intrinsics and
camera-to-base transform (see cameras.py). Measurements from any of them land in the same object
table in the same base frame, so "which camera" is a question about evidence quality, never about
coordinates. Adding a second arm's wrist camera adds an entry, not a code path.

**Cameras hand work to each other geometrically, not by name.** Once the head camera has located
something, any other camera can be asked where that point falls in its own image, and SAM3 can be
prompted with that pixel box. Class-agnostic: the close-up never needs the object's name.

**A close-up has to agree with what it is correcting.** First live run on the robot: a wrist view
half-blocked by the gripper fingers segmented a brick plus the table around it; the centroid moved
only 8 mm so it was accepted, the inflated size then drew a bigger box for the next refine, which
segmented a patch of baseplate — accepted again. So a refinement is now judged on three things,
anchored to what find() first measured rather than to whatever the last refine wrote: the centroid
may move a little, the footprint may shrink (occlusion hides part of it) but not grow (that is
spill-over onto the surroundings), and a camera whose view of the object is mostly the robot's own
body — or that is no closer than the camera which measured it — is not asked at all.

**Geometry is reported with its own uncertainty.** Height is measured against the LOCAL SUPPORT — a
ring of pixels just outside the region — not a global table constant. Every height carries a status,
because depth on dark, thin, glossy and deformable surfaces is routinely contaminated: the values
are present but wrong. A region whose interior reads below the surface it rests on is flagged.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from harness import config
from perception.position_calculator import HandEyeCalculator
from harness.cameras import CameraRig, View, Wrist
from harness.arms import SECOND_ARM
from harness.tools import ToolRegistry, ToolResult

# Depth noise floor on a flat surface. A height difference below this is not a measurement.
# Measured on this rig: 1.6 mm std, 3.2 mm p95 over a flat plate at 0.43 m.
DEPTH_NOISE_M = 0.004
# "This region reads below the surface it rests on", which is physically impossible. Tighter than the
# per-pixel noise because both sides are medians over many pixels.
CONTAMINATION_M = 0.002
RING_INNER_PX = 9        # gap between a region and its support ring (mask edges are unreliable)
RING_OUTER_PX = 27       # outer radius of the support ring

# ---- what a refinement must agree with ----
# Head-camera position error is millimetres to about a centimetre; far beyond that the close-up is
# looking at something else.
MAX_REFINE_CORRECTION_M = 0.05
# The object's footprint as first measured by find() is the reference for every later close-up.
# Occlusion — the gripper, the image edge, a neighbour — can only HIDE part of an object, so a genuine
# close-up sees the same footprint or less. Growth means the mask spread onto the surroundings. Second
# offline check: a brick mask 1.79x the area and 1.5x longer on one side, centroid pulled 10 mm onto the
# baseplate, passed the old symmetric 2.5x bound. So shrinking is allowed generously, growing barely.
REFINE_AREA_RATIO = (0.4, 1.35)      # allowed footprint, as a multiple of the find() footprint
REFINE_EXTENT_GROWTH = 1.3           # no side may grow beyond this factor
REFINE_EXTENT_SHRINK = 0.45          # nor shrink below it (that is a fragment, not the object)
# A camera whose projected box is mostly the robot's own body cannot see the object well enough.
MAX_OCCLUDED_FRAC = 0.35

_COLORS = [(0, 200, 0), (255, 120, 0), (0, 0, 255), (200, 200, 0), (255, 0, 255), (0, 160, 255)]

# One measure() call may trace a whole shape. Bounded so a runaway list cannot stall an episode.
MAX_PATH_POINTS = 64
# How many probed pixels the live window keeps showing, per camera.
MAX_REMEMBERED_PROBES = 24


class ReplayEnv:
    """Offline stand-in for PlayRobotEnv: replays one captured frame (scripts/capture_frame.py).

    Head camera only, or head + wrist when the capture also saved the wrist image, the wrist depth and
    the arm pose it was taken at — the pose is what places the wrist view in the base frame.
    """

    def __init__(self, rgb: np.ndarray, depth_mm: np.ndarray, T_head2base: np.ndarray,
                 wrist_rgb: Optional[np.ndarray] = None, wrist_depth_mm: Optional[np.ndarray] = None,
                 arm_pose: Optional[np.ndarray] = None):
        self._rgb, self._depth, self._tf = rgb, depth_mm, T_head2base
        self._wrgb, self._wdepth, self._pose = wrist_rgb, wrist_depth_mm, arm_pose

    @classmethod
    def from_stem(cls, stem: str) -> "ReplayEnv":
        import os

        def load(suffix):
            return np.load(f"{stem}{suffix}") if os.path.exists(f"{stem}{suffix}") else None

        wrist = cv2.imread(f"{stem}_wrist.png") if os.path.exists(f"{stem}_wrist.png") else None
        return cls(cv2.imread(f"{stem}_rgb.png"), load("_depth.npy"), load("_tf.npy"),
                   wrist_rgb=wrist, wrist_depth_mm=load("_wrist_depth.npy"), arm_pose=load("_arm_pose.npy"))

    @property
    def has_wrist(self) -> bool:
        return self._wrgb is not None and self._pose is not None

    def get_head_camera_frame(self):
        return self._rgb.copy(), self._depth.copy()

    def get_head_camera_transform(self):
        return self._tf.copy()

    def get_handeye_camera_frame(self):
        if self._wrgb is None:
            return None, None
        return self._wrgb.copy(), None if self._wdepth is None else self._wdepth.copy()

    def get_arm_pose(self):
        if self._pose is None:
            return None
        return self._pose[:3, :3].copy(), self._pose[:3, 3].copy()


def _second_wrist(env) -> List:
    """The other arm's wrist camera, if this rig has one and the two arms have been tied together.

    Added here rather than at every call site so that nothing downstream — find, measure, can_see —
    needs to know a second arm exists: it is one more name in `cameras()`, and its points come out in
    the same frame as everything else because the Wrist carries the base-to-base transform.

    Silent when any piece is missing. A single-arm rig, or one whose arms have never been calibrated
    against each other, has no second wrist and should not be told about one.
    """
    # Ask the env whether the arm is actually THERE, not whether the methods exist. They live on
    # PlayRobotEnv unconditionally, so a hasattr probe is true on a single-arm rig too and would
    # register a camera that can never return a frame.
    if not getattr(env, "has_second_arm", False):
        return []
    try:
        T_base2world = config.load_base_to_base()
        if T_base2world is None:
            print("[rig] the left wrist camera is there but the two bases have never been tied "
                  "together; run calibration/play/joint_cal.py. Leaving it out.")
            return []
        intr, extr = config.load_left_hand_eye_calibration()
        calc = HandEyeCalculator(intr, extr)
    except Exception as exc:  # noqa: BLE001  a missing second arm must never stop a single-arm run
        print(f"[rig] no left wrist camera ({type(exc).__name__}: {exc})")
        return []
    print(f"[rig] left wrist camera added; its base sits at "
          f"{(T_base2world[:3, 3] * 1000).round(1).tolist()} mm in the world frame")
    return [Wrist(name="left_wrist", calc=calc,
                  frame=env.get_left_handeye_camera_frame, pose=lambda: env.get_left_arm_pose(tool=False),
                  T_base2world=T_base2world,
                  # Which arm it rides on. Its picture rides along with THAT arm's motions, and only
                  # those; while the other arm works it is reported through can_see but not pictured.
                  arm=SECOND_ARM,
                  note="camera on the OTHER arm's wrist. Same close-up detail as the near wrist, but it "
                       "looks from where that arm is, so it can watch this arm work — including the "
                       "moment this arm's own body hides the thing from every other camera. Its picture "
                       "is NOT sent with every motion; can_see tells you every step what it can see, and "
                       "look(camera='left_wrist') is worth a call when can_see says it sees something "
                       "the head and the near wrist do not. Its points carry the ~8 mm between the two "
                       "bases, so measure with it to see, and re-measure with the acting arm's own "
                       "camera before acting on a millimetre.")]


class Perception:
    def __init__(self, env, segmenter, head_calc, table_z: float, handeye_calc=None,
                 max_instances: int = 8):
        self.env = env
        self.segmenter = segmenter
        self.head_calc = head_calc
        self.table_z = float(table_z)     # fallback support height only, when no ring is visible
        self.max_instances = max_instances
        self.rig = CameraRig(env, head_calc, handeye_calc, wrists=_second_wrist(env))
        self.views: Dict[str, View] = {}                  # camera name → latest frame
        self.objects: Dict[str, Dict[str, Any]] = {}      # id → record
        self._masks: Dict[str, Tuple[str, np.ndarray]] = {}   # id → (camera, mask of that frame)
        self._probes: Dict[str, List[Tuple[int, int]]] = {}   # camera → pixels the model has pointed at
        self.step = 0
        self.narrator = None      # a harness.narrator.Narrator, attached by run_pickplace with --narrator

    def new_episode(self):
        """Drop the previous episode's objects. After the scene is reset their ids point at where things
        used to be, and a stale xyz is worse than none."""
        self.objects.clear()
        self._masks.clear()
        self._probes.clear()
        self.step = 0

    def _narration(self) -> Dict[str, Any]:
        """The narrator's account, for results that observe. Empty when there is no narrator."""
        nar = self.narrator
        if nar is None:
            return {}
        # look() is where the whole account appears; every other result carries only the entries
        # added since the model's previous result (Narrator.news). Reading the whole account here
        # also moves the cursor, so the next result does not repeat what this one showed.
        nar.news()
        return {"what_happened": nar.narrative or "(the narrator has not written anything yet)",
                "narrator": nar.status_line()}

    # ================================================================ frames

    def cameras(self) -> List[str]:
        return self.rig.names()

    def capture(self, camera: str = "head") -> Optional[View]:
        self.step += 1
        v = self.rig.grab(camera, self.step)
        if v is not None:
            self.views[camera] = v
        return v

    def capture_all(self) -> List[View]:
        self.step += 1
        out = []
        for name in self.cameras():
            v = self.rig.grab(name, self.step)
            if v is not None:
                self.views[name] = v
                out.append(v)
        return out

    def view(self, camera: str) -> Optional[View]:
        return self.views.get(camera)

    @property
    def last_rgb(self) -> Optional[np.ndarray]:
        v = self.views.get("head")
        return None if v is None else v.rgb

    @property
    def last_depth(self) -> Optional[np.ndarray]:
        v = self.views.get("head")
        return None if v is None else v.depth

    def wrist_image(self) -> Optional[np.ndarray]:
        v = self.capture("wrist")
        return None if v is None else v.rgb.copy()

    # ================================================================ drawing

    @staticmethod
    def visibility_lines(seen: Dict[str, Any]) -> Dict[str, str]:
        """visibility() as one line per object: the same numbers, a third of the characters.

        The nested form was 30% of everything the model read in a five-minute episode, most of it
        JSON punctuation and key names repeated for every object and camera on every motion.
            "red_handle_1": "head 100% (0.32 m, 0.82 mm/px) | wrist 46% (0.16 m, 0.41 mm/px; the robot's own body covers part of it)"
        """
        out: Dict[str, str] = {}
        for oid, cams in seen.items():
            parts = []
            for cam, e in cams.items():
                vis = e.get("visible", 0.0)
                if e.get("range_m") is None:
                    parts.append(f"{cam} 0% ({e.get('why', 'not visible')})")
                    continue
                detail = f"{e['range_m']:.2f} m, {e['mm_per_px']:.2f} mm/px" + (f"; {e['why']}" if e.get("why") else "")
                parts.append(f"{cam} {vis * 100:.0f}% ({detail})")
            out[oid] = " | ".join(parts)
        return out

    def visibility(self, object_id: Optional[str] = None) -> Dict[str, Any]:
        """How well each camera can see each known object right now, from the latest frames held here.

        Occlusion has only ever surfaced as a reason for refusing something: refine says "61% of the
        object is hidden behind the robot itself", measure says "that pixel is the gripper". Across
        six real episodes those refusals landed seven times and refine never once succeeded, because
        **the arm is its own occluder and it occludes most at exactly the moment it is about to act**.
        A model told this only when it fails cannot plan around it, and every workaround it invents
        costs a round trip. So visibility is a fact reported alongside every motion now, not a gate
        that fires silently.

        Nothing here is new machinery: the self-mask, the projection and the occluded fraction are
        the same ones the refine gate has always used. They were simply never shown to anyone.
        """
        out: Dict[str, Any] = {}
        for oid in ([object_id] if object_id else list(self.objects)):
            rec = self.objects.get(oid)
            if rec is None or not rec.get("xyz"):
                continue
            per_camera: Dict[str, Any] = {}
            for name, view in self.views.items():
                pr = view.project(rec["xyz"])
                if pr is None:
                    per_camera[name] = {"visible": 0.0, "why": "behind this camera"}
                    continue
                u, v, rng = pr
                if not view.contains(u, v):
                    per_camera[name] = {"visible": 0.0, "why": "outside this camera's frame"}
                    continue
                extent = max(rec.get("extent_m") or [0.03])
                box = view.box_around(rec["xyz"], float(extent))
                blocked = view.occluded_fraction(box) if box else 0.0
                entry: Dict[str, Any] = {"visible": round(1.0 - float(blocked), 2),
                                         "range_m": round(float(rng), 3),
                                         "mm_per_px": round(1000.0 / view.px_per_m(float(rng)), 2)}
                # Wording has to match the number. Reusing refine's 0.35 gate here put "mostly the
                # robot's own body" on a view that could see 64% of the object, which reads as "this
                # camera is useless" and is the opposite of true.
                if blocked > 0.65:
                    entry["why"] = "mostly the robot's own body from here"
                elif blocked > 0.3:
                    entry["why"] = "the robot's own body covers part of it from here"
                per_camera[name] = entry
            if per_camera:
                out[oid] = per_camera
        return out

    def overlay(self, camera: str = "head", view: Optional[View] = None,
                masks: bool = True) -> Optional[np.ndarray]:
        """A camera's latest frame with every object drawn on it: its own mask where the object was
        segmented in this very camera, otherwise its position projected in from the base frame.
        Robot pixels (the gripper, in a wrist view) are hatched so it is obvious they were ignored."""
        v = view if view is not None else self.views.get(camera)
        if v is None:
            return None
        img = v.rgb.copy()
        if v.self_mask is not None and v.self_mask.any():
            hatch = np.zeros(v.self_mask.shape, bool)
            hatch[::6, :] = True
            hatch[:, ::6] = True
            img[v.self_mask & hatch] = (90, 90, 90)
        # A snapshot: the live-view thread draws this while the agent thread is adding objects, and
        # iterating the dict itself raises "dictionary changed size during iteration" mid-episode.
        for i, (oid, rec) in enumerate(list(self.objects.items())):
            color = _COLORS[i % len(_COLORS)]
            cam, mask = self._masks.get(oid, (None, None))
            if masks and cam == camera and mask is not None and mask.shape == img.shape[:2]:
                colored = np.zeros_like(img)
                colored[mask > 0] = color
                img = cv2.addWeighted(img, 1.0, colored, 0.35, 0)
                u, v_px = rec["pixel"]
            else:
                pr = v.project(rec["xyz"])
                if pr is None or not v.contains(pr[0], pr[1]):
                    continue
                u, v_px = pr[0], pr[1]
                cv2.drawMarker(img, (int(u), int(v_px)), color, cv2.MARKER_TILTED_CROSS, 14, 2)
            cv2.circle(img, (int(u), int(v_px)), 4, color, -1)
            cv2.putText(img, oid, (int(u) + 6, int(v_px) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        # Where the model has been pointing. Faint, and older probes fade, so the live window shows the
        # trace without competing with the objects drawn above it.
        probes = self._probes.get(camera) or []
        for k, (pu, pv) in enumerate(probes):
            age = (k + 1) / len(probes)
            cv2.circle(img, (int(pu), int(pv)), 4, (0, int(120 + 100 * age), int(160 + 95 * age)), 1,
                       cv2.LINE_AA)
        for a, b in zip(probes, probes[1:]):
            cv2.line(img, a, b, (0, 170, 210), 1, cv2.LINE_AA)
        cv2.putText(img, f"[{camera}] frame {v.step}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        return img

    def _shots(self, cameras: List[str], caption: str) -> List[Tuple[str, np.ndarray]]:
        out = []
        for c in cameras:
            img = self.overlay(c)
            if img is not None:
                out.append((f"[{c}] {caption} {self.views[c].note}", img))
        return out

    # ================================================================ tools

    def look(self, camera: str = "all") -> ToolResult:
        wanted = self.cameras() if camera in ("all", "", None) else [camera]
        bad = [c for c in wanted if c not in self.cameras()]
        if bad:
            return ToolResult.error(f"no camera named {bad}. Available: {self.cameras()}")
        got = [c for c in wanted if (self.capture(c) is not None)]
        if not got:
            return ToolResult.error(f"no frame from {wanted}")
        return ToolResult.json(
            {"ok": True, "frame": self.step, "cameras": [self.views[c].describe() for c in got],
             "known_objects": [self._public(r) for r in self.objects.values()],
             **self._narration(),
             "note": "object positions are from the frame each was last measured in; call find() or "
                     "refine() again after anything moved. Hatched pixels in a wrist image are the "
                     "gripper itself and are never used as evidence."},
            images=self._shots(got, "current view."))

    def find(self, label: str, camera: str = "head", box: Optional[List[int]] = None,
             point: Optional[List[int]] = None) -> ToolResult:
        """Segment by text, or — when the text encoder cannot resolve the thing — by a box or point
        read off the image. A geometric prompt is class agnostic: it does not need a name."""
        label = (label or "").strip()
        if not label:
            return ToolResult.error("label is empty (it names the ids; add a box or point if text fails)")
        if camera not in self.cameras():
            return ToolResult.error(f"no camera named '{camera}'. Available: {self.cameras()}")
        v = self.capture(camera)
        if v is None:
            return ToolResult.error(f"no frame from the {camera} camera")
        if v.depth is None:
            return ToolResult.error(f"the {camera} camera has no depth; use it through refine() instead")
        h, w = v.shape

        # A box says more than a point, so it wins when both are given.
        box_xyxy = None
        if box is not None and len(box) == 4:
            box_xyxy = tuple(int(b) for b in box)
        elif point is not None and len(point) == 2:
            u, p = int(point[0]), int(point[1])
            if not v.contains(u, p):
                return ToolResult.error(f"point [{u}, {p}] is outside the {w}x{h} {camera} image")
            box_xyxy = (u - 18, p - 18, u + 18, p + 18)
        if box_xyxy is not None:
            x0, y0, x1, y1 = box_xyxy
            if not (0 <= min(x0, x1) and max(x0, x1) < w and 0 <= min(y0, y1) and max(y0, y1) < h):
                return ToolResult.error(f"box {list(box_xyxy)} falls outside the {w}x{h} {camera} image")
            masks = self._pick_by_box(self.segmenter.segment_box(v.rgb, box_xyxy, prompt=None), box_xyxy)
            how = f"{camera}: box {list(box_xyxy)}"
        else:
            masks = self.segmenter.segment(v.rgb, label)
            how = f"{camera}: text '{label}'"

        if masks is None or len(masks) == 0:
            hint = ("SAM3 found nothing for this text. Try different wording ONCE, then call find() again "
                    "with box=[x0,y0,x1,y1] around the object — a geometric prompt needs no name and works "
                    "on things the text encoder does not know."
                    if box_xyxy is None else
                    "Nothing segmented inside that box. Check the pixel coordinates against the image, "
                    "or draw a tighter box around the object alone.")
            return ToolResult.json({"ok": True, "label": label, "prompt_used": how, "objects": [], "hint": hint},
                                   images=self._shots([camera], "nothing segmented."))
        if masks.ndim == 2:
            masks = masks[None]

        for oid in [k for k, r in self.objects.items() if r["label"] == label]:
            self.objects.pop(oid)
            self._masks.pop(oid, None)

        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "obj"
        found: List[Dict[str, Any]] = []
        for i in range(min(masks.shape[0], self.max_instances)):
            m = self._as_mask(masks[i], v)
            if m is None:
                continue
            rec = self._measure_region(f"{slug}_{i + 1}", label, m, v)
            if rec is None:
                continue
            # find() is the reference measurement: later refinements are judged against it
            rec["extent_ref_m"] = list(rec["extent_m"])
            rec["area_ref_cm2"] = rec["area_cm2"]
            self.objects[rec["id"]] = rec
            self._masks[rec["id"]] = (camera, m)
            found.append(self._public(rec))

        found.sort(key=lambda o: o["id"])
        return ToolResult.json(
            {"ok": True, "label": label, "prompt_used": how, "frame": v.step, "objects": found},
            images=self._shots([camera], f"objects found for '{label}' are drawn on it — check each id "
                                         f"sits on the right thing."))

    def refine(self, object_id: str, camera: str = "auto") -> ToolResult:
        """RETIRED as a tool on 2026-09-16: refused 12 times across six real episodes, accepted none.

        Its premise defeats itself. It exists to get a closer look, and the only way for the wrist
        camera to get closer is to put the gripper between the camera and the object, so the closest
        view is reliably the most blocked one. The model solved this itself every single time by
        moving somewhere it could see and pointing at a pixel, which is now move_above/tilt plus `measure`.

        Kept here, unregistered, because smoke_cameras_live still exercises it offline and because the
        agreement gate it carries is the only written record of what a bad close-up looks like.
        """
        """Re-measure a known object from the camera that can see it best right now.

        The object's stored 3D position is projected into each candidate camera to place a box
        prompt, so the close-up never needs the object's name. A candidate is only accepted when it
        agrees with the object find() measured: centroid within MAX_REFINE_CORRECTION_M, footprint
        between REFINE_AREA_RATIO and no side grown past REFINE_EXTENT_GROWTH. Views mostly blocked by
        the robot, mostly outside the image, or no closer than the measuring camera are skipped.
        """
        rec = self.objects.get(object_id)
        if rec is None:
            return ToolResult.error(f"unknown object id '{object_id}'. Known: {list(self.objects)}")
        candidates = self.cameras() if camera in ("auto", "", None) else [camera]
        if any(c not in self.cameras() for c in candidates):
            return ToolResult.error(f"no camera named '{camera}'. Available: {self.cameras()}")

        extent_ref = rec.get("extent_ref_m") or rec.get("extent_m") or [0.05, 0.05]
        area_ref = rec.get("area_ref_cm2") or rec.get("area_cm2")
        before = np.asarray(rec["xyz"], float)
        closer_factor = 0.9      # a close-up must be at least this much closer than the measuring camera
        min_in_view = 0.7        # and have at least this much of the object's footprint inside its image

        def inside_fraction(v: View, rng: float, u: float, p: float) -> float:
            half = 0.5 * max(max(extent_ref), 0.01) * v.px_per_m(rng)
            h, w = v.shape
            iw = max(0.0, min(u + half, w) - max(u - half, 0.0))
            ih = max(0.0, min(p + half, h) - max(p - half, 0.0))
            return float(iw * ih / max((2 * half) ** 2, 1e-6))

        # Re-segmenting in the camera that already measured the object, with a projected box, is not a
        # refinement but a worse prompt (offline check: 2.8x the footprint). Only a camera that sees the
        # object from closer adds resolution. An explicitly requested camera is still honoured.
        src_range = None
        if camera in ("auto", "", None):
            src = self.views.get(rec.get("camera")) or self.capture(rec.get("camera") or "head")
            spr = src.project(rec["xyz"]) if src is not None else None
            src_range = spr[2] if spr is not None else None

        tried, accepted, rejected = [], [], []
        for name in candidates:
            v = self.capture(name)
            if v is None or v.depth is None:
                tried.append({"camera": name, "why": "no frame or no depth"})
                continue
            pr = v.project(rec["xyz"])
            if pr is None or not v.contains(pr[0], pr[1], margin=6):
                tried.append({"camera": name, "why": "the object does not fall inside this image"})
                continue
            u, p, rng = pr
            if src_range is not None and rng >= closer_factor * src_range:
                tried.append({"camera": name, "why": f"no closer than the {rec.get('camera')} camera that "
                                                     f"measured it ({rng:.2f} m vs {src_range:.2f} m)"})
                continue
            inside = inside_fraction(v, rng, u, p)
            if inside < min_in_view:
                tried.append({"camera": name, "why": f"only {inside:.0%} of the object lies inside this image"})
                continue
            # occlusion is judged on the object's own footprint, not on the padded prompt box around it
            occ = v.occluded_fraction(v.box_around(rec["xyz"], max(extent_ref), pad=1.0))
            if occ > MAX_OCCLUDED_FRAC:
                tried.append({"camera": name, "why": f"{occ:.0%} of the object is hidden behind the robot "
                                                     f"itself (gripper) in this view"})
                continue
            bx = v.box_around(rec["xyz"], max(extent_ref))
            px = v.px_per_m(rng)
            expected_px = extent_ref[0] * px * extent_ref[1] * px
            masks = self._pick_by_box(self.segmenter.segment_box(v.rgb, bx, prompt=None), bx,
                                      expected_area_px=expected_px)
            if masks is None or len(masks) == 0:
                tried.append({"camera": name, "why": f"nothing of the expected size segmented at "
                                                     f"[{u:.0f}, {p:.0f}]"})
                continue
            m = self._as_mask(masks[0] if masks.ndim > 2 else masks, v)
            if m is None:
                tried.append({"camera": name, "why": "mask was all robot pixels"})
                continue
            cand = self._measure_region(object_id, rec["label"], m, v)
            if cand is None:
                tried.append({"camera": name, "why": "no usable depth inside the mask"})
                continue
            cand["range_m"] = round(rng, 4)
            cand["occluded_frac"] = round(occ, 3)
            verdict = self._agrees(cand, before, extent_ref, area_ref)
            entry = {"camera": name, "range_m": cand["range_m"], "box": list(bx), "mask": m, "cand": cand,
                     **verdict}
            (accepted if verdict["agrees"] else rejected).append(entry)

        if accepted:
            best = min(accepted, key=lambda e: e["range_m"])   # closer camera → finer mm per pixel
            cand = best["cand"]
            # the reference stays what find() measured; a refinement never redefines the object's size
            cand["extent_ref_m"], cand["area_ref_cm2"] = list(extent_ref), area_ref
            self.objects[object_id] = cand
            self._masks[object_id] = (best["camera"], best["mask"])
            return ToolResult.json(
                {"ok": True, "object_id": object_id, "camera": best["camera"], "range_m": best["range_m"],
                 "prompt_used": f"{best['camera']}: projected box {best['box']}",
                 "previous_xyz": [round(float(c), 4) for c in before],
                 "correction_m": best["correction_m"], "area_ratio": best["area_ratio"],
                 "object": self._public(cand),
                 "other_cameras": tried + [self._summary(e) for e in accepted + rejected if e is not best],
                 "note": "measured from the closest camera that agreed with the original measurement; "
                         "this pose supersedes the previous one"},
                images=self._shots([best["camera"]], f"{object_id} re-measured here."))

        if rejected:
            worst = min(rejected, key=lambda e: e["range_m"])
            return ToolResult.json(
                {"ok": False, "object_id": object_id, "kept_previous": True, "disagreement": True,
                 "error": f"no camera's close-up agreed with the original measurement of {object_id}: "
                          + "; ".join(f"{e['camera']}: {e['why']}" for e in rejected),
                 "attempts": tried + [self._summary(e) for e in rejected],
                 "candidate": self._public(worst["cand"]),
                 "hint": "The stored pose is unchanged. Compare the images: if a close-up locked onto the "
                         "table or a fragment, the stored pose is still the best evidence. If the object "
                         "really moved, find() it afresh — that replaces the record."},
                images=self._shots(sorted({e["camera"] for e in rejected} | {"head"} & set(self.views)),
                                   f"{object_id}: close-up disagreed and was not applied."))

        # Being blocked by the robot's own gripper is not the same problem as nothing being able to see
        # it, and the advice differs. It happens exactly when the arm is close enough to grasp, so the
        # model is about to close the fingers on something no camera can currently check. Moving the arm
        # "over it" is useless then: the thing in the way travels with the arm.
        self_blocked = [a for a in tried if "behind the robot itself" in a.get("why", "")]
        if self_blocked:
            hint = ("The only close enough view is blocked by the gripper itself, which is what happens "
                    "right before a grasp. You are about to act on a position no camera can check now. "
                    "Moving the arm will not help — what is in the way moves with it. Either approach from "
                    "a direction that leaves the object beside the fingers rather than behind them, or "
                    "commit to the head camera's measurement and choose the exact spot deliberately: "
                    "measure(u, v) on the part you mean to pinch, rather than the centroid find() gave you.")
        else:
            hint = ("No camera could see it where it is expected. Move the arm over it and refine() "
                    "again, or find() it afresh in the head camera.")
        return ToolResult.json(
            {"ok": False, "object_id": object_id, "kept_previous": True, "attempts": tried,
             "blocked_by_own_gripper": bool(self_blocked), "hint": hint},
            images=self._shots([c for c in candidates if c in self.views], "refine found no usable view."))

    @staticmethod
    def _agrees(cand: Dict[str, Any], before: np.ndarray, extent_ref, area_ref) -> Dict[str, Any]:
        moved = float(np.linalg.norm(np.asarray(cand["xyz"], float) - before))
        area_ratio = (cand["area_cm2"] / area_ref) if area_ref else 1.0
        ratios = [c / max(r, 1e-4) for c, r in zip(sorted(cand["extent_m"]), sorted(extent_ref))]
        grow, shrink = max(ratios), min(ratios)
        why = []
        if moved > MAX_REFINE_CORRECTION_M:
            why.append(f"centroid moved {moved * 1000:.0f} mm (limit {MAX_REFINE_CORRECTION_M * 1000:.0f})")
        if area_ratio > REFINE_AREA_RATIO[1]:
            why.append(f"footprint grew to {area_ratio:.2f}× the original (limit {REFINE_AREA_RATIO[1]}×) — "
                       f"the mask spread onto the surroundings; occlusion can only make it smaller")
        elif area_ratio < REFINE_AREA_RATIO[0]:
            why.append(f"footprint shrank to {area_ratio:.2f}× the original (limit {REFINE_AREA_RATIO[0]}×) — "
                       f"too little of the object is visible to measure it")
        if grow > REFINE_EXTENT_GROWTH:
            why.append(f"a side grew to {grow:.2f}× the original (limit {REFINE_EXTENT_GROWTH}×)")
        if shrink < REFINE_EXTENT_SHRINK:
            why.append(f"a side shrank to {shrink:.2f}× the original (limit {REFINE_EXTENT_SHRINK}×)")
        return {"agrees": not why, "why": "; ".join(why) or "consistent",
                "correction_m": round(moved, 4), "area_ratio": round(area_ratio, 2)}

    @staticmethod
    def _summary(e: Dict[str, Any]) -> Dict[str, Any]:
        return {"camera": e["camera"], "range_m": e["range_m"], "correction_m": e["correction_m"],
                "area_ratio": e["area_ratio"], "why": e["why"]}

    def measure(self, u: Optional[int] = None, v: Optional[int] = None, camera: str = "head",
                radius: int = 4, points: Optional[List[List[int]]] = None) -> ToolResult:
        """Any pixel → 3D point. One pixel, or a whole ordered list of them.

        The plural form is the one that matters for anything that follows a shape. A drawing on a board,
        the edge of a folded cloth, the rim of a container: the model can see it and trace it in the
        image long before the arm is anywhere near, and the list of base-frame points that comes back
        stays true after the arm has moved in and blocked the view. That is the whole trick for working
        on something you cannot see while you work on it — capture it as coordinates while you can see
        it, act on the coordinates, step back to check.
        """
        view = self.views.get(camera) or self.capture(camera)
        if view is None:
            return ToolResult.error(f"no frame from the {camera} camera — call look() first")
        if view.depth is None:
            return ToolResult.error(f"the {camera} camera has no depth")
        radius = int(max(2, min(20, radius)))
        if points:
            return self._measure_path(view, camera, points, radius)
        if u is None or v is None:
            return ToolResult.error("give u and v for one pixel, or points=[[u, v], ...] for an ordered list")

        h, w = view.shape
        u, v = int(u), int(v)
        if not view.contains(u, v):
            return ToolResult.error(f"pixel [{u}, {v}] is outside the {w}x{h} {camera} image")
        if view.self_mask is not None and view.self_mask[v, u]:
            return ToolResult.error(f"pixel [{u}, {v}] in the {camera} image shows the robot's own gripper, "
                                    f"not the scene")
        patch = np.zeros((h, w), bool)
        patch[max(0, v - radius):v + radius + 1, max(0, u - radius):u + radius + 1] = True
        pts, valid_frac = view.backproject(patch)
        if pts is None:
            return ToolResult.error(f"no valid depth within {radius} px of [{u}, {v}] in the {camera} image")
        self._remember_probe(camera, [(u, v)])

        r2 = max(radius, 28)
        wide = np.zeros((h, w), bool)
        wide[max(0, v - r2):v + r2 + 1, max(0, u - r2):u + r2 + 1] = True
        wide_pts, _ = view.backproject(wide)
        relief, status, support, _, floor = self._height_above_support(patch, pts, view)
        out = {"ok": True, "camera": camera, "pixel": [u, v], "radius_px": radius,
               "xyz": [round(float(x), 4) for x in pts.mean(0)],
               "surface_tilt_deg": self._normal_deg(wide_pts) if wide_pts is not None else None,
               "relief_m": relief, "relief_status": status, "relief_floor_m": round(float(floor), 4),
               "surroundings_z_m": support,
               "depth_valid_frac": round(valid_frac, 2),
               "z_spread_m": round(float(np.percentile(pts[:, 2], 75) - np.percentile(pts[:, 2], 25)), 4),
               "image_size": [w, h]}
        out["note"] = {
            "measured": "this spot stands clear of the surface immediately around it — an edge, rim or step",
            # Two different facts used to share one sentence. On a flat surface the floor is the depth
            # noise and "level" is true. At an edge the ring around the spot lies half on the thing and
            # half off it, the floor becomes three times that spread, and an 82 mm difference was
            # reported as "level with what surrounds it (< 4 mm)".
            "below_noise": (f"level with what surrounds it (< {DEPTH_NOISE_M * 1000:.0f} mm) — normal in the "
                            f"middle of any flat surface, and for one flat layer of cloth")
                           if floor <= DEPTH_NOISE_M else
                           (f"no verdict: what surrounds this spot is not one surface — its heights spread "
                            f"enough that only a difference above {floor * 1000:.0f} mm would count, and "
                            f"this one is {(relief or 0.0) * 1000:.0f} mm. That is what the edge of something "
                            f"looks like. xyz is still the measured position of the spot itself."),
            "unreliable": "the depth here reads BELOW the surface around it, which cannot be true: this "
                          "material (dark, glossy, thin or transparent) defeats the depth camera. Use the "
                          "image, or probe a nearby edge instead.",
            "no_support_visible": "nothing valid around this spot to compare against",
        }[status]
        return ToolResult.json(out, image=self._draw_probes(view, [(u, v)]),
                               caption=f"[{camera}] the spot you measured, marked.")

    def _measure_path(self, view: View, camera: str, points: List[List[int]], radius: int) -> ToolResult:
        """An ordered list of pixels → an ordered list of base-frame points, with per-point honesty."""
        h, w = view.shape
        if len(points) > MAX_PATH_POINTS:
            return ToolResult.error(f"{len(points)} points is more than the {MAX_PATH_POINTS} this takes "
                                    f"at once; send the shape in pieces")
        out: List[Dict[str, Any]] = []
        good: List[np.ndarray] = []
        for i, pt in enumerate(points):
            if len(pt) != 2:
                out.append({"i": i, "ok": False, "why": "not a [u, v] pair"})
                continue
            pu, pv = int(pt[0]), int(pt[1])
            entry: Dict[str, Any] = {"i": i, "px": [pu, pv]}
            if not view.contains(pu, pv):
                entry.update(ok=False, why=f"outside the {w}x{h} image")
            elif view.self_mask is not None and view.self_mask[pv, pu]:
                entry.update(ok=False, why="this pixel is the robot's own gripper, not the scene")
            else:
                patch = np.zeros((h, w), bool)
                patch[max(0, pv - radius):pv + radius + 1, max(0, pu - radius):pu + radius + 1] = True
                pts, valid = view.backproject(patch)
                if pts is None:
                    entry.update(ok=False, why="no valid depth here")
                else:
                    xyz = pts.mean(0)
                    good.append(xyz)
                    entry.update(ok=True, xyz=[round(float(c), 4) for c in xyz],
                                 depth_valid_frac=round(valid, 2),
                                 z_spread_m=round(float(np.percentile(pts[:, 2], 75)
                                                        - np.percentile(pts[:, 2], 25)), 4))
            out.append(entry)
        self._remember_probe(camera, [(int(p[0]), int(p[1])) for p in points if len(p) == 2])
        length = float(sum(float(np.linalg.norm(b - a)) for a, b in zip(good, good[1:]))) if len(good) > 1 else 0.0
        payload = {"ok": bool(good), "camera": camera, "asked": len(points), "measured": len(good),
                   "radius_px": radius, "path_length_m": round(length, 4), "points": out}
        if not good:
            payload["error"] = "not one of those pixels gave a depth reading"
        elif len(good) < len(points):
            payload["note"] = ("some pixels gave nothing, and they are marked in the list and crossed out in "
                               "the image. The points that did measure are still an ordered path; the gaps "
                               "are where the depth camera had nothing to say.")
        else:
            payload["note"] = ("these are base-frame coordinates now, so they stay true after the arm moves "
                               "in and blocks the view. follow_path can travel through them.")
        return ToolResult.json(payload, image=self._draw_probes(view, [(int(p[0]), int(p[1]))
                                                                      for p in points if len(p) == 2],
                                                               ok=[e.get("ok", False) for e in out]),
                               caption=f"[{camera}] the {len(points)} pixels you traced, in order.")

    def lowest_visible_surface(self, camera: str = "head",
                               bounds_min=None, bounds_max=None) -> Optional[Dict[str, Any]]:
        """The lowest base-frame z the camera can see inside the region the arm can reach.

        The safety floor has to come from somewhere. A constant tuned against one table stops being a
        safety limit the moment the table changes: it silently becomes a task parameter that has to be
        re-measured for every new surface, which is exactly what this harness exists to avoid. What a
        camera can see it can see on any table, so the floor is derived from the scene each session
        instead of being configured.

        Restricted to the workspace footprint on purpose. The head camera also sees the room past the
        table edge, and the lowest thing in the whole image would put the floor somewhere the arm can
        never go, which protects nothing.
        """
        view = self.views.get(camera)
        if view is None:
            return None
        h, w = view.shape
        mask = np.ones((h, w), bool)
        if view.self_mask is not None:
            mask &= ~view.self_mask          # the arm is not a surface
        pts, valid = view.backproject(mask)
        if pts is None or len(pts) < 200:
            return None
        if bounds_min is not None and bounds_max is not None:
            inside = np.ones(len(pts), bool)
            for k in range(2):               # x and y only; z is what we are measuring
                inside &= (pts[:, k] >= float(bounds_min[k])) & (pts[:, k] <= float(bounds_max[k]))
            if inside.sum() < 200:
                return None
            pts = pts[inside]
        z = pts[:, 2]
        return {"camera": camera, "n": int(len(z)), "valid_frac": round(float(valid), 2),
                "low": round(float(np.percentile(z, 2)), 4),
                "median": round(float(np.median(z)), 4),
                "high": round(float(np.percentile(z, 98)), 4)}

    def _remember_probe(self, camera: str, pixels: List[Tuple[int, int]]):
        """Keep the last probed pixels so the live window can draw where the model has been pointing."""
        kept = self._probes.setdefault(camera, [])
        kept.extend(pixels)
        del kept[:-MAX_REMEMBERED_PROBES]

    @staticmethod
    def _draw_probes(view: View, pixels: List[Tuple[int, int]],
                     ok: Optional[List[bool]] = None) -> Optional[np.ndarray]:
        """The camera image with the probed pixels marked, numbered and joined in order."""
        if view.rgb is None or not pixels:
            return None
        img = view.rgb.copy()
        flags = ok if ok is not None else [True] * len(pixels)
        usable = [p for p, f in zip(pixels, flags) if f]
        for a, b in zip(usable, usable[1:]):
            cv2.line(img, a, b, (0, 220, 255), 1, cv2.LINE_AA)
        for i, ((pu, pv), f) in enumerate(zip(pixels, flags)):
            colour = (0, 220, 255) if f else (60, 60, 235)
            cv2.circle(img, (pu, pv), 5, colour, 2, cv2.LINE_AA)
            if not f:
                cv2.line(img, (pu - 7, pv - 7), (pu + 7, pv + 7), colour, 2, cv2.LINE_AA)
                cv2.line(img, (pu - 7, pv + 7), (pu + 7, pv - 7), colour, 2, cv2.LINE_AA)
            if len(pixels) > 1:
                cv2.putText(img, str(i), (pu + 8, pv - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            colour, 1, cv2.LINE_AA)
        return img

    # ================================================================ geometry

    @staticmethod
    def _as_mask(m: np.ndarray, view: View) -> Optional[np.ndarray]:
        while m.ndim > 2:
            m = m[0]
        m = (m > 0.5).astype(np.uint8)
        h, w = view.shape
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m = view.scene(m).astype(np.uint8)              # never let the gripper count as the object
        return None if not m.any() else m

    @staticmethod
    def _pick_by_box(masks, box_xyxy, expected_area_px: Optional[float] = None) -> Optional[np.ndarray]:
        """A geometric prompt can still return several masks: keep the one the box actually meant.

        With an expected pixel area (known from the object's size and the camera range), masks far
        larger or smaller than the object are discarded outright — that is the table around a small
        object, or a fragment of a large one — and the rest are ranked by box overlap and size fit.
        """
        if masks is None or len(masks) == 0:
            return masks
        if masks.ndim == 2:
            masks = masks[None]
        x0, y0, x1, y1 = box_xyxy
        best, best_score = None, -1.0
        for i in range(masks.shape[0]):
            m = masks[i]
            while m.ndim > 2:
                m = m[0]
            m = m > 0.5
            if not m.any():
                continue
            area = float(m.sum())
            size_fit = 1.0
            if expected_area_px:
                ratio = area / expected_area_px
                if not (0.25 <= ratio <= 1.6):      # occlusion shrinks a mask; growth is spill-over
                    continue
                size_fit = min(ratio, 1.0 / ratio)
            ys, xs = np.where(m)
            inter = (max(0, min(x1, xs.max()) - max(x0, xs.min()))
                     * max(0, min(y1, ys.max()) - max(y0, ys.min())))
            union = ((x1 - x0) * (y1 - y0) + (xs.max() - xs.min()) * (ys.max() - ys.min()) - inter)
            iou = inter / union if union > 0 else 0.0
            score = iou * size_fit if expected_area_px else iou
            if score > best_score:
                best, best_score = m, score
        return None if best is None else best[None].astype(np.uint8)

    def _measure_region(self, oid: str, label: str, mask: np.ndarray, view: View) -> Optional[Dict[str, Any]]:
        """One segmented region in one camera → a full metric record in the base frame."""
        mask_bool = view.scene(mask > 0)
        pts, valid_frac = view.backproject(mask_bool)
        if pts is None:
            return None
        ys, xs = np.where(mask_bool)
        d = view.depth[mask_bool]
        d = d[(d > 100) & (d < 2000)]
        if d.size == 0:
            return None
        depth_m = float(np.median(d)) / 1000.0
        extent = [round(float((ys.max() - ys.min()) * depth_m / view.fy), 4),
                  round(float((xs.max() - xs.min()) * depth_m / view.fx), 4)]
        area_cm2 = float(mask_bool.sum()) * (depth_m / view.fx) * (depth_m / view.fy) * 1e4

        from perception.sam3_segmenter import estimate_orientation_from_mask
        yaw_cam = float(estimate_orientation_from_mask(mask_bool))
        yaw = self._wrap(yaw_cam + float(np.arctan2(view.T_cam2base[1, 0], view.T_cam2base[0, 0])) + np.pi)

        centroid = view.pixel_to_base(float(xs.mean()), float(ys.mean()), depth_m)
        top, status, support, bulk, _ = self._height_above_support(mask_bool, pts, view)
        return {"id": oid, "label": label, "xyz": [round(float(c), 4) for c in centroid], "yaw": yaw,
                "area_cm2": round(area_cm2, 1), "extent_m": extent,
                "height_above_support_m": top, "height_status": status,
                "bulk_above_support_m": bulk, "support_z_m": support,
                "depth_valid_frac": round(valid_frac, 2),
                "z_spread_m": round(float(np.percentile(pts[:, 2], 75) - np.percentile(pts[:, 2], 25)), 4),
                "surface_tilt_deg": self._normal_deg(pts),
                "camera": view.name, "pixel": (float(xs.mean()), float(ys.mean())),
                "frame": view.step, "t": time.time()}

    @staticmethod
    def _wrap(a: float) -> float:
        return float((a + np.pi) % (2 * np.pi) - np.pi)

    @staticmethod
    def _normal_deg(pts: Optional[np.ndarray]) -> Optional[float]:
        """Tilt of the local surface from horizontal, or None when it cannot be resolved.

        A plane fitted to a small patch is dominated by depth noise: the angular uncertainty is about
        atan(residual / lateral extent). Return None rather than a confident-looking number built
        from noise.
        """
        if pts is None or len(pts) < 24:
            return None
        if len(pts) > 4000:                      # a plane needs thousands of points, not the whole mask
            pts = pts[np.random.default_rng(0).choice(len(pts), 4000, replace=False)]
        c = pts.mean(0)
        d = pts - c
        # full_matrices=False: the default builds an N×N matrix — tens of GB for a close-up of a bowl
        _, _, V = np.linalg.svd(d, full_matrices=False)
        n = V[2] / (np.linalg.norm(V[2]) + 1e-12)
        angle = float(np.degrees(np.arccos(min(1.0, abs(n[2])))))
        resid = float(np.abs(d @ n).std())
        extent = float(np.percentile(np.linalg.norm(d[:, :2], axis=1), 90))
        if extent < 1e-4:
            return None
        uncertainty = float(np.degrees(np.arctan2(resid, extent)))
        if uncertainty > 5.0 or uncertainty > 0.5 * max(angle, 1e-6):
            return None
        return round(angle, 1)

    def _support_ring(self, mask: np.ndarray, view: View) -> Tuple[Optional[float], Optional[float]]:
        """Median height and spread of the surface immediately around a region — its local support."""
        m = mask.astype(np.uint8)
        inner = cv2.dilate(m, np.ones((RING_INNER_PX, RING_INNER_PX), np.uint8))
        outer = cv2.dilate(m, np.ones((RING_OUTER_PX, RING_OUTER_PX), np.uint8))
        pts, _ = view.backproject((outer > 0) & (inner == 0))
        if pts is None:
            return None, None
        return float(np.median(pts[:, 2])), float(pts[:, 2].std())

    def _height_above_support(self, mask: np.ndarray, pts: np.ndarray, view: View
                             ) -> Tuple[Optional[float], str, Optional[float], Optional[float], float]:
        """Height of a region above whatever it rests on → (top, status, support_z, bulk, floor).

          floor — the difference that had to be exceeded to count as measured: the depth noise, or
                  three times the spread of the surroundings when those are not one surface.

          top  — 90th percentile of the region (a bowl's rim, a brick's top face)
          bulk — median of the interior, eroded away from the unreliable mask edge

        measured           top stands clear of the support beyond the noise floor
        below_noise        top is within noise of the support — a flush or flat thing
        unreliable         the bulk reads BELOW the surface the region rests on: depth contaminated
        no_support_visible nothing valid around it; falls back to the configured table height
        """
        support, spread = self._support_ring(mask, view)
        fallback = support is None
        if fallback:
            support, spread = self.table_z, 0.0

        top = float(np.percentile(pts[:, 2], 90))
        interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2).astype(bool)
        ipts, _ = view.backproject(interior) if interior.any() else (None, 0.0)
        bulk = float(np.median((ipts if ipts is not None else pts)[:, 2]))

        floor = max(DEPTH_NOISE_M, 3.0 * (spread or 0.0))
        if fallback:
            status = "no_support_visible"
        elif top - support > floor:
            status = "measured"
        elif bulk < support - CONTAMINATION_M:
            status = "unreliable"
        else:
            status = "below_noise"
        return round(top - support, 4), status, round(support, 4), round(bulk - support, 4), floor

    # ================================================================ records

    _CONTAMINATED_NOTE = ("part of the depth inside it reads below the surface it rests on, which cannot "
                          "be true — some of this material (dark, glossy or thin) defeats the depth "
                          "camera. The top height is from the pixels that did return, so treat it as "
                          "approximate and trust x and y more than z.")

    _HEIGHT_NOTE = {
        "measured": None,
        "below_noise": "its top is level with the surface under it — expected for anything flat or flush "
                       "(one layer of cloth, a sheet of paper). Not evidence that it is absent or handled.",
        "unreliable": "the depth inside it reads BELOW the surface it rests on, which cannot be true — this "
                      "material defeats the depth camera. Its x and y are still good; treat its height as "
                      "unknown and judge from the image instead.",
        "no_support_visible": "nothing valid around it to compare against; height is relative to the "
                              "configured table height and may be stale.",
    }

    @classmethod
    def _public(cls, rec: Dict[str, Any]) -> Dict[str, Any]:
        out = {"id": rec["id"], "label": rec["label"], "xyz": rec["xyz"],
               "yaw_deg": round(float(np.degrees(rec["yaw"])), 1), "extent_m": rec["extent_m"],
               "area_cm2": rec["area_cm2"],
               "height_above_support_m": rec["height_above_support_m"],
               # Already measured, and until now filtered out of what the model sees. It was computed
               # only to turn the top into a height above it, so the model got a relative number and
               # no absolute one, and had nothing to place `xyz` against.
               "support_z_m": rec["support_z_m"],
               "bulk_above_support_m": rec["bulk_above_support_m"],
               "height_status": rec["height_status"], "depth_valid_frac": rec["depth_valid_frac"],
               "z_spread_m": rec["z_spread_m"], "surface_tilt_deg": rec["surface_tilt_deg"],
               "measured_by": rec["camera"], "pixel": [int(rec["pixel"][0]), int(rec["pixel"][1])],
               "frame": rec["frame"]}
        note = cls._HEIGHT_NOTE.get(rec["height_status"])
        bulk = rec.get("bulk_above_support_m")
        if bulk is not None and bulk < -CONTAMINATION_M:
            out["depth_contaminated"] = True
            if rec["height_status"] == "measured":
                note = cls._CONTAMINATED_NOTE
        if note:
            out["height_note"] = note
        return out

    def get(self, object_id: str) -> Optional[Dict[str, Any]]:
        return self.objects.get(object_id)

    # ================================================================ registration

    def register_tools(self, reg: ToolRegistry):
        cams = self.cameras()
        cam_enum = {"type": "string", "enum": cams}
        px = {"type": "integer"}

        reg.register(
            "look", f"Take a fresh image. Cameras: {cams}. 'all' (the default) returns one image per "
                    f"camera: the head camera sees the whole table, the wrist camera sees only what the arm "
                    f"is currently over but in more detail. Known object ids are drawn on every image — "
                    f"solid where that camera segmented them, a cross where the position was carried over "
                    f"from another camera. Hatched areas in a wrist image are the gripper itself.",
            {"properties": {"camera": {"type": "string", "enum": cams + ["all"]}}}, self.look)

        reg.register(
            "find", "Detect all instances of an object in one camera and register them as ids with 3D "
                    "positions (metres, robot base frame). Prefer the head camera: it sees the whole table. "
                    "Give a label of 1-3 concrete visual English words. If the text finds nothing or the "
                    "wrong thing, try different wording ONCE, then call find() again with box=[x0,y0,x1,y1] "
                    "tightly around the object, read off that camera's image — a geometric prompt needs no "
                    "name. find() sets the reference size that later refinements are checked against, so "
                    "draw the box around the whole object and nothing else. Read height_status, never the "
                    "height alone.",
            {"properties": {
                "label": {"type": "string", "description": "1-3 English words; also names the returned ids"},
                "camera": cam_enum,
                "box": {"type": "array", "items": px, "minItems": 4, "maxItems": 4,
                        "description": "[x0,y0,x1,y1] pixels tightly around the WHOLE object, in that "
                                       "camera's image. Preferred over point, required for large objects."},
                "point": {"type": "array", "items": px, "minItems": 2, "maxItems": 2,
                          "description": "[u,v] pixel on a small object; segments a 36-pixel neighbourhood, "
                                         "so it returns a fragment of a large one. Ignored if box is given."}},
             "required": ["label"]}, self.find)


        reg.register(
            "measure", "Turn pixels into 3D points in the robot base frame. One pixel with u and v, which "
                       "also reports the local surface tilt and how far the spot stands above what is "
                       "immediately around it; or an ORDERED list with points=[[u, v], ...], which returns "
                       "a base-frame point for each and the total path length. Use one point to choose "
                       "exactly where on something to act. **Use the list to capture a shape you can see "
                       "now and will not be able to see later** \u2014 a drawing to wipe, the edge of a cloth, "
                       "the rim of a container. Once they are base-frame coordinates they stay true after "
                       "the arm moves in and blocks the view, and follow_path can travel through them. "
                       "Pixels with no depth, or that show the gripper, come back marked rather than "
                       "silently dropped, and the returned image shows exactly what you traced.",
            {"properties": {"u": px, "v": px, "camera": cam_enum,
                            "radius": {"type": "integer", "description": "patch half-size in pixels, default 4"},
                            "points": {"type": "array", "minItems": 1,
                                       "items": {"type": "array", "items": px, "minItems": 2, "maxItems": 2},
                                       "description": "an ORDERED list of [u, v] pixels instead of one; "
                                                      "returns a base-frame point for each, in order"}}},
            self.measure)
