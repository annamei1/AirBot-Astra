"""
Skills exposed to the VLM.

Every tool is one physical primitive over robot/motion_atomic.py or harness/motion.py; parameters are
ids and offsets, never bare numbers the model made up:
    move_above(object_id | xyz, dx, dy, dz, yaw_deg)   → move_to_position
    move_relative(dx, dy, dz, yaw_deg)                  → move_to_position from the current TCP
    tilt(pitch_deg, roll_deg)                           → move_to_position, rotation in place
    descend(object_id | z | dz)                         → descend_to_z
    move_until_contact / follow_path                    → GuardedPath (servo, contact-stopped)
    close_gripper(check_grasp) / open_gripper(gap)      → same names
    lift(height)                                        → lift_by
    home() · wait(seconds)

Every target goes through the same preflight (workspace box + reach sphere) and one gated execution
path (_run).

There are no macros. pick/place/release used to compose these into a whole grasp or a whole release, and
each one took a decision that belongs to the model — where on the object to close, that a place ends
with the arm going home — and hid the moment where a look would have caught the mistake. 37 episodes
used them 7 times; the mug whose handle the model found for itself was one pick() had failed on. Every
motion returns what all the cameras see the moment the arm stops, so the model judges and corrects
between any two segments. No settle wait is added after a move: the SDK call is already blocking, so
one segment flows into the next. Nothing here is learned.
"""
from __future__ import annotations

import json
import math
from contextlib import ExitStack, contextmanager, nullcontext
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from harness.narrator import NO_NEWS
from harness.arms import SECOND_ARM, WORLD_ARM, make_arms
from harness.motion import CONTACT_FALL_A, CONTACT_RISE_A, OVERCURRENT_A, GuardedPath
from harness.tools import ToolRegistry, ToolResult
from harness.perception_tools import Perception

REACH_RADIUS_M = 0.50        # targets must stay within a 0.5 m sphere around the arm's base
# How far below the lowest surface the cameras can see the envelope floor sits. It bounds how far a
# missed contact can drive into a surface; it is not meant to be reached in normal operation.
FLOOR_MARGIN_M = 0.020
# The most the lowest reading may sit under the median surface before it is treated as a depth error
# rather than as a lower surface.
FLOOR_SANITY_M = 0.040
MAX_RELATIVE_STEP_M = 0.15   # per-axis bound of one move_relative call
# The cap is 90°, a fully horizontal approach, because that is where the geometry stops making sense for
# a top-down-mounted gripper, not because of any guess about the wrist. Reachability is the manufacturer's
# to decide: set_end_pose refuses what it cannot do, and the arm has not moved when it does. An earlier
# 60° cap here was exactly the kind of invented limit that hides what the hardware can actually do.
MAX_TILT_DEG = 90.0
# Tilting rotates the tool about the TCP, so the fingertips swing through an arc. Close to the table
# that arc ends inside it. Tilt high, then descend.
TILT_MIN_CLEARANCE_M = 0.03
# A planned move can return failure having moved the arm to the requested position anyway, because the
# planner could not meet the requested ORIENTATION. Within this distance of the target the position was
# reached, whatever the return value said, and the caller has no reason to treat it as a failure.
DESCEND_ARRIVAL_TOL_M = 0.005
# How far the gripper closes per increment. Small enough that a hanging or light object is not batted
# aside before the fingers reach it, and the step the "did it stop moving" test is measured against.
GRIPPER_CLOSE_STEP_M = 0.002
# A refused move may still have put the arm exactly where it was asked. Both the position AND the tool
# angle have to match before that counts as having happened.
TOOL_ANGLE_TOL_DEG = 3.0
# The SDK move is blocking: it returns when the arm has arrived, so no settle margin is added after it
# and motion flows straight into the next segment.
SETTLE_AFTER_MOVE_S = 0.0
_TIMED_MOVES = ("move_to_position", "descend_to_z", "lift_by")


class Skills:
    def __init__(self, env, executor, perception: Perception, hover_height: float,
                 grasp_z: Optional[float], table_z: float, bounds_min, bounds_max,
                 abort: Optional[threading.Event] = None,
                 executors: Optional[Dict[str, Any]] = None,
                 arms: Optional[Dict[str, Any]] = None):
        self.env = env
        self.executor = executor
        # One executor per arm. Each is the SAME class bound to a different ArmView, so every atomic
        # action — including the incremental closure that tells one layer of cloth from none — behaves
        # identically on either arm instead of having a second, thinner implementation over here.
        self.executors: Dict[str, Any] = dict(executors or {})
        self.executors.setdefault(WORLD_ARM, executor)
        self.perception = perception
        self.hover = float(hover_height)
        self.grasp_z = None if grasp_z is None else float(grasp_z)
        self.table_z = float(table_z)
        self.bmin, self.bmax = list(bounds_min), list(bounds_max)
        self.floor_source = "configured"
        self._measure_floor()
        self.abort = abort or threading.Event()
        # Every arm on this rig, each one speaking the world frame. One entry on a single-arm rig, so
        # nothing below needs to know how many there are.
        self.arms = arms if arms is not None else make_arms(env)
        self.arm_names = list(self.arms)
        # What each arm is holding, and what its fingers measured at the moment that record was made,
        # so the claim can be re-checked instead of asserted forever. On 2026-09-17 the eraser slipped
        # out during the lift and every result for the next six calls still said holding:
        # black_eraser_1, while the gap closed 2.7 mm and a "contact" 15 mm low was the fingertips
        # hitting the board. Per arm, because two arms hold two different things — that is the point
        # of having two.
        self._hold: Dict[str, Dict[str, Any]] = {n: {"obj": None, "gap": None, "step": None}
                                                 for n in self.arms}
        self._last_descend_obj: Optional[str] = None
        self.history: List[Dict[str, Any]] = []
        # When this episode first commanded the arm to move (time.time()), for the episode's motion time.
        self.first_motion_t: Optional[float] = None
        # Continuous contact-guarded motion: everything that has to touch something softly goes
        # through this rather than through a planned move that only knows it arrived. One per arm —
        # GuardedPath holds no state between calls, but it does hold WHICH arm it drives.
        self.paths = {n: GuardedPath(a, abort=self.abort) for n, a in self.arms.items()}
        self.path = self.paths[WORLD_ARM]           # the composite skills still drive the world arm

    def _measure_floor(self):
        """Put the envelope floor under the surface this scene actually has, not under a tuned one.

        The configured floor came from play_config.json, where it was tuned against one table. On the
        whiteboard it sat 18 mm ABOVE the board, so every descent onto the board
        was clipped short and the arm wiped air. A floor that has to be re-tuned per surface is not a
        safety limit, it is a task parameter in disguise.

        What the floor is actually for is bounding the damage when contact detection misses. Contact
        stays the real stop; this only has to be below anything the arm will work on and above the
        point where pressing would hurt the rig. The cameras can see the lowest surface in reach every
        session, on any table, so it is measured rather than configured. If they cannot, the
        configured value stands and says so.
        """
        try:
            self.perception.capture_all()
            seen = self.perception.lowest_visible_surface(bounds_min=self.bmin, bounds_max=self.bmax)
        except Exception as exc:                      # a floor is not worth crashing a session over
            print(f"[floor] could not measure the scene ({type(exc).__name__}); "
                  f"keeping the configured {self.bmin[2]:+.4f}")
            return
        if not seen:
            print(f"[floor] no usable depth in the workspace; keeping the configured {self.bmin[2]:+.4f}")
            return
        # Guard against a depth reading that is simply wrong. The 2nd percentile already sits below
        # the surface by however noisy the depth is; if it lands more than FLOOR_SANITY_M under the
        # median the camera is seeing through something or past an edge, not a lower table.
        low = float(seen["low"])
        median = float(seen["median"])
        if low < median - FLOOR_SANITY_M:
            print(f"[floor] the 2nd percentile {low:+.4f} is {(median - low) * 1000:.0f} mm under the "
                  f"median {median:+.4f}; that is not a surface. Using the median instead.")
            low = median - FLOOR_SANITY_M
        floor = round(low - FLOOR_MARGIN_M, 4)
        was = self.bmin[2]
        self.bmin[2] = floor
        # self.table_z is "roughly where the work surface is", used for the tilt clearance check and as
        # the height a descend falls back to. It was the same inherited constant, so it
        # was wrong by the same 16 mm. The median of what the cameras see in reach is what it meant all
        # along, and it costs nothing extra now that the scene has been measured.
        self.table_z = round(median, 4)
        self.floor_source = f"measured from the {seen['camera']} camera"
        print(f"[floor] lowest surface in reach {low:+.4f} (median {seen['median']:+.4f}, "
              f"{seen['n']} points) -> floor {floor:+.4f}, was {was:+.4f}; "
              f"work surface {self.table_z:+.4f}. Contact detection is still what stops a descent.")

    # ============================================================ execution core

    def _run(self, actions: List[Dict[str, Any]], source: Optional[Dict[str, Any]] = None,
             clear_context: bool = True, arm: Optional[str] = None) -> Tuple[bool, int, str]:
        """Gated execution. Close-up refinement is no longer wired in
        here: it is an explicit perception step (perception.refine) that the caller — or the model —
        decides to take, so the wrist camera is usable at any moment instead of only before a grasp."""
        ex = self._executor(arm)
        if ex is None:
            return False, 0, (f"the '{arm}' arm has no executor on this rig, so it cannot run planned "
                              f"moves. Its guarded moves (follow_path, move_until_contact) still work.")
        if clear_context:
            ex.clear_context()
        with self._acting(arm):
            return self._run_actions(ex, actions, arm)

    def _run_actions(self, ex, actions: List[Dict[str, Any]], arm: Optional[str]) -> Tuple[bool, int, str]:
        for i, act in enumerate(actions):
            if self.abort.is_set():
                return False, i, "aborted by user"
            name, params = act["action"], dict(act["params"])
            if name in _TIMED_MOVES:
                params.setdefault("wait", SETTLE_AFTER_MOVE_S)
            method = getattr(ex, name, None)
            if method is None:
                return False, i, f"unknown atomic action {name}"
            print(f"  step {i + 1}/{len(actions)}: {name} {params}")
            t0 = time.monotonic()
            ok, result, err = method(**params)
            self.history.append({"action": name, "params": params, "ok": ok, "err": err,
                                 "result": result if isinstance(result, dict) else None,
                                 "seconds": round(time.monotonic() - t0, 2), "t": time.time()})
            if not ok:
                # A refused plan does not mean the arm stayed put. Twice now the planner has returned
                # false having put the arm exactly where it was asked: once on a release descend, and
                # once on a move_above whose requested and achieved tool angles differed by 0.7
                # degrees. Treating that as "nothing happened" derailed a whole episode, because the
                # model believed its approach had failed and improvised from there.
                arrived = self._arrived_anyway(name, params, arm)
                if arrived is None:
                    return False, i, err or f"{name} failed"
                print(f"  {name} reported '{err}' but {arrived}")
                self.history[-1].update(ok=True, note=arrived)
                continue
            if name in ("release_with_contact", "open_gripper"):
                ex._context.pop("fine_z", None)
        return True, -1, ""

    def _arrived_anyway(self, name: str, params: Dict[str, Any],
                        arm: Optional[str] = None) -> Optional[str]:
        """A description of where the arm ended up, when a refused move reached its target regardless.

        None means it genuinely did not get there, and the caller should treat the failure as real.
        """
        tcp = self._tcp(arm)
        if tcp is None:
            return None
        # The tool angle has to be checked too, and for a rotation in place it is the ONLY thing that
        # matters: the position is trivially "reached" because the arm never had to go anywhere. The
        # first version of this check passed a refused tilt as a success and left the model believing
        # it was at 90 degrees while the arm sat at 60.
        want_angles = (params.get("target_yaw"), params.get("pitch"), params.get("roll"))
        if any(a is not None for a in want_angles):
            got = self._tool_angles(arm)
            worst = max(abs(math.degrees(float(w) - g)) for w, g in zip(want_angles, got)
                        if w is not None) if any(a is not None for a in want_angles) else 0.0
            if worst > TOOL_ANGLE_TOL_DEG:
                return None      # the pose the caller asked for did not happen; that is a real failure
        if name == "move_to_position" and params.get("target_xyz") is not None:
            want = [float(v) for v in params["target_xyz"]]
            gap = math.dist(tcp, want)
            if gap <= DESCEND_ARRIVAL_TOL_M:
                return (f"the arm is at {[round(v, 4) for v in tcp]}, {gap * 1000:.1f} mm from the "
                        f"{[round(v, 4) for v in want]} it was sent to, and holding the tool angle it "
                        f"was asked for, so the plan was refused but the pose happened")
        if name == "descend_to_z" and params.get("target_z") is not None:
            gap = abs(tcp[2] - float(params["target_z"]))
            if gap <= DESCEND_ARRIVAL_TOL_M:
                return (f"the arm is at z={tcp[2]:.4f}, {gap * 1000:.1f} mm from the "
                        f"{float(params['target_z']):.4f} it was sent to, so the height was reached and "
                        f"only the exact tool angle was refused")
        return None

    # ---------------- preflight ----------------

    def _preflight(self, xyz, arm: Optional[str] = None) -> Optional[str]:
        """Is this world point safe for THIS arm to go to?

        Two different questions, and they belong in two different frames — getting that wrong is what
        made the second arm useless on its first run.

        Height is a fact about the table, which both arms share: the floor here was measured from the
        scene, in world z. So z is tested in the world frame.

        Reach is a fact about an arm and its own base. The configured box (x forward, y across) and
        the reach sphere describe where an AirBot Play can work relative to ITS OWN base, and both
        arms are the same model — so the point is converted into that arm's base and tested there.
        Tested in world coordinates instead, the sphere is centred on the first arm's base and the box
        stops well short of the second arm's base: every point the second arm can
        comfortably reach reads as out of bounds, and it did. Nothing about the envelope changes for
        the first arm, whose base IS the world origin.
        """
        x, y, z = (float(v) for v in xyz)
        p = [round(x, 3), round(y, 3), round(z, 3)]
        if not (self.bmin[2] <= z <= self.bmax[2]):
            return (f"target {p}: z is outside [{self.bmin[2]}, {self.bmax[2]}], the height band between "
                    f"the measured support surface and the top of the workspace")
        view = self._arm(arm)
        lx, ly, lz = (float(v) for v in view.to_base([x, y, z]))
        where = "" if arm in (None, WORLD_ARM) else \
            f" (in the {view.name} arm's own base that is {[round(lx, 3), round(ly, 3), round(lz, 3)]})"
        if not (self.bmin[0] <= lx <= self.bmax[0] and self.bmin[1] <= ly <= self.bmax[1]):
            return (f"target {p}{where} is outside the {view.name} arm's working box, "
                    f"x {self.bmin[0]}..{self.bmax[0]} and y {self.bmin[1]}..{self.bmax[1]} from its base")
        r = math.sqrt(lx * lx + ly * ly + lz * lz)
        if r > REACH_RADIUS_M:
            return (f"target {p}{where} is {r:.2f} m from the {view.name} arm's base, beyond the "
                    f"{REACH_RADIUS_M} m reach sphere — lower z, move closer to that base, "
                    f"or use the other arm")
        return None

    # ---------------- state helpers ----------------

    def _arm(self, arm: Optional[str] = None):
        """The ArmView to act through. Unknown names are an error the model can read and correct."""
        name = arm or WORLD_ARM
        view = self.arms.get(name)
        if view is None:
            raise KeyError(f"no arm called '{name}' on this robot. Arms: {self.arm_names}")
        return view

    def _known_arm(self, arm: Optional[str]) -> Optional[str]:
        """None if this arm exists, otherwise the error to hand back.

        A name the rig does not have is the model's most likely dual-arm mistake, and it has to read
        as a fact about this robot rather than as a crash. On a single-arm rig it says so plainly,
        which is the answer to 'why did asking for the left arm do nothing'.
        """
        if arm is None or arm in self.arms:
            return None
        if len(self.arms) == 1:
            return (f"this robot has one arm, '{WORLD_ARM}'. There is no '{arm}' to move. "
                    f"(A second arm needs --dual-arm and a base-to-base calibration.)")
        return f"no arm called '{arm}'. Arms on this robot: {self.arm_names}"

    def _path(self, arm: Optional[str] = None) -> GuardedPath:
        return self.paths[arm or WORLD_ARM]

    def _executor(self, arm: Optional[str] = None):
        return self.executors.get(arm or WORLD_ARM)

    def _move_to(self, target, yaw: float, pitch: float, roll: float,
                 arm: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """One planned point-to-point move, for whichever arm. Target and angles are world frame.

        It goes through that arm's executor, which is the same class bound to that arm's view, so a
        move behaves the same on either arm — including the retry and the 'it refused but arrived
        anyway' check that cost a whole episode to learn about.
        """
        ok, _, err = self._run([{"action": "move_to_position",
                                 "params": {"target_xyz": [round(float(v), 4) for v in target],
                                            "target_yaw": yaw, "pitch": pitch, "roll": roll}}],
                               clear_context=False, arm=arm)
        return ok, err

    def _tcp(self, arm: Optional[str] = None) -> Optional[List[float]]:
        p = self._arm(arm).get_tcp_position()
        return None if p is None else [float(v) for v in p]

    def _yaw(self, arm: Optional[str] = None) -> float:
        return self._tool_angles(arm)[0]

    def _tool_angles(self, arm: Optional[str] = None) -> Tuple[float, float, float]:
        """(yaw, pitch, roll) in radians. Falls back to the old tool-z formula on an env that has no
        get_tool_angles, which is correct whenever the tool is pointing straight down."""
        view = self._arm(arm)
        getter = getattr(view, "get_tool_angles", None)
        if getter is not None:
            angles = getter()
            if angles is not None:
                return angles
        pose = view.get_arm_pose()
        if pose is None:
            return 0.0, 0.0, 0.0
        R_m, _ = pose
        return float(math.atan2(R_m[1, 2], R_m[0, 2])), 0.0, 0.0

    def _gap(self, arm: Optional[str] = None) -> Optional[float]:
        try:
            g = self._arm(arm).get_gripper_state().get("gap")
            return None if g is None else round(float(g), 4)
        except Exception:  # noqa: BLE001
            return None

    def _holding(self, arm: Optional[str] = None) -> Optional[str]:
        return self._hold[arm or WORLD_ARM]["obj"]

    def _set_holding(self, object_id: Optional[str], step_m: Optional[float] = None,
                     arm: Optional[str] = None, grip: Optional[Dict[str, Any]] = None):
        """Record what is in this arm's gripper, together with the finger gap that says so and, when the
        closure measured them, the gripper effort at the grasp and while the fingers closed through air."""
        name = arm or WORLD_ARM
        self._hold[name] = {"obj": object_id,
                            "gap": self._gap(name) if object_id else None,
                            "step": (float(step_m) if (object_id and step_m) else None),
                            "grip": (grip if object_id else None)}

    def _grip_effort(self, arm: Optional[str] = None, samples: int = 3) -> Optional[float]:
        """The gripper motor's effort now: the median of a few reads of the SDK's cached feedback."""
        vals = []
        for _ in range(samples):
            try:
                e = self._arm(arm).get_gripper_state().get("effort")
            except Exception:  # noqa: BLE001
                e = None
            if e is not None:
                vals.append(float(e))
            time.sleep(0.02)
        return float(np.median(vals)) if vals else None

    @staticmethod
    def _grip_release_note(at: float, free: float, spread: float, now: float) -> Optional[str]:
        """Has the grip effort fallen back into the range it had while the fingers closed through air?

        Pure measurement against this closure's own numbers, no constant: `free` is the median effort
        of the closure's first steps, when nothing was between the fingers yet, and `spread` is their
        max - min, so free + spread is about the top of that range. A grasp that is still squeezing sits
        well above it; fingers that have lost their object fall back into it. On 2026-09-27 (pyramid,
        foam bricks) two holds stayed at 7.1 and 6.3 during the lift, against tops of 3.0 and 2.4, while
        a brick that slipped out at lift-off and two grasps that had closed on the top of a brick and on
        the baseplate studs fell to 1.3, 1.9 and 2.0, against tops of 2.3, 3.9 and 4.4. The same
        closures' own grasp rule (base + 4 x spread) would have missed one of those three. Replayed over
        all 51 holds of that day's block-bowl and pyramid episodes: it fires during the lift for every
        grasp that had closed on the studs or the top of an object (9), during the touch-down for 3
        bricks that turned in the fingers as they landed in the bowl, and never while a brick was carried.
        """
        top = free + spread
        if at <= top or now > top:
            return None
        return (f"the gripper motor is pushing at {now:.2f}: back within the range it had while the fingers "
                f"were closing through air before the grasp ({free:.2f}, up to {top:.2f}), down from "
                f"{at:.2f} at the grasp. The fingers are not squeezing anything now. If the move that just "
                f"ended touched down, the object may be resting on what it touched rather than dropped")

    def _grasp_drift(self, arm: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """How far the fingers have closed since the grasp, and whether that is more than the grasp's
        own resolution.

        Pure measurement. It does not re-grasp, does not clear `holding`, and does not decide anything;
        it stops the state from asserting a hold nobody has checked since it was made. The threshold is
        the closing increment the MODEL chose for that grasp, not a number invented here: closing
        further than one increment means the fingers travelled past what the closure itself resolved.
        """
        held = self._hold[arm or WORLD_ARM]
        if held["obj"] is None or held["gap"] is None:
            return None
        gap = self._gap(arm)
        if gap is None:
            return None
        closed = float(held["gap"]) - float(gap)
        out: Dict[str, Any] = {"closed_since_grasp_m": round(closed, 4)}
        step = held["step"] or GRIPPER_CLOSE_STEP_M
        doubts = []
        if closed > step:
            doubts.append(f"the fingers have closed {closed * 1000:.1f} mm since the grasp, more than the "
                          f"{step * 1000:.1f} mm increment that made it. Whatever was between them is thinner "
                          f"than it was, or gone")
        g = held.get("grip") or {}
        if all(g.get(k) is not None for k in ("at", "free", "spread")):
            now = self._grip_effort(arm)
            if now is not None:
                out["grip_effort_now"] = round(now, 2)
                out["grip_effort_at_grasp"] = round(float(g["at"]), 2)
                note = self._grip_release_note(float(g["at"]), float(g["free"]), float(g["spread"]), now)
                if note:
                    out["grip_released"] = note
                    doubts.append(note)
        if doubts:
            out["holding_is_doubtful"] = ("; and ".join(doubts) + ". 'holding' below is what was recorded "
                                          "then, not a fresh check — look at the wrist view.")
        return out

    def _state(self, arm: Optional[str] = None) -> Dict[str, Any]:
        name = arm or WORLD_ARM
        tcp = self._tcp(name)
        yaw, pitch, roll = self._tool_angles(name)
        out = {"arm": name,
               "tcp_xyz": None if tcp is None else [round(v, 4) for v in tcp],
               "yaw_deg": round(float(np.degrees(yaw)), 1),
               "pitch_deg": round(float(np.degrees(pitch)), 1),
               "roll_deg": round(float(np.degrees(roll)), 1),
               "gripper_gap_m": self._gap(name), "holding": self._holding(name),
               **(self._grasp_drift(name) or {})}
        if len(self.arms) > 1:
            out["arms"] = self._other_arms(name, tcp)
        return out

    def _other_arms(self, acting: str, tcp: Optional[List[float]]) -> Dict[str, Any]:
        """Where every other arm is, and how far its gripper is from this one's.

        Reported, not enforced. Two arms in one workspace can hit each other, and the distance
        between their grippers is the number that says how close that is — but what counts as too
        close depends on what they are holding and which way they are facing, which is yours to judge,
        not this layer's. If you want a limit enforced, put it in a do() step: {"require":
        {"state.arms.left.gripper_distance_m": {">": 0.08}}} stops the chunk before the step that
        would breach it. That way the number is one you chose for what you are doing.

        The positions are as good as the base-to-base transform, about 8 mm. Ample for "will these
        two collide"; not enough to hand one arm a grasp point the other measured.
        """
        out: Dict[str, Any] = {}
        for other in self.arms:
            if other == acting:
                continue
            p = self._tcp(other)
            entry: Dict[str, Any] = {"tcp_xyz": None if p is None else [round(v, 4) for v in p],
                                     "holding": self._holding(other),
                                     "gripper_gap_m": self._gap(other)}
            if p is not None and tcp is not None:
                entry["gripper_distance_m"] = round(
                    float(np.linalg.norm(np.asarray(p) - np.asarray(tcp))), 4)
            out[other] = entry
        return out

    def _result(self, payload: Dict[str, Any], caption: str,
                arm: Optional[str] = None) -> ToolResult:
        """Assemble a tool result AFTER refreshing the cameras, so it reports what is true now.

        The order used to be wrong and silently so: Python evaluates the payload before the `images=`
        argument, so every `state` was built from the previous frames while only the pictures were
        current. Nothing depended on that until visibility did.

        `can_see` is the point of this: after every motion, how much of each known object each camera
        can actually see from where the arm now is. The arm is its own occluder and it occludes most
        at the moment it is about to act, which is why this belongs next to every move rather than in
        a tool the model has to think to call.
        """
        shots = self._observe(caption, arm)
        # `arm` decides whose state this is. It used to be unconditionally the world arm's, which
        # silently overwrote whatever the caller had already put here: every left-arm move came back
        # reporting the RIGHT arm's tcp, gripper and holding. The motion was correct and the report
        # was of the other hand, which is the worst combination — the model cannot see the error, it
        # can only be confused by it.
        payload["state"] = self._state(arm)
        # What the narrator wrote since the model's previous result. Rides on every result so the
        # model never has to remember to ask; look() carries the whole account. See harness/narrator.py.
        nar = getattr(self.perception, "narrator", None)
        if nar is not None:
            payload["what_happened"] = nar.news() or NO_NEWS
            payload["narrator"] = nar.status_line()
        try:
            seen = self.perception.visibility()
        except Exception as e:  # noqa: BLE001  a reporting failure must never fail the motion
            seen = {}
            print(f"[skills] visibility report skipped: {type(e).__name__}: {e}")
        if seen:
            payload["can_see"] = self.perception.visibility_lines(seen)
        return ToolResult.json(payload, images=shots)

    def _observe(self, caption: str, arm: Optional[str] = None) -> List[Tuple[str, np.ndarray]]:
        """A fresh frame from EVERY camera; a picture from the ones a picture tells something new.

        The head camera shows where things are, the acting arm's wrist shows whether the gripper
        actually did what the head camera cannot resolve. Both are captured and both are pictured.

        "The wrist" means the wrist of the arm that ACTED, which is why this takes `arm`. A left-arm
        motion sends the head view and the LEFT wrist view; the right arm's camera, parked wherever it
        was left, is captured and reported through `can_see` but not pictured. See
        CameraRig.default_image_names for why that split, and not another.
        """
        pictured = set(self.perception.rig.default_image_names(arm or WORLD_ARM))
        shots = []
        for v in self.perception.capture_all():
            if v.name not in pictured:
                continue
            img = self.perception.overlay(v.name)
            if img is not None:
                shots.append((f"[{v.name}] {caption}", img))
        return shots

    @contextmanager
    def _acting(self, arm: Optional[str] = None):
        """Context around every motion: marks this arm as acting for the narrator (its wrist frames are
        sent for these seconds), and stamps the episode's first motion command. Every planned move,
        guarded path and trip home goes through here, so the stamp is the moment the arm was first
        told to move."""
        if self.first_motion_t is None:
            self.first_motion_t = time.time()
        nar = getattr(self.perception, "narrator", None)
        with (nar.acting(arm or WORLD_ARM) if nar is not None else nullcontext()):
            yield

    def new_episode(self):
        """Forget what the previous episode held and did. The model starts every episode with a fresh
        conversation, so anything carried over here is knowledge it never had."""
        self._hold = {n: {"obj": None, "gap": None, "step": None} for n in self.arms}
        self._last_descend_obj = None
        self.history = []
        self.first_motion_t = None

    def _go_home(self):
        with ExitStack() as stack:
            for name in self.arm_names:
                stack.enter_context(self._acting(name))
            return self.env.reset_position()

    def _resolve(self, object_id: Optional[str], xyz: Optional[List[float]], what: str,
                 given_yaw_deg: Optional[float] = None
                 ) -> Tuple[Optional[Dict[str, Any]], Optional[List[float]], Optional[float],
                            Optional[str], Optional[str]]:
        """→ (record|None, xyz, yaw|None, note|None, error).

        When both an id and an explicit xyz arrive, **the xyz wins** and the record is kept only for the
        yaw and the bookkeeping. The id names a whole object and resolves to its centre; an xyz is a
        point the model measured. Asking to go above a specific point *of* an object is a normal thing
        to want, and silently substituting the centre is a trap: on the whiteboard the model asked for a
        point it had measured on the board and was sent 97 mm away to the board's middle, then spent a
        call working out why. The substitution is reported so it is never silent either way.
        """
        rec = None
        if object_id:
            rec = self.perception.get(object_id)
            if rec is None:
                return None, None, None, None, (f"unknown {what} id '{object_id}'. Call find(label) "
                                                f"first. Known: {list(self.perception.objects)}")
        point = [float(v) for v in xyz] if (xyz and len(xyz) == 3) else None
        if rec is None and point is None:
            return None, None, None, None, f"give {what} object_id (preferred) or xyz=[x,y,z] from measure()"
        yaw = float(rec["yaw"]) if rec is not None else None
        if rec is None:
            return None, point, None, None, None
        centre = [float(v) for v in rec["xyz"]]
        if point is None:
            return rec, centre, yaw, None, None
        gap = float(np.linalg.norm(np.asarray(point) - np.asarray(centre)))
        note = None
        if gap > 0.005:
            # The yaw sentence used to be unconditional. With an explicit yaw_deg the move uses that, and
            # the note went on saying the id had supplied -147.4\u00b0 while the arm turned to 0\u00b0.
            whose = (f"The yaw is the {given_yaw_deg:g}\u00b0 you gave" if given_yaw_deg is not None
                     else f"The id supplied the yaw ({math.degrees(yaw):.1f}\u00b0)")
            note = (f"you gave both '{object_id}' and an xyz {gap * 1000:.0f} mm from its centre, so the "
                    f"xyz was used. {whose}. Pass the id alone to act on the object's own position.")
        return rec, point, yaw, note, None

    # ============================================================ Tier-1 skills

    def wait(self, seconds: float = 10.0) -> ToolResult:
        """Do nothing for a while, then observe. For when the world needs time and the model does not.

        A person is rearranging things, something is settling, the other arm is mid-move. The
        narrator keeps watching throughout, so the result's what_happened covers the wait and the
        images are from its end. The cap is a safety bound on an unattended arm, not a judgement.
        """
        s = max(0.0, min(float(seconds), 120.0))
        t0 = time.time()
        while time.time() - t0 < s:
            if self.abort.is_set():
                return ToolResult.error("aborted while waiting")
            time.sleep(0.25)
        return self._result({"ok": True, "waited_s": round(time.time() - t0, 1)}, f"after waiting {s:.0f} s.")

    def home(self, arm: Optional[str] = None) -> ToolResult:
        if (e := self._known_arm(arm)):
            return ToolResult.error(e)
        if arm in (None, WORLD_ARM):
            ok = self._go_home()
        else:
            ok = self._arm(arm).reset_position()
            self._set_holding(None, arm=arm)
        if not ok:
            return ToolResult.error('Home completion was not confirmed; command may still be pending.')
        return self._result({"ok": True, "state": self._state(arm)}, "with the arm at home.", arm)

    # ============================================================ Tier-0 primitives

    def _tilt_rad(self, pitch_deg, roll_deg, keep: bool = True) -> Tuple[float, float, Optional[str]]:
        """Requested tool angles in radians. keep=True means "unspecified" carries the current angle
        forward rather than silently snapping the tool back to vertical mid-task."""
        cur_pitch, cur_roll = self._tool_angles()[1:]
        pitch = cur_pitch if (pitch_deg is None and keep) else math.radians(float(pitch_deg or 0.0))
        roll = cur_roll if (roll_deg is None and keep) else math.radians(float(roll_deg or 0.0))
        lim = math.radians(MAX_TILT_DEG)
        if abs(pitch) > lim or abs(roll) > lim:
            return 0.0, 0.0, (f"pitch and roll must be within ±{MAX_TILT_DEG:.0f}° of straight down; "
                              f"past that the wrist runs out of travel before the pose is useful")
        return pitch, roll, None

    def move_above(self, object_id: Optional[str] = None, xyz: Optional[List[float]] = None,
                   dx: float = 0.0, dy: float = 0.0, dz: Optional[float] = None,
                   yaw_deg: Optional[float] = None, pitch_deg: Optional[float] = None,
                   roll_deg: Optional[float] = None, arm: Optional[str] = None) -> ToolResult:
        if (e := self._known_arm(arm)):
            return ToolResult.error(e)
        rec, base, yaw, resolve_note, err = self._resolve(object_id, xyz, "object", given_yaw_deg=yaw_deg)
        if err:
            return ToolResult.error(err)
        pitch, roll, err = self._tilt_rad(pitch_deg, roll_deg)
        if err:
            return ToolResult.error(err)
        target = [base[0] + float(dx), base[1] + float(dy), base[2] + (self.hover if dz is None else float(dz))]
        yaw = math.radians(yaw_deg) if yaw_deg is not None else (yaw if yaw is not None else self._yaw(arm))
        if (e := self._preflight(target, arm)):
            return ToolResult.error(e)
        ok, err = self._move_to(target, yaw, pitch, roll, arm)
        payload: Dict[str, Any] = {"ok": ok, "error": err, "state": self._state(arm)}
        if resolve_note:
            payload["note"] = resolve_note
        return self._result(payload, "after the move.", arm)

    def move_relative(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                      yaw_deg: Optional[float] = None, pitch_deg: Optional[float] = None,
                      roll_deg: Optional[float] = None, arm: Optional[str] = None) -> ToolResult:
        if (e := self._known_arm(arm)):
            return ToolResult.error(e)
        d = [float(dx), float(dy), float(dz)]
        if max(abs(v) for v in d) > MAX_RELATIVE_STEP_M:
            return ToolResult.error(f"each of dx, dy, dz must be within ±{MAX_RELATIVE_STEP_M} m per call")
        pitch, roll, err = self._tilt_rad(pitch_deg, roll_deg)
        if err:
            return ToolResult.error(err)
        tcp = self._tcp(arm)
        if tcp is None:
            return ToolResult.error("cannot read the current TCP position")
        target = [tcp[k] + d[k] for k in range(3)]
        yaw = math.radians(yaw_deg) if yaw_deg is not None else self._yaw(arm)
        if (e := self._preflight(target, arm)):
            return ToolResult.error(e)
        ok, err = self._move_to(target, yaw, pitch, roll, arm)
        return self._result({"ok": ok, "error": err, "state": self._state(arm)}, "after the move.", arm)

    def tilt(self, pitch_deg: float = 0.0, roll_deg: float = 0.0,
             yaw_deg: Optional[float] = None, arm: Optional[str] = None) -> ToolResult:
        """Rotate the tool where it stands, holding the TCP still.

        pitch leans the approach axis away from straight down, in the horizontal direction given by
        yaw. That is what lets a fingertip get under the edge of something flat, or the tool meet a
        surface that is not horizontal. roll spins the tool about its own approach axis, which does
        nothing visible while the tool is vertical.
        """
        if (e := self._known_arm(arm)):
            return ToolResult.error(e)
        pitch, roll, err = self._tilt_rad(pitch_deg, roll_deg, keep=False)
        if err:
            return ToolResult.error(err)
        tcp = self._tcp(arm)
        if tcp is None:
            return ToolResult.error("cannot read the current TCP position")
        clearance = tcp[2] - self.table_z
        if (abs(pitch) > 1e-6 or abs(roll) > 1e-6) and clearance < TILT_MIN_CLEARANCE_M:
            return ToolResult.error(
                f"the gripper is {clearance * 1000:.0f} mm above the table and tilting swings the "
                f"fingertips down through an arc, into it. lift() to at least "
                f"{TILT_MIN_CLEARANCE_M * 1000:.0f} mm of clearance, tilt there, then descend.")
        yaw = math.radians(yaw_deg) if yaw_deg is not None else self._tool_angles(arm)[0]
        ok, err = self._move_to(tcp, yaw, pitch, roll, arm)
        hint = ("" if ok else
                " The arm did not move. A refusal means the planner could not reach that angle FROM THE "
                "JOINT CONFIGURATION THE ARM IS IN, not that the angle is impossible: the same tilt was "
                "accepted and then refused three poses apart on this rig, because the planner picks a "
                "different joint solution each time. Try it again after moving somewhere else and coming "
                "back, or with a smaller pitch, a different yaw, or a spot closer to the base.")
        return self._result({"ok": ok, "error": (err or "") + hint, "state": self._state(arm)},
                            "after tilting the tool.", arm)

    def _clamp_to_envelope(self, start, target,
                           arm: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
        """The furthest point along start → target that stays inside the safety envelope.

        move_until_contact means "travel until something stops you", and the envelope is one of the
        things that can stop you. Refusing the whole move because its far end is out of bounds would
        make it useless for its main job: feeling for a surface that sits at the floor of the
        workspace, which on this rig is exactly where the table is.
        """
        s0 = np.asarray(start, dtype=float)
        if self._preflight(s0.tolist(), arm) is not None:
            return [round(float(v), 4) for v in s0], "already outside the envelope"
        # Only z can be clamped componentwise here: it is the one axis whose bounds are world-frame.
        # x and y are bounds in the ACTING ARM's base, so clamping the world value against them would
        # be clamping in the wrong frame; the bisection below handles those correctly for either arm.
        t = np.asarray(target, dtype=float).copy()
        t[2] = min(max(float(t[2]), self.bmin[2]), self.bmax[2])
        note = "clipped to the workspace box" if float(np.abs(t - np.asarray(target, float)).max()) > 1e-9 else None
        if self._preflight(t.tolist(), arm) is not None:
            lo, hi, d = 0.0, 1.0, t - s0
            for _ in range(24):
                mid = (lo + hi) / 2
                if self._preflight((s0 + d * mid).tolist(), arm) is None:
                    lo = mid
                else:
                    hi = mid
            t = s0 + d * lo
            note = "clipped to the reach envelope"
        return [round(float(v), 4) for v in t], note

    def move_until_contact(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                           contact_rise_a: Optional[float] = None,
                           contact_current_a: Optional[float] = None,
                           stop_if_stuck: bool = True, arm: Optional[str] = None) -> ToolResult:
        """Travel in a straight line by (dx, dy, dz), stopping the instant the arm meets resistance.

        Unlike every other move here this one is continuous and guarded: the arm current is read
        about 25 times a second while it travels, so it stops on contact rather than after it. It is
        how you find a surface whose height you do not trust, and how you touch something soft
        without crushing it.
        """
        d = [float(dx), float(dy), float(dz)]
        if max(abs(v) for v in d) > MAX_RELATIVE_STEP_M:
            return ToolResult.error(f"each of dx, dy, dz must be within ±{MAX_RELATIVE_STEP_M} m per call")
        if max(abs(v) for v in d) < 1e-5:
            return ToolResult.error("give a direction to travel: dx, dy and dz are all zero")
        if (e := self._known_arm(arm)):
            return ToolResult.error(e)
        tcp = self._tcp(arm)
        if tcp is None:
            return ToolResult.error("cannot read the current TCP position")
        target, clipped = self._clamp_to_envelope(tcp, [tcp[k] + d[k] for k in range(3)], arm)
        room = float(np.linalg.norm(np.asarray(target) - np.asarray(tcp)))
        if room < 1e-4:
            return ToolResult.error(
                f"the safety envelope leaves no room to travel that way from {[round(v, 3) for v in tcp]}. "
                f"Workspace box {self.bmin}..{self.bmax}, reach sphere {REACH_RADIUS_M} m.")
        rise = CONTACT_RISE_A if contact_rise_a is None else float(contact_rise_a)
        with self._acting(arm):
            res = self._path(arm).follow([target], contact_rise_a=rise, contact_current_a=contact_current_a,
                                         stop_when_blocked=bool(stop_if_stuck))
        touched = res.stopped_by in ("contact", "blocked")
        self.history.append({"action": "move_until_contact",
                             "params": {"d": d, "rise_a": rise, "arm": arm or WORLD_ARM},
                             "ok": res.ok, "err": res.error, "t": time.time()})
        out = {"ok": res.ok, "contact": touched, "rise_threshold_a": round(rise, 2),
               "state": self._state(arm), **res.as_dict()}
        if res.rise_used_a is not None and res.noise_a is not None and res.rise_used_a > round(rise, 2):
            out["rise_threshold_raised"] = (
                f"you asked for {rise:g}, but before the test armed this motion's effort already wandered "
                f"{res.noise_a:.2f} from its own baseline with nothing touching it, so {res.rise_used_a:.2f} "
                f"was used. A threshold below that fires on the arm's own jitter.")
        if clipped:
            out["travel_limited"] = (f"{clipped}: {room * 1000:.0f} mm of the "
                                     f"{max(abs(v) for v in d) * 1000:.0f} mm asked for")
        if touched:
            out["felt_by"] = ("the current rising" if res.stopped_by == "contact"
                              else "the arm stopping while it was still being pushed")
            drift = self._grasp_drift(arm) or {}
            held = self._holding(arm)
            if held and res.tcp_xyz and "holding_is_doubtful" in drift:
                out["touched_with"] = (
                    f"unclear. The record says '{held}', but {drift['holding_is_doubtful']} "
                    f"If it is gone, what met the surface was the fingertips and this height is not "
                    f"that object's contact height. The gripper was at z={res.tcp_xyz[2]:.4f}.")
            elif held and res.tcp_xyz:
                # What met the surface is the far face of whatever is in the fingers, not the
                # fingertips. Nobody can know how far a picked-up object sticks out until it touches
                # something, and until then every height that assumes the fingertips is wrong by that
                # amount. One touch on a surface whose height you know settles it for that object.
                out["touched_with"] = (
                    f"the far face of '{held}', not the fingertips. The gripper was at "
                    f"z={res.tcp_xyz[2]:.4f} when it met the surface, so from now on that is the height "
                    f"at which this object touches a surface at that height. Against a surface whose "
                    f"height you know, the difference is how far it sticks out beyond the fingers.")
        elif res.stopped_by == "arrived":
            out["note"] = ("travelled the whole distance without the current rising and without the arm "
                           "being held up, so nothing was touched along the way.")
        return self._result(out, "after the guarded move.", arm)

    def follow_path(self, points: List[List[float]], relative: bool = False,
                    speed_mm_s: Optional[float] = None,
                    contact_rise_a: Optional[float] = None,
                    contact_current_a: Optional[float] = None,
                    stop_if_stuck: Optional[bool] = None,
                    arm: Optional[str] = None) -> ToolResult:
        """Travel through a list of points as ONE continuous motion, holding the current tool angles.

        Point-to-point moves stop at every waypoint, which is fine for reaching and wrong for
        dragging, wiping or smoothing, where the contact has to be unbroken. Give points in base
        metres from measure() or find(); relative=true treats them as offsets from where the gripper
        is now. Any single coordinate may be null, meaning "leave that axis where it is", which is how a
        path travels in xy at a height the previous move established. Set contact_current_a to stop early
        if the path runs into something.
        """
        if not points:
            return ToolResult.error("give at least one point")
        if (e := self._known_arm(arm)):
            return ToolResult.error(e)
        tcp = self._tcp(arm)
        if tcp is None:
            return ToolResult.error("cannot read the current TCP position")
        pts = []
        for i, pt in enumerate(points):
            if len(pt) != 3:
                return ToolResult.error(f"point {i} is not [x, y, z]")
            # A null coordinate means "leave this axis where it is". Per-axis, because `relative` is
            # all-or-nothing across x, y and z and the common case has neither shape: the model has
            # absolute xy it measured, and a z it can only learn from the step before — touch down,
            # then travel along the surface at whatever height that turned out to be. Without this the
            # sequence cannot go in one chunk at all, which is what kept every wipe path in its own
            # model call. Nothing about it is a surface, a height or a wipe; it is one axis held still.
            p = [(tcp[k] if pt[k] is None else
                  (tcp[k] + float(pt[k]) if relative else float(pt[k]))) for k in range(3)]
            if (e := self._preflight(p, arm)):
                return ToolResult.error(f"point {i} {[round(v, 3) for v in p]}: {e}")
            pts.append(p)
        blocked_ends_it = (bool(stop_if_stuck) if stop_if_stuck is not None
                           else (contact_rise_a is not None or contact_current_a is not None))
        # The fall is a second way of seeing the same event, so it follows the rise: a caller that
        # asked for no contact detection wants to travel, not to be stopped by a dip in the current.
        with self._acting(arm):
            res = self._path(arm).follow(pts,
                                         target_speed_ms=(None if speed_mm_s is None
                                                          else float(speed_mm_s) / 1000.0),
                                         contact_rise_a=contact_rise_a,
                                         contact_fall_a=(None if contact_rise_a is None else CONTACT_FALL_A),
                                         contact_current_a=contact_current_a,
                                         stop_when_blocked=blocked_ends_it)
        self.history.append({"action": "follow_path", "params": {"n": len(pts), "arm": arm or WORLD_ARM},
                             "ok": res.ok, "err": res.error, "t": time.time()})
        return self._result({"ok": res.ok, "points": len(pts), "state": self._state(arm),
                             **res.as_dict()}, "after following the path.", arm)

    def descend(self, object_id: Optional[str] = None, z: Optional[float] = None,
                dz: Optional[float] = None, refine: bool = True,
                arm: Optional[str] = None) -> ToolResult:
        if (e := self._known_arm(arm)):
            return ToolResult.error(e)
        tcp = self._tcp(arm)
        if tcp is None:
            return ToolResult.error("cannot read the current TCP position")
        rec = None
        if object_id:
            rec = self.perception.get(object_id)
            if rec is None:
                return ToolResult.error(f"unknown object id '{object_id}'. Known: {list(self.perception.objects)}")
            # An explicit z is normally a height the model just measured in a close-up. Silently
            # replacing it with the calibrated constant would ignore what it asked for.
            base_z = (float(z) if z is not None else
                      (self.grasp_z if self.grasp_z is not None else float(rec["xyz"][2])))
            target_z = base_z + float(dz or 0.0)
            yaw = float(rec["yaw"])
            if math.hypot(tcp[0] - rec["xyz"][0], tcp[1] - rec["xyz"][1]) > 0.05:
                return ToolResult.error(f"the gripper is not above {object_id} (XY distance "
                                        f"{math.hypot(tcp[0] - rec['xyz'][0], tcp[1] - rec['xyz'][1]):.3f} m). "
                                        f"move_above({object_id}) first.")
        elif z is not None:
            target_z, yaw = float(z), self._yaw(arm)
        elif dz is not None:
            target_z, yaw = tcp[2] + float(dz), self._yaw(arm)
        else:
            return ToolResult.error("give object_id (grasp height with wrist-camera refinement), z, or dz")
        if (e := self._preflight([tcp[0], tcp[1], target_z], arm)):
            return ToolResult.error(e)
        # Descending changes the height and nothing else. The executor's descend_to_z defaults to a
        # straight-down tool, so without the current pitch and roll a tilted approach snapped back to
        # vertical on the way down: on 2026-09-23 the fingers, aimed at a drawer handle from a 60°
        # tilt, arrived pointing down at the top of the cabinet. lift already carries the angles.
        _, pitch, roll = self._tool_angles(arm)
        ok, _, err = self._run([{"action": "descend_to_z",
                                 "params": {"target_z": round(target_z, 4), "target_yaw": yaw,
                                            "pitch": pitch, "roll": roll}}],
                               clear_context=False, arm=arm)
        self._last_descend_obj = object_id if ok else None
        return self._result({"ok": ok, "error": err, "descended_to_m": round(target_z, 4),
                             "state": self._state(arm)}, "after descending.", arm)

    def close_gripper(self, check_grasp: bool = True, step_m: Optional[float] = None,
                      grip_rise_a: Optional[float] = None, arm: Optional[str] = None) -> ToolResult:
        """Close in small increments, stopping the moment the fingers stop moving.

        Slamming shut in one command bats a light or hanging object out of the way before the fingers
        reach it — it flicked a pendant aside twice — and it decides "did I grasp anything" from the
        final gap against a fixed 5 mm floor, which makes anything thinner than that unholdable by
        definition. Closing in steps fixes both at once: gently enough not to move what it is reaching
        for, and the answer comes from WHERE the closure stopped, which has no thickness floor.

        `grip_rise_a` is how far above the closure's own effort baseline the fingers push before they
        stop. It answers a different question from "is something between the fingers": meeting an object
        and holding it firmly enough to carry it are not the same event, and stopping at the first is
        what let the eraser slide out three times. Measured on 2026-09-17 across three grasps of the same
        object: rises of 2.55 and 2.30 both slipped during the move, 5.14 held. The default is left where
        it was — first contact — because the right squeeze depends on the object's weight, its surface
        and what is about to be done with it, all of which the model knows and this layer does not.
        """
        if (e := self._known_arm(arm)):
            return ToolResult.error(e)
        step = GRIPPER_CLOSE_STEP_M if step_m is None else max(0.0005, float(step_m))
        ok, _, err = self._run([{"action": "close_gripper",
                                 "params": {"check_grasp": bool(check_grasp), "close_step": step,
                                            "grip_rise": (None if grip_rise_a is None
                                                          else float(grip_rise_a))}}],
                               clear_context=False, arm=arm)
        detail = (self.history[-1].get("result") or {}) if self.history else {}
        if ok and check_grasp:
            grip = {"at": detail.get("effort_now"), "free": detail.get("effort_baseline"),
                    "spread": detail.get("effort_baseline_spread")}
            self._set_holding(self._last_descend_obj or "unknown_object", step, arm=arm, grip=grip)
        gap = self._gap(arm)
        out: Dict[str, Any] = {"ok": ok, "error": err, "close_step_m": round(step, 4),
                               "state": self._state(arm)}
        out.update({k: v for k, v in detail.items()
                    if k in ("stopped_by", "steps", "note") or k.startswith("effort")})
        set_by = detail.get("effort_rise_set_by")
        if grip_rise_a is not None and set_by and set_by != "the requested rise":
            out["grip_rise_not_applied"] = (
                f"you asked for grip_rise_a={float(grip_rise_a):g}, but {set_by}, so the closure stopped "
                f"at {detail.get('effort_rise_used')} above baseline instead. Only a larger value than "
                f"that changes anything.")
        if check_grasp:
            out["grasp_detected"] = ok
            out["evidence"] = detail.get("note") or (
                f"the fingers stopped at {gap * 1000:.1f} mm" if ok and gap
                else "the fingers closed the whole way")
            out["hint"] = ("Two independent signals decide this: where the fingers came to rest, and how "
                           "hard the gripper motor is pushing. The wrist image below is the third. If they "
                           "disagree, believe the image.")
            if ok:
                out["reach_changed"] = ("you are holding something now, so the far end of the arm is no "
                                        "longer the fingertips: it is whatever of this object sticks out "
                                        "past them, by an amount nothing can know until it touches "
                                        "something. move_until_contact downwards onto a surface whose "
                                        "height you know measures it in one move.")
        return self._result(out, "right after closing the gripper.", arm)

    def open_gripper(self, gap: float = 0.08, arm: Optional[str] = None) -> ToolResult:
        if (e := self._known_arm(arm)):
            return ToolResult.error(e)
        ok, _, err = self._run([{"action": "open_gripper", "params": {"gap": float(gap)}}],
                               clear_context=False, arm=arm)
        if ok:
            self._set_holding(None, arm=arm)
        return self._result({"ok": ok, "error": err, "state": self._state(arm)}, "after the move.", arm)

    def lift(self, height: Optional[float] = None, arm: Optional[str] = None) -> ToolResult:
        if (e := self._known_arm(arm)):
            return ToolResult.error(e)
        h = self.hover if height is None else float(height)
        tcp = self._tcp(arm)
        if tcp is None:
            return ToolResult.error("cannot read the current TCP position")
        if (e := self._preflight([tcp[0], tcp[1], tcp[2] + h], arm)):
            return ToolResult.error(e)
        yaw, pitch, roll = self._tool_angles(arm)
        ok, _, err = self._run([{"action": "lift_by",
                                 "params": {"height": h, "target_yaw": yaw,
                                            "pitch": pitch, "roll": roll}}],
                               clear_context=False, arm=arm)
        return self._result({"ok": ok, "error": err, "state": self._state(arm)}, "after lifting.", arm)

    def register_tools(self, reg: ToolRegistry):
        obj = {"type": "string"}
        num = {"type": "number"}
        xyz = {"type": "array", "items": num, "minItems": 3, "maxItems": 3}
        # Same shape, but a coordinate may be null: "leave this axis where it is". Only paths take it,
        # because only a path has a previous move whose result an axis can be inherited from.
        xyz_or_null = {"type": "array", "minItems": 3, "maxItems": 3,
                       "items": {"type": ["number", "null"]}}

        # ---- Tier-1 ----
        reg.register("home", "Return the arm to the home pose with the gripper open.", {"properties": {}},
                     self.home)
        reg.register("wait", "Do nothing for `seconds` (default 10, at most 120), then observe. For when the "
                             "world needs time and you do not: someone is rearranging things, something is "
                             "settling, the other arm is mid-move. The narrator keeps watching throughout; the "
                             "result's what_happened covers the wait.",
                     {"properties": {"seconds": {"type": "number"}}}, self.wait)

        # ---- Tier-0 ----
        reg.register(
            "move_above", "Move the gripper to a point defined by an object id plus offsets: x = obj.x + dx, "
                          "y = obj.y + dy, z = obj.z + dz (dz defaults to the hover height). yaw_deg defaults "
                          "to the object's yaw, pitch_deg and roll_deg to the tool angles you are already "
                          "holding (at the zero joint pose that is horizontal, pitch 90°). Or give xyz from measure() instead of object_id.",
            {"properties": {"object_id": obj, "xyz": xyz, "dx": num, "dy": num, "dz": num, "yaw_deg": num,
                            "pitch_deg": num, "roll_deg": num}},
            self.move_above)
        reg.register(
            "move_relative", f"Move the gripper by (dx, dy, dz) metres from where it is now, each within "
                             f"±{MAX_RELATIVE_STEP_M}. Use for small corrections ('2 cm further left'). "
                             f"Tool angles carry over unless you give new ones.",
            {"properties": {"dx": num, "dy": num, "dz": num, "yaw_deg": num, "pitch_deg": num, "roll_deg": num}},
            self.move_relative)
        reg.register(
            "tilt", f"Rotate the tool where it stands, keeping the gripper at the same point. pitch_deg leans "
                    f"the approach axis away from straight down, in the horizontal direction given by yaw, so "
                    f"the fingers come at something from the side instead of from above — the only way to get "
                    f"a fingertip under the edge of something flat, or to meet a surface that is not level. "
                    f"roll_deg spins the tool about its own approach axis and does nothing visible while the "
                    f"tool is vertical. Both within ±{MAX_TILT_DEG:.0f}°. Tilt with clearance above the surface, "
                    f"then descend: tilting swings the fingertips down through an arc. Not every angle is "
                    f"reachable at every point; a refusal says so and the arm has not moved.",
            {"properties": {"pitch_deg": num, "roll_deg": num, "yaw_deg": num}}, self.tilt)
        reg.register(
            "move_until_contact", "Travel in a straight line by (dx, dy, dz) metres and STOP the instant the "
                                  "arm meets resistance. The current is read about 25 times a second while it "
                                  "travels, so this stops on contact rather than after it. Use it to find a "
                                  "surface whose height the depth camera will not give you, to touch something "
                                  "soft without crushing it, and to learn how far below the reported gripper "
                                  "position the fingertips really reach. The result says whether it stopped on "
                                  "contact or ran the whole distance, and the current it saw either way. "
                                  "Contact is a RISE above what this same motion draws unobstructed, not an "
                                  "absolute number, because a descent draws LESS current than holding still "
                                  "(gravity helps) while free-air motion draws more. The arm ceasing to make "
                                  "progress while it is still being pushed counts as contact too, and for "
                                  "something light it is the better of the two signals. contact_rise_a "
                                  "overrides the rise. For scale, measured on this arm: 40 descents that "
                                  "touched nothing wandered up to 0.98 above their own baseline, most of them "
                                  "0.1 to 0.8, and more when the tool is tilted; a descent that comes to rest "
                                  "on a surface usually shows a FALL of 2.3 to 3.4, which is detected "
                                  "separately. Each motion also measures its own quiet wander before the test "
                                  "arms and never uses a threshold below it; the result reports noise_a and "
                                  "rise_used_a. stop_if_stuck=false keeps going when the "
                                  "arm stops advancing, for when the RESISTANCE IS THE POINT and you mean to "
                                  "push or pull through it — a switch, a latch, a drawer that has to shut. The "
                                  "emergency current ceiling still protects the arm. Default true, which is "
                                  "what feeling for a surface wants.",
            {"properties": {"dx": num, "dy": num, "dz": num, "contact_rise_a": num,
                            "contact_current_a": num, "stop_if_stuck": {"type": "boolean"}}},
            self.move_until_contact)
        reg.register(
            "follow_path", "Travel through a list of [x, y, z] points as ONE continuous motion, holding the "
                           "current tool angles. Every other move stops at its target; this one does not, which "
                           "is what dragging, wiping and smoothing need, because the contact must not break "
                           "between waypoints. relative=true reads the points as offsets from where the gripper "
                           "is now. ANY coordinate may be null, meaning leave that axis where it is \u2014 so "
                       "[[x, y, null], ...] travels in xy at whatever height the arm is already at, which "
                       "is how you stroke a surface you have just touched down on without knowing its "
                       "height in advance. Set contact_rise_a to stop early if the path runs into something; left "
                           "unset, the path is travelled whatever is in the way.",
            {"properties": {"points": {"type": "array", "items": xyz_or_null, "minItems": 1},
                            "relative": {"type": "boolean"},
                            "speed_mm_s": {"type": "number", "description":
                                "travel speed in mm per second. Left out it picks itself: 10 while a "
                                "current-based contact threshold is armed, because a rise or a fall needs "
                                "ticks to be told apart from the motion itself, and 30 otherwise. Set it "
                                "when you know better \u2014 a long wipe whose contact is already "
                                "established can go faster. Capped at 60."},
                            "contact_rise_a": num,
                            "contact_current_a": num, "stop_if_stuck": {"type": "boolean"}},
             "required": ["points"]}, self.follow_path)
        reg.register(
            "descend", "Descend vertically keeping XY. Give EXACTLY ONE of: object_id, to go to that "
                       "object's grasp height (the arm must already be above it, e.g. after move_above); "
                       "z, an absolute height; or dz, a distance to go down, which is negative. They are "
                       "alternatives, not parts of one number: z together with dz descends to z + dz, "
                       "which is almost never what you meant.",
            {"properties": {"object_id": obj, "z": num, "dz": num, "refine": {"type": "boolean"}}}, self.descend)
        reg.register(
            "close_gripper", "Close the gripper in small increments, stopping the instant the fingers stop "
                             "moving. It closes gently, so a light or hanging object is not knocked aside "
                             "before the fingers reach it, and whether something is held is read from WHERE "
                             "the closure stopped rather than from a minimum gap — which means it works on "
                             "something a fraction of a millimetre thick as well as on a brick. step_m sets "
                             "the increment, default 2 mm; make it smaller for something that moves easily. "
                             "Returns the wrist view, which is the only independent check on the answer.",
            {"properties": {"check_grasp": {"type": "boolean"}, "step_m": num,
                            "grip_rise_a": {"type": "number", "description":
                                "how far above the closure's own effort baseline to keep squeezing before "
                                "stopping. Meeting an object and holding it firmly enough to carry it are "
                                "different events, and the default stops at the first one. Raise it for "
                                "anything heavy, smooth, or about to be pressed against something. Lower "
                                "it for something soft that cannot load the motor much, like cloth — a weak "
                                "rise is then accepted and the result says it was weak, because by effort "
                                "alone a soft grasp and an empty gripper look alike. Each closure measures "
                                "its own noise first and cannot go below it. What was actually used "
                                "comes back as effort_rise_used, with effort_rise_set_by saying why, and "
                                "state.closed_since_grasp_m and state.grip_effort_now afterwards tell you whether the hold "
                                "survived: state.grip_released appears when the grip effort has fallen back "
                                "to what the motor drew while the fingers were closing through air."}}},
            self.close_gripper)
        reg.register("open_gripper", "Open the gripper to gap metres (default 0.08).",
                     {"properties": {"gap": num}}, self.open_gripper)
        reg.register("lift", "Lift vertically by height metres (default: hover height).",
                     {"properties": {"height": num}}, self.lift)

        self._register_second_arm(reg)

    # ---- the second arm, if there is one ----

    TWO_ARM_TOOLS = ("move_above", "move_relative", "tilt", "move_until_contact", "follow_path",
                     "descend", "close_gripper", "open_gripper", "lift", "home")

    def _register_second_arm(self, reg: ToolRegistry):
        """Widen the motion tools to take `arm`, but only on a robot that has two.

        Nothing above knows about a second arm. This adds one parameter to the tools that move
        something, and says the two things that are true of this rig and not obvious from the
        parameter itself: the coordinates do not change, and which arm looks matters more than which
        arm reaches.
        """
        if len(self.arms) < 2:
            return
        others = [n for n in self.arm_names if n != WORLD_ARM]
        arm_schema = {"type": "string", "enum": self.arm_names,
                      "description": f"which arm does this. Defaults to '{WORLD_ARM}'."}
        note = (
            f"\nTWO ARMS: this robot has {self.arm_names}. Pass arm='{others[0]}' to move that one; "
            f"leave it out and '{WORLD_ARM}' moves, exactly as before.\n"
            f"Coordinates never change: every xyz is in the same world frame for both arms, the one "
            f"find() and measure() report in. You do not convert anything.\n"
            f"But WHICH ARM LOOKED decides how accurate a point is. A point measured by the head "
            f"camera or by one arm's wrist camera, then reached for by the OTHER arm, carries about "
            f"8 mm of error — the two arms' own kinematics, which no calibration removes. A point "
            f"measured by the SAME arm's wrist camera that then acts on it carries about 1.6 mm, "
            f"because the error cancels. So: to reach roughly, any camera will do; to pinch a single "
            f"layer, look with the arm that is about to pinch — move that arm over the spot, measure "
            f"from its own wrist camera, then act.\n"
            f"Each arm holds its own object; `holding` in the result is the acting arm's. The others "
            f"are under state.arms, with the distance between the grippers, so you can see the two "
            f"converging before they meet. Nothing here stops them touching each other — if you want "
            f"that enforced, put it in a do() step as "
            f"{{\"require\": {{\"state.arms.{others[0]}.gripper_distance_m\": {{\">\": 0.08}}}}}} "
            f"with whatever distance your grippers and their loads actually need.")
        for name in self.TWO_ARM_TOOLS:
            reg.augment(name, {"arm": arm_schema}, note if name == "move_above" else
                        f"\narm: which arm does this ({' or '.join(self.arm_names)}); "
                        f"default '{WORLD_ARM}'. See move_above for what changes with two arms.")
        print(f"[skills] motion tools widened to {self.arm_names}")
