"""
Guarded Cartesian path following: continuous motion, closed on the arm's measured position.

Everything else in the harness is point to point. The arm plans to a pose, arrives, stops, and the
model looks at what changed. That is the right shape for reaching and the wrong shape for contact.
Wiping a surface, dragging a fold flat, pressing down until something resists — these are not a
sequence of arrivals, they are one motion along a path while something watches for the moment it
should end.

Two things the first version got wrong, both found on the robot on 2026-09-16:

**The setpoint must not run away from the arm.** The first version streamed a trajectory computed
from the starting pose at a fixed 3 mm per tick and never looked at where the arm actually was. The
arm tracks more slowly than that, so the commanded point pulled ahead, and past some distance the
controller simply stopped following it. Every move travelled about 25 mm regardless of whether 30 mm
or 60 mm was asked for. Here the setpoint is placed a short lead ahead of the arm's *measured*
progress along the path, and the lead adapts to whatever speed the arm actually manages, so it can
never get further ahead than `LEAD_MAX_M`.

**Contact is a departure from the motion's own baseline, in either direction, not a number.** On this arm the summed joint effort is 3.97 at rest, 6.83
holding a pose with the gripper closed, 8.31 moving through free air, and 6.06 descending freely,
because gravity does the work on the way down. (Units: the vendor SDK documents joint effort in Nm.
This repository's older code called the same quantity amperes, which is where the `_a` suffix on the
parameter names below comes from; the names are kept so the thresholds stay comparable to the runs
that set them, but nothing here is a current.) No absolute threshold separates contact from
motion across those: the inherited 13 A was above everything, and anything below 8.31 A would fire
on a free-air move. So contact is detected as a rise above the baseline this motion establishes in
its own first few ticks, confirmed over consecutive ticks so a single noisy sample cannot stop the
arm. An absolute ceiling is kept, but only as an emergency stop.

A rise is only half of it. Descending onto something that holds the arm up takes LESS effort, not
more, because the surface carries weight the joints were carrying: on 2026-09-16 the gripper came to
rest on a whiteboard and the effort fell 4.92 below the descent's own baseline while the rise test saw
nothing. A rise means the arm is pushing into the world; a fall means the world is holding the arm up.
Both are contact, and both are watched.

There is a second, independent contact signal that costs nothing: if the arm stops making progress
while the setpoint is still ahead of it, something is holding it. For a light contact that barely
moves the current, that is the better evidence of the two, so both are reported.

If the servo mode cannot be entered the same path is walked with ordinary blocking moves at a
coarser step, reported as `mode: stepped`. That is slower and samples the current far less often,
but planning-mode moves are reliable on this arm, so it is a real fallback rather than a formality.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ---- streaming ----
SERVO_HZ = 25.0               # tick rate; also the current sampling rate
PATH_STEP_M = 0.003           # path resampling resolution
LEAD_M = 0.006                # initial distance the setpoint sits ahead of the measured position
LEAD_MIN_M = 0.003
LEAD_MAX_M = 0.010            # hard bound: the setpoint can never be further ahead than this, which
                              # also bounds how hard the arm pushes on whatever stops it
LEAD_GAIN_M = 0.0008          # how fast the lead adapts
TARGET_SPEED_MS = 0.030       # the speed the lead adapts towards for ordinary path following
# A contact search has to be slower than a transfer move. Both detectors need time: the rise test
# needs enough ticks to establish what this motion draws unobstructed, and the stall test needs a
# window long enough that a momentary hesitation is not mistaken for being held. At 30 mm/s a 15 mm
# press is over in twelve ticks and neither one ever arms.
CONTACT_SPEED_MS = 0.010
STEPPED_STEP_M = 0.004        # fallback: one blocking move per step
STEPPED_STALL_STEPS = 3       # consecutive commanded steps that move the arm nowhere = blocked
REACH_TOL_M = 0.004
MIN_TRACK_SPEED_MS = 0.008    # only used to size the deadline
SETTLE_S = 2.0

# ---- stall = progress stops while the setpoint is still ahead ----
STALL_TICKS = 20              # 0.8 s at 25 Hz
STALL_EPS_M = 0.0008
# An arm that has not started moving yet looks exactly like an arm that is being held, and no amount of
# waiting tells the two apart. So the stall test arms only once the arm has demonstrably moved: after
# that, stopping means something stopped it. An arm that never moves at all runs to the timeout, which
# says plainly that it could have been either.
STALL_ARM_M = 0.003

# ---- contact ----
BASELINE_TICKS = 10           # samples taken before the rise test arms itself
CONFIRM_TICKS = 2             # consecutive ticks over threshold before it counts
CONTACT_RISE_A = 2.0          # amperes above this motion's own baseline
# ...and the same distance BELOW it, which is what a descent onto a surface looks like: the surface
# carries the arm and the joints stop holding it up. Set equal to the rise rather than tuned into the
# gap between the two touches measured (-2.93, -4.92) and the descent that touched nothing (-0.02),
# because three points is not enough to tune against. Every motion reports its own `fall_a` so the
# evidence accumulates.
CONTACT_FALL_A = 2.0
OVERCURRENT_A = 17.0          # absolute emergency ceiling
CONTACT_CURRENT_A = None      # absolute contact threshold: off by default, see the module docstring

MAX_PATH_M = 0.60


@dataclass
class PathResult:
    ok: bool
    stopped_by: str        # arrived | contact | blocked | overcurrent | abort | timeout | unreachable | error
    travelled_m: float     # measured, not commanded
    remaining_m: float
    mode: str              # servo | stepped
    baseline_a: Optional[float] = None
    peak_current_a: Optional[float] = None
    current_a: Optional[float] = None
    rise_a: Optional[float] = None
    fall_a: Optional[float] = None       # how far below baseline the current dipped, reported always
    drift_a: Optional[float] = None      # how much of that was gravity torque changing with pose, not contact
    noise_a: Optional[float] = None      # largest wander in the quiet window before the tests armed
    rise_used_a: Optional[float] = None  # the rise threshold actually applied: max(requested, noise)
    fall_used_a: Optional[float] = None
    tcp_xyz: Optional[List[float]] = None
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v != "" and v is not None}


def densify(start: Sequence[float], waypoints: Sequence[Sequence[float]], step: float) -> List[np.ndarray]:
    """The polyline start → waypoints, resampled to points at most `step` apart."""
    pts: List[np.ndarray] = []
    prev = np.asarray(start, dtype=float)
    for wp in waypoints:
        wp = np.asarray(wp, dtype=float)
        dist = float(np.linalg.norm(wp - prev))
        n = max(1, int(np.ceil(dist / max(step, 1e-6))))
        for k in range(1, n + 1):
            pts.append(prev + (wp - prev) * (k / n))
        prev = wp
    return pts


def path_length(start: Sequence[float], waypoints: Sequence[Sequence[float]]) -> float:
    total, prev = 0.0, np.asarray(start, dtype=float)
    for wp in waypoints:
        wp = np.asarray(wp, dtype=float)
        total += float(np.linalg.norm(wp - prev))
        prev = wp
    return total


class _Path:
    """A polyline the arm's measured position can be projected onto, to get its progress along it."""

    def __init__(self, start, waypoints, step: float = PATH_STEP_M):
        self.pts = np.vstack([np.asarray(start, dtype=float)] + densify(start, waypoints, step))
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.total = float(self.s[-1])
        self._i = 0                       # progress never goes backwards, so a path that doubles
                                          # back on itself cannot be mistaken for progress already made

    def progress(self, now: np.ndarray) -> float:
        d = np.linalg.norm(self.pts[self._i:] - now, axis=1)
        self._i += int(np.argmin(d))
        return float(self.s[self._i])

    def point_at(self, arc: float) -> np.ndarray:
        return self.pts[int(np.clip(np.searchsorted(self.s, arc), 0, len(self.pts) - 1))]


class _Contact:
    """Contact as a DEPARTURE from the current this motion draws while it is unobstructed.

    It was a rise only, until a descent onto a whiteboard touched down and the current fell by 4.92 A
    without the detector noticing. Pressing down on something that holds you up takes LESS effort, not
    more: the surface carries part of the arm's weight and the joints stop fighting gravity for it. A
    rise is what a motion pushing INTO something shows; a fall is what a motion resting ON something
    shows. Both mean the arm has met the world.

    Three descents measured on 2026-09-16, all of which reported "arrived":

        touched the board          -2.93 A   and   -4.92 A
        stopped 10 mm above it     -0.02 A

    which is what rules out simple end-of-move deceleration: the descent that touched nothing, and
    decelerated just the same, did not move. Only three points, so the fall threshold is set as
    conservatively as the rise one rather than tuned down into the gap.
    """

    def __init__(self, rise_a: Optional[float], absolute_a: Optional[float],
                 baseline_ticks: int = BASELINE_TICKS, confirm_ticks: int = CONFIRM_TICKS,
                 fall_a: Optional[float] = None):
        self.rise_a = rise_a if (rise_a is None or rise_a > 0) else None
        self.fall_a = fall_a if (fall_a is None or fall_a > 0) else None
        # A threshold of zero is satisfied by every reading, so it reports contact on the first tick
        # and the motion never happens. Treat anything non-positive as "not set" rather than as
        # "stop immediately", which is never what a caller means.
        self.absolute_a = absolute_a if (absolute_a is not None and absolute_a > 0) else None
        self.baseline_ticks = baseline_ticks
        self.confirm_ticks = confirm_ticks
        self.samples: List[float] = []
        self.baseline: Optional[float] = None
        self.noise: Optional[float] = None
        self.peak = 0.0
        self.trough: Optional[float] = None
        self.last: Optional[float] = None
        self._over = 0
        # The fall test reads against a TRAILING reference, not the frozen baseline. See update().
        self._recent: deque = deque(maxlen=baseline_ticks + confirm_ticks)
        self.trailing: Optional[float] = None

    def update(self, cur: Optional[float]) -> Optional[str]:
        """→ a reason string when contact is confirmed, else None."""
        if cur is None:
            return None
        self.last = cur
        self.peak = max(self.peak, cur)
        self.trough = cur if self.trough is None else min(self.trough, cur)
        self._recent.append(cur)
        if self.baseline is None:
            self.samples.append(cur)
            if len(self.samples) >= self.baseline_ticks:
                self.baseline = float(np.median(self.samples))
                # The largest wander this motion showed while nothing was touching it. A threshold below
                # it would fire on something that already happened in the quiet window, so it is the floor
                # under whatever was requested — measured on this motion, in this pose, with no multiplier
                # chosen by anyone. On 2026-09-17 the model asked for 0.035 and 0.05 on a tilted arm whose
                # effort wanders several tenths; both "contacts" were noise, one stopped 26 mm above the
                # towel and the next grasp closed on air.
                self.noise = float(max(abs(v - self.baseline) for v in self.samples))
            return None                      # the rise test is not armed until there is a baseline
        # Reference for the fall test: the median of the recent window, excluding the last few ticks so
        # a step cannot pull its own reference down before it is confirmed. A frozen baseline is wrong
        # for this test and the reason is physical. Joint current tracks gravity torque, and gravity
        # torque changes with configuration; a tilted arm descending 45 mm can shed 2.6 A of current
        # with nothing touching it, smoothly, over a hundred ticks. A surface taking the arm's weight
        # sheds a similar amount in two or three ticks. The magnitudes overlap; the time scales do not.
        # Against a trailing reference the slow drift never opens a gap (the reference drifts with it)
        # while the step opens one immediately. Measured on 2026-09-21: two "contacts" on a towel at
        # 44.1 mm of travel each, 4.6 cm and 2.5 cm above the cloth, from starts 1.5 cm apart — the
        # same travel from different heights is a property of the motion, not of a surface.
        head = list(self._recent)[:-self.confirm_ticks] if len(self._recent) > self.confirm_ticks else []
        self.trailing = float(np.median(head)) if len(head) >= self.baseline_ticks else self.baseline
        hit = None
        if self.absolute_a is not None and cur > self.absolute_a:
            hit = f"{cur:.2f} A over the absolute threshold {self.absolute_a:.2f} A"
        elif self.rise_a is not None and cur > self.baseline + self.rise_used:
            hit = (f"{cur:.2f} A, {cur - self.baseline:+.2f} A above this motion's baseline of "
                   f"{self.baseline:.2f} A")
        elif self.fall_a is not None and cur < self.trailing - self.fall_used:
            hit = (f"{cur:.2f} A, {cur - self.trailing:+.2f} A below the current this motion was "
                   f"drawing a moment ago ({self.trailing:.2f} A) — a surface has just taken the "
                   f"arm's weight")
        if hit is None:
            self._over = 0
            return None
        self._over += 1
        return hit if self._over >= self.confirm_ticks else None

    @property
    def rise_used(self) -> Optional[float]:
        if self.rise_a is None:
            return None
        return max(self.rise_a, self.noise or 0.0)

    @property
    def fall_used(self) -> Optional[float]:
        if self.fall_a is None:
            return None
        return max(self.fall_a, self.noise or 0.0)

    @property
    def rise(self) -> Optional[float]:
        if self.baseline is None or self.last is None:
            return None
        return round(self.last - self.baseline, 2)

    @property
    def drift(self) -> Optional[float]:
        """How far the unobstructed current wandered from the initial baseline over the motion.

        This is the gravity-torque change the fall test used to mistake for contact. Reported so a
        fall figure can be read for what it is: `fall` minus `drift` is roughly the part that was
        sudden."""
        if self.baseline is None or self.trailing is None:
            return None
        return round(self.baseline - self.trailing, 2)

    @property
    def fall(self) -> Optional[float]:
        """How far the current dropped below baseline at its lowest, reported whether or not it fired.

        Every motion contributes a data point this way, so the threshold can be set from evidence
        instead of from the three descents that motivated it."""
        if self.baseline is None or self.trough is None:
            return None
        return round(self.baseline - self.trough, 2)


class GuardedPath:
    """Travel a Cartesian polyline while watching for the moment the motion should end.

    One instance per robot. It holds no motion state between calls: every call reads the current TCP
    pose, travels, and leaves the arm in PLANNING_POS.
    """

    def __init__(self, env, abort: Optional[threading.Event] = None, hz: float = SERVO_HZ,
                 servo_speed: Optional[str] = None, planning_speed: Optional[str] = None):
        self.env = env
        self.abort = abort or threading.Event()
        self.hz = float(hz)
        # Left as None by default: the harness never sets a speed profile, so touching it here would
        # silently change how every other move behaves.
        self.servo_speed = servo_speed
        self.planning_speed = planning_speed
        self._servo_available: Optional[bool] = None

    # ---------------- helpers ----------------

    def _current(self) -> Optional[float]:
        try:
            return self.env.get_arm_total_effort()
        except Exception:  # noqa: BLE001  a failed current read must not stop the arm dead
            return None

    def _quat_now(self) -> Optional[List[float]]:
        try:
            pose = self.env.robot.get_end_pose()
        except Exception:  # noqa: BLE001
            return None
        return None if pose is None else [float(v) for v in pose[1]]

    def _tcp(self) -> Optional[np.ndarray]:
        p = self.env.get_tcp_position()
        return None if p is None else np.asarray(p, dtype=float)

    # ---------------- the one public entry ----------------

    def follow(self, waypoints: Sequence[Sequence[float]], quat: Optional[Sequence[float]] = None,
               contact_rise_a: Optional[float] = CONTACT_RISE_A,
               contact_fall_a: Optional[float] = CONTACT_FALL_A,
               contact_current_a: Optional[float] = CONTACT_CURRENT_A,
               overcurrent_a: float = OVERCURRENT_A, stop_when_blocked: bool = True,
               reach_tol_m: float = REACH_TOL_M,
               target_speed_ms: Optional[float] = None) -> PathResult:
        """Travel from the current TCP through `waypoints`, holding `quat` throughout.

        contact_fall_a: the same distance BELOW this motion's baseline, which is what meeting a surface
        that holds the arm up looks like. None disables it.
        contact_rise_a: amperes above this motion's own baseline that count as contact. None disables
        the rise test. contact_current_a: an absolute threshold, normally left off. stop_when_blocked:
        end the motion when the arm stops making progress, which is what a light contact looks like
        when it barely moves the current. target_speed_ms defaults to a slow contact search whenever
        either detector is on, and to the transfer speed when both are off.
        """
        if not waypoints:
            return PathResult(False, "error", 0.0, 0.0, "servo", error="no waypoints given")
        start = self._tcp()
        if start is None:
            return PathResult(False, "error", 0.0, 0.0, "servo", error="cannot read the current TCP position")
        total = path_length(start, waypoints)
        if total > MAX_PATH_M:
            return PathResult(False, "error", 0.0, total, "servo",
                              error=f"path is {total:.2f} m long, over the {MAX_PATH_M} m limit for one call")
        if total < 1e-5:
            return PathResult(True, "arrived", 0.0, 0.0, "servo", tcp_xyz=[round(float(v), 4) for v in start])
        if quat is None:
            quat = self._quat_now()
            if quat is None:
                return PathResult(False, "error", 0.0, total, "servo",
                                  error="cannot read the current tool orientation")
        quat = [float(v) for v in quat]
        if target_speed_ms is None:
            # "Searching" means a CURRENT-based detector is actually armed, and it decides the speed
            # because those detectors need ticks: a baseline has to settle before a rise or a fall can
            # be told apart from the motion itself. Two things used to get this wrong.
            #
            # A threshold of 0 disables the detector inside _Contact, but this test only asked whether
            # the argument was None, so passing 0 turned the detector off and still crawled at contact
            # speed. And `stop_when_blocked` forced contact speed although the stall test is measured in
            # TIME, not distance: 0.8 s of no progress reads the same at any speed, and the 10 mm lead
            # bounds the push either way.
            #
            # On the whiteboard that cost more than everything else combined. The model passed
            # contact_rise_a=0 for its wipe strokes — it was maintaining contact, not looking for it —
            # and the six strokes covered 1.64 m at 10 mm/s: 164 s of a 360 s run.
            def _armed(v):
                return v is not None and v > 0
            searching = (_armed(contact_rise_a) or _armed(contact_fall_a)
                         or _armed(contact_current_a))
            target_speed_ms = CONTACT_SPEED_MS if searching else TARGET_SPEED_MS
        if target_speed_ms is not None:
            target_speed_ms = float(np.clip(target_speed_ms, 0.002, 0.060))

        if self._servo_available is not False:
            res = self._follow_servo(start, waypoints, quat, contact_rise_a, contact_fall_a,
                                     contact_current_a,
                                     overcurrent_a, stop_when_blocked, reach_tol_m, target_speed_ms)
            if res is not None:
                return res
        return self._follow_stepped(start, waypoints, quat, contact_rise_a, contact_fall_a,
                                    contact_current_a,
                                    overcurrent_a, stop_when_blocked, reach_tol_m)

    # ---------------- servo streaming, closed on the measured position ----------------

    def _follow_servo(self, start, waypoints, quat, rise_a, fall_a, abs_a, over_a, stop_blocked,
                      reach_tol, speed) -> Optional[PathResult]:
        """None means the servo mode could not be entered and the caller should fall back."""
        from airbot_py.arm import RobotMode          # imported late: only this path needs the SDK enum

        robot = self.env.robot
        path = _Path(start, waypoints)
        # Both detectors are sized against how long this particular motion will last, so a short press
        # still gets a baseline and a stall window instead of ending before either one arms.
        expect = max(1, int(path.total / max(speed, 1e-4) * self.hz))
        det = _Contact(rise_a, abs_a, baseline_ticks=int(np.clip(expect // 4, 3, BASELINE_TICKS)),
                       fall_a=fall_a)
        dt = 1.0 / self.hz
        lead = LEAD_M
        per_tick = speed / self.hz
        recent = deque(maxlen=int(np.clip(expect // 3, 8, STALL_TICKS)))
        progress = 0.0
        moved = False
        deadline = time.monotonic() + path.total / min(MIN_TRACK_SPEED_MS, speed) + SETTLE_S

        try:
            if self.servo_speed:
                robot.set_speed_profile(self.servo_speed)
            if not robot.switch_mode(RobotMode.SERVO_CART_POSE):
                self._servo_available = False
                return None
        except Exception as e:  # noqa: BLE001
            print(f"[path] servo mode unavailable ({type(e).__name__}: {e}); falling back to stepped moves")
            self._servo_available = False
            return None
        self._servo_available = True

        print(f"[path] servo {path.total * 100:.1f} cm at {speed * 1000:.0f} mm/s, setpoint "
              f"{lead * 1000:.0f}-{LEAD_MAX_M * 1000:.0f} mm ahead of the measured position, {self.hz:.0f} Hz"
              + (f", contact at +{rise_a:.1f} A over baseline" if rise_a else "")
              + (f", absolute {abs_a:.1f} A" if abs_a else "")
              + (", stop when blocked" if stop_blocked else "")
              + f", emergency {over_a:.1f} A")
        try:
            while True:
                if self.abort.is_set():
                    return self._finish("abort", False, progress, path, "servo", det)
                now = self._tcp()
                if now is None:
                    return self._finish("error", False, progress, path, "servo", det,
                                        error="lost the TCP position mid-motion")
                progress = path.progress(now)
                recent.append(progress)
                moved = moved or progress > STALL_ARM_M

                if progress >= path.total - reach_tol:
                    return self._finish("arrived", True, progress, path, "servo", det)

                cur = self._current()
                if cur is not None and cur > over_a:
                    print(f"[path] emergency stop: {cur:.1f} A > {over_a:.1f} A")
                    return self._finish("overcurrent", False, progress, path, "servo", det)
                why = det.update(cur)
                if why is not None:
                    print(f"[path] contact at {progress * 1000:.1f} mm: {why}")
                    return self._finish("contact", True, progress, path, "servo", det, note=why)

                if (stop_blocked and moved and len(recent) == recent.maxlen
                        and det.baseline is not None
                        and recent[-1] - recent[0] < STALL_EPS_M):
                    print(f"[path] blocked at {progress * 1000:.1f} mm: no progress for "
                          f"{recent.maxlen / self.hz:.1f} s while the setpoint was ahead")
                    return self._finish("blocked", True, progress, path, "servo", det,
                                        note="the arm stopped advancing while the setpoint was still "
                                             "ahead of it, so something is holding it")
                if time.monotonic() > deadline:
                    return self._finish("timeout", False, progress, path, "servo", det,
                                        error=("the arm never moved at all: either the servo stream is not "
                                               "being followed, or something was holding it from the start"
                                               if not moved else
                                               "ran out of time before reaching the end of the path"))

                # adapt the lead to whatever speed the arm actually manages, bounded so it can never
                # run away from the arm the way the first version's open-loop trajectory did
                if len(recent) >= 2:
                    step = recent[-1] - recent[-2]
                    advancing = len(recent) < recent.maxlen or recent[-1] - recent[0] > STALL_EPS_M
                    if step < 0.5 * per_tick and advancing:
                        lead = min(LEAD_MAX_M, lead + LEAD_GAIN_M)
                    elif step > 1.5 * per_tick:
                        lead = max(LEAD_MIN_M, lead - LEAD_GAIN_M)
                robot.servo_cart_pose(path.point_at(min(progress + lead, path.total)).tolist(), quat)
                time.sleep(dt)
        except Exception as e:  # noqa: BLE001
            return self._finish("error", False, progress, path, "servo", det,
                                error=f"{type(e).__name__}: {e}")
        finally:
            try:
                robot.switch_mode(RobotMode.PLANNING_POS)
                if self.planning_speed:
                    robot.set_speed_profile(self.planning_speed)
            except Exception as e:  # noqa: BLE001
                print(f"[path] could not restore PLANNING_POS: {e}")

    # ---------------- blocking fallback ----------------

    def _follow_stepped(self, start, waypoints, quat, rise_a, fall_a, abs_a, over_a, stop_blocked,
                        reach_tol) -> PathResult:
        """One blocking move per step. Reliable on this arm but blunt: the current is sampled once per
        step rather than 25 times a second, and a blocked arm shows up as a step that did not move it
        rather than as a rise."""
        robot = self.env.robot
        path = _Path(start, waypoints, STEPPED_STEP_M)
        det = _Contact(rise_a, abs_a, baseline_ticks=3, confirm_ticks=1, fall_a=fall_a)
        progress = 0.0
        stuck = 0
        print(f"[path] stepped {path.total * 100:.1f} cm in {len(path.pts) - 1} blocking moves"
              + (f", contact at +{rise_a:.1f} A over baseline" if rise_a else ""))
        for pt in path.pts[1:]:
            if self.abort.is_set():
                return self._finish("abort", False, progress, path, "stepped", det)
            cur = self._current()
            if cur is not None and cur > over_a:
                return self._finish("overcurrent", False, progress, path, "stepped", det)
            why = det.update(cur)
            if why is not None:
                print(f"[path] contact at {progress * 1000:.1f} mm: {why}")
                return self._finish("contact", True, progress, path, "stepped", det, note=why)
            try:
                ok = robot.set_end_pose(position=pt.tolist(), orientation=quat, blocking=True)
            except Exception as e:  # noqa: BLE001
                return self._finish("error", False, progress, path, "stepped", det,
                                    error=f"{type(e).__name__}: {e}")
            if ok is False:
                return self._finish("unreachable", False, progress, path, "stepped", det,
                                    error=f"the arm could not reach {[round(float(v), 4) for v in pt]} "
                                          f"in this tool orientation")
            now = self._tcp()
            before, progress = progress, (path.progress(now) if now is not None else progress)
            stuck = stuck + 1 if progress - before < STALL_EPS_M else 0
            if stop_blocked and stuck >= STEPPED_STALL_STEPS and progress > STALL_ARM_M:
                print(f"[path] blocked at {progress * 1000:.1f} mm: {stuck} commanded steps moved it nowhere")
                return self._finish("blocked", True, progress, path, "stepped", det,
                                    note=f"{stuck} commanded steps in a row did not move the arm, so "
                                         f"something is holding it")
        if progress < path.total - reach_tol:
            # Every step was commanded and accepted, yet the arm is short of the end. It was held.
            return self._finish("blocked", True, progress, path, "stepped", det,
                                note="the whole path was commanded but the arm ended short of it, so "
                                     "something stopped it on the way")
        return self._finish("arrived", True, progress, path, "stepped", det)

    # ---------------- result ----------------

    def _finish(self, stopped_by, ok, progress, path, mode, det, error="", note="") -> PathResult:
        tcp = self._tcp()
        return PathResult(
            ok=bool(ok), stopped_by=stopped_by, travelled_m=round(float(progress), 4),
            remaining_m=round(max(0.0, path.total - float(progress)), 4), mode=mode,
            baseline_a=None if det.baseline is None else round(det.baseline, 2),
            peak_current_a=round(det.peak, 2) if det.peak else None,
            current_a=None if det.last is None else round(det.last, 2),
            rise_a=det.rise,
            fall_a=det.fall,
            drift_a=det.drift,
            noise_a=None if det.noise is None else round(det.noise, 2),
            rise_used_a=None if det.rise_used is None else round(det.rise_used, 2),
            fall_used_a=None if det.fall_used is None else round(det.fall_used, 2),
            tcp_xyz=None if tcp is None else [round(float(v), 4) for v in tcp],
            error=error or note)
