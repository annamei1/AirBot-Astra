"""
Play Robot Environment Interface

Provides the same interface as RealRobotEnv (MMK2) but for AirBot Play arm.
This allows motion_atomic.py to be reused without modification.

Key differences from MMK2:
- No TFClient — head camera transform is static from calibration JSON
- No Pose/Position/Orientation wrappers — direct lists to SDK
- Gripper position passed directly (not normalized by max_opening)
- Depth requires D405 raw-to-mm conversion
"""

import cv2
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Callable
from scipy.spatial.transform import Rotation as R

from play_sdk import PlayRealRobot
from robot.feedback_wait import wait_for_target
from play.config import (
    HOME_JOINT, ARM_PORT,
    SECOND_ARM_PORT, SECOND_WRIST_CAMERA_SERIAL,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    HEAD_CAMERA_SERIAL, WRIST_CAMERA_SERIAL,
    HEAD_ZOOM, WRIST_ZOOM,
    D405_DEPTH_RAW_TO_MM,
    MOVE_WAIT_SEC, GRIPPER_CLOSE_WAIT_SEC, GRIPPER_OPEN_WAIT_SEC,
    GRIPPER_RELEASE_WIDTH,
    load_head_camera_calibration, load_hand_eye_calibration,
)


CONFIG_DIR = Path(__file__).parent.parent / "config"

# Below this tilt a pose counts as straight down. Two things follow, and they have to agree with each
# other or a pose read back from the arm cannot be commanded again.
#   · get_tool_angles reports yaw from the tool z-axis and roll as zero, instead of reading yaw off an
#     approach vector whose horizontal part is a few thousandths of noise.
#   · the ±π yaw normalisation that keeps the wrist off its joint limit still applies. An exact-zero
#     test let a 0.1° reading disable it, and an unreachable pose is a far worse outcome than the
#     under-10° change in approach direction that flipping yaw causes at this tilt.
TILT_ZERO_RAD = TILT_DEGENERATE_RAD = np.radians(5.0)

# Where the fingers come to rest when they close on nothing, measured on this arm: 0.2 mm and 0.3 mm on
# two separate empty closes. So a resting gap under EMPTY_GAP_M is an empty gripper, one over HELD_GAP_M
# is holding something, and the band between is genuinely ambiguous and reported as such rather than
# guessed. One layer of cloth falls in that band, which is a fact about this gripper, not a bug.
EMPTY_GAP_M = 0.0005
HELD_GAP_M = 0.0015

# Measured 2026-09-16, the first run in which this arm's gripper effort was ever read. Closing on
# nothing: 0.970 while the fingers still move, settling to 2.024 once they meet and the motor holds.
# Closing on a 8 mm pendant: 9.583. A rise floor of 2.0 clears the empty case by about two and sits
# roughly four times below a real grasp. It replaces a placeholder of 0.02, which was a guess and
# which the empty close would have tripped.
EFFORT_RISE_HELD = 2.0


# ==================== tool orientation, as free functions ====================
# These are pure geometry: they say what "yaw 30°, pitch 0" MEANS as a rotation, and read it back out
# of one. They are frame-agnostic — whatever frame you hand them, you get back. That is the whole
# reason they are not methods: a second arm has to express the same angles in the world frame and then
# convert, and the one thing that must not happen is a second copy of this convention drifting from
# this one. PlayRobotEnv's methods below call these and are unchanged for every existing caller.

def grasp_quaternion(yaw: float, pitch: float = 0.0,
                     roll: float = 0.0) -> Tuple[float, float, float, float]:
    """Tool orientation as a quaternion [x, y, z, w]. All angles in radians.

        R = Rz(yaw) · Ry(π/2 − pitch) · Rx(roll)

    With pitch = roll = 0 this is exactly the straight-down grasp pose the whole codebase used
    before the two extra angles existed: the tool's local x-axis, which is the approach axis,
    maps to base −z, and the tool's local z-axis lies in the horizontal plane pointing along yaw.

    pitch > 0 leans the approach axis away from vertical by that angle, in the horizontal
    direction given by yaw. At yaw = 0 the gripper tips towards +x. That is what lets the
    fingers come at something from the side rather than from directly above, which a flat sheet
    on a table gives no other way to grasp.

    roll spins the tool about its own approach axis. While the tool points straight down that is
    indistinguishable from adding to yaw; it becomes a separate degree of freedom once tilted.

    The yaw normalisation below is only valid straight down. A parallel two-finger gripper
    grasps identically at yaw and yaw ± π only while the approach axis is vertical, because the
    two fingers simply swap places. Once the tool is tilted, yaw ± π swings the approach axis to
    the opposite side of the object, which is a different motion, so it is skipped.
    """
    if abs(pitch) < TILT_ZERO_RAD and abs(roll) < TILT_ZERO_RAD:
        while yaw > np.pi / 2:
            yaw -= np.pi
        while yaw < -np.pi / 2:
            yaw += np.pi
    r_final = (R.from_euler('z', yaw)
               * R.from_euler('y', np.pi / 2 - pitch)
               * R.from_euler('x', roll))
    quat = r_final.as_quat()  # [x, y, z, w]
    return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])


def tool_angles_from_matrix(M: np.ndarray) -> Tuple[float, float, float]:
    """(yaw, pitch, roll) in radians from a tool rotation matrix — the inverse of grasp_quaternion.

    Read off the rotation matrix rather than through a Euler decomposition, because the zyx
    decomposition is degenerate at exactly the pose the arm spends most of its time in: with the
    tool pointing straight down, yaw and roll are the same rotation and scipy is free to split
    them any way it likes. Straight down we therefore take yaw from the tool's z-axis, which is
    what the rest of the code already does, and report roll as zero.
    """
    M = np.asarray(M, float)
    approach = M[:, 0]                      # tool x-axis = approach direction
    pitch = float(np.arccos(float(np.clip(-approach[2], -1.0, 1.0))))
    if pitch < TILT_DEGENERATE_RAD:
        # The horizontal part of the approach vector has magnitude sin(pitch). Near vertical that
        # is a few thousandths and its direction is pure numerical noise, so yaw read off it jumps
        # by tens of degrees between two readings of an arm that has not moved. Take yaw from the
        # tool's z-axis instead, which is exactly what the rest of the code has always used, and
        # call roll zero: at this tilt roll and yaw are the same rotation anyway.
        return float(np.arctan2(M[1, 2], M[0, 2])), pitch, 0.0
    yaw = float(np.arctan2(approach[1], approach[0]))
    M0 = (R.from_euler('z', yaw) * R.from_euler('y', np.pi / 2 - pitch)).as_matrix()
    residual = M0.T @ M
    return yaw, pitch, float(np.arctan2(residual[2, 1], residual[1, 1]))


class PlayRobotEnv:
    """
    Play Robot Environment — same interface as RealRobotEnv.

    Usage:
        env = PlayRobotEnv()
        executor = AtomicMotionExecutor(robot_env=env, ...)
    """

    def __init__(
        self,
        port: int = ARM_PORT,
        head_camera_serial: str = HEAD_CAMERA_SERIAL,
        wrist_camera_serial: str = WRIST_CAMERA_SERIAL,
        config_path: Optional[str] = None,
        second_arm: bool = False,
        second_arm_port: Optional[int] = None,
        second_wrist_camera_serial: Optional[str] = None,
    ):
        # Load config
        if config_path is None:
            config_path = str(CONFIG_DIR / "play_config.json")
        self.config = self._load_config(config_path)

        # Gripper default open width (passed directly to SDK)
        self.gripper_open_width = GRIPPER_RELEASE_WIDTH

        # Camera zoom factors
        self._head_zoom = HEAD_ZOOM
        self._wrist_zoom = WRIST_ZOOM

        # Load calibration (static head camera transform)
        self.head_intrinsics, self.T_head2base = load_head_camera_calibration()

        # Apply digital zoom to head camera intrinsics
        if self._head_zoom > 1.0:
            orig = self.head_intrinsics
            W = orig.get('width', 640)
            H = orig.get('height', 480)
            s = self._head_zoom
            self.head_intrinsics = {
                'fx': orig['fx'] * s,
                'fy': orig['fy'] * s,
                'cx': (orig['cx'] - W * (1 - 1 / s) / 2) * s,
                'cy': (orig['cy'] - H * (1 - 1 / s) / 2) * s,
                'width': W,
                'height': H,
            }

        # Init robot
        #
        # READ THIS BEFORE CHANGING THE PORT MAPPING.
        #
        # Every convenience method on PlayRealRobot — set_end_pose, get_end_pose, set_gripper,
        # get_joint_efforts, set_joint_positions, get_gripper_state — delegates to `self.robot.left`,
        # and in single-arm mode play_sdk parks the one arm there. So `self.left` is not "the left
        # arm": it is "the arm this class drives". The arm whose base IS the world frame has to stay
        # in that slot or all six of those calls silently start driving the other arm, with no error
        # anywhere — the harness would keep computing world coordinates from one arm's calibration
        # and sending them to the other.
        #
        # Hence the inversion, which reads wrong and is right:
        #     play_sdk .left  slot  <- world arm   (port 50050, the one every calibration calls
        #                              "right", the base all measured coordinates live in)
        #     play_sdk .right slot  <- second arm  (port 50052, calibration's "left")
        #
        # The camera slots are NOT inverted, because play_sdk keeps the three Realsense slots fully
        # decoupled from the arm handles. right_wrist_camera stays the world arm's wrist exactly as
        # it is in single-arm mode, and left_wrist_camera is the second arm's. So camera names match
        # the calibration files and only the two arm handles are swapped.
        #
        # local_inference_eef.py (the pi0.5 rollout) maps these the other way round and is also
        # correct: it never touches a convenience method, only the handles.
        self._second_arm_port = second_arm_port if second_arm_port is not None else SECOND_ARM_PORT
        self._dual = bool(second_arm and self._second_arm_port is not None)
        second_wrist_serial = (second_wrist_camera_serial
                               if second_wrist_camera_serial is not None
                               else SECOND_WRIST_CAMERA_SERIAL)

        if self._dual:
            print(f"[PlayRobotEnv] Connecting (world arm port={port}, "
                  f"second arm port={self._second_arm_port})...")
            self.robot = PlayRealRobot(
                left_port=port,                                    # world arm — see the note above
                right_port=self._second_arm_port,                  # second arm
                head_camera_serial=head_camera_serial,
                right_wrist_camera_serial=wrist_camera_serial,     # world arm's wrist
                left_wrist_camera_serial=second_wrist_serial,      # second arm's wrist
                camera_width=CAMERA_WIDTH, camera_height=CAMERA_HEIGHT, camera_fps=CAMERA_FPS,
            )
            self.second_arm = self.robot.right
        else:
            print(f"[PlayRobotEnv] Connecting (port={port})...")
            self.robot = PlayRealRobot(
                port=port,
                head_camera_serial=head_camera_serial,
                right_wrist_camera_serial=wrist_camera_serial,
                camera_width=CAMERA_WIDTH, camera_height=CAMERA_HEIGHT, camera_fps=CAMERA_FPS,
            )
            self.second_arm = None

        # Move to home
        self._go_home()
        print("[PlayRobotEnv] Initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            print(f"[PlayRobotEnv] Warning: Config not found at {path}")
            return {}
        with open(path) as f:
            return json.load(f)

    def _go_home(self):
        self.robot.set_gripper(position=self.gripper_open_width)
        time.sleep(GRIPPER_OPEN_WAIT_SEC)
        self.robot.set_joint_positions(HOME_JOINT, blocking=True)
        time.sleep(1.0)

    # ==================== Arm State ====================

    def get_arm_pose(self, tool: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get current arm end-effector pose.
        Returns: (rotation_matrix_3x3, position_xyz) or None
        tool=True is the grasp point of the mounted fingers (the harness's TCP); tool=False is the
        frame the SDK reports, which is what the wrist camera's extrinsics are relative to.
        """
        try:
            pose = self.robot.get_end_pose(tool=tool)
            if pose is None:
                return None
            position, orientation = pose
            rotation = R.from_quat(orientation).as_matrix()
            return rotation, np.array(position)
        except Exception as e:
            print(f"[PlayRobotEnv] Error getting arm pose: {e}")
            return None

    def get_tcp_position(self) -> Optional[np.ndarray]:
        """Get current TCP position [x, y, z]."""
        pose = self.robot.get_end_pose()
        if pose is None:
            return None
        return np.array(pose[0])

    # ==================== Arm Control ====================

    def move_arm(
        self,
        position: List[float],
        yaw: float,
        wait: float = MOVE_WAIT_SEC,
        check_abort: Optional[Callable[[], bool]] = None,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> bool:
        """Move arm to target position with the given tool angles (radians).

        pitch and roll default to 0, the straight-down pose.
        """
        if check_abort and check_abort():
            return False

        ox, oy, oz, ow = self._compute_grasp_orientation(yaw, pitch, roll)

        try:
            ok = self.robot.set_end_pose(
                position=[float(position[0]), float(position[1]), float(position[2])],
                orientation=[float(ox), float(oy), float(oz), float(ow)],
                blocking=True,
            )
            if not ok:
                tilt = "" if (pitch == 0.0 and roll == 0.0) else \
                    f", pitch={np.degrees(pitch):.1f}°, roll={np.degrees(roll):.1f}°"
                # A refusal does NOT mean the arm stayed put. Observed 2026-09-16: a descend into a bowl
                # returned false with an unreachable ORIENTATION, having moved to the requested position
                # anyway, and the caller treated that as "nothing happened". Say where the arm is.
                here = self.get_tcp_position()
                where = ""
                if here is not None:
                    gap = float(np.linalg.norm(np.asarray(here, float) - np.asarray(position, float)))
                    where = (f"  实际停在 {[round(float(v), 4) for v in here]}"
                             + (f"，距目标 {gap * 1000:.1f} mm：位置其实到了，没达成的是姿态"
                                if gap < 0.005 else f"，距目标 {gap * 1000:.1f} mm"))
                print(f"[PlayRobotEnv] ⚠️ set_end_pose 规划失败/不可达: "
                      f"pos={[round(float(p), 4) for p in position]}, yaw={yaw:.3f}{tilt} "
                      f"(该位姿下该点可能超出可达范围){where}")
                return False
            return self._wait_with_check(wait, check_abort)
        except Exception as e:
            print(f"[PlayRobotEnv] Move arm failed: {e}")
            return False

    def move_arm_xy_only(
        self,
        target_xyz: List[float],
        wait: float = MOVE_WAIT_SEC,
        check_abort: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """Move to new XYZ while preserving current orientation."""
        if check_abort and check_abort():
            return False

        pose = self.robot.get_end_pose()
        if pose is None:
            print("[PlayRobotEnv] Cannot get current pose for xy_only move")
            return False

        _, current_quat = pose

        try:
            self.robot.set_end_pose(
                position=[float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])],
                orientation=[float(current_quat[0]), float(current_quat[1]),
                             float(current_quat[2]), float(current_quat[3])],
                blocking=True,
            )
            return self._wait_with_check(wait, check_abort)
        except Exception as e:
            print(f"[PlayRobotEnv] move_arm_xy_only failed: {e}")
            return False

    def _compute_grasp_orientation(self, yaw: float, pitch: float = 0.0,
                                   roll: float = 0.0) -> Tuple[float, float, float, float]:
        """Tool orientation as a quaternion [x, y, z, w] — see grasp_quaternion above."""
        return grasp_quaternion(yaw, pitch, roll)

    def get_tool_angles(self) -> Optional[Tuple[float, float, float]]:
        """Current (yaw, pitch, roll) in radians — see tool_angles_from_matrix above."""
        pose = self.get_arm_pose()
        if pose is None:
            return None
        M, _ = pose
        return tool_angles_from_matrix(M)

    def reset_position(self) -> bool:
        """Reset arm to home position."""
        print("[PlayRobotEnv] Resetting to home...")
        try:
            self._go_home()
            return True
        except Exception as e:
            print(f"[PlayRobotEnv] Reset failed: {e}")
            return False

    # ==================== Gripper Control ====================

    def open_gripper(self, gap: Optional[float] = None) -> bool:
        """
        Open gripper.
        Args:
            gap: target opening passed directly to SDK (None → default open width)
        """
        position = self.gripper_open_width if gap is None else gap

        try:
            maximum = self.config['grasp'].get('gripper_max_width')
            if maximum is not None:
                position = min(position, float(maximum))
            if not np.isfinite(position) or position < 0:
                raise ValueError('Invalid gripper opening')
            if not self.robot.set_gripper(position=position):
                return False
            if not wait_for_target(self.robot.left.get_eef_pos, [position], 0.002):
                print('[PlayRobotEnv] Gripper opening not confirmed within 15 s; command may still be pending')
                return False
            return True
        except Exception as e:
            print(f"[PlayRobotEnv] Gripper open failed: {e}")
            return False

    def close_gripper(self) -> bool:
        """Close gripper fully, in one command."""
        try:
            self.robot.set_gripper(position=0.0)
            time.sleep(GRIPPER_CLOSE_WAIT_SEC)
            return True
        except Exception as e:
            print(f"[PlayRobotEnv] Gripper close failed: {e}")
            return False

    @staticmethod
    def _effort_trace(efforts, baseline) -> Dict[str, Any]:
        """Every verdict carries the numbers it was reached with.

        The effort threshold below is a placeholder: nobody has ever seen a real reading from this
        motor, because the call that produces it was not wired up. Rather than let a guessed constant
        decide grasps the way a guessed 13 A once decided contacts, the trace goes out with every
        close, so the first real grasp on the robot sets the scale instead of confirming a guess.
        """
        if not efforts:
            return {"effort_trace": "the gripper reported no effort at all"}
        out = {"effort_now": round(efforts[-1], 4), "effort_min": round(min(efforts), 4),
               "effort_max": round(max(efforts), 4), "effort_samples": len(efforts)}
        if baseline is not None:
            out["effort_baseline"] = round(baseline[0], 4)
            out["effort_baseline_spread"] = round(baseline[1], 4)
        out["effort_note"] = ("the threshold that decides this is not calibrated yet — read these "
                              "numbers against each other, not against an absolute value")
        return out

    @staticmethod
    def _weak_grasp_note(rise: float) -> str:
        """Evidence, not a verdict. Said whenever a grasp is called on a rise smaller than any rigid grasp
        has produced, so a lowered request is honoured without its weakness going unmentioned."""
        return (f"this counted as holding on an effort rise of {rise:.2f}. Every rigid grasp measured so far "
                f"rose at least {EFFORT_RISE_HELD:.1f}, and an empty gripper's noise has reached almost 1.0. A "
                f"soft object can legitimately load the motor this little; so can nothing at all. The wrist "
                f"image, and state.closed_since_grasp_m once it is lifted, are what tell them apart.")

    def close_gripper_gradually(self, step: float = 0.002, settle: float = 0.12,
                                stalled_steps: int = 2, floor: float = 0.0,
                                effort_rise: float = EFFORT_RISE_HELD, baseline_steps: int = 3,
                                check_abort: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        """Close the gripper in small increments, and stop as soon as the fingers stop moving.

        Two things wrong with slamming to zero in one command, both of them general:

        A gripper that closes at full speed bats a light or hanging object out of the way before it
        closes on it. Observed on a pendant hanging from a chain, which the fingers flicked aside
        twice; the same is true of anything that can move when touched, which includes cloth.

        And "did it grasp anything" was decided from the final gap against a 5 mm floor, which means
        nothing thinner than 5 mm can ever be reported as held. A single layer of cloth is about half
        a millimetre. Here the answer comes from WHERE the closure stopped: if the fingers quit
        following the commanded gap, something is between them, whatever its thickness.

        Returns a dict; `held` is the signal, `gap` is where the fingers came to rest.
        """
        state = self.get_gripper_state()
        gap = state.get("gap")
        efforts: List[float] = []
        squeeze_baseline = None
        # The thresholds that actually decided, reported with every verdict. A caller's effort_rise is a
        # request, not a guarantee: below this closure's own measured noise it cannot be told apart from
        # motor jitter and the noise decides instead. That used to happen silently — on 2026-09-17 the
        # model asked for 0.5 and 0.7, both lost to noise floors of 1.32 and 5.64, and nothing said so.
        thresholds: Dict[str, Any] = {"effort_rise_requested": round(float(effort_rise), 3)}

        def trace() -> Dict[str, Any]:
            out = self._effort_trace(efforts, squeeze_baseline)
            out.update(thresholds)
            return out

        def moved_at_all() -> bool:
            """Did the fingers travel at all from where they started?

            A stall, and a resting gap wider than the held band, are both evidence of a grasp ONLY
            once the closure has travelled. Something can stop fingers that are closing; nothing can
            hold them open at the width they started from. Without this the two are indistinguishable
            and the wrong one gets reported — on 2026-09-18 the second arm's gripper stalled at its
            opening width three times in a row, each came back as "something is between them", and the
            episode carried a grasp that did not exist until the model looked at a picture and
            overruled it. The yardstick is the caller's own step, not a number invented here.
            """
            return (float(gap) - measured) >= step

        def jammed() -> Dict[str, Any]:
            return {"ok": False, "held": False, "gap": round(measured, 4), "steps": steps,
                    "stopped_by": "gripper did not move", **trace(),
                    "note": f"the gripper never moved: commanded closed from {float(gap) * 1000:.1f} mm "
                            f"and still at {measured * 1000:.1f} mm. That is not a grasp — nothing can "
                            f"hold the fingers open at the width they started from. Either they are "
                            f"jammed (fingers pressed onto a surface carry the arm's weight and then "
                            f"cannot slide), or the gripper did not take the command. Lift a few "
                            f"millimetres clear of whatever the tool is resting on and close again."}

        if state.get("effort") is not None:
            efforts.append(float(state["effort"]))
        if gap is None:
            ok = self.close_gripper()
            return {"ok": ok, "held": None, "gap": None, "steps": 0,
                    "note": "gripper position unreadable; closed in one command instead"}

        # play_sdk logs a line per gripper command; one per 2 mm would bury the terminal.
        sdk_log = logging.getLogger("play_sdk")
        previous_level = sdk_log.level
        target = float(gap)
        measured = float(gap)
        steps = stuck = 0
        try:
            sdk_log.setLevel(logging.WARNING)
            while target > floor + 1e-6:
                if check_abort and check_abort():
                    return {"ok": False, "held": None, "gap": measured, "steps": steps,
                            "note": "aborted part way through the closure"}
                target = max(floor, target - step)
                self.robot.set_gripper(position=target)
                time.sleep(settle)
                steps += 1
                fresh = self.get_gripper_state()
                if fresh.get("effort") is not None:
                    efforts.append(float(fresh["effort"]))
                    # The gripper motor's own effort, which is a signal this codebase has never had:
                    # it was read from a dictionary key that did not exist and came back 0.000 every
                    # time. Compared against what this closure itself draws while the fingers are
                    # still moving freely, exactly as the arm's contact test does, so it needs no
                    # absolute calibration to be useful on the first run.
                    if len(efforts) >= baseline_steps and squeeze_baseline is None:
                        head = sorted(efforts[:baseline_steps])
                        squeeze_baseline = (head[len(head) // 2], max(head) - min(head))
                    if squeeze_baseline is not None:
                        base, spread = squeeze_baseline
                        # The request is the model's to make in either direction; the only floor is
                        # this closure's own measured noise, which is a property of the sensor on this
                        # closure and nothing else. A floor at EFFORT_RISE_HELD was tried and removed the
                        # same day: that number was measured on rigid objects and an empty gripper, so as
                        # a gate the model could not lower it assumed everything grasped is stiff — cloth
                        # is not. A weak grasp is now reported as weak instead of being refused.
                        noise = 4.0 * spread
                        used = max(effort_rise, noise)
                        trigger = base + used
                        thresholds["effort_rise_used"] = round(used, 3)
                        thresholds["effort_rise_set_by"] = (
                            "the requested rise" if effort_rise >= noise
                            else f"this closure's own noise (4 x {spread:.3f} spread), which was larger "
                                 f"than the {effort_rise:.3f} requested")
                        if efforts[-1] > trigger:
                            if efforts[-1] - base < EFFORT_RISE_HELD:
                                thresholds["effort_weak_grasp"] = self._weak_grasp_note(efforts[-1] - base)
                            return {"ok": True, "held": True, "gap": round(measured, 4), "steps": steps,
                                    "stopped_by": "effort",
                                    **trace(),
                                    "note": f"the gripper is pushing at {efforts[-1]:.3f} against "
                                            f"{base:.3f} while it was still closing freely, so it is "
                                            f"squeezing something — this works at any thickness"}
                now = fresh.get("gap")
                if now is None:
                    continue
                moved, measured = measured - float(now), float(now)
                # Position alone is not enough to call a grasp, and at small steps it is not even a
                # measurement. The fingers report to about 0.2-0.3 mm, so a 0.5 mm step makes
                # "did it move less than 0.4 x step" a question about quantisation: it fired twice on
                # an empty gripper, at 45.1 mm and 31.6 mm. If the fingers have genuinely stopped then
                # the motor is pushing on whatever stopped them, so the effort must agree. Measured:
                # real grasps rose 4.5 to 5.8 above baseline, the two false positives rose under 1.0.
                stuck = stuck + 1 if moved < 0.4 * step else 0
                if stuck >= stalled_steps and squeeze_baseline is not None and efforts:
                    confirm = effort_rise
                    thresholds["effort_confirm_used"] = round(confirm, 3)
                    if efforts[-1] < squeeze_baseline[0] + confirm:
                        stuck = 0          # the fingers did not really stop; that was sensor noise
                if stuck >= stalled_steps and not moved_at_all():
                    return jammed()
                if stuck >= stalled_steps:
                    if squeeze_baseline is not None and efforts and \
                            efforts[-1] - squeeze_baseline[0] < EFFORT_RISE_HELD:
                        thresholds["effort_weak_grasp"] = self._weak_grasp_note(efforts[-1] - squeeze_baseline[0])
                    return {"ok": True, "held": True, "gap": round(measured, 4), "steps": steps,
                            "stopped_by": "position", **trace(),
                            # This used to end "so something is between them". Twice in one episode the
                            # fingers had stalled at 59 and 67 mm over a 55 mm block, because they were
                            # resting ON it, and the model took the verdict at its word. A stall is a
                            # measurement; what it means is for the wrist image and the next lift to say.
                            "note": f"the fingers stopped following the command at {measured * 1000:.1f} mm "
                                    f"after {steps} steps. A stall can be the object between the fingertips, "
                                    f"or the fingers pressing down on top of something; compare the gap with "
                                    f"the object's size, and let the wrist image and the next lift's "
                                    f"closed_since_grasp_m decide"}
            # The loop can also run out before the stall test has two consecutive readings to compare,
            # which is what happens with something thin: the fingers track the command all the way down
            # and only the last step is short. So the final resting gap decides it instead.
            if measured > HELD_GAP_M:
                if not moved_at_all():
                    return jammed()
                return {"ok": True, "held": True, "gap": round(measured, 4), "steps": steps,
                        "stopped_by": "resting gap", **trace(),
                        "note": f"the fingers came to rest {measured * 1000:.1f} mm apart instead of "
                                f"touching, so something is between them"}
            if measured <= EMPTY_GAP_M:
                return {"ok": True, "held": False, "gap": round(measured, 4), "steps": steps,
                        "stopped_by": "fully closed", **trace(),
                        "note": "the fingers closed the whole way and the gripper never pushed against "
                                "anything, so nothing is between them"}
            return {"ok": True, "held": None, "gap": round(measured, 4), "steps": steps,
                    "stopped_by": "ambiguous gap", **trace(),
                    "note": f"the fingers came to rest {measured * 1000:.1f} mm apart, inside the band where "
                            f"an empty gripper and a single thin sheet look alike by position, and the "
                            f"gripper effort never rose either. Look at the image."}
        except Exception as e:  # noqa: BLE001
            print(f"[PlayRobotEnv] gradual gripper close failed: {e}")
            return {"ok": False, "held": None, "gap": measured, "steps": steps, "note": str(e)}
        finally:
            sdk_log.setLevel(previous_level)

    def get_gripper_state(self) -> Dict[str, Optional[float]]:
        """
        Get gripper state matching RealRobotEnv interface.
        Returns: {'effort': float_or_None, 'gap': float_or_None}
        - effort: gripper motor current (joint_6)
        - gap: gripper eef position from SDK
        """
        effort = None
        gap = None

        try:
            gap = self.robot.get_gripper_state()  # single float from get_eef_pos()[0]

            # The gripper motor has its own call. It was previously looked for as 'joint_6' inside
            # get_joint_efforts(), which is built from the six ARM joints and so tops out at joint_5:
            # the key never existed, the lookup always missed, and every grasp in every run reported
            # 0.000. The hardware was publishing it the whole time.
            getter = getattr(self.robot, "get_gripper_effort", None)
            if getter is not None:
                value = getter()
                effort = None if value is None else abs(float(value))
            if effort is None:
                efforts = self.robot.get_joint_efforts()
                if efforts is not None and 'joint_6' in efforts:
                    effort = abs(efforts['joint_6'])
        except Exception as e:
            print(f"[PlayRobotEnv] Get gripper state failed: {e}")

        return {'effort': effort, 'gap': gap}

    def get_arm_total_effort(self) -> Optional[float]:
        """Get total effort (current) of arm joints."""
        try:
            efforts = self.robot.get_joint_efforts()
            if efforts is None:
                return None
            values = list(efforts.values())
            arm_values = values[:6] if len(values) >= 6 else values
            return sum(abs(v) for v in arm_values)
        except Exception:
            return None

    # ==================== Camera ====================

    def _zoom_frame(self, rgb: np.ndarray, depth: np.ndarray,
                    zoom: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply digital zoom to RGB + depth pair."""
        if zoom <= 1.0:
            return rgb, depth
        h, w = rgb.shape[:2]
        crop_w, crop_h = int(w / zoom), int(h / zoom)
        x0, y0 = (w - crop_w) // 2, (h - crop_h) // 2
        rgb_z = cv2.resize(
            rgb[y0:y0 + crop_h, x0:x0 + crop_w], (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
        depth_z = cv2.resize(
            depth[y0:y0 + crop_h, x0:x0 + crop_w], (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        return rgb_z, depth_z

    def _zoom_rgb(self, rgb: np.ndarray, zoom: float) -> np.ndarray:
        """Apply digital zoom to RGB only."""
        if zoom <= 1.0:
            return rgb
        h, w = rgb.shape[:2]
        crop_w, crop_h = int(w / zoom), int(h / zoom)
        x0, y0 = (w - crop_w) // 2, (h - crop_h) // 2
        return cv2.resize(
            rgb[y0:y0 + crop_h, x0:x0 + crop_w], (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

    def get_head_camera_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get head camera RGB and depth.
        Returns: (bgr_image, depth_mm) — depth converted to millimeters
        """
        if self.robot.head_camera is None:
            return None, None
        try:
            rgb, depth = next(self.robot.head_camera)
            if rgb is None or depth is None:
                return None, None
            rgb, depth = self._zoom_frame(rgb, depth, self._head_zoom)
            # Convert D405 raw depth to millimeters
            depth_mm = (depth.astype(np.float32) * D405_DEPTH_RAW_TO_MM).astype(np.uint16)
            return rgb, depth_mm
        except Exception as e:
            print(f"[PlayRobotEnv] Head camera failed: {e}")
            return None, None

    def get_handeye_camera_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get hand-eye (wrist) camera RGB + depth (with zoom)."""
        if self.robot.right_wrist_camera is None:
            return None, None
        try:
            rgb, depth = next(self.robot.right_wrist_camera)
            if rgb is None:
                return None, None
            rgb, depth = self._zoom_frame(rgb, depth, self._wrist_zoom)
            # Convert D405 raw depth to millimeters (same sensor as head camera)
            depth_mm = (depth.astype(np.float32) * D405_DEPTH_RAW_TO_MM).astype(np.uint16) if depth is not None else None
            return rgb, depth_mm
        except Exception as e:
            print(f"[PlayRobotEnv] Hand-eye camera failed: {e}")
            return None, None

    # ==================== Second arm ====================
    #
    # Named "left" throughout because that is what the calibration files call this arm
    # (hand_eye_extrinsics_left.json, and base_to_base_extrinsics.json carries left_base -> world).
    # It is NOT play_sdk's `.left` handle, which holds the world arm — see the note in __init__.

    @property
    def has_second_arm(self) -> bool:
        return self.second_arm is not None

    def get_left_handeye_camera_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Second arm's wrist camera: RGB + depth in mm. Same treatment as the world arm's wrist."""
        if not self.has_second_arm or self.robot.left_wrist_camera is None:
            return None, None
        try:
            rgb, depth = next(self.robot.left_wrist_camera)
            if rgb is None:
                return None, None
            rgb, depth = self._zoom_frame(rgb, depth, self._wrist_zoom)
            depth_mm = (depth.astype(np.float32) * D405_DEPTH_RAW_TO_MM).astype(np.uint16) \
                if depth is not None else None
            return rgb, depth_mm
        except Exception as e:
            print(f"[PlayRobotEnv] Second-arm hand-eye camera failed: {e}")
            return None, None

    def get_left_arm_pose(self, tool: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Second arm's TCP pose, in the SECOND ARM'S OWN BASE — not the world frame.

        Deliberately not converted here. Whoever uses it has to compose base_to_base, and the
        8 mm loop-closure that composition carries is the arms' own kinematics, so the caller should
        know it is crossing bases. Perception does this in one place, for the camera transform.
        """
        if not self.has_second_arm:
            return None
        try:
            pose = self.second_arm.get_end_pose(tool=tool)
            if pose is None:
                return None
            position, orientation = pose
            return R.from_quat(orientation).as_matrix(), np.array(position)
        except Exception as e:
            print(f"[PlayRobotEnv] Error getting second arm pose: {e}")
            return None

    # ==================== TF ====================

    def get_head_camera_transform(self) -> Optional[np.ndarray]:
        """
        Get 4x4 transform from head camera to base.
        Static from calibration JSON (Play has no live TF service).
        """
        return self.T_head2base.copy()

    # ==================== Utility ====================

    def _wait_with_check(
        self, duration: float,
        check_abort: Optional[Callable[[], bool]] = None,
    ) -> bool:
        interval = 0.05
        elapsed = 0.0
        while elapsed < duration:
            if check_abort and check_abort():
                return False
            time.sleep(interval)
            elapsed += interval
        return True

    def disconnect(self):
        """Clean up resources."""
        if self.robot.head_camera:
            self.robot.head_camera.stop()
        if self.robot.right_wrist_camera:
            self.robot.right_wrist_camera.stop()
        print("[PlayRobotEnv] Disconnected")
