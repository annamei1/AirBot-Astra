"""
One arm, seen from the world frame.

Everything the harness has ever measured lives in one frame: the first arm's base. The head camera's
extrinsics, the workspace envelope, every coordinate in every log. A second arm does not share that
frame — it has its own base, and its controller only accepts poses expressed in it. So there has to be
exactly one place where a world coordinate becomes that arm's coordinate, and this is it.

An ArmView presents an arm with the same method names the motion layer already uses on the env, and
always in the world frame. GuardedPath takes one and does not know or care which arm it drives;
follow_path, move_until_contact and the rest take the name of an arm and get one of these. Nothing
downstream branches on which arm is acting.

Two implementations, deliberately not one:

  WorldArm   the arm whose base IS the world frame. Every method forwards to PlayRobotEnv, which is
             the code that has driven this arm for every run so far. No transform, no reimplementation,
             nothing new to go wrong on the path that already works.

  SecondArm  any other arm. The same interface, through the measured base-to-base transform, driving
             its own play_sdk handle directly rather than through the convenience methods — those are
             hardwired to one slot (see the note in robot/play_env.py) and would drive the wrong arm.

THE ACCURACY THAT MATTERS, because it decides how this should be used:

The two bases were tied together to about 8 mm, and that residual is the arms' own kinematics, not the
calibration — it was measured as 8.6 mm on one chain and 2.9 mm on the other. So a point the HEAD
camera measures and this arm is then commanded to lands with that error in it: good enough to reach
for, not good enough to pinch a single layer of cloth.

But a point this arm's OWN wrist camera measures goes out through the same transform it came in
through, and the base-to-base error cancels exactly — what is left is that arm's hand-eye residual,
1.65 mm. That is not a detail: it means the right way to use the second arm is the way the first one
already works. Look with the arm that is about to act, then act.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from robot.play_env import PlayRobotEnv, grasp_quaternion, tool_angles_from_matrix


class _SharedWithEnv:
    """Methods taken whole from PlayRobotEnv instead of reimplemented.

    They are written against a small interface — get_gripper_state, close_gripper, robot.set_gripper,
    and two pure helpers — which is exactly what an ArmView provides, so they run unchanged on either
    arm. Copying them would be worse than it looks: the incremental closure is the most carefully
    measured code here, and the thing a drifting copy would get wrong is how one layer of cloth is
    told apart from an empty gripper, which is the whole task.
    """
    close_gripper_gradually = PlayRobotEnv.close_gripper_gradually
    # staticmethod(...) is not decoration for its own sake. Reading `PlayRobotEnv._effort_trace` off the
    # class unwraps the descriptor and hands back a PLAIN function; storing that here would re-bind it
    # as an instance method and pass `self` as its first argument. That is not hypothetical — it cost
    # the first dual-arm run its only left-hand grasp, as
    # "_weak_grasp_note() takes 1 positional argument but 2 were given", at the exact moment the fingers
    # closed on the cloth.
    _effort_trace = staticmethod(PlayRobotEnv.__dict__["_effort_trace"].__func__)
    _weak_grasp_note = staticmethod(PlayRobotEnv.__dict__["_weak_grasp_note"].__func__)


class WorldArm(_SharedWithEnv):
    """The arm whose base defines the world frame: PlayRobotEnv itself, under a common name."""

    def __init__(self, env, name: str = "right"):
        self.name = name
        self.env = env
        self.robot = env.robot            # GuardedPath reaches through this for servo/pose calls
        self.T_base2world = np.eye(4)

    # ---- state ----
    # This arm's base IS the world frame, so both conversions are the identity. They exist so that
    # callers (the reach envelope, above all) can convert without asking which arm they hold.
    def to_world(self, p):               return np.asarray(p, float).flatten()[:3]
    def to_base(self, p):                return np.asarray(p, float).flatten()[:3]

    def get_tcp_position(self):          return self.env.get_tcp_position()
    def get_arm_pose(self):              return self.env.get_arm_pose()
    def get_tool_angles(self):           return self.env.get_tool_angles()
    def get_arm_total_effort(self):      return self.env.get_arm_total_effort()
    def get_gripper_state(self):         return self.env.get_gripper_state()

    # ---- motion ----
    def move_arm(self, position, yaw, wait=None, check_abort=None, pitch=0.0, roll=0.0) -> bool:
        kw = {} if wait is None else {"wait": wait}
        return self.env.move_arm(position, yaw, check_abort=check_abort, pitch=pitch, roll=roll, **kw)

    def open_gripper(self, gap: Optional[float] = None) -> bool:  return self.env.open_gripper(gap)
    def close_gripper(self) -> bool:                              return self.env.close_gripper()
    def set_gripper(self, position: float) -> bool:
        self.env.robot.set_gripper(position=position)
        return True

    def reset_position(self) -> bool:    return self.env.reset_position()

    # ---- what AtomicMotionExecutor asks of a robot_env, beyond the above ----
    def get_head_camera_transform(self):    return self.env.get_head_camera_transform()
    def get_handeye_camera_frame(self):     return self.env.get_handeye_camera_frame()


class SecondArm(_SharedWithEnv):
    """An arm whose base is not the world frame. Same interface; the transform lives here.

    It drives its play_sdk handle directly. The convenience methods on PlayRealRobot all delegate to
    one fixed slot, so calling them here would move the other arm with this arm's coordinates.
    """

    def __init__(self, env, handle, T_base2world, name: str = "left",
                 sdk_slot: str = "right", rest_joints: Optional[List[float]] = None):
        self.name = name
        self.env = env
        self.handle = handle
        self.sdk_slot = sdk_slot          # which play_sdk slot this arm sits in, for the gripper calls
        self.rest_joints = list(rest_joints or [0.0] * 6)
        self.T_base2world = np.asarray(T_base2world, float)
        self.T_world2base = np.linalg.inv(self.T_base2world)
        self.robot = _WorldFrameHandle(handle, self.T_base2world)

    # ---- frame conversion ----

    def to_world(self, p) -> np.ndarray:
        return (self.T_base2world @ np.append(np.asarray(p, float).flatten()[:3], 1.0))[:3]

    def to_base(self, p) -> np.ndarray:
        return (self.T_world2base @ np.append(np.asarray(p, float).flatten()[:3], 1.0))[:3]

    # ---- state ----

    def get_arm_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(rotation, position) in the WORLD frame."""
        pose = self.robot.get_end_pose()
        if pose is None:
            return None
        from scipy.spatial.transform import Rotation as R
        position, orientation = pose
        return R.from_quat(orientation).as_matrix(), np.array(position)

    def get_tcp_position(self) -> Optional[np.ndarray]:
        pose = self.robot.get_end_pose()
        return None if pose is None else np.array(pose[0])

    def get_tool_angles(self) -> Optional[Tuple[float, float, float]]:
        pose = self.get_arm_pose()
        return None if pose is None else tool_angles_from_matrix(pose[0])

    def get_arm_total_effort(self) -> Optional[float]:
        """Sum of |effort| over this arm's six joints.

        Read straight off this arm's handle. The env's version goes through PlayRealRobot's
        get_joint_efforts, whose keys CHANGE in dual-arm mode — joint_0..5 becomes left_joint_0..5
        plus right_joint_0..5 — and which then takes the first six values by dict order. That happens
        to be the world arm's six, which is luck, not a contract. Contact detection is built on this
        number, so this arm reads its own.
        """
        try:
            eff = self.handle.get_joint_eff()
            if not eff:
                return None
            return float(sum(abs(float(v)) for v in list(eff)[:6]))
        except Exception:  # noqa: BLE001  a failed current read must not stop the arm dead
            return None

    def get_gripper_state(self) -> Dict[str, Optional[float]]:
        effort = gap = None
        try:
            eef = self.handle.get_eef_pos()
            if eef is not None and len(eef):
                gap = float(eef[0])
            e = self.handle.get_eef_eff()
            if e is not None and len(e):
                effort = abs(float(e[0]))
        except Exception as exc:  # noqa: BLE001
            print(f"[{self.name}] get gripper state failed: {exc}")
        return {"effort": effort, "gap": gap}

    # ---- motion ----

    def move_arm(self, position, yaw: float, wait=None, check_abort=None,
                 pitch: float = 0.0, roll: float = 0.0) -> bool:
        """Move to a WORLD position with WORLD tool angles.

        yaw, pitch and roll mean here exactly what they mean for the other arm — the same
        grasp_quaternion builds them — because they are built in the world frame and only then
        rotated into this arm's base. An arm mounted facing the other way would otherwise silently
        read every angle backwards.
        """
        if check_abort and check_abort():
            return False
        quat = grasp_quaternion(yaw, pitch, roll)
        try:
            ok = self.robot.set_end_pose(position=[float(v) for v in position],
                                         orientation=list(quat), blocking=True)
            if not ok:
                here = self.get_tcp_position()
                where = ""
                if here is not None:
                    gap = float(np.linalg.norm(np.asarray(here, float) - np.asarray(position, float)))
                    where = (f"  实际停在 {[round(float(v), 4) for v in here]}，距目标 {gap * 1000:.1f} mm"
                             + ("：位置其实到了，没达成的是姿态" if gap < 0.005 else ""))
                print(f"[{self.name}] ⚠️ set_end_pose 规划失败/不可达: "
                      f"pos={[round(float(p), 4) for p in position]}, yaw={yaw:.3f}{where}")
            return bool(ok)
        except Exception as exc:  # noqa: BLE001
            print(f"[{self.name}] move failed: {exc}")
            return False

    def set_gripper(self, position: float) -> bool:
        try:
            self.handle.set_gripper(position=float(position))
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[{self.name}] set gripper failed: {exc}")
            return False

    def open_gripper(self, gap: Optional[float] = None) -> bool:
        import time
        from play.config import GRIPPER_OPEN_WAIT_SEC
        ok = self.set_gripper(self.env.gripper_open_width if gap is None else gap)
        time.sleep(GRIPPER_OPEN_WAIT_SEC)
        return ok

    def close_gripper(self) -> bool:
        import time
        from play.config import GRIPPER_CLOSE_WAIT_SEC
        ok = self.set_gripper(0.0)
        time.sleep(GRIPPER_CLOSE_WAIT_SEC)
        return ok

    def get_head_camera_transform(self):
        """The head camera is not on any arm, and its extrinsics are already in the world frame."""
        return self.env.get_head_camera_transform()

    def get_handeye_camera_frame(self):
        """THIS arm's wrist camera, not the other one's.

        The executor's refine path asks its robot_env for 'the wrist camera', meaning the one on the
        arm it is driving. Handing it the other arm's would be the worst kind of wrong: a plausible
        image of the right scene from the wrong place.
        """
        return self.env.get_left_handeye_camera_frame()

    def reset_position(self) -> bool:
        """Back to this arm's rest joints, gripper open.

        Zero, not the tuned home pose: home_joint was hand-tuned against the first arm's mount and
        means nothing on this one. Zero is defined the same way for both.
        """
        try:
            self.open_gripper()
            self.handle.set_joint_positions(self.rest_joints, blocking=True)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[{self.name}] reset failed: {exc}")
            return False


class _WorldFrameHandle:
    """A play_sdk arm handle that speaks the world frame, for GuardedPath to servo through.

    GuardedPath reads a quaternion out of get_end_pose and hands it straight back to
    servo_cart_pose, so the two have to agree about which frame they are in. Wrapping both here is
    what keeps that true; anything that moves this arm and is NOT wrapped here would be sending
    world coordinates to a controller expecting its own base.
    """

    def __init__(self, handle, T_base2world):
        self._h = handle
        self.T_b2w = np.asarray(T_base2world, float)
        self.T_w2b = np.linalg.inv(self.T_b2w)

    def _out(self, position, orientation):
        """this arm's base → world"""
        from scipy.spatial.transform import Rotation as R
        p = (self.T_b2w @ np.append(np.asarray(position, float).flatten()[:3], 1.0))[:3]
        q = R.from_matrix(self.T_b2w[:3, :3] @ R.from_quat(list(orientation)).as_matrix()).as_quat()
        return [float(v) for v in p], [float(v) for v in q]

    def _in(self, position, orientation):
        """world → this arm's base"""
        from scipy.spatial.transform import Rotation as R
        p = (self.T_w2b @ np.append(np.asarray(position, float).flatten()[:3], 1.0))[:3]
        q = R.from_matrix(self.T_w2b[:3, :3] @ R.from_quat(list(orientation)).as_matrix()).as_quat()
        return [float(v) for v in p], [float(v) for v in q]

    # ---- the five calls GuardedPath makes ----

    def get_end_pose(self):
        pose = self._h.get_end_pose()
        if pose is None:
            return None
        return self._out(pose[0], pose[1])

    def set_end_pose(self, position, orientation, blocking=True) -> bool:
        p, q = self._in(position, orientation)
        return bool(self._h.set_end_pose(p, q, blocking=blocking))

    def servo_cart_pose(self, position, orientation) -> None:
        p, q = self._in(position, orientation)
        self._h.servo_cart_pose(p, q)

    def switch_mode(self, mode):
        return self._h.switch_mode(mode)

    def set_speed_profile(self, *a, **kw):
        fn = getattr(self._h, "set_speed_profile", None)
        if fn is None:
            raise AttributeError("this arm handle has no set_speed_profile")
        return fn(*a, **kw)

    def set_gripper(self, position: float = 1.0, **kw) -> bool:
        return bool(self._h.set_gripper(position=position))


# ==================== construction ====================

WORLD_ARM = "right"      # the arm whose base is the world frame — what every calibration file calls it
SECOND_ARM = "left"


def make_arms(env) -> Dict[str, Any]:
    """Every arm on this rig, keyed by name, each one speaking the world frame.

    Always contains the world arm. Contains the second one only when the rig has it AND the two bases
    have been tied together, because without that transform there is no way to say where this arm is
    in the frame everything else is measured in — and an arm that cannot be located is worse than no
    arm at all.
    """
    arms: Dict[str, Any] = {WORLD_ARM: WorldArm(env, name=WORLD_ARM)}

    handle = getattr(env, "second_arm", None)
    if handle is None:
        return arms
    try:
        from harness import config
        T = config.load_base_to_base()
    except Exception as exc:  # noqa: BLE001
        print(f"[arms] second arm present but base-to-base failed to load ({exc}); "
              f"it stays out of reach")
        return arms
    if T is None:
        print("[arms] second arm present but the two bases have never been tied together; "
              "run calibration/play/joint_cal.py. It stays out of reach.")
        return arms

    from play.config import ZERO_JOINT
    arms[SECOND_ARM] = SecondArm(env, handle, T, name=SECOND_ARM, sdk_slot="right",
                                 rest_joints=ZERO_JOINT)
    print(f"[arms] two arms: {list(arms)} — '{SECOND_ARM}' drives through the measured base-to-base "
          f"transform, its base at {(np.asarray(T)[:3, 3] * 1000).round(1).tolist()} mm in the world frame")
    return arms
