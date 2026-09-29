"""
Robot environment for the harness: configurable rest pose, thread-safe camera access.

The arm does not move on connect, and its rest pose defaults to the arm's own zero position, not a
hand-tuned pose that already points the wrist camera at the table: the head camera finds an object
anywhere on the table and the wrist camera is brought to it.

Both cameras are also read by the live display thread while perception is reading them, so every
frame grab is serialised through one lock.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from play.config import GRIPPER_OPEN_WAIT_SEC, HOME_JOINT, ZERO_JOINT
from robot.play_env import PlayRobotEnv
from robot.feedback_wait import wait_for_target

REST_POSES = {"zero": ZERO_JOINT, "home": HOME_JOINT}


class HarnessEnv(PlayRobotEnv):
    """PlayRobotEnv that stays still on connect and rests wherever you tell it to.

    rest="zero"  the arm's zero joint position (default): the wrist camera starts pointing away from
                 the work area, so the two cameras have to cooperate to reach anything.
    rest="home"  the hand-tuned arm.home_joint pose from config/play_config.json.
    rest="none"  never travel to a rest pose; retreats after a failure just open the gripper.
    """

    def __init__(self, rest: str = "zero", move_on_start: bool = False, **kwargs):
        if rest not in ("zero", "home", "none"):
            raise ValueError(f"rest must be zero, home or none, not {rest!r}")
        self._rest = rest
        self._cam_lock = threading.RLock()
        self._connecting = True
        super().__init__(**kwargs)
        self._connecting = False
        print(f"[harness] connected without moving; rest pose is '{rest}'")
        if move_on_start and rest != "none":
            self.reset_position()

    # PlayRobotEnv.__init__ and reset_position both funnel through _go_home
    def _go_home(self):
        """Every arm to its rest pose — not just the one the convenience methods drive.

        This used to move only the world arm, because PlayRealRobot's set_joint_positions
        delegates to a single slot. The second arm then started each session wherever the last
        one left it, which quietly breaks the rule both arms are supposed to share: an arm starts
        from its zero joint pose, and a close-up view is earned during the task rather than set up
        beforehand. An arm parked somewhere convenient is a pre-positioned camera by another name,
        and it also makes runs non-comparable, since its wrist view differs from session to session.
        """
        if self._connecting or self._rest == "none":
            return
        joints = REST_POSES[self._rest]
        print(f"[harness] moving to the '{self._rest}' rest pose")
        if not self.open_gripper():
            raise RuntimeError('Rest aborted: gripper opening not confirmed')
        if not self.robot.set_joint_positions(joints, blocking=True):
            raise RuntimeError('Rest command failed; execution state is uncertain')
        if not wait_for_target(self.robot.get_joint_q, joints, 0.01745):
            raise RuntimeError('Rest joint positions not confirmed within 15 s')
        # The second arm has no convenience method of its own; drive its handle directly. Zero is
        # defined the same way for both arms, while the tuned 'home' pose was measured against the
        # first arm's mount and means nothing on the other one.
        if getattr(self, "second_arm", None) is not None:
            print(f"[harness] second arm to its zero joint pose")
            try:
                self.second_arm.set_gripper(position=self.gripper_open_width)
                time.sleep(GRIPPER_OPEN_WAIT_SEC)
                self.second_arm.set_joint_positions(list(ZERO_JOINT), blocking=True)
            except Exception as exc:  # noqa: BLE001  never let the second arm block the first
                print(f"[harness] could not rest the second arm: {type(exc).__name__}: {exc}")

    # ---- serialise camera access: perception and the live view both read these ----

    def get_head_camera_frame(self):
        with self._cam_lock:
            return super().get_head_camera_frame()

    def get_handeye_camera_frame(self):
        with self._cam_lock:
            return super().get_handeye_camera_frame()

    def get_left_handeye_camera_frame(self):
        with self._cam_lock:
            return super().get_left_handeye_camera_frame()
