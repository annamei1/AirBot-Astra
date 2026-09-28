"""
Capture one head-camera RGB + depth frame (+ the head→base transform) so perception tools can be
developed offline with ReplayEnv, without the robot.

    python -m harness.scripts.capture_frame            # → harness/logs/frames/<ts>_{rgb.png,depth.npy,tf.npy}
    python -m harness.scripts.smoke_perception --frame harness/logs/frames/<ts>

**The arm does not move**, so the capture shows the scene a real episode starts in. Pass --move-to-home
to travel to the configured home pose first.
"""
import argparse
import time

import cv2
import numpy as np

from harness import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--move-to-home", action="store_true",
                    help="travel to the configured home pose (arm.home_joint) first, instead of capturing where the arm is")
    args = ap.parse_args()

    from harness.robot_env import HarnessEnv
    from play.config import HEAD_CAMERA_SERIAL, WRIST_CAMERA_SERIAL, ARM_PORT

    # Cameras and arm come from config/play_config.json.
    print(f"arm port {ARM_PORT} · head camera {HEAD_CAMERA_SERIAL} · wrist camera {WRIST_CAMERA_SERIAL}")
    env = HarnessEnv(rest="home" if args.move_to_home else "none",
                     move_on_start=args.move_to_home)
    try:
        time.sleep(1.0)
        rgb, depth = env.get_head_camera_frame()
        if rgb is None or depth is None:
            raise SystemExit("no head-camera frame — check the serial above and the USB connection")
        tf = env.get_head_camera_transform()
        config.FRAME_DIR.mkdir(parents=True, exist_ok=True)
        stem = config.FRAME_DIR / time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f"{stem}_rgb.png", rgb)
        np.save(f"{stem}_depth.npy", depth)
        np.save(f"{stem}_tf.npy", tf)
        valid = depth[depth > 0]
        print(f"saved {stem}_{{rgb.png,depth.npy,tf.npy}}\n"
              f"  rgb {rgb.shape}  depth {depth.shape}  valid {100.0 * valid.size / depth.size:.0f}%  "
              f"median {np.median(valid):.0f} mm (≈ camera-to-table distance)")

        wrist = env.get_handeye_camera_frame()
        wrist_rgb, wrist_depth = wrist if isinstance(wrist, tuple) else (wrist, None)
        if wrist_rgb is None:
            print("  wrist camera: NO FRAME — close-up measurements over an object will fail")
        else:
            cv2.imwrite(f"{stem}_wrist.png", wrist_rgb)
            if wrist_depth is not None:
                np.save(f"{stem}_wrist_depth.npy", wrist_depth)
            pose = env.get_arm_pose()
            if pose is not None:
                # A wrist view is only meaningful with the arm pose it was taken at:
                # T_cam2base = T_gripper2base (saved here) @ hand-eye extrinsics.
                T_g2b = np.eye(4)
                T_g2b[:3, :3] = np.asarray(pose[0], float)
                T_g2b[:3, 3] = np.asarray(pose[1], float).flatten()
                np.save(f"{stem}_arm_pose.npy", T_g2b)
            print(f"  wrist camera OK {wrist_rgb.shape} → {stem}_wrist.png "
                  f"(+depth, +arm pose, so the wrist view can be replayed offline)")

        print(f"\nnext: python -m harness.scripts.smoke_perception --frame {stem} --label \"black block\"")
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:
        env.disconnect()
        import os
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)          # realsense threads otherwise keep the interpreter alive at exit;
                             # flush first, because _exit skips it and a pipe would lose everything


if __name__ == "__main__":
    main()
