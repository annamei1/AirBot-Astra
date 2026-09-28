# Calibration files / 标定文件

**These are the calibration results of OUR rig. They are correct only for our arms and cameras,
mounted exactly where they were when we calibrated. If you use different arms or cameras, or move
any of them, you must recalibrate in your own setup before running the harness.**

**这里是我们实验台的标定结果，只对我们这套机械臂和相机、并且是在标定时的安装位置下才正确。
如果你换了机械臂或相机，或者移动了其中任何一个，都必须在你自己的环境里重新标定，然后再运行 harness。**

## What each file is / 每个文件是什么

| File / 文件 | What it gives / 内容 | Used when / 何时用到 |
|---|---|---|
| `head_camera_extrinsics.json` | Pose of the fixed head (environment) camera in the right arm's base frame, and its intrinsics. 固定头部（环境）相机在右臂基座坐标系下的位姿，以及它的内参。 | Always. 始终需要。 |
| `hand_eye_extrinsics_right.json` | Pose of the right arm's wrist camera relative to the arm's end frame (hand-eye), and its intrinsics. 右臂腕部相机相对机械臂末端的位姿（手眼标定），以及内参。 | Always. 始终需要。 |
| `hand_eye_extrinsics_left.json` | The same for the left arm's wrist camera. 左臂腕部相机的手眼标定。 | Two-arm runs (`--dual-arm`). 双臂模式。 |
| `base_to_base_extrinsics.json` | Pose of the left arm's base in the right arm's base frame, so both arms share one world frame. 左臂基座在右臂基座坐标系下的位姿，让两条臂共用一个世界坐标系。 | Two-arm runs (`--dual-arm`). 双臂模式。 |

The world frame is the **right arm's base**: x forward, y left, z up, metres. Each file also records
the camera serial number it was made with; the serials the harness opens are set in
`config/play_config.json` (`cameras.head_serial`, `cameras.wrist_serial`, `second_arm.wrist_serial`),
together with the arm ports.

世界坐标系是**右臂的基座**：x 向前、y 向左、z 向上，单位米。每个文件里也记录了标定时所用相机的序列号；
harness 实际打开哪台相机由 `config/play_config.json` 决定（`cameras.head_serial`、`cameras.wrist_serial`、
`second_arm.wrist_serial`），机械臂端口也在那里设置。

## Our rig / 我们的实验台

- Two AirBot Play arms (firmware / SDK **v5.1.6**) with G2 grippers, ports 50050 (right) and 50052 (left).
  两台 AirBot Play 机械臂（固件 / SDK **v5.1.6**），G2 夹爪，端口 50050（右）和 50052（左）。
- Three Intel RealSense D405: one fixed head camera about 0.5 m above the table, one wrist camera on each arm.
  三台 Intel RealSense D405：一台固定在桌面上方约 0.5 m 的头部相机，每条臂各一台腕部相机。

## When to recalibrate / 什么时候必须重新标定

- New arms, new cameras, or a different rig. 换了机械臂、相机，或者是另一套实验台。
- A camera or an arm base was moved, bumped or re-mounted — even slightly. A 2–3° knock on the head
  camera already moves what it measures by several millimetres.
  相机或机械臂基座被移动、碰到或重新安装过，哪怕只是轻微的。头部相机被碰歪 2–3°，测量结果就会偏几毫米。
- The contact probe (`python -m harness.scripts.probe_contact`) reports that the head and wrist cameras
  disagree on the height of the same spot by much more than they used to.
  接触探针（`python -m harness.scripts.probe_contact`）显示头部相机和腕部相机对同一点的高度差比以前大很多。

Calibrate the wrist cameras (hand-eye) first, then the head camera and the base-to-base transform,
which are both measured through a wrist camera.
先标定腕部相机（手眼），再标定头部相机和双臂基座之间的变换，因为后两者都是借助腕部相机测出来的。
