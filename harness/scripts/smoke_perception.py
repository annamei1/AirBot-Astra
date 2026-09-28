"""
Step-3 smoke test, no robot: run find()/measure() on a captured frame and show the overlay.

    python -m harness.scripts.smoke_perception --frame harness/logs/frames/<ts> --label "black block"

Pass criteria: one id per visible instance, xyz within the workspace, ids drawn on the right objects
(written next to the frame as *_overlay.jpg). Read height_status rather than the height alone:
"measured" is a real height, "unreliable" means the depth inside that object is contaminated (dark or
glossy surfaces do this) and only its x, y should be trusted.
"""
import argparse
import os

import cv2
import numpy as np

from harness import config
from harness.perception_tools import Perception, ReplayEnv


def _handeye():
    from perception.position_calculator import HandEyeCalculator
    intr, extr = config.load_hand_eye_calibration()
    return HandEyeCalculator(intr, extr)


def build_head_calc():
    from perception.position_calculator import create_head_calculator
    intr, _ = config.load_head_camera_calibration()
    if config.HEAD_ZOOM > 1.0:   # digital-zoom correction for a cropped head image
        s, W, H = config.HEAD_ZOOM, intr.get("width", 640), intr.get("height", 480)
        intr = {"fx": intr["fx"] * s, "fy": intr["fy"] * s,
                "cx": (intr["cx"] - W * (1 - 1 / s) / 2) * s, "cy": (intr["cy"] - H * (1 - 1 / s) / 2) * s,
                "width": W, "height": H}
    return create_head_calculator(intrinsics=intr, calib_dir=config.CALIB_DIR, enable_z_compensation=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True,
                    help="path STEM, without the _rgb.png suffix, e.g. harness/logs/frames/20260915-2210")
    ap.add_argument("--label", default="black block")
    ap.add_argument("--camera", default="head", help="which replayed camera to segment in")
    ap.add_argument("--box", default=None, help="x0,y0,x1,y1 read off the image, as the model would give")
    ap.add_argument("--point", default=None, help="u,v: with --label it aims find(), alone it runs measure()")
    args = ap.parse_args()

    # Tab completion hands you the stem with a trailing underscore, which then looks for "..__rgb.png".
    stem = args.frame.rstrip("_")
    if not os.path.exists(f"{stem}_rgb.png"):
        raise SystemExit(f"no frame at {stem}_rgb.png\n"
                         f"give the stem without any suffix, e.g. "
                         f"{os.path.join(os.path.dirname(stem) or '.', '20260916-054215')}")
    from perception.sam3_segmenter import create_segmenter
    seg = create_segmenter(config.SAM3_CHECKPOINT, confidence=config.SAM3_CONFIDENCE)
    env = ReplayEnv.from_stem(stem)
    per = Perception(env, seg, build_head_calc(), table_z=config.TABLE_SURFACE_Z,
                     handeye_calc=_handeye() if env.has_wrist else None)
    print(f"cameras replayed: {per.cameras()}")

    box = [int(v) for v in args.box.split(",")] if args.box else None
    point = [int(v) for v in args.point.split(",")] if args.point else None
    if point and not args.label:
        print(per.measure(point[0], point[1], camera=args.camera).text)
        return
    res = per.find(args.label, camera=args.camera, box=box, point=point)
    print(res.text)
    if res.image is not None:
        out = f"{args.frame}_overlay.jpg"
        cv2.imwrite(out, res.image)
        print("overlay →", out)
    if point:
        print("\nmeasure() at that pixel, which needs no label and no segmentation:")
        print(per.measure(point[0], point[1], camera=args.camera).text)


if __name__ == "__main__":
    main()
