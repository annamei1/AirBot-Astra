"""Run the production SAM3 segmenter on an image, without robot hardware.

    python -m harness.scripts.check_sam3 image.jpg --prompt child
    python -m harness.scripts.check_sam3 image.jpg --box 430 200 700 680
"""
import argparse
import json
import time

import cv2

from harness import config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('image', help='Path to an image')
    ap.add_argument('--prompt', help='Text prompt; optional with --box')
    ap.add_argument('--box', nargs=4, type=float, metavar=('X0', 'Y0', 'X1', 'Y1'))
    ap.add_argument('--confidence', type=float, default=config.SAM3_CONFIDENCE)
    args = ap.parse_args()
    if not args.prompt and args.box is None:
        ap.error('provide --prompt or --box')
    image = cv2.imread(args.image)
    if image is None:
        ap.error(f'cannot read image: {args.image}')
    from perception.sam3_segmenter import create_segmenter
    start = time.monotonic()
    segmenter = create_segmenter(config.SAM3_CHECKPOINT, confidence=args.confidence)
    load_seconds = time.monotonic() - start
    start = time.monotonic()
    masks = (segmenter.segment_box(image, args.box, args.prompt) if args.box is not None
             else segmenter.segment(image, args.prompt))
    print(json.dumps({'device': segmenter.device, 'load_seconds': load_seconds,
                      'inference_seconds': time.monotonic() - start,
                      'mask_count': len(masks) if masks is not None else 0,
                      'mask_shape': list(masks.shape) if masks is not None else None}, indent=2))


if __name__ == '__main__':
    main()
