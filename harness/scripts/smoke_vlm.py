"""
Step-1 smoke test, no robot: does the model drive a find → pick → place loop over fake tools?

    python -m harness.scripts.smoke_vlm                 # synthetic image (block + bowl drawn with OpenCV)
    python -m harness.scripts.smoke_vlm --image x.jpg   # a real head-camera image

The stub tools answer like the real ones: distinct ids and 3D poses per label, empty result + hint
when a label matches nothing. Pass: the model finds both objects, then picks the block by the id the
tool returned and places it into the bowl by id. Prints latency and token usage per round.
"""
import argparse
import json

import cv2
import numpy as np

from harness import config
from harness.tools import ToolRegistry, ToolResult
from harness.vlm_client import Session, make_client_from_env

MAX_ROUNDS = 8

# what the fake perception "knows" — same shape as perception_tools.Perception.find()
SCENE = {
    "black_block_1": {"keys": ("block", "brick", "cube", "积木"), "label": "black block",
                      "xyz": [0.32, 0.12, -0.053], "yaw_deg": 88.0, "extent_m": [0.11, 0.05],
                      "height_above_support_m": 0.013, "area_cm2": 52.0},
    "green_bowl_1": {"keys": ("bowl", "container", "dish", "碗"), "label": "green bowl",
                     "xyz": [0.30, -0.10, -0.030], "yaw_deg": 0.0, "extent_m": [0.15, 0.15],
                     "height_above_support_m": 0.036, "area_cm2": 180.0},
}


def synthetic_scene() -> np.ndarray:
    img = np.full((480, 640, 3), 235, np.uint8)
    cv2.rectangle(img, (150, 220), (260, 270), (30, 30, 30), -1)          # black block
    cv2.circle(img, (450, 260), 70, (60, 160, 60), -1)                     # green bowl
    cv2.circle(img, (450, 260), 50, (90, 200, 90), -1)
    return img


class FakeTools:
    def __init__(self):
        self.holding = None
        self.picked = None
        self.placed = None

    def find(self, label: str) -> ToolResult:
        key = (label or "").lower()
        objects = [
            {"id": oid, "label": o["label"], "xyz": o["xyz"], "yaw_deg": o["yaw_deg"],
             "extent_m": o["extent_m"], "area_cm2": o["area_cm2"],
             "height_above_support_m": o["height_above_support_m"], "height_status": "measured"}
            for oid, o in SCENE.items() if any(k in key or k in (label or "") for k in o["keys"])
        ]
        if not objects:
            return ToolResult.json({"ok": True, "label": label, "objects": [],
                                    "hint": "nothing matched that text; try another wording"})
        return ToolResult.json({"ok": True, "label": label, "objects": objects})

    def pick(self, object_id: str) -> ToolResult:
        if object_id not in SCENE:
            return ToolResult.error(f"unknown object id '{object_id}'. Call find() first.")
        if self.holding:
            return ToolResult.error(f"gripper already holds '{self.holding}'")
        self.holding = self.picked = object_id
        return ToolResult.json({"ok": True, "object_id": object_id, "gripper_gap_m": 0.031,
                                "note": "grasp verified by gripper gap"})

    def place(self, target_id: str = None, xyz=None, z_offset: float = 0.0) -> ToolResult:
        if not self.holding:
            return ToolResult.error("nothing in the gripper; pick() first")
        if target_id and target_id not in SCENE:
            return ToolResult.error(f"unknown target id '{target_id}'")
        held, self.holding = self.holding, None
        self.placed = (held, target_id or xyz)
        return ToolResult.json({"ok": True, "object_id": held, "target": target_id or xyz})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    args = ap.parse_args()
    img = cv2.imread(args.image) if args.image else synthetic_scene()
    if img is None:
        raise SystemExit(f"cannot read {args.image}")

    fake = FakeTools()
    reg = ToolRegistry()
    reg.register("find", "Detect all objects matching a short English label in the current image. "
                         "Returns one id per instance with its 3D pose in the robot base frame.",
                 {"properties": {"label": {"type": "string"}}, "required": ["label"]}, fake.find)
    reg.register("pick", "Grasp an object by the id find() returned.",
                 {"properties": {"object_id": {"type": "string"}}, "required": ["object_id"]}, fake.pick)
    reg.register("place", "Put the held object onto/into a target by the id find() returned.",
                 {"properties": {"target_id": {"type": "string"},
                                 "xyz": {"type": "array", "items": {"type": "number"},
                                         "minItems": 3, "maxItems": 3},
                                 "z_offset": {"type": "number"}}}, fake.place)

    client = make_client_from_env()
    session = Session("You control a robot arm through tools. Describe the image in one sentence, then act. "
                      "Never guess coordinates: only ids returned by find() go into pick() and place().",
                      log_path=config.LOG_DIR / "smoke_vlm.jsonl", max_edge=config.IMAGE_MAX_EDGE)
    session.user("Task: put the black block into the green bowl.", images=[img])

    saw_tool_call = False
    for round_ in range(1, MAX_ROUNDS + 1):
        reply = client.step(session, tools=reg.schemas())
        session.assistant(reply)
        print(f"\n--- round {round_}  {reply.elapsed:.1f}s  usage={reply.usage}")
        if reply.text:
            print("text:", reply.text)
        if not reply.tool_calls:
            if saw_tool_call:
                print("no tool call this round — the model stopped on its own (read its text above)")
            else:
                print("no tool call at all → does this endpoint/model support function calling?")
            break
        saw_tool_call = True
        for c in reply.tool_calls:
            print("tool call:", c.name, json.dumps(c.arguments, ensure_ascii=False))
            res = reg.dispatch(c)
            print("      →", res.text[:200])
            session.tool_result(c.id, res.text)
        if fake.placed:
            break

    print()
    if fake.picked == "black_block_1" and fake.placed and fake.placed[1] == "green_bowl_1":
        print("PASS: found both objects, picked the block by id, placed it into the bowl by id.")
    elif fake.picked:
        print(f"PARTIAL: picked {fake.picked}, but the place step did not complete.")
    else:
        print("FAIL: never called pick() with an id returned by find(). Read the rounds above.")


if __name__ == "__main__":
    main()
