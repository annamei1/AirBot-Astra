"""
What the acting model is told about the gripper it is using: measured facts, no advice.

The model knows the size of every object it finds and never knew the size of its own fingers. On
2026-09-22 it lowered a pair of fingers 59 mm wide into a drawer 125 mm wide, aiming at the centre,
and hit the wall; nothing in the harness had told it the width. The facts come from a JSON file
(harness/grippers/airbot_g2.json for the stock fingers; pass another with --gripper-facts), so
different fingers bring their own numbers. Where the fingertips sit relative to the reported tool position
is not here: the model learns that by touching, as before.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_gripper_facts(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finger_sentence(f: Dict[str, Any]) -> str:
    w, t = f["width_across_closing_mm"], f["thickness_along_closing_mm"]
    return (f"{f['finger_length_mm']:.0f} mm long from its mount to the tip; across the closing direction "
            f"{w['max']:.0f} mm wide at its widest ({w['at_mm_from_mount']:.0f} mm from the mount), {w['mid_length']:.0f} mm at "
            f"mid-length and {w['tip']:.0f} mm at the tip; along the closing direction {t['max']:.0f} mm thick at the mount end, "
            f"{t['mid_length']:.0f} mm at mid-length and {t['tip']:.0f} mm at the tip")


def gripper_paragraph(f: Dict[str, Any], arms: Optional[list] = None) -> str:
    where = "on each arm" if arms and len(arms) > 1 else "on this arm"
    head = f"THE GRIPPER {where} ({f.get('name', 'mounted gripper')}): a {f.get('gripper_type', 'parallel two-finger')} gripper"
    fingers = f.get("fingers") or {}
    if f.get("fingers_identical", True) or not fingers.get("right"):
        s = f"{head}; the two fingers are mirror images of each other. Each finger is {_finger_sentence(f)}."
    else:
        s = (f"{head}; the two fingers are NOT alike. The left finger is {_finger_sentence(fingers['left'])}. "
             f"The right finger is {_finger_sentence(fingers['right'])}.")
    s += f" The contact faces open to {f['max_opening_mm']:.0f} mm apart."
    if f.get("description"):
        s += f" {f['description'].strip()}"
    return s
