"""
Live window: every camera side by side, plus what the model is doing right now.

The point is to watch the two cameras cooperate. Each tile is that camera's live feed with the known
objects drawn on it, so you can see the head camera holding a position while the wrist camera is
carried to it. The panel underneath shows the model's last sentence, the tool it is running, the
plan with its verified steps, and a running count, so a pause in the arm is explained rather than
mysterious.

OpenCV cannot draw CJK glyphs, so panel text is transliterated to ASCII; the terminal keeps the
original. q or Esc in the window sets the abort event, exactly like Ctrl+C.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (26, 26, 26)
DIM = (150, 150, 150)
OK_C = (120, 220, 120)
BAD_C = (110, 110, 250)
BUSY_C = (90, 200, 250)


def _ascii(s: str, limit: int = 300) -> str:
    """Drop what OpenCV cannot draw. A line that was entirely CJK becomes a pointer to the terminal
    rather than a row of dots, which reads as noise."""
    kept = "".join(ch for ch in str(s) if 32 <= ord(ch) < 127).strip()
    if not kept:
        return "[non-latin text, see terminal]"
    return kept[:limit]


def _wrap(s: str, width: int) -> List[str]:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


class LiveView:
    def __init__(self, perception, abort: Optional[threading.Event] = None,
                 title: str = "", tile: tuple = (640, 480), panel_h: int = 210):
        title = title or "harness — " + " | ".join(perception.cameras())
        self.perception = perception
        self.abort = abort
        self.title = title
        self.tw, self.th = tile
        self.panel_h = panel_h
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.events = deque(maxlen=6)
        self.busy = "starting"
        self.model_text = ""
        self.plan: List[Dict[str, Any]] = []
        self.stats = {"model_calls": 0, "tool_calls": 0}
        self.t0 = time.time()

    # ---------------- what the agent tells us ----------------

    def set_busy(self, text: str, color=BUSY_C):
        with self._lock:
            self.busy, self.busy_color = text, color

    def set_model_text(self, text: str):
        with self._lock:
            self.model_text = text

    def set_plan(self, plan: List[Dict[str, Any]]):
        with self._lock:
            self.plan = list(plan or [])

    def set_stats(self, **kw):
        with self._lock:
            self.stats.update(kw)

    def note(self, text: str, ok: Optional[bool] = None):
        with self._lock:
            self.events.append((time.time(), text, ok))

    # ---------------- the window ----------------

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            pass

    def _loop(self):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, 1280, self.th + self.panel_h)
        while not self._stop.is_set():
            try:
                frame = self._compose()
                if frame is not None:
                    cv2.imshow(self.title, frame)
            except Exception as e:  # noqa: BLE001  the view must never take the robot down
                print(f"[viz] {type(e).__name__}: {e}")
            key = cv2.waitKey(40) & 0xFF
            if key in (ord("q"), 27) and self.abort is not None:
                print("\n[viz] q pressed — aborting")
                self.abort.set()
        cv2.destroyAllWindows()

    def _compose(self) -> Optional[np.ndarray]:
        tiles = []
        for name in self.perception.cameras():
            view = self.perception.rig.grab(name)
            if view is None:
                img = np.full((self.th, self.tw, 3), 40, np.uint8)
                cv2.putText(img, f"{name}: no frame", (16, self.th // 2), FONT, 0.7, DIM, 2)
                tiles.append(img)
                continue
            drawn = self.perception.overlay(name, view=view, masks=False)
            img = cv2.resize(drawn if drawn is not None else view.rgb, (self.tw, self.th))
            cv2.rectangle(img, (0, 0), (self.tw, 26), BG, -1)
            eye = "" if view.eye_xyz is None else f"  eye [{view.eye_xyz[0]:+.3f} {view.eye_xyz[1]:+.3f} {view.eye_xyz[2]:+.3f}]"
            cv2.putText(img, f"{name}{eye}", (8, 19), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        if not tiles:
            return None
        return np.vstack([np.hstack(tiles), self._panel(self.tw * len(tiles))])

    def _panel(self, width: int) -> np.ndarray:
        with self._lock:
            busy, model_text = self.busy, self.model_text
            plan, events, stats = list(self.plan), list(self.events), dict(self.stats)
            color = getattr(self, "busy_color", BUSY_C)
        p = np.full((self.panel_h, width, 3), 18, np.uint8)
        cols = max(40, width // 9)
        y = 24
        cv2.putText(p, f"NOW: {_ascii(busy, cols)}", (10, y), FONT, 0.6, color, 2, cv2.LINE_AA)
        y += 24
        cv2.putText(p, f"t={time.time() - self.t0:5.0f}s   model calls {stats.get('model_calls', 0)}"
                       f"   tool calls {stats.get('tool_calls', 0)}", (10, y), FONT, 0.45, DIM, 1, cv2.LINE_AA)
        y += 22
        for line in _wrap(_ascii(model_text, 240), cols)[:2]:
            cv2.putText(p, line, (10, y), FONT, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
            y += 18
        y += 4
        for t, text, ok in events[-4:]:
            c = OK_C if ok else (BAD_C if ok is False else DIM)
            cv2.putText(p, _ascii(text, cols), (10, y), FONT, 0.42, c, 1, cv2.LINE_AA)
            y += 17
        if plan:
            x = width // 2 + 20
            cv2.putText(p, "PLAN", (x, 24), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            yy = 46
            for step in plan[:5]:
                mark = {"done": "[x]", "failed": "[!]", "skipped": "[-]"}.get(step["status"], "[ ]")
                c = {"done": OK_C, "failed": BAD_C}.get(step["status"], DIM)
                cv2.putText(p, f"{mark} {_ascii(step['step'], cols // 2)}", (x, yy), FONT, 0.42, c, 1, cv2.LINE_AA)
                yy += 17
        return p
