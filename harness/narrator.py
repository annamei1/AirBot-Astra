"""
A second copy of the model, watching the workspace the whole time and writing down what happens.

Every other tool observes when the arm stops. Between two of the model's decisions the world can
change on its own — a person rearranges things, an object rolls, the other arm works — and nothing
looks. A VLA sees every frame but remembers one or two of them; the model here sees frames only
at decision points but keeps a text history forever. The narrator closes the gap from the other
side: it sees the frames the acting model does not, and turns them into text the acting model
keeps.

How it runs. One thread grabs the head camera at a fixed rate into a buffer. Another takes every
frame that arrived since its last call, together with the very first frame of the episode and the
account so far, sends them to the model, and appends what comes back. The next window closes when
that call returns, so the window is as long as the model takes and never piles up; nothing here
chooses a duration. The first frame is the anchor every entry is written against. All other frames
are seen once and then live only as text — and on disk, for replay.

The account is a list of entries the harness keeps, not a text the model rewrites. The first
version asked for the whole account back every call, and a quiet window cost as much as a busy one:
eleven seconds and 666 tokens to retype three thousand characters in which nothing had changed.
Now the model replies only with what the window adds — an entry, a correction, or NO CHANGE — and
the harness stamps it with the window's time span and appends it. Consecutive NO CHANGE windows
merge into one line with one span. That is report-by-exception with a heartbeat, the same shape as
a SCADA outstation: the master holds the point table, the outstation reports what changed.

The calls form one conversation, not one prompt per window. On the route we use the prompt cache
pays out only when a previously processed prompt is a prefix of the new one (measured 2026-09-22),
so re-sending the account inside a fresh prompt every call cached nothing, 41 calls out of 41, and
cost more than the version it replaced. In a conversation every window is a new turn on top of the
last prompt, the earlier replies ARE the account, and only the new frames are paid at full price.
The old frames stay in the history until it passes a token budget; then a new conversation starts,
seeded with the anchor and the rendered account, and the frames are dropped in bulk — one cache
miss per generation instead of one per call.

The wrist cameras are a supplement, not a second narrator. The fixed camera is the account's frame
of reference; the wrist view moves with the arm, so in it things shift because the arm moved, and a
narrator that did not know that would report a still table as a moving one. So wrist frames go in
only for the seconds their arm was acting — a planned move, a guarded path, the gripper opening or
closing, going home — which the skills layer marks exactly, no threshold; they are labelled by arm
and camera, and the model is told what they are for: what happens between the fingers, which the
fixed camera cannot see. Off those seconds the wrist frames are recorded to disk and not sent.

What it knows about the task: nothing. It describes what moved and where it went, in the terms it
sees. The acting model reads the entries added since its previous tool result as `what_happened`
in every result, and the whole account in look(). When the narrator is unsure — two alike things
crossed, something was hidden — it says so, and that is the acting model's cue to look for itself.
"""
from __future__ import annotations

import json
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from harness.vlm_client import Session, VLMClient

# The camera rate. One frame a second follows a hand rearranging things on a table; the acting
# model can ask for a different rate per episode if a task needs it.
NARRATOR_HZ = 1.0
# Frames per narrator call are bounded only because the API has a per-request image limit, not
# to save anything. If a call took long enough to accumulate more than this, the window is sampled
# evenly and the entry says how many it covered.
NARRATOR_MAX_FRAMES = 20
# Wrist frames per arm per call, sampled the same way. They only ever cover the seconds the arm
# was acting, so a window rarely has more than a handful.
NARRATOR_MAX_WRIST_FRAMES = 8
# When a call's prompt passes this many tokens the conversation is restarted from the account.
# A low-detail frame costs about 240 tokens on this model (measured 2026-09-22), so at one frame a
# second the history grows ~14k tokens a minute, and every call re-reads all of it at the cached
# price (about a tenth). The frames themselves are paid once whatever the budget; what the budget
# sets is the re-read: over ten minutes at ~200 calls, a 40k budget re-reads ~200k tokens, 15k
# re-reads ~100k, and each restart costs one uncached read of the rendered account, a few
# thousand tokens. So the budget is small — about a minute of watching per generation.
NARRATOR_CONTEXT_TOKENS = 15000
# What a tool result says when the account has no entry newer than the model's previous result.
NO_NEWS = "(no new entries since your previous result)"

NARRATOR_SYSTEM = """You are watching a robot's workspace through a fixed camera above the table. You are
one half of the robot's mind: the other half plans and acts, but it only looks at the world when
its arm stops. You look the whole time. The account is the only record it has of what happened
while it was not looking.

This is one continuous conversation. Its first message carries the very first frame of the session,
the anchor; each message after that brings the frames since your last reply, about a second apart.
Your earlier replies are the account so far. When the conversation is restarted to keep it short,
its first message carries the anchor again and the account so far as a dated list. You do not
rewrite the account. Reply only with what the new frames add to it:

- When the account is empty, your first entry describes the anchor frame itself: what is on the
  table and where, in enough detail that later entries can name each thing by where it was first
  seen. Only then do the rules below apply.
- If nothing changed in these frames, reply exactly: NO CHANGE
- Otherwise write what moved and where it ended up, concretely: which thing, from where, to where,
  in the terms you can see (position on the table, relation to other things, what it is next to).
  Say what a hand is handling and what it left where. Things that look alike can only be told apart
  by their history, so name them by where they were first seen and keep those names. Seconds within
  the window help; the harness stamps the entry with the window's time span.
- If these frames show that an earlier entry was wrong, start a line with CORRECTION, say which
  entry, and say what is right instead. Do not otherwise restate old entries.

Some messages also carry frames from a camera on the robot's wrist, marked as such, for the seconds
that arm was acting. That view moves with the arm: things shift in it because the arm moved, not
because they moved. Do not narrate it as a second scene. Use it for what the fixed camera cannot
see — what is between the fingers, whether they closed on the thing or beside it, whether it turned,
slipped or came free while held — and write that into the account in the fixed camera's terms.

When two similar things cross or one is hidden and you cannot be certain which is which afterwards,
say so plainly and say what you saw. The other half will decide whether to look closer. A confident
wrong account is worse than an honest uncertain one.

Reply with the entry only — no preamble, no headings."""

_QUIET = re.compile(r"^\W*(no change|no changes|nothing changed|nothing has changed)\W*$", re.I)
_CORRECTION = re.compile(r"^\W*correction\b[\s:\-–—]*", re.I)


def _span(t_from: float, t_to: float) -> str:
    a, b = int(round(t_from)), int(round(t_to))
    return f"{a} s" if a == b else f"{a}–{b} s"


def _render(e: Dict[str, Any]) -> str:
    if e["kind"] == "quiet":
        return f"[{_span(e['t_from'], e['t_to'])}] no change"
    if e["kind"] == "correction":
        return f"[correction, written at {int(round(e['t_to']))} s] {e['text']}"
    return f"[{_span(e['t_from'], e['t_to'])}] {e['text']}"


def parse_reply(text: str) -> List[Tuple[str, str]]:
    """The model's reply → [(kind, text)], kind in event | quiet | correction.

    Kept separate from the threads so the protocol can be tested without a camera or a model.
    """
    text = (text or "").strip()
    if not text or _QUIET.match(text):
        return [("quiet", "")]
    out: List[Tuple[str, str]] = []
    body: List[str] = []
    for line in text.splitlines():
        if _CORRECTION.match(line):
            out.append(("correction", _CORRECTION.sub("", line, count=1).strip()))
        else:
            body.append(line)
    joined = " ".join(l.strip() for l in body if l.strip())
    if joined and not _QUIET.match(joined):
        out.insert(0, ("event", joined))
    return out or [("quiet", "")]


class Narrator:
    def __init__(self, perception, make_client: Callable[[], VLMClient], hz: float = NARRATOR_HZ,
                 camera: str = "head", max_edge: int = 768, wrists: Optional[Dict[str, str]] = None):
        self.perception = perception
        self.make_client = make_client
        self.hz = float(hz)
        self.camera = camera
        self.max_edge = max_edge
        self.wrists: Dict[str, str] = dict(wrists or {})         # arm name -> that arm's wrist camera
        self._frames: List[Tuple[float, np.ndarray]] = []       # (t since start, BGR)
        # per wrist camera: (t, BGR, acting) — acting says whether its arm was moving something,
        # opening or closing the gripper, or going home when the frame was taken
        self._wrist_frames: Dict[str, List[Tuple[float, np.ndarray, bool]]] = {c: [] for c in self.wrists.values()}
        self._acting: Dict[str, int] = {a: 0 for a in self.wrists}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._client: Optional[VLMClient] = None
        self._session: Optional[Session] = None                 # the current generation's conversation
        self.generations: int = 0
        self._log: Optional[Path] = None
        self._frame_dir: Optional[Path] = None
        self.t0: float = 0.0
        self.entries: List[Dict[str, Any]] = []                 # kind, t_from, t_to, text
        self._cursor: Tuple[int, float] = (0, -1.0)             # what the acting model has been given
        self.calls: int = 0
        self.frames_seen: int = 0
        self.last_call_s: Optional[float] = None
        self.error: Optional[str] = None
        self._paused_at: Optional[float] = None                 # t since start, while paused
        self._pause_why: str = ""
        self._gap_note: Optional[Tuple[float, float, str]] = None   # told to the model with the next window

    # ---------------- pause ----------------

    def pause(self, why: str = "the acting model's API was not answering"):
        """Stop taking, saving and sending frames until resume().

        Used while the acting model's request is timing out: the arm cannot move, so every frame would
        be the same picture, written to disk and sent to the model once a second for nothing. The pause
        is recorded as a gap in the account, never as time watched with nothing happening.
        """
        with self._lock:
            if self._paused_at is not None or not self.t0:
                return
            self._paused_at = time.time() - self.t0
            self._pause_why = why
        print(f"[narrator] paused at {self._paused_at:.0f} s ({why}): no frames taken, saved or sent")

    def resume(self):
        with self._lock:
            if self._paused_at is None:
                return
            t_from, t_to, why = self._paused_at, time.time() - self.t0, self._pause_why
            self._paused_at = None
            self.entries.append({"kind": "gap", "t_from": t_from, "t_to": t_to,
                                 "text": f"not watched: {why}"})
            self._gap_note = (t_from, t_to, why)
        print(f"[narrator] resumed at {t_to:.0f} s after a {t_to - t_from:.0f} s gap")

    # ---------------- lifecycle ----------------

    def start(self, ep_dir: Path):
        # Idempotent: a second episode in the same process, or one after an episode that died mid-way,
        # must not leave two watchers running. Agent.run has no finally around the loop, so this is
        # where a stale instance is cleaned up.
        if any(t.is_alive() for t in self._threads):
            self.stop()
        with self._lock:
            self._frames = []
            self._wrist_frames = {c: [] for c in self.wrists.values()}
            self.entries = []
            self._cursor = (0, -1.0)
        self.calls = 0; self.frames_seen = 0; self.last_call_s = None; self.error = None
        self._paused_at, self._pause_why, self._gap_note = None, "", None
        self._session = None; self.generations = 0
        self._log = ep_dir / "narrator.jsonl"
        # One folder per camera.
        self._frame_dir = ep_dir / "images"
        for cam in [self.camera, *self.wrists.values()]:
            (self._frame_dir / cam).mkdir(parents=True, exist_ok=True)
        self._client = self.make_client()
        self.t0 = time.time()
        self._stop.clear()
        self._threads = [threading.Thread(target=self._capture_loop, daemon=True, name="narrator-capture"),
                         threading.Thread(target=self._narrate_loop, daemon=True, name="narrator-model")]
        for t in self._threads:
            t.start()
        print(f"[narrator] watching the {self.camera} camera at {self.hz:g} Hz"
              + (f"; {', '.join(f'{c} while the {a} arm acts' for a, c in self.wrists.items())}" if self.wrists else ""))

    @contextmanager
    def acting(self, arm: str):
        """Marks the seconds an arm is acting; wrist frames taken meanwhile are the ones the narrator sees."""
        with self._lock:
            self._acting[arm] = self._acting.get(arm, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                self._acting[arm] = max(0, self._acting.get(arm, 0) - 1)

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5.0)
        if self._log:
            try:
                (self._log.parent / "narrator_account.txt").write_text(self.narrative, encoding="utf-8")
            except OSError:
                pass
        print(f"[narrator] stopped after {self.calls} calls over {self.frames_seen} frames, "
              f"{len(self.entries)} entries, {self.generations} conversation(s)")

    # ---------------- what the acting model reads ----------------

    @property
    def narrative(self) -> str:
        """The whole account, rendered. look() carries this."""
        with self._lock:
            return "\n".join(_render(e) for e in self.entries)

    def news(self) -> str:
        """Entries added since the previous call to news(), rendered; empty if none.

        Every tool result carries this, so the acting model's context accumulates the account once,
        as it happens, instead of a full copy per result. A quiet entry that has grown since it was
        last delivered is delivered again with its new span.
        """
        with self._lock:
            idx, seen_to = self._cursor
            ents = self.entries
            out: List[str] = []
            if 0 < idx <= len(ents) and ents[idx - 1]["kind"] == "quiet" and ents[idx - 1]["t_to"] > seen_to:
                out.append(_render(ents[idx - 1]))
            out += [_render(e) for e in ents[idx:]]
            self._cursor = (len(ents), ents[-1]["t_to"] if ents else seen_to)
        return "\n".join(out)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            n = len(self._frames)
            covered = self.entries[-1]["t_to"] if self.entries else None
        out: Dict[str, Any] = {"frames_recorded": n, "narrator_calls": self.calls,
                               "recording_s": round(time.time() - self.t0, 1) if self.t0 else 0.0}
        if covered is not None:
            out["account_covers_upto_s"] = round(covered, 1)   # frames after this are still being written
        if self.last_call_s is not None:
            out["last_call_s"] = self.last_call_s
        out["conversation"] = self.generations
        if self.error:
            out["narrator_error"] = self.error
        return out

    def status_line(self) -> str:
        """status() in one line, for tool results: the model reads it every step, so it costs every step."""
        st = self.status()
        line = (f"account covers up to {st.get('account_covers_upto_s', 0)} s of {st['recording_s']} s recorded "
                f"({st['narrator_calls']} calls, conversation {st.get('conversation', 1)})")
        if st.get("narrator_error"):
            line += f"; ERROR: {st['narrator_error']}"
        return line

    def frame_at(self, t: float) -> Optional[Tuple[float, np.ndarray]]:
        """The recorded frame nearest to t seconds since the start, for replay."""
        with self._lock:
            if not self._frames:
                return None
            return min(self._frames, key=lambda f: abs(f[0] - t))

    # ---------------- the account ----------------

    def _append(self, parsed: List[Tuple[str, str]], t_from: float, t_to: float):
        with self._lock:
            for kind, text in parsed:
                last = self.entries[-1] if self.entries else None
                if kind == "quiet" and last is not None and last["kind"] == "quiet":
                    last["t_to"] = t_to             # heartbeat: extend, do not add a line
                    continue
                self.entries.append({"kind": kind, "t_from": t_from, "t_to": t_to, "text": text})

    # ---------------- threads ----------------

    def _capture_loop(self):
        period = 1.0 / self.hz
        nxt = time.time()
        while not self._stop.is_set():
            if self._paused_at is not None:
                nxt = time.time() + period
                self._stop.wait(period)
                continue
            v = self.perception.rig.grab(self.camera, 0)
            if v is not None:
                t = time.time() - self.t0
                with self._lock:
                    self._frames.append((t, v.rgb))
                    k = len(self._frames)
                try:
                    cv2.imwrite(str(self._frame_dir / self.camera / f"{k:05d}_{t:07.1f}s.jpg"), v.rgb,
                                [cv2.IMWRITE_JPEG_QUALITY, 80])
                except Exception:  # noqa: BLE001  a failed save must not stop the watching
                    pass
            for arm, cam in self.wrists.items():
                w = self.perception.rig.grab(cam, 0)
                if w is None:
                    continue
                t = time.time() - self.t0
                with self._lock:
                    acting = self._acting.get(arm, 0) > 0
                    self._wrist_frames[cam].append((t, w.rgb, acting))
                    k = len(self._wrist_frames[cam])
                try:
                    cv2.imwrite(str(self._frame_dir / cam / f"{k:05d}_{t:07.1f}s{'_acting' if acting else ''}.jpg"),
                                w.rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                except Exception:  # noqa: BLE001
                    pass
            nxt += period
            self._stop.wait(max(0.0, nxt - time.time()))

    def _narrate_loop(self):
        sent = 0
        sent_w: Dict[str, int] = {c: 0 for c in self.wrists.values()}
        while not self._stop.is_set():
            with self._lock:
                new = self._frames[sent:]
                first = self._frames[0] if self._frames else None
                wrist_new = {c: self._wrist_frames[c][sent_w[c]:] for c in sent_w}
            if not new or first is None:
                self._stop.wait(0.2)
                continue
            sent += len(new)
            # Wrist frames of this window, only those taken while the arm was acting. The rest stay on
            # disk. Each arm's group is sampled on its own so one busy arm cannot crowd out the head.
            wrist_parts: List[Tuple[str, Any]] = []
            wrist_sent = 0
            for arm, cam in self.wrists.items():
                frames = wrist_new.get(cam, [])
                sent_w[cam] += len(frames)
                acting = [(t, f) for t, f, a in frames if a]
                if not acting:
                    continue
                n_act = len(acting)
                if n_act > NARRATOR_MAX_WRIST_FRAMES:
                    idx = np.linspace(0, n_act - 1, NARRATOR_MAX_WRIST_FRAMES).round().astype(int)
                    acting = [acting[i] for i in idx]
                wrist_parts.append(("text", f"Then {len(acting)} images from the {arm} arm's wrist camera ({cam}), "
                                            f"taken while that arm was acting, from {acting[0][0]:.0f} s to "
                                            f"{acting[-1][0]:.0f} s" + (f" (sampled from {n_act})" if n_act != len(acting) else "")
                                            + ". This view moves with the arm."))
                wrist_parts += [("image", f) for _, f in acting]
                wrist_sent += len(acting)
            covered = len(new)
            if len(new) > NARRATOR_MAX_FRAMES:
                idx = np.linspace(0, len(new) - 1, NARRATOR_MAX_FRAMES).round().astype(int)
                new = [new[i] for i in idx]
            t_a, t_b = new[0][0], new[-1][0]
            window = (f"The next {len(new)} images are the frames from {t_a:.0f} s to {t_b:.0f} s"
                      + (f" (sampled from {covered})" if covered != len(new) else "") + ", about a second apart. ")
            gap = self._gap_note
            if gap is not None and t_b >= gap[1]:
                window = (f"No frames were taken from {gap[0]:.0f} s to {gap[1]:.0f} s ({gap[2]}). Anything that "
                          f"differs from the last frame before that happened during the gap. ") + window
                self._gap_note = None
            head: List[Tuple[str, Any]] = []
            if self._session is None:
                # A new generation: the anchor, and the account so far if there is one. Stable things
                # first, so a restarted conversation can itself be extended and cached from here on.
                self._session = Session(NARRATOR_SYSTEM, log_path=None, max_edge=self.max_edge)
                self.generations += 1
                account = self.narrative
                head = [("text", "The first image is the first frame of the session, the anchor."), ("image", first[1])]
                if account:
                    head.append(("text", "The account so far, from earlier in this session:\n" + account))
                    ask = "Reply with what they add to the account, or NO CHANGE."
                else:
                    ask = "The account is empty: begin it with what the anchor frame shows, then what these frames add."
            else:
                ask = "Reply with what they add to the account, or NO CHANGE."
            self._session.user_parts(head + [("text", window + ask)] + [("image", f) for _, f in new] + wrist_parts,
                                     detail="low")
            t_call = time.time()
            try:
                reply = self._client.step(self._session)
                self.error = None
            except Exception as exc:  # noqa: BLE001  the watcher must outlive a failed call
                self.error = f"{type(exc).__name__}: {exc}"[:200]
                print(f"[narrator] call failed: {self.error}")
                self._session.entries.pop()         # no reply to pair it with; the next turn starts clean
                self._stop.wait(2.0)
                continue
            self._session.assistant(reply)
            parsed = parse_reply(reply.text)
            self._append(parsed, t_a, t_b)
            self.last_call_s = round(time.time() - t_call, 1)
            self.calls += 1
            self.frames_seen += covered
            if self._log:
                with open(self._log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"call": self.calls, "t_from": round(t_a, 1), "t_to": round(t_b, 1),
                                        "frames": covered, "sent": len(new), "elapsed_s": self.last_call_s,
                                        "usage": reply.usage, "reply": (reply.text or "").strip(),
                                        "entries": len(self.entries), "conversation": self.generations,
                                        "wrist_sent": wrist_sent},
                                       ensure_ascii=False) + "\n")
            if ((reply.usage or {}).get("input_tokens") or 0) > NARRATOR_CONTEXT_TOKENS:
                print(f"[narrator] conversation {self.generations} reached "
                      f"{reply.usage.get('input_tokens'):,} tokens; the next window starts a new one from the account")
                self._session = None
