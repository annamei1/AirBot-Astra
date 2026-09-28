"""
The agent loop: instruction + head image → model → tool calls → observations → ... → done().

Task decomposition is a structured tool: the model must call plan(steps) before acting and
mark_step(...) as steps are verified; done() is refused while planned steps are still open. Metric
numbers come from perception tools.
Everything is logged under harness/logs/<episode>/: session.jsonl (the whole conversation), the images the
model saw, the narrator's frames and account, and outcome.json.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from harness.tools import ToolRegistry, ToolResult
from harness.gripper import gripper_paragraph
from harness.vlm_client import Session, VLMClient

WARN_STEPS_LEFT = 5      # start telling the model to wrap up this many calls before the limit


def build_system_prompt(table_z: float, bounds_min, bounds_max, hover: float,
                        cameras: Optional[List[str]] = None,
                        arms: Optional[List[str]] = None,
                        narrator: bool = False, keep_images: int = 1,
                        gripper: Optional[Dict[str, Any]] = None, rest: str = "zero") -> str:
    cams = cameras or ["head", "wrist"]
    # The zero joint pose leaves the gripper pointing forward, and a tool angle left out carries over from
    # the current pose. On 2026-09-26 four of six episodes made their first move out of the zero pose
    # without a pitch, so the gripper stayed horizontal; three were refused by the planner, the fourth
    # was accepted and drove the side of the gripper into the table (34 A, arm server lost). The prompt
    # had said the gripper points straight down by default, which is true everywhere except where every
    # episode starts.
    starts = (" Every episode starts with the arm there, and the arm returns there between episodes."
              if rest == "zero" else "")
    arms = arms or ["right"]
    # Measured facts about the mounted fingers (harness/gripper.py); nothing about how to use them.
    grip = (gripper_paragraph(gripper, arms) + "\n\n") if gripper else ""
    # The prompt is an f-string, so this lands in the numbered list as {imgs}.
    imgs = ("""   Images do not stay. Each result's images are in front of you for the decisions that follow; when the
   context grows long they are withdrawn, and what remains is what you did with them — the ids find()
   recorded, the pixels you measured, the results themselves. Never rely on an earlier image: look() again.
""" if keep_images > 0 else "")
    narr = ("""THE OTHER HALF OF YOU IS WATCHING. A second copy of this model watches the head camera continuously —
one frame a second, the whole session — and keeps a written account of everything that happens on the
table, as dated entries. Every tool result carries `what_happened`: the entries added since your previous
result, so the account reaches you once, as it happens; look() carries the whole account. It is the only
record of what moved while you were not looking: you observe when your arm stops, it observes all the
time. Read it before deciding. When it says it is unsure which of two alike things is which, that is your
cue to look() for yourself. If the task involves waiting for something to happen — a person moving
things, an object being handed over — use wait(seconds): the narrator keeps watching, and the result's
what_happened covers the wait.

""" if narrator else "")
    # One arm or two is the first thing that has to be true, because it decides whether "the gripper"
    # names something unambiguous. The per-tool `arm` parameter says how; this says that.
    body = (f"""You control TWO 6-DoF AirBot Play arms, {arms}, each with a parallel two-finger gripper, on one
table. Every motion tool takes arm=; leave it out and '{arms[0]}' moves. Both arms' coordinates are in ONE
frame — the '{arms[0]}' arm's base — so a point from find() or measure() is reachable by either without
conversion. Accuracy is not symmetric though: a point one arm's wrist camera measured and that SAME arm
then acts on is good to about 1.6 mm, while handing it to the other arm costs about 8 mm, which is the
two arms' own kinematics and nothing can calibrate it away. To place roughly, either arm; to pinch one
layer of cloth, look with the arm that is about to pinch.
The two arms can collide with each other — nothing stops them. state.arms gives the other arm's gripper
position and its distance from this one after every move."""
            if len(arms) > 1 else
            """You control a single 6-DoF AirBot Play robot arm with a parallel two-finger gripper on a table.""")
    return f"""{body}
Coordinates are metres in the robot base frame: x forward, y left, z up.
**At the arm's zero joint pose the gripper points straight forward, horizontally: pitch 90°.**{starts}
Every motion tool keeps the tool angles you leave out, so a first move out of the zero pose without
pitch_deg stays horizontal; give pitch_deg=0 (straight down) when you want to come from above.
Straight down is the pose for most work, and most of the time nothing else is needed. The tool can be
tilted with tilt(pitch_deg, roll_deg): pitch leans the approach axis away from vertical, in the horizontal
direction given by yaw, and roll spins the tool about that axis.
**The two fingers close along an axis perpendicular to yaw, and at roll = 0 that axis stays horizontal at
every pitch.** So something hanging vertically is pinched by coming in at roll 0, with the fingers meeting
it from either side. Rolling 90° turns the finger axis vertical, so the fingers would close on a hanging
object from above and below, which pushes it out of the way instead of gripping it. **Leave roll at 0
unless you have a specific reason;** yaw and pitch decide the approach, roll almost never helps. A tilt is how the fingers reach the side of
something instead of the top of it, and the only way to get a fingertip under the edge of something too flat
to grasp from above. Measured on this arm at a point in front of the base: a tilt up to about 60° works at
yaw 0 and ±45° and ±90°, a fully horizontal 90° approach only at yaw ±45°, and tilts pointing back over the
base are the first to be refused. Treat that as rough: the same tilt is sometimes accepted and sometimes
refused depending on the joint configuration the arm is in, because the planner picks a different solution
each time. A refusal means not from here, not impossible, and the arm has not moved when you get one.
Table surface z ≈ {table_z:.3f}. Reachable workspace:
x ∈ [{bounds_min[0]}, {bounds_max[0]}], y ∈ [{bounds_min[1]}, {bounds_max[1]}], z ∈ [{bounds_min[2]}, {bounds_max[2]}],
and no further than 0.5 m from the base (far points must be low).

Cameras: {cams}. Both are RGB-D and both measure into the same base frame, so a position from either is
directly comparable. They see different things and you choose which to use:
  head  — fixed above the table, sees everything at once but from ~0.4 m, so small detail is coarse.
  wrist — on the moving arm, sees only what the arm is currently over, but from centimetres away, so it
          resolves far finer detail. It is the camera that tells you whether a grasp actually closed on
          the object, and it is useless for searching.
They cooperate through the base frame, not through each other: anything either camera measures lands in the
same coordinates, so a position taken from one is directly usable by the other and survives the arm moving.
Changing what a camera can see is just moving the arm, which you already do: carry the wrist camera to a
tilted standoff beside the object, or step out of the head camera's line of sight. find(camera=..., box=...)
and measure(u, v, camera=...) are what read the scene once you can see it.

{grip}You act ONLY through tools, in this order:
{narr}1. look() to see the scene from all cameras.
2. plan(steps): write the ordered physical steps for the task before moving anything. Think about what has to be
   true before each step (gripper open before descending onto an object, object grasped before lifting, clearance
   above a container rim before moving over it, release inside the container, arm back home at the end).
3. find(label) for every object the plan mentions → object ids with measured 3D poses. Never guess coordinates.
4. Act ONE SHORT SEGMENT AT A TIME. Every motion tool returns fresh images from every camera the moment the
   arm stops, so the normal loop is: read the images, decide the next single move, execute it, read the new
   images. The primitives are that loop: move_above, move_relative, tilt, descend, move_until_contact,
   close_gripper, open_gripper, lift, follow_path. There are no macros: a grasp is your own sequence of
   these, and where on the object you close is your decision — the centre of a thing is not always where
   fingers can hold it.
   When you need a better look before touching something, move there yourself with move_above, move_relative
   or tilt and then measure: can_see in every result tells you whether the move worked.
5. After each action read the returned state and EVERY image: the head image says where things are, the wrist
   image says what is actually between the fingers. mark_step(i, "done", evidence) only for steps you verified
   from a NEW observation; mark "failed" and replan if not.
{imgs}6. done(success, summary) when every planned step is verified (or, after at most 3 attempts on the same step,
   done(success=false) with what went wrong).

Rules:
- Labels for find() are 1–3 concrete English words describing appearance ("black block", "green bowl").
  If the text finds nothing or the wrong thing, try different wording ONCE, then call find() again with
  point=[u,v] on the object (or box=[x0,y0,x1,y1] around it) read off the image: that segments by geometry,
  needs no name, and works on things the text encoder does not know. It still returns a normal id.
  measure(u, v, camera) turns any pixel into a 3D point, in whichever camera shows it best. Use it to choose
  WHERE ON an object to act, not just to probe elsewhere: find() reports a centroid, which is where the
  object's mass is, and that is not always where you can close the fingers. On anything that is not a
  compact blob — something long, something hanging, an edge, a handle, a rim — point at the part you
  actually mean to pinch and grasp the point measure() gives you back.
  measure(points=[[u, v], ...]) does the same for a whole ordered list at once, and that is how you capture
  a shape you can see now and will not be able to see once the arm is working on it: trace it in the image,
  get base-frame coordinates back, act on the coordinates, step back to check. Coordinates outlive the view.
- Two primitives are continuous rather than point-to-point, and they are the only ones that can feel anything.
  move_until_contact travels until the arm meets resistance, reading the current about 25 times a second, so it
  stops ON contact rather than after it: that is how you find a surface the depth camera will not give you, how
  you learn how far below the reported gripper position the fingertips actually reach, and how you touch
  something soft without crushing it. follow_path travels through several points without stopping at them,
  which is what dragging, wiping or smoothing needs, because the contact must not break between waypoints.
  Contact is a DEPARTURE from what the same motion draws unobstructed, not an absolute number, and it goes
  both ways. Pushing into something costs the motors more, so the current RISES. Coming to rest on something
  that holds the arm up costs them less, because the surface carries weight the joints were carrying, so the
  current FALLS. A descent onto a table shows the fall, a sideways press shows the rise, and both mean the
  arm has met the world. It is not an absolute number: a
  descent draws less current than holding still because gravity helps, and free-air motion draws more, so no
  fixed number separates them. The arm ceasing to make progress while it is still being pushed counts as
  contact too, and for something light that barely loads the motors it is the better of the two signals.
  Both report what they saw. One that says it arrived without contact means neither signal fired along the
  whole path, which is evidence but not proof that nothing is there: read baseline_a, peak_current_a and
  fall_a, which is reported whether or not it fired.
  Those numbers are joint effort, which the arm's own documentation gives in Nm; the parameter names say
  "current" for historical reasons only. Compare them against each other, never against an absolute idea
  of what a newton-metre should be.
- close_gripper has two independent signals and they cost nothing: where the fingers came to rest, and how
  hard the gripper motor pushed. A thin or soft thing may be invisible to the first and obvious to the
  second. The wrist image is the third and it settles disagreements.
- **You are your own occluder, and you block the view most at the moment you are about to act.** Every
  motion result carries `can_see`: for each object you know about, how much of it each camera can see FROM
  WHERE THE ARM NOW IS, how far away, and how many millimetres one pixel covers there. Read it before
  committing to a grasp. A camera that is closer is not automatically better: the wrist camera routinely
  resolves four times finer and sees almost none of the object, because the fingers are in front of it.
  When nothing can see the target, the answer is to move somewhere that can and look again, not to reach
  anyway. But visibility is one consideration and not the first: **a pose you can see from is not
  automatically a pose you can act from**. The approach direction is decided by what is in the way of the
  fingers, and only then adjusted for what the cameras can see. If a vertical approach would put the
  gripper's body through something, tilt and come in from the side, whatever can_see says about the view
  from straight above. Coordinates in the base frame survive occlusion, so something measured while you could see it
  stays usable after you cannot: measure first, then act blind, then step back and check.
- **Picking something up changes where the end of the arm is.** The fingertips already reach a few
  millimetres past the position the gripper reports; a held tool reaches further still, by however much of
  it sticks out, and nothing can know that number until the thing touches something. So the first time you
  are going to press, wipe or place with a tool in the fingers, spend one move_until_contact downwards onto
  a surface whose height you already know. The gripper height at contact is then the height at which that
  tool meets that surface, and you can work from it for every stroke afterwards.
- **When you already know the next few moves, send them as one do([...]) instead of one call each.** The
  round trip between calls is most of the wall clock on a short run of motions whose shape you already
  know — approach, touch down, stroke. Give each step a `require` naming a field of that tool's own result
  and the value it must have for the next step to make sense; the run stops at the first one that does not
  hold and hands you everything up to it. Keep steps in separate calls wherever you genuinely need to look
  at an image before choosing the next one — only the last step in a chunk returns camera views.
  A step that only RECORDS can lead a chunk: mark_step about what you have already seen, then the moves
  that follow from it, in the same call. And a path step can carry null coordinates, so
  `move_until_contact` then `follow_path([[x, y, null], ...])` strokes at the height the touch just
  found — the whole approach-touch-stroke sequence fits in one chunk without knowing that height first.
- Positions age. Anything you measured before the arm moved something may be stale: find() it again.
- The gripper holds one object at a time. A result with "ok": false explains why — read it before retrying.
- A bowl's detected z is about its rim. To put something in, come down over it with move_until_contact and open; the contact stop is what says you are low enough.
- Hover height above objects is {hover:.2f} m. The wrist camera only sees what the arm is over, and it sees
  past the fingers only when the tool is tilted, so a close look is a move first and a measurement second.
- Read measurements for what they claim. Heights are relative to whatever the thing rests on, and carry a
  height_status: "measured" is a real height; "below_noise" means the depth camera cannot separate it from
  the surface under it — normal for dark, thin, flat or soft things, and NOT evidence that it is flat,
  missing or already handled. depth_valid_frac and z_spread_m say how solid a reading is. When a height you
  need is below_noise, get the evidence another way: look at the image, compare before/after images, use
  the gripper's own feedback, measure(u, v) somewhere the geometry does show up (an edge, a fold, a rim),
  or descend to a safe height and move_until_contact downwards to feel where the surface actually is.
- Keep your text short: one sentence on what you see and what you do next, then the tool call."""


class Agent:
    def __init__(self, client: VLMClient, registry: ToolRegistry, perception, system_prompt: str,
                 log_root: Path, max_steps: int = 40, max_edge: int = 768, abort=None, viz=None,
                 narrator=None, keep_logs: int = 1, keep_images: int = 1, compact_at: int = 30000,
                 gripper: Optional[Dict[str, Any]] = None, run_meta: Optional[Dict[str, Any]] = None):
        self.run_meta = dict(run_meta or {})   # rig settings every outcome carries, e.g. the arm speed profile
        self.narrator = narrator                # harness.narrator.Narrator or None
        if narrator is not None:
            # While this model's request is timing out, the narrator stops taking and saving frames.
            client.on_stall = lambda: narrator.pause("the acting model's API was not answering")
            client.on_recover = narrator.resume
        self.gripper = gripper                  # its measured facts (harness/gripper.py), recorded alongside
        self.keep_logs = keep_logs              # episode directories kept under log_root, this one included
        # When a call's prompt has grown by compact_at tokens since the last compaction (or the start),
        # the images of all but the newest keep_images image-bearing turns are withdrawn, once. The
        # trigger is growth, not the prompt's absolute size: a prompt whose TEXT alone exceeds a fixed
        # budget would otherwise compact on every call, withdrawing one or two images each time — the
        # per-call history edit that the cache punishes hardest. That happened on 2026-09-22: after the
        # first compaction of a 65-call episode every call re-read 45-54k tokens at full price, 90% of
        # the episode's uncached tokens. See Session.compact.
        self.keep_images = keep_images
        self.compact_at = compact_at
        self.client = client
        self.registry = registry
        self.perception = perception
        self.system_prompt = system_prompt
        self.log_root = Path(log_root)
        self.max_steps = max_steps
        self.max_edge = max_edge
        self.abort = abort                      # threading.Event or None
        self.viz = viz                          # harness.viz.LiveView or None
        self._outcome: Optional[Dict[str, Any]] = None
        self.plan: List[Dict[str, Any]] = []
        registry.register(
            "plan", "Record the ordered physical steps for the task before acting (task decomposition). "
                    "Call it once at the start; call again to replace the plan after a failure.",
            {"properties": {"steps": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
             "required": ["steps"]}, self._plan)
        registry.register(
            "mark_step", "Update one planned step after verifying it from a new observation.",
            {"properties": {"index": {"type": "integer", "description": "0-based index into the plan"},
                            "status": {"type": "string", "enum": ["done", "failed", "skipped"]},
                            "evidence": {"type": "string", "description": "what you saw / measured"}},
             "required": ["index", "status", "evidence"]}, self._mark_step)
        registry.register(
            "done", "Finish the episode. Refused while planned steps are still pending.",
            {"properties": {"success": {"type": "boolean"}, "summary": {"type": "string"}},
             "required": ["success", "summary"]}, self._done)

    # ---------------- ledger tools ----------------

    def _plan(self, steps: List[str]) -> ToolResult:
        self.plan = [{"index": i, "step": s, "status": "pending", "evidence": ""} for i, s in enumerate(steps)]
        return ToolResult.json({"ok": True, "plan": self.plan})

    def _mark_step(self, index: int, status: str, evidence: str) -> ToolResult:
        if not (0 <= index < len(self.plan)):
            return ToolResult.error(f"index {index} out of range, plan has {len(self.plan)} steps")
        self.plan[index].update(status=status, evidence=evidence)
        pending = [p["index"] for p in self.plan if p["status"] == "pending"]
        return ToolResult.json({"ok": True, "plan": self.plan, "pending": pending})

    def _done(self, success: bool, summary: str) -> ToolResult:
        pending = [p for p in self.plan if p["status"] == "pending"]
        if success and pending:
            return ToolResult.error("plan still has pending steps: "
                                    + "; ".join(f"[{p['index']}] {p['step']}" for p in pending)
                                    + ". Verify them with look()/find() and mark_step(), or done(success=false).")
        self._outcome = {"success": bool(success), "summary": summary, "plan": self.plan}
        return ToolResult.json({"ok": True, "episode_finished": True}, done=True)

    # ---------------- loop ----------------

    _EPISODE_NAME = re.compile(r"^\d{8}-\d{6}$")

    def _prune_logs(self, current: Path):
        """Delete older episode directories so that keep_logs of them remain, the current one included.

        An episode with the narrator on writes a frame a second per camera — 140 MB an hour from the
        head camera alone — and during testing nobody goes back to last week's run. keep_logs <= 0
        keeps everything, which is what an evaluation session wants.
        """
        if self.keep_logs <= 0:
            return
        episodes = sorted(d for d in self.log_root.iterdir()
                          if d.is_dir() and self._EPISODE_NAME.match(d.name) and d != current)
        for d in episodes[:max(0, len(episodes) - (self.keep_logs - 1))]:
            try:
                shutil.rmtree(d)
            except OSError as exc:          # a root-owned leftover, a file open in another window
                print(f"[harness] could not remove old log {d.name}: {exc}")

    def run(self, instruction: str) -> Dict[str, Any]:
        ep = time.strftime("%Y%m%d-%H%M%S")
        ep_dir = self.log_root / ep
        ep_dir.mkdir(parents=True, exist_ok=True)
        self._prune_logs(ep_dir)
        # A placeholder, replaced at the end. If the process never gets there (a crash, a lost arm
        # server, a forced quit) the episode still says what it was and that its record is incomplete.
        with open(ep_dir / "outcome.json", "w") as f:
            json.dump({"success": False, "record_complete": False,
                       "summary": "the episode did not end normally: the process stopped before done() "
                                  "or the step limit",
                       "episode": ep, "instruction": instruction, "log": str(ep_dir),
                       **self.run_meta}, f, indent=2, ensure_ascii=False)
        if self.narrator is not None:
            self.narrator.start(ep_dir)         # before the first look(): its first frame is the anchor
        session = Session(self.system_prompt, ep_dir / "session.jsonl", max_edge=self.max_edge)
        self._outcome, self.plan = None, []
        n_calls, n_tool_calls, t_start = 0, 0, time.time()
        compact_base: Optional[int] = None      # prompt size right after the last compaction

        first = self.perception.look()
        for k, (_, img) in enumerate(first.images):
            self._save_image(ep_dir, 0, f"look_{k}", img)
        session.user(f"Task: {instruction}\n\nInitial observation: {first.text}\n"
                     + "\n".join(c for c, _ in first.images),
                     images=[im for _, im in first.images])

        for step in range(1, self.max_steps + 1):
            if self.abort is not None and self.abort.is_set():
                self._outcome = {"success": False, "summary": "aborted by user"}
                break
            if self.viz:
                self.viz.set_busy(f"GPT-6 thinking (step {step})")
            reply = self.client.step(session, tools=self.registry.schemas())
            n_calls += 1
            session.assistant(reply)
            toks = (reply.usage or {}).get("input_tokens") or 0
            if toks and compact_base is None:
                compact_base = toks
            if self.compact_at and toks and toks - compact_base > self.compact_at:
                n = session.compact(self.keep_images)
                compact_base = None                 # re-based on the next call's (smaller) prompt
                if n:
                    print(f"[harness] context grew to {toks:,} tokens: withdrew {n} earlier images "
                          f"(the next call re-reads the text once, then the cache holds again)")
            if self.viz:
                self.viz.set_stats(model_calls=n_calls, tool_calls=n_tool_calls)
                if reply.text:
                    self.viz.set_model_text(reply.text)
            if reply.text:
                print(f"\n[agent #{step}] {reply.text}")
            if not reply.tool_calls:
                session.user("Please continue with a tool call. If the task is finished and verified, call done().")
                continue

            pending: List[Tuple[str, np.ndarray]] = []
            finished = False
            for call in reply.tool_calls:
                print(f"[tool #{step}] {call.name}({json.dumps(call.arguments, ensure_ascii=False)})")
                if self.viz:
                    self.viz.set_busy(f"{call.name}({json.dumps(call.arguments, ensure_ascii=False)})")
                result = self.registry.dispatch(call)
                n_tool_calls += 1
                print(f"[tool #{step}] → {result.text[:300]}")
                if self.viz:
                    good = '"ok": false' not in result.text
                    self.viz.note(f"{call.name}: {result.text[:110]}", ok=good)
                    self.viz.set_plan(self.plan)
                    self.viz.set_stats(model_calls=n_calls, tool_calls=n_tool_calls)
                session.tool_result(call.id, result.text)
                for k, (caption, img) in enumerate(result.images):
                    self._save_image(ep_dir, step, f"{call.name}_{k}" if k else call.name, img)
                    pending.append((caption or f"Observation after {call.name}.", img))
                if result.done:
                    finished = True
            left = self.max_steps - step
            warning = ""
            if 0 < left <= WARN_STEPS_LEFT:
                # The 02:34 and 04:15 runs both finished the physical task and were scored failures
                # because the step limit arrived before done() did. The model cannot see the limit
                # unless it is told, so tell it while there is still room to act on it.
                warning = (f"\n\n[harness] {left} tool call(s) left in this episode. Finish what is in the "
                           f"gripper, verify what you can, and call done() before they run out — an "
                           f"episode that hits the limit is recorded as a failure whatever the table "
                           f"looks like.")
            if pending:
                session.user("\n".join(c for c, _ in pending) + warning,
                             images=[im for _, im in pending])
            elif warning:
                session.user(warning.strip())
            if finished:
                break

        if self.narrator is not None:
            self.narrator.stop()
        outcome = self._outcome or {"success": False, "summary": f"step limit {self.max_steps} reached",
                                    "plan": self.plan}
        outcome.update({"record_complete": True, "episode": ep, "instruction": instruction, "model_calls": n_calls,
                        "tool_calls": n_tool_calls, "elapsed_s": round(time.time() - t_start, 1),
                        "images_sent": session.n_images, "log": str(ep_dir),
                        **self.run_meta,
                        "gripper": (self.gripper.get("name") if self.gripper else None)})
        with open(ep_dir / "outcome.json", "w") as f:
            json.dump(outcome, f, indent=2, ensure_ascii=False)
        print(f"\n[episode] {json.dumps({k: v for k, v in outcome.items() if k != 'plan'}, ensure_ascii=False)}")
        return outcome

    @staticmethod
    def _save_image(ep_dir: Path, step: int, name: str, img: Optional[np.ndarray]):
        if img is not None:
            d = ep_dir / "images" / "model_view"
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{step:03d}_{name}.jpg"), img)
