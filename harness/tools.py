"""
Tool registry: JSON-schema tools the VLM can call, dispatched to python functions.

A tool returns a ToolResult: the text the model reads (usually JSON), zero or more images the model
should see afterwards (one per camera, when a tool observes), and a `done` flag that ends the
episode. Exceptions inside a tool never crash the loop: they come back to the model as
{"ok": false, "error": ...} so it can read the reason and try something else.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from harness.vlm_client import ToolCall
from harness.narrator import NO_NEWS


@dataclass
class ToolResult:
    text: str
    images: List[Tuple[str, np.ndarray]] = field(default_factory=list)   # (caption, BGR)
    done: bool = False

    # --- convenience for the common single-image case ---
    @property
    def image(self) -> Optional[np.ndarray]:
        return self.images[0][1] if self.images else None

    @property
    def image_caption(self) -> str:
        return self.images[0][0] if self.images else ""

    @classmethod
    def json(cls, data: Any, image: Optional[np.ndarray] = None, caption: str = "",
             images: Optional[List[Tuple[str, np.ndarray]]] = None, done: bool = False):
        shots = list(images or [])
        if image is not None:
            shots.insert(0, (caption, image))
        return cls(text=json.dumps(data, ensure_ascii=False), images=shots, done=done)

    @classmethod
    def error(cls, message: str):
        return cls(text=json.dumps({"ok": False, "error": message}, ensure_ascii=False))


# ---- chunking: several already-registered tools in one model call ----
# A cap, not a policy. Long chunks make a failure expensive to diagnose and defeat the point of
# keeping the model in the loop.
MAX_CHUNK_STEPS = 8
# `do` inside `do` would recurse; `done` inside `do` would end the episode from inside a batch, where
# the model has not seen the result it is declaring success on.
NOT_CHUNKABLE = ("do", "done")


def dig(data: Any, path: str) -> Tuple[bool, Any]:
    """Follow a dotted path into nested dicts. → (found, value)."""
    cur = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def satisfies(value: Any, want: Any) -> bool:
    """Does `value` meet `want`? A literal means equality; {">": x} and friends compare.

    Deliberately generic: it reads whatever field the result already publishes, so it needs no
    vocabulary of its own and nothing here knows what any particular tool or task means.
    """
    if isinstance(want, dict) and len(want) == 1:
        op, target = next(iter(want.items()))
        if op in (">", "<", ">=", "<=", "!=", "=="):
            try:
                a, b = float(value), float(target)
            except (TypeError, ValueError):
                return (value != target) if op == "!=" else (value == target) if op == "==" else False
            return {">": a > b, "<": a < b, ">=": a >= b,
                    "<=": a <= b, "!=": a != b, "==": a == b}[op]
    return value == want


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, parameters: Dict[str, Any],
                 fn: Callable[..., ToolResult]):
        """parameters: JSON schema of the arguments object ({"type":"object","properties":...})."""
        parameters = dict(parameters)
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        parameters.setdefault("additionalProperties", False)
        self._tools[name] = {"description": description, "parameters": parameters, "fn": fn}

    def names(self) -> List[str]:
        return list(self._tools)

    def augment(self, name: str, properties: Dict[str, Any], note: str = ""):
        """Add parameters to an already-registered tool, and a line to its description.

        For capabilities the rig either has or does not have. A second arm is the case this exists
        for: writing `arm` into ten schemas up front would put it in front of a model driving a
        one-armed robot, which then has a parameter to wonder about and a choice that does not exist.
        Registering the tools once and widening them only when the arm is really there keeps the
        single-arm prompt exactly as it was.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"cannot augment unknown tool '{name}'. Available: {self.names()}")
        tool["parameters"]["properties"].update(properties)
        if note:
            tool["description"] = tool["description"].rstrip() + "\n" + note

    def schemas(self) -> List[Dict[str, Any]]:
        return [{"type": "function",
                 "function": {"name": n, "description": t["description"], "parameters": t["parameters"]}}
                for n, t in self._tools.items()]

    # ---------------- chunking ----------------

    def register_chunk(self, name: str = "do"):
        """Expose `do`: run several already-registered tools in one model call.

        This adds no ability. Every step is a tool the model could already call, with the arguments it
        would already have passed. What it removes is the round trip between them — on the whiteboard a
        grasp cost three round trips of thinking for about eight seconds of motion, and the model was
        right not to split its descents because splitting cost more than it saved.

        Nothing here decides anything. The model writes the sequence AND the condition each step has to
        meet to go on; this executes them in order and stops at the first condition that does not hold,
        handing back what happened up to that point. A condition is a field of the result the tool
        already publishes, so there is no vocabulary to learn and nothing in this file knows what any
        tool or task means.

        Images come back only from the last step that ran. Carrying two views out of every step would
        put the tokens straight back, and the reason to batch at all is that the model does not need to
        look in between.
        """
        def do(steps: List[Dict[str, Any]]) -> ToolResult:
            if not isinstance(steps, list) or not steps:
                return ToolResult.error("do(steps=[...]) needs a non-empty list of steps")
            if len(steps) > MAX_CHUNK_STEPS:
                return ToolResult.error(f"{len(steps)} steps is more than the {MAX_CHUNK_STEPS} this "
                                        f"runs at once; send the sequence in pieces")
            # Validate everything before moving anything: a chunk refused on its fourth step has
            # already moved the arm three times for nothing.
            for i, st in enumerate(steps):
                if not isinstance(st, dict) or "tool" not in st:
                    return ToolResult.error(f"step {i} needs a 'tool' name and optional 'args'/'require'")
                t = st["tool"]
                if t in NOT_CHUNKABLE:
                    return ToolResult.error(f"step {i}: '{t}' cannot go inside do()")
                if t not in self._tools:
                    return ToolResult.error(f"step {i}: unknown tool '{t}'. Available: {self.names()}")
                if not isinstance(st.get("args", {}), dict):
                    return ToolResult.error(f"step {i}: 'args' must be an object")
                if not isinstance(st.get("require", {}) or {}, dict):
                    return ToolResult.error(f"step {i}: 'require' must be an object of field -> value")

            ran: List[Dict[str, Any]] = []
            images: List[Tuple[str, np.ndarray]] = []
            stopped: Optional[str] = None
            for i, st in enumerate(steps):
                t = st["tool"]
                res = self.invoke(t, dict(st.get("args", {})))
                try:
                    payload = json.loads(res.text)
                except (ValueError, TypeError):
                    payload = {"text": res.text}
                images = list(res.images)          # only the last step's views survive
                entry: Dict[str, Any] = {"i": i, "tool": t, "result": payload}
                ran.append(entry)

                if isinstance(payload, dict) and payload.get("ok") is False:
                    entry["stopped_here"] = "the tool reported ok: false"
                    stopped = f"step {i} ({t}) failed"
                    break
                unmet = []
                for path, want in (st.get("require") or {}).items():
                    found, got = dig(payload, path)
                    if not found:
                        unmet.append(f"{path} is not in what {t} returned")
                    elif not satisfies(got, want):
                        unmet.append(f"{path} came back {got!r}, you required {want!r}")
                if unmet:
                    entry["stopped_here"] = "; ".join(unmet)
                    stopped = f"step {i} ({t}) did not meet what you required"
                    break

            # Each observing step carries the narrator's entries since the step before it, so inside
            # a chunk they are disjoint slices of the account: join them at the top, in order, and keep
            # the last status. (When every result carried the whole account, keeping the last copy was
            # right; with slices it would silently drop what happened during the earlier steps.)
            news: List[str] = []
            nar_status = None
            last_seen = None
            for e in ran:
                r = e.get("result")
                if isinstance(r, dict) and "what_happened" in r:
                    t = r.pop("what_happened")
                    if t and t != NO_NEWS:
                        news.append(t)
                    nar_status = r.pop("narrator", nar_status)
                # can_see describes the view from where the arm was after THAT step; every later step
                # moved it, so only the last one is current. The others were a fifth of a chunk's text.
                if isinstance(r, dict) and "can_see" in r:
                    last_seen = r.pop("can_see")
            out = {"ok": stopped is None, "asked": len(steps), "ran": len(ran), "steps": ran}
            if last_seen is not None:
                out["can_see"] = last_seen
            if nar_status is not None:
                out["what_happened"] = "\n".join(news) if news else NO_NEWS
                out["narrator"] = nar_status
            if stopped:
                out["stopped"] = stopped
                out["not_run"] = [s["tool"] for s in steps[len(ran):]]
                out["note"] = ("the arm is wherever that step left it. The remaining steps were not "
                               "attempted, and the state and images below are from where it stopped.")
            return ToolResult.json(out, images=images,
                                   caption=f"[do] after {ran[-1]['tool'] if ran else 'nothing'}, "
                                           f"step {len(ran)} of {len(steps)}.")

        self.register(
            name,
            "Run several of these same tools in ONE call, in order, instead of one call each. Nothing "
            "new happens: every step is a tool you could call on its own, with the arguments you would "
            "have passed. What you save is the round trip between them, which is most of the wall clock "
            "when a sequence is short motions you already know the shape of \u2014 approach, touch down, "
            "stroke. Use it when you can say in advance what the next steps are and what each one has to "
            "achieve for the next to make sense.\n"
            "Each step is {\"tool\": name, \"args\": {...}, \"require\": {...}}. `require` is how you "
            "stay in control: it names fields of that tool's OWN result and the values they must have, "
            "e.g. {\"contact\": true}, {\"stopped_by\": \"arrived\"}, {\"state.holding\": \"cup_1\"}, "
            "{\"state.gripper_gap_m\": {\">\": 0.01}}. Dotted paths reach into nested fields. The run "
            "stops at the first step whose result is not ok or whose require is unmet, and hands you "
            "everything up to and including it, so a surprise still comes straight back to you.\n"
            "Only the LAST step that ran returns camera views \u2014 batching is worth it precisely when "
            "you do not need to look in between. Split the sequence wherever you do. done() and do() "
            "itself cannot be steps: call done() on its own, after the chunk.",
            {"properties": {"steps": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "tool": {"type": "string", "description": "name of a tool in this same list"},
                    "args": {"type": "object", "description": "that tool's arguments"},
                    "require": {"type": "object", "description":
                        "field of that tool's result -> the value it must have, before the next step runs"}},
                "required": ["tool"]}}},
             "required": ["steps"]},
            do)

    def invoke(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Run one registered tool by name. The one path everything goes through.

        `do` runs its steps through here rather than fabricating a ToolCall, which carries a
        `raw_arguments` string that only means something for a call the model actually made.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error(f"unknown tool '{name}'. Available: {self.names()}")
        try:
            result = tool["fn"](**arguments)
        except TypeError as e:
            return ToolResult.error(f"bad arguments for {name}: {e}. Schema: "
                                    f"{json.dumps(tool['parameters'])}")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return ToolResult.error(f"{name} raised {type(e).__name__}: {e}")
        if not isinstance(result, ToolResult):
            return ToolResult.json(result)
        return result

    def dispatch(self, call: ToolCall) -> ToolResult:
        return self.invoke(call.name, call.arguments)
