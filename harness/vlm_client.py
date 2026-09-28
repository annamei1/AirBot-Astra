"""
Multimodal VLM client with function calling.

Backends selected by HARNESS_VLM_API (default "responses"):
  codex     : Codex App Server with ChatGPT login; structured decisions become harness tool calls.
  responses : OpenAI /v1/responses — reasoning + function tools together (required for gpt-6-astra).
              Turns are chained with previous_response_id, so each call uploads only the NEW items
              (tool outputs, new images); the server keeps the reasoning context.
  chat      : /v1/chat/completions — for gateways / other models. With tools, reasoning_effort is
              forced to "none" when the endpoint demands it.

Session = the episode transcript in a backend-neutral form (+ jsonl log). The client converts it
to whichever wire format it needs.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ==================== images ====================

def encode_image(bgr: np.ndarray, max_edge: int = 768, quality: int = 85) -> str:
    """BGR ndarray → JPEG data URI, longest edge ≤ max_edge."""
    h, w = bgr.shape[:2]
    scale = min(1.0, max_edge / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ==================== message types ====================

@dataclass
class ToolCall:
    id: str                     # call_id (responses) / tool_call id (chat)
    name: str
    arguments: Dict[str, Any]
    raw_arguments: str


@dataclass
class Reply:
    text: str
    tool_calls: List[ToolCall]
    usage: Dict[str, Any]
    elapsed: float
    response_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


# What stands in for an image the model was shown earlier and is no longer carried.
IMAGE_WITHHELD = "[image no longer carried: it was shown once, with the result it came with]"


class Session:
    """Backend-neutral transcript of one episode.

    entries: {"kind": "user", "parts": [{"type": "text", "text"} | {"type": "image", "url", "detail"}]}
             {"kind": "assistant", "text", "tool_calls": [ToolCall]}
             {"kind": "tool", "call_id", "content"}

    The transcript is sent whole every call, and on the route we use the prompt cache pays out only
    when a previously processed prompt is a prefix of the new one (measured 2026-09-22: a shared
    prefix that was never a whole prompt scores 0; a continued conversation scores everything but
    the new turn). So history is never edited turn by turn. Images are withdrawn only by compact(),
    in bulk, when the context has grown past a budget: one cache miss per compaction instead of one
    per call. Withdrawing the previous turn's image on every call cost 3.7x in a lamp episode.
    """

    def __init__(self, system_prompt: str, log_path: Optional[Path] = None, max_edge: int = 768):
        self.system_prompt = system_prompt
        self.entries: List[Dict[str, Any]] = []
        self.log_path = Path(log_path) if log_path else None
        self.max_edge = max_edge
        self.n_images = 0
        self.response_id: Optional[str] = None      # responses backend: last response in the chain
        self.sent_upto: int = 0                     # responses backend: entries already delivered
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log({"event": "system", "content": system_prompt})

    def user(self, text: str, images: Optional[List[np.ndarray]] = None, detail: str = "high"):
        self.user_parts([("text", text)] + [("image", im) for im in (images or [])], detail=detail)

    def user_parts(self, parts: List[Tuple[str, Any]], detail: str = "high"):
        """A user turn whose text and images interleave in the order given: [("text", str) | ("image", array)].

        Order matters for prefix caching: whatever is the same bytes every call belongs before
        whatever changes, and only this lets a caller put a fixed image ahead of a growing text.
        """
        out: List[Dict[str, Any]] = []
        for kind, v in parts:
            if kind == "image":
                out.append({"type": "image", "url": encode_image(v, self.max_edge), "detail": detail})
                self.n_images += 1
            else:
                out.append({"type": "text", "text": str(v)})
        self.entries.append({"kind": "user", "parts": out})
        self._log({"event": "user", "text": "\n".join(p["text"] for p in out if p["type"] == "text"),
                   "n_images": sum(p["type"] == "image" for p in out)})

    def assistant(self, reply: Reply):
        self.entries.append({"kind": "assistant", "text": reply.text, "tool_calls": reply.tool_calls})
        self._log({"event": "assistant", "text": reply.text,
                   "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in reply.tool_calls],
                   "usage": reply.usage, "elapsed_s": round(reply.elapsed, 2), "response_id": reply.response_id})

    def tool_result(self, call_id: str, content: str):
        self.entries.append({"kind": "tool", "call_id": call_id, "content": content})
        self._log({"event": "tool", "call_id": call_id, "content": content})   # whole, never truncated

    def compact(self, keep_images: int = 1) -> int:
        """Replace the images of all but the newest keep_images image-bearing turns with a text stub,
        permanently. Returns how many images were withdrawn. 0 keeps everything."""
        with_images = [i for i, e in enumerate(self.entries)
                       if e["kind"] == "user" and any(p["type"] == "image" for p in e["parts"])]
        drop = with_images[:-keep_images] if keep_images > 0 else with_images
        n = 0
        for i in drop:
            for part in self.entries[i]["parts"]:
                if part["type"] == "image":
                    part.clear(); part.update({"type": "text", "text": IMAGE_WITHHELD}); n += 1
        if n:
            self._log({"event": "compact", "images_withdrawn": n, "keep_images": keep_images})
        return n

    # ---- wire formats ----
    def chat_messages(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        for e in self.entries:
            if e["kind"] == "user":
                content: List[Dict[str, Any]] = []
                for p in e["parts"]:
                    if p["type"] == "image":
                        content.append({"type": "image_url", "image_url": {"url": p["url"], "detail": p["detail"]}})
                    else:
                        content.append({"type": "text", "text": p["text"]})
                out.append({"role": "user", "content": content})
            elif e["kind"] == "assistant":
                msg: Dict[str, Any] = {"role": "assistant", "content": e["text"] or ""}
                if e["tool_calls"]:
                    msg["tool_calls"] = [{"id": c.id, "type": "function",
                                          "function": {"name": c.name, "arguments": c.raw_arguments}}
                                         for c in e["tool_calls"]]
                out.append(msg)
            else:
                out.append({"role": "tool", "tool_call_id": e["call_id"], "content": e["content"]})
        return out

    def responses_input(self, since: int = 0) -> List[Dict[str, Any]]:
        """Items not yet delivered. Assistant entries are never re-sent (the server has them)."""
        out: List[Dict[str, Any]] = []
        # The server keeps what was delivered, so compact() has no effect on this backend.
        for e in self.entries[since:]:
            if e["kind"] == "user":
                content: List[Dict[str, Any]] = []
                for p in e["parts"]:
                    if p["type"] == "image":
                        content.append({"type": "input_image", "image_url": p["url"], "detail": p["detail"]})
                    else:
                        content.append({"type": "input_text", "text": p["text"]})
                out.append({"role": "user", "content": content})
            elif e["kind"] == "tool":
                out.append({"type": "function_call_output", "call_id": e["call_id"], "output": e["content"]})
        return out

    def _log(self, record: Dict[str, Any]):
        if not self.log_path:
            return
        record["t"] = time.time()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ==================== client ====================

class _Waiting:
    """Print progress while a model call is in flight.

    The model call is the longest single thing in the loop and was the only one that said nothing
    while it ran. When one stalls — 2026-09-18, a call that never returned — the screen shows the last
    tool result and then nothing for the full 180 s timeout, and again for each retry: up to nine
    minutes that are indistinguishable from a hung harness. This does not change what happens, only
    whether you can see it happening.
    """

    def __init__(self, label: str, first_s: float = 20.0, every_s: float = 15.0, enabled: bool = True):
        self.label, self.first, self.every, self.enabled = label, first_s, every_s, enabled
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def __enter__(self):
        if not self.enabled:
            return self
        t0 = time.time()

        def tick():
            if self._stop.wait(self.first):
                return
            while not self._stop.is_set():
                print(f"[VLM] {self.label}: still waiting, {time.time() - t0:.0f}s elapsed", flush=True)
                if self._stop.wait(self.every):
                    return

        self._t = threading.Thread(target=tick, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        return False


class VLMClient:
    def __init__(self, model: str, api_key: str = "", base_url: Optional[str] = None,
                 reasoning_effort: Optional[str] = "medium", timeout: float = 60.0,
                 temperature: Optional[float] = None, max_retries: int = 2, verbose: bool = True,
                 api: str = "responses"):
        self.model = model
        if api not in ("responses", "chat", "codex"):
            raise ValueError(f"Unknown VLM backend: {api}")
        self.api = api
        self.reasoning_effort = reasoning_effort or None
        self.temperature = temperature
        self.max_retries = max_retries
        self.verbose = verbose
        self.timeout = float(timeout)
        # Called when a request times out, and again once the call is over (answered or given up).
        # The acting model's client uses them to pause the narrator: while no reply comes the arm is
        # still, so every frame would be the same picture, saved and paid for.
        self.on_stall: Optional[Any] = None
        self.on_recover: Optional[Any] = None
        self._stalled = False
        kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        if api == "codex":
            self._client = None
        else:
            from openai import OpenAI
            self._client = OpenAI(**kwargs)
        if verbose:
            endpoint = 'ChatGPT subscription' if api == 'codex' else (base_url or 'api.openai.com')
            print(f"[VLM] model={model} api={self.api} endpoint={endpoint} "
                  f"reasoning={self.reasoning_effort}")

    # ---------------- public ----------------

    def step(self, session: Session, tools: Optional[List[Dict]] = None,
             tool_choice: str = "auto") -> Reply:
        """One model call. Does NOT append to the session (the caller appends the Reply)."""
        self._stalled = False
        try:
            if self.api == "codex":
                from harness.codex_backend import step
                try:
                    with _Waiting("Codex", enabled=self.verbose):
                        return step(self, session, tools, tool_choice)
                except Exception as e:
                    self._note_error(e)
                    raise
            if self.api == "responses":
                return self._step_responses(session, tools, tool_choice)
            return self._step_chat(session, tools, tool_choice)
        finally:
            if self._stalled and self.on_recover is not None:
                try:
                    self.on_recover()
                except Exception as e:  # noqa: BLE001
                    print(f"[VLM] on_recover failed: {e}")

    def _note_error(self, e: Exception):
        """A timed-out request means the endpoint is not answering: say so once per call."""
        if not (type(e).__name__ == "APITimeoutError" or "timed out" in str(e).lower()):
            return
        if self.verbose:
            print(f"[VLM] no reply within {self.timeout:.0f} s", flush=True)
        if not self._stalled:
            self._stalled = True
            if self.on_stall is not None:
                try:
                    self.on_stall()
                except Exception as e2:  # noqa: BLE001
                    print(f"[VLM] on_stall failed: {e2}")

    # ---------------- responses backend ----------------

    def _step_responses(self, session: Session, tools, tool_choice) -> Reply:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "instructions": session.system_prompt,
            "input": session.responses_input(session.sent_upto),
            "store": True,
        }
        if session.response_id:
            kwargs["previous_response_id"] = session.response_id
        if tools:
            kwargs["tools"] = [{"type": "function", "name": t["function"]["name"],
                                "description": t["function"]["description"],
                                "parameters": t["function"]["parameters"]} for t in tools]
            kwargs["tool_choice"] = tool_choice
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                with _Waiting(f"attempt {attempt + 1}/{self.max_retries + 1}", enabled=self.verbose):
                    resp = self._client.responses.create(**kwargs)
                reply = self._parse_responses(resp, time.time() - t0)
                session.response_id = reply.response_id
                session.sent_upto = len(session.entries)
                return reply
            except Exception as e:  # noqa: BLE001
                msg, last_err = str(e), e
                self._note_error(e)
                if "reasoning" in msg and "reasoning" in kwargs:
                    kwargs.pop("reasoning"); self.reasoning_effort = None
                    print("[VLM] endpoint rejected reasoning → dropped"); continue
                if "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature"); self.temperature = None
                    print("[VLM] endpoint rejected temperature → dropped"); continue
                if self.verbose:
                    print(f"[VLM] error (attempt {attempt + 1}/{self.max_retries + 1}): {msg[:300]}")
                if any(k in msg.lower() for k in ("authentication", "invalid_api_key", "model_not_found")):
                    break
                time.sleep(1.5)
        raise RuntimeError(f"VLM call failed: {last_err}")

    @staticmethod
    def _parse_responses(resp: Any, elapsed: float) -> Reply:
        calls: List[ToolCall] = []
        texts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            t = getattr(item, "type", None)
            if t == "function_call":
                raw = item.arguments or "{}"
                try:
                    args = json.loads(raw)
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    args = {}
                calls.append(ToolCall(id=item.call_id, name=item.name, arguments=args, raw_arguments=raw))
            elif t == "message":
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) == "output_text":
                        texts.append(part.text)
        text = (getattr(resp, "output_text", None) or "\n".join(texts) or "").strip()
        usage: Dict[str, Any] = {}
        u = getattr(resp, "usage", None)
        if u is not None:
            usage = {"input_tokens": getattr(u, "input_tokens", None),
                     "output_tokens": getattr(u, "output_tokens", None),
                     "total_tokens": getattr(u, "total_tokens", None)}
            det = getattr(u, "input_tokens_details", None)
            if det is not None and getattr(det, "cached_tokens", None) is not None:
                usage["cached_tokens"] = det.cached_tokens
            det = getattr(u, "output_tokens_details", None)
            if det is not None and getattr(det, "reasoning_tokens", None) is not None:
                usage["reasoning_tokens"] = det.reasoning_tokens
        return Reply(text=text, tool_calls=calls, usage=usage, elapsed=elapsed, response_id=resp.id,
                     raw={"status": getattr(resp, "status", None)})

    # ---------------- chat backend ----------------

    def _step_chat(self, session: Session, tools, tool_choice) -> Reply:
        kwargs: Dict[str, Any] = {"model": self.model, "messages": session.chat_messages()}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                with _Waiting(f"attempt {attempt + 1}/{self.max_retries + 1}", enabled=self.verbose):
                    resp = self._client.chat.completions.create(**kwargs)
                return self._parse_chat(resp, time.time() - t0)
            except Exception as e:  # noqa: BLE001
                msg, last_err = str(e), e
                self._note_error(e)
                if "reasoning_effort" in msg:
                    if tools and kwargs.get("reasoning_effort") != "none":
                        kwargs["reasoning_effort"] = "none"        # chat + tools: reasoning must be off
                        print("[VLM] chat endpoint: tools require reasoning_effort='none' → set"); continue
                    if "reasoning_effort" in kwargs:
                        kwargs.pop("reasoning_effort"); self.reasoning_effort = None
                        print("[VLM] endpoint rejected reasoning_effort → dropped"); continue
                if "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature"); self.temperature = None
                    print("[VLM] endpoint rejected temperature → dropped"); continue
                if self.verbose:
                    print(f"[VLM] error (attempt {attempt + 1}/{self.max_retries + 1}): {msg[:300]}")
                if any(k in msg.lower() for k in ("authentication", "invalid_api_key", "model_not_found")):
                    break
                time.sleep(1.5)
        raise RuntimeError(f"VLM call failed: {last_err}")

    @staticmethod
    def _parse_chat(resp: Any, elapsed: float) -> Reply:
        choice = resp.choices[0]
        msg = choice.message
        calls: List[ToolCall] = []
        for tc in (msg.tool_calls or []):
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw)
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args, raw_arguments=raw))
        usage = {}
        u = getattr(resp, "usage", None)
        if u is not None:
            usage = {"input_tokens": getattr(u, "prompt_tokens", None),
                     "output_tokens": getattr(u, "completion_tokens", None),
                     "total_tokens": getattr(u, "total_tokens", None)}
            # Everything a gateway or provider volunteers beyond the three basics. Cached and
            # reasoning tokens are what a latency figure has to be read against, and OpenRouter
            # reports the real per-call cost, which an end-of-month bill cannot give per run.
            # All optional: absent fields stay absent.
            pd = getattr(u, "prompt_tokens_details", None)
            if pd is not None and getattr(pd, "cached_tokens", None) is not None:
                usage["cached_tokens"] = pd.cached_tokens
            cd = getattr(u, "completion_tokens_details", None)
            if cd is not None and getattr(cd, "reasoning_tokens", None) is not None:
                usage["reasoning_tokens"] = cd.reasoning_tokens
            raw = u.model_dump() if hasattr(u, "model_dump") else {}
            if raw.get("cost") is not None:
                usage["cost_usd"] = raw["cost"]
        return Reply(text=(msg.content or "").strip(), tool_calls=calls, usage=usage, elapsed=elapsed,
                     raw={"finish_reason": choice.finish_reason})


def make_client_from_env(verbose: bool = True) -> VLMClient:
    from harness import config
    config.require_vlm_key()
    return VLMClient(model=config.VLM_MODEL, api_key=config.VLM_API_KEY, base_url=config.VLM_BASE_URL,
                     reasoning_effort=config.VLM_REASONING, timeout=config.VLM_TIMEOUT, verbose=verbose,
                     api=config.VLM_API)
