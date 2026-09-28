"""ChatGPT-authenticated Codex App Server adapter (no API key).

Each decision uses an ephemeral thread and the current Session transcript. Structured
output is translated to harness ToolCalls; only the harness executes those calls.
This intentionally does not keep hidden server history, so Session.compact works.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid


class AppServer:
    """One bounded stdio JSON-RPC connection. Never reads or copies login tokens."""

    def __init__(self, timeout=60.0, executable=None):
        executable = executable or os.environ.get("HARNESS_CODEX_BIN", "codex")
        if not shutil.which(executable):
            raise RuntimeError("Codex CLI not found. Install it and run `codex login` with ChatGPT.")
        self.deadline = time.monotonic() + timeout
        self.messages = queue.Queue()
        self.pending = []
        self.next_id = 0
        self.workdir = tempfile.TemporaryDirectory(prefix="airbot-codex-")
        self.stderr = tempfile.TemporaryFile(mode="w+b")
        env = os.environ.copy()
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "HARNESS_VLM_API_KEY", "OPENAI_BASE_URL"):
            env.pop(key, None)
        cmd = [executable, "app-server", "--listen", "stdio://"]
        # Retain the user's managed ChatGPT login, but disable agent capabilities.
        for feature in ("shell_tool", "unified_exec", "apps", "plugins", "hooks",
                        "multi_agent", "browser_use", "computer_use", "image_generation",
                        "memories", "skill_search", "view_image", "code_mode", "code_mode_host"):
            cmd += ["-c", f"features.{feature}=false"]
        cmd += ["-c", 'web_search="disabled"', "-c", 'forced_login_method="chatgpt"']
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self.stderr, text=True, bufsize=1,
                                         cwd=self.workdir.name, env=env)
        except BaseException:
            self.stderr.close()
            self.workdir.cleanup()
            raise
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.rpc("initialize", {"clientInfo": {"name": "airbot_astra", "version": "0.1.0"},
                                    "capabilities": {"experimentalApi": True}})
            self.send({"method": "initialized", "params": {}})
            account = self.rpc("account/read", {"refreshToken": False}).get("account") or {}
            if account.get("type") != "chatgpt":
                raise RuntimeError("Codex backend requires a ChatGPT login. Run `codex login` first.")
            self.account_type = account["type"]
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.proc.stdout:
                try:
                    self.messages.put(json.loads(line))
                except json.JSONDecodeError:
                    self.messages.put(RuntimeError("Invalid JSON from Codex App Server"))
        finally:
            self.messages.put(EOFError("Codex App Server closed its connection"))

    def send(self, message):
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def receive(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Codex request timed out")
        try:
            message = self.messages.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError("Codex request timed out") from exc
        if isinstance(message, BaseException):
            raise message
        if "id" in message and "method" in message:
            self.send({"id": message["id"], "error": {"code": -32601,
                       "message": "This integration supports structured decisions only"}})
            raise RuntimeError(f"Unexpected Codex capability request: {message['method']}")
        return message

    def rpc(self, method, params):
        self.next_id += 1
        request_id = self.next_id
        self.send({"id": request_id, "method": method, "params": params})
        while True:
            message = self.receive()
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(f"Codex {method}: {message['error'].get('message', 'RPC error')}")
                return message["result"]
            self.pending.append(message)

    def event(self):
        return self.pending.pop(0) if self.pending else self.receive()

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.reader.join(timeout=1)
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.stderr.close()
        self.workdir.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def transcript_input(session):
    """Preserve ordering and tool-result ids, including images after observations."""
    parts = []
    for entry in session.entries:
        kind = entry["kind"]
        if kind == "user":
            parts.append({"type": "text", "text": "USER observation/instruction:"})
            for part in entry["parts"]:
                parts.append({"type": "image", "url": part["url"]} if part["type"] == "image"
                             else {"type": "text", "text": part["text"]})
        elif kind == "assistant":
            record = {"text": entry["text"], "tool_calls": [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in entry["tool_calls"]]}
            parts.append({"type": "text", "text": "ASSISTANT decision: " + json.dumps(record)})
        elif kind == "tool":
            parts.append({"type": "text", "text": "TOOL result: " + json.dumps(entry)})
    return parts


OUTPUT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["text", "tool_calls"],
    "properties": {
        "text": {"type": "string"},
        "tool_calls": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "arguments_json"],
            "properties": {"name": {"type": "string"}, "arguments_json": {"type": "string"}}}},
    },
}


def parse_decision(text, tools, tool_choice):
    from harness.vlm_client import ToolCall
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("text"), str) or not isinstance(data.get("tool_calls"), list):
        raise ValueError("Invalid Codex decision envelope")
    allowed = {t["function"]["name"] for t in (tools or [])}
    calls = []
    for call in data["tool_calls"]:
        name, raw = call["name"], call["arguments_json"]
        if name not in allowed or tool_choice == "none":
            raise ValueError(f"Codex returned a disallowed tool: {name}")
        args = json.loads(raw)
        if not isinstance(args, dict):
            raise ValueError(f"Codex arguments for {name} must be an object")
        calls.append(ToolCall("codex_" + uuid.uuid4().hex, name, args, raw))
    if tool_choice == "required" and not calls:
        raise ValueError("Codex did not return a required tool call")
    return data["text"], calls


def step(client, session, tools, tool_choice):
    from harness.vlm_client import Reply
    if tool_choice not in ("auto", "none", "required"):
        raise ValueError("Codex backend supports tool_choice auto, none, or required")
    if tool_choice == "required" and not tools:
        raise ValueError("tool_choice required needs tools")
    started = time.monotonic()
    instructions = session.system_prompt + "\n\n" + (
        "You are the decision engine of an external harness. Return ONLY the requested JSON decision. "
        "The transcript below contains prior observations, decisions, and actual tool results. "
        "Continue from its end. To call a harness tool, put its name and JSON-encoded argument object "
        "in tool_calls using arguments_json. The harness executes it AFTER this reply. "
        "Never invent results or claim an action executed without a TOOL result. "
        "Do not use Codex tools, files, shell commands, web, plugins, or other agents. "
        "If no action is needed, return an empty tool_calls array.\n"
        f"Tool choice: {tool_choice}. Available harness tool schemas:\n" + json.dumps(tools or []))
    with AppServer(client.timeout) as server:
        thread = server.rpc("thread/start", {
            "model": client.model, "modelProvider": "openai", "ephemeral": True,
            "cwd": server.workdir.name, "sandbox": "read-only", "approvalPolicy": "never",
            "baseInstructions": instructions, "developerInstructions": "",
            "environments": [], "config": {"project_doc_max_bytes": 0},
        })["thread"]["id"]
        params = {"threadId": thread, "input": transcript_input(session),
                  "outputSchema": OUTPUT_SCHEMA, "environments": []}
        if client.reasoning_effort:
            params["effort"] = client.reasoning_effort
        turn = server.rpc("turn/start", params)["turn"]["id"]
        final_text = None
        usage = {}
        while True:
            event = server.event()
            p = event.get("params", {})
            if p.get("threadId") != thread:
                continue
            method = event.get("method")
            if method == "item/completed" and p.get("turnId") == turn:
                item = p["item"]
                if item.get("type") == "agentMessage" and item.get("phase") != "commentary":
                    final_text = item.get("text")
            elif method == "thread/tokenUsage/updated":
                u = p.get("tokenUsage", {}).get("last", {})
                usage = {dst: u[src] for src, dst in (
                    ("inputTokens", "input_tokens"), ("outputTokens", "output_tokens"),
                    ("totalTokens", "total_tokens"), ("cachedInputTokens", "cached_tokens"),
                    ("reasoningOutputTokens", "reasoning_tokens")) if src in u}
            elif method == "turn/completed" and p.get("turn", {}).get("id") == turn:
                result = p["turn"]
                if result.get("status") != "completed":
                    raise RuntimeError(f"Codex turn {result.get('status')}: {result.get('error')}")
                break
        if not final_text:
            raise RuntimeError("Codex completed without a structured decision")
        text, calls = parse_decision(final_text, tools, tool_choice)
        return Reply(text, calls, usage, time.monotonic() - started, response_id=turn,
                     raw={"backend": "codex", "thread_id": thread, "auth": server.account_type})
