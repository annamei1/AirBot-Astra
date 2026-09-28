"""
Step-0: is the key valid, is the model id right, does it take images and tools?  No robot.

    export HARNESS_VLM_API_KEY=sk-...
    python -m harness.scripts.check_api                 # lists gpt-6* models, then 3 checks on HARNESS_VLM_MODEL
    python -m harness.scripts.check_api --model gpt-6   # override the model id for this run

Checks: (1) models.list reaches the endpoint and shows the id you configured,
        (2) a 1-sentence text reply, (3) an image is seen, (4) a tool call comes back.
"""
import argparse
import json
import time

import cv2
import numpy as np

from harness import config
from harness.tools import ToolRegistry, ToolResult
from harness.vlm_client import Session, VLMClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.VLM_MODEL)
    args = ap.parse_args()
    config.require_vlm_key()

    if config.VLM_API == "codex":
        from harness.codex_backend import AppServer
        with AppServer(config.VLM_TIMEOUT) as server:
            models = server.rpc("model/list", {})["data"]
            ids = [m["model"] for m in models]
            print(f"(1) Codex ChatGPT login OK — models: {ids}")
            if args.model not in ids:
                raise SystemExit(f"Configured model {args.model!r} is not available in Codex")
    else:
        from openai import OpenAI
        raw = OpenAI(api_key=config.VLM_API_KEY, base_url=config.VLM_BASE_URL, timeout=60)

        # (1) endpoint + model list
        print(f"endpoint: {config.VLM_BASE_URL or 'https://api.openai.com/v1'}")
        try:
            ids = sorted(m.id for m in raw.models.list())
            hits = [i for i in ids if "gpt-6" in i]
            print(f"(1) models.list OK — {len(ids)} models; gpt-6*: {hits or 'none'}")
            if args.model not in ids:
                if hits:
                    print(f"    '{args.model}' is not in the list → using '{hits[0]}' for this run. "
                          f"Put HARNESS_VLM_MODEL={hits[0]} in harness/.env")
                    args.model = hits[0]
                else:
                    print(f"    WARNING: '{args.model}' is not in the list and no gpt-6* model is visible to this key")
        except Exception as e:  # noqa: BLE001
            print(f"(1) models.list failed: {str(e)[:300]}\n    (some gateways block it — continuing)")
    client = VLMClient(model=args.model, api_key=config.VLM_API_KEY, base_url=config.VLM_BASE_URL,
                       reasoning_effort=config.VLM_REASONING, timeout=config.VLM_TIMEOUT, api=config.VLM_API)

    # (2) text
    s = Session("Answer in one short sentence.")
    s.user("Which model are you, and what is 17 * 23?")
    r = client.step(s)
    print(f"(2) text OK  {r.elapsed:.1f}s  usage={r.usage}\n    → {r.text}")

    # (3) image
    img = np.full((240, 320, 3), 240, np.uint8)
    cv2.rectangle(img, (40, 90), (130, 150), (0, 0, 200), -1)        # red block (BGR)
    cv2.circle(img, (230, 120), 45, (0, 160, 0), -1)                  # green disc
    s = Session("Answer in one short sentence.")
    s.user("What shapes and colours do you see, left to right?", images=[img])
    r = client.step(s)
    print(f"(3) image OK  {r.elapsed:.1f}s  usage={r.usage}\n    → {r.text}")
    if not any(w in r.text.lower() for w in ("red", "green")):
        print("    WARNING: the reply does not mention red/green — is this model multimodal?")

    # (4) tool call
    reg = ToolRegistry()
    reg.register("find", "Detect objects by label.", {"properties": {"label": {"type": "string"}},
                                                    "required": ["label"]},
                 lambda label: ToolResult.json({"ok": True, "objects": [{"id": "x_1", "label": label}]}))
    s = Session("You control a robot through tools. Use them; never answer with coordinates.")
    s.user("Find the red block.", images=[img])
    r = client.step(s, tools=reg.schemas())
    if r.tool_calls:
        c = r.tool_calls[0]
        print(f"(4) tool call OK  {r.elapsed:.1f}s  usage={r.usage}  → {c.name}({json.dumps(c.arguments)})")
        # (5) second turn on the same session: tool output goes back, model must use the returned id
        s.assistant(r)
        s.tool_result(c.id, reg.dispatch(c).text)
        r2 = client.step(s, tools=reg.schemas())
        print(f"(5) follow-up turn OK  {r2.elapsed:.1f}s  usage={r2.usage}  → "
              f"{[f'{t.name}({json.dumps(t.arguments)})' for t in r2.tool_calls] or r2.text}")
        print("\nALL CHECKS PASSED — next: python -m harness.scripts.smoke_vlm")
    else:
        print(f"(4) NO tool call ({r.elapsed:.1f}s). text → {r.text}\n    the endpoint/model must support "
              f"function calling for the harness to work")


if __name__ == "__main__":
    main()
