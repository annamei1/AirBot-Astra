"""
harness — a VLM-driven harness for AirBot Play arms: the model reasons, the harness measures.

Layout:
    config.py            VLM endpoint from environment variables, robot constants from play.config
    vlm_client.py        multimodal chat + function calling + episode session/log
    tools.py             tool registry (JSON schema → python function) and ToolResult
    cameras.py           one View per camera frame: intrinsics + camera-to-base, project / backproject
    perception_tools.py  look / find / refine / measure across every camera into one object table
    skills.py            the motion primitives over AtomicMotionExecutor / GuardedPath
    narrator.py          a second model that watches the head camera and keeps an account of what happened
    agent.py             the loop: instruction + image → tool calls → observations → done
    gripper.py           the gripper facts (grippers/*.json) the model is told
    scripts/             smoke tests, frame capture, real-robot entry point
"""
