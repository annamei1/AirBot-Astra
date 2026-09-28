"""
Real-robot entry point: GPT-6 plus the harness tools on the AirBot Play arm.

    export HARNESS_VLM_API_KEY=...  HARNESS_VLM_MODEL=gpt-6-astra
    python -m harness.scripts.run_pickplace "把黑色积木放进绿色的碗里"
    python -m harness.scripts.run_pickplace --start home --no-viz "..."

The arm does NOT travel to the hand-tuned home pose on connect, and does not return there between
attempts. It rests at its zero joint position, where the wrist camera is not already aimed at the
table, so the head camera has to find the object and the wrist camera has to be carried to it. Pass
--start home for the old behaviour, or --start none to leave the arm wherever it is.

A live window shows both camera feeds and what the model is doing; q or Esc there aborts. Ctrl+C
aborts too, and the arm always comes home: one press stops after the current action and then parks,
a second press parks immediately without waiting for it, and only a third leaves the arm where it is.
Logs land in harness/logs/<episode>/ with session.jsonl, every image the model saw, and outcome.json.
"""
import argparse
import json
import os
import pathlib
import re
import signal
import sys
import threading
import time

from harness import config
from harness.agent import Agent, build_system_prompt
from harness.perception_tools import Perception
from harness.robot_env import HarnessEnv
from harness.skills import Skills
from harness.tools import ToolRegistry
from harness.viz import LiveView
from harness.vlm_client import make_client_from_env

# --arm-speed: 'slow' and 'default' are the SDK's own speed profiles. 'launch' is what a freshly started
# arm server holds (play_sdk._ArmHandle.LAUNCH_SPEED), which is what the harness ran with until
# 2026-09-26 because it never set a profile. The SDK's 'fast' is not offered: the SDK itself calls it
# experimental, and it multiplies the servo scale the guarded paths run on by a hundred.
ARM_SPEEDS = ("default", "slow", "launch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instruction", nargs="?", default=None)
    ap.add_argument("--instruction-file", type=str, default=None,
                    help="read the instruction from a file instead of the command line. Use this for "
                         "anything long: a prompt containing \" or > gets mangled by the shell before "
                         "python ever sees it, and the failure looks like the command not running")
    ap.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    ap.add_argument("--start", choices=["zero", "home", "none"], default="zero",
                    help="rest pose: zero (default), the old tuned home, or never travel to one")
    ap.add_argument("--move-on-start", action="store_true",
                    help="go to the rest pose immediately after connecting instead of staying still")
    ap.add_argument("--gripper-facts", type=str,
                    default=str(config.HARNESS_DIR / "grippers" / "airbot_g2.json"),
                    help="JSON describing the mounted gripper (finger sizes, opening, tool point); it goes into "
                         "the system prompt as facts. Default: the stock AirBot G2 fingers. Pass '' to omit.")
    ap.add_argument("--dual-arm", action="store_true",
                    help="connect the second arm too, so its wrist camera joins the rig. The harness "
                         "still drives only the world arm; this adds a third view, not a second hand")
    ap.add_argument("--compact-at", type=int, default=30000,
                    help="when the prompt has grown by this many tokens since the last compaction, withdraw "
                         "earlier images from the context in one go (one cache miss). 0 never compacts "
                         "(default: 30000)")
    ap.add_argument("--keep-images", type=int, default=1,
                    help="image-bearing turns that keep their images at a compaction (default: 1)")
    ap.add_argument("--keep-logs", type=int, default=1,
                    help="episode log directories to keep under harness/logs, this run included; "
                         "older ones are deleted when a run starts. 0 keeps everything (default: 1)")
    ap.add_argument("--narrator", action="store_true",
                    help="run a second copy of the model that watches the head camera continuously and "
                         "keeps a written account; every tool result then carries it as what_happened. "
                         "For tasks where the world changes while the arm is still. Costs a model call "
                         "every few seconds for the whole episode")
    ap.add_argument("--narrator-hz", type=float, default=None,
                    help="narrator capture rate (default from harness/narrator.py, 1 Hz)")
    ap.add_argument("--arm-speed", choices=ARM_SPEEDS, default="default",
                    help="speed profile written to every arm server at start: the SDK's 'default' or "
                         "'slow', or 'launch' (a fresh server's values, what the harness used before "
                         "2026-09-26). All five parameters are read back into each episode's record.")
    ap.add_argument("--no-viz", action="store_true", help="run without the live camera window")
    args = ap.parse_args()
    if args.instruction_file:
        if args.instruction:
            raise SystemExit("give an instruction OR --instruction-file, not both")
        args.instruction = pathlib.Path(args.instruction_file).read_text(encoding="utf-8").strip()
        if not args.instruction:
            raise SystemExit(f"{args.instruction_file} is empty")
        print(f"[harness] instruction from {args.instruction_file} "
              f"({len(args.instruction)} chars, {args.instruction.count(chr(10)) + 1} lines)")
    config.require_vlm_key()

    from perception.sam3_segmenter import create_segmenter
    from perception.position_calculator import HandEyeCalculator
    from robot.motion_atomic import AtomicMotionExecutor
    from harness.scripts.smoke_perception import build_head_calc

    # Refuse to connect if a tool body reads a name that does not exist. Python resolves globals at
    # call time, so `import` proves nothing about the tool layer, and the tool layer is the one part
    # of this repo that cannot be unit-tested because every tool moves the arm. On 2026-09-16 that
    # cost a whole whiteboard session: the model was holding the eraser when follow_path raised
    # NameError. 0.1 s here against a wasted run there.
    from harness.scripts.check_names import main as check_names
    if not os.environ.get('HARNESS_SKIP_NAME_CHECK') and check_names() != 0:
        raise SystemExit("undefined names above; fix them before moving the arm "
                         "(skip with HARNESS_SKIP_NAME_CHECK=1)")

    env = HarnessEnv(rest=args.start, move_on_start=args.move_on_start, second_arm=args.dual_arm)
    segmenter = create_segmenter(config.SAM3_CHECKPOINT, confidence=config.SAM3_CONFIDENCE)
    head_calc = build_head_calc()
    he_intr, he_extr = config.load_hand_eye_calibration()
    handeye_calc = HandEyeCalculator(he_intr, he_extr)
    executor = AtomicMotionExecutor(robot_env=env, segmenter=segmenter, head_calc=None,
                                    handeye_calc=handeye_calc, llm_planner=None)

    abort = threading.Event()
    executor.set_callbacks(check_abort=abort.is_set)

    # One executor per arm. An ArmView is a drop-in robot_env — same method names, world frame — so the
    # second arm gets the SAME atomic actions rather than a thinner second implementation. The one that
    # matters is the incremental gripper closure: it is what tells one layer of cloth from an empty
    # gripper, and a folding task needs that from both hands.
    from harness.arms import SECOND_ARM, WORLD_ARM, make_arms
    arms = make_arms(env)               # built once here and handed to Skills, so both halves of the
    executors = {}                      # harness drive the same view objects
    if SECOND_ARM in arms:
        left_intr, left_extr = config.load_left_hand_eye_calibration()
        executors[SECOND_ARM] = AtomicMotionExecutor(
            robot_env=arms[SECOND_ARM], segmenter=segmenter, head_calc=None,
            handeye_calc=HandEyeCalculator(left_intr, left_extr), llm_planner=None)
        executors[SECOND_ARM].set_callbacks(check_abort=abort.is_set)

    parked = threading.Event()
    interrupts = {"n": 0}

    def go_rest():
        """Open every gripper, then take every arm to the rest pose. Blocking."""
        # Both grippers open BEFORE either arm travels. With two arms the order matters: cloth held
        # between them means an arm that parks while the other still grips drags the cloth — and
        # whatever else is on the table — with it.
        for name, view in arms.items():
            if name != WORLD_ARM:
                view.open_gripper(0.08)
        env.open_gripper(0.08)
        env.reset_position()
        for name, view in arms.items():
            if name != WORLD_ARM and args.start != "none":
                view.reset_position()

    def park(reason: str):
        """Open the gripper and travel to the rest pose. Runs once, whichever path reaches it."""
        if parked.is_set():
            return
        parked.set()
        # The interrupt that brought us here must not cancel the journey home: the executor and the
        # guarded path both check this flag, and a set flag would turn the parking move into a no-op.
        abort.clear()
        where = "wherever it is (rest pose is 'none')" if args.start == "none" else f"the '{args.start}' pose"
        try:
            print(f"\n[harness] {reason}: opening the gripper and returning to {where}")
            go_rest()
        except Exception as e:  # noqa: BLE001  never let cleanup mask the original exit
            print(f"[harness] could not park the arm: {type(e).__name__}: {e}")

    def on_sigint(*_):
        """Three presses, each with a clearly different promise.

        The previous version exited on the second press without moving, which left the arm wherever
        the interrupt caught it — often leaning into something with an object in its fingers. Parking
        is now what a second press does, on its own thread so that a main thread stuck inside a
        blocking SDK call cannot prevent it, and only a third press abandons the arm.
        """
        interrupts["n"] += 1
        if interrupts["n"] == 1:
            print("\n[harness] interrupt: stopping after the current action, then returning to the rest "
                  "pose. Ctrl+C again to go home immediately; a third time to quit and leave the arm.")
            abort.set()
        elif interrupts["n"] == 2:
            threading.Thread(target=lambda: (park("interrupted"), os._exit(130)), daemon=True).start()
        else:
            print("\n[harness] quitting without parking — the arm stays where it is")
            os._exit(130)

    signal.signal(signal.SIGINT, on_sigint)

    perception = Perception(env, segmenter, head_calc, table_z=config.TABLE_SURFACE_Z,
                            handeye_calc=handeye_calc)
    narrator = None
    if args.narrator:
        from harness.narrator import NARRATOR_HZ, Narrator
        # Same model, same reasoning setting as the acting half, so the two halves speak alike.
        # Each arm's wrist camera, by the names the rig uses; the narrator sends an arm's wrist
        # frames only for the seconds that arm is acting.
        cams = perception.cameras()
        wrists = {a: c for a, c in ((WORLD_ARM, "wrist"), (SECOND_ARM, "left_wrist")) if c in cams}
        narrator = Narrator(perception, lambda: make_client_from_env(verbose=False),
                            hz=args.narrator_hz or NARRATOR_HZ, max_edge=config.IMAGE_MAX_EDGE, wrists=wrists)
        perception.narrator = narrator
    skills = Skills(env, executor, perception, hover_height=config.HOVER_HEIGHT, grasp_z=config.GRASP_Z,
                    table_z=config.TABLE_SURFACE_Z, bounds_min=config.WORKSPACE_BOUNDS_MIN,
                    bounds_max=config.WORKSPACE_BOUNDS_MAX, abort=abort, executors=executors, arms=arms)
    # Every arm by the harness's own names, as play_sdk handles (PlayRealRobot calls the world arm 'left'
    # on a two-arm rig; the harness calls it 'right').
    handles = {WORLD_ARM: env.robot.left}
    if SECOND_ARM in arms and getattr(env, "second_arm", None) is not None:
        handles[SECOND_ARM] = env.second_arm
    speed_params = {}
    for name, h in handles.items():
        speed_params[name] = h.set_speed(args.arm_speed)
        print(f"[harness] {name} arm speed '{args.arm_speed}', read back: {speed_params[name]}")
        if speed_params[name] is None:
            print(f"[harness] WARNING: the {name} arm server did not return its speed parameters")
    reg = ToolRegistry()
    perception.register_tools(reg)
    skills.register_tools(reg)

    viz = None
    if not args.no_viz:
        viz = LiveView(perception, abort=abort)
        viz.start()

    gripper = None
    facts_path = args.gripper_facts or None
    if facts_path:
        from harness.gripper import load_gripper_facts
        if not os.path.exists(facts_path):
            raise SystemExit(f"{facts_path} not found (--gripper-facts)")
        gripper = load_gripper_facts(facts_path)
        print(f"[harness] gripper facts from {facts_path}: {gripper['finger_length_mm']} mm fingers, "
              f"{gripper['width_across_closing_mm']['max']} mm wide, open to {gripper['max_opening_mm']} mm")
        tool = gripper.get("tool") or {}
        off = tool.get("tool_offset_in_reported_frame_m")
        if off is not None:
            env.robot.set_tool_offset(off)
            print(f"[harness] tool point = SDK point + {[round(v * 1000, 1) for v in off]} mm (in the SDK frame): "
                  f"fingertip {tool.get('finger_tip_forward_of_flange_m', 0) * 1000:.1f} mm ahead of the flange, "
                  f"grasp point {tool.get('grasp_point_behind_tip_m', 0) * 1000:.1f} mm behind it")
    agent = Agent(make_client_from_env(), reg, perception,
                  build_system_prompt(config.TABLE_SURFACE_Z, config.WORKSPACE_BOUNDS_MIN,
                                      config.WORKSPACE_BOUNDS_MAX, config.HOVER_HEIGHT,
                                      cameras=perception.cameras(), arms=skills.arm_names,
                                      narrator=narrator is not None, keep_images=args.keep_images,
                                      gripper=gripper, rest=args.start),
                  log_root=config.LOG_DIR, max_steps=args.max_steps, max_edge=config.IMAGE_MAX_EDGE,
                  abort=abort, viz=viz, narrator=narrator, keep_logs=args.keep_logs,
                  keep_images=args.keep_images, compact_at=args.compact_at,
                  gripper=gripper,
                  run_meta={"arm_speed": args.arm_speed, "speed_params": speed_params})
    # Registered last so it can dispatch every other tool, including ones added after it here.
    reg.register_chunk()

    print(f"[harness] tools: {reg.names()}")

    try:
        while True:
            instruction = args.instruction or input("\n[harness] 任务指令 / instruction (q to quit): ").strip()
            if not instruction or instruction.lower() == "q":
                break
            perception.new_episode()
            skills.new_episode()
            agent.run(instruction)
            abort.clear()
            print(f"\n[harness] episode over: opening the gripper and returning to the '{args.start}' pose")
            go_rest()
            if args.instruction:
                break
    except KeyboardInterrupt:
        abort.set()
    finally:
        try:
            park("finished")
        finally:
            env.disconnect()
            sys.stdout.flush()
            sys.stderr.flush()
            # The RealSense reader threads are not daemons: without this the interpreter hangs
            # here after everything is done, and Ctrl+C is already swallowed by the handler.
            os._exit(0)


if __name__ == "__main__":
    main()
