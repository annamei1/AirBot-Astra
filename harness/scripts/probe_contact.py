"""
Measure what "contact" actually looks like on this arm, instead of inheriting a threshold.

Run 1 (2026-09-16) produced these, and they are why the motion layer was rewritten:

    resting, gripper open                3.97 A
    holding the pose, gripper closed     6.83 A
    free-air motion, peak                8.31 A
    descending freely, peak              6.06 A

A descent draws LESS than holding still, because gravity does the work. No absolute threshold can
separate contact from motion across that spread: the inherited 13 A sat above everything, so both
releases in the 02:34 run drove an object into a bowl without ever registering contact. Contact is
now a rise above the baseline each motion establishes for itself, plus a second signal that costs
nothing — the arm ceasing to make progress while the setpoint is still ahead of it.

That run also showed the path follower was not tracking: every servo move travelled about 25 mm no
matter whether 30 mm or 60 mm was commanded, because the streamed setpoint was computed from the
starting pose and ran away from the arm. Step 4 below is the check that this is fixed: it must now
say "arrived", not "timeout".

    python -m harness.scripts.probe_contact
    python -m harness.scripts.probe_contact --motion stepped      # if servo tracking is still wrong
    python -m harness.scripts.probe_contact --x 0.28 --y 0.10 --max-drop 0.08

The gripper is CLOSED for the probe so the fingers meet the table together, and the descent stops at
the first sign of resistance or after --max-drop. Keep the area under the probe point clear.
"""
import argparse
import os
import signal
import statistics
import sys
import threading
import time

from harness import config
from harness.motion import CONTACT_FALL_A, CONTACT_RISE_A, OVERCURRENT_A, GuardedPath
from harness.robot_env import HarnessEnv


def surface_under(rig, x: float, y: float, nominal_z: float, win: int = 5, only=None) -> dict:
    """Measured height of whatever is under (x, y), from each camera that can see it.

    Run 2 descended to 3 mm below the configured table height and still touched nothing, because the
    configured height is not the physical one: `TABLE_SURFACE_Z` is -0.066 while the wrist camera
    measured the plate at the brick at -0.075. A probe that is trying to find out where the surface
    is has no business taking that constant on trust, so it measures first and descends second.
    """
    import numpy as np
    out = {}
    for name in rig.names():
        if only is not None and name not in only:
            continue
        view = rig.grab(name)
        if view is None or view.depth is None:
            continue
        pr = view.project([x, y, nominal_z])
        if pr is None:
            continue
        u, v, _ = pr
        if not view.contains(u, v, margin=win + 1):
            continue
        patch = view.depth[int(v) - win:int(v) + win + 1, int(u) - win:int(u) + win + 1].astype(float)
        good = patch[(patch > 30) & (patch < 2000)]
        if good.size < 0.3 * patch.size:
            out[name] = {"z": None, "why": f"only {good.size}/{patch.size} pixels carry depth"}
            continue
        d = float(np.median(good)) / 1000.0
        out[name] = {"z": float(view.pixel_to_base(u, v, d)[2]), "px": [int(u), int(v)],
                     "range_m": round(d, 4), "valid": round(float(good.size) / patch.size, 2),
                     "spread_mm": round(float(np.percentile(good, 90) - np.percentile(good, 10)), 1)}
    return out


def sample(env, seconds: float, label: str) -> dict:
    vals, end = [], time.monotonic() + seconds
    while time.monotonic() < end:
        c = env.get_arm_total_effort()
        if c is not None:
            vals.append(float(c))
        time.sleep(0.02)
    if not vals:
        print(f"  {label:<34} no readings — get_arm_total_effort() returned None every time")
        return {}
    out = {"n": len(vals), "min": min(vals), "median": statistics.median(vals), "max": max(vals)}
    print(f"  {label:<34} median {out['median']:5.2f} A   min {out['min']:5.2f}   max {out['max']:5.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=0.30)
    ap.add_argument("--y", type=float, default=0.00)
    ap.add_argument("--start-above", type=float, default=0.05, help="start this far above the table (m)")
    ap.add_argument("--max-drop", type=float, default=0.09, help="give up after descending this far (m)")
    ap.add_argument("--rise", type=float, default=CONTACT_RISE_A, help="contact = this many A above baseline")
    ap.add_argument("--fall", type=float, default=CONTACT_FALL_A,
                    help="contact = this many A BELOW baseline, which is what resting on a surface looks like")
    ap.add_argument("--floor-margin", type=float, default=0.025,
                    help="hard backstop: how far below the workspace floor the probe may ever reach (m)")
    ap.add_argument("--below-surface", type=float, default=0.004,
                    help="how far below the MEASURED surface to aim, so contact is certain (m)")
    ap.add_argument("--motion", choices=["auto", "servo", "stepped"], default="auto")
    ap.add_argument("--start", choices=["zero", "home", "none"], default="none")
    ap.add_argument("--arm-speed", choices=["default", "slow", "launch", "keep"], default="default",
                    help="speed profile to write first, as run_pickplace does; 'keep' leaves the server as it is")
    args = ap.parse_args()

    table_z = float(config.TABLE_SURFACE_Z)
    z_start = table_z + args.start_above
    print(f"\nProbe point [{args.x:.3f}, {args.y:.3f}], starting {args.start_above * 100:.0f} cm above the "
          f"table (z={z_start:.4f}), descending at most {args.max_drop * 100:.0f} cm.")
    print(f"Contact = {args.rise:.1f} A above the motion's own baseline, or the arm stopping while pushed.")
    print(f"Emergency stop: {OVERCURRENT_A:.1f} A.   Motion mode: {args.motion}.")
    print("Clear the area under the probe point, then press Enter (Ctrl+C to abort).")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        print("aborted")
        return

    env = HarnessEnv(rest=args.start)
    if args.arm_speed != "keep":
        print(f"[probe] arm speed '{args.arm_speed}', read back: {env.robot.left.set_speed(args.arm_speed)}")
    else:
        print(f"[probe] arm speed left as it is: {env.robot.left.speed_params()}")
    abort = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: (print("\n[probe] interrupt"), abort.set()))
    path = GuardedPath(env, abort=abort)
    if args.motion == "stepped":
        path._servo_available = False

    rig = None
    try:
        from harness.cameras import CameraRig
        from harness.scripts.smoke_perception import build_head_calc
        from perception.position_calculator import HandEyeCalculator
        he_intr, he_extr = config.load_hand_eye_calibration()
        rig = CameraRig(env, build_head_calc(), HandEyeCalculator(he_intr, he_extr))
    except Exception as e:  # noqa: BLE001  the currents are still worth having without the cameras
        print(f"[probe] cameras unavailable ({type(e).__name__}: {e}); will fall back to the "
              f"configured table height")

    tracking_ok = None
    free_peak = free_base = None
    try:
        print("\n1. resting current")
        sample(env, 2.0, "resting, gripper open")

        print("\n2. closing the gripper so the fingers meet the table together")
        env.close_gripper()
        time.sleep(0.5)

        # The head camera has to read the surface BEFORE the arm parks over it. Measuring afterwards
        # is measuring the robot: on 2026-09-16 it put the surface at +0.1038 while the wrist, which
        # was right above it, said -0.0623 — the head was looking at the arm's own body, 166 mm out.
        # The wrist is the opposite case: it can only read the point once it is there. So each camera
        # measures from the pose it can actually see the point from.
        print("\n3. measuring from the head camera, before the arm stands over the point")
        head_reading = {}
        if rig is not None:
            try:
                head_reading = surface_under(rig, args.x, args.y, table_z, only={"head"})
            except Exception as exc:  # noqa: BLE001  one camera failing is not a reason to stop probing
                print(f"   the head camera could not measure it ({type(exc).__name__}: {exc}); "
                      f"the wrist at step 6 will have to carry the probe")
        if not head_reading:
            print("   the head camera returned nothing for this point")
        for name, m in head_reading.items():
            if m.get("z") is None:
                print(f"   {name:<6} no reading: {m['why']}")
            else:
                print(f"   {name:<6} surface z {m['z']:+.4f}   range {m['range_m']:.3f} m   "
                      f"valid {m['valid']:.0%}   spread {m['spread_mm']:.1f} mm")

        print("\n4. travelling to the probe point")
        if not env.move_arm([args.x, args.y, z_start], 0.0, wait=0.0):
            print("   the arm could not reach the probe point; try a smaller x or y")
            return
        time.sleep(0.3)
        sample(env, 1.0, "holding the pose, gripper closed")

        print("\n5. free air: 3 cm up and back down, nothing in the way, no contact stop")
        print("   (this is the tracking check: it must say 'arrived', not 'timeout')")
        for label, dz in (("up", 0.03), ("down", -0.03)):
            z = float(env.get_tcp_position()[2])
            # Both contact signals off, not just the rise: this step is meant to run to the end
            # untouched, and a descent through free air dips below its own baseline as gravity takes
            # over, which is exactly what the fall detector is looking for.
            res = path.follow([[args.x, args.y, z + dz]], contact_rise_a=None, contact_fall_a=None,
                              stop_when_blocked=False)
            print(f"   {label:<5} {res.stopped_by:<9} travelled {res.travelled_m * 1000:5.1f} mm of "
                  f"{abs(dz) * 1000:.0f}   baseline {res.baseline_a} A   peak {res.peak_current_a} A   "
                  f"({res.mode})")
            tracking_ok = (res.stopped_by == "arrived") if tracking_ok is None else \
                (tracking_ok and res.stopped_by == "arrived")
            free_peak = max(free_peak or 0.0, res.peak_current_a or 0.0)
            free_base = res.baseline_a if res.baseline_a is not None else free_base

        print("\n6. measuring from the wrist camera, now that it is over the point")
        measured = dict(head_reading)
        if rig is not None:
            measured.update(surface_under(rig, args.x, args.y, table_z,
                                          only={n for n in rig.names() if n != "head"}))
        for name, m in measured.items():
            if m.get("z") is None:
                print(f"   {name:<6} no reading: {m['why']}")
            else:
                print(f"   {name:<6} surface z {m['z']:+.4f}   range {m['range_m']:.3f} m   "
                      f"valid {m['valid']:.0%}   spread {m['spread_mm']:.1f} mm")
        # Take the CLOSEST camera, not the lowest reading. Run 3 took the lowest and got the head
        # camera's, which reads 8 mm low on this rig; it then aimed 8 mm deeper than it needed to and
        # was saved by the contact stop. The closest view is also the one refine() already trusts.
        usable = {k: m for k, m in measured.items() if m.get("z") is not None}
        if usable:
            best = min(usable, key=lambda k: usable[k]["range_m"])
            surface = usable[best]["z"]
            print(f"   taking {best}, the closest view at {usable[best]['range_m']:.3f} m: {surface:+.4f}, "
                  f"against the configured {table_z:+.4f} ({(surface - table_z) * 1000:+.1f} mm)")
            spread = [m["z"] for m in usable.values()]
            if len(spread) > 1:
                print(f"   the cameras disagree by {(max(spread) - min(spread)) * 1000:.1f} mm on this spot")
        else:
            surface = table_z
            print(f"   no camera could measure it; falling back to the configured {table_z:+.4f}")

        print("\n7. descending onto it, stopping on contact")
        before = float(env.get_tcp_position()[2])
        # Deep enough that the fingertips must meet the surface whatever their offset turns out to be,
        # and no deeper. If both contact detectors fail this margin is all that protects the surface.
        bottom = max(surface - args.below_surface, before - args.max_drop,
                     float(config.WORKSPACE_BOUNDS_MIN[2]) - args.floor_margin)
        print(f"   descending to z={bottom:+.4f}, {(surface - bottom) * 1000:.0f} mm below the measured "
              f"surface")
        res = path.follow([[args.x, args.y, bottom]], contact_rise_a=args.rise,
                          contact_fall_a=args.fall)
        after = float(env.get_tcp_position()[2])
        print(f"   {res.stopped_by}   travelled {res.travelled_m * 1000:.1f} mm   baseline {res.baseline_a} A"
              f"   peak {res.peak_current_a} A   at {res.current_a} A   rise {res.rise_a} A")

        print("\n" + "=" * 74)
        print(f"TRACKING   {'ok, the arm reaches what it is asked to reach' if tracking_ok else 'STILL WRONG — the servo path did not arrive'}")
        if not tracking_ok and args.motion != "stepped":
            print("           re-run with --motion stepped to get the contact numbers anyway")
        print(f"CURRENTS   free-air peak {free_peak:.2f} A" + (f", baseline {free_base:.2f} A" if free_base else ""))

        config_offset_mm = float(config.FINGERTIP_OFFSET_M) * 1000
        if res.stopped_by in ("contact", "blocked"):
            if res.stopped_by != "contact":
                how = "the arm stopped while still being pushed"
            elif res.rise_a is not None and res.rise_a < 0:
                how = (f"the effort FELL {abs(res.rise_a):.2f} below this motion's baseline — the "
                       f"surface took the arm's weight")
            else:
                how = f"the effort ROSE {res.rise_a:+.2f} above this motion's baseline"
            offset_mm = (after - surface) * 1000
            print(f"CONTACT    felt because {how}")
            print(f"           gripper z {after:+.4f}   measured surface {surface:+.4f}")
            # Only the fingertip reading if the fingers really did meet the surface the cameras
            # measured. A detector that fired early and a surface reading that is too high look
            # identical from here, so say the gap and let the numbers be compared across runs rather
            # than announcing a body parameter from one of them.
            print(f"           gap at the stop {offset_mm:+.1f} mm "
                  f"(fingertip reach IF the fingers met that surface; "
                  f"config says {config_offset_mm:+.1f} mm)")
            print( "           (a LOWER bound: the arm stopped after the current had already risen, so the")
            print( "            fingers had pressed in a little by the time it noticed)")
            if res.rise_a is not None:
                print(f"           rise at contact {res.rise_a:+.2f} A over a baseline of {res.baseline_a} A")
                if res.rise_a < args.rise * 1.5:
                    print(f"           that is barely over the {args.rise:.1f} A threshold. If the arm ever stops")
                    print( "           early in free air, lower the speed or raise the rise a little.")
            else:
                print("           the current never rose measurably; only the stall signal caught it, which")
                print("           means current alone cannot detect this contact and stop_when_blocked is")
                print("           doing the real work. That is fine, but do not rely on the rise for cloth.")
        elif res.stopped_by == "arrived":
            print(f"NO CONTACT   the arm reached z={after:+.4f} and nothing held it up.")
            print(f"           The surface was measured at {surface:+.4f} and the descent aimed for "
                  f"{bottom:+.4f}.")
            if bottom > surface - args.below_surface + 1e-6:
                print(f"           That was short of the aim: a backstop cut it off. Raise --floor-margin.")
            else:
                print(f"           So the fingertips reach LESS far below the reported gripper position than")
                print(f"           the 3-9 mm the README assumes, or the surface reading is too low.")
                print(f"           Re-run with --below-surface {args.below_surface + 0.004:.3f}.")
            print(f"           peak {res.peak_current_a} A over a baseline of {res.baseline_a} A.")
        else:
            print(f"ENDED AS   {res.stopped_by}: {res.error}")
        print("=" * 74)

    except BaseException:
        # os._exit(0) in the finally skips Python's own traceback, so until now an exception anywhere
        # in this probe simply vanished: the 09:5x run printed step 3's header, then "retreating", and
        # nothing else — no error, no clue, and the arm never moved. Print it while the interpreter is
        # still alive to do it.
        import traceback
        traceback.print_exc()
    finally:
        try:
            print("\nretreating and opening the gripper")
            tcp = env.get_tcp_position()
            if tcp is not None:
                env.move_arm([float(tcp[0]), float(tcp[1]), table_z + 0.12], 0.0, wait=0.0)
            env.open_gripper(0.08)
        finally:
            env.disconnect()
            sys.stdout.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
