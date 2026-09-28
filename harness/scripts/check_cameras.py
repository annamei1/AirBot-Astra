"""Which cameras are here, how they are connected, and what that costs.

Run before a dual-arm session. Three D405 on one machine is where a rig stops being about the
harness and starts being about USB: a camera that negotiates USB 2.1 instead of 3.2 has about a
tenth of the bandwidth and will either stall or silently drop to a frame rate nothing here expects.
This says which, in five seconds, instead of leaving it to look like the program hanging.
"""
import sys

import pyrealsense2 as rs

WANT = {"230422271972": "head", "230422271433": "right wrist", "218622271178": "left wrist"}


def main() -> int:
    devs = list(rs.context().query_devices())
    if not devs:
        print("no RealSense devices found at all")
        return 1
    print(f"{len(devs)} RealSense device(s)\n")
    bad = 0
    for d in devs:
        def info(k, default="?"):
            try:
                return d.get_info(k)
            except Exception:  # noqa: BLE001  not every device publishes every field
                return default
        sn = info(rs.camera_info.serial_number)
        usb = info(rs.camera_info.usb_type_descriptor)
        role = WANT.get(sn, "NOT ONE OF OURS")
        ok = usb.startswith("3")
        bad += (not ok)
        print(f"  {sn}  {role:<15} {info(rs.camera_info.name)}")
        print(f"      USB {usb}   {'ok' if ok else 'FALLEN BACK TO USB 2 — this is the problem'}")
        print(f"      firmware {info(rs.camera_info.firmware_version)}  port {info(rs.camera_info.physical_port)}")
    missing = [s for s in WANT if not any(
        d.get_info(rs.camera_info.serial_number) == s for d in devs)]
    if missing:
        print(f"\nmissing: {[(s, WANT[s]) for s in missing]}")
        bad += len(missing)
    print("\nall three on USB 3 and present" if not bad else
          f"\n{bad} problem(s) above — fix these before a dual-arm run")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
