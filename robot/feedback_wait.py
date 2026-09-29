"""Bounded feedback verification; acceptance is not completion."""
import math
import time


def wait_for_target(read, target, tolerance, timeout=15.0):
    deadline = time.monotonic() + timeout
    consecutive = 0
    while True:
        values = read()
        matched = (values is not None and len(values) == len(target)
                   and all(math.isfinite(v) and abs(v - t) <= tolerance
                           for v, t in zip(values, target)))
        consecutive = consecutive + 1 if matched else 0
        if consecutive >= 3:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
