# Black-block center-of-mass experiment — 2026-09-30 20:39:57 UTC

Prompt: “Find the position of center of mass of the black block. Plan the experiment with the given setup, then execute the experiment. An approximate interval is fine.”

The harness reported success after 36 model calls, 44 tool calls, 67 images sent, and 860.1 seconds. It placed the block across the red support, released fully, and checked its stationary balance and end clearance in multiple views. One balance placement was used; no tipping-threshold trials or repeated placements were performed.

Reported result: longitudinal COM projection at world y = 0.170–0.203 m in the balanced experimental pose, expanded from measured support edges at approximately y = 0.180–0.193 m for measurement uncertainty. This interval does not refer to the returned block position. Transverse and vertical COM were not experimentally localized.

The block was returned to the mat and the working arm was verified home. Contact flags remained false during placement probes; the harness relied on effort changes and visual checks. Two inspection orientations were rejected by the planner. A second-arm move-group timeout was logged during homing; later cleanup completed and the harness exited normally, but a verified second-arm return pose is not established by this archive.

Preserved here: full session, outcome, model-view images, console output, exact instruction, run metadata, and an archive-time runtime source patch. SHA256SUMS.json covers every other archive file. This directory is outside harness log retention.
