# Can2 red-knob task — CPU-isolated run

Archived episode: `20260929-011046` (UTC, 2026-09-29).

Command: “move the red circle knob along the swiggling trajectory to the other end of the green panel”.

Single arm: can2; slow speed; Codex subscription backend using `gpt-6-astra`, medium reasoning.
CPU assignments: can2 runtime 0–1, can3 runtime 2–3, harness 4–9; SAM3 used two CPU threads with passive OpenMP waiting.

Outcome: unsuccessful after 624.5 seconds, 21 model calls and 28 tool calls. Three grasp attempts did not establish a verified grasp. No sliding motion was attempted. The arm returned home with the gripper open.

- `session.jsonl`: timestamped model messages, tool calls and results.
- `outcome.json`: recorded result and plan status.
- `images/`: original camera/model-view images from the episode.
- `console.txt`: complete console output, including final cleanup.
- `SHA256SUMS.json`: checksums of the archived episode files and console output.

This directory is outside `harness/logs`, so normal run-log retention cannot remove it. Original log paths inside the records are preserved as provenance.

This run used local, uncommitted CPU-isolation, completion-verification, gripper-configuration and camera changes. This archive commit contains run evidence only; it does not commit those implementation changes.
