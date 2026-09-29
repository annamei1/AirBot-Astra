# Can2 red-knob task — CPU-isolated retry

Archived episode: `20260929-014117` (UTC, 2026-09-29).

Instruction: “move the red circle knob along the swiggling trajectory to the other end of the green panel”.

Single arm: can2; slow speed; Codex subscription backend using `gpt-6-astra`, medium reasoning.
CPU assignments: can2 runtime 0–1, can3 runtime 2–3, harness 4–9; SAM3 used two threads and passive OpenMP waiting.

Outcome: unsuccessful after 661.3 seconds, 22 model calls and 26 tool calls. Three grasp attempts failed verification. The panel shifted during contact, no sliding motion was attempted, and the knob did not reach the opposite end. The arm returned home with the gripper open.

- `session.jsonl`: timestamped model messages, tool calls and results.
- `outcome.json`: recorded outcome and plan status.
- `images/`: original images saved for this episode.
- `console.txt`: console output including final cleanup.
- `SHA256SUMS.json`: checksums for the original episode files and console output.

Archived outside `harness/logs` so ordinary log retention cannot prune this run. Original paths in the records are retained for provenance.

CPU isolation and completion checks used in this run are committed in `ed9fda3`. The local shared-context RealSense camera fix in `play_sdk.py` was also in use and remains uncommitted.
