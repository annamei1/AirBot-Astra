# Taped-box T stacking — interrupted run

Episode `20260930-191157` (UTC, 2026-09-30).

The exact task prompt is in `instruction.txt`: stand one black box vertically and place the other horizontally on top, preserving parallel taped edges to form a fattened letter T.

**Outcome: interrupted by the operator before stacking was completed.** The first upright placement tipped. On the second attempt, the harness reported that the wrist view confirmed the base stood independently after release and horizontal withdrawal. It was locating the upright top and second box when the operator requested a stop. No completed T or final taped-edge alignment was verified.

The harness stopped through its immediate-home interrupt handler (exit 130). Independent SDK feedback confirmed the right arm returned to the configured home pose within one degree on every joint, state IDLE, with the gripper open at 69.4 mm.

Right arm: can3, port 50052; slow speed. Codex subscription backend: GPT-6 Astra, high reasoning. Cameras: 640×480 RGB/depth at 5 FPS. Harness CPUs 4–9, arm runtimes on 0–1 and 2–3.

- `session.jsonl`: model decisions and complete tool feedback.
- `console.txt`: captured console output, including interruption and homing.
- `images/`: saved camera/model views.
- `outcome.json`: harness outcome snapshot; consult interruption notes above for final operator action.
- `run-metadata.json`: launch settings and independently verified final arm state.
- `runtime-source.patch`: tracked runtime changes relative to the recorded base commit.
- `SHA256SUMS.json`: SHA-256 hashes of archived files other than the checksum manifest.

Preserved outside `harness/logs` so log retention cannot prune this run. Local credentials and environment files are excluded.
