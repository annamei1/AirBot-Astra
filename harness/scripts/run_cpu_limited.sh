#!/usr/bin/env bash
# Keep perception away from CPUs reserved for the robot runtime.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${HARNESS_CPUSET:?Set HARNESS_CPUSET to CPUs not assigned to the robot runtime}"
export SAM3_CPU_THREADS="${SAM3_CPU_THREADS:-2}"
export OMP_NUM_THREADS="$SAM3_CPU_THREADS"
export MKL_NUM_THREADS="$SAM3_CPU_THREADS"
export OPENBLAS_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0
if [[ "${1:-}" == "--python" ]]; then
    shift
    exec taskset -c "$HARNESS_CPUSET" .venv/bin/python -u "$@"
fi
exec taskset -c "$HARNESS_CPUSET" .venv/bin/python -u -m harness.scripts.run_pickplace "$@"
