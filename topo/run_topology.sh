#!/bin/bash
# ON/OFF SWITCHBOARD for run/train_qfi.py + run/train_cfi.py.
#
# Turn each (stage, topology) combination on/off individually below --
# only the ones set to true actually run, ONCE each (however many steps
# that is depends entirely on STEPS_PER_CALL inside train_qfi.py /
# train_cfi.py -- edit that if you want more/fewer steps, not this
# script). Jobs that ARE turned on run in parallel (one background
# process per job); jobs that are off are skipped entirely, not even
# launched.
#
# Re-run this script again any time to keep training further -- both
# scripts resume automatically from their last checkpoint.
#
# HOW TOPOLOGY GETS INTO train_qfi.py/train_cfi.py: those scripts
# normally take TOPOLOGY/LAYERS as in-file constants (edit-then-run, no
# arguments) for a single manual run. This script instead sets the
# TOPOLOGY/LAYERS ENVIRONMENT VARIABLES per background process (the
# scripts fall back to their in-file defaults if unset -- see their
# docstrings), since one file can't hold several different TOPOLOGY
# values for several concurrent processes. You never type an environment
# variable yourself -- it's internal to how this script launches jobs.
#
# Usage:
#   ./run_topology.sh
set -euo pipefail

# ---- turn each job on/off here ----
qfi_star=false
qfi_ring=false
qfi_full=false
cfi_star=false
cfi_ring=false
cfi_full=true
# ------------------------------------

LAYERS=1   # applies to every job turned on above; re-run with a different
           # LAYERS (and the switches you want) to add another layer count

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/run"

run_job() {
    local script="$1" topo="$2"
    ( export TOPOLOGY="$topo" LAYERS="$LAYERS"; python3 "$script" ) &
}

echo "=== QFI (theta) jobs ==="
pids=()
for topo in star ring full; do
    var="qfi_$topo"
    if [ "${!var}" = "true" ]; then
        echo "  [on]  qfi_$topo"
        run_job train_qfi.py "$topo"
        pids+=($!)
    else
        echo "  [off] qfi_$topo"
    fi
done
for pid in "${pids[@]:-}"; do [ -n "$pid" ] && wait "$pid"; done

echo "=== CFI (mu) jobs ==="
pids=()
for topo in star ring full; do
    var="cfi_$topo"
    if [ "${!var}" = "true" ]; then
        echo "  [on]  cfi_$topo"
        run_job train_cfi.py "$topo"
        pids+=($!)
    else
        echo "  [off] cfi_$topo"
    fi
done
for pid in "${pids[@]:-}"; do [ -n "$pid" ] && wait "$pid"; done

echo "=== done. ==="
echo "Edit plot/plot_loss.py (TOPOLOGY/STAGE/LAYERS_LIST), then:"
echo "    cd $SCRIPT_DIR/plot && python3 plot_loss.py"
