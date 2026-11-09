#!/bin/bash
#SBATCH --job-name=axion
#SBATCH --output=slurm_%x_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#
# Runs run_topology.sh (which loops run/train_qfi.py + run/train_cfi.py)
# on one node.
#
# IMPORTANT: TOPOLOGY (and LAYERS) are now in-file constants at the top
# of run/train_qfi.py and run/train_cfi.py, not a command-line argument
# (see run_topology.sh). Since this job just calls that script, edit
# TOPOLOGY/LAYERS in those two files BEFORE submitting this job. For
# genuinely parallel per-topology training across multiple SLURM nodes
# (the original point of this script), keep a separate copy of this
# whole code/topo/ directory per topology/node, each with its own
# TOPOLOGY constant set -- since TOPOLOGY now lives in the file, not in
# an argument, one shared directory can only run one topology at a time.
#
# Usage:
#   sbatch --job-name=star slurm_topology.sbatch [qfi_loops] [cfi_loops]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# activate your environment here if needed, e.g.:
# module load python/3.10
# source ~/venvs/axion/bin/activate

bash "$SCRIPT_DIR/run_topology.sh" "$QFI_LOOPS" "$CFI_LOOPS"
