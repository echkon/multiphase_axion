#!/bin/bash
# NOTE: TOPOLOGY is now an in-file constant in run/train_qfi.py and
# run/train_cfi.py (not a command-line argument), so a single code/topo/
# directory can only run one topology at a time. To submit all 3
# topologies as separate SLURM jobs (the original point of this script),
# make 3 copies of the whole code/topo/ directory (e.g. topo_star/,
# topo_ring/, topo_full/), set TOPOLOGY accordingly in each copy's
# run/train_qfi.py + run/train_cfi.py, then submit each copy's
# slurm_topology.sbatch separately:
#
#   sbatch --job-name=star topo_star/slurm_topology.sbatch
#   sbatch --job-name=ring topo_ring/slurm_topology.sbatch
#   sbatch --job-name=full topo_full/slurm_topology.sbatch
#
# With 2 compute nodes, SLURM runs 2 concurrently and starts the 3rd as
# soon as one finishes.
squeue -u "$USER"
