# Axion-network training/recovery pipeline — user guide

## Background

The physical encoding is Chen et al.'s `U_DM(delta, alpha)` (PRD 110,
115021, Eq. 3.12): a small rotation by angle `delta` about an axis at
angle `alpha` in the qubit's x-y Bloch plane. `alpha` is a genuinely
random, unknown DM phase (constant within one coherence window, different
from window to window).

Estimating `(delta, alpha)` directly (the natural "polar" parametrization)
is a dead end: the Fisher information for `alpha` collapses as
`Q_alpha,alpha ~ 19 * delta^2` -- a coordinate artifact of the map
`(delta, alpha) -> (x, y) = (delta*cos(alpha), delta*sin(alpha))`, whose
Jacobian is singular at `delta -> 0`.

**Fix (used throughout this pipeline):** estimate the Cartesian pair
`(x, y)` instead, at a fixed fiducial point `(x0, y0) = (delta0, 0)`.
Cartesian coordinates have no origin singularity.

## Directory layout

```
topo/
  run/          all compute scripts (training + recovery). Edit the
                constants at the top of a script, then run it with no
                arguments. Writes results into ../data/.
  data/         every checkpoint, loss history, and results JSON --
                nothing else lives here, and nothing here is meant to be
                edited by hand.
  plot/         all plotting scripts. Reads from ../data/, writes PNGs
                into ../figure/. Kept separate from run/ so you can
                re-plot (or combine data with other curves) without
                re-running any (possibly slow) computation.
  figure/       every PNG this pipeline produces.
  run_topology.sh, slurm_topology.sbatch, submit_all.sh
                shell drivers that just call the scripts in run/ in a
                loop / on a cluster -- see their own comments.
```

## Naming convention (IMPORTANT -- read this once)

Every file in `data/` is named with an explicit, ALWAYS-PRESENT
`_{TOPOLOGY}_L{LAYERS}` tag -- e.g. `ckpt_qfi_best_star_L1.pkl`,
`costs_cfi_ring_L2.json`. There is **no special case for `LAYERS=1`**
(earlier versions of this pipeline dropped the `_L1` suffix for
backward compatibility; that's gone now -- everything is uniform, so a
`star, LAYERS=1` run and a `star, LAYERS=2` run are just two files that
differ by the same tag pattern, never a special vs. general case).

Every script in `run/` and `plot/` has `TOPOLOGY` and `LAYERS` (plus
whatever else it needs) as **in-file constants** at the top -- edit
those, then run the script with no command-line arguments and no
environment variables. This is deliberate (matches how this whole
pipeline is meant to be operated): open the file, change the numbers you
want, run it.

## What's in `run/`

### `train_qfi.py` — train theta (state prep)

Trains the entangling state-preparation ansatz (`stargraph`/`ringgraph`/
`fullgraph`, N=5 qubits, selected via the `TOPOLOGY` constant) to
**minimize** `Tr(Q_xy^-1)`, the quantum Cramer-Rao cost for jointly
estimating `(x, y)` at the fixed fiducial point.

```python
TOPOLOGY = 'star'   # star | ring | full
LAYERS = 1
LR = 0.05
STEPS_PER_CALL = 10  # use ~4 for 'full', heavier ansatz
```
```bash
cd topo/run
python3 train_qfi.py
```
Resumes automatically from its checkpoint -- just run it again to do
more steps. Writes `data/ckpt_qfi_{tag}.pkl` (latest),
`data/ckpt_qfi_best_{tag}.pkl` (lowest cost ever seen -- what downstream
scripts use), `data/costs_qfi_{tag}.json` (loss history), where
`tag = {TOPOLOGY}_L{LAYERS}`.

### `train_cfi.py` — train mu (measurement)

With theta **fixed** (loaded from `data/ckpt_qfi_best_{tag}.pkl`), trains
the measurement stage to minimize the classical FIM cost `Tr(C_xy^-1)`
(`C <= Q` always).

The measurement ansatz does not have to match the state-prep ansatz.
`POST_TOPOLOGY`/`POST_LAYERS` default to `None`, meaning "same as
`TOPOLOGY`/`LAYERS`" (mu = inverse of whatever graph theta used, same
layer count -- the common case, and the only case earlier versions of
this pipeline supported). Set them independently to try a
different/cheaper measurement ansatz against a fixed theta:

```python
TOPOLOGY = 'star'   # must match train_qfi.py
LAYERS = 1            # must match train_qfi.py
POST_TOPOLOGY = None  # e.g. 'full' to use a different measurement ansatz
POST_LAYERS = None    # e.g. 1 to use a different measurement layer count
LR = 0.05
STEPS_PER_CALL = 10
```
```bash
python3 train_cfi.py
```
Writes `data/ckpt_cfi_{tag}.pkl`, `data/ckpt_cfi_best_{tag}.pkl` (used
downstream), `data/costs_cfi_{tag}.json`, where `tag` is
`{TOPOLOGY}_L{LAYERS}` when the measurement ansatz matches theta's
(default), or `{TOPOLOGY}_L{LAYERS}_post_{POST_TOPOLOGY}_L{POST_LAYERS}`
when it was set to something different -- so a generalized run never
collides with or overwrites the default one.

### `mass_scan_reach.py` — analytic discovery-reach data

Computes `eig_min(C_xy)` (the worse-estimated of the two Cartesian
directions) for every topology that has a trained
`data/ckpt_cfi_best_{topology}_L{LAYERS}.pkl`, and saves
`data/eig_min_L{LAYERS}.json`. **Compute only** -- see
`plot/plot_mass_scan_reach.py` for the actual reach-curve figure
(`delta_min(N_try) = Z / sqrt(N_try * eig_min)`, standard Cramer-Rao
scaling).

```python
LAYERS = 1
```
```bash
python3 mass_scan_reach.py
```

### `bayes_recover.py` — sequential Bayesian recovery of delta

Ported from a single-qubit sequential-Bayes
template (`likelihood -> bayesian_update -> prior=posterior`, repeated
per measurement), generalized to this project's N=5-qubit circuit and to
scanning `delta` directly (not `(x,y)` jointly, not marginalizing
`alpha`): `y` is fixed at 0 and `x=delta` is scanned (allowed negative),
the same fiducial direction `train_qfi.py`/`train_cfi.py` optimize and
`mass_scan_reach.py`'s `eig_min(C)` comes from. The literal per-shot
sequential update is done via an exact, fast counts-based shortcut (see
the script's docstring for the derivation) rather than a Python loop over
up to `1e5`-`1e12` shots.

**Compute only** -- saves `data/bayes_recover_results_{tag}.json`
(bias/var sweep over N, plus the full posterior at each `N_POST_LIST`
value). See `plot/plot_bayes_recover.py` for the figures.

```python
TOPOLOGY = 'star'
LAYERS = 1
N_TRY = 1e5              # top of the bias/var sweep
N_POST_LIST = [1e2, 1e3, 1e5]
N_SWEEP_POINTS = 20
GRID_N = 500
```
```bash
python3 bayes_recover.py
```
Also prints a diagnostic: the trained `C` at the fiducial point. If
`eig_min(C)` is tiny, no `N_try` will narrow the posterior -- that's a
training issue (check `data/costs_cfi_{tag}.json` is actually
decreasing), not a bug in this script.

### `mass_scan_bayes.py` — Bayesian recovery across masses

Same sequential-Bayes method as `bayes_recover.py`, run at ~10 mass
points `m_a` spanning `mass_scan_reach.py`'s range, each against its own
`g_true = g_KSVZ(m_a)`. **Compute only** -- saves
`data/mass_scan_bayes_results_{tag}.json` (per-mass mean/std of `g_hat`
plus the raw per-run values). See `plot/plot_mass_scan_bayes.py`.

```python
TOPOLOGY = 'star'
LAYERS = 1
N_TRY = 2.1e5    # matches mass_scan_reach.py's "realistic" 1-year-scan curve
N_MA_POINTS = 10
N_RUNS = 5
```
```bash
python3 mass_scan_bayes.py
```
Note: `g_hat` will only land close to `g_true` once `N_TRY` is large
enough for that mass (per the reach curve -- `N_TRY=2.1e5` is BELOW
discovery threshold almost everywhere; try `N_TRY=1e12` to sanity-check
recovery instead).

## What's in `plot/`

All plot scripts read from `../data/` and write PNGs to `../figure/`,
using the same `TOPOLOGY`/`LAYERS` in-file-constant convention.

- **`plot_loss.py`** -- `TOPOLOGY`, `STAGE` (`qfi`|`cfi`), `LAYERS_LIST`.
  Auto-detects which layer counts in `LAYERS_LIST` have a cost-history
  file and draws ALL of them as separate lines on one figure
  (`figure/loss_{stage}_{topology}.png`).
- **`plot_vs_layers.py`** -- `TOPOLOGIES`, `LAYERS_LIST`. Bar chart of
  `Tr(Q^-1)`/`Tr(C^-1)` vs. layer count, all topologies side by side
  (`figure/bound_vs_layers.png`).
- **`plot_mass_scan_reach.py`** -- `LAYERS`, `N_TRY_REALISTIC`,
  `N_TRY_IDEALIZED`. Reads `data/eig_min_L{LAYERS}.json`, draws the
  discovery-reach curve vs. KSVZ/DFSZ (`figure/reach_L{LAYERS}.png`).
- **`plot_bayes_recover.py`** -- `TOPOLOGY`, `LAYERS`. Reads
  `data/bayes_recover_results_{tag}.json`, draws bias/var-vs-N and
  posterior-vs-N figures.
- **`plot_mass_scan_bayes.py`** -- `TOPOLOGY`, `LAYERS`,
  `SHOW_ANALYTIC_CRB`, `SHOW_KSVZ_DFSZ`. Reads
  `data/mass_scan_bayes_results_{tag}.json`, overlays the empirical
  `(m_a, g_hat)` points on the analytic CRB curve
  (`figure/mass_scan_bayes_{tag}.png`).

## Changing the number of ansatz layers

Just set `LAYERS` to whatever you want in `run/train_qfi.py` and
`run/train_cfi.py` (must match between the two) -- every filename
automatically carries the right `_L{LAYERS}` tag, so different layer
counts never collide. Nothing else needs to change (the naming
convention is uniform, see above).

## Suggested workflow

```bash
cd topo/run
# edit train_qfi.py: TOPOLOGY='star', LAYERS=1, bump STEPS_PER_CALL to e.g. 500
python3 train_qfi.py
cd ../plot
# edit plot_loss.py: TOPOLOGY='star', STAGE='qfi'
python3 plot_loss.py   # check it has plateaued; re-run train_qfi.py if not

cd ../run
# edit train_cfi.py: TOPOLOGY='star', LAYERS=1
python3 train_cfi.py
cd ../plot
# edit plot_loss.py: STAGE='cfi'
python3 plot_loss.py   # check convergence again

# repeat train_qfi.py/train_cfi.py for TOPOLOGY='ring', 'full'

cd ../run
# edit bayes_recover.py: TOPOLOGY='star', N_TRY=1e12, then:
python3 bayes_recover.py     # sanity check
cd ../plot && python3 plot_bayes_recover.py

# once star, ring, full all have trained data/ckpt_cfi_best_*.pkl:
cd ../run && python3 mass_scan_reach.py
cd ../plot && python3 plot_mass_scan_reach.py
```

To avoid re-running `train_qfi.py`/`train_cfi.py` by hand many times,
`run_topology.sh` loops both for you (edit `TOPOLOGY`/`LAYERS` in the two
scripts first, then run with no topology argument):
```bash
cd topo
./run_topology.sh 200 200   # <qfi_loops> <cfi_loops>
```
For a SLURM cluster, see `slurm_topology.sbatch` and `submit_all.sh` --
since `TOPOLOGY` is now a file constant rather than a runtime argument,
running multiple topologies in parallel across nodes needs a separate
directory copy per topology (see `submit_all.sh`'s comments).
