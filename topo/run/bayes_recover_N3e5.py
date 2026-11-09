import sys, os, pickle, json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'axion_reach_results'))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
import numpy as np
import onecircuit
from onecircuit import *
from onecircuit.metric import _safe_probs, cfim, sld_qfim
from axion_coupling import g_agg_from_delta
from bayes_core import sequential_recover as _core_sequential_recover, g_stat_properties as _core_g_stat_properties

"""
Edit the constants below, then just run:
    python3 bayes_recover.py

COMPUTE + SAVE ONLY -- no plotting here. plot/plot_bayes_recover.py reads
the JSON this saves and draws the combined figure (kept separate so plots
can be tweaked without re-running the recovery).

The actual recovery algorithm lives in bayes_core.py (shared with
mass_scan_bayes.py, not duplicated here) -- a LINE-FOR-LINE PORT of the
user's own bayesian_inference.py, generalized from a single-qubit binary
outcome (k in {0,1}) to this project's N=5-qubit circuit (k in
{0, ..., 31}), and from "phase" to "delta". This file just builds the
circuit-specific inputs (delta_grid, p_grid, p_true) and calls it:

  - `delta_grid` is defined ONCE, as a FIXED constant range -- exactly
    like bayesian_inference.py's `phase = np.linspace(-np.pi, np.pi, 500)`,
    reused UNCHANGED for every num_k. It does NOT depend on N_try, and its
    width is NOT derived from eig_min(C)/Fisher-info/any N value -- it's
    just a plain constant (GRID_HALF_WIDTH_FACTOR * delta_true), same as
    your own fixed [-pi, pi]. GRID_N is kept large so the peak still has
    enough points to look like a smooth bell curve at large N_try.
  - For EACH N_try value (in the sweep AND in N_POST_LIST), a FRESH,
    INDEPENDENT batch of N_try outcomes is drawn -- exactly like
    bayesian_inference.py's `for num_k in num_k_list: k_values =
    np.random.choice(...)` drawing new k_values every iteration. There is
    no shared/carried-over realization between different N values.
  - `stat_properties` normalizes with `np.sum(f)` (not np.trapz), exactly
    matching bayesian_inference.py's `mean = x @ f / np.sum(f)`.

SEQUENTIAL UPDATE, DONE EFFICIENTLY: doing exactly what
bayesian_inference.py's `for k in k_values: posterior = bayesian_update(...);
prior = posterior` loop does, one measurement at a time, is mathematically
IDENTICAL to computing it in one shot via counts, because likelihoods of
independent shots multiply in any order:

    prior(delta) * prod_{shots} P(k_shot | delta)
        = prior(delta) * exp( sum_k counts[k] * log P(k | delta) )

So below, `counts = bincount(outcomes)` + one matrix-vector product
against the (fixed, precomputed-ONCE) likelihood table reproduces the
exact same final posterior as N_try individual bayesian_update() calls,
but stays fast even at N_try = 1e5-1e12. Because delta_grid is now FIXED
(not recomputed per N), the likelihood table p_grid is computed ONCE for
the whole script, not once per N -- much faster than before.

Saves everything plot_bayes_recover.py needs: the N sweep (bias/var vs N,
in BOTH delta-space and g-space), and the full posterior (delta_grid +
density) at each N in N_POST_LIST.
"""

# ---- edit these before running ----
TOPOLOGY = 'full'                    # star | ring | full
LAYERS = 3                           # must match the LAYERS used to train the checkpoint
POST_TOPOLOGY = None                 # measurement ansatz's topology per train_cfi.py; None = same as TOPOLOGY
POST_LAYERS = 3                      # measurement ansatz's layer count per train_cfi.py; None = same as LAYERS
N_TRY = 3e5                          # target/max exposure (top end of the sweep)
N_POST_LIST = [1e5, 2e5, 3e5]         # exposures for which the full posterior is saved
N_SWEEP_POINTS = 20                  # number of N values swept (log-spaced, 1 .. N_TRY)
GRID_N = 5000                        # number of candidate delta values (bumped up from 500 --
                                      # needed to resolve N_POST_LIST's widest-to-narrowest span)
GRID_HALF_WIDTH_SIGMAS = 8           # grid half-width = SIGMAS * sigma_theory(min(N_POST_LIST))
# written as data/bayes_recover_results_{tag}_N{N_TRY}.json -- N_TRY is baked
# into the filename so different exposures don't overwrite each other
# ------------------------------------
TOPOLOGY = os.environ.get('TOPOLOGY', TOPOLOGY)  # only used by run_topology.sh, see train_qfi.py's docstring
LAYERS = int(os.environ.get('LAYERS', LAYERS))
POST_TOPOLOGY = POST_TOPOLOGY or TOPOLOGY
POST_LAYERS = LAYERS if POST_LAYERS is None else POST_LAYERS

topo = TOPOLOGY
# same ALWAYS-both naming as train_cfi.py: {topo}_L{layers}_post_{post_topology}_L{post_layers}
tag = f'{topo}_L{LAYERS}_post_{POST_TOPOLOGY}_L{POST_LAYERS}'


def _fmt_n(n):
    """1e5 -> '1e05', 2.1e5 -> '2.1e05' -- compact, filename-safe."""
    return f'{n:.4g}'.replace('+', '').replace('.', 'p')


# N_TRY is baked into the output filename so different exposures don't
# overwrite each other's saved results -- see OUT_JSON below.
run_tag = f'{tag}_N{_fmt_n(N_TRY)}'

with open(os.path.join(DATA_DIR, f'ckpt_cfi_best_{tag}.pkl'), 'rb') as f:
    circuit0 = pickle.load(f)
prep_stage = circuit0[0]
post_stage = circuit0[2]

N = 5
m_a = 1.0  # ueV, matches the hardcoded delta_true/g_true below
delta_true = 6.498913663721062e-06  # delta(g_KSVZ, m_a=1ueV)
g_true = 3.918992394086675e-16


def probs_for_delta(d):
    """P(k | delta), the full likelihood vector over computational-basis
    outcomes k -- the generalization of your likelihood(phase, k)."""
    c = [prep_stage, [sensing_xy, N, 1.0, [d, 0.0]], post_stage]
    p = np.clip(_safe_probs(c), 0, None)
    return p / p.sum()


# ============================================================
# DIAGNOSTIC ONLY: trained C (classical) and Q (quantum) at the fiducial
# point -- NOT used anywhere in the recovery math or the grid below,
# only to draw the CB/nu and QB/nu reference-scaling lines in the
# variance panel.
# ============================================================
_C_check = np.real(cfim([prep_stage, [sensing_xy, N, 1.0, [delta_true, 0.0]], post_stage]))
_eigs = np.linalg.eigvalsh(_C_check)
eig_min = max(_eigs.min(), 1e-12)
print(f'--- diagnostic: trained C at fiducial point ({run_tag}) ---')
print(f'C =\n{_C_check}')
print(f'eigenvalues(C) = {_eigs}')
if _eigs.min() < 1e-3:
    print('WARNING: eig_min(C) is tiny.')

_Q_check = np.real(sld_qfim([prep_stage, [sensing_xy, N, 1.0, [delta_true, 0.0]], prep_stage]))
_eigs_q = np.linalg.eigvalsh(_Q_check)
eig_min_q = max(_eigs_q.min(), 1e-12)
print(f'eigenvalues(Q) = {_eigs_q}')
print()

# ============================================================
# ONE fixed delta_grid for the ENTIRE script -- exactly like your
# `phase = np.linspace(-np.pi, np.pi, 500)`: a single range and point
# count, chosen ONCE and reused unchanged for every N_try.
#
# WIDTH: delta_true (6.5e-6) is NOT the right scale to base this on --
# your trained circuit's actual sensitivity is eig_min(C) ~= 60 (just
# measured above), giving a theoretical CRB sigma of
# 1/sqrt(N*eig_min) ~= 0.013 at N=100 and ~= 2.8e-4 even at N=2.1e5 --
# BOTH thousands of times WIDER than delta_true itself. A window sized
# off delta_true (like the earlier [0, 2*delta_true] attempt) is ~1000x
# too narrow: the likelihood barely curves at all across such a tiny
# slice, so the "posterior" is just a near-uniform box, giving a flat
# line instead of a peak, and a "variance" that's really just the box's
# own (b-a)^2/12 -- not measuring anything about N.
#
# Fix: size the ONE fixed half-width from the theoretical CRB sigma AT
# THE SMALLEST N in N_POST_LIST (the widest posterior you're actually
# asking to plot), generously padded -- computed from eig_min(C), which
# the script already measures from the real trained checkpoint above (no
# more guessing). Still a single constant, computed once.
# ============================================================
_sigma_ref = 1.0 / np.sqrt(min(N_POST_LIST) * eig_min)
GRID_HALF_WIDTH = GRID_HALF_WIDTH_SIGMAS * _sigma_ref
delta_grid = np.linspace(delta_true - GRID_HALF_WIDTH, delta_true + GRID_HALF_WIDTH, GRID_N)
print(f'delta_grid: {delta_true - GRID_HALF_WIDTH:.3e} .. {delta_true + GRID_HALF_WIDTH:.3e} '
      f'({GRID_N} pts, dx={2 * GRID_HALF_WIDTH / GRID_N:.3e}), '
      f'sigma_theory(N={min(N_POST_LIST):.0e})={_sigma_ref:.3e}, '
      f'sigma_theory(N_TRY)={1.0 / np.sqrt(N_TRY * eig_min):.3e}')
print()

# likelihood table computed ONCE on the fixed grid (not once per N,
# because the grid no longer changes with N) -- GRID_N circuit evals total
p_grid = np.array([probs_for_delta(d) for d in delta_grid])  # (GRID_N, n_outcomes)
p_true = probs_for_delta(delta_true)


def sequential_recover(N_try):
    """Thin wrapper around bayes_core.sequential_recover(): supplies this
    script's module-level delta_grid/p_grid/p_true/delta_true (all fixed,
    computed once above) -- see bayes_core.py for the actual algorithm
    (shared with mass_scan_bayes.py, not duplicated here)."""
    return _core_sequential_recover(delta_grid, p_grid, p_true, N_try, delta_true)


def g_stat_properties(delta_grid, posterior, true_g):
    """Thin wrapper around bayes_core.g_stat_properties()."""
    return _core_g_stat_properties(delta_grid, posterior, g_agg_from_delta, m_a, true_g)


if __name__ == '__main__':
    # ============================================================
    # 1. BIAS and VAR vs. N -- both delta-space AND g-space. Each N_try
    # is an independent fresh recovery (fresh draw, fresh prior), exactly
    # like your num_k_list loop.
    # ============================================================
    N_list = np.unique(np.round(np.logspace(0, np.log10(N_TRY), N_SWEEP_POINTS)).astype(np.int64))
    bias_list, var_list = [], []
    g_bias_list, g_var_list = [], []

    print(f'--- {run_tag}: sequential Bayes recovery, N = 1 .. {N_TRY:.0e} ({len(N_list)} points) ---')
    for num_k in N_list:
        post_k, mean, bias, var, std = sequential_recover(int(num_k))
        bias_list.append(bias)
        var_list.append(var)
        g_mean, g_bias, g_var = g_stat_properties(delta_grid, post_k, g_true)
        g_bias_list.append(g_bias)
        g_var_list.append(g_var)
        print(f'  N={num_k:.3e}: mean={mean:.4e}  bias={bias:.3e}  var={var:.3e}  std={std:.3e}  '
              f'| g_bias={g_bias:.3e}  g_var={g_var:.3e}')

    # ============================================================
    # 2. full posterior at each N_POST_LIST value -- same fixed
    # delta_grid as everything else, so all curves share one x-axis.
    # ============================================================
    posteriors = []
    print(f'\n--- {run_tag}: posterior at each N_POST_LIST value ---')
    for N_try in N_POST_LIST:
        post, mean, bias, var, std = sequential_recover(int(N_try))
        g_mean, g_bias, g_var = g_stat_properties(delta_grid, post, g_true)
        print(f'  N={N_try:.0e}: mean={mean:.3e}  std={std:.3e}  | g_mean={g_mean:.3e}  g_bias={g_bias:.3e}')
        posteriors.append({
            'N_try': N_try,
            'delta_grid': delta_grid.tolist(),
            'posterior': post.tolist(),
            'mean': float(mean), 'bias': float(bias), 'var': float(var), 'std': float(std),
            'g_mean': float(g_mean), 'g_bias': float(g_bias), 'g_var': float(g_var),
        })

    results = {
        'topology': topo,
        'LAYERS': LAYERS,
        'POST_TOPOLOGY': POST_TOPOLOGY,
        'POST_LAYERS': POST_LAYERS,
        'N_TRY': N_TRY,
        'm_a': m_a,
        'delta_true': delta_true,
        'g_true': g_true,
        'eig_min': float(eig_min),        # classical FI (worst-case eigen-direction) -> CB/nu
        'eig_min_q': float(eig_min_q),     # quantum FI (worst-case eigen-direction) -> QB/nu
        'N_list': N_list.tolist(),
        'bias_list': [float(b) for b in bias_list],
        'var_list': [float(v) for v in var_list],
        'g_bias_list': [float(b) for b in g_bias_list],
        'g_var_list': [float(v) for v in g_var_list],
        'posteriors': posteriors,
    }
    out_path = os.path.join(DATA_DIR, f'bayes_recover_results_{run_tag}.json')
    json.dump(results, open(out_path, 'w'), indent=2)
    print(f'\nsaved {out_path}')
