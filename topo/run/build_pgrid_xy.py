import sys, os, pickle, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'axion_reach_results'))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
import numpy as np
import onecircuit
from onecircuit import *
from onecircuit.metric import _safe_probs, cfim

TOPOLOGY = 'full'; LAYERS = 3; POST_TOPOLOGY = 'full'; POST_LAYERS = 3
tag = f'{TOPOLOGY}_L{LAYERS}_post_{POST_TOPOLOGY}_L{POST_LAYERS}'
N = 5
m_a = 1.0
delta_true = 6.498913663721062e-06

with open(os.path.join(DATA_DIR, f'ckpt_cfi_best_{tag}.pkl'), 'rb') as f:
    circuit0 = pickle.load(f)
prep_stage = circuit0[0]
post_stage = circuit0[2]


def probs_for_xy(x, y):
    c = [prep_stage, [sensing_xy, N, 1.0, [x, y]], post_stage]
    p = np.clip(_safe_probs(c), 0, None)
    return p / p.sum()


# ---- fixed scale: eig_min(C) at the fiducial (delta_true, 0) point,
# reused from the earlier bayes_recover.py diagnostic (C is close to
# isotropic: eigs = [60.199, 61.296], off-diag ~ -0.30 -- a shared
# symmetric grid for x and y is a reasonable approximation). ----
eig_min = 60.19909415

# ---- 1D axis grid, shared by x and y: fine "core" resolving the CRB
# scale down to N=1e5 (8-sigma half-width), plus a log-spaced "wing" out
# to R_MAX=1.0 rad to cover the N=1..1e5 broad-posterior tail (same
# core+wing idea used for the earlier 1D delta_grid, now applied per-axis
# so the 2D (x,y) grid can resolve both the tiny delta_true~6.5e-6 scale
# and the O(1) prior range needed at low N). ----
GRID_HALF_WIDTH_SIGMAS = 8
sigma_ref = 1.0 / np.sqrt(1e5 * eig_min)
core_half_width = GRID_HALF_WIDTH_SIGMAS * sigma_ref
CORE_N = 61
WING_N = 20
R_MAX = 1.0

core = np.linspace(-core_half_width, core_half_width, CORE_N)
wing_pos = np.logspace(np.log10(core_half_width * 1.05), np.log10(R_MAX), WING_N)
axis_grid = np.unique(np.concatenate([-wing_pos[::-1], core, wing_pos]))
NAX = len(axis_grid)
print(f'axis_grid: {NAX} pts, core_half_width={core_half_width:.4e}, range=[{axis_grid[0]:.4e},{axis_grid[-1]:.4e}]')

CKPT = os.path.join(SCRIPT_DIR, 'pgrid_checkpoint_xy.npz')

# probe n_outcomes
p0 = probs_for_xy(delta_true, 0.0)
n_out = len(p0)

if os.path.exists(CKPT):
    d = np.load(CKPT)
    p_grid = d['p_grid']
    done = d['done']
    assert p_grid.shape == (NAX, NAX, n_out), f'{p_grid.shape} vs {(NAX,NAX,n_out)}'
else:
    p_grid = np.zeros((NAX, NAX, n_out))
    done = np.zeros((NAX, NAX), dtype=bool)

total = NAX * NAX
n_done0 = int(done.sum())
print(f'resuming: {n_done0}/{total} done')

t0 = time.time()
BUDGET_S = 33.0
idxs = np.argwhere(~done)
n_this_call = 0
for (i, j) in idxs:
    x = axis_grid[i]
    y = axis_grid[j]
    p_grid[i, j] = probs_for_xy(x, y)
    done[i, j] = True
    n_this_call += 1
    if time.time() - t0 > BUDGET_S:
        break

np.savez(CKPT, p_grid=p_grid, done=done, axis_grid=axis_grid,
         core_half_width=core_half_width, eig_min=eig_min)
n_done1 = int(done.sum())
print(f'this call: {n_this_call} evals in {time.time()-t0:.1f}s; total {n_done1}/{total} '
      f'({100*n_done1/total:.1f}%)')
if n_done1 == total:
    print('DONE building 2D (x,y) likelihood grid.')
