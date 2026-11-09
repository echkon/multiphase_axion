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
from onecircuit.metric import _safe_probs

TOPOLOGY = 'full'; LAYERS = 3; POST_TOPOLOGY = 'full'; POST_LAYERS = 3
tag = f'{TOPOLOGY}_L{LAYERS}_post_{POST_TOPOLOGY}_L{POST_LAYERS}'
N = 5
m_a = 1.0
delta_true = 6.498913663721062e-06
eig_min = 60.19909415

# ---- same exposure scale established yesterday ----
N_MONTH = int(2.63e10)
N_6MO = int(1.578e11)
N_YEAR = int(3.156e11)
N_TRY = N_YEAR

with open(os.path.join(DATA_DIR, f'ckpt_cfi_best_{tag}.pkl'), 'rb') as f:
    circuit0 = pickle.load(f)
prep_stage = circuit0[0]
post_stage = circuit0[2]


def probs_for_xy(x, y):
    c = [prep_stage, [sensing_xy, N, 1.0, [x, y]], post_stage]
    p = np.clip(_safe_probs(c), 0, None)
    return p / p.sum()


# ---- r-grid: physics only depends on r=sqrt(x^2+y^2), theta=atan2(y,x)
# (sensing_xy(x,y) is the SAME rotation as sensing_dm(alpha,[delta]) with
# r<->delta, theta<->alpha -- see circuit.py docstrings). This lets us use
# an efficient POLAR (r,theta) grid instead of an infeasible dense
# Cartesian N_x*N_y grid: core resolves the CRB scale down to N_MONTH
# (needed since N sweeps up to N_YEAR, finer than N_MONTH), wing extends
# out (log-spaced) to R_MAX=1.0 to cover the N=1 broad-posterior tail. ----
GRID_HALF_WIDTH_SIGMAS = 8
sigma_ref = 1.0 / np.sqrt(N_MONTH * eig_min)
core_half_width = GRID_HALF_WIDTH_SIGMAS * sigma_ref  # ~6.36e-6, matches yesterday's 1D core
CORE_N = 600
WING_N = 200
R_MAX = 1.0

core = np.linspace(max(delta_true - core_half_width, 1e-10), delta_true + core_half_width, CORE_N)
wing = np.logspace(np.log10(core_half_width * 1.05 + delta_true), np.log10(R_MAX), WING_N)
r_grid = np.unique(np.concatenate([core, wing]))
NR = len(r_grid)

NTHETA = 16
theta_grid = np.linspace(0, 2 * np.pi, NTHETA, endpoint=False)

print(f'r_grid: {NR} pts, range=[{r_grid[0]:.4e},{r_grid[-1]:.4e}], core_half_width={core_half_width:.4e}')
print(f'theta_grid: {NTHETA} pts')
print(f'total grid points: {NR * NTHETA}')

CKPT = os.path.join(SCRIPT_DIR, 'pgrid_checkpoint_polar.npz')

p0 = probs_for_xy(delta_true, 0.0)
n_out = len(p0)

if os.path.exists(CKPT):
    d = np.load(CKPT)
    p_grid = d['p_grid']
    done = d['done']
    assert p_grid.shape == (NR, NTHETA, n_out), f'{p_grid.shape} vs {(NR, NTHETA, n_out)}'
else:
    p_grid = np.zeros((NR, NTHETA, n_out))
    done = np.zeros((NR, NTHETA), dtype=bool)

total = NR * NTHETA
n_done0 = int(done.sum())
print(f'resuming: {n_done0}/{total} done')

t0 = time.time()
BUDGET_S = 33.0
idxs = np.argwhere(~done)
n_this_call = 0
for (i, j) in idxs:
    r = r_grid[i]
    th = theta_grid[j]
    x = r * np.cos(th)
    y = r * np.sin(th)
    p_grid[i, j] = probs_for_xy(x, y)
    done[i, j] = True
    n_this_call += 1
    if time.time() - t0 > BUDGET_S:
        break

np.savez(CKPT, p_grid=p_grid, done=done, r_grid=r_grid, theta_grid=theta_grid,
         core_half_width=core_half_width, eig_min=eig_min,
         N_MONTH=N_MONTH, N_6MO=N_6MO, N_YEAR=N_YEAR, delta_true=delta_true)
n_done1 = int(done.sum())
print(f'this call: {n_this_call} evals in {time.time()-t0:.1f}s; total {n_done1}/{total} '
      f'({100*n_done1/total:.1f}%)')
if n_done1 == total:
    print('DONE building polar (r,theta) likelihood grid.')
