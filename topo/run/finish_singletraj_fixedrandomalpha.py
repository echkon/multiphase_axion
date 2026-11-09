import sys, os, pickle, json, time
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
from axion_coupling import g_agg_from_delta

TOPOLOGY = 'full'; LAYERS = 3; POST_TOPOLOGY = 'full'; POST_LAYERS = 3
tag = f'{TOPOLOGY}_L{LAYERS}_post_{POST_TOPOLOGY}_L{POST_LAYERS}'
N = 5
m_a = 1.0
delta_true = 6.498913663721062e-06
g_true = 3.918992394086675e-16

with open(os.path.join(DATA_DIR, f'ckpt_cfi_best_{tag}.pkl'), 'rb') as f:
    circuit0 = pickle.load(f)
prep_stage = circuit0[0]
post_stage = circuit0[2]


def probs_for_xy(x, y):
    c = [prep_stage, [sensing_xy, N, 1.0, [x, y]], post_stage]
    p = np.clip(_safe_probs(c), 0, None)
    return p / p.sum()


grid = np.load(os.path.join(SCRIPT_DIR, 'pgrid_checkpoint_polar.npz'))
p_grid = grid['p_grid']
r_grid = grid['r_grid']
theta_grid = grid['theta_grid']
N_MONTH = int(grid['N_MONTH']); N_6MO = int(grid['N_6MO']); N_YEAR = int(grid['N_YEAR'])
NR, NTHETA = len(r_grid), len(theta_grid)
n_out = p_grid.shape[-1]
print(f'loaded polar grid: NR={NR}, NTHETA={NTHETA}')

log_p_grid = np.log(p_grid + 1e-300)
w_r = np.gradient(r_grid)
w_theta = 2 * np.pi / NTHETA
COS_T = np.cos(theta_grid)
SIN_T = np.sin(theta_grid)
R_COL = r_grid[:, None]


def posterior_polar(counts):
    loglik = log_p_grid @ counts
    loglik -= loglik.max()
    L = np.exp(loglik)
    g_unnorm = L * R_COL
    Z = np.sum(g_unnorm * w_r[:, None]) * w_theta
    return g_unnorm / Z


def moments_polar(g):
    hat_x = np.sum(g * R_COL * COS_T[None, :] * w_r[:, None]) * w_theta
    hat_y = np.sum(g * R_COL * SIN_T[None, :] * w_r[:, None]) * w_theta
    marginal_r = np.sum(g, axis=1) * w_theta
    return hat_x, hat_y, marginal_r


rng = np.random.default_rng()

# ---- ONE random alpha for the ENTIRE trajectory (drawn once, hidden from
# the recovery algorithm, only used to generate the synthetic data) ----
alpha0 = rng.uniform(0, 2 * np.pi)
x0 = delta_true * np.cos(alpha0)
y0 = delta_true * np.sin(alpha0)
p_true = probs_for_xy(x0, y0)
print(f'alpha0 = {alpha0:.4f} rad (fixed for the whole trajectory), (x0,y0)=({x0:.4e},{y0:.4e})')

N_SWEEP_POINTS = 100
n_log = np.unique(np.round(np.logspace(0, np.log10(N_YEAR), N_SWEEP_POINTS)).astype(np.int64))
checkpoints = np.unique(np.concatenate([n_log, [N_MONTH, N_6MO, N_YEAR]]))
checkpoints.sort()
print(f'{len(checkpoints)} checkpoints, n=1..{N_YEAR:.3e}')

# ---- single continuous trajectory: independent multinomial increments
# between consecutive checkpoints, ALL drawn from the SAME p_true (fixed
# alpha0) -- exact for one running total. ----
prev_n = 0
cum_counts = np.zeros(n_out, dtype=np.int64)
running = {}
t0 = time.time()
for n in checkpoints:
    delta_n = int(n - prev_n)
    if delta_n > 0:
        cum_counts = cum_counts + rng.multinomial(delta_n, p_true)
    prev_n = int(n)
    running[int(n)] = cum_counts.copy()
print(f'trajectory built in {time.time()-t0:.1f}s')

n_list, bias_delta_list, bias_g_list, hat_delta_list, hat_g_list = [], [], [], [], []
snapshots = []
t0 = time.time()
for n in checkpoints:
    g = posterior_polar(running[int(n)])
    hat_x, hat_y, marginal_r = moments_polar(g)
    hat_delta = float(np.sqrt(hat_x ** 2 + hat_y ** 2))
    hat_g = float(g_agg_from_delta(hat_delta, m_a))
    bias_delta = hat_delta - delta_true
    bias_g = hat_g - g_true
    n_list.append(int(n)); bias_delta_list.append(bias_delta); bias_g_list.append(bias_g)
    hat_delta_list.append(hat_delta); hat_g_list.append(hat_g)
    if int(n) in (N_MONTH, N_6MO, N_YEAR):
        print(f'  [checkpoint] n={n:.4e}  delta_hat={hat_delta:.4e}  '
              f'bias_delta={bias_delta:.3e}  bias_g={bias_g:.3e}')
        snapshots.append({
            'N_try': int(n), 'hat_x': float(hat_x), 'hat_y': float(hat_y),
            'hat_delta': hat_delta, 'hat_g': hat_g, 'marginal_r': marginal_r.tolist(),
        })
print(f'{len(checkpoints)} checkpoints evaluated in {time.time()-t0:.1f}s')

results = {
    'topology': TOPOLOGY, 'LAYERS': LAYERS, 'POST_TOPOLOGY': POST_TOPOLOGY, 'POST_LAYERS': POST_LAYERS,
    'N_YEAR': N_YEAR, 'N_MONTH': N_MONTH, 'N_6MO': N_6MO,
    'm_a': m_a, 'delta_true': delta_true, 'g_true': g_true, 'alpha0': float(alpha0),
    'x0': float(x0), 'y0': float(y0),
    'r_grid': r_grid.tolist(),
    'N_list': n_list,
    'bias_delta_list': bias_delta_list, 'bias_g_list': bias_g_list,
    'hat_delta_list': hat_delta_list, 'hat_g_list': hat_g_list,
    'snapshots': snapshots,
    'note': 'ONE continuous sequential trajectory, n=1..N_YEAR, cumulative counts from '
            'independent multinomial increments between checkpoints (exact for a single '
            'running experiment). A SINGLE random DM phase alpha0 is drawn ONCE (hidden from '
            'the recovery, only used to generate data) and held fixed for the whole '
            'trajectory -- the recovery never assumes alpha0=0, only that it is unknown '
            'but constant, exactly as in the manuscript\'s physical setup.',
}
out_path = os.path.join(DATA_DIR, f'bayes_recover_results_{tag}_singletraj_fixedrandomalpha.json')
json.dump(results, open(out_path, 'w'), indent=2)
print(f'saved {out_path}')
