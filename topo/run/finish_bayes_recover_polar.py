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


# ---- load the precomputed polar (r,theta) likelihood grid ----
grid = np.load(os.path.join(SCRIPT_DIR, 'pgrid_checkpoint_polar.npz'))
p_grid = grid['p_grid']          # (NR, NTHETA, n_outcomes)
r_grid = grid['r_grid']
theta_grid = grid['theta_grid']
N_MONTH = int(grid['N_MONTH']); N_6MO = int(grid['N_6MO']); N_YEAR = int(grid['N_YEAR'])
assert grid['done'].all(), 'grid not fully built'
NR, NTHETA = len(r_grid), len(theta_grid)
print(f'loaded polar grid: NR={NR}, NTHETA={NTHETA}, N_MONTH={N_MONTH:.3e}, N_6MO={N_6MO:.3e}, N_YEAR={N_YEAR:.3e}')

log_p_grid = np.log(p_grid + 1e-300)  # (NR, NTHETA, n_outcomes)

# trapezoid weights for the nonuniform r_grid
w_r = np.gradient(r_grid)  # good approximation to trapz weights for a dense, smooth grid
# theta is uniform & periodic -> rectangle rule weight is exact
w_theta = 2 * np.pi / NTHETA

COS_T = np.cos(theta_grid)  # (NTHETA,)
SIN_T = np.sin(theta_grid)
R_COL = r_grid[:, None]     # (NR,1)


def posterior_polar(counts):
    loglik = log_p_grid @ counts   # (NR, NTHETA)
    loglik -= loglik.max()
    L = np.exp(loglik)
    g_unnorm = L * R_COL           # Jacobian r, broadcasts over theta
    Z = np.sum(g_unnorm * w_r[:, None]) * w_theta
    g = g_unnorm / Z
    return g  # (NR, NTHETA), integrates (sum*w_r*w_theta) to 1


def moments_polar(g):
    hat_x = np.sum(g * R_COL * COS_T[None, :] * w_r[:, None]) * w_theta
    hat_y = np.sum(g * R_COL * SIN_T[None, :] * w_r[:, None]) * w_theta
    marginal_r = np.sum(g, axis=1) * w_theta   # (NR,), integrates to 1 over r_grid
    return hat_x, hat_y, marginal_r


rng = np.random.default_rng(20260728)

# ---- continuous bias-vs-n sweep, n = 1 .. N_YEAR, FRESH independent
# random alpha for EACH n (per explicit instruction) ----
N_SWEEP_POINTS = 40
N_list = np.unique(np.round(np.logspace(0, np.log10(N_YEAR), N_SWEEP_POINTS)).astype(np.int64))

alpha_list, hat_x_list, hat_y_list = [], [], []
hat_delta_list, hat_g_list, bias_delta_list, bias_g_list = [], [], [], []

t0 = time.time()
for n in N_list:
    alpha = rng.uniform(0, 2 * np.pi)
    x0 = delta_true * np.cos(alpha)
    y0 = delta_true * np.sin(alpha)
    p_true = probs_for_xy(x0, y0)
    counts = rng.multinomial(int(n), p_true)
    g = posterior_polar(counts)
    hat_x, hat_y, _ = moments_polar(g)
    hat_delta = float(np.sqrt(hat_x ** 2 + hat_y ** 2))
    hat_g = float(g_agg_from_delta(hat_delta, m_a))
    bias_delta = hat_delta - delta_true
    bias_g = hat_g - g_true
    alpha_list.append(float(alpha)); hat_x_list.append(float(hat_x)); hat_y_list.append(float(hat_y))
    hat_delta_list.append(hat_delta); hat_g_list.append(hat_g)
    bias_delta_list.append(float(bias_delta)); bias_g_list.append(float(bias_g))
    print(f'  n={n:.4e}  alpha={alpha:.3f}  delta_hat={hat_delta:.4e}  bias_delta={bias_delta:.3e}  bias_g={bias_g:.3e}')
print(f'sweep done in {time.time()-t0:.1f}s')

# ---- posterior-over-delta snapshots at the 1-month / 6-month / 1-year
# checkpoints (each with its OWN fresh random alpha) ----
snapshots = []
for n in (N_MONTH, N_6MO, N_YEAR):
    alpha = rng.uniform(0, 2 * np.pi)
    x0 = delta_true * np.cos(alpha)
    y0 = delta_true * np.sin(alpha)
    p_true = probs_for_xy(x0, y0)
    counts = rng.multinomial(n, p_true)
    g = posterior_polar(counts)
    hat_x, hat_y, marginal_r = moments_polar(g)
    hat_delta = float(np.sqrt(hat_x ** 2 + hat_y ** 2))
    hat_g = float(g_agg_from_delta(hat_delta, m_a))
    print(f'[snapshot] n={n:.4e}  alpha={alpha:.3f}  (x0,y0)=({x0:.3e},{y0:.3e})  '
          f'delta_hat={hat_delta:.4e}')
    snapshots.append({
        'N_try': int(n), 'alpha': float(alpha), 'x0': float(x0), 'y0': float(y0),
        'hat_x': float(hat_x), 'hat_y': float(hat_y), 'hat_delta': hat_delta, 'hat_g': hat_g,
        'marginal_r': marginal_r.tolist(),
    })

results = {
    'topology': TOPOLOGY, 'LAYERS': LAYERS, 'POST_TOPOLOGY': POST_TOPOLOGY, 'POST_LAYERS': POST_LAYERS,
    'N_YEAR': N_YEAR, 'N_MONTH': N_MONTH, 'N_6MO': N_6MO,
    'm_a': m_a, 'delta_true': delta_true, 'g_true': g_true,
    'r_grid': r_grid.tolist(),
    'N_list': N_list.tolist(),
    'alpha_list': alpha_list, 'hat_x_list': hat_x_list, 'hat_y_list': hat_y_list,
    'hat_delta_list': hat_delta_list, 'hat_g_list': hat_g_list,
    'bias_delta_list': bias_delta_list, 'bias_g_list': bias_g_list,
    'snapshots': snapshots,
    'note': 'delta_0 fixed; for EACH n an INDEPENDENT fresh random DM phase alpha is drawn '
            '(true point (x0,y0)=delta_0(cos alpha,sin alpha)); Bayesian recovery on a polar '
            '(r,theta) grid (Jacobian-corrected to match a flat prior in Cartesian (x,y)); '
            'hat_x,hat_y are posterior means, hat_delta=sqrt(hat_x^2+hat_y^2), hat_g=g(hat_delta). '
            'Exposure scale matches the 1-year targeted-mass run: N_MONTH=2.63e10, '
            'N_6MO=1.578e11, N_YEAR=3.156e11.',
}
out_path = os.path.join(DATA_DIR, f'bayes_recover_results_{tag}_polar_1yr_randomalpha.json')
json.dump(results, open(out_path, 'w'), indent=2)
print(f'saved {out_path}')
