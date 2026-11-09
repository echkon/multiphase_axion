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
N_TRY = int(2.1e5)
N_SWEEP_POINTS = 20
N_POST_SNAPSHOTS = [1e3, 1e5, 2.1e5]

with open(os.path.join(DATA_DIR, f'ckpt_cfi_best_{tag}.pkl'), 'rb') as f:
    circuit0 = pickle.load(f)
prep_stage = circuit0[0]
post_stage = circuit0[2]


def probs_for_xy(x, y):
    c = [prep_stage, [sensing_xy, N, 1.0, [x, y]], post_stage]
    p = np.clip(_safe_probs(c), 0, None)
    return p / p.sum()


# ---- load the precomputed 2D (x,y) likelihood grid ----
grid = np.load(os.path.join(SCRIPT_DIR, 'pgrid_checkpoint_xy.npz'))
p_grid = grid['p_grid']            # (NAX, NAX, n_outcomes), p_grid[i,j] = P(k | x=axis_grid[i], y=axis_grid[j])
axis_grid = grid['axis_grid']      # shared x/y axis
assert grid['done'].all(), 'grid not fully built yet'
NAX = len(axis_grid)
X, Y = np.meshgrid(axis_grid, axis_grid, indexing='ij')  # X[i,j]=axis_grid[i], Y[i,j]=axis_grid[j]
print(f'loaded grid: {NAX}x{NAX}, range=[{axis_grid[0]:.3e},{axis_grid[-1]:.3e}]')

log_p_grid = np.log(p_grid + 1e-300)  # (NAX, NAX, n_outcomes)


def posterior_2d(counts):
    """counts: (n_outcomes,) -> normalized 2D posterior density over (x,y),
    integral over the (x,y) plane (via nested trapz) = 1."""
    loglik = log_p_grid @ counts  # (NAX, NAX)
    loglik -= loglik.max()
    likelihood = np.exp(loglik)
    prior = np.ones((NAX, NAX))
    unnorm = likelihood * prior
    norm = np.trapz(np.trapz(unnorm, axis_grid, axis=1), axis_grid, axis=0)
    return unnorm / norm


def moments_2d(posterior):
    hat_x = np.trapz(np.trapz(posterior * X, axis_grid, axis=1), axis_grid, axis=0)
    hat_y = np.trapz(np.trapz(posterior * Y, axis_grid, axis=1), axis_grid, axis=0)
    R = np.sqrt((X - hat_x) ** 2 + (Y - hat_y) ** 2)
    var_r = np.trapz(np.trapz(posterior * R ** 2, axis_grid, axis=1), axis_grid, axis=0)
    return hat_x, hat_y, np.sqrt(var_r)


rng = np.random.default_rng(20260727)

N_list = np.unique(np.round(np.logspace(0, np.log10(N_TRY), N_SWEEP_POINTS)).astype(np.int64))
alpha_list, x0_list, y0_list = [], [], []
hat_x_list, hat_y_list, hat_delta_list, hat_g_list = [], [], [], []
bias_delta_list, bias_g_list, std_delta_list = [], [], []

t0 = time.time()
for n in N_list:
    alpha = rng.uniform(0, 2 * np.pi)   # FRESH random DM phase for THIS n
    x0 = delta_true * np.cos(alpha)
    y0 = delta_true * np.sin(alpha)
    p_true = probs_for_xy(x0, y0)       # one fresh circuit eval at the true (x0,y0)
    counts = rng.multinomial(int(n), p_true)
    posterior = posterior_2d(counts)
    hat_x, hat_y, std_r = moments_2d(posterior)
    hat_delta = np.sqrt(hat_x ** 2 + hat_y ** 2)
    hat_g = float(g_agg_from_delta(hat_delta, m_a))
    bias_delta = hat_delta - delta_true
    bias_g = hat_g - g_true

    alpha_list.append(float(alpha)); x0_list.append(float(x0)); y0_list.append(float(y0))
    hat_x_list.append(float(hat_x)); hat_y_list.append(float(hat_y))
    hat_delta_list.append(float(hat_delta)); hat_g_list.append(hat_g)
    bias_delta_list.append(float(bias_delta)); bias_g_list.append(float(bias_g))
    std_delta_list.append(float(std_r))
    print(f'  n={n:.3e}  alpha={alpha:.3f}  (x0,y0)=({x0:.3e},{y0:.3e})  '
          f'(x_hat,y_hat)=({hat_x:.3e},{hat_y:.3e})  '
          f'delta_hat={hat_delta:.3e}  bias_delta={bias_delta:.3e}  bias_g={bias_g:.3e}')
print(f'sweep done in {time.time()-t0:.1f}s')

# ---- full 2D posterior snapshots at a few representative N, each with
# its OWN fresh random alpha (same convention as the sweep above) ----
snapshots = []
for n in N_POST_SNAPSHOTS:
    n = int(n)
    alpha = rng.uniform(0, 2 * np.pi)
    x0 = delta_true * np.cos(alpha)
    y0 = delta_true * np.sin(alpha)
    p_true = probs_for_xy(x0, y0)
    counts = rng.multinomial(n, p_true)
    posterior = posterior_2d(counts)
    hat_x, hat_y, std_r = moments_2d(posterior)
    hat_delta = float(np.sqrt(hat_x ** 2 + hat_y ** 2))
    hat_g = float(g_agg_from_delta(hat_delta, m_a))
    print(f'[snapshot] n={n:.3e}  alpha={alpha:.3f}  (x0,y0)=({x0:.3e},{y0:.3e})  '
          f'(x_hat,y_hat)=({hat_x:.3e},{hat_y:.3e})  delta_hat={hat_delta:.3e}')
    snapshots.append({
        'N_try': n, 'alpha': float(alpha), 'x0': float(x0), 'y0': float(y0),
        'hat_x': float(hat_x), 'hat_y': float(hat_y),
        'hat_delta': hat_delta, 'hat_g': hat_g,
        'posterior': posterior.tolist(),
    })

results = {
    'topology': TOPOLOGY, 'LAYERS': LAYERS, 'POST_TOPOLOGY': POST_TOPOLOGY, 'POST_LAYERS': POST_LAYERS,
    'N_TRY': N_TRY, 'm_a': m_a, 'delta_true': delta_true, 'g_true': g_true,
    'axis_grid': axis_grid.tolist(),
    'N_list': N_list.tolist(),
    'alpha_list': alpha_list, 'x0_list': x0_list, 'y0_list': y0_list,
    'hat_x_list': hat_x_list, 'hat_y_list': hat_y_list,
    'hat_delta_list': hat_delta_list, 'hat_g_list': hat_g_list,
    'bias_delta_list': bias_delta_list, 'bias_g_list': bias_g_list,
    'std_delta_list': std_delta_list,
    'snapshots': snapshots,
    'note': 'delta_0 fixed; for EACH n in N_list a FRESH random DM phase alpha '
            'is drawn, giving true point (x0,y0)=(delta_0 cos alpha, delta_0 sin alpha). '
            'Bayesian recovery is done jointly on the 2D (x,y) grid (built once from the '
            'trained full_L3 circuit); hat_x,hat_y are posterior means, and '
            'hat_delta=sqrt(hat_x^2+hat_y^2), hat_g=g(hat_delta) are derived quantities.',
}
out_path = os.path.join(DATA_DIR, f'bayes_recover_results_{tag}_xy_randomalpha.json')
json.dump(results, open(out_path, 'w'), indent=2)
print(f'saved {out_path}')
