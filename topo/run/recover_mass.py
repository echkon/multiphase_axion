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
from onecircuit.metric import _safe_probs, cfim
from axion_coupling import delta_axion, g_agg_from_delta, tau_DM_us
from bayes_core import sequential_recover as _core_sequential_recover

"""
NEW, STANDALONE script -- does NOT modify mass_scan_bayes.py,
bayes_recover.py, or bayes_core.py. Edit MASS_UEV below, then run:
    python3 recover_mass.py

TARGETED (single-mass) Bayesian recovery -- the opposite search strategy
from mass_scan_bayes.py. mass_scan_bayes.py splits a fixed total scan
time across ~1.5e6 frequency bins to cover the whole [0.1,10] ueV window
(a broadband scan), giving only N_try~2.1e5 per mass point at 1 year.
This script instead dedicates the ENTIRE exposure to ONE hypothesis mass
MASS_UEV -- e.g. if you have an independent reason (astrophysical/
cosmological prior) to suspect the axion is near that particular mass.
No splitting across bins means MUCH higher N_try per point: 1 year
dedicated to a single mass gives N_try ~ 3.16e11 (vs. 2.1e5/bin in a full
scan) -- close to the "idealized" N_try=1e12 curve from the old
fig3_new.png reference figure, and actually achievable on a realistic
timescale (~3.2 years to hit N_try=1e12 exactly, at THIS mass only).

N_try derivation: entire N_YEARS of exposure goes into shots of length
tau = min(tau_DM(MASS_UEV), T2=100us) (same tau saturation logic as
axion_coupling.delta_axion/g_agg_from_delta), so
    N_try = (N_YEARS * 3.156e7 s) / tau_s
-- no frequency-bin division at all, unlike mass_scan_bayes.py's
N_try_for_scan().

Reuses bayes_core.sequential_recover() (the SAME shared algorithm used
by bayes_recover.py and mass_scan_bayes.py, not re-implemented here) and
the same eig_min(C)-based grid-half-width fix (grid sized off the
circuit's real resolving power, not off delta_true itself, which is
usually thousands of times too narrow -- see bayes_core.py/
bayes_recover.py's docstrings for the full story).

MODEL selects which benchmark coupling g_target(m_a) is used as the
"true" value being recovered -- 'KSVZ' (default, same as before) or
'DFSZ' (weaker coupling at the same mass -- a harder target, since
smaller delta_true means a smaller signal at fixed N_try). Lets you
repeat the exact same exercise against the DFSZ benchmark line to see
whether the network is also sensitive enough there, not just at KSVZ.

Saves to
data/targeted_mass_bayes_results_{TOPOLOGY}_L{LAYERS}_post_{POST_TOPOLOGY}_L{POST_LAYERS}_{MODEL}_m{MASS_UEV}ueV_N{N_TRY}.json
-- MODEL, mass, AND N_TRY are ALL baked into the filename (same
convention as bayes_recover.py/mass_scan_bayes.py's N_TRY-in-filename
fix), so re-running at a different mass, model, or N_YEARS never
overwrites a previous result -- you can build up a whole collection of
targeted points (including BOTH KSVZ and DFSZ runs side by side), one
run per (mass, model), and plot them all together later (separate plot
script, written once you have data to show -- this script only computes
and saves).
"""

# ---- edit these before running ----
TOPOLOGY = 'full'                    # star | ring | full
LAYERS = 3                           # must match the LAYERS used to train the checkpoint
POST_TOPOLOGY = None                 # measurement ansatz's topology per train_cfi.py; None = same as TOPOLOGY
POST_LAYERS = 3                      # measurement ansatz's layer count per train_cfi.py; None = same as LAYERS
MASS_UEV = 1.0                       # <-- set the ONE target mass [ueV] here before each run
MODEL = 'KSVZ'                       # 'KSVZ' or 'DFSZ' -- which benchmark coupling is the
                                      # "true" g_target(m_a) being recovered
N_YEARS = 1.0                        # total exposure dedicated to this single mass, in years
N_RUNS = 5                           # independent Bayesian recoveries averaged at this mass
GRID_N = 5000                        # number of candidate delta values scanned per recovery
GRID_HALF_WIDTH_SIGMAS = 8           # grid half-width = SIGMAS * sigma_theory(N_TRY)
# ------------------------------------
TOPOLOGY = os.environ.get('TOPOLOGY', TOPOLOGY)
LAYERS = int(os.environ.get('LAYERS', LAYERS))
MASS_UEV = float(os.environ.get('MASS_UEV', MASS_UEV))  # lets you launch several masses as
                                                          # separate background processes, e.g.:
                                                          #   MASS_UEV=2.7 python3 recover_mass.py &
                                                          #   MASS_UEV=4.0 python3 recover_mass.py &
MODEL = os.environ.get('MODEL', MODEL)  # e.g. MODEL=DFSZ python3 recover_mass.py &
assert MODEL in ('KSVZ', 'DFSZ'), f"MODEL must be 'KSVZ' or 'DFSZ', got {MODEL!r}"
N_YEARS = float(os.environ.get('N_YEARS', N_YEARS))  # e.g. N_YEARS=0.5 python3 recover_mass.py &
                                                        # -- N_TRY (derived from N_YEARS) is baked
                                                        # into the output filename, so different
                                                        # N_YEARS never overwrite each other
POST_TOPOLOGY = POST_TOPOLOGY or TOPOLOGY
POST_LAYERS = LAYERS if POST_LAYERS is None else POST_LAYERS

topo = TOPOLOGY
# same ALWAYS-both naming as train_cfi.py: {topo}_L{layers}_post_{post_topology}_L{post_layers}
tag = f'{topo}_L{LAYERS}_post_{POST_TOPOLOGY}_L{POST_LAYERS}'


def _fmt_n(n):
    """1e5 -> '1e05', 2.1e5 -> '2p1e05', 2.7 -> '2p7' -- compact,
    filename-safe. Same formatter as bayes_recover.py/mass_scan_bayes.py."""
    return f'{n:.4g}'.replace('+', '').replace('.', 'p')


with open(os.path.join(DATA_DIR, f'ckpt_cfi_best_{tag}.pkl'), 'rb') as f:
    circuit0 = pickle.load(f)
prep_stage = circuit0[0]
post_stage = circuit0[2]

N = 5
alpha_fs = 1 / 137.036
hbar_eVs = 6.582119569e-16  # eV*s
sec_per_year = 3.156e7       # matches Chen et al.'s convention, same constant as mass_scan_bayes.py
T2_us = 100.0
x0_fiducial = 6.498913663721062e-06  # same fiducial point train_cfi.py optimized (x0, y0=0)


def g_KSVZ(m_a_ueV):
    return (alpha_fs / (2 * np.pi)) * 1.92 * m_a_ueV / 5.69e12


def g_DFSZ(m_a_ueV):
    return (alpha_fs / (2 * np.pi)) * 0.7467 * m_a_ueV / 5.69e12


def probs_for_delta(d):
    c = [prep_stage, [sensing_xy, N, 1.0, [d, 0.0]], post_stage]
    p = np.clip(_safe_probs(c), 0, None)
    return p / p.sum()


# ---- N_try for a SINGLE targeted mass: the ENTIRE N_YEARS exposure goes
# ---- into shots at THIS one mass -- no splitting across frequency bins
# ---- (that's what makes this so much bigger than mass_scan_bayes.py's
# ---- per-bin N_try at the same N_YEARS) ----
tau_us = min(tau_DM_us(MASS_UEV), T2_us)
tau_s = tau_us * 1e-6
N_TRY = (N_YEARS * sec_per_year) / tau_s

run_tag = f'{tag}_{MODEL}_m{_fmt_n(MASS_UEV)}ueV_N{_fmt_n(N_TRY)}'

# ---- eig_min(C) measured ONCE at the trained fiducial point -- used only
# ---- to size the grid half-width below (same fix as bayes_recover.py/
# ---- mass_scan_bayes.py -- delta_true itself is thousands of times too
# ---- narrow a scale to size the grid on) ----
_C_check = np.real(cfim([prep_stage, [sensing_xy, N, 1.0, [x0_fiducial, 0.0]], post_stage]))
eig_min = max(np.linalg.eigvalsh(_C_check).min(), 1e-12)
_sigma_theory_N_TRY = 1.0 / np.sqrt(N_TRY * eig_min)
GRID_HALF_WIDTH = GRID_HALF_WIDTH_SIGMAS * _sigma_theory_N_TRY
print(f'--- {run_tag}: N_years={N_YEARS}, tau={tau_us:.2f}us, N_try={N_TRY:.4e}, '
      f'eig_min(C)={eig_min:.4e}, grid half-width={GRID_HALF_WIDTH:.4e} ---')


def sequential_recover(delta_true, N_try):
    """Thin wrapper around bayes_core.sequential_recover(): builds this
    mass's (delta_grid, p_grid, p_true), then delegates the actual draw +
    counts-shortcut Bayes update to bayes_core.py -- the SAME shared
    algorithm bayes_recover.py and mass_scan_bayes.py use, not
    re-implemented here."""
    hw = GRID_HALF_WIDTH
    delta_grid = np.linspace(delta_true - hw, delta_true + hw, GRID_N)
    p_grid = np.array([probs_for_delta(d) for d in delta_grid])
    p_true = probs_for_delta(delta_true)

    _, mean, _, _, _ = _core_sequential_recover(delta_grid, p_grid, p_true, N_try, delta_true)
    return mean


if __name__ == '__main__':
    g_model_fn = g_KSVZ if MODEL == 'KSVZ' else g_DFSZ
    g_target = g_model_fn(MASS_UEV)
    delta_true, _ = delta_axion(g_target, MASS_UEV)

    deltas_hat = np.array([sequential_recover(delta_true, N_TRY) for _ in range(N_RUNS)])
    g_hats = np.array([g_agg_from_delta(abs(d), MASS_UEV) for d in deltas_hat])

    print(f'  m_a={MASS_UEV:.4f} ueV [{MODEL}]: g_target={g_target:.3e}  '
          f'g_hat={g_hats.mean():.3e} +- {g_hats.std():.3e}  '
          f'ratio={g_hats.mean() / g_target:.3f}')

    results = {
        'topology': topo, 'LAYERS': LAYERS,
        'POST_TOPOLOGY': POST_TOPOLOGY, 'POST_LAYERS': POST_LAYERS,
        'model': MODEL,
        'mass_ueV': MASS_UEV, 'N_years': N_YEARS, 'N_TRY': N_TRY,
        'tau_us': tau_us, 'eig_min_C': float(eig_min),
        'g_target': float(g_target), 'delta_true': float(delta_true),
        'g_hat_runs': g_hats.tolist(),
        'g_hat_mean': float(g_hats.mean()),
        'g_hat_std': float(g_hats.std()),
    }

    out_path = os.path.join(DATA_DIR, f'targeted_mass_bayes_results_{run_tag}.json')
    json.dump(results, open(out_path, 'w'), indent=2)
    print(f'saved {out_path}')
