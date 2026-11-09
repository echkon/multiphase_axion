import os, json, glob
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIGURE_DIR = os.path.join(SCRIPT_DIR, '..', 'figure')
os.makedirs(FIGURE_DIR, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
NOISE counterpart of plot_recover_mass.py -- does NOT modify
plot_recover_mass.py or any run/*.py. Edit the constants below, then run:
    python3 plot_recover_mass_noise.py

Draws KSVZ/DFSZ, ADMX/RBF+UF haloscopes, and CAST/HB-AGB astro bounds,
PLUS overlays every
data/targeted_mass_bayes_network_noise_optimized_results_{tag}_*_{noise_tag}_qfi{Q}_cfi{C}.json
file it finds (written by run/recover_mass_noise.py, one file per
single-mass targeted run under the qubit-network noise model), grouped
by (MODEL, N_years) exactly like plot_recover_mass.py.

TOPOLOGY/LAYERS/POST_TOPOLOGY/POST_LAYERS/T1_US/T2_US/CZ_ERROR_PROB/
OPTIMIZE_QFI/OPTIMIZE_CFI below must match whatever combination you
actually ran with run/recover_mass_noise.py (they select which noise
tag's files get globbed) -- everything else (KSVZ/DFSZ lines, haloscope/
astro bounds, styling) is identical to plot_recover_mass.py so the two
figures are directly comparable side by side.
"""

# ---- edit these before running -- must match run/recover_mass_noise.py ----
TOPOLOGY = 'ring'
LAYERS = 1
POST_TOPOLOGY = None       # None = same as TOPOLOGY
POST_LAYERS = 2            # None = same as LAYERS
T1_US = 100.0
T2_US = 100.0
CZ_ERROR_PROB = 0.005
OPTIMIZE_QFI = False
OPTIMIZE_CFI = False
SHOW_EXPERIMENTS = True
SHOW_ASTRO = True
YLIM = (1e-18, 1e-9)
# ------------------------------------

POST_TOPOLOGY = POST_TOPOLOGY or TOPOLOGY
POST_LAYERS = LAYERS if POST_LAYERS is None else POST_LAYERS

# (fill_color, line_color, label-or-None, label x [ueV], label y [GeV^-1])
EXPERIMENTS = [
    ('ADMX2018.txt', 'goldenrod', 'darkgoldenrod', 'ADMX', 3.6, 3e-10),
    ('ADMX2021.txt', 'goldenrod', 'darkgoldenrod', None, None, None),
    ('RBF.txt', 'silver', 'dimgray', 'RBF+UF', 7.0, 1.4e-13),
    ('UF.txt', 'silver', 'dimgray', None, None, None),
]

# (g_bound [GeV^-1], line_color, fill_color, linestyle, label, label x [ueV], va)
ASTRO_BOUNDS = [
    (5.67e-11, 'mediumpurple', 'mediumpurple', (0, (5, 1)), 'CAST helioscope', 0.7, 'bottom'),
    (0.47e-10, 'mediumvioletred', 'mediumvioletred', (0, (1, 1)), 'HB/AGB stars (globular clusters)', 0.55, 'top'),
]

# closed-form KSVZ/DFSZ, computed directly here (NOT read from any run/*.py
# output file) -- same formula/constants as recover_mass_noise.py
alpha_fs = 1 / 137.036


def g_KSVZ(m_a_ueV):
    return (alpha_fs / (2 * np.pi)) * 1.92 * m_a_ueV / 5.69e12


def g_DFSZ(m_a_ueV):
    return (alpha_fs / (2 * np.pi)) * 0.7467 * m_a_ueV / 5.69e12


def _fmt_n(n):
    """Same compact filename formatter as recover_mass_noise.py."""
    return f'{n:.4g}'.replace('+', '').replace('.', 'p')


def load_step_envelope(fname):
    """Load an AxionLimits polygon-style limit file (mass [ueV], coupling
    [GeV^-1]) with top-of-plot bracket rows at g=1 marking polygon
    closure, and collapse to the lower (most-constraining) envelope."""
    m, g = [], []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.replace(',', ' ').split()
            mm, gg = float(parts[0]), float(parts[1])
            if gg >= 1.0:
                continue
            m.append(mm)
            g.append(gg)
    m, g = np.array(m), np.array(g)
    order = np.argsort(m)
    return m[order], g[order]


if __name__ == '__main__':
    tag = f'{TOPOLOGY}_L{LAYERS}_post_{POST_TOPOLOGY}_L{POST_LAYERS}'
    noise_tag = f'T1{_fmt_n(T1_US)}_T2{_fmt_n(T2_US)}_czerr{_fmt_n(CZ_ERROR_PROB)}'
    qfi_cfi_suffix = f'qfi{int(OPTIMIZE_QFI)}_cfi{int(OPTIMIZE_CFI)}'

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    # ---- overlay every noise-targeted-mass single-point result found,
    # ---- grouped by (MODEL, N_years) -- see docstring ----
    noise_glob = os.path.join(
        DATA_DIR,
        f'targeted_mass_bayes_network_noise_optimized_results_{tag}_*_{noise_tag}_{qfi_cfi_suffix}.json')
    noise_paths = sorted(glob.glob(noise_glob))
    if not noise_paths:
        raise SystemExit(f'No targeted_mass_bayes_network_noise_optimized_results_{tag}_*_{noise_tag}_'
                          f'{qfi_cfi_suffix}.json files found in {DATA_DIR} -- run run/recover_mass_noise.py first '
                          f'(check TOPOLOGY/LAYERS/POST_LAYERS/T1_US/T2_US/CZ_ERROR_PROB/OPTIMIZE_QFI/OPTIMIZE_CFI '
                          f'above match what you actually ran)')

    by_group = {}  # (model, N_years) -> {'mass': [], 'mean': [], 'std': [], 'ntry': []}
    for p in noise_paths:
        r = json.load(open(p))
        model = r.get('model', 'KSVZ')
        n_years = round(r.get('N_years', 1.0), 6)
        g = by_group.setdefault((model, n_years), {'mass': [], 'mean': [], 'std': [], 'ntry': []})
        g['mass'].append(r['mass_ueV'])
        g['mean'].append(r['g_hat_mean'])
        g['std'].append(r['g_hat_std'])
        g['ntry'].append(r['N_TRY'])

    print(f'found {len(by_group)} (model, N_years) group(s): '
          f'{sorted(by_group.keys(), key=lambda k: (k[0], k[1]))}')

    MODEL_COLOR = {'KSVZ': 'tab:green', 'DFSZ': 'tab:purple'}
    MODEL_MARKER = {'KSVZ': 'D', 'DFSZ': '^'}
    # longer exposure -> darker/more opaque; shorter exposure -> lighter --
    # sorted descending so the longest-exposure group gets alpha=1.0
    years_sorted = sorted({k[1] for k in by_group}, reverse=True)
    alpha_for_years = {y: max(0.35, 1.0 - 0.55 * i / max(len(years_sorted) - 1, 1))
                       for i, y in enumerate(years_sorted)}

    m_a_all = []
    for (model, n_years), d in sorted(by_group.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        mass = np.array(d['mass']); mean = np.array(d['mean'])
        std = np.array(d['std']); ntry = np.array(d['ntry'])
        order = np.argsort(mass)
        mass, mean, std, ntry = mass[order], mean[order], std[order], ntry[order]
        m_a_all.append(mass)
        alpha = alpha_for_years[n_years]
        dur_label = f'{n_years*12:.2g}mo' if n_years < 1 else f'{n_years:.2g}yr'
        ax.errorbar(mass * 1e-6, mean, yerr=std, fmt=MODEL_MARKER[model], color=MODEL_COLOR[model],
                    alpha=alpha, ms=7, capsize=3, zorder=5,
                    label=f'noisy targeted, {model} target, {dur_label} (N_try~{ntry.mean():.1e})')
        print(f'overlaid {len(mass)} {model}-target, {dur_label} point(s): {sorted(mass.tolist())} ueV')

    t_mass = np.concatenate(m_a_all)

    # KSVZ/DFSZ -- always computed directly here, never depends on any
    # run/*.py output file
    m_dense = np.logspace(np.log10(0.1), np.log10(10), 200)
    ax.plot(m_dense * 1e-6, g_KSVZ(m_dense), 'k:', lw=1.6, label='KSVZ')
    ax.plot(m_dense * 1e-6, g_DFSZ(m_dense), 'k-.', lw=1.6, label='DFSZ')

    if SHOW_EXPERIMENTS:
        for fname, fill_color, line_color, label, tx, ty in EXPERIMENTS:
            path = os.path.join(DATA_DIR, fname)
            if not os.path.exists(path):
                print(f'[skip] {path} not found')
                continue
            m, g = load_step_envelope(path)
            ax.fill_between(m, g, 1.0, color=fill_color, alpha=0.55, lw=0, zorder=2)
            ax.plot(m, g, color=line_color, lw=1.0, zorder=3)
            if label:
                ax.text(tx * 1e-6, ty, label, fontsize=8, color=line_color, ha='center')

    x_lo, x_hi = float(t_mass.min()) * 1e-6, float(t_mass.max()) * 1e-6
    if SHOW_ASTRO:
        for g_bound, line_color, fill_color, ls, label, tx, va in ASTRO_BOUNDS:
            ax.axhline(g_bound, color=line_color, lw=1.4, ls=ls, zorder=1)
            ax.fill_between([x_lo, x_hi], [g_bound, g_bound], [1.0, 1.0],
                             color=fill_color, alpha=0.12, zorder=0)
            y_text = g_bound * (1.5 if va == 'bottom' else 0.55)
            ax.text(tx * 1e-6, y_text, label, fontsize=8, color=line_color, va=va, ha='center')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(x_lo, x_hi)
    if YLIM is not None:
        ax.set_ylim(*YLIM)
    ax.set_xlabel('axion mass m_a (eV)')
    ax.set_ylabel(r'$g_{a\gamma\gamma}$  [GeV$^{-1}$]')
    ax.set_title(f'{tag}, {noise_tag}: NOISY targeted-mass Bayesian recovery vs.\nhaloscope and astrophysical constraints',
                 fontsize=10)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3, which='major')
    fig.tight_layout()

    out_path = os.path.join(FIGURE_DIR, f'targeted_mass_bayes_noise_{tag}_{noise_tag}_{qfi_cfi_suffix}.png')
    fig.savefig(out_path, dpi=140)
    print(f'saved {out_path}')
