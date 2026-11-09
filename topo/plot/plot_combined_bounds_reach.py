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
Combines plot_noise.py (Tr(Q^-1)/Tr(C^-1) vs. layer count, UNDER THE SAME
NOISE MODEL -- reads data/optimal_noise_results_{noise_tag}.json, written
by run/train_noise.py) and plot_recover_mass_noise.py (NOISY targeted-mass
Bayesian recovery vs. haloscope/astro bounds) into ONE figure, side by
side:

    +------------------+---------------------------+
    |  Tr(Q^-1) vs L1   |                           |
    |  (bar, 3 topos,   |   NOISY targeted-mass      |
    |   NOISY)          |   Bayesian recovery vs.    |
    +------------------+   haloscope/astro bounds   |
    |  Tr(C^-1) vs L2   |   (NOISY)                  |
    |  (bar, best topo, |                           |
    |   NOISY)          |                           |
    +------------------+---------------------------+

Left column: quantum bound on top, classical bound on bottom (stacked) --
BOTH computed under the qubit-network noise model (T1/T2 relaxation + CZ
depolarizing error), NOT the noiseless checkpoints plot_vs_layers.py uses.
Right column: the noisy mass-scan plot, spanning the full height.

Pure PLOTTING -- no onecircuit/qiskit import needed, everything is read
from already-computed JSON files. Does NOT modify plot_noise.py,
plot_vs_layers.py, or plot_recover_mass_noise.py -- standalone script,
edit the constants below then run:
    python3 plot_combined_bounds_reach.py
"""

# ---- noise-model tag shared by BOTH panels (must match run/train_noise.py
# and run/recover_mass_noise.py) ----
T1_US = 100.0
T2_US = 100.0
CZ_ERROR_PROB = 0.005

# ---- noisy targeted-mass recovery constants (same as plot_recover_mass_noise.py) ----
NOISE_TOPOLOGY = 'ring'
NOISE_LAYERS = 1
NOISE_POST_TOPOLOGY = None
NOISE_POST_LAYERS = 2
OPTIMIZE_QFI = False
OPTIMIZE_CFI = False
SHOW_EXPERIMENTS = False
SHOW_ASTRO = False
YLIM = (1e-18, 1e-14)

OUT_NAME = 'noise_combined_bounds_reach.eps'
# ------------------------------------

NOISE_POST_TOPOLOGY = NOISE_POST_TOPOLOGY or NOISE_TOPOLOGY
NOISE_POST_LAYERS = NOISE_LAYERS if NOISE_POST_LAYERS is None else NOISE_POST_LAYERS

colors = {'star': 'tab:blue', 'ring': 'tab:green', 'full': 'tab:red'}
labels = {'star': 'star', 'ring': 'ring', 'full': 'circulant'}
BAR_WIDTH = 0.15

EXPERIMENTS = [
    ('ADMX2018.txt', 'goldenrod', 'darkgoldenrod', 'ADMX', 3.6, 3e-10),
    ('ADMX2021.txt', 'goldenrod', 'darkgoldenrod', None, None, None),
    ('RBF.txt', 'silver', 'dimgray', 'RBF+UF', 7.0, 1.4e-13),
    ('UF.txt', 'silver', 'dimgray', None, None, None),
]

ASTRO_BOUNDS = [
    (5.67e-11, 'mediumpurple', 'mediumpurple', (0, (5, 1)), 'CAST helioscope', 0.7, 'bottom'),
    (0.47e-10, 'mediumvioletred', 'mediumvioletred', (0, (1, 1)), 'HB/AGB stars (globular clusters)', 0.55, 'top'),
]

alpha_fs = 1 / 137.036


def g_KSVZ(m_a_ueV):
    return (alpha_fs / (2 * np.pi)) * 1.92 * m_a_ueV / 5.69e12


def g_DFSZ(m_a_ueV):
    return (alpha_fs / (2 * np.pi)) * 0.7467 * m_a_ueV / 5.69e12


# axion mass <-> Compton frequency, f = m_a c^2 / h  (m_a in eV, f in Hz)
H_EV_S = 4.135667696e-15  # Planck constant [eV s]


def mass_eV_to_freq_Hz(m_eV):
    return np.asarray(m_eV) / H_EV_S


def freq_Hz_to_mass_eV(f_Hz):
    return np.asarray(f_Hz) * H_EV_S


def _fmt_n(n):
    return f'{n:.4g}'.replace('+', '').replace('.', 'p')


def load_step_envelope(fname):
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
    noise_tag = f'T1{_fmt_n(T1_US)}_T2{_fmt_n(T2_US)}_czerr{_fmt_n(CZ_ERROR_PROB)}'

    # ================= left column: NOISY Q/C vs. layers =================
    # -- read data/optimal_noise_results_{noise_tag}.json (run/train_noise.py's
    # -- output), same file plot_noise.py draws from -- NOT the noiseless
    # -- checkpoints plot_vs_layers.py uses.
    noise_bounds_path = os.path.join(DATA_DIR, f'optimal_noise_results_{noise_tag}.json')
    if not os.path.exists(noise_bounds_path):
        raise SystemExit(f'{noise_bounds_path} not found -- run run/train_noise.py for '
                          f'(T1_US={T1_US}, T2_US={T2_US}, CZ_ERROR_PROB={CZ_ERROR_PROB}) first')
    nb = json.load(open(noise_bounds_path))

    TOPOLOGIES = nb['topologies']
    LAYERS_LIST = nb['layers_list']
    POST_LAYERS_LIST = nb['post_layers_list']
    q_results = {topo: {int(L): v for L, v in d.items()} for topo, d in nb['q_results'].items()}
    c_results = {int(PL): v for PL, v in nb['c_results'].items()}
    best_topo, best_L = nb['best_topo'], nb['best_L']

    all_vals = [v for topo in TOPOLOGIES for v in q_results[topo].values() if np.isfinite(v)]
    all_vals += [v for v in c_results.values() if np.isfinite(v)]
    ymin, ymax = (min(all_vals), max(all_vals)) if all_vals else (1e-3, 1.0)
    y_bottom, y_top = ymin * 0.7, ymax * 2.2

    # ================= right panel: noisy targeted-mass recovery =================
    tag = f'{NOISE_TOPOLOGY}_L{NOISE_LAYERS}_post_{NOISE_POST_TOPOLOGY}_L{NOISE_POST_LAYERS}'
    qfi_cfi_suffix = f'qfi{int(OPTIMIZE_QFI)}_cfi{int(OPTIMIZE_CFI)}'

    noise_glob = os.path.join(
        DATA_DIR,
        f'targeted_mass_bayes_network_noise_optimized_results_{tag}_*_{noise_tag}_{qfi_cfi_suffix}.json')
    noise_paths = sorted(glob.glob(noise_glob))
    if not noise_paths:
        raise SystemExit(f'No targeted_mass_bayes_network_noise_optimized_results_{tag}_*_{noise_tag}_'
                          f'{qfi_cfi_suffix}.json files found in {DATA_DIR} -- run run/recover_mass_noise.py first')

    by_group = {}
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
    years_sorted = sorted({k[1] for k in by_group}, reverse=True)
    alpha_for_years = {y: max(0.35, 1.0 - 0.55 * i / max(len(years_sorted) - 1, 1))
                       for i, y in enumerate(years_sorted)}

    # ================= figure layout =================
    # Left column: 2 rows (Q top, C bottom). Right column: 1 tall panel.
    fig = plt.figure(figsize=(12.5, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.3], height_ratios=[1, 1],
                           wspace=0.28, hspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])   # Tr(Q^-1) vs L1 -- top-left (NOISY)
    ax2 = fig.add_subplot(gs[1, 0])   # Tr(C^-1) vs L2 -- bottom-left (NOISY)
    ax = fig.add_subplot(gs[:, 1])    # noisy mass scan -- full-height right

    # ---- top-left: Tr(Q^-1) bar chart, NOISY, all 3 topologies ----
    n_topo = len(TOPOLOGIES)
    x1 = np.arange(len(LAYERS_LIST))
    for i, topo in enumerate(TOPOLOGIES):
        offset = (i - (n_topo - 1) / 2) * BAR_WIDTH
        q_vals = [q_results[topo].get(L, np.nan) for L in LAYERS_LIST]
        ax1.bar(x1 + offset, q_vals, width=BAR_WIDTH, bottom=y_bottom, color=colors[topo], label=labels[topo])

    ax1.set_xlabel(r'$L_1$', fontsize=14)
    ax1.set_ylabel(r'$\mathrm{Tr}(Q^{-1})$', fontsize=14)
    ax1.set_yscale('log')
    ax1.set_xticks(x1)
    ax1.set_xticklabels([str(L) for L in LAYERS_LIST], fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(alpha=0.3, axis='y', which='both')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_ylim(y_bottom, y_top)

    # ---- bottom-left: Tr(C^-1) bar chart, NOISY, best topology only
    # (matches train_noise.py's design: it picks the single best (topo, L1)
    # from the quantum-bound sweep, then only sweeps L2 for that one combo) ----
    x2 = np.arange(len(POST_LAYERS_LIST))
    c_vals = [c_results.get(PL, np.nan) for PL in POST_LAYERS_LIST]
    ax2.bar(x2, c_vals, width=BAR_WIDTH, bottom=y_bottom, color=colors[best_topo],
            label=f'{labels[best_topo]} ($L_1$={best_L})')

    ax2.set_xlabel(r'$L_2$', fontsize=14)
    ax2.set_ylabel(r'$\mathrm{Tr}(C^{-1})$', fontsize=14)
    ax2.set_yscale('log')
    ax2.set_xticks(x2)
    ax2.set_xticklabels([str(PL) for PL in POST_LAYERS_LIST], fontsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_xlim(-0.5, len(POST_LAYERS_LIST) - 0.5)
    ax2.grid(alpha=0.3, axis='y', which='both')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_ylim(y_bottom, y_top)

    # ---- right: noisy targeted-mass Bayesian recovery ----
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
                    alpha=alpha, ms=11, mew=1.5, capsize=4, zorder=5,
                    label=f'noisy targeted, {model} target, {dur_label} (N_try~{ntry.mean():.1e})')
        print(f'overlaid {len(mass)} {model}-target, {dur_label} point(s): {sorted(mass.tolist())} ueV')

    t_mass = np.concatenate(m_a_all)

    m_dense = np.logspace(np.log10(0.1), np.log10(10), 200)
    ax.plot(m_dense * 1e-6, g_KSVZ(m_dense), 'k:', lw=1.6, label='KSVZ')
    ax.plot(m_dense * 1e-6, g_DFSZ(m_dense), 'k-.', lw=1.6, label='DFSZ')

    # same EPS-runaway-bar fix as plot_recover_mass.py -- fill up to the
    # axis top (YLIM[1]) instead of 1.0
    FILL_TOP = YLIM[1] if YLIM is not None else 1.0

    if SHOW_EXPERIMENTS:
        for fname, fill_color, line_color, label, tx, ty in EXPERIMENTS:
            path = os.path.join(DATA_DIR, fname)
            if not os.path.exists(path):
                print(f'[skip] {path} not found')
                continue
            m, g = load_step_envelope(path)
            ax.fill_between(m, g, FILL_TOP, color=fill_color, alpha=0.55, lw=0, zorder=2)
            ax.plot(m, g, color=line_color, lw=1.0, zorder=3)
            if label:
                ax.text(tx * 1e-6, ty, label, fontsize=8, color=line_color, ha='center')

    x_lo, x_hi = float(t_mass.min()) * 1e-6, float(t_mass.max()) * 1e-6
    if SHOW_ASTRO:
        astro_sorted = sorted(ASTRO_BOUNDS, key=lambda a: a[0])
        for i, (g_bound_v, line_color, fill_color, ls, label, tx, va) in enumerate(astro_sorted):
            fill_top_i = astro_sorted[i + 1][0] if i + 1 < len(astro_sorted) else FILL_TOP
            ax.axhline(g_bound_v, color=line_color, lw=1.4, ls=ls, zorder=1)
            ax.fill_between([x_lo, x_hi], [g_bound_v, g_bound_v], [fill_top_i, fill_top_i],
                             color=fill_color, alpha=0.15, zorder=0)
            y_text = g_bound_v * (1.5 if va == 'bottom' else 0.55)
            ax.text(tx * 1e-6, y_text, label, fontsize=8, color=line_color, va=va, ha='center')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(x_lo, x_hi)
    if YLIM is not None:
        ax.set_ylim(*YLIM)
    ax.set_xlabel('axion mass m_a (eV)', fontsize=14)
    ax.set_ylabel(r'$g_{a\gamma\gamma}$  [GeV$^{-1}$]', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3, which='major')

    # secondary x-axis on top: Compton frequency f = m_a/h corresponding to
    # the mass shown on the bottom axis
    secax = ax.secondary_xaxis('top', functions=(mass_eV_to_freq_Hz, freq_Hz_to_mass_eV))
    secax.set_xlabel('frequency f (Hz)', fontsize=13)
    secax.tick_params(axis='x', labelsize=11)

    out_path = os.path.join(FIGURE_DIR, OUT_NAME)
    fig.savefig(out_path, dpi=140)
    print(f'saved {out_path}')
