import os, json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIGURE_DIR = os.path.join(SCRIPT_DIR, '..', 'figure')
os.makedirs(FIGURE_DIR, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
NEW, STANDALONE script -- does NOT modify run/train_noise.py or
plot/plot_vs_layers.py. Edit the constants below, then just run:
    python3 plot_noise.py

Pure PLOTTING -- reads data/optimal_noise_results_{noise_tag}.json
(written once by run/train_noise.py after it finishes training) and
redraws the two-panel figure. Re-run this as many times as you like
while tweaking figsize/aspect ratio, gridlines, label sizes, colors,
titles, y-limits, etc. -- NO retraining, loads instantly.

If you change T1_US/T2_US/CZ_ERROR_PROB below to point at a different
noise combination, make sure run/train_noise.py has already been run
(and finished) for that combination -- otherwise the matching JSON file
won't exist yet.
"""

# ---- edit these before running ----
T1_US = 100.0
T2_US = 100.0
CZ_ERROR_PROB = 0.005

# ---- figure styling -- this is what you'll actually be iterating on ----
FIGSIZE = (13, 4.5)          # (width, height) in inches -- overall aspect ratio
BAR_WIDTH = 0.15
DPI = 140
GRID_ALPHA = 0.3
GRID_AXIS = 'y'               # 'y' | 'x' | 'both'
LABEL_FONTSIZE = 11
TITLE_FONTSIZE = 11
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 8
YLIM = None                   # None = auto (shared across both panels); or e.g. (1e-2, 1e-1)
OUT_PNG = 'bound_vs_layers_network_noise.png'
# ------------------------------------

colors = {'star': 'tab:blue', 'ring': 'tab:green', 'full': 'tab:red'}
labels = {'star': 'star', 'ring': 'ring', 'full': 'circulant'}


def _fmt_n(n):
    return f'{n:.4g}'.replace('+', '').replace('.', 'p')


noise_tag = f'T1{_fmt_n(T1_US)}_T2{_fmt_n(T2_US)}_czerr{_fmt_n(CZ_ERROR_PROB)}'


if __name__ == '__main__':
    results_path = os.path.join(DATA_DIR, f'optimal_noise_results_{noise_tag}.json')
    if not os.path.exists(results_path):
        raise SystemExit(f'{results_path} not found -- run run/train_noise.py for this '
                          f'(T1_US={T1_US}, T2_US={T2_US}, CZ_ERROR_PROB={CZ_ERROR_PROB}) combo first')
    r = json.load(open(results_path))

    TOPOLOGIES = r['topologies']
    LAYERS_LIST = r['layers_list']
    POST_LAYERS_LIST = r['post_layers_list']
    # JSON round-trips int dict keys as strings -- cast back
    q_results = {topo: {int(L): v for L, v in d.items()} for topo, d in r['q_results'].items()}
    c_results = {int(PL): v for PL, v in r['c_results'].items()}
    best_topo, best_L, best_q = r['best_topo'], r['best_L'], r['best_q']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    n_topo = len(TOPOLOGIES)
    x1 = np.arange(len(LAYERS_LIST))
    for i, topo in enumerate(TOPOLOGIES):
        offset = (i - (n_topo - 1) / 2) * BAR_WIDTH
        q_vals = [q_results[topo].get(L, np.nan) for L in LAYERS_LIST]
        ax1.bar(x1 + offset, q_vals, width=BAR_WIDTH, color=colors[topo], label=labels[topo])
    ax1.set_xlabel('state-prep (theta) layers', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel(r'$\mathrm{Tr}(Q^{-1})$  (quantum bound, noisy)', fontsize=LABEL_FONTSIZE)
    ax1.set_yscale('log')
    ax1.set_xticks(x1)
    ax1.set_xticklabels([str(L) for L in LAYERS_LIST], fontsize=TICK_FONTSIZE)
    ax1.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax1.grid(alpha=GRID_ALPHA, axis=GRID_AXIS)
    ax1.legend(fontsize=LEGEND_FONTSIZE)
    ax1.set_title(f'Quantum Cramer-Rao bound vs. prep layers\n(T1={T1_US}us, T2={T2_US}us, CZ_err={CZ_ERROR_PROB})',
                  fontsize=TITLE_FONTSIZE)

    x2 = np.arange(len(POST_LAYERS_LIST))
    c_vals = [c_results.get(PL, np.nan) for PL in POST_LAYERS_LIST]
    ax2.bar(x2, c_vals, width=BAR_WIDTH, color=colors[best_topo], label=f'{labels[best_topo]} (prep={best_L})')
    ax2.set_xlabel('measurement (mu) layers', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel(r'$\mathrm{Tr}(C^{-1})$  (classical bound, noisy)', fontsize=LABEL_FONTSIZE)
    ax2.set_yscale('log')
    ax2.set_xticks(x2)
    ax2.set_xticklabels([str(PL) for PL in POST_LAYERS_LIST], fontsize=TICK_FONTSIZE)
    ax2.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax2.grid(alpha=GRID_ALPHA, axis=GRID_AXIS)
    ax2.legend(fontsize=LEGEND_FONTSIZE)
    ax2.set_title(f'Classical Cramer-Rao bound vs. measurement layers\n(winning prep: {labels[best_topo]}, L={best_L})',
                  fontsize=TITLE_FONTSIZE)

    if YLIM is not None:
        ax1.set_ylim(*YLIM)
        ax2.set_ylim(*YLIM)
    else:
        all_vals = [v for topo in TOPOLOGIES for v in q_results[topo].values() if np.isfinite(v)]
        all_vals += [v for v in c_results.values() if np.isfinite(v)]
        if all_vals:
            ymin, ymax = min(all_vals), max(all_vals)
            ax1.set_ylim(ymin * 0.5, ymax * 2)
            ax2.set_ylim(ymin * 0.5, ymax * 2)

    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, OUT_PNG)
    fig.savefig(out_path, dpi=DPI)
    print(f'saved {out_path}')
