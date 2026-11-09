import sys, os, pickle
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, CODE_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIGURE_DIR = os.path.join(SCRIPT_DIR, '..', 'figure')
os.makedirs(FIGURE_DIR, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import onecircuit
from onecircuit import *
from onecircuit.metric import sld_qfim, cfim

"""
Plot Tr(Q^-1) (quantum bound) vs. state-prep layer count, and Tr(C^-1)
(classical bound) vs. MEASUREMENT (post) layer count, for each topology.

Edit the constants below, then just run:
    python3 plot_vs_layers.py

  - LEFT panel (quantum bound, bar chart): loads
    data/ckpt_qfi_best_{topology}_L{L}.pkl for each L in LAYERS_LIST --
    this is Tr(Q^-1) at the trained theta with L layers (post-stage
    doesn't affect Q at all, so no "post" concept here). x-axis = LAYERS_LIST.

  - RIGHT panel (classical bound, line chart): scans the FULL Cartesian
    product of LAYERS_LIST (prep) x POST_LAYERS_LIST (post), i.e.
    1-1, 1-2, 1-3, 1-4, 2-1, ..., 4-4, loading
    data/ckpt_cfi_best_{topology}_L{prep_L}_post_{topology}_L{PL}.pkl
    for every (prep_L, PL) pair. Each (topology, prep_L) combination that
    has at least one matching file becomes its own line, plotted against
    PL on the x-axis (x-axis = POST_LAYERS_LIST); color = topology, line
    style = prep_L.

Any (topology, layer) combination with no matching checkpoint yet is
skipped with a printed warning -- you don't have to have trained every
combination before running this.
"""

# ---- edit these before running ----
TOPOLOGIES = ['star', 'ring', 'full']
LAYERS_LIST = [1, 2, 3, 4]       # LEFT panel x-axis (QFI prep layers); also the "prep" half of every CFI combo
POST_LAYERS_LIST = [1, 2, 3, 4]  # RIGHT panel x-axis (CFI post/measurement layers)
OUT_PNG = 'bound_vs_layers.eps'
# ------------------------------------

N = 5
x0, y0 = 6.498913663721062e-06, 0.0
REG = 1e-6

colors = {'star': 'tab:blue', 'ring': 'tab:green', 'full': 'tab:red'}
labels = {'star': 'star', 'ring': 'ring', 'full': 'circulant'}
BAR_WIDTH = 0.15  # fixed bar width, shared by BOTH panels -- every bar is
                  # this wide regardless of how many bars share a tick


def _load_circuit(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def q_bound(prep_stage):
    circuit = [prep_stage, [sensing_xy, N, 1.0, [x0, y0]], prep_stage]
    # post-stage doesn't affect Q (QFI is basis-independent under unitaries),
    # so any placeholder unitary stage works here; we reuse prep_stage's
    # graph type for a harmless, definitely-unitary placeholder.
    Q = np.real(sld_qfim(circuit)) + REG * np.eye(2)
    return np.trace(np.linalg.inv(Q))


def c_bound(prep_stage, post_stage):
    circuit = [prep_stage, [sensing_xy, N, 1.0, [x0, y0]], post_stage]
    C = np.real(cfim(circuit)) + REG * np.eye(2)
    return np.trace(np.linalg.inv(C))


if __name__ == '__main__':
    q_results = {topo: {} for topo in TOPOLOGIES}
    # c_results[topo][prep_L][PL] = Tr(C^-1)
    c_results = {topo: {L: {} for L in LAYERS_LIST} for topo in TOPOLOGIES}

    # ---- left panel data: Tr(Q^-1) vs. prep layer count ----
    for topo in TOPOLOGIES:
        for L in LAYERS_LIST:
            qfi_path = os.path.join(DATA_DIR, f'ckpt_qfi_best_{topo}_L{L}.pkl')
            if os.path.exists(qfi_path):
                prep_stage = _load_circuit(qfi_path)[0]
                q_results[topo][L] = q_bound(prep_stage)
            else:
                print(f'[skip] {qfi_path} not found')

    # ---- right panel data: Tr(C^-1) for every (prep_L, PL) combo ----
    for topo in TOPOLOGIES:
        for prep_L in LAYERS_LIST:
            for PL in POST_LAYERS_LIST:
                cfi_tag = f'{topo}_L{prep_L}_post_{topo}_L{PL}'
                cfi_path = os.path.join(DATA_DIR, f'ckpt_cfi_best_{cfi_tag}.pkl')
                if os.path.exists(cfi_path):
                    circ = _load_circuit(cfi_path)
                    c_results[topo][prep_L][PL] = c_bound(circ[0], circ[2])
                else:
                    print(f'[skip] {cfi_path} not found')

    # ---- y-limits computed BEFORE plotting -- needed for the `bottom=`
    # ---- argument below, which fixes a real EPS-export bug: on a log
    # ---- y-axis, ax.bar()'s default bottom=0 maps to log(0)=-inf, so the
    # ---- Agg/PNG backend silently clips the bar rectangle at the visible
    # ---- axes edge, but the EPS (vector) backend instead emits the TRUE,
    # ---- unclipped rectangle path running to -inf -- this is exactly the
    # ---- huge runaway bar seen when opening the .eps in Illustrator.
    # ---- Passing an explicit, finite bottom= (matching the y-axis floor)
    # ---- avoids the -inf coordinate entirely. ----
    all_vals = [v for topo in TOPOLOGIES for v in q_results[topo].values() if np.isfinite(v)]
    all_vals += [v for topo in TOPOLOGIES for prep_L in LAYERS_LIST
                 for v in c_results[topo][prep_L].values() if np.isfinite(v)]
    ymin, ymax = (min(all_vals), max(all_vals)) if all_vals else (1e-3, 1.0)
    y_bottom, y_top = ymin * 0.7, ymax * 2.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5))

    # ---- LEFT panel: grouped bar chart, x = LAYERS_LIST ----
    n_topo = len(TOPOLOGIES)
    x1 = np.arange(len(LAYERS_LIST))
    for i, topo in enumerate(TOPOLOGIES):
        offset = (i - (n_topo - 1) / 2) * BAR_WIDTH
        q_vals = [q_results[topo].get(L, np.nan) for L in LAYERS_LIST]
        ax1.bar(x1 + offset, q_vals, width=BAR_WIDTH, bottom=y_bottom, color=colors[topo], label=labels[topo])

    ax1.set_xlabel(r'$L_1$', fontsize=16)
    ax1.set_ylabel(r'$\mathrm{Tr}(Q^{-1})$', fontsize=16)
    ax1.set_yscale('log')
    ax1.set_xticks(x1)
    ax1.set_xticklabels([str(L) for L in LAYERS_LIST], fontsize=13)
    ax1.tick_params(axis='y', labelsize=13)
    ax1.grid(alpha=0.3, axis='y', which='both')
    ax1.legend(fontsize=11, loc='upper right')

    # ---- RIGHT panel: grouped bar chart, x = POST_LAYERS_LIST.
    # At EACH x-tick independently, only the (topology, prep_L) combos that
    # actually have data are placed, centered on that tick -- so a tick
    # with only 1 of the 12 possible combos gets a single bar sitting
    # right on the tick, not shifted off into an empty slot reserved for
    # combos that don't exist at that PL. Bar width is the SAME fixed
    # BAR_WIDTH as the left panel (not resized by how many bars share a
    # tick), so every bar in the whole figure is identically wide. Color =
    # topology, alpha shade = prep_L (darker = higher prep_L).
    x2 = np.arange(len(POST_LAYERS_LIST))
    n_prep = len(LAYERS_LIST)
    already_labeled = set()
    for k, PL in enumerate(POST_LAYERS_LIST):
        present = [(topo, prep_L) for topo in TOPOLOGIES for prep_L in LAYERS_LIST
                   if PL in c_results[topo][prep_L]]
        n_present = len(present)
        for idx, (topo, prep_L) in enumerate(present):
            offset = (idx - (n_present - 1) / 2) * BAR_WIDTH
            j = LAYERS_LIST.index(prep_L)
            alpha = 0.35 + 0.65 * (j / max(n_prep - 1, 1))
            key = (topo, prep_L)
            label = f'{labels[topo]} ($L_1$={prep_L})' if key not in already_labeled else None
            already_labeled.add(key)
            ax2.bar(x2[k] + offset, c_results[topo][prep_L][PL], width=BAR_WIDTH, bottom=y_bottom,
                    color=colors[topo], alpha=alpha, label=label)

    ax2.set_xlabel(r'$L_2$', fontsize=16)
    ax2.set_ylabel(r'$\mathrm{Tr}(C^{-1})$', fontsize=16)
    ax2.set_yscale('log')
    ax2.set_xticks(x2)
    ax2.set_xticklabels([str(PL) for PL in POST_LAYERS_LIST], fontsize=13)
    ax2.tick_params(axis='y', labelsize=13)
    ax2.set_xlim(-0.5, len(POST_LAYERS_LIST) - 0.5)
    ax2.grid(alpha=0.3, axis='y', which='both')
    ax2.legend(fontsize=11, ncol=3, loc='upper right')

    # give both panels the same y-scale so quantum/classical bounds are
    # directly visually comparable (ymin/ymax computed above, before plotting)
    ax1.set_ylim(y_bottom, y_top)
    ax2.set_ylim(y_bottom, y_top)

    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, OUT_PNG)
    fig.savefig(out_path, dpi=140)
    print(f'saved {out_path}')
    print('Q results:', q_results)
    print('C results:', c_results)
