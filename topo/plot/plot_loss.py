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
Edit the constants below, then just run:
    python3 plot_loss.py

For STAGE='qfi': LAYERS_LIST sweeps the PREP (theta) layer count, e.g.
data/costs_qfi_{TOPOLOGY}_L1.json, ..._L2.json, ... -- PREP_LAYERS is
unused.

For STAGE='cfi': train_cfi.py's tag ALWAYS spells out both prep and post
ansatz explicitly (see its docstring): {TOPOLOGY}_L{PREP_LAYERS}_post_{POST_TOPOLOGY}_L{L}.
A theta checkpoint is typically trained ONCE at a fixed layer count
(PREP_LAYERS), then CFI is tried at several different measurement layer
counts against that SAME fixed theta -- so here, LAYERS_LIST sweeps the
POST (measurement) layer count, with PREP_LAYERS held fixed. Example:
if you have data/costs_cfi_full_L3_post_full_L1.json,
..._full_L3_post_full_L2.json, ..., set TOPOLOGY='full',
PREP_LAYERS=3, LAYERS_LIST=[1,2,3,4].

Whichever ones actually have a cost-history file get plotted together as
separate lines on the SAME figure; a layer count with no matching file
yet is skipped with a printed warning.

BURN_IN steps are skipped only when auto-scaling the y-axis (log scale,
tightly fit to the post-burn-in range -- a huge loss at random-init time
otherwise forces the axis to stretch way past the converged values you
actually care about); every point is still plotted -- see
plot_vs_layers.py if you instead want to scan ALL (prep, post) layer
combinations at once.
"""

# ---- edit these before running ----
TOPOLOGY = 'full'          # star | ring | full
STAGE = 'cfi'                # qfi | cfi
LAYERS_LIST = [1, 2, 3, 4]  # STAGE='qfi': prep layers to compare. STAGE='cfi': POST (measurement) layers to compare
PREP_LAYERS = 3             # only used when STAGE='cfi': the FIXED prep/theta layer count these CFI runs used
POST_TOPOLOGY = None        # only used when STAGE='cfi'; None = same as TOPOLOGY (default/common case)
BURN_IN = 5                 # steps skipped only when auto-scaling the y-axis (see docstring)
# ------------------------------------

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(7, 4.5))
    label_metric = '1/Q' if STAGE == 'qfi' else '1/C'
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(LAYERS_LIST), 2)))

    found_any = False
    ylim_vals = []
    for L, color in zip(LAYERS_LIST, colors):
        if STAGE == 'cfi':
            post_topology = POST_TOPOLOGY or TOPOLOGY
            tag = f'{TOPOLOGY}_L{PREP_LAYERS}_post_{post_topology}_L{L}'
            label = f'post L={L} (prep L={PREP_LAYERS})'
        else:
            tag = f'{TOPOLOGY}_L{L}'
            label = f'L={L}'
        path = os.path.join(DATA_DIR, f'costs_{STAGE}_{tag}.json')
        if not os.path.exists(path):
            print(f'[skip] {path} not found')
            continue
        costs = json.load(open(path))
        ax.plot(range(1, len(costs) + 1), costs, lw=1.6, color=color, label=label)
        print(f'{label}: {len(costs)} steps, final loss={costs[-1]:.5f}, best={min(costs):.5f}, '
              f'range=[{min(costs):.3e}, {max(costs):.3e}]')
        found_any = True
        # skip the first BURN_IN steps when deciding the y-range, so a huge
        # loss at random-init time doesn't compress the converged region
        tail = costs[BURN_IN:] if len(costs) > BURN_IN else costs
        ylim_vals.extend(tail)

    if not found_any:
        print(f'No matching costs_{STAGE}_{TOPOLOGY}_* files found in {DATA_DIR} for layers={LAYERS_LIST}')
        raise SystemExit(1)

    ax.set_xlabel('training step')
    ax.set_ylabel(f'loss ({label_metric})')
    ax.set_yscale('log')
    ax.set_title(f'{TOPOLOGY} - {STAGE.upper()} loss vs iteration')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    # log y-axis, tightly fit to the post-burn-in tail's actual range (small
    # multiplicative margin in log-space, not the old lo*0.5/hi*2 padding) --
    # lines with a huge early value just run off the top instead of forcing
    # the axis to stretch down to cover it
    if ylim_vals:
        lo, hi = min(ylim_vals), max(ylim_vals)
        if lo > 0 and hi > lo:
            margin = (hi / lo) ** 0.1
            ax.set_ylim(lo / margin, hi * margin)

    fig.tight_layout()

    out = os.path.join(FIGURE_DIR, f'loss_{STAGE}_{TOPOLOGY}.png')
    fig.savefig(out, dpi=140)
    print(f'saved {out}')
