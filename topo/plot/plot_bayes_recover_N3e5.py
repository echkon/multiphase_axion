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
Edit the constants below, then run:
    python3 plot_bayes_recover.py

Reads data/bayes_recover_results_{TOPOLOGY}_L{LAYERS}_post_{POST_TOPOLOGY}_L{POST_LAYERS}_N{N_TRY}.json
(written by run/bayes_recover.py) and draws ONE combined 3-panel figure
into figure/bayes_recover_{tag}.png, matching the user's reference style:

  (a) LEFT, big panel: posterior over delta at each N_POST_LIST value
      (all sharing ONE fixed delta_grid -- see bayes_recover.py's
      sequential_recover() docstring for why -- so peak heights are
      directly comparable: narrower peak = taller peak, since each curve
      independently integrates to 1 via np.trapz; this WAS verified
      numerically to be exact, see chat), plus a vertical delta_true line.

  (b) TOP RIGHT: BIAS(delta) vs. N only, semilog-x, linear y, with a
      dotted zero-bias reference line.

  (c) BOTTOM RIGHT: BIAS(g) vs. N only, semilog-x, linear y, with a
      dotted zero-bias reference line. Note g = g_agg_from_delta(|delta|)
      is built from |delta| (always >= 0), so its bias is systematically
      POSITIVE whenever the posterior is still much wider than delta_true
      itself (E[|X|] >= |E[X]| for any distribution) -- this is expected
      physics, not a bug; it only approaches 0 once N is large enough
      that the posterior narrows down to delta_true's own tiny scale.

  Variance is no longer plotted here (dropped per request).

Data and plotting are fully decoupled: run bayes_recover.py ONCE per
(topology, layers, post_layers, N_TRY) config you care about, then run
THIS script as many times as you like (tweaking colors/labels/styling)
without ever re-running the (slow) recovery.
"""

# ---- edit these before running ----
TOPOLOGY = 'full'
LAYERS = 3
POST_TOPOLOGY = None   # None = same as TOPOLOGY (must match what bayes_recover.py used)
POST_LAYERS = 3        # None = same as LAYERS (must match what bayes_recover.py used)
N_TRY = 3e5             # must match the N_TRY bayes_recover.py was run with
IN_JSON = None          # if None, built automatically from the constants above
# ------------------------------------

POST_TOPOLOGY = POST_TOPOLOGY or TOPOLOGY
POST_LAYERS = LAYERS if POST_LAYERS is None else POST_LAYERS


def _fmt_n(n):
    """Same formatter as run/bayes_recover.py: 1e5 -> '1e05', 2.1e5 -> '2p1e05'."""
    return f'{n:.4g}'.replace('+', '').replace('.', 'p')


if __name__ == '__main__':
    # same ALWAYS-both naming as train_cfi.py / run/bayes_recover.py
    tag = f'{TOPOLOGY}_L{LAYERS}_post_{POST_TOPOLOGY}_L{POST_LAYERS}_N{_fmt_n(N_TRY)}'
    in_path = IN_JSON or os.path.join(DATA_DIR, f'bayes_recover_results_{tag}.json')
    results = json.load(open(in_path))

    delta_true = results['delta_true']
    N_list = np.array(results['N_list'])
    bias_list = np.array(results['bias_list'])
    g_bias_list = np.array(results['g_bias_list'])

    fig = plt.figure(figsize=(11.5, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.08)
    ax_post = fig.add_subplot(gs[:, 0])
    ax_bias = fig.add_subplot(gs[0, 1])
    ax_bias_g = fig.add_subplot(gs[1, 1])

    # ---- (a) posterior over delta, shared grid across N_POST_LIST ----
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(results['posteriors'])))
    for entry, color in zip(results['posteriors'], palette):
        N_try = entry['N_try']
        delta_grid = np.array(entry['delta_grid'])
        post = np.array(entry['posterior'])
        # normalization check, printed so you can verify it yourself: this
        # MUST equal 1.0 (it's a density -- integral over delta_grid = 1,
        # NOT a bounded-by-1 "probability" -- a narrow peak legitimately
        # has a LARGE density value while still integrating to 1, same as
        # a narrow Gaussian N(0, 0.001) has a huge peak height)
        norm_check = np.trapz(post, delta_grid)
        print(f'  nu={N_try:.0e}: trapz(posterior, delta_grid) = {norm_check:.10f}  (should be 1.0)')
        ax_post.plot(delta_grid, post, color=color, lw=2.0, label=fr'$\nu = {N_try:.0e}$')
    ax_post.axvline(delta_true, color='red', linestyle='--', lw=1.6, label=r'$\delta_{\rm true}$')
    ax_post.set_xlabel(r'$\delta$')
    ax_post.set_ylabel('Posterior Probability Density')
    ax_post.legend(fontsize=9)
    ax_post.grid(alpha=0.3)
    ax_post.text(0.03, 0.95, '(a)', transform=ax_post.transAxes, fontsize=13, fontweight='bold', va='top')

    # ---- (b) bias(delta) vs. N ----
    ax_bias.plot(N_list, bias_list, color='tab:blue', lw=1.8, label=r'Bias $\delta$')
    ax_bias.axhline(0, color='black', linestyle=':', lw=1.2)
    ax_bias.set_xscale('log')
    ax_bias.set_ylabel(r'Bias $\bar\delta - \delta_{\rm true}$')
    ax_bias.tick_params(axis='x', labelbottom=False)
    ax_bias.grid(alpha=0.3)
    ax_bias.legend(fontsize=8, loc='upper right')
    ax_bias.text(0.02, 0.05, '(b)', transform=ax_bias.transAxes, fontsize=13, fontweight='bold', va='bottom')

    # ---- (c) bias(g) vs. N ----
    ax_bias_g.plot(N_list, g_bias_list, color='tab:orange', lw=1.8, label=r'Bias $g$')
    ax_bias_g.axhline(0, color='black', linestyle=':', lw=1.2)
    ax_bias_g.set_xscale('log')
    ax_bias_g.set_xlabel(r'$\nu$')
    ax_bias_g.set_ylabel(r'Bias $\bar g - g_{\rm true}$')
    ax_bias_g.grid(alpha=0.3)
    ax_bias_g.legend(fontsize=8, loc='upper right')
    ax_bias_g.text(0.02, 0.05, '(c)', transform=ax_bias_g.transAxes, fontsize=13, fontweight='bold', va='bottom')

    fig.suptitle(f'{tag}: sequential Bayes recovery', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(FIGURE_DIR, f'bayes_recover_{tag}.eps')
    fig.savefig(out_path, dpi=140)
    print(f'saved {out_path}')
