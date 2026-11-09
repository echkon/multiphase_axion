import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figure')

results = json.load(open(os.path.join(DATA_DIR, 'bayes_recover_results_full_L3_post_full_L3_polar_1yr_randomalpha.json')))

delta_true = results['delta_true']
r_grid = np.array(results['r_grid'])
N_list = np.array(results['N_list'])
bias_delta = np.array(results['bias_delta_list'])
bias_g = np.array(results['bias_g_list'])
snapshots = results['snapshots']

fig = plt.figure(figsize=(10.5, 5.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.08, wspace=0.28)
ax_post = fig.add_subplot(gs[:, 0])
ax_bias = fig.add_subplot(gs[0, 1])
ax_bias_g = fig.add_subplot(gs[1, 1])

labels = {results['N_MONTH']: '1 month', results['N_6MO']: '6 months', results['N_YEAR']: '1 year'}
palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(snapshots)))
for snap, color in zip(snapshots, palette):
    n = snap['N_try']
    marginal_r = np.array(snap['marginal_r'])
    lbl = labels.get(n, f'n={n:.2e}')
    ax_post.plot(r_grid, marginal_r, color=color, lw=2.0, label=f'{lbl} ($n={n:.3e}$)')
ax_post.axvline(delta_true, color='red', linestyle='--', lw=1.6, label=r'$\delta_{0}$')
ax_post.set_xlim(delta_true - 3e-6, delta_true + 3e-6)
ax_post.set_xlabel(r'$\delta$')
ax_post.set_ylabel('Posterior Probability Density')
ax_post.legend(fontsize=8)
ax_post.grid(alpha=0.3)
ax_post.text(0.03, 0.95, '(a)', transform=ax_post.transAxes, fontsize=13, fontweight='bold', va='top')

ax_bias.plot(N_list, bias_delta, color='tab:blue', lw=1.4, marker='.', ms=3)
ax_bias.axhline(0, color='black', linestyle=':', lw=1.2)
for nn in (results['N_MONTH'], results['N_6MO'], results['N_YEAR']):
    ax_bias.axvline(nn, color='gray', linestyle=':', lw=0.8, alpha=0.6)
ax_bias.set_xscale('log')
ax_bias.set_yscale('symlog', linthresh=1e-8)
ax_bias.set_ylabel(r'$\hat\delta - \delta_{0}$')
ax_bias.tick_params(axis='x', labelbottom=False)
ax_bias.grid(alpha=0.3)
ax_bias.text(0.02, 0.05, '(b)', transform=ax_bias.transAxes, fontsize=13, fontweight='bold', va='bottom')

ax_bias_g.plot(N_list, bias_g, color='tab:orange', lw=1.4, marker='.', ms=3)
ax_bias_g.axhline(0, color='black', linestyle=':', lw=1.2)
for nn in (results['N_MONTH'], results['N_6MO'], results['N_YEAR']):
    ax_bias_g.axvline(nn, color='gray', linestyle=':', lw=0.8, alpha=0.6)
ax_bias_g.set_xscale('log')
ax_bias_g.set_yscale('symlog', linthresh=1e-19)
ax_bias_g.set_xlabel(r'$n$')
ax_bias_g.set_ylabel(r'$\hat g - g_{0}$')
ax_bias_g.grid(alpha=0.3)
ax_bias_g.text(0.02, 0.05, '(c)', transform=ax_bias_g.transAxes, fontsize=13, fontweight='bold', va='bottom')

fig.suptitle(r'random-$\alpha$ 2D (Cartesian) Bayesian recovery: $n=1$..1 year, '
             r'checkpoints at 1 month/6 months/1 year', fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_path = os.path.join(FIGURE_DIR, 'bayes_recover_polar_1yr_randomalpha.eps')
fig.savefig(out_path, dpi=140, bbox_inches='tight')
print(f'saved {out_path}')
png_path = os.path.join(FIGURE_DIR, 'fig3_polar_1yr_preview.png')
fig.savefig(png_path, dpi=140, bbox_inches='tight')
print(f'saved {png_path}')
