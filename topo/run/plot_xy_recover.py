import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figure')

results = json.load(open(os.path.join(DATA_DIR, 'bayes_recover_results_full_L3_post_full_L3_xy_randomalpha.json')))

delta_true = results['delta_true']
eig_min = 60.19909415
N_list = np.array(results['N_list'])
hat_delta = np.array(results['hat_delta_list'])
bias_g = np.array(results['bias_g_list'])
snapshots = results['snapshots']
axis_grid = np.array(results['axis_grid'])

sigma_theory = 1.0 / np.sqrt(N_list * eig_min)
rice_floor = sigma_theory * np.sqrt(np.pi / 2)  # Rician-mean floor when delta_true << sigma

fig = plt.figure(figsize=(12, 7.5))
gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.32)
ax_delta = fig.add_subplot(gs[0, 0])
ax_g = fig.add_subplot(gs[1, 0])
ax_snap = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 1])]

ax_delta.loglog(N_list, hat_delta, 'o-', color='tab:blue', ms=4, label=r'$\hat\delta=\sqrt{\hat x^2+\hat y^2}$')
ax_delta.loglog(N_list, rice_floor, '--', color='gray', label=r'$\sigma(n)\sqrt{\pi/2}$ (Rician floor)')
ax_delta.axhline(delta_true, color='red', linestyle=':', label=r'$\delta_0$')
ax_delta.set_xlabel(r'$n$')
ax_delta.set_ylabel(r'$\hat\delta$')
ax_delta.legend(fontsize=8)
ax_delta.grid(alpha=0.3, which='both')
ax_delta.text(0.03, 0.95, '(a)', transform=ax_delta.transAxes, fontsize=12, fontweight='bold', va='top')

ax_g.set_xscale('log')
ax_g.set_yscale('log')
ax_g.plot(N_list, np.abs(bias_g), 'o-', color='tab:orange', ms=4)
ax_g.set_xlabel(r'$n$')
ax_g.set_ylabel(r'$|\hat g - g_0|~[{\rm GeV^{-1}}]$')
ax_g.grid(alpha=0.3, which='both')
ax_g.text(0.03, 0.95, '(b)', transform=ax_g.transAxes, fontsize=12, fontweight='bold', va='top')

labels = ['(c)', '(d)', '(e)']
for ax, snap, lbl in zip(ax_snap, snapshots, labels):
    n = snap['N_try']
    post = np.array(snap['posterior'])
    x0, y0 = snap['x0'], snap['y0']
    hx, hy = snap['hat_x'], snap['hat_y']
    win = 6 * max(1.0 / np.sqrt(n * eig_min), abs(x0), abs(y0), 1e-6)
    mask = (np.abs(axis_grid) <= win)
    idx = np.where(mask)[0]
    sub = post[np.ix_(idx, idx)]
    extent = [axis_grid[idx[0]], axis_grid[idx[-1]], axis_grid[idx[0]], axis_grid[idx[-1]]]
    im = ax.imshow(sub.T, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    ax.plot(x0, y0, marker='*', color='red', ms=12, label=r'$(x_0,y_0)$')
    ax.plot(hx, hy, marker='x', color='white', ms=9, mew=2, label=r'$(\hat x,\hat y)$')
    ax.plot(0, 0, marker='+', color='cyan', ms=8)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title(f'$n={n:.2e}$', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.text(0.03, 0.93, lbl, transform=ax.transAxes, fontsize=12, fontweight='bold',
            va='top', color='white')

fig.suptitle(r'2D Bayesian recovery with per-$n$ random dark-matter phase $\alpha$ '
              r'($\delta_0$ fixed, $(x_0,y_0)=\delta_0(\cos\alpha,\sin\alpha)$)', fontsize=10)

out_path = os.path.join(FIGURE_DIR, 'bayes_recover_xy_randomalpha.eps')
fig.savefig(out_path, dpi=140, bbox_inches='tight')
print(f'saved {out_path}')
png_path = os.path.join(FIGURE_DIR, 'fig3_xy_randomalpha_preview.png')
fig.savefig(png_path, dpi=140, bbox_inches='tight')
print(f'saved {png_path}')
