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
NEW, STANDALONE script -- does NOT modify plot_mass_scan_bayes.py (or any
run/*.py). Edit the constants below, then run:
    python3 plot_recover_mass.py

Draws KSVZ/DFSZ, ADMX/RBF+UF haloscopes, and CAST/HB-AGB astro bounds,
PLUS overlays every data/targeted_mass_bayes_results_{tag}_*.json file it
finds (written by run/recover_mass.py, one file per single-mass
targeted run), grouped by (MODEL, N_years): color = model (green=KSVZ,
purple=DFSZ target), and alpha/marker-fill = exposure duration (longer
exposure = darker/filled, shorter = lighter/hollow-ish), since different
N_years runs at the SAME masses land at nearly the SAME (x, y) position
(0.5yr vs. 1yr only differs by ~1.4x in error-bar width -- see chat) and
would otherwise silently overlap/hide each other if drawn identically.
recover_mass.py's MODEL constant selects KSVZ vs. DFSZ, tagged in
both the filename and the saved JSON's 'model' field (older files saved
before MODEL existed have no 'model' field and are treated as KSVZ, since
that was the only option before). Missing/not-yet-run (mass, model,
N_years) combinations are simply absent from the plot -- no need to wait
for a whole batch to finish; re-run this as many times as you like as
more targeted runs complete.

Does NOT plot the broadband mass_scan_bayes.py empirical points, and does
NOT plot the eig_min(C)-based 1yr/5yr theory-reach curves -- both of
those were built on a "divide 1 year across ~1.5e6 frequency bins"
premise that does not match how you actually want mass_scan_bayes.py's
10 illustrative m_a points to be interpreted (you scan a small, FIXED set
of discrete masses, each with its own dedicated Bayes recovery -- exactly
what recover_mass.py does -- not a continuous dense frequency
scan). Removed per explicit request rather than left in as a
now-misleading reference curve.
"""

# ---- edit these before running ----
TOPOLOGY = 'full'
LAYERS = 3
POST_TOPOLOGY = None       # None = same as TOPOLOGY
POST_LAYERS = 3            # None = same as LAYERS
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
    (6.6e-11, 'mediumpurple', 'mediumpurple', (0, (5, 1)), 'CAST helioscope', 0.7, 'bottom'),
    (0.47e-10, 'mediumvioletred', 'mediumvioletred', (0, (1, 1)), 'HB/AGB stars (globular clusters)', 0.55, 'top'),
]

# closed-form KSVZ/DFSZ, computed directly here (NOT read from any run/*.py
# output file) -- same formula/constants as mass_scan_bayes.py/recover_mass.py
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

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    # ---- overlay every targeted-mass single-point result found, grouped by
    # ---- (MODEL, N_years) -- see docstring for why N_years matters here ----
    targeted_glob = os.path.join(DATA_DIR, f'targeted_mass_bayes_results_{tag}_*.json')
    targeted_paths = sorted(glob.glob(targeted_glob))
    if not targeted_paths:
        raise SystemExit(f'No targeted_mass_bayes_results_{tag}_*.json files found in {DATA_DIR} -- run run/recover_mass.py first')

    by_group = {}  # (model, N_years) -> {'mass': [], 'mean': [], 'std': [], 'ntry': []}
    n_skipped_legacy = 0
    for p in targeted_paths:
        r = json.load(open(p))
        if 'model' not in r:
            # Old files saved before MODEL existed -- these duplicate the SAME
            # (mass, N_years) points as newer, explicitly-tagged KSVZ runs
            # (same topology/layers/exposure), so overlaying them just draws
            # a second, stale marker on top of the current KSVZ result.
            # Skip them entirely rather than silently treating them as KSVZ.
            n_skipped_legacy += 1
            continue
        model = r['model']
        n_years = round(r.get('N_years', 1.0), 6)
        g = by_group.setdefault((model, n_years), {'mass': [], 'mean': [], 'std': [], 'ntry': []})
        g['mass'].append(r['mass_ueV'])
        g['mean'].append(r['g_hat_mean'])
        g['std'].append(r['g_hat_std'])
        g['ntry'].append(r['N_TRY'])

    # Within each (model, N_years) group, some old runs used a slightly
    # SHIFTED m_a grid (~5% higher than the canonical grid) as a manual
    # trick to keep 1mo/1yr points from landing exactly on top of each
    # other on the plot. Now that duration is distinguished by marker
    # shape/color instead, that shift is no longer needed and just leaves
    # near-duplicate pairs of points (same group, ~5% apart in mass) that
    # look like 2 markers for the "same" m_a. The SHIFTED (higher-mass)
    # member of each pair is the newer run -- keep that one and drop the
    # older, unshifted (lower-mass) member.
    NEAR_DUP_REL_TOL = 0.08
    n_dropped_shifted = 0
    for key, g in by_group.items():
        order = np.argsort(g['mass'])
        mass = np.array(g['mass'])[order]
        mean = np.array(g['mean'])[order]
        std = np.array(g['std'])[order]
        ntry = np.array(g['ntry'])[order]
        keep = [0]
        for i in range(1, len(mass)):
            if mass[i] <= mass[keep[-1]] * (1 + NEAR_DUP_REL_TOL):
                # current point is within tolerance of the last kept one --
                # it's the newer (shifted) run, so replace the kept index
                keep[-1] = i
                n_dropped_shifted += 1
                continue
            keep.append(i)
        keep = np.array(keep)
        by_group[key] = {'mass': mass[keep].tolist(), 'mean': mean[keep].tolist(),
                          'std': std[keep].tolist(), 'ntry': ntry[keep].tolist()}

    print(f'found {len(by_group)} (model, N_years) group(s): '
          f'{sorted(by_group.keys(), key=lambda k: (k[0], k[1]))}'
          f' -- skipped {n_skipped_legacy} legacy (no "model" field) file(s), '
          f'dropped {n_dropped_shifted} shifted near-duplicate point(s)')

    # Every (model, duration) group gets its OWN maximally-distinct color
    # (not just 2 colors shared across durations) AND its own marker shape,
    # drawn OPEN/hollow (mfc='none') so overlapping points stay readable --
    # avoids relying on alpha shading, which looks identical once
    # rasterized at EPS-export darkness.
    GROUP_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                     'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    DURATION_MARKERS = ['D', 'o', 's', '^', 'P', 'X']
    years_sorted = sorted({k[1] for k in by_group}, reverse=True)
    marker_for_years = {y: DURATION_MARKERS[i % len(DURATION_MARKERS)] for i, y in enumerate(years_sorted)}

    # Legend/plot order: KSVZ before DFSZ, and within each model, shorter
    # exposure (1mo) before longer (1yr) -- i.e. KSVZ-1mo, KSVZ-1yr,
    # DFSZ-1mo, DFSZ-1yr.
    MODEL_ORDER = {'KSVZ': 0, 'DFSZ': 1}
    sorted_groups = sorted(by_group.items(), key=lambda kv: (MODEL_ORDER.get(kv[0][0], 99), kv[0][1]))
    color_for_group = {key: GROUP_COLORS[i % len(GROUP_COLORS)] for i, (key, _) in enumerate(sorted_groups)}

    MS, MEW = 9, 1.7  # marker size / edge width -- shared by main plot and inset

    m_a_all = []
    plotted_groups = []  # (model, n_years, mass, mean, std, marker, color) -- reused by the inset below
    for (model, n_years), d in sorted_groups:
        mass = np.array(d['mass']); mean = np.array(d['mean'])
        std = np.array(d['std']); ntry = np.array(d['ntry'])
        order = np.argsort(mass)
        mass, mean, std, ntry = mass[order], mean[order], std[order], ntry[order]
        m_a_all.append(mass)
        marker = marker_for_years[n_years]
        color = color_for_group[(model, n_years)]
        dur_label = f'{n_years*12:.2g}mo' if n_years < 1 else f'{n_years:.2g}yr'
        ax.errorbar(mass * 1e-6, mean, yerr=std, fmt=marker, mfc='none', mec=color, ecolor=color,
                    ms=MS, mew=MEW, capsize=3, zorder=5,
                    label=rf'{model}, {dur_label} ($\nu\sim{ntry.mean():.1e}$)')
        plotted_groups.append((model, n_years, mass, mean, std, marker, color))
        print(f'overlaid {len(mass)} {model}-target, {dur_label} point(s): {sorted(mass.tolist())} ueV')

    t_mass = np.concatenate(m_a_all)

    # KSVZ/DFSZ -- always computed directly here, never depends on any
    # run/*.py output file
    m_dense = np.logspace(np.log10(0.1), np.log10(10), 200)
    ax.plot(m_dense * 1e-6, g_KSVZ(m_dense), 'k:', lw=1.6, label='KSVZ')
    ax.plot(m_dense * 1e-6, g_DFSZ(m_dense), 'k-.', lw=1.6, label='DFSZ')

    # Same EPS-export bug as the bar charts in plot_vs_layers.py: filling up
    # to y=1.0 puts the fill's top edge at log10(1.0)=0, far above the
    # visible axis top (YLIM[1]=1e-9, i.e. log10=-9). The PNG/Agg backend
    # silently clips this at the frame, but the EPS (vector) backend emits
    # the TRUE, unclipped path all the way up to y=1.0 -- invisible in the
    # normal preview, but a huge overshooting bar in Illustrator. Fill up
    # to the axis top instead of 1.0.
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

    # Small log-scale margin so edge markers (m_a = 0.1 ueV ~ 1e-7 eV on the
    # left, 10 ueV ~ 1e-5 eV on the right) aren't clipped by the axis frame.
    X_MARGIN_FACTOR = 1.25
    x_lo = float(t_mass.min()) * 1e-6 / X_MARGIN_FACTOR
    x_hi = float(t_mass.max()) * 1e-6 * X_MARGIN_FACTOR
    if SHOW_ASTRO:
        # Fill each bound only up to the NEXT bound above it (stacked,
        # non-overlapping bands) instead of all the way to the axis top --
        # CAST and HB/AGB sit close together on this log scale, so filling
        # both to FILL_TOP used to stack into one hard-to-read solid block.
        # Now HB/AGB (the lower bound) only shades the thin sliver up to
        # where CAST begins, and CAST shades from there up to the top.
        astro_sorted = sorted(ASTRO_BOUNDS, key=lambda a: a[0])
        for i, (g_bound, line_color, fill_color, ls, label, tx, va) in enumerate(astro_sorted):
            fill_top_i = astro_sorted[i + 1][0] if i + 1 < len(astro_sorted) else FILL_TOP
            ax.axhline(g_bound, color=line_color, lw=1.4, ls=ls, zorder=1)
            ax.fill_between([x_lo, x_hi], [g_bound, g_bound], [fill_top_i, fill_top_i],
                             color=fill_color, alpha=0.15, zorder=0)
            y_text = g_bound * (1.5 if va == 'bottom' else 0.55)
            ax.text(tx * 1e-6, y_text, label, fontsize=8, color=line_color, va=va, ha='center')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(x_lo, x_hi)
    if YLIM is not None:
        ax.set_ylim(*YLIM)
    ax.set_xlabel('axion mass m_a (eV)')
    ax.set_ylabel(r'$g_{a\gamma\gamma}$  [GeV$^{-1}$]')
    ax.set_title(f'{tag}: targeted-mass Bayesian recovery vs.\nhaloscope and astrophysical constraints', fontsize=10)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3, which='major')

    # secondary x-axis on top: Compton frequency f = m_a/h corresponding to
    # the mass shown on the bottom axis
    secax = ax.secondary_xaxis('top', functions=(mass_eV_to_freq_Hz, freq_Hz_to_mass_eV))
    secax.set_xlabel('frequency f (Hz)')

    # ---- inset: zoom around m_a ~ 1e-6 eV (1 ueV) -- 2 masses below and 1
    # ---- above, y-axis restricted to just the KSVZ/DFSZ band so the model
    # ---- separation and the marker/color identities are clearly visible.
    # ---- Each (model, duration) group has its OWN m_a grid (durations are
    # ---- shifted ~5% apart from each other), so the "2 below + 1 above"
    # ---- selection is made PER GROUP -- picking one common set of mass
    # ---- values across groups would silently drop whichever groups don't
    # ---- happen to land on those exact values (this is why the 1-month
    # ---- groups were missing from the inset before). The inset x-range is
    # ---- then the union of every group's own selected masses, so all
    # ---- durations end up visible together.
    # ---- Placed in the empty region between the astro bounds (top) and
    # ---- the KSVZ/DFSZ lines (bottom-left) -- move INSET_BBOX to taste. ----
    inset_sel_per_group = []  # (model, n_years, mass, mean, std, marker, color, sel_mask)
    inset_masses_all = []
    for model, n_years, mass, mean, std, marker, color in plotted_groups:
        masses_g = sorted(set(np.round(mass, 6).tolist()))
        below = sorted([m for m in masses_g if m < 1.0], key=lambda m: abs(m - 1.0))[:2]
        above = sorted([m for m in masses_g if m >= 1.0], key=lambda m: abs(m - 1.0))[:1]
        group_inset_masses = below + above
        sel = np.isin(np.round(mass, 6), group_inset_masses)
        inset_sel_per_group.append((model, n_years, mass, mean, std, marker, color, sel))
        inset_masses_all.extend(group_inset_masses)

    if len(set(np.round(inset_masses_all, 6).tolist())) >= 2:
        inset_masses_all = sorted(inset_masses_all)
        INSET_BBOX = (0.14, 0.40, 0.34, 0.30)  # (x0, y0, width, height), axes-fraction
        axins = ax.inset_axes(INSET_BBOX)

        m_lo, m_hi = inset_masses_all[0] * 0.7, inset_masses_all[-1] * 1.4
        m_dense_in = np.logspace(np.log10(m_lo), np.log10(m_hi), 200)
        axins.plot(m_dense_in * 1e-6, g_KSVZ(m_dense_in), 'k:', lw=1.4)
        axins.plot(m_dense_in * 1e-6, g_DFSZ(m_dense_in), 'k-.', lw=1.4)

        y_vals_in = list(g_KSVZ(np.array(inset_masses_all))) + list(g_DFSZ(np.array(inset_masses_all)))
        for model, n_years, mass, mean, std, marker, color, sel in inset_sel_per_group:
            if sel.any():
                axins.errorbar(mass[sel] * 1e-6, mean[sel], yerr=std[sel], fmt=marker,
                                mfc='none', mec=color, ecolor=color, ms=MS, mew=MEW,
                                capsize=3, zorder=5)
                y_vals_in += list(mean[sel] - std[sel]) + list(mean[sel] + std[sel])

        axins.set_xscale('log'); axins.set_yscale('log')
        axins.set_xlim(m_lo * 1e-6, m_hi * 1e-6)
        y_vals_in = np.array(y_vals_in)
        y_vals_in = y_vals_in[np.isfinite(y_vals_in) & (y_vals_in > 0)]
        if len(y_vals_in):
            axins.set_ylim(y_vals_in.min() * 0.5, y_vals_in.max() * 2.0)
        axins.tick_params(labelsize=7)
        axins.grid(alpha=0.3, which='major')
        ax.indicate_inset_zoom(axins, edgecolor='black')
        print(f'[inset] zoomed on m_a = {sorted(set(np.round(inset_masses_all, 6).tolist()))} ueV (per-group selection)')
    else:
        print('[inset] fewer than 2 distinct masses found near 1 ueV -- inset skipped')

    fig.tight_layout()

    out_path = os.path.join(FIGURE_DIR, f'targeted_mass_bayes_{tag}.eps')
    fig.savefig(out_path, dpi=140)
    print(f'saved {out_path}')
