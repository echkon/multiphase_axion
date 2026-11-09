import numpy as np

"""
Shared core of the sequential-Bayes recovery algorithm, used by BOTH
bayes_recover.py (single mass, full posterior/bias/var vs. N) and
mass_scan_bayes.py (many masses, one N_TRY, g_hat vs. m_a) -- so the
actual math lives in exactly ONE place instead of being copy-pasted into
both scripts.

LINE-FOR-LINE PORT of the user's own bayesian_inference.py: bayesian_update
normalizes via np.trapz (a density, integral=1), stat_properties normalizes
via np.sum (matching `mean = x @ f / np.sum(f)` exactly). sequential_recover
draws a FRESH, INDEPENDENT batch of N_try outcomes and does the sequential
update prior -> posterior -> prior once per outcome -- computed via the
counts shortcut (mathematically identical to the literal shot-by-shot loop,
since likelihoods of independent shots multiply in any order; stays fast
even at N_try = 1e5-1e12). See bayes_recover.py's module docstring for the
full derivation/justification of the counts shortcut.

Callers are responsible for building delta_grid and p_grid (the
per-delta-grid-point likelihood table, from their own trained circuit) --
this module has no dependency on onecircuit at all, just numpy.
"""


def bayesian_update(prior, delta_grid, likelihood_grid):
    """posterior = likelihood * prior, renormalized (trapz, i.e. a proper
    density: integral over delta_grid = 1)."""
    posterior = likelihood_grid * prior
    posterior /= np.trapz(posterior, delta_grid)
    return posterior


def stat_properties(f, x, true_val):
    """mean/bias/var/std, normalized with np.sum(f) -- matches
    bayesian_inference.py's stat_properties() exactly."""
    mean = x @ f / np.sum(f)
    bias = mean - true_val
    var = (x - mean) ** 2 @ f / np.sum(f)
    std = np.sqrt(var)
    return mean, bias, var, std


def sequential_recover(delta_grid, p_grid, p_true, N_try, delta_true):
    """Draw a FRESH, independent batch of N_try outcomes from p_true, then
    run the sequential Bayes update -- via the counts shortcut. Returns
    (posterior, mean, bias, var, std).

    delta_grid: (GRID_N,) array, the FIXED grid this call uses (caller
        decides -- may be the same grid every call, or rebuilt per call,
        e.g. re-centered on a different delta_true for each mass point).
    p_grid: (GRID_N, n_outcomes) likelihood table, P(k | delta) for every
        delta in delta_grid -- computed ONCE per distinct delta_grid.
    p_true: (n_outcomes,) the TRUE likelihood vector at delta_true, used
        only to draw synthetic measurement outcomes.
    """
    # np.random.multinomial(N_try, p_true) draws the SAME distribution of
    # per-outcome counts as "draw N_try iid categorical samples, then
    # histogram them" (np.random.choice + np.bincount, used previously) --
    # but WITHOUT ever materializing an array of N_try individual samples.
    # That materialization is what broke recover_mass.py's N_try~
    # 3e11-1e12 runs (np.random.choice(..., size=int(N_try), ...) tried to
    # allocate a 2.3 TiB array and crashed) -- multinomial() only ever
    # allocates an array the size of p_true (a handful of outcomes),
    # regardless of how large N_try is, and is verified fast/exact even at
    # N_try=1e12 (see chat).
    counts = np.random.multinomial(int(N_try), p_true)

    prior = np.ones_like(delta_grid)
    prior /= np.trapz(prior, delta_grid)

    # equivalent to: for k in outcomes: prior = bayesian_update(prior, delta_grid, p_grid[:, k])
    loglik_grid = np.log(p_grid + 1e-300) @ counts
    loglik_grid -= loglik_grid.max()
    likelihood_grid = np.exp(loglik_grid)
    posterior = bayesian_update(prior, delta_grid, likelihood_grid)

    mean, bias, var, std = stat_properties(posterior, delta_grid, delta_true)
    return posterior, mean, bias, var, std


def g_stat_properties(delta_grid, posterior, g_agg_from_delta, m_a, true_g):
    """g-space bias/var of the SAME posterior, by weighting
    g(delta) = g_agg_from_delta(|delta|, m_a) under the delta-posterior --
    sum-normalized, same convention as stat_properties()."""
    g_grid = g_agg_from_delta(np.abs(delta_grid), m_a)
    norm = np.sum(posterior)
    mean_g = (g_grid @ posterior) / norm
    bias_g = mean_g - true_g
    var_g = ((g_grid - mean_g) ** 2 @ posterior) / norm
    return mean_g, bias_g, var_g
