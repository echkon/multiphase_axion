"""
Axion-photon coupling -> single-qubit encoding angle delta(g_agg, m_a),
following the transmon-capacitor E-field mechanism of Chen et al.,
PRD 110, 115021 (2024) [arXiv:2409.19461], Eqs. (2.6)-(3.15).

delta == eta*tau, the (small) rotation angle imprinted on ONE qubit during
one interrogation window tau. This is exactly the object that plays the
role of the physical parameter to be estimated in the VQM/QFIM-CFIM
framework (Santoso & Ho, PRD 112, L081301) -- it does not matter that it's
now sourced by axion-photon coupling instead of whatever coupling was used
before: QFI only cares about (N, network ansatz, decoherence sigma), not
the numerical value of delta itself.

We reuse Chen et al.'s own normalized Eq. (3.15) rather than re-deriving
the natural-units prefactors from scratch, since it is already numerically
validated in their paper:

p_ge^(1) = eta^2 tau^2
         = 0.11 * (g_agg/1e-10 GeV^-1)^2 * (m_a/1ueV)^-1
                * (B0/1T)^2 * (tau/100us)^2 * kappa^2
                * (C/0.1pF) * (d/100um)^2 * (rho_DM/0.45 GeV/cm^3)

delta = sqrt(p_ge^(1)) = eta*tau
"""

import numpy as np


def tau_DM_us(m_a_ueV, v_DM=1e-3):
    """DM coherence time in microseconds, tau_DM ~ 1/(m_a v_DM^2).
    Chen et al. Eq. (2.9): tau_DM ~ 100 us * (10 ueV / m_a[ueV])
    => tau_DM[us] = 1000 / m_a[ueV]  (for the fiducial v_DM=1e-3 they use).
    """
    return 100.0 * (10.0 / m_a_ueV) * (1e-3 / v_DM) ** 2


def delta_axion(g_agg, m_a_ueV, tau_us=None, T2_qubit_us=100.0,
                 B0_T=5.0, kappa=1.0, C_pF=0.1, d_um=100.0,
                 rho_DM=0.45):
    """Per-qubit encoding rotation angle delta = eta*tau for a trial
    axion-photon coupling g_agg [GeV^-1] at mass m_a_ueV [ueV].

    If tau_us is None, use tau = min(tau_DM(m_a), T2_qubit_us) -- the
    interrogation window is capped by the qubit coherence time, not just
    by tau_DM. This matters a lot at low mass, where tau_DM(m_a) grows
    without bound (tau_DM ~ 1000us/m_a[ueV]) and would otherwise blow past
    any realistic qubit T2 -- and past the small-angle (delta<<1)
    perturbative regime the whole eta*tau formula assumes.
    """
    if tau_us is None:
        tau_us = min(tau_DM_us(m_a_ueV), T2_qubit_us)

    p_ge1 = (0.11
             * (g_agg / 1e-10) ** 2
             * (m_a_ueV / 1.0) ** -1
             * (B0_T / 1.0) ** 2
             * (tau_us / 100.0) ** 2
             * kappa ** 2
             * (C_pF / 0.1)
             * (d_um / 100.0) ** 2
             * (rho_DM / 0.45))
    return np.sqrt(p_ge1), tau_us


def g_agg_from_delta(delta, m_a_ueV, tau_us=None, T2_qubit_us=100.0,
                      B0_T=5.0, kappa=1.0, C_pF=0.1, d_um=100.0,
                      rho_DM=0.45):
    """Invert: given an achieved (minimum-resolvable) delta at mass m_a,
    solve for the corresponding g_agg.

    BUGFIX: this used to default tau_us to tau_DM_us(m_a_ueV) WITHOUT the
    T2 cap applied in delta_axion(), silently breaking the round-trip
    delta_axion <-> g_agg_from_delta inverse relationship (off by a factor
    of tau_DM/T2 in p_ge1, i.e. sqrt(tau_DM/T2) in g_agg) whenever
    tau_DM(m_a) > T2 and tau_us was not passed explicitly. Now mirrors
    delta_axion's min(tau_DM, T2) capping exactly.
    """
    if tau_us is None:
        tau_us = min(tau_DM_us(m_a_ueV), T2_qubit_us)
    p_ge1 = delta ** 2
    factor = (0.11
              * (m_a_ueV / 1.0) ** -1
              * (B0_T / 1.0) ** 2
              * (tau_us / 100.0) ** 2
              * kappa ** 2
              * (C_pF / 0.1)
              * (d_um / 100.0) ** 2
              * (rho_DM / 0.45))
    g_agg2_over_1e10sq = p_ge1 / factor
    return 1e-10 * np.sqrt(g_agg2_over_1e10sq)


if __name__ == "__main__":
    # sanity check against Chen et al.'s own worked numbers:
    # at m_a=1ueV, g_agg=1e-10 GeV^-1, B0=1T, tau=100us, kappa=1,
    # C=0.1pF, d=100um, rho_DM=0.45 -> p_ge1 = 0.11 exactly (by construction)
    delta, tau = delta_axion(g_agg=1e-10, m_a_ueV=1.0, tau_us=100.0,
                              B0_T=1.0, kappa=1.0)
    print(f"check: delta={delta:.6f}, delta^2={delta**2:.6f} (expect 0.11)")

    # round trip
    g_back = g_agg_from_delta(delta, m_a_ueV=1.0, tau_us=100.0, B0_T=1.0, kappa=1.0)
    print(f"round-trip g_agg = {g_back:.4e} (expect 1e-10)")

    # mass scan example: tau=min(tau_DM(m_a),T2=100us), B0=5T, kappa=1, g_agg=1e-11
    for m_a_ueV in [0.01, 0.1, 1, 10, 100]:
        d, tau = delta_axion(g_agg=1e-11, m_a_ueV=m_a_ueV)
        print(f"m_a={m_a_ueV:>6.2f} ueV: tau={tau:>8.2f} us "
              f"(tau_DM={tau_DM_us(m_a_ueV):>10.1f} us), delta={d:.3e}")
