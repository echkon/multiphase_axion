import sys, os, time, pickle, json, copy
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'axion_reach_results'))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
import numpy as np
import qiskit
import onecircuit
from onecircuit import *
from onecircuit.util import step_size
from onecircuit.optimizing import gradient_func
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error
from axion_coupling import delta_axion, g_agg_from_delta, tau_DM_us
from bayes_core import sequential_recover as _core_sequential_recover

"""
NEW, STANDALONE script -- does NOT modify train_qfi.py, train_cfi.py,
recover_mass.py, or anything in onecircuit/.
Edit the constants below, then run:
    python3 recover_mass_noise.py

Single-mass Bayesian recovery under a REAL qiskit-Aer NOISE MODEL for the
qubit NETWORK (verified against this environment's actual installed
qiskit==1.4.6 / qiskit_aer==0.17.2 API -- NoiseModel, thermal_relaxation_
error, depolarizing_error all confirmed present and, for the
density_matrix simulation method used here, EXACT/deterministic -- not
Monte Carlo shot noise; cross-checked to agree with manual Kraus
application to machine precision). Two physically distinct noise sources:

  1. IDLE T1+T2 RELAXATION during the sensing window --
     qiskit_aer.noise.thermal_relaxation_error(T1_US, T2_US, tau_us),
     the standard superconducting-qubit relaxation channel (energy decay
     AND dephasing together). Applied independently to each of the N
     qubits via a placeholder `id` gate per qubit (its own pipeline
     stage: [prep, sens, idle, post]).
  2. NETWORK (gate-level, topology-dependent) noise on every CZ gate of
     the state-prep AND measurement ansatz, restricted to exactly the
     qubit PAIRS that topology's ansatz actually entangles (see
     topology_edges() below). Modeled as depolarizing_error(
     CZ_ERROR_PROB, 2) per edge (default 0.005 = 0.5% per CZ).

PLUS two independent OPTIONS to re-optimize the ansatz under this noise
before recovering:

    OPTIMIZE_QFI = True/False  -- re-optimize the state-prep ansatz
        (theta) to minimize Tr(Q_xy^-1) computed on the NOISY circuit
        (i.e. train_qfi.py's objective, but with the qubit-network noise
        model folded into the cost), instead of reusing the existing
        noiseless-trained theta as-is.
    OPTIMIZE_CFI = True/False  -- same idea for the measurement ansatz
        (mu), minimizing Tr(C_xy^-1) on the noisy circuit (train_cfi.py's
        objective under noise).

Both default to False: the noiseless-optimal (theta, mu) from
ckpt_cfi_best_{tag}.pkl is usually already a reasonable choice even under
noise, and re-optimizing is expensive (see PERFORMANCE below) -- turn
either flag on only when you actually want to check whether a
noise-aware ansatz beats the noiseless-optimal one at this specific
(T1, T2, CZ_ERROR_PROB). With both flags False, this reduces to a plain
noisy Bayesian recovery using the checkpoints as-is.

WHY THIS NEEDS ITS OWN QFI/CFI IMPLEMENTATIONS (can't call
onecircuit.metric.sld_qfim/cfim directly): those evolve circuits via
exact qiskit.quantum_info.DensityMatrix, which cannot express "attach
this depolarizing error to CZ gates on qubits (a,b) only" -- that is
specifically a qiskit_aer.noise.NoiseModel + AerSimulator feature
(confirmed exact/deterministic: two runs of the identical circuit/
noise_model produce bit-identical density matrices; also cross-checked
against manual Kraus application for a simple 1-qubit case, max abs diff
~2e-16). So this script defines its own rho_from_stages/probs_from_stages
(AerSimulator-backed), grad_rho_network + sld_qfim_network (a plain-numpy
port of onecircuit.metric.sld_qfim_eigens's closed-form SLD formula --
same double sum over eigenpairs, 2*Re[<i|drho_k|j><j|drho_l|i>]/(lambda_i+
lambda_j), just via np.linalg.eigh instead of tqix's eigenx/dotx/daggx,
so it needs no tqix-specific behavior), and cfim_network (a small,
self-contained, LINE-FOR-LINE port of onecircuit.metric.cfim()'s math).
onecircuit.optimizing.gradient_func IS reused unchanged -- it is a fully
generic finite-difference gradient over cost_func(circuit), agnostic to
what cost_func does internally, so no duplication needed there.

TRAINING DESIGN:
  - QFI circuit = [prep, sens(fixed x0,y0), idle] -- NO post/measurement
    stage. QFI is the POVM-independent bound; the network's noisy CZ
    gates inside `prep` and the idle T1/T2 relaxation after `sens` both
    count toward it, but a `post` stage must NOT be included (unlike
    train_qfi.py's noiseless circuit, which harmlessly included an
    inert `post` since a UNITARY after the point of interest cannot
    change SLD-QFI -- but a NOISY channel after that point strictly can
    only ever DECREASE it, so including noisy `post` here would
    understate the true achievable bound).
  - CFI circuit = [prep(fixed), sens(fixed x0,y0), idle, post(trained)]
    -- the same 4-stage pipeline described above.
  - Both start (when OPTIMIZE_*=True and no network-noise checkpoint
    exists yet) as a WARM START from the existing noiseless-optimal
    theta/mu in ckpt_cfi_best_{tag}.pkl -- adapting a known-good solution
    to noise, not training from random scratch. Resumable exactly like
    train_qfi.py/train_cfi.py: re-running this script with OPTIMIZE_QFI/
    OPTIMIZE_CFI=True picks up from its own checkpoint
    (ckpt_qfi_best_network_noise_..., ckpt_cfi_best_network_noise_...)
    and keeps improving it; each noise combination (T1, T2,
    CZ_ERROR_PROB) gets its own checkpoint file (never overwrites the
    noiseless baseline, or a different noise strength's result).
  - Both use gradient_func(..., method='two_point') -- matching
    axion_reach_results/train_axion_local.py's choice for noisy training
    (train_qfi.py/train_cfi.py use 'psr' for the noiseless case).

PERFORMANCE: each AerSimulator call costs ~5-10 ms. One QFI/CFI cost
evaluation needs 5 calls (1 rho/prob + 2 params x 2 shifts); one
gradient_func step over ~180 params ('full', L=3) needs ~360 cost
evaluations => roughly 15-20s PER GRADIENT STEP for either QFI or CFI
training. Default QFI_STEPS_PER_CALL=CFI_STEPS_PER_CALL=4 (matching
train_qfi.py's "~4 for full" note) => roughly 1-2 minutes per training
stage, plus roughly 30-60s for the final Bayes grid build (an
AerSimulator call is ~5-10x slower than the exact statevector path used
by the noiseless/dephasing-only scripts) -- a few minutes total per
(mass, model) run when both flags are on; near-instant when both are off
(falls straight through to loading ckpt_cfi_best_{tag}.pkl).

Saves the FINAL Bayes-recovery result to
data/targeted_mass_bayes_network_noise_optimized_results_{tag}_{MODEL}_m{MASS_UEV}ueV_N{N_TRY}_T1{T1_US}_T2{T2_US}_czerr{CZ_ERROR_PROB}_qfi{0|1}_cfi{0|1}.json
-- OPTIMIZE_QFI/OPTIMIZE_CFI are baked into the filename too, so
comparing "optimized under noise" vs. "noiseless ansatz reused as-is"
never collides.
"""

ansatz = {
    'star': (stargraph, stargraph_inv),
    'ring': (ringgraph, ringgraph_inv),
    'full': (fullgraph, fullgraph_inv),
}

# ---- edit these before running ----
TOPOLOGY = 'ring'                    # star | ring | full
LAYERS = 1                           # must match the LAYERS used to train the checkpoint
POST_TOPOLOGY = None                 # measurement ansatz's topology; None = same as TOPOLOGY
POST_LAYERS = 2                      # measurement ansatz's layer count; None = same as LAYERS
MASS_UEV = 1.0                       # <-- set the ONE target mass [ueV] here before each run
MODEL = 'KSVZ'                       # 'KSVZ' or 'DFSZ'
N_YEARS = 1.0                        # total exposure dedicated to this single mass, in years
T1_US = 100.0                        # per-qubit energy-relaxation time [us] (idle noise)
T2_US = 100.0                        # per-qubit dephasing time [us] (idle noise); must have T2 <= 2*T1
CZ_ERROR_PROB = 0.005                # depolarizing-error probability per CZ gate (network/gate noise)
OPTIMIZE_QFI = False                 # re-optimize state-prep (theta) under noise before recovering?
OPTIMIZE_CFI = False                 # re-optimize measurement ansatz (mu) under noise?
QFI_STEPS_PER_CALL = 4               # gradient steps to run THIS invocation if OPTIMIZE_QFI (resumable)
CFI_STEPS_PER_CALL = 4               # gradient steps to run THIS invocation if OPTIMIZE_CFI (resumable)
QFI_LR = 0.05
CFI_LR = 0.05
QFI_REG = 1e-6
CFI_REG = 1e-6
N_RUNS = 5                           # independent Bayesian recoveries averaged at this mass
GRID_N = 5000                        # number of candidate delta values scanned per recovery
GRID_HALF_WIDTH_SIGMAS = 8           # grid half-width = SIGMAS * sigma_theory(N_TRY)
# ------------------------------------
TOPOLOGY = os.environ.get('TOPOLOGY', TOPOLOGY)
LAYERS = int(os.environ.get('LAYERS', LAYERS))
MASS_UEV = float(os.environ.get('MASS_UEV', MASS_UEV))  # lets you launch several masses as
                                                          # separate background processes, e.g.:
                                                          #   MASS_UEV=0.1 python3 recover_mass_noise.py &
                                                          #   MASS_UEV=1.0 python3 recover_mass_noise.py &
                                                          # -- same env-var pattern as recover_mass.py,
                                                          # for a direct noiseless-vs-noise comparison.
MODEL = os.environ.get('MODEL', MODEL)  # e.g. MODEL=DFSZ python3 recover_mass_noise.py &
N_YEARS = float(os.environ.get('N_YEARS', N_YEARS))  # e.g. N_YEARS=0.5 python3 recover_mass_noise.py &
assert MODEL in ('KSVZ', 'DFSZ'), f"MODEL must be 'KSVZ' or 'DFSZ', got {MODEL!r}"
assert T2_US <= 2 * T1_US, f'thermal_relaxation_error requires T2 <= 2*T1 (got T1={T1_US}, T2={T2_US})'
POST_TOPOLOGY = POST_TOPOLOGY or TOPOLOGY
POST_LAYERS = LAYERS if POST_LAYERS is None else POST_LAYERS

prep_tag = f'{TOPOLOGY}_L{LAYERS}'                     # matches train_qfi.py's tag (prep-only)
tag = f'{prep_tag}_post_{POST_TOPOLOGY}_L{POST_LAYERS}'  # matches train_cfi.py's tag (prep+post)


def _fmt_n(n):
    """1e5 -> '1e05', 0.005 -> '0p005' -- compact, filename-safe. Same
    formatter as bayes_recover.py/mass_scan_bayes.py/
    targeted_mass_bayes*.py."""
    return f'{n:.4g}'.replace('+', '').replace('.', 'p')


noise_tag = f'T1{_fmt_n(T1_US)}_T2{_fmt_n(T2_US)}_czerr{_fmt_n(CZ_ERROR_PROB)}'

N = 5
alpha_fs = 1 / 137.036
sec_per_year = 3.156e7
T2_us_saturation = 100.0  # same DM-coherence saturation cap as every other script here
x0_fiducial = 6.498913663721062e-06


def g_KSVZ(m_a_ueV):
    return (alpha_fs / (2 * np.pi)) * 1.92 * m_a_ueV / 5.69e12


def g_DFSZ(m_a_ueV):
    return (alpha_fs / (2 * np.pi)) * 0.7467 * m_a_ueV / 5.69e12


def topology_edges(topology, num_qubits):
    """CZ-gate qubit pairs for each ansatz -- copied from
    onecircuit/ansatz.py's stargraph/ringgraph/fullgraph, duplicated
    here since this script is standalone."""
    Nq = num_qubits
    if topology == 'star':
        edges = [(0, i) for i in range(1, Nq)]
    elif topology == 'ring':
        edges = [(i, i + 1) for i in range(Nq - 1)] + [(0, Nq - 1)]
    elif topology == 'full':
        edges = [(i, (i + 1) % Nq) for i in range(Nq)] + [(i, (i + 2) % Nq) for i in range(Nq)]
    else:
        raise ValueError(f'unknown topology {topology!r}')
    return sorted(set(tuple(sorted(e)) for e in edges))


def idle_qc(num_qubits, time, params):
    """Placeholder stage: one `id` gate per qubit; the actual T1/T2
    relaxation is injected by NOISE_MODEL's thermal_relaxation_error
    attached to the 'id' instruction, not by this function."""
    qc = qiskit.QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        qc.id(q)
    return qc


# ---- tau/N_try -- identical formula to recover_mass.py ----
tau_us = min(tau_DM_us(MASS_UEV), T2_us_saturation)
tau_s = tau_us * 1e-6
N_TRY = (N_YEARS * sec_per_year) / tau_s

# ---- NoiseModel: idle T1/T2 relaxation (per qubit, at THIS mass's
# ---- tau_us) + per-edge CZ depolarizing error for BOTH prep and
# ---- post/measurement topologies ----
NOISE_MODEL = NoiseModel()
_idle_err = thermal_relaxation_error(T1_US, T2_US, tau_us)
for q in range(N):
    NOISE_MODEL.add_quantum_error(_idle_err, ['id'], [q])

_edges = sorted(set(topology_edges(TOPOLOGY, N)) | set(topology_edges(POST_TOPOLOGY, N)))
_cz_err = depolarizing_error(CZ_ERROR_PROB, 2)
for (a, b) in _edges:
    NOISE_MODEL.add_quantum_error(_cz_err, ['cz'], [a, b])

BACKEND = AerSimulator(method='density_matrix')

print(f'--- {tag}_{noise_tag}: T1={T1_US}us, T2={T2_US}us (idle @ tau={tau_us:.2f}us), '
      f'CZ_ERROR_PROB={CZ_ERROR_PROB} on {len(_edges)} edges, '
      f'OPTIMIZE_QFI={OPTIMIZE_QFI}, OPTIMIZE_CFI={OPTIMIZE_CFI} ---')


def _compose_stages(stages):
    num_qubits = stages[0][1]
    qc = qiskit.QuantumCircuit(num_qubits)
    for s in stages:
        qc &= s[0](s[1], s[2], s[3])
    return qc


def rho_from_stages(stages):
    """Exact (deterministic) density matrix for a stage list, evolved
    through AerSimulator's density_matrix method with NOISE_MODEL
    attached -- see module docstring."""
    qc = _compose_stages(stages)
    qc.save_density_matrix()
    result = BACKEND.run(qc, noise_model=NOISE_MODEL, shots=1).result()
    return np.asarray(result.data(0)['density_matrix'])


def probs_from_stages(stages):
    rho = rho_from_stages(stages)
    p = np.clip(np.real(np.diag(rho)), 0, None)
    return p / p.sum()


def grad_rho_network(stages):
    """LINE-FOR-LINE port of onecircuit.metric._grad_rho's math (2-point
    parameter-shift on stages[1][3], the sensing (x,y) params), with
    rho_from_stages() as the backend instead of onecircuit.metric.
    _safe_rho."""
    dp = []
    s = step_size
    for i in range(len(stages[1][3])):
        s1, s2 = copy.deepcopy(stages), copy.deepcopy(stages)
        s1[1][3][i] += s
        s2[1][3][i] -= s
        dp.append((rho_from_stages(s1) - rho_from_stages(s2)) / (2 * s))
    return dp


def sld_qfim_network(stages):
    """Plain-numpy port of onecircuit.metric.sld_qfim_eigens's closed-form
    SLD-QFI formula (same double sum over eigenpairs of rho,
    2*Re[<i|drho_k|j><j|drho_l|i>]/(lambda_i+lambda_j)), via
    np.linalg.eigh instead of tqix's eigenx/dotx/daggx -- see module
    docstring for why this duplication is necessary and deliberate."""
    rho = rho_from_stages(stages)
    grho = grad_rho_network(stages)
    d = len(grho)
    eigvals, eigvecs = np.linalg.eigh(rho)
    n = len(eigvals)
    H = np.zeros((d, d), dtype=complex)
    for k in range(d):
        for l in range(d):
            acc = 0.0
            for i in range(n):
                for j in range(n):
                    de = eigvals[i] + eigvals[j]
                    if de > 1e-14:
                        vi, vj = eigvecs[:, i], eigvecs[:, j]
                        num1 = np.conj(vi) @ grho[k] @ vj
                        num2 = np.conj(vj) @ grho[l] @ vi
                        acc += 2 * np.real(num1 * num2) / de
            H[k, l] = acc
    return np.real(H)


def cfim_network(stages):
    """LINE-FOR-LINE port of onecircuit.metric.cfim()'s math, using
    probs_from_stages() (AerSimulator-backed) instead of
    onecircuit.metric._safe_probs()."""
    prob = probs_from_stages(stages)
    d = len(stages[1][3])
    idxs = np.where(prob <= 1e-17)
    prob = np.delete(prob, idxs)

    dpro = []
    for i in range(d):
        s1, s2 = copy.deepcopy(stages), copy.deepcopy(stages)
        s1[1][3][i] += step_size
        s2[1][3][i] -= step_size
        gr = (probs_from_stages(s1) - probs_from_stages(s2)) / (2 * step_size)
        dpro.append(np.delete(gr, idxs))

    C = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            C[i, j] = np.sum(dpro[i] * dpro[j] / prob)
    return C


CKPT_CFI_BASELINE = os.path.join(DATA_DIR, f'ckpt_cfi_best_{tag}.pkl')  # noiseless baseline (theta, mu)

# ============================================================
# STAGE 1: state-prep (theta) -- re-optimize under noise, or reuse as-is
# ============================================================
if OPTIMIZE_QFI:
    CKPT_QFI = os.path.join(DATA_DIR, f'ckpt_qfi_network_noise_{prep_tag}_{noise_tag}.pkl')
    CKPT_QFI_BEST = os.path.join(DATA_DIR, f'ckpt_qfi_best_network_noise_{prep_tag}_{noise_tag}.pkl')
    LOG_QFI = os.path.join(DATA_DIR, f'costs_qfi_network_noise_{prep_tag}_{noise_tag}.json')

    def cost_qfi_network(circuit):
        """circuit = [prep_stage] (1-element wrapper, so gradient_func's
        generic circuit[i][3] indexing with train_opt=[0] mutates the
        prep params)."""
        stages = [circuit[0], [sensing_xy, N, 1.0, [x0_fiducial, 0.0]], [idle_qc, N, 1.0, []]]
        Q = sld_qfim_network(stages) + QFI_REG * np.eye(2)
        return np.trace(np.linalg.inv(Q))

    if os.path.exists(CKPT_QFI):
        with open(CKPT_QFI, 'rb') as f:
            qfi_circuit = pickle.load(f)
        qfi_costs = json.load(open(LOG_QFI)) if os.path.exists(LOG_QFI) else []
        print(f'[QFI-network] resumed, {len(qfi_costs)} steps so far')
    else:
        with open(CKPT_CFI_BASELINE, 'rb') as f:
            baseline_prep = pickle.load(f)[0]
        qfi_circuit = [[baseline_prep[0], baseline_prep[1], baseline_prep[2],
                         np.array(baseline_prep[3], dtype=float).copy()]]
        qfi_costs = []
        print('[QFI-network] starting fresh, warm-started from noiseless baseline theta')

    if os.path.exists(CKPT_QFI_BEST):
        with open(CKPT_QFI_BEST, 'rb') as f:
            qfi_best_circuit = pickle.load(f)
        qfi_best_cost = cost_qfi_network(qfi_best_circuit)
    else:
        qfi_best_circuit = qfi_circuit
        qfi_best_cost = cost_qfi_network(qfi_circuit)

    t0 = time.time()
    for _step in range(QFI_STEPS_PER_CALL):
        grad = np.array(gradient_func(qfi_circuit, cost_qfi_network, [0], 'two_point', None))
        qfi_circuit[0][3] = np.array(qfi_circuit[0][3]) - QFI_LR * grad
        c = cost_qfi_network(qfi_circuit)
        qfi_costs.append(c)
        if c < qfi_best_cost:
            qfi_best_cost = c
            qfi_best_circuit = [[qfi_circuit[0][0], qfi_circuit[0][1], qfi_circuit[0][2],
                                  np.array(qfi_circuit[0][3])]]
    dt = time.time() - t0

    with open(CKPT_QFI, 'wb') as f:
        pickle.dump(qfi_circuit, f)
    with open(CKPT_QFI_BEST, 'wb') as f:
        pickle.dump(qfi_best_circuit, f)
    json.dump(qfi_costs, open(LOG_QFI, 'w'))

    print(f'[QFI-network] ran {QFI_STEPS_PER_CALL} steps in {dt:.1f}s, total so far: {len(qfi_costs)} steps')
    print(f'[QFI-network] current Tr(Q^-1) = {qfi_costs[-1]:.4e}, BEST so far = {qfi_best_cost:.4e}')

    prep_stage = qfi_best_circuit[0]
else:
    # Prefer an already-noise-optimized theta if one exists (produced by a
    # previous OPTIMIZE_QFI=True run of THIS script, or by run/train_noise.py
    # -- both save the SAME 1-element [prep_stage] checkpoint format at the
    # SAME path, ckpt_qfi_best_network_noise_{prep_tag}_{noise_tag}.pkl).
    # Falling back straight to the NOISELESS baseline here would silently
    # ignore any noise-aware training you already did.
    _qfi_noise_ckpt = os.path.join(DATA_DIR, f'ckpt_qfi_best_network_noise_{prep_tag}_{noise_tag}.pkl')
    if os.path.exists(_qfi_noise_ckpt):
        with open(_qfi_noise_ckpt, 'rb') as f:
            prep_stage = pickle.load(f)[0]
        print(f'[QFI] OPTIMIZE_QFI=False -- reusing existing NOISE-OPTIMIZED theta from {_qfi_noise_ckpt}')
    else:
        with open(CKPT_CFI_BASELINE, 'rb') as f:
            prep_stage = pickle.load(f)[0]
        print('[QFI] OPTIMIZE_QFI=False -- no noise-optimized checkpoint found, using noiseless-trained theta unchanged')

# ============================================================
# STAGE 2: measurement ansatz (mu) -- re-optimize under noise, or reuse as-is
# ============================================================
if OPTIMIZE_CFI:
    CKPT_CFI = os.path.join(DATA_DIR, f'ckpt_cfi_network_noise_{tag}_{noise_tag}.pkl')
    CKPT_CFI_BEST = os.path.join(DATA_DIR, f'ckpt_cfi_best_network_noise_{tag}_{noise_tag}.pkl')
    LOG_CFI = os.path.join(DATA_DIR, f'costs_cfi_network_noise_{tag}_{noise_tag}.json')

    def cost_cfi_network(circuit):
        """circuit = [prep_stage(fixed), post_stage] -- train_opt=[1]
        mutates only the post (measurement) params."""
        stages = [circuit[0], [sensing_xy, N, 1.0, [x0_fiducial, 0.0]], [idle_qc, N, 1.0, []], circuit[1]]
        C = cfim_network(stages) + CFI_REG * np.eye(2)
        return np.trace(np.linalg.inv(C))

    if os.path.exists(CKPT_CFI):
        with open(CKPT_CFI, 'rb') as f:
            cfi_circuit = pickle.load(f)
        cfi_costs = json.load(open(LOG_CFI)) if os.path.exists(LOG_CFI) else []
        print(f'[CFI-network] resumed, {len(cfi_costs)} steps so far')
    else:
        with open(CKPT_CFI_BASELINE, 'rb') as f:
            baseline_post = pickle.load(f)[2]
        cfi_circuit = [prep_stage, [baseline_post[0], baseline_post[1], baseline_post[2],
                                     np.array(baseline_post[3], dtype=float).copy()]]
        cfi_costs = []
        print('[CFI-network] starting fresh, warm-started from noiseless baseline mu')
    cfi_circuit[0] = prep_stage  # always keep in sync with STAGE 1's result (matters when resuming
                                 # after OPTIMIZE_QFI changed between runs)

    if os.path.exists(CKPT_CFI_BEST):
        with open(CKPT_CFI_BEST, 'rb') as f:
            cfi_best_circuit = pickle.load(f)
        cfi_best_cost = cost_cfi_network(cfi_best_circuit)
    else:
        cfi_best_circuit = cfi_circuit
        cfi_best_cost = cost_cfi_network(cfi_circuit)

    t0 = time.time()
    for _step in range(CFI_STEPS_PER_CALL):
        grad = np.array(gradient_func(cfi_circuit, cost_cfi_network, [1], 'two_point', None))
        cfi_circuit[1][3] = np.array(cfi_circuit[1][3]) - CFI_LR * grad
        c = cost_cfi_network(cfi_circuit)
        cfi_costs.append(c)
        if c < cfi_best_cost:
            cfi_best_cost = c
            cfi_best_circuit = [cfi_circuit[0], [cfi_circuit[1][0], cfi_circuit[1][1], cfi_circuit[1][2],
                                                  np.array(cfi_circuit[1][3])]]
    dt = time.time() - t0

    with open(CKPT_CFI, 'wb') as f:
        pickle.dump(cfi_circuit, f)
    with open(CKPT_CFI_BEST, 'wb') as f:
        pickle.dump(cfi_best_circuit, f)
    json.dump(cfi_costs, open(LOG_CFI, 'w'))

    print(f'[CFI-network] ran {CFI_STEPS_PER_CALL} steps in {dt:.1f}s, total so far: {len(cfi_costs)} steps')
    print(f'[CFI-network] current Tr(C^-1) = {cfi_costs[-1]:.4e}, BEST so far = {cfi_best_cost:.4e}')

    post_stage = cfi_best_circuit[1]
else:
    # Same idea as STAGE 1's skip branch: prefer an already-noise-optimized
    # mu if one exists at data/ckpt_cfi_best_network_noise_{tag}_{noise_tag}.pkl
    # -- written by either a previous OPTIMIZE_CFI=True run of THIS script,
    # OR by run/train_noise.py (both save the SAME 2-element
    # [prep_stage, post_stage] format, post at index 1 -- NOT the noiseless
    # baseline's 3-element [prep, sens, post] format, post at index 2).
    # If you swept POST_LAYERS_LIST in train_noise.py, set POST_LAYERS
    # here to whichever value you picked from that sweep's plot before
    # running -- there is no single "the" winner for CFI the way STAGE 1
    # auto-picks one for QFI, since train_noise.py trains every PL.
    _cfi_noise_ckpt = os.path.join(DATA_DIR, f'ckpt_cfi_best_network_noise_{tag}_{noise_tag}.pkl')
    if os.path.exists(_cfi_noise_ckpt):
        with open(_cfi_noise_ckpt, 'rb') as f:
            post_stage = pickle.load(f)[1]
        print(f'[CFI] OPTIMIZE_CFI=False -- reusing existing NOISE-OPTIMIZED mu from {_cfi_noise_ckpt}')
    else:
        with open(CKPT_CFI_BASELINE, 'rb') as f:
            post_stage = pickle.load(f)[2]
        print('[CFI] OPTIMIZE_CFI=False -- no noise-optimized checkpoint found, using noiseless-trained mu unchanged')

# ============================================================
# STAGE 3: targeted single-mass Bayesian recovery, using whichever
# (prep_stage, post_stage) resulted from STAGE 1/2 above
# ============================================================
run_tag = (f'{tag}_{MODEL}_m{_fmt_n(MASS_UEV)}ueV_N{_fmt_n(N_TRY)}_{noise_tag}'
           f'_qfi{int(OPTIMIZE_QFI)}_cfi{int(OPTIMIZE_CFI)}')


def probs_for_delta(d):
    stages = [prep_stage, [sensing_xy, N, 1.0, [d, 0.0]], [idle_qc, N, 1.0, []], post_stage]
    return probs_from_stages(stages)


_fiducial_stages = [prep_stage, [sensing_xy, N, 1.0, [x0_fiducial, 0.0]], [idle_qc, N, 1.0, []], post_stage]
_C_check = cfim_network(_fiducial_stages)
eig_min = max(np.linalg.eigvalsh(_C_check).min(), 1e-12)
_sigma_theory_N_TRY = 1.0 / np.sqrt(N_TRY * eig_min)
GRID_HALF_WIDTH = GRID_HALF_WIDTH_SIGMAS * _sigma_theory_N_TRY
print(f'--- {run_tag}: N_years={N_YEARS}, tau={tau_us:.2f}us, N_try={N_TRY:.4e}, '
      f'eig_min(C_final)={eig_min:.4e}, grid half-width={GRID_HALF_WIDTH:.4e} ---')

if __name__ == '__main__':
    g_model_fn = g_KSVZ if MODEL == 'KSVZ' else g_DFSZ
    g_target = g_model_fn(MASS_UEV)
    delta_true, _ = delta_axion(g_target, MASS_UEV)

    hw = GRID_HALF_WIDTH
    delta_grid = np.linspace(delta_true - hw, delta_true + hw, GRID_N)
    print(f'  building p_grid ({GRID_N} points) under network noise -- this is the slow step...')
    p_grid = np.array([probs_for_delta(d) for d in delta_grid])
    p_true = probs_for_delta(delta_true)
    print('  p_grid built.')

    deltas_hat = []
    for _ in range(N_RUNS):
        _, mean, _, _, _ = _core_sequential_recover(delta_grid, p_grid, p_true, N_TRY, delta_true)
        deltas_hat.append(mean)
    deltas_hat = np.array(deltas_hat)
    g_hats = np.array([g_agg_from_delta(abs(d), MASS_UEV) for d in deltas_hat])

    print(f'  m_a={MASS_UEV:.4f} ueV [{MODEL}], network noise (T1={T1_US}us, T2={T2_US}us, '
          f'CZ_err={CZ_ERROR_PROB}), OPTIMIZE_QFI={OPTIMIZE_QFI}, OPTIMIZE_CFI={OPTIMIZE_CFI}: '
          f'g_target={g_target:.3e}  g_hat={g_hats.mean():.3e} +- {g_hats.std():.3e}  '
          f'ratio={g_hats.mean() / g_target:.3f}')

    results = {
        'topology': TOPOLOGY, 'LAYERS': LAYERS,
        'POST_TOPOLOGY': POST_TOPOLOGY, 'POST_LAYERS': POST_LAYERS,
        'model': MODEL,
        'mass_ueV': MASS_UEV, 'N_years': N_YEARS, 'N_TRY': N_TRY,
        'T1_us': T1_US, 'T2_us': T2_US, 'cz_error_prob': CZ_ERROR_PROB,
        'optimize_qfi': OPTIMIZE_QFI, 'optimize_cfi': OPTIMIZE_CFI,
        'noise_edges': _edges,
        'tau_us': tau_us, 'eig_min_C_final': float(eig_min),
        'g_target': float(g_target), 'delta_true': float(delta_true),
        'g_hat_runs': g_hats.tolist(),
        'g_hat_mean': float(g_hats.mean()),
        'g_hat_std': float(g_hats.std()),
    }

    out_path = os.path.join(DATA_DIR, f'targeted_mass_bayes_network_noise_optimized_results_{run_tag}.json')
    json.dump(results, open(out_path, 'w'), indent=2)
    print(f'saved {out_path}')
