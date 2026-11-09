import sys, os, time, pickle, json, copy
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, CODE_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIGURE_DIR = os.path.join(SCRIPT_DIR, '..', 'figure')
os.makedirs(FIGURE_DIR, exist_ok=True)
import numpy as np
import qiskit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import onecircuit
from onecircuit import *
from onecircuit.util import step_size
from onecircuit.optimizing import gradient_func
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error

"""
NEW, STANDALONE script -- does NOT modify train_qfi.py, train_cfi.py,
plot/plot_vs_layers.py, or recover_mass_noise.py.
Edit the constants below, then just run:
    python3 train_noise.py

Everything is an IN-FILE CONSTANT, no environment variables -- edit,
then run, exactly like train_qfi.py/train_cfi.py.

The noisy analogue of plot/plot_vs_layers.py's two-panel figure (see
the image you shared), but SELF-CONTAINED: this one script both trains
and plots, and automatically feeds the winning config from stage 1 into
stage 2, instead of you manually training everything with train_qfi.py/
train_cfi.py first and only plotting afterward.

NO DEPENDENCE ON NOISELESS CHECKPOINTS: earlier versions of this script
warm-started every noisy training run from train_qfi.py's/train_cfi.py's
NOISELESS checkpoints, which meant a (topology, L) combo the noisy QFI
sweep happened to pick as the winner could be stuck with no matching
noiseless CFI checkpoint to warm-start STAGE 2 from -- a real, confusing
dead end. Both stages now train FROM SCRATCH (fresh random params, via
VariationalCircuit.add_params() -- the SAME fresh-init pattern
train_qfi.py/train_cfi.py themselves use when THEY have no checkpoint
yet) directly under noise whenever no noisy checkpoint already exists.
No noiseless baseline is read or required anywhere in this script.

STAGE 1 (QFI over topology x layers): for every (topology, L) in
TOPOLOGIES x LAYERS_LIST, resumes from
data/ckpt_qfi_network_noise_{topology}_L{L}_{noise_tag}.pkl if it
exists, else starts fresh, and runs QFI_STEPS gradient steps minimizing
Tr(Q_xy^-1) computed on the NOISY circuit [prep, sens(fixed x0,y0),
idle] (same physics/verification as
recover_mass_noise.py's sld_qfim_network -- see
that script's docstring: AerSimulator density_matrix method, exact/
deterministic, cross-checked against the analytic SLD-QFI formula).
Plots the LEFT panel (bar chart, Tr(Q^-1) vs. layers, grouped by
topology) exactly like plot_vs_layers.py's left panel.

QFI_STEPS=0 IS THE "just compare what I already trained" MODE: no new
gradient steps run anywhere, and any (topology, L) with no existing
checkpoint yet is [skip]ped (not trained from scratch just to show a
meaningless 0-step random point) -- so if you already ran a full
TOPOLOGIES x LAYERS_LIST sweep with QFI_STEPS>0 in a PREVIOUS invocation
and just want to redo STAGE 2 (CFI) without paying for more QFI
training, set QFI_STEPS=0 here and the LEFT panel still shows the FULL
existing comparison across every trained config, not just one.

SELECTION: among every (topology, L) that has a result (trained this run
or loaded from an existing checkpoint), picks the one with the LOWEST
noisy Tr(Q^-1) (the best achievable quantum bound under this noise) --
printed clearly when the script runs.

STAGE 2 (CFI sweep, always runs): fixes prep to the STAGE 1 winner, then
for every PL in POST_LAYERS_LIST resumes from
data/ckpt_cfi_network_noise_{topology}_L{L}_post_{topology}_L{PL}_{noise_tag}.pkl
if it exists, else starts fresh, and runs CFI_STEPS gradient steps
minimizing Tr(C_xy^-1) on the NOISY circuit [prep(fixed), sens, idle,
post]. Plots the RIGHT panel -- a single bar series for the winning
topology (matching your reference image's "circulant (prep=3)"
single-series style, since only one prep config is ever the winner).

SAVES data/optimal_noise_results_{noise_tag}.json (q_results, c_results,
best_topo/best_L, and the swept config lists) once training finishes,
THEN draws a quick preview plot -- but the figure you'll actually want
to iterate on (aspect ratio, gridlines, label sizes, colors, ...) should
be tuned in plot/plot_noise.py, which just loads that JSON and
redraws -- no need to re-run this (slow) training script every time you
tweak the figure's look.

Both stages are RESUMABLE the same way train_qfi.py/train_cfi.py are:
re-running this script picks up every (topology, L)/(topology, PL)
checkpoint where it left off and keeps improving it (a NEW winning
config after more training simply gets picked automatically next time).
Every noise combination (T1_US, T2_US, CZ_ERROR_PROB) gets its own set
of checkpoints (never collides with the noiseless baseline or a
different noise strength).

PERFORMANCE: each noisy Tr(Q^-1)/Tr(C^-1) evaluation costs ~5 AerSimulator
calls (~5-10ms each); one gradient step needs ~2x(num params) evaluations
-- for 'full' L=4 (the heaviest config, 240 params) that is roughly
30-40s PER STEP. Default QFI_STEPS=0 (no new QFI training -- just
compares whatever's already checkpointed), CFI_STEPS=10. With
TOPOLOGIES=['star','ring','full'] x LAYERS_LIST=[1,2,3,4] and
QFI_STEPS>0, a single invocation covers up to 12 QFI configs + 4 CFI
configs; expect anywhere from ~15 minutes (star/ring dominate, few
params) to well over an hour if 'full' L=3/L=4 configs are included --
run this under nohup and re-run it later to keep refining every config
further.

Output figure: figure/{OUT_PNG} (default 'bound_vs_layers_network_noise.png').
"""

ansatz = {
    'star': (stargraph, stargraph_inv),
    'ring': (ringgraph, ringgraph_inv),
    'full': (fullgraph, fullgraph_inv),
}
colors = {'star': 'tab:blue', 'ring': 'tab:green', 'full': 'tab:red'}
labels = {'star': 'star', 'ring': 'ring', 'full': 'circulant'}

# ---- edit these before running ----
TOPOLOGIES = ['star', 'ring', 'full']
LAYERS_LIST = [1, 2, 3, 4]        # LEFT panel x-axis (QFI prep layers)
POST_LAYERS_LIST = [1, 2, 3, 4]   # RIGHT panel x-axis (CFI post/measurement layers), for the WINNING prep only
N = 5
T1_US = 100.0                      # per-qubit energy-relaxation time [us] (idle noise)
T2_US = 100.0                      # per-qubit dephasing time [us] (idle noise); must have T2 <= 2*T1
CZ_ERROR_PROB = 0.005              # depolarizing-error probability per CZ gate (network/gate noise)
QFI_STEPS = 200                     # gradient steps THIS invocation runs per (topology, L) config (resumable).
                                    # 0 = train NOTHING new -- just load whatever
                                    # ckpt_qfi_best_network_noise_{topology}_L{L}_{noise_tag}.pkl checkpoints
                                    # ALREADY exist (from a previous run with QFI_STEPS>0) and compare them;
                                    # any (topology, L) with no existing checkpoint is skipped, not trained
                                    # from scratch. Set >0 to actually train (new configs start fresh, or
                                    # RESUME + run 20 MORE steps on top of whatever was already trained).
CFI_STEPS = 200                     # gradient steps THIS invocation runs per post-layers config (resumable).
                                    # Same 0-means-"just load, don't train" convention as QFI_STEPS.
QFI_LR = 0.05
CFI_LR = 0.05
REG = 1e-6
BAR_WIDTH = 0.15
OUT_PNG = 'bound_vs_layers_network_noise.png'
# ------------------------------------
assert T2_US <= 2 * T1_US, f'thermal_relaxation_error requires T2 <= 2*T1 (got T1={T1_US}, T2={T2_US})'

x0_fiducial, y0_fiducial = 6.498913663721062e-06, 0.0
T2_us_saturation = 100.0  # DM-coherence saturation cap (same constant used everywhere else in this
                          # project) -- idle-noise duration for TRAINING is fixed at this, since
                          # training happens at the fiducial point, independent of any specific
                          # axion mass (mass only enters later, in the recovery script)
BACKEND = AerSimulator(method='density_matrix')


def _fmt_n(n):
    """1e5 -> '1e05', 0.005 -> '0p005' -- compact, filename-safe. Same
    formatter as bayes_recover.py/mass_scan_bayes.py/
    targeted_mass_bayes*.py."""
    return f'{n:.4g}'.replace('+', '').replace('.', 'p')


noise_tag = f'T1{_fmt_n(T1_US)}_T2{_fmt_n(T2_US)}_czerr{_fmt_n(CZ_ERROR_PROB)}'


def topology_edges(topology, num_qubits):
    """CZ-gate qubit pairs for each ansatz -- copied from
    onecircuit/ansatz.py's stargraph/ringgraph/fullgraph (see
    recover_mass_noise.py's docstring for the full
    derivation)."""
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
    """Placeholder stage: one `id` gate per qubit; the NoiseModel attaches
    the actual T1/T2 relaxation to the 'id' instruction."""
    qc = qiskit.QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        qc.id(q)
    return qc


def build_noise_model(edge_topologies):
    """NoiseModel with idle T1/T2 relaxation on every qubit + per-edge CZ
    depolarizing error on the UNION of the given topologies' edges (pass
    a single topology for a prep-only QFI circuit, or the same topology
    twice/once for a prep+post CFI circuit -- since prep and post share
    the same topology once STAGE 2 fixes prep to the STAGE 1 winner)."""
    nm = NoiseModel()
    idle_err = thermal_relaxation_error(T1_US, T2_US, T2_us_saturation)
    for q in range(N):
        nm.add_quantum_error(idle_err, ['id'], [q])
    edges = set()
    for topo in edge_topologies:
        edges |= set(topology_edges(topo, N))
    cz_err = depolarizing_error(CZ_ERROR_PROB, 2)
    for (a, b) in sorted(edges):
        nm.add_quantum_error(cz_err, ['cz'], [a, b])
    return nm


def _compose_stages(stages):
    num_qubits = stages[0][1]
    qc = qiskit.QuantumCircuit(num_qubits)
    for s in stages:
        qc &= s[0](s[1], s[2], s[3])
    return qc


def rho_from_stages(stages, noise_model):
    qc = _compose_stages(stages)
    qc.save_density_matrix()
    result = BACKEND.run(qc, noise_model=noise_model, shots=1).result()
    return np.asarray(result.data(0)['density_matrix'])


def probs_from_stages(stages, noise_model):
    rho = rho_from_stages(stages, noise_model)
    p = np.clip(np.real(np.diag(rho)), 0, None)
    return p / p.sum()


def grad_rho_network(stages, noise_model):
    dp = []
    s = step_size
    for i in range(len(stages[1][3])):
        s1, s2 = copy.deepcopy(stages), copy.deepcopy(stages)
        s1[1][3][i] += s
        s2[1][3][i] -= s
        dp.append((rho_from_stages(s1, noise_model) - rho_from_stages(s2, noise_model)) / (2 * s))
    return dp


def sld_qfim_network(stages, noise_model):
    """Plain-numpy port of onecircuit.metric.sld_qfim_eigens's closed-form
    SLD-QFI formula -- see recover_mass_noise.py's
    docstring (numerically verified there against the analytic 4*Var(Z)
    result for a Ramsey toy circuit)."""
    rho = rho_from_stages(stages, noise_model)
    grho = grad_rho_network(stages, noise_model)
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


def cfim_network(stages, noise_model):
    """LINE-FOR-LINE port of onecircuit.metric.cfim()'s math -- identical
    to recover_mass_noise.py's cfim_network()."""
    prob = probs_from_stages(stages, noise_model)
    d = len(stages[1][3])
    idxs = np.where(prob <= 1e-17)
    prob = np.delete(prob, idxs)

    dpro = []
    for i in range(d):
        s1, s2 = copy.deepcopy(stages), copy.deepcopy(stages)
        s1[1][3][i] += step_size
        s2[1][3][i] -= step_size
        gr = (probs_from_stages(s1, noise_model) - probs_from_stages(s2, noise_model)) / (2 * step_size)
        dpro.append(np.delete(gr, idxs))

    C = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            C[i, j] = np.sum(dpro[i] * dpro[j] / prob)
    return C


def train_qfi_noisy(topology, L, noise_model, steps, ckpt_path, ckpt_best_path, log_path):
    """Resumable noisy-QFI training for ONE (topology, L) config -- trains
    FROM SCRATCH (fresh random theta, via VariationalCircuit.add_params(),
    the same fresh-init pattern train_qfi.py itself uses) the first time,
    resumes from ckpt_path afterward. NO noiseless checkpoint is read.
    Returns (best_cost, best_circuit) where best_circuit = [prep_stage]."""

    def cost_fn(circuit):
        stages = [circuit[0], [sensing_xy, N, 1.0, [x0_fiducial, y0_fiducial]], [idle_qc, N, 1.0, []]]
        Q = sld_qfim_network(stages, noise_model) + REG * np.eye(2)
        return np.trace(np.linalg.inv(Q))

    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            circuit = pickle.load(f)
        costs = json.load(open(log_path)) if os.path.exists(log_path) else []
    else:
        fwd, inv = ansatz[topology]
        prep = [fwd, N, L, []]
        sens = [sensing_xy, N, 1.0, [x0_fiducial, y0_fiducial]]
        post = [inv, N, L, []]
        vqc0 = VariationalCircuit([prep, sens, post])
        vqc0.add_params()
        fresh_prep = vqc0.circuit[0]
        circuit = [[fresh_prep[0], fresh_prep[1], fresh_prep[2], np.array(fresh_prep[3], dtype=float)]]
        costs = []

    if os.path.exists(ckpt_best_path):
        with open(ckpt_best_path, 'rb') as f:
            best_circuit = pickle.load(f)
        best_cost = cost_fn(best_circuit)
    else:
        best_circuit = circuit
        best_cost = cost_fn(circuit)

    for _ in range(steps):
        grad = np.array(gradient_func(circuit, cost_fn, [0], 'two_point', None))
        circuit[0][3] = np.array(circuit[0][3]) - QFI_LR * grad
        c = cost_fn(circuit)
        costs.append(c)
        if c < best_cost:
            best_cost = c
            best_circuit = [[circuit[0][0], circuit[0][1], circuit[0][2], np.array(circuit[0][3])]]

    with open(ckpt_path, 'wb') as f:
        pickle.dump(circuit, f)
    with open(ckpt_best_path, 'wb') as f:
        pickle.dump(best_circuit, f)
    json.dump(costs, open(log_path, 'w'))
    return best_cost, best_circuit


def train_cfi_noisy(prep_stage, post_topology, PL, noise_model, steps, ckpt_path, ckpt_best_path, log_path):
    """Resumable noisy-CFI training for ONE post-layers config, prep held
    FIXED to prep_stage (the STAGE 1 winner). Trains FROM SCRATCH (fresh
    random mu, via VariationalCircuit.add_params(), the same fresh-init
    pattern train_cfi.py itself uses) the first time, resumes from
    ckpt_path afterward. NO noiseless checkpoint is read. Returns
    (best_cost, best_circuit) where best_circuit = [prep_stage, post_stage]."""

    def cost_fn(circuit):
        stages = [circuit[0], [sensing_xy, N, 1.0, [x0_fiducial, y0_fiducial]], [idle_qc, N, 1.0, []], circuit[1]]
        C = cfim_network(stages, noise_model) + REG * np.eye(2)
        return np.trace(np.linalg.inv(C))

    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            circuit = pickle.load(f)
        costs = json.load(open(log_path)) if os.path.exists(log_path) else []
    else:
        _, inv = ansatz[post_topology]
        sens = [sensing_xy, N, 1.0, [x0_fiducial, y0_fiducial]]
        post = [inv, N, PL, []]
        vqc0 = VariationalCircuit([prep_stage, sens, post])
        vqc0.add_params()
        circuit3 = vqc0.circuit
        circuit3[0] = prep_stage  # keep prep fixed (not re-randomized by add_params) -- same
                                  # defensive line train_cfi.py itself uses
        circuit = [circuit3[0], circuit3[2]]
        costs = []
    circuit[0] = prep_stage  # always keep in sync with STAGE 1's winner

    if os.path.exists(ckpt_best_path):
        with open(ckpt_best_path, 'rb') as f:
            best_circuit = pickle.load(f)
        best_cost = cost_fn(best_circuit)
    else:
        best_circuit = circuit
        best_cost = cost_fn(circuit)

    for _ in range(steps):
        grad = np.array(gradient_func(circuit, cost_fn, [1], 'two_point', None))
        circuit[1][3] = np.array(circuit[1][3]) - CFI_LR * grad
        c = cost_fn(circuit)
        costs.append(c)
        if c < best_cost:
            best_cost = c
            best_circuit = [circuit[0], [circuit[1][0], circuit[1][1], circuit[1][2], np.array(circuit[1][3])]]

    with open(ckpt_path, 'wb') as f:
        pickle.dump(circuit, f)
    with open(ckpt_best_path, 'wb') as f:
        pickle.dump(best_circuit, f)
    json.dump(costs, open(log_path, 'w'))
    return best_cost, best_circuit


if __name__ == '__main__':
    print(f'=== train_noise.py: T1={T1_US}us, T2={T2_US}us, CZ_ERROR_PROB={CZ_ERROR_PROB} ===')

    # ============================================================
    # STAGE 1: noisy QFI over topology x layers. QFI_STEPS=0 -> load and
    # COMPARE whatever checkpoints already exist (no new training, no
    # config trained from scratch just to show a 0-step random point);
    # QFI_STEPS>0 -> actually train (resume existing, or start fresh for
    # any (topology, L) with nothing saved yet).
    # ============================================================
    tag_qfi = 'QFI (loading existing checkpoints, QFI_STEPS=0 -- no new training)' if QFI_STEPS == 0 \
        else f'QFI ({QFI_STEPS} new gradient steps per config)'
    print(f'=== STAGE 1: {tag_qfi} ===')
    q_results = {topo: {} for topo in TOPOLOGIES}
    best_prep_by_config = {}

    for topo in TOPOLOGIES:
        nm_prep = build_noise_model([topo])
        for L in LAYERS_LIST:
            prep_tag = f'{topo}_L{L}'
            ckpt = os.path.join(DATA_DIR, f'ckpt_qfi_network_noise_{prep_tag}_{noise_tag}.pkl')
            ckpt_best = os.path.join(DATA_DIR, f'ckpt_qfi_best_network_noise_{prep_tag}_{noise_tag}.pkl')
            log = os.path.join(DATA_DIR, f'costs_qfi_network_noise_{prep_tag}_{noise_tag}.json')
            if QFI_STEPS == 0 and not os.path.exists(ckpt_best):
                print(f'  [skip] no existing checkpoint for {prep_tag} and QFI_STEPS=0 (nothing to compare)')
                continue
            t0 = time.time()
            best_cost, best_circuit = train_qfi_noisy(topo, L, nm_prep, QFI_STEPS, ckpt, ckpt_best, log)
            print(f'  [{prep_tag}] Tr(Q^-1)_noisy = {best_cost:.4e}  ({time.time() - t0:.1f}s)')
            q_results[topo][L] = best_cost
            best_prep_by_config[(topo, L)] = best_circuit[0]

    flat = [(topo, L, val) for topo in TOPOLOGIES for L, val in q_results[topo].items()]
    if not flat:
        raise SystemExit('no QFI checkpoints found/trained -- set QFI_STEPS>0 to train at least one '
                          '(topology, L) config under this noise combo first')
    best_topo, best_L, best_q = min(flat, key=lambda t: t[2])
    best_prep_stage = best_prep_by_config[(best_topo, best_L)]
    print(f'=== winning prep config: topology={best_topo}, layers={best_L}, Tr(Q^-1)_noisy={best_q:.4e} ===')

    # ============================================================
    # STAGE 2: noisy CFI sweep over measurement layers, prep FIXED to the winner
    # ============================================================
    print('=== STAGE 2: noisy CFI sweep over measurement layers ===')
    c_results = {}
    nm_post = build_noise_model([best_topo])
    for PL in POST_LAYERS_LIST:
        cfi_tag = f'{best_topo}_L{best_L}_post_{best_topo}_L{PL}'
        ckpt = os.path.join(DATA_DIR, f'ckpt_cfi_network_noise_{cfi_tag}_{noise_tag}.pkl')
        ckpt_best = os.path.join(DATA_DIR, f'ckpt_cfi_best_network_noise_{cfi_tag}_{noise_tag}.pkl')
        log = os.path.join(DATA_DIR, f'costs_cfi_network_noise_{cfi_tag}_{noise_tag}.json')
        t0 = time.time()
        best_cost, best_circuit = train_cfi_noisy(best_prep_stage, best_topo, PL, nm_post, CFI_STEPS, ckpt, ckpt_best, log)
        print(f'  [{cfi_tag}] Tr(C^-1)_noisy = {best_cost:.4e}  ({time.time() - t0:.1f}s)')
        c_results[PL] = best_cost

    # ============================================================
    # SAVE RESULTS -- so the figure can be re-styled/re-plotted (ratio,
    # grid, labels, ...) as many times as needed via
    # plot/plot_noise.py WITHOUT re-running this (slow) training.
    # This script itself is compute-only from here on except for one
    # quick default preview plot below.
    # ============================================================
    # PLOT_TOPOLOGIES/PLOT_LAYERS_LIST: just the full swept lists -- any
    # (topology, L) that got [skip]ped above simply has no entry in
    # q_results, and the bar chart below already handles that (np.nan,
    # no bar drawn), so nothing special needed here.
    PLOT_TOPOLOGIES = TOPOLOGIES
    PLOT_LAYERS_LIST = LAYERS_LIST

    results_out = {
        'T1_us': T1_US, 'T2_us': T2_US, 'cz_error_prob': CZ_ERROR_PROB,
        'topologies': PLOT_TOPOLOGIES, 'layers_list': PLOT_LAYERS_LIST, 'post_layers_list': POST_LAYERS_LIST,
        'q_results': q_results,          # {topology: {L: Tr(Q^-1)}}
        'c_results': c_results,          # {PL: Tr(C^-1)}  -- winning prep only
        'best_topo': best_topo, 'best_L': best_L, 'best_q': best_q,
    }
    results_path = os.path.join(DATA_DIR, f'optimal_noise_results_{noise_tag}.json')
    json.dump(results_out, open(results_path, 'w'), indent=2)
    print(f'saved {results_path}')
    print('Q results (noisy):', q_results)
    print('C results (noisy, winning prep only):', c_results)
    print('now edit/re-run plot/plot_noise.py any time to re-style the figure without re-training')

    # ---- quick default preview plot (same style as plot/plot_noise.py's
    # ---- defaults) so you get SOMETHING immediately; the real iteration on
    # ---- ratio/grid/labels happens in plot/plot_noise.py ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    n_topo = len(PLOT_TOPOLOGIES)
    x1 = np.arange(len(PLOT_LAYERS_LIST))
    for i, topo in enumerate(PLOT_TOPOLOGIES):
        offset = (i - (n_topo - 1) / 2) * BAR_WIDTH
        q_vals = [q_results[topo].get(L, np.nan) for L in PLOT_LAYERS_LIST]
        ax1.bar(x1 + offset, q_vals, width=BAR_WIDTH, color=colors[topo], label=labels[topo])
    ax1.set_xlabel('state-prep (theta) layers')
    ax1.set_ylabel(r'$\mathrm{Tr}(Q^{-1})$  (quantum bound, noisy)')
    ax1.set_yscale('log')
    ax1.set_xticks(x1)
    ax1.set_xticklabels([str(L) for L in PLOT_LAYERS_LIST])
    ax1.grid(alpha=0.3, axis='y')
    ax1.legend(fontsize=8)
    ax1.set_title(f'Quantum Cramer-Rao bound vs. prep layers\n(T1={T1_US}us, T2={T2_US}us, CZ_err={CZ_ERROR_PROB})')

    x2 = np.arange(len(POST_LAYERS_LIST))
    c_vals = [c_results.get(PL, np.nan) for PL in POST_LAYERS_LIST]
    ax2.bar(x2, c_vals, width=BAR_WIDTH, color=colors[best_topo],
            label=f'{labels[best_topo]} (prep={best_L})')
    ax2.set_xlabel('measurement (mu) layers')
    ax2.set_ylabel(r'$\mathrm{Tr}(C^{-1})$  (classical bound, noisy)')
    ax2.set_yscale('log')
    ax2.set_xticks(x2)
    ax2.set_xticklabels([str(PL) for PL in POST_LAYERS_LIST])
    ax2.grid(alpha=0.3, axis='y')
    ax2.legend(fontsize=8)
    ax2.set_title(f'Classical Cramer-Rao bound vs. measurement layers\n(winning prep: {labels[best_topo]}, L={best_L})')

    all_vals = [v for topo in PLOT_TOPOLOGIES for v in q_results[topo].values() if np.isfinite(v)]
    all_vals += [v for v in c_results.values() if np.isfinite(v)]
    if all_vals:
        ymin, ymax = min(all_vals), max(all_vals)
        ax1.set_ylim(ymin * 0.5, ymax * 2)
        ax2.set_ylim(ymin * 0.5, ymax * 2)

    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, OUT_PNG)
    fig.savefig(out_path, dpi=140)
    print(f'saved {out_path} (preview -- re-style via plot/plot_noise.py)')
