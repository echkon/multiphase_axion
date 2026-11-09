import sys, os, time, pickle, json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, CODE_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
import numpy as np
import onecircuit
from onecircuit import *
from onecircuit.metric import cfim
from onecircuit.optimizing import gradient_func

"""
Edit the constants below, then just run:
    python3 train_cfi.py

Single, unified script for ALL topologies/layer counts (see
train_qfi.py's docstring for the naming convention: every data file gets
an explicit "_{TOPOLOGY}_L{LAYERS}" suffix, always).

With theta (state prep) FIXED from train_qfi.py's best checkpoint, train
the measurement stage (mu) to minimize the classical multiparameter
Cramer-Rao cost Tr(C_xy^-1) -- same Cartesian (x, y) parametrization as
train_qfi.py.

The measurement ansatz does NOT have to match the state-prep ansatz: by
default POST_TOPOLOGY/POST_LAYERS mirror TOPOLOGY/LAYERS (mu = the
inverse of whatever graph theta used, same layer count -- the common
case), but you're free to set them independently, e.g. TOPOLOGY='star',
LAYERS=2 for theta with POST_TOPOLOGY='full', POST_LAYERS=1 for mu, to
see whether a different/cheaper measurement ansatz still recovers most of
Tr(C_xy^-1).

NAMING: same "no special case" rule as everywhere else in this pipeline
(see train_qfi.py's docstring re: LAYERS=1) -- the output tag ALWAYS
spells out both the prep and post ansatz explicitly:
    {TOPOLOGY}_L{LAYERS}_post_{POST_TOPOLOGY}_L{POST_LAYERS}
even when POST_TOPOLOGY/POST_LAYERS equal TOPOLOGY/LAYERS (the common
case). There is no shortened/collapsed filename for the matched-ansatz
case, and POST_TOPOLOGY/POST_LAYERS = None NEVER appears in a filename
(None is only a way to say "same as TOPOLOGY/LAYERS" when editing this
file -- it's resolved to concrete values before anything is named).

NOTE: an earlier version of this script used "params += lr*grad" here,
based on a mistaken claim that qiskit's .inverse() flips the gradient
sign for this stage. That is incorrect: gradient_func (optimizing.py)
computes cost1/cost2 by rebuilding and evaluating the FULL circuit
(including .inverse()) at each shifted parameter value, so the returned
gradient already has the correct sign for cost(circuit) as a function of
circuit[2][3] -- no extra sign flip is needed. The "+=" version performed
gradient ASCENT (Tr(C^-1) increasing with training instead of decreasing).
Standard descent ("-=") is correct, matching train_qfi.py.
"""

ansatz = {
    'star': (stargraph, stargraph_inv),
    'ring': (ringgraph, ringgraph_inv),
    'full': (fullgraph, fullgraph_inv),
}

# ---- edit these before running ----
TOPOLOGY = 'star'        # star | ring | full -- must match the checkpoint train_qfi.py wrote
LAYERS = 1                # must match the LAYERS used in train_qfi.py for this topology
POST_TOPOLOGY = None      # measurement ansatz's topology; None = same as TOPOLOGY (default/common case)
POST_LAYERS = None        # measurement ansatz's layer count; None = same as LAYERS
N = 5
x0, y0 = 6.498913663721062e-06, 0.0
LR = 0.05
STEPS_PER_CALL = 10       # use ~4 for 'full', heavier ansatz
REG = 1e-6
# ------------------------------------
TOPOLOGY = os.environ.get('TOPOLOGY', TOPOLOGY)  # only used by run_topology.sh, see train_qfi.py's docstring
LAYERS = int(os.environ.get('LAYERS', LAYERS))
POST_TOPOLOGY = os.environ.get('POST_TOPOLOGY', POST_TOPOLOGY)
POST_LAYERS = os.environ.get('POST_LAYERS', POST_LAYERS)
POST_LAYERS = int(POST_LAYERS) if POST_LAYERS is not None else None

POST_TOPOLOGY = POST_TOPOLOGY or TOPOLOGY
POST_LAYERS = LAYERS if POST_LAYERS is None else POST_LAYERS

fwd, _ = ansatz[TOPOLOGY]          # state-prep ansatz (theta) -- must match the loaded checkpoint
_, inv = ansatz[POST_TOPOLOGY]     # measurement ansatz (mu) -- independently chosen

prep_tag = f'{TOPOLOGY}_L{LAYERS}'
post_tag = f'{POST_TOPOLOGY}_L{POST_LAYERS}'
tag = f'{prep_tag}_post_{post_tag}'  # ALWAYS both, never collapsed, even when post==prep

CKPT_PREP = os.path.join(DATA_DIR, f'ckpt_qfi_best_{prep_tag}.pkl')
CKPT = os.path.join(DATA_DIR, f'ckpt_cfi_{tag}.pkl')
CKPT_BEST = os.path.join(DATA_DIR, f'ckpt_cfi_best_{tag}.pkl')
LOG = os.path.join(DATA_DIR, f'costs_cfi_{tag}.json')


def C_matrix(circuit):
    circuit = list(circuit)
    circuit[1] = [sensing_xy, N, 1.0, [x0, y0]]
    return np.real(cfim(circuit))


def cost(circuit):
    C = C_matrix(circuit) + REG * np.eye(2)
    return np.trace(np.linalg.inv(C))


if os.path.exists(CKPT):
    with open(CKPT, 'rb') as f:
        circuit = pickle.load(f)
    all_costs = json.load(open(LOG)) if os.path.exists(LOG) else []
    print(f"[{tag}] resumed, {len(all_costs)} steps so far")
else:
    with open(CKPT_PREP, 'rb') as f:
        prep_circuit = pickle.load(f)
    prep = prep_circuit[0]  # exact [fwd, N, num_layers, params] as trained
    sens = [sensing_xy, N, 1.0, [x0, y0]]
    post = [inv, N, POST_LAYERS, []]
    circuit = [prep, sens, post]
    vqc0 = VariationalCircuit(circuit)
    vqc0.add_params()
    circuit = vqc0.circuit
    circuit[0] = prep  # keep theta fixed (not re-randomized by add_params)
    all_costs = []
    print(f"[{tag}] starting fresh from trained theta, random mu")

if os.path.exists(CKPT_BEST):
    with open(CKPT_BEST, 'rb') as f:
        best_circuit = pickle.load(f)
    best_cost = cost(best_circuit)
else:
    best_circuit = circuit
    best_cost = cost(circuit)

t0 = time.time()
for step in range(STEPS_PER_CALL):
    grad = np.array(gradient_func(circuit, cost, [2], 'psr', None))
    circuit[2][3] = np.array(circuit[2][3]) - LR * grad
    c = cost(circuit)
    all_costs.append(c)
    if c < best_cost:
        best_cost = c
        best_circuit = [list(s) if i != 2 else [s[0], s[1], s[2], np.array(circuit[2][3])]
                         for i, s in enumerate(circuit)]
dt = time.time() - t0

with open(CKPT, 'wb') as f:
    pickle.dump(circuit, f)
with open(CKPT_BEST, 'wb') as f:
    pickle.dump(best_circuit, f)
json.dump(all_costs, open(LOG, 'w'))

C_now = C_matrix(best_circuit)
print(f"[{tag}] ran {STEPS_PER_CALL} steps in {dt:.1f}s, total so far: {len(all_costs)} steps")
print(f"[{tag}] current Tr(C_xy^-1) = {all_costs[-1]:.4e}, BEST so far = {best_cost:.4e}")
print(f"[{tag}] C_xy(best) =\n{C_now}")
