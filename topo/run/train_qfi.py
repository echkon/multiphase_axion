import sys, os, time, pickle, json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, CODE_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
import numpy as np
import onecircuit
from onecircuit import *
from onecircuit.metric import sld_qfim
from onecircuit.optimizing import gradient_func

"""
Edit the constants below, then just run:
    python3 train_qfi.py

Single, unified script for ALL topologies (star/ring/full) and ALL layer
counts -- no more per-topology folders, no more special-casing LAYERS=1
vs LAYERS!=1. Every data file gets an explicit "_{TOPOLOGY}_L{LAYERS}"
suffix, always, so nothing is ever ambiguous or silently overwritten
regardless of which topology/layer count you run.

TOPOLOGY/LAYERS below also fall back to the TOPOLOGY/LAYERS environment
variables if set (`os.environ.get(..., <the in-file default>)`), purely
so run_topology.sh can launch star/ring/full in parallel as separate
processes without three copies of this file. This changes NOTHING about
the normal manual workflow: for a single run, just edit the defaults
below and run `python3 train_qfi.py` with no environment variables at
all, same as before.

CARTESIAN multiparameter version (final design -- see conversation).

Background: jointly estimating (delta, alpha) directly (POLAR
parametrization of Chen et al.'s U_DM, Eq. 3.12) has an exact coordinate
singularity as delta -> 0: the Jacobian d(x,y)/d(delta,alpha), with
x=delta*cos(alpha), y=delta*sin(alpha), is singular there (determinant =
delta), so Q_alpha,alpha ~ delta^2 REGARDLESS of the ansatz or which
alpha0 you evaluate at. FIX: estimate the CARTESIAN pair (x, y) instead --
ordinary Cartesian coordinates have no origin singularity. `sensing_xy`
(onecircuit/circuit.py) differentiates x and y directly (z fixed at 0).

This script trains the state-prep ansatz (theta) to minimize the
multiparameter Cramer-Rao cost Tr(Q_xy^-1), evaluated at one FIXED
fiducial point (x0, y0) (x0=delta0, y0=0 -- alpha0's exact value doesn't
matter here, unlike the polar version).

train_cfi.py does the same for the classical FIM (Tr(C_xy^-1)) with
theta fixed from this script's best checkpoint.
"""

ansatz = {
    'star': (stargraph, stargraph_inv),
    'ring': (ringgraph, ringgraph_inv),
    'full': (fullgraph, fullgraph_inv),
}

# ---- edit these before running ----
TOPOLOGY = 'star'        # star | ring | full
LAYERS = 1                # number of ansatz layers
N = 5
delta0 = 6.498913663721062e-06
x0, y0 = delta0, 0.0     # fiducial point; alpha0's value is irrelevant in (x,y)
LR = 0.05
STEPS_PER_CALL = 10       # steps per invocation (use ~4 for 'full', heavier ansatz)
REG = 1e-6                # tiny regularizer, just for numerical safety
# ------------------------------------
TOPOLOGY = os.environ.get('TOPOLOGY', TOPOLOGY)  # only used by run_topology.sh, see docstring
LAYERS = int(os.environ.get('LAYERS', LAYERS))

fwd, inv = ansatz[TOPOLOGY]
tag = f'{TOPOLOGY}_L{LAYERS}'
CKPT = os.path.join(DATA_DIR, f'ckpt_qfi_{tag}.pkl')
CKPT_BEST = os.path.join(DATA_DIR, f'ckpt_qfi_best_{tag}.pkl')
LOG = os.path.join(DATA_DIR, f'costs_qfi_{tag}.json')


def Q_matrix(circuit):
    circuit = list(circuit)
    circuit[1] = [sensing_xy, N, 1.0, [x0, y0]]
    return np.real(sld_qfim(circuit))


def cost(circuit):
    Q = Q_matrix(circuit) + REG * np.eye(2)
    return np.trace(np.linalg.inv(Q))


if os.path.exists(CKPT):
    with open(CKPT, 'rb') as f:
        circuit = pickle.load(f)
    all_costs = json.load(open(LOG)) if os.path.exists(LOG) else []
    print(f"[{tag}] resumed, {len(all_costs)} steps so far")
else:
    prep = [fwd, N, LAYERS, []]
    sens = [sensing_xy, N, 1.0, [x0, y0]]
    post = [inv, N, LAYERS, []]
    circuit = [prep, sens, post]
    vqc0 = VariationalCircuit(circuit)
    vqc0.add_params()
    circuit = vqc0.circuit
    all_costs = []
    print(f"[{tag}] starting fresh (Cartesian x,y multiparameter QFI)")

if os.path.exists(CKPT_BEST):
    with open(CKPT_BEST, 'rb') as f:
        best_circuit = pickle.load(f)
    best_cost = cost(best_circuit)
else:
    best_circuit = circuit
    best_cost = cost(circuit)

t0 = time.time()
for step in range(STEPS_PER_CALL):
    grad = np.array(gradient_func(circuit, cost, [0], 'psr', None))
    circuit[0][3] = np.array(circuit[0][3]) - LR * grad
    c = cost(circuit)
    all_costs.append(c)
    if c < best_cost:
        best_cost = c
        best_circuit = [list(s) if i != 0 else [s[0], s[1], s[2], np.array(circuit[0][3])]
                         for i, s in enumerate(circuit)]
dt = time.time() - t0

with open(CKPT, 'wb') as f:
    pickle.dump(circuit, f)
with open(CKPT_BEST, 'wb') as f:
    pickle.dump(best_circuit, f)
json.dump(all_costs, open(LOG, 'w'))

Q_now = Q_matrix(best_circuit)
print(f"[{tag}] ran {STEPS_PER_CALL} steps in {dt:.1f}s, total so far: {len(all_costs)} steps")
print(f"[{tag}] current Tr(Q_xy^-1) = {all_costs[-1]:.4e}, BEST so far = {best_cost:.4e}")
print(f"[{tag}] Q_xy(best) =\n{Q_now}")
