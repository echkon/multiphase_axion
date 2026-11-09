# -*- coding: utf-8 -*-
""" Everything belong to circuit

CALLING CONVENTION (applies to every stage function used in a
VariationalCircuit pipeline, i.e. every function that can appear as
circuit[i][0] in a stage list [func, num_qubits, coef, params]):

    func(num_qubits, coef, params) -> qiskit.QuantumCircuit

`coef` and `params` play two DIFFERENT roles, and it is important not to
mix them up, because `params` is exactly the array that gets
finite-difference-differentiated -- but WHAT that derivative means depends
on which kind of stage the function builds:

  - Ansatz / state-preparation stages (e.g. `stargraph`, `stargraph_inv`,
    `fullgraph`, `ringgraph` in ansatz.py):
      coef   = num_layers (an int, FIXED -- how many repeated layers of
               the ansatz to stack; never differentiated).
      params = the ansatz's trainable rotation angles (length depends on
               num_qubits/num_layers). These get differentiated by
               `VariationalCircuit.fit(train_opt=[...])` during TRAINING,
               to search for the state that maximizes QFI/CFI. They carry
               no direct physical meaning by themselves.

  - Sensing / encoding stages (e.g. `sensing_qc`, `sensing_dm` below):
    here params is not a training knob -- it is (or determines) the
    PHYSICAL parameter we want to estimate (e.g. the axion-induced
    rotation angle delta). `onecircuit.metric.sld_qfim`/`cfim` differentiate
    with respect to `params` too, but for a completely different reason:
    to compute how distinguishable rho(delta) is from rho(delta+eps), i.e.
    the actual Quantum/Classical Fisher Information for delta -- NOT a
    training update.
      coef   = whatever is FIXED and irrelevant to that Fisher-information
               question (e.g. the interrogation time `time`, or, for
               `sensing_dm`, the DM phase `alpha` -- alpha is a real,
               physical, per-shot-random quantity, but it is NOT the
               parameter we are estimating, so it must never appear in
               `params` or `sld_qfim`/`cfim` will silently compute
               spurious extra "derivatives" with respect to it (see
               `sensing_dm`'s docstring for why this matters).
      params = the physical parameter(s) actually being estimated.

  - Noise stages (e.g. `dephasing_qc`, `collective_dephasing_qc` in
    noise.py): coef = time, params = [y] (a fixed decay-rate constant).
    Never differentiated by anything in practice, even though the
    machinery would technically allow it.

Getting this split wrong for a sensing stage is a real, easy-to-hit bug:
if a quantity is put into `params` that should not be estimated (or is
redundant with another physical parameter), sld_qfim/cfim will return a
bigger matrix than you want and picking the "obviously right-looking"
diagonal entry from it (e.g. [2,2]) can silently give the WRONG number.
See `sensing_qc` vs `sensing_dm` below for a concrete example of this
trap and the fix.
"""

import qiskit
import numpy as np
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from tqix import *

def tensor_product(state, num_qubits: int):
    tensor_product = state
    I = eyex(2)
    for _ in range(1, num_qubits):
        tensor_product = np.kron(tensor_product, I)
    
    return tensor_product

def ghz_qc(num_qubits: int,
           *args,
           **kwargs) -> qiskit.QuantumCircuit:
    """Create GHZ state

    Args:
        num_qubits: number of qubits
        *kwargs: for other parameter
        **kwargs: for other parameters

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    qc = qiskit.QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(0, num_qubits - 1):
        qc.cx(0, i + 1)
    qc.barrier()
    
    return qc

def exc_qc(num_qubits: int,
           *args,
           **kwargs) -> qiskit.QuantumCircuit:
    """Create excited state

    Args:
        num_qubits: number of qubits
        params (np.ndarray): number of parameters
        **kwargs: for other parameters

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    qc = qiskit.QuantumCircuit(num_qubits)
    
    for i in range(0, num_qubits):
        qc.x(i)
    qc.barrier()

    return qc

def gh3_qc(num_qubits: int,
           *args,
           **kwargs) -> qiskit.QuantumCircuit:
    """Create GH3 state

    Args:
        num_qubits: number of qubits
        *kwargs: for other parameter
        **kwargs: for other parameters

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    qc = qiskit.QuantumCircuit(num_qubits)

    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    tensor_x = tensor_product(sx, num_qubits)
    tensor_y = tensor_product(sy, num_qubits)
    tensor_z = tensor_product(sz, num_qubits)

    _, eigenstate_x = eigenx(tensor_x)
    _, eigenstate_y = eigenx(tensor_y)
    _, eigenstate_z = eigenx(tensor_z)

    ghx = 1/np.sqrt(2) * (eigenstate_x[0] + eigenstate_x[-1])
    ghy = 1/np.sqrt(2) * (eigenstate_y[0] + eigenstate_y[-1])
    ghz = 1/np.sqrt(2) * (eigenstate_z[0] + eigenstate_z[-1])
    gh3 = normx(ghx + ghy + ghz)

    ghz = ghz.flatten().tolist()
    gh3 = gh3.flatten().tolist()

    qc.initialize(gh3, list(range(num_qubits)))
    op = Statevector(qc).data
    qc.barrier()

    return qc

def stargraph_qc(num_qubits, *args, **kwargs)-> qiskit.QuantumCircuit:
    
    """Create start graph 

    Args:
        - num_qubits: number of qubits
        - *args, **kwargs: None

    Returns:
        - QuantumCircuit: the added circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    
    for i in range(0, num_qubits):
        qc.h(i)
    for i in range(1, num_qubits):
        qc.cz(0, i)
    qc.barrier()   
    
    return qc

def ringgraph_qc(num_qubits, *args, **kwargs)-> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(0, num_qubits - 1, 2):
        qc.cz(i, i + 1)
    if n % 2 == 1:
        for i in range(1, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        qc.cz(0, num_qubits - 1)
        qc.barrier()
    return qc

def sensing_qc(num_qubits: int,
            time: float,
            params: np.ndarray) -> qiskit.QuantumCircuit:
    """Apply the SAME general single-qubit rotation phase_gate(time, params)
    to every qubit, where params = [x, y, z] are treated as THREE
    INDEPENDENT differentiable parameters by sld_qfim/cfim (see the module
    docstring at the top of this file).

    This is the right stage to use ONLY when the physical encoding really
    is along a single, fixed, already-known axis, so that two of the three
    components are held at exactly 0 (e.g. params=[0, 0, delta], the pure
    Z-rotation used for the magnon/Ramsey encoding H_q^(R)=chi<n>sigma_z in
    Manu1/Manu2). In that case sld_qfim(circuit)[2,2] correctly equals the
    QFI for delta, because x=y=0 identically and only z=delta varies.

    It is the WRONG stage to use when the encoding direction is not fixed
    along a single Pauli axis by construction -- e.g. the axion-photon
    coupling of Chen et al. (Eq. 3.12), whose rotation axis is
    (cos(alpha), sin(alpha), 0) for a random per-shot phase alpha. If you
    pass params=[delta*cos(alpha), delta*sin(alpha), 0] here, sld_qfim
    will return a 3x3 matrix that differentiates x, y, AND z separately
    (including the z-component, which is pinned at 0 and physically
    irrelevant) -- taking [2,2] from that matrix answers a different
    question (sensitivity to a hypothetical, nonexistent z-rotation) and
    silently gives the wrong QFI. Use `sensing_dm`/`phase_gate_dm` below
    instead for that case: they take alpha as a separate, fixed argument
    (not folded into params), so delta is the only differentiable
    parameter.

    Args:
        num_qubits (int): number of qubits
        time (float): interrogation time, fixed (not differentiated)
        params (np.ndarray): [x, y, z] rotation-generator components,
            ALL differentiated by sld_qfim/cfim

    Returns:
        qiskit.QuantumCircuit: circuit with phase_gate applied to every qubit
    """
    qc = qiskit.QuantumCircuit(num_qubits)

    for i in range(0,num_qubits):
        qc.append(phase_gate(time, params),[i])

    return qc

def sensing_xy(num_qubits: int,
               time: float,
               params: np.ndarray) -> qiskit.QuantumCircuit:
    """CARTESIAN 2-parameter variant of `sensing_qc`: differentiates ONLY
    x, y (z is fixed at 0.0, not differentiated) -- see conversation.

    Motivation: `sensing_dm_joint`'s (delta, alpha) POLAR parametrization
    of the same physical U_DM has an exact coordinate singularity at
    delta=0 -- the Jacobian d(x,y)/d(delta,alpha) = [[cos a, -delta sin a],
    [sin a, delta cos a]] is singular as delta->0 (determinant = delta），
    which is why Q_alpha,alpha ~ delta^2 there (verified numerically,
    independent of alpha0's value). This is a property of the (delta,
    alpha) coordinate choice near the origin, not necessarily of the
    underlying physics: x = delta*cos(alpha), y = delta*sin(alpha) are
    perfectly ordinary CARTESIAN coordinates with no origin singularity,
    so the QFIM in (x, y) should be well-conditioned even as delta -> 0.
    This function lets you compute/train against that Cartesian QFIM
    directly, then convert (x_hat, y_hat) -> delta_hat = sqrt(x^2+y^2),
    alpha_hat = atan2(y, x) as a LAST step (with the standard caveat that
    magnitude estimation near the origin from noisy Cartesian estimates
    has a well-known positive bias, e.g. as in Rician-distributed radar/
    MRI magnitude estimation -- so this is not automatically free lunch,
    but is a structurally different, worth-testing estimator).

    Args:
        num_qubits (int): number of qubits
        time (float): interrogation time, fixed (not differentiated)
        params (np.ndarray): [x, y] -- BOTH differentiated; z is
            implicitly fixed at 0.0 inside this function

    Returns:
        qiskit.QuantumCircuit: circuit with phase_gate(time, [x,y,0])
            applied to every qubit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    x, y = params[0], params[1]

    for i in range(0, num_qubits):
        qc.append(phase_gate(time, [x, y, 0.0]), [i])

    return qc

def phase_gate(t, phis): #unitary
    x,y,z = phis
    w = np.sqrt(x*x + y*y + z*z) #omega
    wt = t * w

    # NOTE: sin(wt)/w has a removable singularity at w=0 (limit = t), but
    # naive "np.sin(wt)/w" raises ZeroDivisionError / gives nan there. This
    # matters in practice: the QFI's internal finite-difference derivative
    # (onecircuit.metric._grad_rho, step_size=1e-4 from util.py) evaluates
    # phase_gate at delta +/- step_size, and for realistic small encoding
    # angles (delta ~ 1e-6 or smaller, as for a real axion-photon coupling)
    # one of those two perturbed points can land exactly on / very close to
    # w=0. Use np.sinc (normalized: sinc(x)=sin(pi x)/(pi x)) to get the
    # correct, numerically stable value sin(wt)/w = t*sinc(wt/pi) at w=0
    # (and everywhere else) instead.
    sinc_wt_over_w = t * np.sinc(wt / np.pi)

    u11 = np.cos(wt) - 1j*z*sinc_wt_over_w
    u12 = (-1j*x - y)*sinc_wt_over_w
    u21 = (-1j*x + y)*sinc_wt_over_w
    u22 = np.cos(wt) + 1j*z*sinc_wt_over_w

    umat =  np.array([[u11,u12],[u21, u22]])
    gate = UnitaryGate(umat, 'sensing_qc')

    return gate

def sensing_dm(num_qubits: int,
                alpha: float,
                phase: np.ndarray) -> qiskit.QuantumCircuit:
    """Apply the axion-photon (dark-matter) encoding U_DM to every qubit,
    via phase_gate_dm(alpha, phase) -- see its docstring for the exact
    unitary. This matches Chen et al., Eq. (3.12), up to an alpha -> alpha+pi
    relabeling (physically irrelevant: alpha is uniform on [0, 2*pi), and
    cos^2, sin^2, cos*sin are all invariant under that shift, so QFI/CFI
    computed from this function are identical to Eq. (3.12) evaluated
    directly -- verified numerically against `phase_gate`-based encodings).

    THIS FUNCTION HAS EXACTLY ONE DIFFERENTIABLE PARAMETER: delta =
    phase[0]. alpha is the real, per-shot-random DM phase (constant within
    one coherence time, different from shot to shot -- Chen et al. Sec.
    II) but it is NOT the parameter being estimated, so it is correctly
    passed through the `alpha` argument (the pipeline's `coef` slot: fixed,
    never differentiated by sld_qfim/cfim), not folded into `phase`
    (the pipeline's `params` slot, which IS differentiated).

    Concretely, as a VariationalCircuit pipeline stage:
        [sensing_dm, num_qubits, alpha, [delta]]
    gives sld_qfim(circuit) a 1x1 matrix whose single entry is the true
    QFI for delta at that alpha (do NOT use `sensing_qc` with
    params=[delta*cos(alpha), delta*sin(alpha), 0] instead -- that treats
    x, y, z as three independent parameters and picking any single entry,
    e.g. [2,2], out of the resulting 3x3 matrix does NOT give the QFI for
    delta; see `sensing_qc`'s docstring for why).

    Args:
        num_qubits (int): number of qubits
        alpha (float): DM phase for this shot, fixed (not differentiated)
        phase (np.ndarray): [delta] -- length-1 array, the single physical
            parameter differentiated by sld_qfim/cfim

    Returns:
        qiskit.QuantumCircuit: circuit with U_DM(delta, alpha) applied to
            every qubit
    """
    qc = qiskit.QuantumCircuit(num_qubits)

    for i in range(0,num_qubits):
        qc.append(phase_gate_dm(alpha,phase),[i])  # alpha out of phase

    return qc

def sensing_dms(num_qubits: int,
                time: float,
                phase: np.ndarray) -> qiskit.QuantumCircuit:
    """Variant of `sensing_dm` where alpha is instead read from phase[0]
    (with the actual delta nested one level deeper, at phase[1][0] once it
    reaches `phase_gate_dm`). NOTE: unlike `sensing_dm`, if this is used as
    a VariationalCircuit stage with params=phase, alpha ends up INSIDE the
    differentiated params array too -- sld_qfim/cfim would then compute an
    (unwanted) derivative with respect to alpha as well, alongside delta.
    Only use this for contexts where that 2-parameter differentiation is
    actually intended; otherwise prefer `sensing_dm`, which keeps alpha
    out of `params` correctly.

    Args:
        num_qubits (int): number of qubits
        time (float): unused directly here (kept for pipeline-signature
            consistency with other sensing stages; phase_gate_dm has no
            explicit time argument)
        phase (np.ndarray): [alpha, [delta, ...]] -- phase[0] is used as
            alpha, phase[1] is passed on as the `phase` argument of
            phase_gate_dm (whose own phase[0] = delta)

    Returns:
        qiskit.QuantumCircuit: circuit with U_DM applied to every qubit
    """
    qc = qiskit.QuantumCircuit(num_qubits)

    for i in range(0,num_qubits):
        qc.append(phase_gate_dm(phase[0],np.array(phase[1])),[i])  # alpha = phase[0]

    return qc

def sensing_dm_joint(num_qubits: int,
                      time: float,
                      phase: np.ndarray) -> qiskit.QuantumCircuit:
    """Multiparameter variant of `sensing_dm`: BOTH delta and alpha are
    differentiated by sld_qfim/cfim, giving a genuine 2x2 QFIM/CFIM for
    the joint (delta, alpha) multiparameter estimation problem -- see
    conversation ("multiparameter metrology" pivot: instead of treating
    alpha as a fixed nuisance coef, estimate it jointly with delta and
    design theta/mu to minimize Tr(F^-1) over the pair).

    NOTE: `sensing_dms` above looks similar but is NOT usable for this --
    its `phase = [alpha, [delta]]` is a NESTED list, and the generic
    finite-difference loop in `onecircuit.metric._grad_rho`/`cfim` does
    `circuit[1][3][i] += step_size` for each i, which fails/is wrong on a
    nested entry (`phase[1]` is itself a list, not a scalar). This
    function instead uses a FLAT params list, `phase = [delta, alpha]`,
    matching what `_grad_rho`/`cfim` actually expect (a flat list of
    scalars, one per differentiated parameter) -- both entries are
    genuinely independent scalars, differentiated correctly by index.

    Concretely, as a VariationalCircuit pipeline stage:
        [sensing_dm_joint, num_qubits, 1.0, [delta, alpha]]
    gives sld_qfim(circuit) / cfim(circuit) a proper 2x2 matrix: index 0
    = delta, index 1 = alpha, off-diagonal entries are the real
    delta-alpha cross Fisher information.

    Args:
        num_qubits (int): number of qubits
        time (float): unused (kept for pipeline-signature consistency;
            phase_gate_dm has no explicit time argument)
        phase (np.ndarray): [delta, alpha] -- BOTH differentiated by
            sld_qfim/cfim

    Returns:
        qiskit.QuantumCircuit: circuit with U_DM(delta, alpha) applied to
            every qubit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    delta, alpha = phase[0], phase[1]

    for i in range(0, num_qubits):
        qc.append(phase_gate_dm(alpha, [delta]), [i])

    return qc

def sensing_dm_i(num_qubits: int,
                index: list,
                phase: np.ndarray) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits)
    
    for i in index:
        qc.append(phase_gate_dm(phase,0.0),[i])
        
    return qc

def phase_gate_dm(alpha: float,
                  phase: np.ndarray):
    """Dark-matter (axion-photon) single-qubit encoding unitary:

        U_DM = [[cos(delta),          i*exp(-i*alpha)*sin(delta)],
                [i*exp(i*alpha)*sin(delta), cos(delta)          ]]

    delta = phase[0] is the physical parameter being estimated (delta =
    eta*tau, set by the unknown axion-photon coupling g_agg -- see
    axion_coupling.py). alpha is the random DM phase for this shot
    (fixed/known for the purposes of building this one gate, but not the
    thing being estimated).

    This is Chen et al.'s U_DM, Eq. (3.12), up to alpha -> alpha + pi
    (an overall sign flip on the off-diagonal terms) -- physically
    irrelevant since alpha is uniform on [0, 2*pi) and every quantity that
    matters (cos^2(alpha), sin^2(alpha), cos(alpha)*sin(alpha)) is
    invariant under that shift. Verified numerically to agree with a
    directly-built Chen-Eq.-3.12 matrix up to that sign convention.

    Args:
        alpha (float): random DM phase for this shot (fixed, not the
            estimated parameter)
        phase (np.ndarray): [delta, ...] -- phase[0] = delta, the
            estimated rotation angle

    Returns:
        UnitaryGate: the single-qubit U_DM(delta, alpha) gate
    """
    delta = phase[0]

    u11 = np.cos(delta)
    u12 = 1j * np.exp(-1j * alpha) * np.sin(delta)
    u21 = 1j * np.exp(1j * alpha) * np.sin(delta)
    u22 = np.cos(delta)

    U_DM =  np.array([[u11,u12],[u21, u22]])
    gate = UnitaryGate(U_DM, 'sensing_dm')

    return gate
