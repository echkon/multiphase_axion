# -*- coding: utf-8 -*-
""" Toolbox for quantum metrology
    Calculate Holevo bound using semidefinite programing
    Calculate SLD and RLD bounds
    Author: Le Bin Ho @ 2022
"""

import qiskit
import qiskit.quantum_info as qi
import numpy as np
import itertools
from numpy.linalg import inv, multi_dot, norm
from scipy.linalg import sqrtm, solve_sylvester, expm
import copy
from tqix.qobj import eigenx,dotx,daggx
from qiskit.quantum_info import SparsePauliOp , Operator, Statevector, DensityMatrix

from onecircuit.algorithm import QuantumCircuit
from onecircuit.util import step_size
from onecircuit.measurement import QuantumMeasurement

def chebyshev_probability(data, delta):
    return (sum(1 for x in data if x >= delta) / len(data))


def _stage_qc(stage):
    """Build the qiskit circuit for a single pipeline stage
    [func, num_qubits, coef, params]."""
    return stage[0](stage[1], stage[2], stage[3])


def _find_collective_noise_stage(circuit):
    """Return the index of a `collective_dephasing_qc` stage in `circuit`,
    or None if there isn't one."""
    from onecircuit.noise import collective_dephasing_qc
    for i, stage in enumerate(circuit):
        if stage[0] is collective_dephasing_qc:
            return i
    return None


def _safe_rho(circuit):
    """Return the final density matrix (plain ndarray) for `circuit`.

    Ordinary circuits are simply composed and evolved through Qiskit, exactly
    as before (`rho = qi.DensityMatrix(QuantumCircuit(circuit)).data`). If
    `circuit` contains a `collective_dephasing_qc` noise stage, that stage is
    instead applied directly to the density matrix via its exact closed
    form (numpy, O(d^2)), bypassing Qiskit's Kraus->SuperOp materialization
    for full-width multi-qubit channels. Qiskit always expands a multi-qubit
    Kraus channel to a d^2 x d^2 SuperOp before applying it (regardless of
    whether it is attached via qiskit_aer.noise.QuantumError or
    qiskit.quantum_info.Kraus, and regardless of whether it is applied via a
    circuit instruction or DensityMatrix.evolve), which needs O(d^4) memory
    and becomes prohibitive beyond N~7-8 qubits. Applying the (diagonal)
    collective-dephasing decay directly to rho avoids this entirely and
    scales with the same O(d^2)-O(d^3) cost as the rest of this module, so
    it does not add any bottleneck beyond the existing SLD eigendecomposition
    (see `collective_dephasing_qc` in noise.py for the physics derivation).
    """
    noise_idx = _find_collective_noise_stage(circuit)
    if noise_idx is None:
        circuit_qc = QuantumCircuit(circuit)
        return qi.DensityMatrix(circuit_qc).data

    num_qubits = circuit[0][1]

    qc_pre = qiskit.QuantumCircuit(num_qubits)
    for i in range(noise_idx):
        qc_pre = qc_pre & _stage_qc(circuit[i])
    rho = qi.DensityMatrix(qc_pre).data

    _, time_c, y_c = circuit[noise_idx][1], circuit[noise_idx][2], circuit[noise_idx][3]
    y_c = y_c[0] if isinstance(y_c, list) else y_c
    sigma2 = 1 - np.exp(-time_c * y_c)
    idx = np.array(list(itertools.product([0, 1], repeat=num_qubits)))
    z_tot = np.sum(np.where(idx == 0, -1, 1), axis=1)
    Delta = z_tot[:, None] - z_tot[None, :]
    rho = rho * np.exp(-sigma2 * Delta ** 2 / 8.0)

    qc_post = qiskit.QuantumCircuit(num_qubits)
    for i in range(noise_idx + 1, len(circuit)):
        qc_post = qc_post & _stage_qc(circuit[i])
    if len(qc_post.data) > 0:
        U = qi.Operator(qc_post).data
        rho = U @ rho @ U.conj().T

    return rho


def _safe_probs(circuit):
    """Diagonal (computational-basis) measurement probabilities for
    `circuit`, via `_safe_rho` -- see its docstring."""
    return np.real(np.diag(_safe_rho(circuit)))

def sld_qfim(circuit, method = 'eigens'):
    
    """calculate QFIM using SLD with different methods
    
    Args:
        - qc : initial circuit
        - qcirs: set of circuits (we always qc_add as model, no call here)

    Returns:
        - sld_qfim
    """
    if method == 'eigens':
        return sld_qfim_eigens(circuit)
    elif method == 'inverse':
        return sld_qfim_inverse(circuit)
    elif method == 'sylvester':
        return sld_qfim_sylvester(circuit)
    else:
        raise ValueError(
            "Method is not avilable: please use 'eigens', 'inverse', or 'sylvester' ")
    
    
    
def sld_qfim_inverse(circuit):
    
    """ calculate the QFIM bases SLD 
    
    Args:
        - qc : initial circuit
        - qcirs: set of circuits (we always qc_add as model, no call here)

    Returns:
        - qfim 
    """ 
    
    rho = _safe_rho(circuit) #rho after evolve
    grho = _grad_rho(circuit)

    d = len(grho) # number of paramaters
    H = np.zeros((d,d), dtype = complex)
    invM = _inv_M(rho)
    
    vec_grho = []  
    for i in range(d):
        vec_grho.append(_vectorize(grho[i]))

    for i in range(d):
        for j in range(d): 
            H[i,j] = 2*multi_dot([np.conjugate(vec_grho[i]).T, invM, vec_grho[j]]) 
    return np.real(H)  


def sld_qfim_sylvester(circuit):
    
    """ calculate the QFIM bases SLD 
    
    Args:
        - qc : initial circuit
        - qcirs: set of circuits (we always qc_add as model, no call here)

    Returns:
        - qfim 
    """ 
    
    rho = _safe_rho(circuit) #rho after evolve
    grho = _grad_rho(circuit)

    d = len(grho) # number of paramaters
    H = np.zeros((d,d), dtype = complex)
    L = sylvester(rho,rho,grho)
        
    for i in range(d):
        for j in range(d):
            H[i,j] = 0.5*np.trace(rho @ AntiCommutator(L[i],L[j]))

    return np.real(H)     


def sld_qfim_eigens(circuit):
    
    """ calculate the QFIM bases eigenvalues
    
    Args:
        - qc : initial circuit
        - qcirs: set of circuits (we always qc_add as model, no call here)

    Returns:
        - qfim 
    """ 
    
    rho = _safe_rho(circuit) #rho after evolve
    grho = _grad_rho(circuit)

    d = len(grho) # number of paramaters
    H = np.zeros((d,d), dtype = complex)

    eigens = eigenx(rho)
        
    for k in range(d):
        for l in range(d):
            H[k,l] = 0.0
            for i in range(len(eigens[0])):
                for j in range(len(eigens[0])):
                    de = eigens[0][i] + eigens[0][j]
                    if de > 10e-15:
                        num1 = dotx(daggx(eigens[1][i]),grho[k],eigens[1][j])
                        num2 = dotx(daggx(eigens[1][j]),grho[l],eigens[1][i])
                        H[k,l] += 2*np.real(num1*num2)/de
                        
    return np.real(H)  


def sld_bound(circuit):
    """ return the SLD bound 
    
    Args:
        - qc
        - qcirs

    Returns:
        - sld bound
    """
    
    d = len(circuit[1][3])
    W = np.identity(d)
    
    sld = sld_qfim(circuit)
    return np.real(np.trace(W @ inv(sld + np.eye(len(sld)) * 10e-10)))


def rld_qfim(circuit): 
    
    """ calculate the QFIM bases RLD 
    
    Analytical:
        H_ij = tr(rho*L_i*L_j^\dag)
            = tr(L_j^\dag*rho*L_i)

        submit rho*L_i = der_i(rho) : this is rld
        we get: H_ij = tr(L_j^\dag*dev_i(rho))
        
        apply tr(A^dag*B) = vec(A)^dag*vec(B), 
        we have: H_ij = vec(L_j)^dag*vec[dev_i(rho)]
        here: vec[L_j)] = (I x rho)^{-1)*vec(L_j)
        
    Args:
        - qc
        - qcirs

    Returns:
        - qfim via rld   
    """ 
    rho = _safe_rho(circuit) #rho after evolve
    grho = _grad_rho(circuit)

    d = len(grho) # number of estimated parameters
    R = np.zeros((d,d), dtype = complex)
    IR = _i_R(rho)
    
    vec_grho = []
    for i in range(d):
        vec_grho.append(_vectorize(grho[i]))
    
    for i in range(d):
        for j in range(d): 
            vecLj = np.conjugate(multi_dot([IR, vec_grho[j]])).T
            R[i,j] = multi_dot([vecLj, vec_grho[i]])  
            
    return R


def rld_bound(qc, qcirs):
    """ return the SLD bound 
    
    Args:
        - qc
        - cirs
        - params
        - coefs
        - W: weight matrix

    Returns:
        - rld bound
    """
    
    d = len(qcirs[1][2]) #[1]:u_phase,[1][2]:phases
    W = np.identity(d)
        
    rld = rld_qfim(qc, qcirs)
    invrld = inv(rld + np.eye(len(rld)) * 10e-10)
    R1 = np.trace(W @ np.real(invrld))
    R2 = norm(multi_dot([sqrtm(W), np.imag(invrld),sqrtm(W)])) 
    
    return R1 + R2


def cfim(circuit):
    """ return the classical fisher information matrix
    
    Args:
        - circuit
        
    Returns:
        - cfim
    """
    # measurements
    prob = _safe_probs(circuit)

    dpro = []
    d = len(circuit[1][3]) #[1]:sensing_qc,[1][3]:phases
    s = step_size

    #remove zeros indexs
    idxs = np.where(prob<=10e-18)
    prob = np.delete(prob, idxs)


    for i in range(0, d):
        # We use the parameter-shift rule explicitly
        # to compute the derivatives
        circuit1, circuit2 = copy.deepcopy(circuit), copy.deepcopy(circuit)
        circuit1[1][3][i] += s #[1]:sensing_qc,[1][3]:phases
        circuit2[1][3][i] -= s

        gr = (_safe_probs(circuit1) - _safe_probs(circuit2))/(2*s)
        gr = np.delete(gr, idxs)
        dpro.append(np.array(gr))
    
    matrix = np.zeros((d,d), dtype = float)
    for i in range(d):
        for j in range(d):      
            matrix[i,j] = np.sum(dpro[i] * dpro[j] / prob)

    return matrix
    

def cls_bound(circuit):
    """ return the classical bound 
    
    Args:
        - qc
        - qcirs
        - W: weight matrix

    Returns:
        - rld bound
    """
    
    #list2str = list(map(lambda f: f.__name__, cirs)) 
    #idx = list2str.index('u_phase') 
    #d = len(params[idx])
        
    clf = cfim(circuit)
    #print(f'CFIM = {clf}')
    W = np.identity(len(clf))
    
    return np.trace(W @ inv(clf + np.eye(len(clf)) * 10e-10))


def rd_bound(circuit):
    sld = sld_bound(circuit)
    cls = cls_bound(circuit)
    
    return 1-sld/cls


def _vectorize(rho):
    # return a vectorized of rho
    # rho: a matrices (data)
    vec_rho = np.reshape(rho, (len(rho)**2,1), order='F')
    return vec_rho


def _grad_rho(circuit):
    """ calculate drho by parameter-shift rule
    
    Args:
        - circuit
        
    Return:
        - gradient of state density w.r.t. all phases
    """
    
    dp = [] #array of matrices rho
    s = step_size
    
    for i in range(0, len(circuit[1][3])):
        circuit1, circuit2 = copy.deepcopy(circuit), copy.deepcopy(circuit)
        circuit1[1][3][i] += s
        circuit2[1][3][i] -= s

        dp.append((_safe_rho(circuit1) - _safe_rho(circuit2))/(2*s))

    return dp


def _inv_M(rho, epsilon = 10e-10): 
    """ return inverse matrix M 
        M = rho.conj()*I + I*rho.conj()
    
    Args:
        - quantum state rho (data)

    Returns:
        - inverse matrix M 
    """
    
    d = len(rho)
    M = np.kron(np.conj(rho), np.identity(d)) + np.kron(np.identity(d), rho)
    return inv(M + np.eye(len(M)) * epsilon)


def _i_R(rho, epsilon = 10e-10):
    """ return inverse of matrix R 
        R = I*rho.conj()
    
    Args:
        - quantum state rho (data)

    Returns:
        - inverse matrix R 
    """    
    d = len(rho)
    R = np.kron(np.identity(d), rho)
    return inv(R) #inv(R + np.eye(len(R)) * epsilon)


def sylvester(A,B,C):
    """ solve the sylvester function:
        AX + XB = 2*C # the symmetric logarithmic derivative
        here A,B = rho for SLD, i.e., rho @ L + L @ rho = 2*drho
            A = rho, B = 0 for RLD, i.e., rho @ L = 2*drho
            C = drho
    Args:
        - rho: input quantum state
        - drho: input derivative of quantum state
    Returns:
        - X operator
    """
    
    lenC = len(C) #we have several drho (gradient of rho)
    X = []
    
    for i in range(lenC):
        L = solve_sylvester(A, B, 2*C[i])
        X.append(L)
        
    #print(X)  
    return X

def AntiCommutator(A,B):
    # AB + BA
    return A @ B + B @ A   

def calculate_SLD(circuit):
    # to calculate SLD

    rho = _safe_rho(circuit) #rho after evolve
    grho = _grad_rho(circuit)

    d = len(grho) # number of paramaters
    Ls = []
    eigens = eigenx(rho)
    
    for i in range(d):
        L = np.zeros((len(rho),len(rho)), dtype = complex) 
        for k in range(len(eigens[0])):
            for l in range(len(eigens[0])):
                de = eigens[0][k] + eigens[0][l]
                if de > 10e-15:
                    coe1 = dotx(daggx(eigens[1][k]),grho[i],eigens[1][l])/de
                    coe2 = eigens[1][k]@daggx(eigens[1][l])
                    L += 2*coe1*coe2
        Ls.append(L)                
    return Ls                  

def fidelity_state(circuit):
    
    """calculate fidelity
    
    Args:
        - qc : initial circuit

    Returns:
        - fidelity of two operators:
        - 1/4^N * Tr[UVdagg]
    """
    qc = QuantumCircuit(circuit)
    rho = qi.DensityMatrix(qc).data #rho after evolve
    return np.abs(rho[0][0])**2

def frobenius_norm(circuit):
    """calculate the frobenius norm 
    
    Args:
        - uvd
        
    Returns:
        ||u-vd_dagger||      
    """
    u = circuit[0]
    vd = circuit[1]
    
    u = QuantumCircuit([u])
    vd = QuantumCircuit([vd])
    
    u_data = qi.Operator(u).data
    vd_data = qi.Operator(vd).data

    # Compute the difference matrix U - V^\dagger
    diff = (u_data.real - vd_data.conj().T.real) + (u_data.imag - vd_data.conj().T.imag)

    # Compute and return the Frobenius norm of the difference
    return np.linalg.norm(diff, 'fro')
    

def infidelity_state(uvd):
    return 1 - fidelity_state(uvd)

def fidelity_operator(uvd):
    """calculate fidelity
    
    Args:
        - qc : initial circuit

    Returns:
        - fidelity of two operators:
        - 1/4^N * Tr[UVdagg]
    """
    qc = QuantumCircuit(uvd)
    uvd_op = qi.Operator(qc).data
    fid = 1/4 ** qc.num_qubits * np.abs(np.trace(uvd_op))**2
    return fid

def infidelity_operator(uvd):
    return 1-fidelity_operator(uvd) 

def zz_ham(num_qubits):
    """
    Generates a Hamiltonian in the form of a SparsePauliOp matrix,
    following the pattern ZZ...I... + IZZ...I... + ... II...ZZ.

    Args:
        num_qubits: The number of qubits.

    Returns:
        A NumPy matrix representing the Hamiltonian.
    """

    if num_qubits < 2:
        return np.array([[0]])  # Or handle this case as you wish

    pauli_list = []
    for i in range(num_qubits - 1):
        pauli_string = ""
        for j in range(num_qubits):
            if j == i or j == i + 1:
                pauli_string += "Z"
            else:
                pauli_string += "I"
        pauli_list.append((pauli_string, 1.0))

    hamiltonian_op = SparsePauliOp.from_list(pauli_list)
    return hamiltonian_op

def zpz_ham(num_qubits):
    """
    Generates a Hamiltonian in the form of a SparsePauliOp matrix,
    following the pattern ZZ...I... + IZZ...I... + ... II...ZZ.

    Args:
        num_qubits: The number of qubits.

    Returns:
        A NumPy matrix representing the Hamiltonian.
    """

    pauli_list = [("Z" + (num_qubits - 1) * "I", 1.0), ("I"+ "Z" + "I"* (num_qubits - 2),1)]

    hamiltonian_op = SparsePauliOp.from_list(pauli_list)
    return hamiltonian_op

def expectation_value(circuit):
    from qiskit.primitives import Estimator  # lazy import -- see measurement.py

    num_qubits = circuit[0][1]
    #ham = zz_ham(num_qubits)
    ham = zpz_ham(num_qubits)

    qc = QuantumCircuit(circuit)
    sampler = Estimator()
    job = sampler.run(circuits=[qc], observables=[ham]) 
    result = job.result() # Extract the expectation value 
    expectation_value = result.values[0]

    return expectation_value

def overlap(circuit1, circuit2):
    from qiskit.primitives import Estimator  # lazy import -- see measurement.py

    num_qubits = circuit[0][1]
    ham = SparsePauliOp.from_list([("Z" * num_qubits, 1.0)])

    qc = QuantumCircuit(circuit)
    sampler = Estimator()
    job = sampler.run(circuits=[qc], observables=[ham]) 
    result = job.result() # Extract the expectation value 
    expectation_value = result.values[0]

    return expectation_value

def expectation_value_var(circuit):
    from qiskit.primitives import Estimator  # lazy import -- see measurement.py
    num_qubits = circuit[0][1]

    # Define the Hamiltonian (observable)
    ham = SparsePauliOp.from_list([("Z" * num_qubits, 1.0)])

    qc = QuantumCircuit(circuit)
    sampler = Estimator()

    # Compute expectation value ⟨H⟩
    job1 = sampler.run(circuits=[qc], observables=[ham]) 
    result1 = job1.result()
    expectation_value = result1.values[0]

    # Compute expectation value of H^2, which is just ⟨H²⟩
    ham_squared = ham @ ham  # Squaring the Hamiltonian
    job2 = sampler.run(circuits=[qc], observables=[ham_squared])
    result2 = job2.result()
    expectation_value_squared = result2.values[0]

    # Compute variance: Var(H) = ⟨H²⟩ - ⟨H⟩²
    variance = expectation_value_squared - expectation_value**2

    return variance
    
def expectation_value_exp(circuit):
    """
    Calculates the expectation value of exp(-i*ham) for a given circuit.

    Args:
        circuit (QuantumCircuit): The quantum circuit.

    Returns:
        complex: The expectation value.
    """
    num_qubits = circuit[0][1]  # Get the number of qubits directly from the QuantumCircuit
    ham = generate_hamiltonian(num_qubits)

    # Convert SparsePauliOp to a dense matrix
    ham_matrix = ham.to_matrix()

    # Calculate exp(-1j * ham) using scipy.linalg.expm
    expm_ham_matrix = expm(-1j * ham_matrix)

    # Convert the resulting matrix back to an Operator for use with Estimator
    expm_ham_op = Operator(expm_ham_matrix)

    from qiskit.primitives import Estimator  # lazy import -- see measurement.py
    sampler = Estimator()
    qc = QuantumCircuit(circuit)
    job = sampler.run(circuits=[qc], observables=[expm_ham_op])
    result = job.result()
    expectation_value = np.abs(result.values[0])

    return expectation_value


def overlap(circuit1, circuit2):

    qc1 = QuantumCircuit(circuit1)
    qc2 = QuantumCircuit(circuit2)
    # Convert circuits to statevectors
    psi1 = Statevector(qc1).data
    psi2 = Statevector(qc2).data

    # Compute the inner product (overlap)
    overlap_value = np.abs(np.vdot(psi1, psi2))**2 

    return overlap_value

def cost_density(circuit):
    num_qubits = circuit[0][1]
    ham = SparsePauliOp.from_list([("ZZ" + (num_qubits - 2) * "I", 1.0)]).to_matrix()
    
    qc = QuantumCircuit(circuit)
    rho = DensityMatrix(qc).data  # Get the density matrix representation
    
    cost = np.trace(rho @ ham).real 

    return cost 