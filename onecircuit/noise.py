# -*- coding: utf-8 -*-
""" Variational quantum metrology
    Author: Le Bin Ho @ 2023
"""

import qiskit
import numpy as np
import itertools

from qiskit.quantum_info import Kraus
#from qiskit.providers.aer.noise import QuantumError
from qiskit_aer.noise import QuantumError


# various noise via Kraus operators
def dephasing_qc(num_qubits: int, 
                time: float = 1.0, 
                y: np.ndarray = []) -> qiskit.QuantumCircuit:
    """Add dephasing to the circuit

    Args:
        - num_qubits = number of qubits
        - time: time coefs
        - y: gamma in params
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    lamb = 1 - np.exp(-time*y)
    
    # kraus operators
    k1 = np.array([[1, 0],[0, np.sqrt(1 - lamb)]])
    k2 = np.array([[0, 0],[0, np.sqrt(lamb)]])
    
    noise_ops = Kraus([k1,k2])
    kraus_to_error = QuantumError(noise_ops) 
    
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def bit_flip(num_qubits: int, 
            time: float = 1.0, 
            y: np.ndarray = []) -> qiskit.QuantumCircuit:
    """Add bit flip to the circuit

    Args:
        - num_qubit: number of qubits
        - t: time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    lamb = 1 - np.exp(-time*y)
    
    noise_ops = Kraus([np.sqrt(1-lamb) * np.array([[1, 0], [0, 1]]),
                   np.sqrt(lamb) * np.array([[0, 1], [1, 0]])])
    kraus_to_error = QuantumError(noise_ops)
    
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def phase_flip(num_qubits: int, 
            time: float = 1.0, 
            y: np.ndarray = []) -> qiskit.QuantumCircuit:
    """Add phase flip to the circuit

    Args:
        - num qubits
        - t: time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    lamb = 1 - np.exp(-time*y)
    
    noise_ops = Kraus([np.sqrt(1-lamb) * np.array([[1, 0], [0, 1]]),
                   np.sqrt(lamb) * np.array([[1, 0], [0, -1]])])
    kraus_to_error = QuantumError(noise_ops)
    
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def depolarizing(num_qubits: int, 
            time: float = 1.0, 
            y: np.ndarray = []) -> qiskit.QuantumCircuit:
    """Add depolarizing to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t: time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    lamb = 1 - np.exp(-time*y)
    
    noise_ops = Kraus([np.sqrt(1-lamb) * np.array([[1, 0], [0, 1]]),
                       np.sqrt(lamb/3.) * np.array([[0, 1], [1, 0]]),
                       np.sqrt(lamb/3.) * np.array([[0, -1j], [1j, 0]]),
                       np.sqrt(lamb/3.) * np.array([[1, 0], [0, -1]])
                        ])
    kraus_to_error = QuantumError(noise_ops)
    
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def markovian(num_qubits: int, 
            time: float = 1.0, 
            y: np.ndarray = []) -> qiskit.QuantumCircuit:
    """Add bit flip to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t: time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    qt = 1 - np.exp(-y*time) 
    
    k1 = np.array([[np.sqrt(1-qt) ,0], [0,1]])
    k2 = np.array([[np.sqrt(qt),0], [0,0]])
    
    noise_ops = Kraus([k1,k2])
    kraus_to_error = QuantumError(noise_ops)

    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.append(kraus_to_error,[i])

    return qc


def collective_dephasing_qc(num_qubits: int,
                time: float = 1.0,
                y: np.ndarray = [],
                num_kraus: int = 21) -> qiskit.QuantumCircuit:
    """Add COLLECTIVE (shared-bath) dephasing to the circuit.

    Unlike `dephasing_qc` above, which applies an INDEPENDENT noise
    realization to each qubit (one separate single-qubit Kraus channel per
    qubit), this channel applies a SINGLE shared random Z-rotation
    exp(-i*xi*Z_tot/2), with Z_tot = sum_i sigma_z^(i), to ALL qubits at
    once, where xi ~ Gaussian(0, sigma^2) is drawn ONCE per shot and seen
    identically by every qubit. This models a common-mode noise source --
    e.g. qubits that all read out the same shared magnon/cavity mode, so
    that fluctuations of the mode itself dephase every qubit together
    rather than independently.

    Closed form: averaging exp(-i*xi*Z_tot/2) rho exp(i*xi*Z_tot/2) over
    xi~N(0,sigma^2) damps the coherence between computational-basis states
    differing by total-Z eigenvalue Delta by exp(-sigma^2*Delta^2/8). This
    is implemented as a genuine multi-qubit Kraus channel (applied to all
    num_qubits at once, NOT one channel per qubit), using Gauss-Hermite
    quadrature to discretize the Gaussian average into `num_kraus` diagonal
    Kraus operators. The default num_kraus=21 keeps the quadrature error
    negligible for sigma up to ~0.3 and num_qubits up to a few tens.

    Args:
        - num_qubits = number of qubits
        - time: time coefs
        - y: rate parameter; sigma^2 = 1 - exp(-time*y), the same
          lamb2y-style convention as the other channels in this module, here
          giving the collective dephasing variance sigma^2
        - num_kraus: number of Gauss-Hermite quadrature points (default 21)

    Return
        - qc
    """
    y = y[0] if type(y) == list else y
    sigma2 = 1 - np.exp(-time * y)
    sigma = np.sqrt(sigma2)

    # total-Z eigenvalue of each computational basis state (|g>=-1, |e>=+1
    # per qubit, matching the sign convention used for H_q^(R),N in the
    # manuscript)
    idx = np.array(list(itertools.product([0, 1], repeat=num_qubits)))
    z_tot = np.sum(np.where(idx == 0, -1, 1), axis=1)

    # Gauss-Hermite quadrature for (1/(sigma*sqrt(2*pi))) * int dxi
    # exp(-xi^2/(2*sigma^2)) * g(xi), via xi = sqrt(2)*sigma*x
    nodes, weights = np.polynomial.hermite.hermgauss(num_kraus)
    xi_vals = np.sqrt(2) * sigma * nodes
    w_vals = weights / np.sqrt(np.pi)

    kraus_list = []
    for xi, w in zip(xi_vals, w_vals):
        phases = np.exp(-1j * xi * z_tot / 2.0)
        kraus_list.append(np.sqrt(w) * np.diag(phases))

    noise_ops = Kraus(kraus_list)
    kraus_to_error = QuantumError(noise_ops)

    qc = qiskit.QuantumCircuit(num_qubits)
    qc.append(kraus_to_error, list(range(num_qubits)))

    return qc


def non_markovian(num_qubits: int,
            time: float = 1.0, 
            y: np.ndarray = []) -> qiskit.QuantumCircuit:
    """Add bit flip to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t: time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    tc = 20.0 #fixed in rXiv:2305.08289
    qt = 1 - np.exp(-y*time**2/(2*tc))
    
    k1 = np.array([[np.sqrt(1-qt) ,0], [0,1]])
    k2 = np.array([[np.sqrt(qt),0], [0,0]])
    
    noise_ops = Kraus([k1,k2])
    kraus_to_error = QuantumError(noise_ops)  
    
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.append(kraus_to_error,[i])
    
    return qc