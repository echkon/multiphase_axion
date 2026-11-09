# -*- coding: utf-8 -*-
""" Entanglement measure
"""

import qiskit
import tqix
import numpy as np
from math import comb

def concentratable_entanglement(
    qc: qiskit.QuantumCircuit):
    """Calculate Concentratable Entanglement (CE)

    Args:
        qc (qiskit.QuantumCircuit): circuit to be calculated

    Returns:
        (complex): CE
    """
    
    num_qubits = qc[0][1] # ancila qubits
    swap_qc = create_swap_test_state(qc.copy()) # full qubits
    
    list_measured = list(range(0, num_qubits))
    prob = tqix.vqa.circuits.measure(swap_qc, list_measured)
    return 1 - prob

def concentratable_entanglement_theor(
    qc: qiskit.QuantumCircuit):
    """Calculate Concentratable Entanglement (CE)

    Args:
        qc (qiskit.QuantumCircuit): circuit to be calculated

    Returns:
        (complex): CE
    """
    
    num_qubits = qc[0][1] # ancila qubits
    swap_qc = create_swap_test_state(qc.copy()) # full qubits
    
    list_measured = list(range(0, num_qubits))
    prob = tqix.vqa.circuits.measure_theor(swap_qc, list_measured)
    return 1 - prob

    
def create_swap_test_state(
    qc: qiskit.QuantumCircuit):
    """Create swap test state

    Args:
        qc (qiskit.QuantumCircuit): init circuit

    Returns:
        (qiskit.QuantumCircuit): swap test circuit
    """
    
    num_qubits = qc[0][1] # ancila qubits
    qubits_list_first = list(range(num_qubits, 2*num_qubits))
    qubits_list_second = list(range(2*num_qubits, 3*num_qubits))
    
    # Create swap test circuit
    swap_test_circuit = qiskit.QuantumCircuit(3*num_qubits, num_qubits)
    
    # Add initial circuit the first time
    swap_test_circuit = swap_test_circuit.compose(qc, qubits=qubits_list_first)
    
    # Add initial circuit the second time
    swap_test_circuit = swap_test_circuit.compose(qc, qubits=qubits_list_second)
    swap_test_circuit.barrier()
    
    # Add hadamard gate
    swap_test_circuit.h(list(range(0, num_qubits)))
    swap_test_circuit.barrier()
        
    for i in range(num_qubits):
        # Add control-swap gate
        swap_test_circuit.cswap(i, i+num_qubits, i+2*num_qubits)
    swap_test_circuit.barrier()
    
    # Add hadamard gate
    swap_test_circuit.h(list(range(0, num_qubits)))
    swap_test_circuit.barrier()
    
    return swap_test_circuit

def ce_max(n):
    """
    return maximun CE with input n-qubit
    Ref: arXiv:2207.11997v1 (2022)
    """
    ce = 0
    for j in range (0,n+1):
        ce += comb(n,j)/2**np.min([j,n-j])
    ce = ce/2**n
    ce = 1-ce
    return ce
