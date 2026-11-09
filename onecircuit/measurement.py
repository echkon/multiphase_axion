# -*- coding: utf-8 -*-
""" Everything belong to circuit
"""

import qiskit
import tqix
import numpy as np
from onecircuit.algorithm import QuantumCircuit
from onecircuit.util import meas_method

def QuantumMeasurement(sensor, method = meas_method):
    if method == 'theory':
        return measure_theor(sensor)
    elif method == 'theory0':
        return measure_theor0(sensor)
    elif method == 'experiment':
        return measureall(sensor)
    else:
        raise ValueError(
            'No measurement method was given')
    
    
def measure(sensor):
    """Measuring the quantum circuit which fully measurement gates

    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (Numpy array): List of measured qubit

    Returns:
        - float: Frequency of 00..0 cbit
    """    
    n = len(qubits)
    if cbits == []:
        cbits = qubits.copy()
    for i in range(0, n):
        qc.measure(qubits[i], cbits[i])
    
    counts = qiskit.execute(
            qc, backend = tqix.vqa.constants.backend,
            shots = tqix.vqa.constants.num_shots).result().get_counts()

    return counts.get("0" * len(qubits), 0) / tqix.vqa.constants.num_shots


def measure_theor(sensor):
    """Measuring the quantum circuit which fully measurement gates

    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (Numpy array): List of measured qubit

    Returns:
        - float: Frequency of 00..0 cbit by Born rule
    """    
    rho = qiskit.quantum_info.DensityMatrix(sensor).data
    probs = []
    for i in range(len(rho)):
        probs.append(np.real(rho[i,i]))
    return np.array(probs)


def measure_theor0(sensor):
    """Measuring the quantum circuit which fully measurement gates

    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (Numpy array): List of measured qubit
    
    Returns:
        - float: Frequency of 00..0 cbit by Born rule
    """
    rho = qiskit.quantum_info.DensityMatrix(sensor).data
    prob = np.real(rho[0,0])
    return prob


def measureall(sensor):
    """Measuring the quantum circuit which fully measurement gates

    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (Numpy array): List of measured qubit

    Returns:
        - float: all cbits
    """
    
    from qiskit.primitives import Sampler  # lazy import: only needed for this
                                            # shot-based ('experiment') method,
                                            # so a newer qiskit without the old
                                            # V1 Sampler doesn't break every
                                            # other (theory-based) code path
    sensor.measure_all()
    sampler = Sampler()
    results = sampler.run([sensor], shots=1000).result().quasi_dists[0]

    # Calculate the probabilities for each bit value
    probs = []
    for i in range(16):
        probs.append(results.get(i, 0.0))  # Assigns the default value 0.0 if key is missing
    return np.array(probs)


def measure_born(qc: qiskit.QuantumCircuit):
    """Measuring the quantum circuit which fully measurement gates

    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (Numpy array): List of measured qubit

    Returns:
        - float: all cbits
    """
    
    n = qc.num_qubits
    d = 2**n
    probs = []

    rho = qiskit.quantum_info.DensityMatrix.from_instruction(qc.copy()).data

    for i in range(0, d):
        probs.append(np.real(rho[i,i]))

    return  np.array(probs) 
