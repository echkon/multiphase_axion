"""
>>> OneCircuit: One solution for all quantum circuit needs
________________________________
>>> copyright (c) 2024 and later
>>> authors: Binho Le
>>> all rights reserved
________________________________
"""

import qiskit
import numpy as np
from onecircuit.util import init_params

def hea_ansatz(num_qubits: int, 
            num_layers: int, 
            params: np.ndarray) -> qiskit.QuantumCircuit:
    
    qc = qiskit.QuantumCircuit(num_qubits)

    if len(params) != num_layers*3*num_qubits:
        raise ValueError(
            f'The number of parameter must be {num_layers*3*num_qubits}')
    
    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(params[j], i)
            qc.rx(params[j+1], i)
            qc.ry(params[j+2], i)
            j += 3
        for i in range(0, num_qubits-1, 1):
            qc.cz(i, i+1)

    return qc

def hea_ansatz2(num_qubits: int, 
            num_layers: int, 
            params: np.ndarray) -> qiskit.QuantumCircuit:
    
    qc = qiskit.QuantumCircuit(num_qubits)

    if len(params) != num_layers*3*num_qubits:
        raise ValueError(
            f'The number of parameter must be {num_layers*3*num_qubits}')
    
    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(params[j]**2, i)
            qc.rx(params[j+1]**2, i)
            qc.ry(params[j+2]**2, i)
            j += 3
        for i in range(0, num_qubits-1, 1):
            qc.cz(i, i+1)

    return qc
        
def random_pauli_ansatz(num_qubits: int, 
            num_layers: int, 
            params: np.ndarray) -> qiskit.QuantumCircuit:
    
    if len(params) != num_layers*num_qubits:
        raise ValueError(
            f'The number of parameter must be {num_layers*num_qubits}')
    
    qc = qiskit.QuantumCircuit(num_qubits)

    j=0
    for i in range(0, num_qubits):
        qc.rz(np.pi/4, i) 
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            rand_pauli = np.random.randint(0,3)
            if rand_pauli == 0:
                qc.rx(params[j], i)
            if rand_pauli == 1:
                qc.ry(params[j], i)
            if rand_pauli == 2:
                qc.rz(params[j], i)
            j += 1
        for i in range(0, num_qubits-1, 1):
            qc.cz(i, i+1)

    return qc

def linegraph(num_qubits: int,
              num_layers: int,
              params: np.ndarray) -> qiskit.QuantumCircuit:
    """Create linear graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """

    qc = qiskit.QuantumCircuit(num_qubits)
    
    if len(params) != num_layers*(6*num_qubits):
        raise ValueError(
            f'The number of parameter must be {num_layers*(6*num_qubits)}')
    
    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits, 1):
            qc.rx(params[j], i)
            qc.ry(params[j+1], i)
            qc.rz(params[j+2], i)
            j += 3
        for i in range(0, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        for i in range(1, num_qubits - 1, 1):
            qc.rx(params[j], i)
            qc.ry(params[j+1], i)
            qc.rz(params[j+2], i)
            j += 3
        for i in range(1, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        qc.barrier()
        
    return qc
    
def linegraph_inv(num_qubits: int,
                  num_layers: int,
                  params: np.ndarray)-> qiskit.QuantumCircuit:
    """Create inverse linear graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    return linegraph(num_qubits,num_layers,params).inverse()

def ringgraph(num_qubits: int,
              num_layers: int,
              params: np.ndarray)-> qiskit.QuantumCircuit:
    """Create ring graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """

    qc = qiskit.QuantumCircuit(num_qubits)
    if len(params) != num_layers*(6*num_qubits):
        raise ValueError(
            f'The number of parameter must be {num_layers*(6*num_qubits)}')

    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.rx(params[j], i)
            qc.ry(params[j+1], i)
            qc.rz(params[j+2], i)
            j += 3
        for i in range(0, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        if num_qubits % 2 == 1:
            for i in range(0, num_qubits - 1):
                qc.rx(params[j], i)
                qc.ry(params[j+1], i)
                qc.rz(params[j+2], i)
                j += 3
        else:
            for i in range(0, num_qubits):
                qc.rx(params[j], i)
                qc.ry(params[j+1], i)
                qc.rz(params[j+2], i)
                j += 3
        for i in range(1, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        if num_qubits % 2 == 1:
            qc.rx(params[j], num_qubits - 1)
            qc.ry(params[j+1], num_qubits - 1)
            qc.rz(params[j+2], num_qubits - 1)
            j += 3
        qc.cz(0, num_qubits - 1)
        qc.barrier()

    return qc
    
def ringgraph_inv(num_qubits: int,
                  num_layers: int,
                  params: np.ndarray)-> qiskit.QuantumCircuit:
    """Create inverse ring graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    return ringgraph(num_qubits,num_layers,params).inverse()

def stargraph(num_qubits: int,
              num_layers: int,
              params: np.ndarray) -> qiskit.QuantumCircuit:
    """Create star graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """

    qc = qiskit.QuantumCircuit(num_qubits)
    
    if len(params) != num_layers*(6*num_qubits-6):
        raise ValueError(
            f'The number of parameter must be {num_layers*(6*num_qubits-6)}')
    
    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.rx(params[j], i)
            qc.ry(params[j+1], i)
            qc.rz(params[j+2], i)
            j += 3
        qc.cz(0, 1)
        for i in range(2, num_qubits):
            qc.rx(params[j], 0)
            qc.ry(params[j+1], 0)
            qc.rz(params[j+2], 0)
            j += 3
            qc.cz(0, i)
        qc.barrier()

    return qc

def stargraph_inv(num_qubits: int,
                  num_layers: int,
                  params: np.ndarray)-> qiskit.QuantumCircuit:
    """Create invert star graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """

    return stargraph(num_qubits,num_layers,params).inverse()

def fullgraph(num_qubits: int,
              num_layers: int,
              params: np.ndarray) -> qiskit.QuantumCircuit:
    """Create a FIXED-DEGREE-4 circulant graph ansatz -- NOT a genuine
    all-to-all (complete) graph.

    Each qubit i is entangled directly with exactly its 4 nearest
    neighbors on an index ring: i-2, i-1, i+1, i+2 (mod num_qubits).
    This is a deliberate hardware constraint: a true complete graph needs
    degree num_qubits-1 per qubit, which only fits within a 4-connection-
    per-qubit hardware budget up to num_qubits=5 -- beyond that it is not
    physically realizable. The circulant design keeps every qubit's
    degree fixed at 4 REGARDLESS of num_qubits, so it scales to any
    num_qubits while staying within that budget (it also reduces to the
    exact complete graph K_5 when num_qubits=5, since neighbors
    i-2,i-1,i+1,i+2 already cover all 4 other qubits in that case -- so
    this is a strict, backward-compatible generalization of the old
    all-to-all fullgraph for the num_qubits=5 case used throughout this
    project's runs so far).

    Edges: (i, i+1 mod N) for all i [the ring edges, same as ringgraph]
           (i, i+2 mod N) for all i [the added "skip-one" chord edges]
    giving 2*num_qubits edges total, each visited once, in that fixed
    order, applying a fresh RX,RY,RZ to BOTH endpoints of an edge
    immediately before its CZ (same per-edge pattern as the old
    all-to-all fullgraph used).

    Args:
        num_qubits: number of qubits (must be >= 5; for fewer qubits the
            +-1/+-2 neighbor sets overlap or become degenerate)
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """

    if num_qubits < 5:
        raise ValueError(
            'fullgraph (fixed-degree-4 circulant) requires num_qubits >= 5; '
            'for fewer qubits the i+-1/i+-2 neighbor sets overlap or '
            'duplicate edges.')

    qc = qiskit.QuantumCircuit(num_qubits)

    edges = [(i, (i + 1) % num_qubits) for i in range(num_qubits)] + \
            [(i, (i + 2) % num_qubits) for i in range(num_qubits)]
    num_edges = len(edges)  # = 2 * num_qubits, independent of topology size

    if len(params) != num_layers * 6 * num_edges:
        raise ValueError(
            f'The number of parameter must be {num_layers * 6 * num_edges}')

    j = 0
    for l in range(0, num_layers, 1):
        for (a, b) in edges:
            qc.rx(params[j], a)
            qc.ry(params[j+1], a)
            qc.rz(params[j+2], a)
            j += 3
            qc.rx(params[j], b)
            qc.ry(params[j+1], b)
            qc.rz(params[j+2], b)
            j += 3
            qc.cz(a, b)
        qc.barrier()

    return qc

def fullgraph_inv(num_qubits: int,
                  num_layers: int,
                  params: np.ndarray)-> qiskit.QuantumCircuit:
    """Create inverse of the fixed-degree-4 circulant graph ansatz (see
    `fullgraph`'s docstring for the graph structure).

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    return fullgraph(num_qubits,num_layers,params).inverse()

def stargraph9(num_qubits: int,
               num_layers: int,
               params: np.ndarray) -> qiskit.QuantumCircuit:
    """Create a star2 graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """

    qc = qiskit.QuantumCircuit(num_qubits)
    
    if len(params) != num_layers*(63):
        raise ValueError(
            f'The number of parameter must be {num_layers*(63)}')
    
    if num_qubits % 2 == 0:  # N9: qubits: 0,1,...,8
        num_branches = int(num_qubits/2)
    else:
        num_branches = int((num_qubits-1)/2)  # N9: num_branches = 4

    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits, 1):
            qc.rx(params[j], i)
            qc.ry(params[j+1], i)
            qc.rz(params[j+2], i)
            j += 3
        for i in range(1, num_branches+1, 1):  # N9: i = 1, 2, 3, 4
            qc.rx(params[j], 0)
            qc.ry(params[j+1], 0)
            qc.rz(params[j+2], 0)
            j += 3
            qc.cz(0, 2*i-1)  # N9: (0,1), (0,3), (0,5), (0,7)
            if num_qubits > 2*i:
                qc.rx(params[j], 2*i-1)
                qc.ry(params[j+1], 2*i-1)
                qc.rz(params[j+2], 2*i-1)
                j += 3
                qc.rx(params[j], 2*i)
                qc.ry(params[j+1], 2*i)
                qc.rz(params[j+2], 2*i)
                j += 3
                qc.cz(2*i-1, 2*i)  # N9: (1,2), (3,4), (5,6), (7,8)
        qc.barrier() 
        
    return qc

def stargraph9_inv(num_qubits: int,
               num_layers: int,
               params: np.ndarray)-> qiskit.QuantumCircuit:
    """Create inverse star2 graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    return stargraph9(num_qubits,num_layers,params).inverse()

def starringgraph9(num_qubits: int,
                   num_layers: int,
                   params: np.ndarray) -> qiskit.QuantumCircuit:
    """Create a star-ring graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """

    qc = qiskit.QuantumCircuit(num_qubits)
    
    if len(params) != num_layers*(84):  # Only for N9, as is difficult to derive a general solution
        raise ValueError(
            f'The number of parameter must be {num_layers*(84)}')

    depth = 0
    while True:  # N9: depth = 1
        max_nodes = (2*depth+1) ** 2
        if num_qubits <= max_nodes:
            break
        depth += 1

    j = 0
    for l in range(0, num_layers, 1):
        for i in range(1, depth+1, 1):  # N9: i = 1
            first_node = (2*i-1) ** 2  # N9: first_node = 1
            max_nodes = (2*i+1) ** 2 - 1  # N9: last_node = 8
            if i != depth:
                last_node = max_nodes
            else:
                last_node = num_qubits-1
            for n in range(first_node, last_node+1, 1):  # N9: i = 1: n = 1,2,...,8
                if n == first_node:
                    qc.rx(params[j], 0)
                    qc.ry(params[j+1], 0)
                    qc.rz(params[j+2], 0)
                    j += 3
                    qc.rx(params[j], n)
                    qc.ry(params[j+1], n)
                    qc.rz(params[j+2], n)
                    j += 3
                    qc.cz(0, n)  # (0,1)
                else:
                    qc.rx(params[j], n-1)
                    qc.ry(params[j+1], n-1)
                    qc.rz(params[j+2], n-1)
                    j += 3
                    qc.rx(params[j], n)
                    qc.ry(params[j+1], n)
                    qc.rz(params[j+2], n)
                    j += 3
                    qc.cz(n-1, n)  # (1,2),(2,3),...,(7,8)
                    if n % 2 == 1:
                        qc.rx(params[j], 0)
                        qc.ry(params[j+1], 0)
                        qc.rz(params[j+2], 0)
                        j += 3
                        qc.rx(params[j], n)
                        qc.ry(params[j+1], n)
                        qc.rz(params[j+2], n)
                        j += 3
                        qc.cz(0, n)  # (0,3),(0,5),(0,7)
            if num_qubits > last_node:
                qc.rx(params[j], first_node)
                qc.ry(params[j+1], first_node)
                qc.rz(params[j+2], first_node)
                j += 3
                qc.rx(params[j], last_node)
                qc.ry(params[j+1], last_node)
                qc.rz(params[j+2], last_node)
                j += 3
                qc.cz(first_node, last_node)  # (1,8)

        qc.barrier()
        
    return qc

def starringgraph9_inv(num_qubits: int,
                       num_layers: int,
                       params: np.ndarray)-> qiskit.QuantumCircuit:
    """Create inverse star_ring graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    return starringgraph9(num_qubits,num_layers,params).inverse()

def fullgraph9(num_qubits: int,
               num_layers: int,
               params: np.ndarray) -> qiskit.QuantumCircuit:
    """Create a full2 graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """

    qc = qiskit.QuantumCircuit(num_qubits)
    
    if len(params) != num_layers*(108):  # Only for N9, as is difficult to derive a general solution
        raise ValueError(
            f'The number of parameter must be {num_layers*(108)}')

    depth = 0
    while True:  # N9: depth = 1
        max_nodes = (2*depth+1) ** 2
        if num_qubits <= max_nodes:
            break
        depth += 1

    j = 0
    for l in range(0, num_layers, 1):
        for i in range(1, depth+1, 1):  # N9: i = 1
            first_node = (2*i-1) ** 2  # N9: first_node = 1
            max_nodes = (2*i+1) ** 2 - 1  # N9: last_node = 8
            if i != depth:
                last_node = max_nodes
            else:
                last_node = num_qubits-1
            for n in range(first_node, last_node+1, 1):  # N9: i = 1: n = 1,2,...,8
                if n == first_node:
                    qc.rx(params[j], 0)
                    qc.ry(params[j+1], 0)
                    qc.rz(params[j+2], 0)
                    j += 3
                    qc.rx(params[j], n)
                    qc.ry(params[j+1], n)
                    qc.rz(params[j+2], n)
                    j += 3
                    qc.cz(0, n)  # (0,1)
                else:
                    qc.rx(params[j], n-1)
                    qc.ry(params[j+1], n-1)
                    qc.rz(params[j+2], n-1)
                    j += 3
                    qc.rx(params[j], n)
                    qc.ry(params[j+1], n)
                    qc.rz(params[j+2], n)
                    j += 3
                    qc.cz(n-1, n)  # (1,2),(2,3),...,(7,8)
                    if n % 2 == 1:
                        qc.rx(params[j], 0)
                        qc.ry(params[j+1], 0)
                        qc.rz(params[j+2], 0)
                        j += 3
                        qc.rx(params[j], n)
                        qc.ry(params[j+1], n)
                        qc.rz(params[j+2], n)
                        j += 3
                        qc.cz(0, n)  # (0,3),(0,5),(0,7)
                    if n % 2 == 0:
                        if n == last_node:
                            qc.rx(params[j], first_node+1)
                            qc.ry(params[j+1], first_node+1)
                            qc.rz(params[j+2], first_node+1)
                            j += 3
                            qc.rx(params[j], last_node)
                            qc.ry(params[j+1], last_node)
                            qc.rz(params[j+2], last_node)
                            j += 3
                            qc.cz(first_node+1, last_node)  # (2,8)
                        else:
                            qc.rx(params[j], n)
                            qc.ry(params[j+1], n)
                            qc.rz(params[j+2], n)
                            j += 3
                            qc.rx(params[j], n+2)
                            qc.ry(params[j+1], n+2)
                            qc.rz(params[j+2], n+2)
                            j += 3
                            qc.cz(n, n+2)  # (2,4), (4,6), (6,8)
            if num_qubits > last_node:
                qc.rx(params[j], first_node)
                qc.ry(params[j+1], first_node)
                qc.rz(params[j+2], first_node)
                j += 3
                qc.rx(params[j], last_node)
                qc.ry(params[j+1], last_node)
                qc.rz(params[j+2], last_node)
                j += 3
                qc.cz(first_node, last_node)  # (1,8)
        qc.barrier()
        
    return qc

def fullgraph9_inv(num_qubits: int,
                   num_layers: int,
                   params: np.ndarray)-> qiskit.QuantumCircuit:
    """Create inverse full2 graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    return fullgraph9(num_qubits,num_layers,params).inverse()

def star_ring_graph(num_qubits: int,
                    num_layers: int,
                    params: np.ndarray) -> qiskit.QuantumCircuit:
    """Create a star_ring graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """

    qc = qiskit.QuantumCircuit(num_qubits)
    
    if len(params) != 3*num_qubits+num_layers*(84):  # Only for N9, as is difficult to derive a general solution
        raise ValueError(
            f'The number of parameter must be {3*num_qubits+num_layers*(84)}')

    depth = 0
    while True:  # N9: depth = 1
        max_nodes = (2*depth+1) ** 2
        if num_qubits <= max_nodes:
            break
        depth += 1

    j = 0
    for i in range(0, num_qubits, 1):
        qc.rx(params[j], i)
        qc.ry(params[j+1], i)
        qc.rz(params[j+2], i)
        j += 3
    for l in range(0, num_layers, 1):
        for i in range(1, depth+1, 1):  # N9: i = 1
            first_node = (2*i-1) ** 2  # N9: first_node = 1
            max_nodes = (2*i+1) ** 2 - 1  # N9: last_node = 8
            if i != depth:
                last_node = max_nodes
            else:
                last_node = num_qubits-1
            for n in range(first_node, last_node+1, 1):  # N9: i = 1: n = 1,2,...,8
                if n == first_node:
                    qc.cz(0, n)  # (0,1)
                    qc.rx(params[j], 0)
                    qc.ry(params[j+1], 0)
                    qc.rz(params[j+2], 0)
                    j += 3
                    qc.rx(params[j], n)
                    qc.ry(params[j+1], n)
                    qc.rz(params[j+2], n)
                    j += 3
                else:
                    qc.cz(n-1, n)  # (1,2),(2,3),...,(7,8)
                    qc.rx(params[j], n-1)
                    qc.ry(params[j+1], n-1)
                    qc.rz(params[j+2], n-1)
                    j += 3
                    qc.rx(params[j], n)
                    qc.ry(params[j+1], n)
                    qc.rz(params[j+2], n)
                    j += 3
                    if n % 2 == 1:
                        qc.cz(0, n)  # (0,3),(0,5),(0,7)
                        qc.rx(params[j], 0)
                        qc.ry(params[j+1], 0)
                        qc.rz(params[j+2], 0)
                        j += 3
                        qc.rx(params[j], n)
                        qc.ry(params[j+1], n)
                        qc.rz(params[j+2], n)
                        j += 3
            if num_qubits > last_node:
                qc.cz(first_node, last_node)  # (1,8)
                qc.rx(params[j], first_node)
                qc.ry(params[j+1], first_node)
                qc.rz(params[j+2], first_node)
                j += 3
                qc.rx(params[j], last_node)
                qc.ry(params[j+1], last_node)
                qc.rz(params[j+2], last_node)
                j += 3

        qc.barrier()
        
    return qc

def star_ring_graph_inv(num_qubits: int,
                        num_layers: int,
                        params: np.ndarray)-> qiskit.QuantumCircuit:
    """Create inverse star_ring graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    return star_ring_graph(num_qubits,num_layers,params).inverse()

def full2graph(num_qubits: int,
               num_layers: int,
               params: np.ndarray) -> qiskit.QuantumCircuit:
    """Create a full2 graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """

    qc = qiskit.QuantumCircuit(num_qubits)
    
    if len(params) != 3*num_qubits+num_layers*(108):  # Only for N9, as is difficult to derive a general solution
        raise ValueError(
            f'The number of parameter must be {3*num_qubits+num_layers*(108)}')

    depth = 0
    while True:  # N9: depth = 1
        max_nodes = (2*depth+1) ** 2
        if num_qubits <= max_nodes:
            break
        depth += 1

    j = 0
    for i in range(0, num_qubits, 1):
        qc.rx(params[j], i)
        qc.ry(params[j+1], i)
        qc.rz(params[j+2], i)
        j += 3
    for l in range(0, num_layers, 1):
        for i in range(1, depth+1, 1):  # N9: i = 1
            first_node = (2*i-1) ** 2  # N9: first_node = 1
            max_nodes = (2*i+1) ** 2 - 1  # N9: last_node = 8
            if i != depth:
                last_node = max_nodes
            else:
                last_node = num_qubits-1
            for n in range(first_node, last_node+1, 1):  # N9: i = 1: n = 1,2,...,8
                if n == first_node:
                    qc.cz(0, n)  # (0,1)
                    qc.rx(params[j], 0)
                    qc.ry(params[j+1], 0)
                    qc.rz(params[j+2], 0)
                    j += 3
                    qc.rx(params[j], n)
                    qc.ry(params[j+1], n)
                    qc.rz(params[j+2], n)
                    j += 3
                else:
                    qc.cz(n-1, n)  # (1,2),(2,3),...,(7,8)
                    qc.rx(params[j], n-1)
                    qc.ry(params[j+1], n-1)
                    qc.rz(params[j+2], n-1)
                    j += 3
                    qc.rx(params[j], n)
                    qc.ry(params[j+1], n)
                    qc.rz(params[j+2], n)
                    j += 3
                    if n % 2 == 1:
                        qc.cz(0, n)  # (0,3),(0,5),(0,7)
                        qc.rx(params[j], 0)
                        qc.ry(params[j+1], 0)
                        qc.rz(params[j+2], 0)
                        j += 3
                        qc.rx(params[j], n)
                        qc.ry(params[j+1], n)
                        qc.rz(params[j+2], n)
                        j += 3
                    if n % 2 == 0:
                        if n == last_node:
                            qc.cz(first_node+1, last_node)  # (2,8)
                            qc.rx(params[j], first_node+1)
                            qc.ry(params[j+1], first_node+1)
                            qc.rz(params[j+2], first_node+1)
                            j += 3
                            qc.rx(params[j], last_node)
                            qc.ry(params[j+1], last_node)
                            qc.rz(params[j+2], last_node)
                            j += 3
                        else:
                            qc.cz(n, n+2)  # (2,4), (4,6), (6,8)
                            qc.rx(params[j], n)
                            qc.ry(params[j+1], n)
                            qc.rz(params[j+2], n)
                            j += 3
                            qc.rx(params[j], n+2)
                            qc.ry(params[j+1], n+2)
                            qc.rz(params[j+2], n+2)
                            j += 3
            if num_qubits > last_node:
                qc.cz(first_node, last_node)  # (1,8)
                qc.rx(params[j], first_node)
                qc.ry(params[j+1], first_node)
                qc.rz(params[j+2], first_node)
                j += 3
                qc.rx(params[j], last_node)
                qc.ry(params[j+1], last_node)
                qc.rz(params[j+2], last_node)
                j += 3
        qc.barrier()
        
    return qc

def full2graph_inv(num_qubits: int,
                   num_layers: int,
                   params: np.ndarray)-> qiskit.QuantumCircuit:
    """Create inverse full2 graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    return full2graph(num_qubits,num_layers,params).inverse()

def linegraph_eco(num_qubits: int,
                  num_layers: int,
                  params: np.ndarray) -> qiskit.QuantumCircuit:
    """Create linear graph ansatz economy

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    qc = qiskit.QuantumCircuit(num_qubits)
    if len(params) != num_layers*num_qubits:
        raise ValueError(
            f'The number of parameter must be {num_layers*num_qubits}')

    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(params[j], i)
            j += 1
        for i in range(0, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        for i in range(1, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        qc.barrier()
    
    return qc

def linegraph_eco_inv(num_qubits: int,
                      num_layers: int,
                      params: np.ndarray)-> qiskit.QuantumCircuit:
    """Create inverse linear graph ansatz economy

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    return linegraph_eco(num_qubits,num_layers,params).inverse()

def ringgraph_eco(num_qubits: int,
                  num_layers: int, 
                  params: np.ndarray) -> qiskit.QuantumCircuit:
    """Create ring graph ansatz economy

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """

    qc = qiskit.QuantumCircuit(num_qubits)

    if len(params) != num_layers*num_qubits:
        raise ValueError(
            f'The number of parameter must be {num_layers*num_qubits}')

    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(params[j], i)
            j += 1
        for i in range(0, num_qubits-1):
            qc.cz(i, i + 1)
        qc.cz(0, num_qubits - 1)
        qc.barrier()
    
    return qc

def ringgraph_eco_inv(num_qubits: int,
                      num_layers: int,
                      params: np.ndarray)-> qiskit.QuantumCircuit:
    """Create inverse ring graph ansatz economy

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    return ringgraph_eco(num_qubits,num_layers,params).inverse()

def stargraph_eco(num_qubits: int,
                  num_layers: int,
                  params: np.ndarray) -> qiskit.QuantumCircuit:
    """Create star graph ansatz economy

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """

    qc = qiskit.QuantumCircuit(num_qubits)
    
    if len(params) != num_layers*num_qubits:
        raise ValueError(
            f'The number of parameter must be {num_layers*num_qubits}')
    
    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(params[j], i)
            j += 1
        for i in range(1, num_qubits):
            qc.cz(0, i)
        qc.barrier()   
        
    return qc

def stargraph_eco_inv(num_qubits: int,
                      num_layers: int,
                      params: np.ndarray)-> qiskit.QuantumCircuit:
    """Create invert star graph ansatz economy

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """

    return stargraph_eco(num_qubits,num_layers,params).inverse()

def squeezing(
    num_qubits,
    num_layers: int,
    params: np.ndarray)-> qiskit.QuantumCircuit:
    
    """Create squeezing ansatz

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        thetas (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """ 
    
    qc = qiskit.QuantumCircuit(num_qubits)
    if len(params) != num_layers*num_qubits*(num_qubits+1):
        raise ValueError(
            f'The number of parameter must be {num_layers*num_qubits*(num_qubits+1)}')
        
    k = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(params[k], i)
            k += 1
            
        # GMS_z gate
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                qc.rzz(params[k], i, j)
                k += 1
        # RX gate
        for i in range(0, num_qubits):
            qc.rx(params[k], i)
            k += 1

        # GMS_x gate
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                qc.rxx(params[k], i, j)   
                k += 1
        qc.barrier()    
    return qc

def squeezing_inv(
    num_qubits,
    num_layers: int,
    params: np.ndarray)-> qiskit.QuantumCircuit:
    
    """Create squeezing ansatz

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        thetas (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """ 
    
    return squeezing(num_qubits,num_layers,params).inverse()

def random_ansatz(num_qubits: int, 
            cnot_pairs, 
            params: np.ndarray) -> qiskit.QuantumCircuit:
    
    """Create random ansatz

    Args:
        num_qubits: number of qubits
        cnot_pairs: random set of cnot position
        params (np.ndarray): number of parameters

    Returns:
        One random quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    
    D = len(cnot_pairs)

    if len(params) != 6*D:
        raise ValueError(
            f'The number of parameter must be {6*D}')    
    
    j = 0
    for (control, target) in cnot_pairs:
        qc.rx(params[j], control)
        qc.ry(params[j+1], control)
        qc.rz(params[j+2], control)
        qc.rx(params[j+3], target)
        qc.ry(params[j+4], target)
        qc.rz(params[j+5], target)
        qc.cx(control, target)
        qc.barrier() 
        j += 6
        
    return qc

def create_num_params(single_sensor):
    """ to create a list of number of parameters:
    
    Args:
        - single_sensor = [qc_name, num_qubits, num_layers, params]

    Returns:
        - total number of paramaters
    """

    qc_name = single_sensor[0].__name__
    num_qubits = single_sensor[1]
    num_layers = single_sensor[2]
    cnot_pairs = single_sensor[2]
    
    if qc_name == 'linegraph' or qc_name == 'linegraph_inv':
        num_params = num_layers*6*num_qubits
    elif qc_name == 'ringgraph' or qc_name == 'ringgraph_inv':
        num_params = num_layers*6*num_qubits
    elif qc_name == 'stargraph' or qc_name == 'stargraph_inv':
            num_params = num_layers*6*(num_qubits-1)
    elif qc_name == 'fullgraph' or qc_name == 'fullgraph_inv':
        num_params = num_layers*3*(num_qubits-1)*num_qubits
    elif qc_name == 'stargraph9' or qc_name == 'stargraph9_inv':
        num_params = num_layers*(63)
    elif qc_name == 'starringgraph9' or qc_name == 'starringgraph9_inv':
        num_params = num_layers*(84)
    elif qc_name == 'fullgraph9' or qc_name == 'fullgraph9_inv':
        num_params = num_layers*(108)
    elif qc_name == 'ghz_qc' or qc_name == 'gh3_qc':
        num_params = 0
    elif qc_name == 'exc_qc':
        num_params = num_qubits
    elif qc_name == 'star_ring_graph' or qc_name == 'star_ring_graph_inv':
        num_params = 3*num_qubits+num_layers*(84)
    elif qc_name == 'full2graph' or qc_name == 'full2graph_inv':
        num_params = 3*num_qubits+num_layers*(108)
    elif qc_name == 'stargraph_eco' or qc_name == 'stargraph_eco_inv':
        num_params = num_layers*num_qubits
    elif qc_name == 'ringgraph_eco' or qc_name == 'ringgraph_eco_inv':
        num_params = num_layers*num_qubits
    elif qc_name == 'linegraph_eco' or qc_name == 'linegraph_eco_inv':
        num_params = num_layers*num_qubits
    elif qc_name == 'squeezing' or qc_name == 'squeezing_inv' :
        num_params = num_layers*num_qubits*(num_qubits+1)  
    elif qc_name == 'random_ansatz' or qc_name == 'random_ansatz_inv':
        num_params = 6*len(cnot_pairs)
    elif qc_name == 'hea_ansatz' or qc_name == 'hea_ansatz_inv' or qc_name == 'hea_ansatz2':
        num_params = 3*num_layers*num_qubits
    elif qc_name == 'random_pauli_ansatz' or qc_name == 'random_pauli_ansatz_inv':
        num_params = num_layers*num_qubits    
    #elif qc_name == 'u_phase':
    #    num_params = 3 # fixed
    else:
        raise ValueError(
            'No ansatz was given.') 
    
    return int(num_params)


def create_params(single_sensor):
    
    """ to create a list of parameters:
    
    Args:
        - single_sensor = [qc_name, num_qubits, num_layers, params]
        - value: initial value
        
    Returns:
        - update params
    """
    
    num_params = create_num_params(single_sensor)

    if init_params == -1: # -1 means random
        params = (np.random.uniform(0, 2 * np.pi, num_params))    
    else:
        params = [init_params] * num_params
        
    return params