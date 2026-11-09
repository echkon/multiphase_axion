import sys, os, pickle, copy, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'..','..'))
import qiskit
from onecircuit import *
from onecircuit.util import step_size
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error
N=5; T1=100.0; T2=100.0; CZ=0.005; x0=6.498913663721062e-06
D=os.path.join(HERE,'..','data')+'/'
prep=pickle.load(open(D+'ckpt_qfi_best_network_noise_ring_L1_T1100_T2100_czerr0p005.pkl','rb'))[0]
post=pickle.load(open(D+'ckpt_cfi_best_network_noise_ring_L1_post_ring_L2_T1100_T2100_czerr0p005.pkl','rb'))[1]
edges=sorted(set(tuple(sorted(e)) for e in [(i,i+1) for i in range(N-1)]+[(0,N-1)]))
def idle_qc(n,t,p):
    qc=qiskit.QuantumCircuit(n)
    for q in range(n): qc.id(q)
    return qc
B=AerSimulator(method='density_matrix')
def model(tau):
    nm=NoiseModel()
    if tau>0:
        e=thermal_relaxation_error(T1,T2,tau)
        for q in range(N): nm.add_quantum_error(e,['id'],[q])
    for a,b in edges: nm.add_quantum_error(depolarizing_error(CZ,2),['cz'],[a,b])
    return nm
def probs(st,nm):
    qc=qiskit.QuantumCircuit(N)
    for s in st: qc &= s[0](s[1],s[2],s[3])
    qc.save_density_matrix()
    r=np.asarray(B.run(qc,noise_model=nm,shots=1).result().data(0)['density_matrix'])
    p=np.clip(np.real(np.diag(r)),0,None); return p/p.sum()
def C_of(tau):
    nm=model(tau)
    st=[prep,[sensing_xy,N,1.0,[x0,0.0]],[idle_qc,N,1.0,[]],post]
    p=probs(st,nm); m=p>1e-17; dp=[]
    for i in range(2):
        s1,s2=copy.deepcopy(st),copy.deepcopy(st)
        s1[1][3][i]+=step_size; s2[1][3][i]-=step_size
        dp.append(((probs(s1,nm)-probs(s2,nm))/(2*step_size))[m])
    p=p[m]
    return np.array([[np.sum(dp[i]*dp[j]/p) for j in range(2)] for i in range(2)])
taus=np.concatenate([np.linspace(2,20,10),np.linspace(25,300,23)])
out=[]
for t in taus:
    C=C_of(t); Ci=np.linalg.inv(C); out.append((t,C[0,0],C[1,1],C[0,1],Ci[0,0],np.trace(Ci)))
    print(t, C[0,0],C[1,1],Ci[0,0],flush=True)
np.save(D+'tau_scan.npy',np.array(out))
