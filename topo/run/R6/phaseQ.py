import os as _os, sys as _sys
_HERE=_os.path.dirname(_os.path.abspath(__file__)); _ROOT=_os.path.join(_HERE,'..','..','..')
_sys.path[:0]=[_ROOT, _os.path.join(_ROOT,'axion_reach_results'), _os.path.join(_HERE,'..')]
import sys,pickle,numpy as np

import qiskit
from qiskit.quantum_info import Statevector
from onecircuit import *
N=5; D=_os.path.join(_ROOT,'topo','data')+'/'; cc=pickle.load(open(D+'ckpt_cfi_best_full_L3_post_full_L3.pkl','rb'))
def sv(st):
    qc=qiskit.QuantumCircuit(N)
    for s in st: qc &= s[0](s[1],s[2],s[3])
    return Statevector(qc).data
al=np.linspace(0,2*np.pi,16,endpoint=False)
def rhobar(d):
    r=0
    for a in al:
        v=sv([cc[0],[sensing_xy,N,1.0,[d*np.cos(a),d*np.sin(a)]]]); r=r+np.outer(v,v.conj())
    return r/len(al)
def qfi(r,dr):
    w,V=np.linalg.eigh(r); drV=V.conj().T@dr@V; F=0
    for i in range(len(w)):
        for j in range(len(w)):
            s=w[i]+w[j]
            if s>1e-14*max(w): F+=2*abs(drV[i,j])**2/s
    return F
for d in [1e-2,1e-3,1e-4]:
    h=d*1e-3; F=qfi(rhobar(d),(rhobar(d+h)-rhobar(d-h))/(2*h))
    # inverse-prep measurement: probability of returning to initial |0..0>
    print(d,'QFI_avg=%.3f'%F)
import qiskit.quantum_info as qi
qc=qiskit.QuantumCircuit(N); qc &= cc[0][0](cc[0][1],cc[0][2],cc[0][3]); Uinv=qi.Operator(qc).data.conj().T
for d in [1e-3,1e-5]:
    h=d*1e-3
    P=lambda dd: np.real(np.diag(Uinv@rhobar(dd)@Uinv.conj().T))
    p=P(d); dp=(P(d+h)-P(d-h))/(2*h); m=p>1e-30
    print(d,'CFI inverse-prep=%.3f'%np.sum(dp[m]**2/p[m]), 'p_background(00000)=%.3e'%(1-p[0]))
