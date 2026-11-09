import os as _os, sys as _sys
_HERE=_os.path.dirname(_os.path.abspath(__file__)); _ROOT=_os.path.join(_HERE,'..','..','..')
_sys.path[:0]=[_ROOT, _os.path.join(_ROOT,'axion_reach_results'), _os.path.join(_HERE,'..')]
import sys,pickle,copy,numpy as np
__file__=_os.path.join(_HERE,'..','tau_scan.py'); exec(open(__file__).read().split('taus=')[0])
from onecircuit.circuit import phase_gate
from onecircuit.metric import cfim
EPS=np.zeros(N)
def sens_nu(n,t,p):
    qc=qiskit.QuantumCircuit(n)
    for i in range(n): qc.append(phase_gate(t*(1+EPS[i]),[p[0],p[1],0.0]),[i])
    return qc
cc=pickle.load(open(D+'ckpt_cfi_best_full_L3_post_full_L3.pkl','rb'))
def C_clean():
    circ=list(cc); circ[1]=[sens_nu,N,1.0,[x0,0.0]]; return np.real(cfim(circ))
def C_noisy(tau=100.0):
    nm=model(tau); st=[prep,[sens_nu,N,1.0,[x0,0.0]],[idle_qc,N,1.0,[]],post]
    p=probs(st,nm); m=p>1e-17; dp=[]
    for i in range(2):
        s1,s2=copy.deepcopy(st),copy.deepcopy(st); s1[1][3][i]+=step_size; s2[1][3][i]-=step_size
        dp.append(((probs(s1,nm)-probs(s2,nm))/(2*step_size))[m])
    p=p[m]; return np.array([[np.sum(dp[i]*dp[j]/p) for j in range(2)] for i in range(2)])
rng=np.random.default_rng(1)
b0=np.trace(np.linalg.inv(C_clean())); b1=np.trace(np.linalg.inv(C_noisy())); print('uniform',b0,b1,flush=True)
for sig in [0.05,0.1,0.2]:
    r0=[];r1=[]
    for k in range(10):
        EPS[:]=rng.normal(0,sig,N)
        r0.append(np.trace(np.linalg.inv(C_clean()))/b0); r1.append(np.trace(np.linalg.inv(C_noisy()))/b1)
    EPS[:]=0
    print(sig,'clean mean %.4f max %.4f'%(np.mean(r0),np.max(r0)),'noisy mean %.4f max %.4f'%(np.mean(r1),np.max(r1)),flush=True)
