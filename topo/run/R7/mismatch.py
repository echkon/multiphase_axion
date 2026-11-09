import os as _os, sys as _sys
_HERE=_os.path.dirname(_os.path.abspath(__file__)); _ROOT=_os.path.join(_HERE,'..','..','..')
_sys.path[:0]=[_ROOT, _os.path.join(_ROOT,'axion_reach_results'), _os.path.join(_HERE,'..')]
import sys,copy,numpy as np
__file__=_os.path.join(_HERE,'..','tau_scan.py'); exec(open(__file__).read().split('taus=')[0])
from onecircuit.circuit import phase_gate
def mk(T1,T2,cz):
    nm=NoiseModel(); e=thermal_relaxation_error(T1,T2,100.0)
    for q in range(N): nm.add_quantum_error(e,['id'],[q])
    for a,b in edges: nm.add_quantum_error(depolarizing_error(cz,2),['cz'],[a,b])
    return nm
def sens(n,t,p):
    qc=qiskit.QuantumCircuit(n)
    for i in range(n): qc.append(phase_gate(t,[p[0],p[1],p[2]]),[i])
    return qc
def ro(p,e):
    if e==0: return p
    M=np.array([[1-e,e],[e,1-e]]); P=p.reshape([2]*N)
    for ax in range(N): P=np.moveaxis(np.tensordot(M,P,axes=([1],[ax])),0,ax)
    return P.reshape(-1)
def P(x,y,cfg):
    st=[cfg['prep'],[sens,N,1.0,[x,y,cfg['z']]],[idle_qc,N,1.0,[]],cfg['post']]
    return ro(probs(st,cfg['nm']),cfg['ro'])
h=1e-4
def lin(cfg):
    p0=P(0,0,cfg); J=np.stack([(P(h,0,cfg)-P(-h,0,cfg))/(2*h),(P(0,h,cfg)-P(0,-h,cfg))/(2*h)],1); return p0,J
base=dict(prep=prep,post=post,z=0.0,ro=0.0,nm=mk(100,100,0.005))
rng=np.random.default_rng(0)
def pert(st,s):
    s2=copy.deepcopy(st); s2[3]=np.array(s2[3])+rng.normal(0,s,len(s2[3])); return s2
cases={'T2 90':dict(base,nm=mk(100,90,0.005)),'T1 90':dict(base,nm=mk(90,100,0.005)),
 'cz 0.006':dict(base,nm=mk(100,100,0.006)),'ro 1%':dict(base,ro=0.01),
 'calib 1e-3':dict(base,prep=pert(prep,1e-3),post=pert(post,1e-3)),'drift z 1e-3':dict(base,z=1e-3)}
d0=6.498913663721062e-06
pt0,Jt=lin(base)
for k,cfg in cases.items():
    # data from cfg (true world), likelihood = base model
    pd0,Jd=lin(cfg); pm0,Jm=pt0,Jt
    W=np.diag(1/np.clip(pm0,1e-15,None)); A=np.linalg.inv(Jm.T@W@Jm)@Jm.T@W
    null=A@(pd0-pm0); sig=A@(pd0+Jd@[d0,0]-pm0)
    # with null calibration: model offset replaced by measured offset pd0
    W2=np.diag(1/np.clip(pd0,1e-15,None)); A2=np.linalg.inv(Jm.T@W2@Jm)@Jm.T@W2
    sigc=A2@(Jd@[d0,0])
    print('%-12s null |x*|=%.3e  signal x*=%.3e  calibrated x*/d0=%.4f'%(k,np.hypot(*null),np.hypot(*sig),np.hypot(*sigc)/d0),flush=True)
