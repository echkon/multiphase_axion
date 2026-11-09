# Fig. 3 redo: noiseless circulant L1=3, measurement = inverse of state preparation (mu=theta),
# DM phase redrawn at every coherence interval (n_c repetitions), phase-averaged likelihood.
import os as _os, sys as _sys
_HERE=_os.path.dirname(_os.path.abspath(__file__)); _ROOT=_os.path.join(_HERE,'..','..','..')
_sys.path[:0]=[_ROOT, _os.path.join(_ROOT,'axion_reach_results'), _os.path.join(_HERE,'..')]
import sys,pickle,json,time,numpy as np

import qiskit, qiskit.quantum_info as qi
from onecircuit import *
from axion_coupling import g_agg_from_delta
N=5; D=_os.path.join(_ROOT,'topo','data')+'/'
cc=pickle.load(open(D+'ckpt_cfi_best_full_L3_post_full_L3.pkl','rb'))
U=qi.Operator(cc[0][0](cc[0][1],cc[0][2],cc[0][3])).data; Ud=U.conj().T
psi0=U[:,0]
m_a=1.0; delta_true=6.498913663721062e-06; g_true=3.918992394086675e-16
t_cycle=120.0; tau_DM=1000.0/m_a; n_c=int(tau_DM//t_cycle)   # repetitions per coherence interval
def s1(x,y):
    w=np.hypot(x,y); sc=np.sinc(w/np.pi)
    return np.array([[np.cos(w),(-1j*x-y)*sc],[(-1j*x+y)*sc,np.cos(w)]])
def probs(x,y):
    s=s1(x,y); S=s
    for _ in range(N-1): S=np.kron(S,s)
    a=Ud@(S@psi0); return np.abs(a)**2
NA=32; al=np.linspace(0,2*np.pi,NA,endpoint=False)
def pbar(d): return np.mean([probs(d*np.cos(a),d*np.sin(a)) for a in al],0)
# delta grid: dense core + log wings, uniform prior in the (x,y) plane -> weight delta
core=np.linspace(0,4e-5,4001)[1:]; wing=np.logspace(np.log10(4.05e-5),0,150)
dg=np.concatenate([core,wing]); w_d=np.gradient(dg)
t0=time.time(); P=np.array([pbar(d) for d in dg]); logP=np.log(P+1e-300); print('grid',time.time()-t0,flush=True)
N_MONTH,N_6MO,N_YEAR=26300000000,157800000000,315600000000
n_log=np.unique(np.round(np.logspace(0,np.log10(N_YEAR),100)).astype(np.int64))
cps=np.unique(np.concatenate([n_log,[N_MONTH,N_6MO,N_YEAR]]))
rng=np.random.default_rng(2026)
KMAX=20000   # max distinct phases drawn per increment (each block = many coherence intervals)
def increment(dn):
    nint=int(np.ceil(dn/n_c)); K=min(nint,KMAX)
    sizes=np.full(K,dn//K); sizes[:dn-sizes.sum()]+=1
    c=np.zeros(2**N,dtype=np.int64)
    for a,sz in zip(rng.uniform(0,2*np.pi,K),sizes):
        c+=rng.multinomial(int(sz),probs(delta_true*np.cos(a),delta_true*np.sin(a)))
    return c
cum=np.zeros(2**N,dtype=np.int64); prev=0; out=dict(N_list=[],hat_delta=[],bias_delta=[],bias_g=[]); snaps=[]
for n in cps:
    if n>prev: cum=cum+increment(int(n-prev)); prev=int(n)
    ll=logP@cum; ll-=ll.max(); post=np.exp(ll)*dg; post/=np.sum(post*w_d)
    hd=float(np.sum(dg*post*w_d)); hg=float(g_agg_from_delta(hd,m_a))
    out['N_list'].append(int(n)); out['hat_delta'].append(hd); out['bias_delta'].append(hd-delta_true); out['bias_g'].append(hg-g_true)
    if n in (N_MONTH,N_6MO,N_YEAR):
        print(n,hd,hd-delta_true,hg-g_true,flush=True); snaps.append(dict(N_try=int(n),hat_delta=hd,hat_g=hg,post=post.tolist()))
res=dict(m_a=m_a,delta_true=delta_true,g_true=g_true,n_c=n_c,N_MONTH=N_MONTH,N_6MO=N_6MO,N_YEAR=N_YEAR,delta_grid=dg.tolist(),snapshots=snaps,**out,
 note='noiseless circulant L1=3; readout = inverse state preparation (mu=theta); phase redrawn every n_c repetitions (coherence interval); likelihood phase-averaged; uniform prior in (x,y)')
json.dump(res,open(D+'fig3_randomphase.json','w')); print('done')
