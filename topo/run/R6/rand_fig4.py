# Fig. 4 check: noiseless circulant, mu=theta, phase redrawn every coherence interval,
# posterior on (x,y) (fixed-phase likelihood, uniform prior), delta_hat = posterior mean of sqrt(x^2+y^2).
import os as _os, sys as _sys
_HERE=_os.path.dirname(_os.path.abspath(__file__)); _ROOT=_os.path.join(_HERE,'..','..','..')
_sys.path[:0]=[_ROOT, _os.path.join(_ROOT,'axion_reach_results'), _os.path.join(_HERE,'..')]
import json,time,numpy as np
exec(open(_os.path.join(_HERE,'rand_fig3.py')).read().split('m_a=1.0')[0])
from axion_coupling import delta_axion, g_agg_from_delta
NA=None
af=1/137.036
gK=lambda m:(af/(2*np.pi))*1.92*m/5.69e12; gD=lambda m:(af/(2*np.pi))*0.7467*m/5.69e12
def s1(x,y):
    w=np.hypot(x,y); sc=np.sinc(w/np.pi); return np.array([[np.cos(w),(-1j*x-y)*sc],[(-1j*x+y)*sc,np.cos(w)]])
def probs(x,y):
    s=s1(x,y); S=s
    for _ in range(N-1): S=np.kron(S,s)
    return np.abs(Ud@(S@psi0))**2
r=np.concatenate([np.linspace(0,4e-5,2001)[1:],np.logspace(np.log10(4.05e-5),0,150)]); NT=16; th=np.linspace(0,2*np.pi,NT,endpoint=False); wr=np.gradient(r)
LP=np.log(np.array([[probs(ri*np.cos(t),ri*np.sin(t)) for t in th] for ri in r])+1e-300)
rng=np.random.default_rng(7); KMAX=10000; t_cycle=120.0
masses=np.geomspace(0.1,10,10); out=[]
for model,gf in [('KSVZ',gK),('DFSZ',gD)]:
  for m in masses:
    g0=gf(m); d0,_=delta_axion(g0,m); n_c=max(1,int((1000.0/m)//t_cycle))
    for NT_ in [26300000000,315600000000]:
      gh=[]
      for run in range(5):
        K=min(int(np.ceil(NT_/n_c)),KMAX); sizes=np.full(K,NT_//K); sizes[:NT_-sizes.sum()]+=1
        c=np.zeros(2**N,dtype=np.int64)
        for a,sz in zip(rng.uniform(0,2*np.pi,K),sizes): c+=rng.multinomial(int(sz),probs(d0*np.cos(a),d0*np.sin(a)))
        ll=LP@c; ll-=ll.max(); g=np.exp(ll)*r[:,None]; mr=np.sum(g,1); mr/=np.sum(mr*wr)
        gh.append(float(g_agg_from_delta(np.sum(r*mr*wr),m)))
      out.append(dict(model=model,mass_ueV=float(m),N_try=NT_,n_c=n_c,g_target=g0,g_hat_runs=gh,g_hat_mean=float(np.mean(gh)),g_hat_std=float(np.std(gh))))
      print(model,'%.4f'%m,NT_,'%.4e %.4e %.4e'%(g0,np.mean(gh),np.std(gh)),flush=True)
json.dump(out,open(D+'fig4_randomphase.json','w'),indent=1); print('done')
