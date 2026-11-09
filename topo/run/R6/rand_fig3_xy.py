# Same as rand_fig3.py, but the posterior is computed on the (x,y) plane (polar grid, uniform prior in (x,y)),
# with the fixed-phase likelihood p(k|x,y) as in the original Fig. 3; delta_hat = posterior mean of sqrt(x^2+y^2).
import os as _os, sys as _sys
_HERE=_os.path.dirname(_os.path.abspath(__file__)); _ROOT=_os.path.join(_HERE,'..','..','..')
_sys.path[:0]=[_ROOT, _os.path.join(_ROOT,'axion_reach_results'), _os.path.join(_HERE,'..')]
import json,time,numpy as np
exec(open(_os.path.join(_HERE,'rand_fig3.py')).read().split('# delta grid')[0])
r_old=np.array(json.load(open('/dev/stdin'))) if False else None
NR=800; r=np.concatenate([np.linspace(0,4e-5,651)[1:],np.logspace(np.log10(4.05e-5),0,150)]); NT=16; th=np.linspace(0,2*np.pi,NT,endpoint=False)
wr=np.gradient(r)
t0=time.time(); LP=np.log(np.array([[probs(ri*np.cos(t),ri*np.sin(t)) for t in th] for ri in r])+1e-300); print('grid',time.time()-t0,flush=True)
N_MONTH,N_6MO,N_YEAR=26300000000,157800000000,315600000000
n_log=np.unique(np.round(np.logspace(0,np.log10(N_YEAR),100)).astype(np.int64))
cps=np.unique(np.concatenate([n_log,[N_MONTH,N_6MO,N_YEAR]]))
rng=np.random.default_rng(2026); KMAX=20000
def increment(dn):
    nint=int(np.ceil(dn/n_c)); K=min(nint,KMAX); sizes=np.full(K,dn//K); sizes[:dn-sizes.sum()]+=1
    c=np.zeros(2**N,dtype=np.int64)
    for a,sz in zip(rng.uniform(0,2*np.pi,K),sizes): c+=rng.multinomial(int(sz),probs(delta_true*np.cos(a),delta_true*np.sin(a)))
    return c
cum=np.zeros(2**N,dtype=np.int64); prev=0; out=dict(N_list=[],hat_delta=[],bias_delta=[],bias_g=[]); snaps=[]
for n in cps:
    if n>prev: cum=cum+increment(int(n-prev)); prev=int(n)
    ll=LP@cum; ll-=ll.max(); g=np.exp(ll)*r[:,None]; g/=np.sum(g*wr[:,None])*(2*np.pi/NT)
    mr=np.sum(g,1)*(2*np.pi/NT)          # radial marginal
    hd=float(np.sum(r*mr*wr)); hg=float(g_agg_from_delta(hd,m_a))
    out['N_list'].append(int(n)); out['hat_delta'].append(hd); out['bias_delta'].append(hd-delta_true); out['bias_g'].append(hg-g_true)
    if n in (N_MONTH,N_6MO,N_YEAR): print(n,hd,hd-delta_true,hg-g_true,flush=True); snaps.append(dict(N_try=int(n),hat_delta=hd,hat_g=hg,post=mr.tolist()))
res=dict(m_a=m_a,delta_true=delta_true,g_true=g_true,n_c=n_c,N_MONTH=N_MONTH,N_6MO=N_6MO,N_YEAR=N_YEAR,delta_grid=r.tolist(),snapshots=snaps,**out,
 note='noiseless circulant L1=3; readout mu=theta (inverse state preparation); phase redrawn every n_c repetitions; posterior on (x,y) with fixed-phase likelihood and uniform prior; delta_hat = posterior mean of sqrt(x^2+y^2)')
json.dump(res,open(D+'fig3_randomphase_xy.json','w')); print('done')
