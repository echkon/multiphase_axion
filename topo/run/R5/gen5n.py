import os as _os, sys as _sys
_HERE=_os.path.dirname(_os.path.abspath(__file__)); _ROOT=_os.path.join(_HERE,'..','..','..')
_sys.path[:0]=[_ROOT, _os.path.join(_ROOT,'axion_reach_results'), _os.path.join(_HERE,'..')]
import sys,copy,numpy as np
__file__=_os.path.join(_HERE,'..','tau_scan.py'); exec(open(__file__).read().split('taus=')[0])
def C_at(d,a,tau=100.0):
    nm=model(tau)
    st=[prep,[sensing_xy,N,1.0,[d*np.cos(a),d*np.sin(a)]],[idle_qc,N,1.0,[]],post]
    p=probs(st,nm); m=p>1e-17; dp=[]
    for i in range(2):
        s1,s2=copy.deepcopy(st),copy.deepcopy(st)
        s1[1][3][i]+=step_size; s2[1][3][i]-=step_size
        dp.append(((probs(s1,nm)-probs(s2,nm))/(2*step_size))[m])
    p=p[m]; return np.array([[np.sum(dp[i]*dp[j]/p) for j in range(2)] for i in range(2)])
ds=np.geomspace(8.0e-7,2.1e-5,5); als=np.linspace(0,2*np.pi,8,endpoint=False)
Ci0=np.linalg.inv(C_at(x0,0.0)); print('fid',np.trace(Ci0),Ci0[0,0],flush=True)
R=[]
for d in ds:
  for a in als:
    Ci=np.linalg.inv(C_at(d,a)); u=np.array([np.cos(a),np.sin(a)]); R.append((d,a,np.trace(Ci),u@Ci@u))
R=np.array(R); np.save('gen5_noisy.npy',R)
print('TrCi',R[:,2].min(),R[:,2].max(),'marg',R[:,3].min(),R[:,3].max())
