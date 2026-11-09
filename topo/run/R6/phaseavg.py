import os as _os, sys as _sys
_HERE=_os.path.dirname(_os.path.abspath(__file__)); _ROOT=_os.path.join(_HERE,'..','..','..')
_sys.path[:0]=[_ROOT, _os.path.join(_ROOT,'axion_reach_results'), _os.path.join(_HERE,'..')]
import sys,pickle,numpy as np

from onecircuit import *
from onecircuit.metric import _safe_probs
N=5; D=_os.path.join(_ROOT,'topo','data')+'/'; cc=pickle.load(open(D+'ckpt_cfi_best_full_L3_post_full_L3.pkl','rb'))
def P(x,y):
    c=list(cc); c[1]=[sensing_xy,N,1.0,[x,y]]; p=np.clip(_safe_probs(c),0,None); return p/p.sum()
al=np.linspace(0,2*np.pi,32,endpoint=False)
def Pbar(d): return np.mean([P(d*np.cos(a),d*np.sin(a)) for a in al],0)
for d in [1e-5,1e-3,1e-2]:
    h=d*1e-2; pb=Pbar(d); dp=(Pbar(d+h)-Pbar(d-h))/(2*h); m=pb>1e-14
    print(d,'F_avg=',np.sum(dp[m]**2/pb[m]))
