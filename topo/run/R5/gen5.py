import os as _os, sys as _sys
_HERE=_os.path.dirname(_os.path.abspath(__file__)); _ROOT=_os.path.join(_HERE,'..','..','..')
_sys.path[:0]=[_ROOT, _os.path.join(_ROOT,'axion_reach_results'), _os.path.join(_HERE,'..')]
import sys,pickle,numpy as np

from onecircuit import *
from onecircuit.metric import cfim
N=5; x0=6.498913663721062e-06
D=_os.path.join(_ROOT,'topo','data')+'/'
cc=pickle.load(open(D+'ckpt_cfi_best_full_L3_post_full_L3.pkl','rb'))
def C_at(d,a):
    circ=list(cc); circ[1]=[sensing_xy,N,1.0,[d*np.cos(a),d*np.sin(a)]]
    return np.real(cfim(circ))
ds=np.geomspace(8.0e-7,2.1e-5,7); als=np.linspace(0,2*np.pi,12,endpoint=False)
C0=C_at(x0,0.0); Ci0=np.linalg.inv(C0); print('fid TrCi',np.trace(Ci0),'xx',Ci0[0,0])
R=[]
for d in ds:
  for a in als:
    C=C_at(d,a); Ci=np.linalg.inv(C); u=np.array([np.cos(a),np.sin(a)])
    R.append((d,a,np.trace(Ci),u@Ci@u))
R=np.array(R); np.save('gen5_noiseless.npy',R)
print('TrCi range',R[:,2].min(),R[:,2].max(),' rel',R[:,2].min()/np.trace(Ci0),R[:,2].max()/np.trace(Ci0))
print('delta-marginal range',R[:,3].min(),R[:,3].max())
