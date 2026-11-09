import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
plt.rcParams.update({'mathtext.fontset':'cm','font.size':16,'axes.labelsize':20,'axes.linewidth':1.5,
                     'xtick.major.width':1.5,'ytick.major.width':1.5,'xtick.major.size':5,'ytick.major.size':5})
import os
HERE=os.path.dirname(os.path.abspath(__file__))
a=np.load(os.path.join(HERE,'..','data','tau_scan.npy')); t=a[:,0]; Ci=a[:,4]
f=interp1d(t,np.log(Ci),kind='cubic'); T2=100.; TOH=20.
ref=np.sqrt(np.exp(f(T2))*(T2+TOH))/T2
def S(tau,toh): return np.sqrt(np.exp(f(tau))*(tau+toh))/tau/ref
LW=2.5; MS=11; YL=(0.8,3.0)
fig,ax=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
D=np.linspace(0.08,1,400); toh=T2*(1/D-1)
ax[0].plot(D,[S(np.array([T2]),x)[0] for x in toh],'k',lw=LW)
ax[0].plot(T2/(T2+TOH),1,'s',color='tab:blue',ms=MS,zorder=5)
ax[0].set_xlabel(r'$\tau/t_{\rm cycle}$'); ax[0].set_xlim(0,1.02)
tt=np.linspace(5,300,3000); s=S(tt,TOH); i=s.argmin()
ax[1].plot(tt/T2,s,'k',lw=LW)
ax[1].plot(tt[i]/T2,s[i],'o',color='tab:red',ms=MS,zorder=5)
ax[1].plot(1,1,'s',color='tab:blue',ms=MS,zorder=5)
ax[1].set_xlabel(r'$\tau/T_2$'); ax[1].set_xlim(0,3)
ax[0].set_ylabel(r'$\Delta g_{a\gamma\gamma}/\Delta g_{a\gamma\gamma}^{(0)}$')
for k,(x,l) in enumerate(zip(ax,['(a)','(b)'])):
    x.set_ylim(*YL); x.grid(True,alpha=0.4); x.text(0.05 if k else 0.85,0.88,l,transform=x.transAxes,fontsize=17)
plt.tight_layout(); plt.savefig(os.path.join(HERE,'..','figure','fig6.eps'))
