import os as _os
_TOPO=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')
_os.makedirs(_os.path.join(_TOPO,'figure'),exist_ok=True)
import json,glob,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'mathtext.fontset':'cm','font.size':14,'axes.linewidth':2})
new=json.load(open(_os.path.join(_TOPO,'data','fig4_randomphase.json')))
af=1/137.036; m=np.geomspace(0.08,12,200)
gK=(af/(2*np.pi))*1.92*m/5.69e12; gD=(af/(2*np.pi))*0.7467*m/5.69e12
old={}
for mod in ('KSVZ','DFSZ'):
    for fn in glob.glob(_os.path.join(_TOPO,'data','targeted_mass_bayes_results_full_L3_post_full_L3_%s_m*_N*.json'%mod)):
        d=json.load(open(fn)); old[(mod,round(d['mass_ueV'],3),float('%.3e'%d['N_TRY']))]=d
fig,ax=plt.subplots(figsize=(7,5.2))
ax.plot(m,gK,'k:',lw=1.8,label='KSVZ'); ax.plot(m,gD,'k-.',lw=1.8,label='DFSZ')
mk={26300000000:('o',r'$N_{\rm try}=2.630\times10^{10}$'),315600000000:('D',r'$N_{\rm try}=3.156\times10^{11}$')}
sh={26300000000:0.96,315600000000:1.04}
for NT,(sym,lab) in mk.items():
    for mod,col in [('KSVZ','tab:green'),('DFSZ','tab:purple')]:
        pts=[p for p in new if p['N_try']==NT and p['model']==mod]
        x=np.array([p['mass_ueV'] for p in pts])*sh[NT]
        ax.errorbar(x,[p['g_hat_mean'] for p in pts],[p['g_hat_std'] for p in pts],fmt=sym,color=col,ms=7,capsize=3,label=f'{mod}, random phase, '+lab)
        o=[old.get((mod,round(p['mass_ueV'],3),float('%.3e'%NT))) for p in pts]
        xo=[xi for xi,oo in zip(x,o) if oo]; 
        ax.plot(np.array(xo)*1.02,[oo['g_hat_mean'] for oo in o if oo],sym,mfc='none',mec='k',ms=9,mew=1.2,label=(f'fixed phase (old), '+lab) if mod=='KSVZ' else None)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.08,12)
ax.set_xlabel(r'$m_a$ [$\mu$eV]'); ax.set_ylabel(r'$g_{a\gamma\gamma}$ [GeV$^{-1}$]')
ax.legend(fontsize=8,loc='upper left'); ax.grid(alpha=.3,which='both')
fig.tight_layout(); fig.savefig(_os.path.join(_TOPO,'figure','fig4_randomphase.eps')); fig.savefig(_os.path.join(_TOPO,'figure','fig4_randomphase_preview.png'),dpi=90)
print('ok')
