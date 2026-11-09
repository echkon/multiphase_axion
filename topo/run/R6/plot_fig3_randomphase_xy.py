import os as _os
_TOPO=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')
_os.makedirs(_os.path.join(_TOPO,'figure'),exist_ok=True)
import json,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
r=json.load(open(_os.path.join(_TOPO,'data','fig3_randomphase_xy.json')))
dg=np.array(r['delta_grid']); d0=r['delta_true']; N=np.array(r['N_list'])
fig=plt.figure(figsize=(10.5,5.5)); gs=fig.add_gridspec(2,2,width_ratios=[1.15,1],hspace=0.08,wspace=0.28)
a=fig.add_subplot(gs[:,0]); b=fig.add_subplot(gs[0,1]); c=fig.add_subplot(gs[1,1])
for s,col in zip(r['snapshots'],plt.cm.viridis(np.linspace(0.15,0.85,3))):
    a.plot(dg,s['post'],color=col,lw=2,label='$n=%s\\times10^{%d}$'%(('%.3e'%s['N_try']).split('e')[0],int(('%.3e'%s['N_try']).split('e')[1])))
a.axvline(d0,color='red',ls='--',lw=1.6,label=r'$\delta_{\rm true}$')
a.set_xlim(d0-3e-6,d0+3e-6); a.set_xlabel(r'$\delta=\sqrt{x^2+y^2}$'); a.set_xlabel(r'$\delta$'); a.set_ylabel('Posterior probability density'); a.legend(fontsize=9); a.grid(alpha=.3)
a.text(0.03,0.95,'(a)',transform=a.transAxes,fontsize=13,fontweight='bold',va='top')
for ax,y,lab,col in [(b,np.abs(r['bias_delta']),r'$|\hat\delta-\delta_{\rm true}|$','tab:blue'),(c,np.abs(r['bias_g']),r'$|\hat g_{a\gamma\gamma}-g_{\rm true}|$','tab:orange')]:
    ax.plot(N,y,color=col,lw=1.4)
    for nn in (r['N_MONTH'],r['N_6MO'],r['N_YEAR']): ax.axvline(nn,color='gray',ls=':',lw=.8,alpha=.6)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylabel(lab); ax.grid(alpha=.3,which='both')
b.tick_params(axis='x',labelbottom=False); c.set_xlabel(r'$n$')
b.text(0.02,0.05,'(b)',transform=b.transAxes,fontsize=13,fontweight='bold'); c.text(0.02,0.05,'(c)',transform=c.transAxes,fontsize=13,fontweight='bold')
fig.savefig(_os.path.join(_TOPO,'figure','fig3_randomphase_xy.eps'),bbox_inches='tight'); fig.savefig(_os.path.join(_TOPO,'figure','fig3_randomphase_xy.png'),dpi=90,bbox_inches='tight')
