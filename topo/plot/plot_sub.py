# Figures S2-S5 of the Supplemental Material (EPS) -> topo/figure/
import json,glob,os,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
def sci(v,pos):
    e=int(np.floor(np.log10(v))); m=v/10**e
    return r'$%g\times10^{%d}$'%(round(m,1),e)
def ticks(ax,t):
    ax.yaxis.set_major_locator(FixedLocator(t)); ax.yaxis.set_major_formatter(FuncFormatter(sci))
    ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_formatter(NullFormatter())
plt.rcParams.update({'mathtext.fontset':'cm','font.size':13,'axes.linewidth':1.8,
  'xtick.major.width':1.5,'ytick.major.width':1.5,'xtick.direction':'in','ytick.direction':'in'})
import os; _TOPO=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
D=os.path.join(_TOPO,'data')+'/'; OUT=os.path.join(_TOPO,'figure')+'/'; os.makedirs(OUT, exist_ok=True)
J=lambda f: json.load(open(D+f))
cols=['#1b9e77','#d95f02','#7570b3','#e7298a']
def tag(ax,s,loc='ul'):
    x,y,va,ha={'ul':(0.02,0.97,'top','left'),'ll':(0.02,0.03,'bottom','left'),'ur':(0.98,0.97,'top','right')}[loc]
    ax.text(x,y,s,transform=ax.transAxes,fontsize=15,va=va,ha=ha,color='w',zorder=10,
            bbox=dict(boxstyle='square,pad=0.15',fc='k',ec='k'))
names={'star':'star','ring':'ring','full':'circulant'}
# ---- S2: QFI training ----
fig,ax=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey='row')
nz='_T1100_T2100_czerr0p005'
for j,t in enumerate(['star','ring','full']):
    for L in range(1,5):
        c=J(f'costs_qfi_{t}_L{L}.json'); ax[0,j].plot(np.arange(1,len(c)+1),c,color=cols[L-1],lw=1.8,label=f'$L_1={L}$')
        c=J(f'costs_qfi_network_noise_{t}_L{L}{nz}.json'); ax[1,j].plot(np.arange(1,len(c)+1),c,color=cols[L-1],lw=1.8,label=f'$L_1={L}$')
    ax[0,j].set_title(names[t]); ax[0,j].set_yscale('log'); ax[1,j].set_yscale('log')
    ax[1,j].set_xlabel('iteration')
    tag(ax[0,j],'(%s)'%'abc'[j],'ur'); tag(ax[1,j],'(%s)'%'def'[j],'ur')
for (r,t,L,fn) in [(0,'full',3,'costs_qfi_full_L3.json'),(1,'ring',1,f'costs_qfi_network_noise_ring_L1{nz}.json')]:
    c=np.array(J(fn)); k=int(np.argmin(c)); jj=['star','ring','full'].index(t)
    ax[r,jj].plot(k+1,c[k],'*',color=cols[L-1],ms=18,mec='k',mew=1.0,zorder=5)
ticks(ax[1,0],[0.6,0.8,1.0,1.4]); ticks(ax[0,0],[0.03,0.04,0.06,0.1])
ax[0,0].set_ylabel(r'${\rm Tr}(\mathbf{Q}^{-1})$, noiseless',fontsize=17); ax[1,0].set_ylabel(r'${\rm Tr}(\mathbf{Q}^{-1})$, noisy',fontsize=17)
ax[0,2].legend(fontsize=15,frameon=False,handlelength=1.6,loc='center right')
fig.tight_layout(); fig.savefig(OUT+'figS2.eps'); plt.close()
# ---- S3: CFI training ----
fig,ax=plt.subplots(1,2,figsize=(10,4))
for L in range(1,5):
    c=J(f'costs_cfi_full_L3_post_full_L{L}.json')[:400]; ax[0].plot(np.arange(1,len(c)+1),np.minimum.accumulate(c),color=cols[L-1],lw=2,label=f'$L_2={L}$')
    c=J(f'costs_cfi_network_noise_ring_L1_post_ring_L{L}{nz}.json'); ax[1].plot(np.arange(1,len(c)+1),c,color=cols[L-1],lw=1.8,label=f'$L_2={L}$')
ax[0].axhline(J('costs_qfi_full_L3.json')[np.argmin(J('costs_qfi_full_L3.json'))],color='k',ls='--',lw=1.4,label=r'${\rm Tr}(\mathbf{Q}^{-1})$')
ax[1].axhline(min(J(f'costs_qfi_network_noise_ring_L1{nz}.json')),color='k',ls='--',lw=1.4,label=r'${\rm Tr}(\mathbf{Q}^{-1})$')
for a,s,t in zip(ax,'ab',['circulant, $L_1=3$, noiseless','ring, $L_1=1$, noisy']):
    a.set_yscale('log'); a.set_xlabel('iteration',fontsize=15); a.set_title(t); tag(a,f'({s})','ll')
ax[0].set_ylim(0.026,0.058); ax[1].set_ylim(0.4,7); ticks(ax[0],[0.03,0.04,0.05]); ticks(ax[1],[0.5,1,2,4])
ax[0].set_ylabel(r'${\rm Tr}(\mathbf{C}^{-1})$',fontsize=17)
for k,(L,fn) in enumerate([(3,'costs_cfi_full_L3_post_full_L3.json'),(2,f'costs_cfi_network_noise_ring_L1_post_ring_L2{nz}.json')]):
    c=np.array(J(fn)[:400]); i=int(np.argmin(c)); ax[k].plot(i+1,c[i],'*',color=cols[L-1],ms=18,mec='k',mew=1.0,zorder=5)
ax[0].legend(fontsize=14,frameon=False,loc='upper right',handlelength=1.6)
fig.tight_layout(); fig.savefig(OUT+'figS3.eps'); plt.close()
# ---- S4: robustness over delta and alpha ----
A=np.load('run/R5/gen5_noiseless.npy'); B=np.load('run/R5/gen5_noisy.npy')
fig,ax=plt.subplots(1,2,figsize=(10,4))
for R,lab,c,m in [(A,'circulant, noiseless','tab:blue','o'),(B,'ring, noisy','tab:red','s')]:
    ref=np.median(R[:,2]); ax[0].plot(R[:,0],(R[:,2]/ref-1),m,color=c,ms=7,mfc='none',mew=1.4,label=lab)
    for d in np.unique(R[:,0])[[0,-1]]:
        sel=R[:,0]==d; o=np.argsort(R[sel,1]); first=d==np.unique(R[:,0])[0]; ax[1].plot(R[sel,1][o],R[sel,3][o]/np.mean(R[:,3]),m+('-' if first else '--'),color=c,ms=7,lw=1.4,mfc='none',mew=1.4,label=(lab if first else None))
ax[0].set_xscale('log'); ax[0].set_xlabel(r'$\delta$',fontsize=20); ax[0].set_ylabel(r'$\Delta{\rm Tr}(\mathbf{C}^{-1})/{\rm Tr}(\mathbf{C}^{-1})$',fontsize=17)
ax[0].yaxis.set_major_locator(FixedLocator([-2e-5,-1e-5,0,1e-5,2e-5])); ax[0].yaxis.set_major_formatter(FuncFormatter(lambda v,p: '0' if abs(v)<1e-12 else r'$%d\times10^{-5}$'%round(v*1e5)))
ax[0].legend(fontsize=14,frameon=False); tag(ax[0],'(a)')
lo,hi=ax[1].get_ylim(); ax[1].set_ylim(lo,hi+0.9*(hi-lo)); ax[1].legend(fontsize=14,frameon=False,loc='upper right')
ax[1].set_xlabel(r'$\alpha$',fontsize=20); ax[1].set_ylabel(r'$\mathbf{u}^T\mathbf{C}^{-1}\mathbf{u}$ (normalized)',fontsize=17); tag(ax[1],'(b)')
ax[1].set_xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]); ax[1].set_xticklabels(['0',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
fig.tight_layout(); fig.savefig(OUT+'figS4.eps'); plt.close()
# ---- S5: random phase between coherence intervals ----
fx=J('bayes_recover_results_full_L3_post_full_L3_singletraj_fixedrandomalpha.json'); rp=J('fig3_randomphase_xy.json')
fig,ax=plt.subplots(1,2,figsize=(11,4.3))
ax[0].plot(fx['N_list'],np.abs(fx['bias_g_list']),color='k',lw=2.2,label='fixed phase')
ax[0].plot(rp['N_list'],np.abs(rp['bias_g']),color='tab:red',lw=2.2,label='random phase')
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel(r'$n$',fontsize=18); ax[0].set_ylabel(r'$|\hat g_{a\gamma\gamma}-g_{\rm true}|$ [GeV$^{-1}$]',fontsize=17)
ax[0].legend(fontsize=14,frameon=False); tag(ax[0],'(a)','ll')
new=J('fig4_randomphase.json'); af=1/137.036; m=np.geomspace(0.08,12,200)
ax[1].plot(m,(af/(2*np.pi))*1.92*m/5.69e12,'k:',lw=2.2,label='KSVZ'); ax[1].plot(m,(af/(2*np.pi))*0.7467*m/5.69e12,'k-.',lw=2.2,label='DFSZ')
# the data folder also holds untagged files from an earlier run whose
# values are the KSVZ ones; glob each model explicitly so that they
# cannot be read as DFSZ
old={}
for _mod in ('KSVZ','DFSZ'):
    for fn in glob.glob(D+'targeted_mass_bayes_results_full_L3_post_full_L3_%s_m*_N3p156e11.json'%_mod):
        d=json.load(open(fn)); old[(_mod,round(d['mass_ueV'],3))]=d
for mod,c in [('KSVZ','tab:green'),('DFSZ','tab:purple')]:
    p=[q for q in new if q['model']==mod and q['N_try']==315600000000]
    x=np.array([q['mass_ueV'] for q in p])
    ax[1].errorbar(x*1.03,[q['g_hat_mean'] for q in p],[q['g_hat_std'] for q in p],fmt='D',color=c,ms=8,capsize=3,elinewidth=1.6,label=f'{mod}, random phase')
    o=[old.get((mod,round(q['mass_ueV'],3))) for q in p]
    ax[1].errorbar(x[[k for k,v in enumerate(o) if v]]/1.03,[v['g_hat_mean'] for v in o if v],[v['g_hat_std'] for v in o if v],fmt='o',mfc='none',mew=1.6,color=c,ms=8,capsize=3,elinewidth=1.6,label=f'{mod}, fixed phase')
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel(r'$m_a$ [$\mu$eV]',fontsize=17); ax[1].set_ylabel(r'$\hat g_{a\gamma\gamma}$ [GeV$^{-1}$]',fontsize=17)
ax[1].set_ylim(1e-17,6e-14)
h,l=ax[1].get_legend_handles_labels()
lg1=ax[1].legend(h[:2],l[:2],fontsize=14,frameon=False,loc='lower right',handlelength=2.2); ax[1].add_artist(lg1)
ax[1].legend(h[2:],l[2:],fontsize=12,frameon=False,loc='upper left',bbox_to_anchor=(0.0,0.88))
tag(ax[1],'(b)')
fig.tight_layout(); fig.savefig(OUT+'figS5.eps'); plt.close()
print('ok')
