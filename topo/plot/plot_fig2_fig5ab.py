import numpy as np, matplotlib, json, os
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figure'); os.makedirs(OUT, exist_ok=True)
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FixedLocator, FuncFormatter, NullFormatter
plt.rcParams.update({'mathtext.fontset':'cm','font.size':15,'axes.linewidth':2.5,
  'xtick.major.width':2,'ytick.major.width':2,'ytick.minor.width':1.5,'xtick.major.size':6,'ytick.major.size':6})
COL={'star':'#4F4181','ring':'#146786','full':'#61AC8A'}
LAB={'star':'star','ring':'ring','full':'circulant'}
L=[1,2,3,4]; BW=0.18
q0={'star':[0.05449365581940051,0.03397031448167772,0.03034356625566694,0.03021298550414292],
    'ring':[0.0681486994588608,0.03619257302618448,0.031184838039633466,0.030331348534761018],
    'full':[0.032528192055428,0.029961443670573754,0.02959604580395407,0.029653934800831797]}
c0=[0.03639599269784253,0.03496587688640713,0.0329259258280743,0.032096783071276384]
qn={'star':[0.6854371331881319,0.7586471388523073,0.6545168314842319,0.7762750857547573],
    'ring':[0.6065890543467377,0.7706639958881005,0.6953138124441176,0.7466338283029544],
    'full':[0.8601876069453732,0.7063220350780104,0.7434447054131821,0.7619522967759192]}
cn=[1.458042038906957,1.4166166724779075,1.8587115502761453,2.130974401335598]
def sci(v,pos):
    e=int(np.floor(np.log10(v))); m=v/10**e
    return r'$%g\times10^{%d}$'%(round(m,1),e) if abs(m-1)>1e-9 else r'$10^{%d}$'%e
def tag(ax,s,loc='left'):
    x=0.02 if loc=='left' else 0.98
    ax.text(x,0.97,s,transform=ax.transAxes,ha=loc,va='top',color='w',fontsize=17,
            bbox=dict(boxstyle='square,pad=0.15',fc='k',ec='k'))
def qpanel(ax,q,ylab,b0):
    x=np.arange(4)
    for i,t in enumerate(['star','ring','full']):
        ax.bar(x+(i-1)*BW,np.array(q[t])-b0,BW,bottom=b0,color=COL[t],label=LAB[t],zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(L); ax.set_xlabel(r'$L_1$',fontsize=20); ax.set_ylabel(ylab,fontsize=20)
def cpanel(ax,c,lab,col,ylab,b0):
    x=np.arange(4); ax.bar(x,np.array(c)-b0,BW,bottom=b0,color=col,label=lab,zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(L); ax.set_xlabel(r'$L_2$',fontsize=20)
    if ylab: ax.set_ylabel(ylab,fontsize=20)
def logaxis(ax,ticks,lim):
    ax.set_yscale('log'); ax.set_ylim(*lim)
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(FuncFormatter(sci))
    ax.yaxis.set_minor_formatter(NullFormatter()); ax.grid(True,axis='y',which='major',color='0.8',lw=1.5,zorder=0)
YQ=r'${\rm Tr}(\mathbf{Q}^{-1})$'; YC=r'${\rm Tr}(\mathbf{C}^{-1})$'
# ---- Fig. 2 ----
fig,ax=plt.subplots(1,2,figsize=(6.4,3.2),sharey=True)
qpanel(ax[0],q0,YQ,0.025); cpanel(ax[1],c0,'circulant\n($L_1$=3)',COL['full'],YC,0.025)
for a in ax: logaxis(a,[0.03,0.04,0.05,0.07],(0.025,0.085))
for a in ax: a.yaxis.set_minor_locator(FixedLocator([]))
ax[1].tick_params(labelleft=False); ax[1].set_ylabel(YC,fontsize=20); ax[1].yaxis.set_label_position('left')
ax[0].legend(loc='upper right',fontsize=13,framealpha=1); ax[1].legend(loc='upper right',fontsize=13,framealpha=1)
tag(ax[0],'(a)'); tag(ax[1],'(b)')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig2.eps')); plt.savefig(os.path.join(OUT,'fig2.png'),dpi=110); plt.close()
# ---- Fig. 5 (a),(b) ----
fig,ax=plt.subplots(2,1,figsize=(3.6,5.6))
qpanel(ax[0],qn,YQ,0.4); cpanel(ax[1],cn,'ring ($L_1$=1)',COL['ring'],YC,0.4)
for a in ax: logaxis(a,[0.5,1,2],(0.4,9))
for a in ax: a.yaxis.set_minor_locator(FixedLocator([]))
ax[0].legend(loc='upper right',fontsize=11,framealpha=1)
ax[1].legend(loc='upper right',fontsize=10,framealpha=1,handlelength=1.2)
tag(ax[0],'(a)'); tag(ax[1],'(b)')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig5ab.eps')); plt.savefig(os.path.join(OUT,'fig5ab.png'),dpi=110); plt.close()
