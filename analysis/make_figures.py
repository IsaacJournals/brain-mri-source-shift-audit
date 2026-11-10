import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "Inter" in f: fm.fontManager.addfont(f)
plt.rcParams.update({"font.family":"Inter","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.titleweight":"semibold","axes.titlesize":11,"savefig.dpi":300,"axes.edgecolor":"#5B6770","axes.labelcolor":"#1F2A33","xtick.color":"#1F2A33","ytick.color":"#1F2A33"})
C={"blue":"#2E6FAC","teal":"#1B9E8A","orange":"#E08A2E","purple":"#7A5BA6","red":"#C8553D","grey":"#5B6770","lgrey":"#C9D1D8","ink":"#1F2A33"}
ARCH={"resnet18":"ResNet-18","efficientnet_b0":"EfficientNet-B0","convnext_tiny":"ConvNeXt-T"}
ACOL={"resnet18":C["blue"],"efficientnet_b0":C["teal"],"convnext_tiny":C["orange"]}
import os
F = os.environ.get("P1_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")) + "/"
def save(fig,name):
    fig.savefig(name+".png",bbox_inches="tight"); fig.savefig(name+".pdf",bbox_inches="tight"); plt.close(fig)
def letter(ax,s): ax.text(-0.12,1.06,s,transform=ax.transAxes,fontsize=13,fontweight="bold",va="bottom",color=C["ink"])

# ---------- Fig 2 containment ----------
src=["Figshare","SARTAJ","Br35H","Nickparvar","BRISC"]; key=["figshare","sartaj","br35h","nickparvar","brisc"]
M={"br35h":{"brisc":1.67,"figshare":0.07,"nickparvar":23.99,"sartaj":12.19},
   "brisc":{"br35h":5.97,"figshare":93.77,"nickparvar":68.85,"sartaj":80.85},
   "figshare":{"br35h":0.07,"brisc":75.53,"nickparvar":67.10,"sartaj":71.84},
   "nickparvar":{"br35h":58.37,"brisc":80.48,"figshare":99.15,"sartaj":99.17},
   "sartaj":{"br35h":42.13,"brisc":70.20,"figshare":74.84,"nickparvar":79.86}}
A=np.full((5,5),np.nan)
for i,a in enumerate(key):
    for j,b in enumerate(key):
        if a!=b: A[i,j]=M[a][b]
pd.DataFrame(A,index=src,columns=src).to_csv("Fig2_containment_source.csv")
fig,axs=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw={"width_ratios":[1.25,1]})
ax=axs[0]; cm=matplotlib.colors.LinearSegmentedColormap.from_list("b",["#FFFFFF","#BBD3EA",C["blue"],"#173A5E"])
im=ax.imshow(A,cmap=cm,vmin=0,vmax=100)
for i in range(5):
    for j in range(5):
        if i==j: ax.text(j,i,"–",ha="center",va="center",color=C["grey"])
        else: ax.text(j,i,f"{A[i,j]:.0f}",ha="center",va="center",fontsize=10,color="white" if A[i,j]>55 else C["ink"],fontweight="semibold")
ax.set_xticks(range(5),src,rotation=30,ha="right"); ax.set_yticks(range(5),src)
ax.set_xlabel("Dataset whose images are looked up"); ax.set_ylabel("Dataset that contains them")
for s_ in ax.spines.values(): s_.set_visible(False)
cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.03); cb.set_label("Images contained (%)")
letter(ax,"a")
ax=axs[1]
ext=pd.read_csv(F+"addonC/external_containment.csv")
names={"bdneuro":"BDNeuro-MRI","pmram":"PMRAM"}
y=np.arange(len(ext))
ax.barh(y,ext.contained_pct,color=C["blue"],height=0.5,label="Near-duplicate of a Kaggle-family image")
ax.barh(y,100-ext.contained_pct,left=ext.contained_pct,color=C["lgrey"],height=0.5,label="New image (used as external test)")
for i,r in ext.iterrows():
    ax.text(r.contained_pct/2,i,f"{r.contained_pct:.0f}%",ha="center",va="center",color="white",fontweight="semibold")
    ax.text(r.contained_pct+(100-r.contained_pct)/2,i,f"{100-r.contained_pct:.0f}%",ha="center",va="center",color=C["ink"])
    ax.text(101,i,f"n = {int(r.n):,}",va="center",fontsize=9,color=C["grey"])
ax.set_yticks(y,[names[d] for d in ext.dataset]); ax.set_xlim(0,115); ax.set_xlabel("Share of images (%)")
ax.legend(loc="upper center",bbox_to_anchor=(0.45,-0.22),frameon=False,ncol=1,fontsize=9)
letter(ax,"b")
fig.tight_layout(); save(fig,"Fig2_containment")

# ---------- Fig 3 replication ----------
r=pd.read_csv(F+"replication_per_seed.csv")
order=[("OFFICIAL","official_test","Official test"),("OFFICIAL","official_test_clean","Clean test\n(no duplicates)"),("CLUSTER","cluster_test","Cluster-disjoint\nsplit"),("CROSS","cross_figshare","New source\n(Figshare)")]
fig,ax=plt.subplots(figsize=(7.5,4.2))
for arch,col,lab in [("mlhnet_style",C["purple"],"Kaprekar-style (re-implemented)"),("khan_cnn",C["red"],"Khan-style CNN (re-implemented)")]:
    m=[];sd=[]
    for p,t,_ in order:
        v=r[(r.arch==arch)&(r.protocol==p)&(r.test==t)].ba; m.append(v.mean()); sd.append(v.std(ddof=1))
    x=np.arange(len(order)); ax.errorbar(x,m,yerr=sd,color=col,marker="o",ms=8,lw=2.2,capsize=4,label=lab)
    dy=0.03 if arch=="mlhnet_style" else -0.045
    for xi,mi in zip(x,m): ax.text(xi+0.06,mi+dy,f"{mi:.2f}",color=col,fontsize=9,fontweight="semibold")
ax.axhline(0.984,ls="--",color=C["grey"],lw=1); ax.text(3.45,0.984,"Median reported\nin literature (98.4%)",va="center",fontsize=8.5,color=C["grey"])
ax.set_xticks(range(len(order)),[o[2] for o in order]); ax.set_ylabel("Balanced accuracy"); ax.set_ylim(0.25,1.05); ax.set_xlim(-0.3,3.4)
ax.legend(frameon=False,loc="lower left",fontsize=9)
save(fig,"Fig3_replication")

# ---------- Fig 4 cross-source ----------
d=pd.read_csv(F+"per_seed_metrics_FULL.csv")
dirs=[("bin_br35h_raw","cross_sartaj","Br35H → SARTAJ\n(tumour vs none)"),("bin_sartaj_raw","cross_br35h","SARTAJ → Br35H\n(tumour vs none)"),
      ("type_figshare_raw","cross_sartaj","Figshare → SARTAJ\n(tumour type)"),("type_sartaj_raw","cross_figshare","SARTAJ → Figshare\n(tumour type)")]
fig,axs=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={"width_ratios":[1.5,1]})
ax=axs[0]; w=0.12
for k,(p,ct,lab) in enumerate(dirs):
    for a_i,a in enumerate(ARCH):
        g=d[(d.protocol==p)&(d.arch==a)]
        ins=g[g.test=="in_source"].ba; cr=g[g.test==ct].ba
        xc=k+(a_i-1)*0.26
        ax.plot([xc-w/2,xc+w/2],[ins.mean(),cr.mean()],color=ACOL[a],lw=1.6)
        ax.scatter(xc-w/2,ins.mean(),color=ACOL[a],s=36,zorder=3); ax.scatter(xc+w/2,cr.mean(),facecolor="white",edgecolor=ACOL[a],s=36,lw=1.8,zorder=3)
ax.set_xticks(range(4),[x[2] for x in dirs],fontsize=9); ax.set_ylabel("Balanced accuracy"); ax.set_ylim(0.55,1.0)
from matplotlib.lines import Line2D
h=[Line2D([],[],color=ACOL[a],lw=2,label=ARCH[a]) for a in ARCH]+[Line2D([],[],marker="o",color=C["grey"],ls="",label="Same source (filled)"),Line2D([],[],marker="o",mfc="white",color=C["grey"],ls="",label="New source (open)")]
ax.legend(handles=h,frameon=False,fontsize=8.5,ncol=2,loc="lower left"); letter(ax,"a")
ax=axs[1]
e=pd.read_csv(F+"addonC/external_eval_full_models.csv")
rows=[("type_figshare_raw","Figshare type, raw"),("type_figshare_harm","Figshare type, harmonised"),("type_sartaj_raw","SARTAJ type, raw"),("four_nickparvar_raw_CLUSTER","Nickparvar 4-class, cluster split"),("four_nickparvar_raw_OFFICIAL","Nickparvar 4-class, official split")]
yy=np.arange(len(rows))[::-1]
for (p,lab),yv in zip(rows,yy):
    for ds,col,off in [("external_bdneuro",C["blue"],0.15),("external_pmram",C["purple"],-0.15)]:
        v=e[(e.protocol==p)&(e.test==ds)].ba
        if len(v): ax.errorbar(v.mean(),yv+off,xerr=v.std(ddof=1),fmt="o",color=col,capsize=3,ms=6)
ax.set_yticks(yy,[x[1] for x in rows],fontsize=9); ax.set_xlabel("Balanced accuracy on new external images"); ax.set_xlim(0.3,1.0)
ax.legend(handles=[Line2D([],[],marker="o",color=C["blue"],ls="",label="BDNeuro-MRI (n = 832)"),Line2D([],[],marker="o",color=C["purple"],ls="",label="PMRAM (n = 522)")],frameon=False,fontsize=8.5,loc="lower left")
letter(ax,"b"); fig.tight_layout(); save(fig,"Fig4_cross_source")

# ---------- Fig 5 CCIT ----------
s=pd.read_csv(F+"addon/ccit_summary.csv")
tl={"T1":"T1 · training-source background on new-source images","T2":"T2 · new-source background on same-source images","T3":"T3 · re-encode with other source's size and JPEG tables","T4_in":"T4 · blank outside the brain (same source)","T4_cross":"T4 · blank outside the brain (new source)"}
pl={"bin_br35h_raw":"Br35H","bin_sartaj_raw":"SARTAJ","type_figshare_raw":"Figshare type","type_sartaj_raw":"SARTAJ type"}
fig,ax=plt.subplots(figsize=(8.5,9))
ylab=[];y=0;ticks=[]
for t in ["T1","T2","T3","T4_in","T4_cross"]:
    ax.text(-0.115,y+0.4,tl[t],fontsize=9.5,fontweight="semibold",color=C["ink"],transform=ax.get_yaxis_transform())
    y+=1
    for p in pl:
        for a in ARCH:
            r_=s[(s.test==t)&(s.protocol==p)&(s.arch==a)].iloc[0]
            ax.plot([r_.ci_lo,r_.ci_hi],[y,y],color=ACOL[a],lw=2); ax.scatter(r_.mean_delta_ba,y,color=ACOL[a],s=22,zorder=3)
            y+=0.55
        ticks.append((y-1.1,pl[p])); y+=0.35
    y+=0.6
ax.axvline(0,color=C["ink"],lw=1); 
for v in (-0.05,0.05): ax.axvline(v,color=C["red"],ls="--",lw=1)
ax.text(0.052,0.2,"±0.05 pre-declared\nmeaningful effect",color=C["red"],fontsize=8.5,va="top")
ax.set_yticks([t[0] for t in ticks],[t[1] for t in ticks],fontsize=8.5); ax.invert_yaxis(); ax.set_xlim(-0.1,0.1)
ax.set_xlabel("Change in balanced accuracy after the edit (mean, 95% CI over 5 seeds)")
ax.legend(handles=[Line2D([],[],color=ACOL[a],lw=2,marker="o",label=ARCH[a]) for a in ARCH],frameon=False,loc="lower right",fontsize=8.5)
save(fig,"Fig5_ccit")

# ---------- Fig 6 attribution ----------
te=pd.read_csv(F+"tumour_enrichment.csv"); be=pd.read_csv(F+"background_enrichment_in_vs_cross.csv",header=[0,1],index_col=[0,1])
fig,axs=plt.subplots(1,2,figsize=(11,4))
ax=axs[0]; xs=np.arange(3); wd=0.36
for k,(m,col,lab) in enumerate([("gradcam",C["orange"],"Grad-CAM"),("intgrad",C["teal"],"Integrated Gradients")]):
    v=[te[(te.arch==a)&(te.method==m)].enrich_ratio_of_means.iloc[0] for a in ARCH]
    b=ax.bar(xs+(k-0.5)*wd,v,wd,color=col,label=lab)
    for xi,vi in zip(xs+(k-0.5)*wd,v): ax.text(xi,vi+0.08,f"{vi:.1f}×",ha="center",fontsize=9)
ax.axhline(1,ls="--",color=C["ink"],lw=1); ax.text(2.55,1.05,"no focus\n(share = area)",fontsize=8,color=C["grey"])
ax.set_xticks(xs,list(ARCH.values())); ax.set_ylabel("Tumour attribution ÷ tumour area"); ax.legend(frameon=False,fontsize=9); letter(ax,"a")
ax=axs[1]
for k,(sg,col,lab) in enumerate([("in",C["blue"],"Same source"),("cross",C["purple"],"New source")]):
    v=[be.loc[(a,"gradcam"),("bg_share",sg)] for a in ARCH]
    ax.bar(xs+(k-0.5)*wd,np.array(v)*100,wd,color=col,label=lab)
ar=np.mean([be.loc[(a,"gradcam"),("bg_area","in")] for a in ARCH])*100
ax.axhline(ar,ls="--",color=C["ink"],lw=1); ax.text(-0.45,ar+1,f"background area ≈ {ar:.0f}% of image",fontsize=8.5,color=C["grey"])
ax.set_xticks(xs,list(ARCH.values())); ax.set_ylabel("Grad-CAM attribution on background (%)"); ax.set_ylim(0,40); ax.legend(frameon=False,fontsize=9,loc="center right"); letter(ax,"b")
fig.tight_layout(); save(fig,"Fig6_attribution")

# ---------- Fig 7 remedies ----------
st=pd.read_csv(F+"addonC/sourceshield_stats.csv")
vl={"ERM_harm":"Harmonisation","mask_only":"Brain mask","DRO_only":"Group DRO","SourceShield":"SourceShield (mask + DRO)"}
vc={"ERM_harm":C["purple"],"mask_only":C["grey"],"DRO_only":C["teal"],"SourceShield":C["orange"]}
tests=[("in_source","Same-source test"),("external_bdneuro","BDNeuro-MRI (external)"),("external_pmram","PMRAM (external)")]
fig,axs=plt.subplots(1,3,figsize=(13,4.6),sharey=True)
for ax,(t,tt) in zip(axs,tests):
    y=0; yt=[]
    for v in vl:
        for a,mk in [("resnet18","o"),("efficientnet_b0","s")]:
            r_=st[(st.test==t)&(st.variant==v)&(st.arch==a)].iloc[0]
            ax.plot([r_.ci_lo,r_.ci_hi],[y,y],color=vc[v],lw=2.2); ax.scatter(r_.mean_diff_ba,y,color=vc[v],marker=mk,s=40,zorder=3,edgecolor="white")
            if r_.t_p_holm<0.05: ax.text(r_.ci_hi+0.005,y,"*",color=vc[v],fontsize=14,va="center")
            y+=1
        yt.append((y-1.5,vl[v])); y+=0.8
    ax.axvline(0,color=C["ink"],lw=1); ax.axvline(0.05,color=C["red"],ls="--",lw=1)
    ax.set_title(tt); ax.set_xlim(-0.12,0.16); ax.set_xlabel("Gain in balanced accuracy vs standard training")
axs[0].set_yticks([t[0] for t in yt],[t[1] for t in yt]); axs[0].invert_yaxis()
axs[1].text(0.058,1.9,"+0.05 pre-declared\nsuccess threshold",color=C["red"],fontsize=8.5,va="center")
axs[0].legend(handles=[Line2D([],[],marker="o",color=C["grey"],ls="",label="ResNet-18"),Line2D([],[],marker="s",color=C["grey"],ls="",label="EfficientNet-B0")],frameon=False,fontsize=8.5,loc="lower right")
fig.tight_layout(); save(fig,"Fig7_remedies")

# ---------- Fig 8 literature ----------
L=pd.read_csv(F + "P1_literature_inflation_audit_pilot.csv")
fig,axs=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={"width_ratios":[1,1.2]})
ax=axs[0]; ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
def box(x,y,w,h,t,col):
    ax.add_patch(matplotlib.patches.FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08,rounding_size=0.25",fc="white",ec=col,lw=2))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=9.5)
box(1,8.2,8,1.3,"Records identified (SciSpace, 2024–2026): 100",C["red"])
box(1,5.9,8,1.3,"After removing duplicate records: 91",C["red"])
box(1,3.6,8,1.3,"Coded: dataset, split unit, duplicate removal,\nexternal test, reported accuracy (91)",C["red"])
box(0.2,0.6,4.4,2.1,"Reported accuracy\navailable: 74",C["red"]); box(5.4,0.6,4.4,2.1,"Tested on another\ndataset: 7\n(4 of them recycled)",C["red"])
for (y1,y2) in [(8.2,7.2),(5.9,4.9)]: ax.annotate("",xy=(5,y2),xytext=(5,y1),arrowprops=dict(arrowstyle="-|>",color=C["grey"],lw=1.8))
ax.annotate("",xy=(2.4,2.7),xytext=(4,3.6),arrowprops=dict(arrowstyle="-|>",color=C["grey"],lw=1.8)); ax.annotate("",xy=(7.6,2.7),xytext=(6,3.6),arrowprops=dict(arrowstyle="-|>",color=C["grey"],lw=1.8))
ax.text(0,10.2,"a",fontsize=13,fontweight="bold")
ax=axs[1]
acc=pd.to_numeric(L.reported_acc,errors="coerce")
bins=np.arange(86,101,1)
ax.hist(acc.dropna(),bins=bins,color=C["red"],alpha=0.85,edgecolor="white")
for v,lab,col in [(98.4,"Literature median 98.4%",C["red"]),(94.5,"This study: official test 94.5%",C["teal"]),(76.5,"This study: clean test 76.5%",C["blue"]),(65.0,"This study: new source ≈ 65–79%",C["purple"])]:
    pass
ax.axvline(acc.median(),color=C["ink"],ls="--",lw=1.2); ax.text(acc.median()-0.2,ax.get_ylim()[1]*0.6,f"median\n{acc.median():.1f}%",ha="right",fontsize=9)
ax.set_xlabel("Reported test accuracy (%)"); ax.set_ylabel("Papers"); ax.set_xlim(86,100.5)
ins=ax.inset_axes([0.05,0.45,0.42,0.5])
labs=["Literature\n(median)","Official\ntest","Clean\ntest","New\nsource"]; vals=[98.4,94.6,76.5,70.4]; cols=[C["red"],C["teal"],C["blue"],C["purple"]]
ins.bar(range(4),vals,color=cols); ins.set_ylim(50,100); ins.set_xticks(range(4),labs,fontsize=7); ins.tick_params(axis="y",labelsize=7)
for i,v in enumerate(vals): ins.text(i,v+1,f"{v:.0f}",ha="center",fontsize=7)
ins.set_title("Same dataset family: reported accuracy vs\nbalanced accuracy measured in this study",fontsize=7)
ax.text(-0.08,1.06,"b",transform=ax.transAxes,fontsize=13,fontweight="bold")
fig.tight_layout(); save(fig,"Fig8_literature")
print("done")
