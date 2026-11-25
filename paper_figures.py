"""Build print figures from the supplied front-pocket demo and experiment outputs."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE
from experiment import CLASSES, WINDOW, aligned, dictionary

OUT = HERE / "results" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family":["Times New Roman","DejaVu Serif"], "mathtext.fontset":"stix",
                     "font.size":9, "axes.labelsize":9, "legend.fontsize":8,
                     "axes.spines.top":False, "axes.spines.right":False,
                     "axes.linewidth":0.6, "lines.linewidth":1.15, "savefig.dpi":350})
colors = ["#215278","#a65628","#2b7768","#563f6c"]
def save(fig, name):
    fig.savefig(OUT / (name+".png"),facecolor="white")
    plt.close(fig)

def examples():
    p=REPO/"demo/subject1"
    x, y=aligned(p/"test/Step1")
    t, c, scale=dictionary(p)
    fig, axes=plt.subplots(2,2,figsize=(7.12,3.5),layout="constrained",sharex="col")
    start=next(i for i in range(len(x)-39) if (y[i:i+40]=="walk").all())
    win=next(i for i in range(20,len(x)-WINDOW) if (y[i-20:i+WINDOW]=="walk").all())
    sample=x[win:win+WINDOW]
    best=(np.linalg.norm(t-sample,axis=1)/scale).mean(axis=1).argmin()
    for k in range(2):
        a,b=axes[k]
        for j in range(3):
            a.plot(x[start:start+40,3*k+j],color=colors[j],ls=["-","--",":"][j],label="xyz"[j])
        b.plot(sample[:,3*k],"o-",ms=3,color=colors[0],label="Test: walk")
        b.plot(t[best,:,3*k],"s--",ms=3,color=colors[1],label="Template: "+CLASSES[c[best]])
        a.set_ylabel(["Acceleration (g)","Angular rate (rad/s)"][k])
        b.set_ylabel(["Acceleration, x (g)","Angular rate, x (rad/s)"][k])
    for i,title in enumerate(["(a)  Walking segment","(b)  Nearest training template"]):
        axes[0,i].set_title(title,fontsize=10)
        axes[0,i].legend(loc="upper center",bbox_to_anchor=(.5,1.02),ncol=3,frameon=False)
        axes[0,i].margins(y=.34)
        axes[1,i].set_xlabel("Sample index")
    save(fig,"examples")

def confusion():
    fig,axs=plt.subplots(1,2,figsize=(7.12,3.05),layout="constrained",sharey=True)
    for ax,m,title in zip(axs,["raw","scaled"],["(a)  Raw fusion","(b)  Scaled fusion"]):
        cm=pd.read_csv(REPO/"results"/("confusion_"+m+".csv"),index_col=0).loc[CLASSES,CLASSES]
        pct=100*cm.div(cm.sum(axis=1),axis=0)
        ax.imshow(pct,vmin=0,vmax=100,cmap="Blues")
        for i,j in np.ndindex(5,5):
            val=pct.iloc[i,j]
            ax.text(j,i,f"{val:.1f}",ha="center",va="center",color="white" if val>55 else "black")
        labels=["Sit","Stand","Walk","Up","Down"]
        ax.set(xticks=range(5),yticks=range(5),xticklabels=labels,yticklabels=labels,xlabel="Predicted activity")
        ax.set_title(title,fontsize=10)
        ax.tick_params(length=0,pad=3)
    axs[0].set_ylabel("Recorded activity")
    save(fig,"confusion")

def trends():
    results=REPO/"results"
    fig, ax=plt.subplots(figsize=(3.45,2.35),layout="constrained")
    lag=pd.read_csv(results/"lag_sweep.csv")
    for i,(m,label) in enumerate([("raw","Raw fusion"),("scaled","Scaled fusion")]):
        ax.plot(lag.lag,lag[m]*100,["o-","s--"][i],ms=3.5,color=colors[i],label=label)
    ax.set(xlabel="Gyroscope delay (samples)",ylabel="Macro recall (%)",xticks=lag.lag,ylim=(62,75))
    ax.legend(frameon=False,ncol=2,loc="upper center")
    save(fig,"lag")
    fig, ax=plt.subplots(figsize=(3.45,2.6),layout="constrained")
    speed=pd.read_csv(results/"by_step.csv").set_index("step").loc[["Step3/"+s+"_walk" for s in ("slow","normal","fast")]]
    for i,(m,label) in enumerate([("acc","Accelerometer"),("gyro","Gyroscope"),("raw","Raw fusion"),("scaled","Scaled fusion")]):
        ax.plot(["Slow","Normal","Fast"],speed[m]*100,["o-","s--","^:","d-."][i],ms=3.5,color=colors[i],label=label)
    ax.set(ylabel="Walking recall (%)",ylim=(0,105))
    ax.legend(frameon=False,ncol=2,loc="lower center")
    save(fig,"speed")
def error_slices():
    frame=pd.read_csv(REPO/"results/predictions.csv")
    clean=frame[frame.condition=="clean"]
    walk=clean[(clean.truth=="walk") & clean.step.str.startswith("Step4/")]
    counts=pd.crosstab(walk.step,walk.raw).reindex(columns=CLASSES,fill_value=0)
    pct=100*counts.div(counts.sum(axis=1),axis=0)
    fig,ax=plt.subplots(figsize=(3.45,2.65),layout="constrained")
    im=ax.imshow(pct,aspect="auto",cmap="Blues",vmin=0,vmax=100)
    for i,j in np.ndindex(pct.shape):
        val=pct.iloc[i,j]
        ax.text(j,i,f"{val:.0f}",ha="center",va="center",color="white" if val>55 else "black")
    ax.set(xticks=range(5),xticklabels=["Sit","Stand","Walk","Up","Down"],
           yticks=range(4),yticklabels=["Down / toward","Down / away","Up / toward","Up / away"],
           xlabel="Predicted activity",ylabel="Phone / screen orientation")
    ax.tick_params(length=0)
    save(fig,"pose")
    fig,ax=plt.subplots(figsize=(3.45,2.5),layout="constrained")
    x=np.arange(len(CLASSES))
    for i,(m,label) in enumerate([("raw","Raw fusion"),("scaled","Scaled fusion")]):
        recall=frame[m].eq(frame.truth).groupby([frame.condition,frame.truth]).mean().unstack()
        diff=(recall.loc["delay"]-recall.loc["clean"]).reindex(CLASSES)*100
        ax.bar(x+(i-.5)*.35,diff,.35,label=label,color=colors[i])
    ax.axhline(0,color="black",lw=.6)
    ax.set(xticks=x,xticklabels=["Sit","Stand","Walk","Up","Down"],ylabel="Recall change (pp)")
    ax.legend(frameon=False,loc="lower left",fontsize=8)
    save(fig,"change")

if __name__=="__main__":
    examples()
    confusion()
    trends()
    error_slices()