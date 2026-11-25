"""Compare class support, personal scales, and accuracy by walking speed."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment import dictionary, CLASSES

def plot(data, out):
    counts=pd.read_csv(out/"dictionary.csv").groupby("activity").templates.sum().reindex(CLASSES)
    truth=pd.read_csv(out/"by_truth.csv").set_index("truth").n.reindex(CLASSES)
    fig,ax=plt.subplots(figsize=(5.5,3.05))
    x=np.arange(5)
    for offset,series,label,color in [(-.18,counts,"Training templates","#337e83"),(.18,truth,"Test windows","#194e73")]:
        ax.bar(x+offset,100*series/series.sum(),.36,label=label,color=color)
    ax.set(xticks=x,xticklabels=["Sit","Stand","Walk","Up","Down"],ylabel="Share (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out/"support.png",dpi=210)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(5.6,2.8))
    for p in sorted(data.glob("subject*")):
        _,_,scale=dictionary(p)
        for k,ax in enumerate(axes):
            ax.plot(["x","y","z"],scale[k*3:k*3+3],"o-",label=p.name.replace("subject","S"))
    for ax,label in zip(axes,["Acc SD (g)","Gyro SD (rad/s)"]):
        ax.set(ylabel=label,ylim=(0,None))
    axes[0].legend(ncol=2,fontsize=8,frameon=False)
    fig.tight_layout()
    fig.savefig(out/"scales.png",dpi=210)
    plt.close(fig)
    data=pd.read_csv(out/"by_step.csv").set_index("step")
    keys=["Step3/"+s+"_walk" for s in ("slow","normal","fast")]
    if not all(k in data.index for k in keys):
        return
    fig,ax=plt.subplots(figsize=(5.6,2.9))
    for col,label,color in [("acc","Acc","#8ca6b8"),("gyro","Gyro","#3e7d83"),("raw","Raw fusion","#184b70"),("scaled","Scaled fusion","#c4753a")]:
        ax.plot(["Slow","Normal","Fast"],100*data.loc[keys,col],"o-",label=label,color=color)
    ax.set(ylabel="Walking recall (%)",ylim=(0,105))
    ax.legend(ncol=2,fontsize=8,frameon=False)
    fig.tight_layout()
    fig.savefig(out/"speed.png",dpi=210)
    plt.close(fig)

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("results"))
    a=p.parse_args()
    plot(a.data,a.out)
