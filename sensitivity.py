"""Sweep gyroscope lag over the same windows, with a seven-sample history."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment import aligned, dictionary, classify, CLASSES, WINDOW

def sweep(data, out):
    results = []
    for subject in sorted(data.glob("subject*")):
        t, c, scale = dictionary(subject)
        for file in sorted((subject / "test").rglob("acc_front.csv")):
            x, y = aligned(file.parent)
            starts = np.arange(8,len(x)-WINDOW+1,5)
            starts = starts[np.isin(y[starts+5],CLASSES)]
            if not len(starts):
                continue
            indices = starts[:,None]+np.arange(WINDOW)
            for lag in (0,1,2,3,5,7):
                z = x[indices].copy()
                z[:,:,3:] = x[indices-lag,3:]
                predictions = classify(z,t,c,scale,False)
                for label, pred in zip(y[starts+5],predictions):
                    results.append([lag,label,CLASSES[pred[2]],CLASSES[pred[3]]])
    frame=pd.DataFrame(results,columns=["lag","truth","raw","scaled"])
    rows=[]
    for lag,part in frame.groupby("lag"):
        rows.append([lag,len(part),*[(part[m]==part.truth).groupby(part.truth).mean().mean()
                                    for m in ("raw","scaled")]])
    scores=pd.DataFrame(rows,columns=["lag","n","raw","scaled"])
    out.mkdir(exist_ok=True,parents=True)
    scores.to_csv(out/"lag_sweep.csv",index=False)
    fig,ax=plt.subplots(figsize=(5.6,2.9))
    for col, label, color in [("raw","Raw fusion","#174d70"),("scaled","Scaled fusion","#bd753d")]:
        ax.plot(scores.lag,100*scores[col],"o-",label=label,color=color)
    ax.set(xlabel="Gyroscope lag (samples)",ylabel="Macro recall (%)",xticks=scores.lag)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out/"lag_sweep.png",dpi=210,bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("results"))
    a=p.parse_args()
    sweep(a.data,a.out)
