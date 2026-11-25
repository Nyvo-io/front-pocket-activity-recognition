"""Summarize predictions and render figures from an experiment directory."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment import CLASSES, METHODS

def main(folder):
    frame = pd.read_csv(folder / "predictions.csv")
    clean = frame[frame.condition == "clean"].copy()
    clean["protocol"] = clean.step.str.split("/").str[0]
    colors = ["#8ca6b8", "#3e7d83", "#184b70", "#c4753a"]
    labels = ["Acc", "Gyro", "Raw fusion", "Scaled fusion"]
    for group in ("truth","step","subject","protocol"):
        rows = []
        for key, part in clean.groupby(group):
            rows.append([key,len(part),*[(part[m] == part.truth).mean() for m in METHODS]])
        pd.DataFrame(rows,columns=[group,"n",*METHODS]).to_csv(folder / f"by_{group}.csv",index=False)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    scores = pd.read_csv(folder / "scores.csv")
    def save(name):
        plt.tight_layout()
        plt.savefig(folder / (name + ".png"),dpi=210,bbox_inches="tight")
        plt.close()
    fig, ax = plt.subplots(figsize=(5.7,2.9))
    base = scores[scores.condition == "clean"].set_index("method").loc[METHODS]
    x = np.arange(4)
    ax.bar(x-.17,base.accuracy*100,.34,label="Accuracy",color=colors[1])
    ax.bar(x+.17,base.macro_recall*100,.34,label="Macro recall",color=colors[2])
    ax.set(xticks=x,xticklabels=labels,ylabel="Score (%)",ylim=(0,100))
    ax.legend(ncol=2,frameon=False)
    save("baseline")
    for method in ["raw","scaled"]:
        table = pd.crosstab(clean.truth,clean[method]).reindex(index=CLASSES,columns=CLASSES,fill_value=0)
        table.to_csv(folder / f"confusion_{method}.csv")
        pct = table.div(table.sum(axis=1).replace(0,np.nan),axis=0)*100
        fig,ax = plt.subplots(figsize=(4.4,3.1))
        ax.imshow(pct,vmin=0,vmax=100,cmap="Blues",aspect="auto")
        for i,j in np.ndindex(5,5):
            if pd.notna(pct.iloc[i,j]):
                ax.text(j,i,f"{pct.iloc[i,j]:.0f}",ha="center",va="center",
                        color="white" if pct.iloc[i,j]>55 else "black")
        short=["Sit","Stand","Walk","Up","Down"]
        ax.set(xticks=range(5),yticks=range(5),xticklabels=short,yticklabels=short,
               xlabel="Predicted activity",ylabel="True activity")
        save("matrix_"+method)
    fig,ax=plt.subplots(figsize=(5.7,3.0))
    for m,c,l in zip(METHODS,colors,labels):
        part=scores[scores.method==m]
        ax.plot(part.condition,part.macro_recall*100,"o-",label=l,color=c)
    ax.set(ylabel="Macro recall (%)",ylim=(50,83))
    ax.legend(ncol=2,frameon=False,fontsize=8)
    save("stress")
    fig,ax=plt.subplots(figsize=(5.7,3.0))
    data=pd.read_csv(folder/"by_protocol.csv")
    for m,c,l in zip(METHODS,colors,labels):
        ax.plot(data.protocol,data[m]*100,"o-",color=c,label=l)
    ax.set(ylabel="Accuracy (%)",ylim=(0,105))
    ax.legend(ncol=2,frameon=False,fontsize=8)
    save("steps")

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("results"))
    main(p.parse_args().out)
