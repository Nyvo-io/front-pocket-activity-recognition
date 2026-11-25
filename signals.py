"""Plot measured signals and the nearest training templates for one test window."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment import aligned, dictionary, CLASSES, WINDOW

def draw(data, out):
    out.mkdir(parents=True, exist_ok=True)
    subject = data / "subject1"
    x, labels = aligned(subject / "test" / "Step1")
    # First complete walking window preceded by two seconds of walking.
    i = next(k for k in range(20,len(x)-WINDOW) if (labels[k-20:k+WINDOW]=="walk").all())
    templates, names, scale = dictionary(subject)
    window = x[i:i+WINDOW]
    axes = np.linalg.norm(templates-window,axis=1)
    match = (axes/scale).mean(axis=1).argmin()
    fig,plots=plt.subplots(2,1,figsize=(5.6,4.1),sharex=True)
    for d,ax in enumerate(plots):
        axis = 3*d
        ax.plot(window[:,axis], "o-",color="#174d70",label="Test: walk")
        ax.plot(templates[match,:,axis], "s--",color="#bd753d",label="Template: "+CLASSES[names[match]])
        ax.set_ylabel(["Acc x (g)","Gyro x (rad/s)"][d])
    plots[0].legend(frameon=False,ncol=2,fontsize=9)
    plots[1].set_xlabel("Sample in window")
    fig.tight_layout()
    fig.savefig(out/"match.png",dpi=210,bbox_inches="tight")
    plt.close(fig)
    fig,plots=plt.subplots(2,1,figsize=(5.6,4.1),sharex=True)
    k=next(j for j in range(0,len(x)-40) if (labels[j:j+40]=="walk").all())
    for d,ax in enumerate(plots):
        for axis,color in enumerate(["#164f73","#cc793d","#46928c"]):
            ax.plot(np.arange(40),x[k:k+40,3*d+axis],color=color,label="xyz"[axis])
        ax.set_ylabel(["Acc (g)","Gyro (rad/s)"][d])
    plots[0].legend(frameon=False,ncol=3)
    plots[1].set_xlabel("Aligned sample index")
    fig.tight_layout()
    fig.savefig(out/"record.png",dpi=210,bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("results"))
    args=p.parse_args()
    draw(args.data,args.out)
