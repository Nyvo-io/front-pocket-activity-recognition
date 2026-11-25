"""Stratify the existing clean predictions by the ten original window labels."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import aligned, WINDOW, METHODS

def summarize(data, results):
    predictions = pd.read_csv(results / "predictions.csv")
    clean = predictions.loc[predictions.condition == "clean"].copy()
    for (subject, step), group in clean.groupby(["subject", "step"]):
        _, labels = aligned(data / subject / "test" / step)
        center = group.center_index.to_numpy()
        truth = group.truth.to_numpy()
        if not np.array_equal(labels[center], truth):
            raise ValueError(f"Labels do not match saved predictions: {subject}/{step}")
        offsets = np.arange(WINDOW) - 5
        uniform = (labels[center[:, None] + offsets] == truth[:, None]).all(axis=1)
        clean.loc[group.index, "context"] = np.where(uniform, "uniform", "mixed")
    rows = []
    for (context, activity), group in clean.groupby(["context", "truth"]):
        rows.append([context, activity, len(group), *[int(group[m].eq(group.truth).sum()) for m in METHODS]])
    pd.DataFrame(rows, columns=["context","truth","windows",*METHODS]).to_csv(
        results / "window_context.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    summarize(args.data,args.out)
