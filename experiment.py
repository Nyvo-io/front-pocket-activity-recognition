"""Personal inertial-template matching on paired front-pocket recordings."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["sit", "stand", "walk", "stairUp", "stairDown"]
METHODS = ["acc", "gyro", "raw", "scaled"]
WINDOW = 10


def aligned(path):
    acc, gyro = [pd.read_csv(path / f"{sensor}_front.csv", keep_default_na=False)
                 for sensor in ("acc", "gyro")]
    clock = acc.columns[0]
    t = acc[clock].to_numpy()
    g = gyro[clock].to_numpy()
    keep = (t >= g[0]) & (t <= g[-1])
    a = acc.loc[keep, ["x", "y", "z"]].to_numpy()
    b = np.column_stack([np.interp(t[keep], g, gyro[axis]) for axis in "xyz"])
    labels = acc.loc[keep, "label"].str.replace(r"[12]$", "", regex=True)
    labels = labels.replace(dict.fromkeys(["slow", "normal", "fast"], "walk"))
    return np.column_stack([a, b]), labels.to_numpy()


def dictionary(subject):
    rows, windows, classes = [], [], []
    for file in sorted((subject / "training").rglob("acc_front.csv")):
        x, y = aligned(file.parent)
        rows.append(x[np.isin(y, CLASSES)])
        for i in range(len(x) - WINDOW + 1):
            if y[i] in CLASSES and np.all(y[i:i + WINDOW] == y[i]):
                windows.append(x[i:i + WINDOW])
                classes.append(CLASSES.index(y[i]))
    scale = np.maximum(np.concatenate(rows).std(axis=0), 1e-8)
    return np.array(windows), np.array(classes), scale


def classify(windows, templates, classes, scale, missing):
    predictions = []
    for j in range(0, len(windows), 32):
        axes = slice(0, 3) if missing else slice(None)
        delta = windows[j:j + 32, None, :, axes] - templates[None, :, :, axes]
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        acc = distances[:, :, :3].mean(axis=2)
        acc_pred = classes[acc.argmin(axis=1)]
        if missing:
            raw = acc
            gyro_pred = np.full(len(acc), -1, dtype=int)
        else:
            gyro = distances[:, :, 3:].mean(axis=2)
            gyro_pred = classes[gyro.argmin(axis=1)]
            raw = (acc + gyro) / 2
        scaled = (distances / scale[axes]).mean(axis=2)
        predictions.extend(np.column_stack([acc_pred, gyro_pred, classes[raw.argmin(axis=1)],
                                            classes[scaled.argmin(axis=1)]]))
    return np.array(predictions)


def run(data, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, counts = [], []
    rng = np.random.default_rng(20261125)
    for subject in sorted(data.glob("subject*")):
        templates, classes, scale = dictionary(subject)
        for k, name in enumerate(CLASSES):
            counts.append([subject.name, name, int((classes == k).sum())])
        for file in sorted((subject / "test").rglob("acc_front.csv")):
            x, labels = aligned(file.parent)
            starts = np.arange(3, len(x) - WINDOW + 1, 5)
            starts = starts[np.isin(labels[starts + 5], CLASSES)]
            if not len(starts):
                continue
            indices = starts[:, None] + np.arange(WINDOW)
            for condition in ("clean", "delay", "noise", "missing"):
                z = x.copy()
                if condition == "delay":
                    z[3:, 3:] = x[:-3, 3:]
                elif condition == "noise":
                    z[:, 3:] += rng.normal(size=z[:, 3:].shape) * scale[3:] * 0.5
                preds = classify(z[indices], templates, classes, scale, condition == "missing")
                for start, pred in zip(starts, preds):
                    values = [CLASSES[k] if k >= 0 else "" for k in pred]
                    rows.append([subject.name, str(file.parent.relative_to(subject / "test")),
                                 start + 5, condition, labels[start + 5], *values])
    if not rows:
        raise ValueError("No evaluable windows. Check the data directory.")
    frame = pd.DataFrame(rows, columns=["subject","step","center_index","condition","truth",*METHODS])
    frame.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(counts, columns=["subject","activity","templates"]).to_csv(out / "dictionary.csv", index=False)
    records = []
    for condition, part in frame.groupby("condition", sort=False):
        for name in METHODS:
            if (part[name] == "").all():
                continue
            recalls = (part[name] == part.truth).groupby(part.truth).mean()
            records.append([condition, name, len(part), (part[name] == part.truth).mean(), recalls.mean()])
    summary = pd.DataFrame(records, columns=["condition","method","n","accuracy","macro_recall"])
    summary.to_csv(out / "scores.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.data, args.out)
