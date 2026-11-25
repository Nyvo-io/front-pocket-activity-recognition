"""Bonus experiments: adjacent-window majority voting, multi-level noise sweep, and per-method efficiency."""
import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from experiment import CLASSES, METHODS, WINDOW, aligned, classify, dictionary


def majority_vote_3(seq):
    """3-window majority vote (prev, curr, next) within a recording; center window breaks ties."""
    n = len(seq)
    if n < 3:
        return list(seq)
    out = list(seq)
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n, i + 2)
        window_votes = seq[lo:hi]
        counts = Counter(window_votes)
        best_cnt = max(counts.values())
        winners = [k for k, v in counts.items() if v == best_cnt]
        out[i] = seq[i] if seq[i] in winners else winners[0]
    return out


def run_voting_ablation(out: Path) -> pd.DataFrame:
    preds = pd.read_csv(out / "predictions.csv")
    clean = preds[preds.condition == "clean"].copy()
    voted = clean.copy()
    for (_, _), grp in clean.groupby(["subject", "step"], sort=False):
        for m in METHODS:
            voted.loc[grp.index, m] = majority_vote_3(grp[m].tolist())

    rows = []
    for scheme, df in [("single_window", clean), ("vote3_smooth", voted)]:
        for m in METHODS:
            acc = (df[m] == df.truth).mean()
            macro_rec = (df[m] == df.truth).groupby(df.truth).mean().mean()
            rows.append([scheme, m, len(df), acc, macro_rec])
    res = pd.DataFrame(rows, columns=["scheme", "method", "n", "accuracy", "macro_recall"])
    res.to_csv(out / "voting_ablation.csv", index=False)
    return res


def run_noise_sweep(data: Path, out: Path) -> pd.DataFrame:
    levels = (0.0, 0.25, 0.5, 1.0, 1.5)
    seeds = [20261125 + k for k in range(5)]

    # Cache dictionaries and clean aligned recordings once
    cached = []
    for subject in sorted(data.glob("subject*")):
        templates, classes, scale = dictionary(subject)
        recs = []
        for file in sorted((subject / "test").rglob("acc_front.csv")):
            x, labels = aligned(file.parent)
            starts = np.arange(3, len(x) - WINDOW + 1, 5)
            starts = starts[np.isin(labels[starts + 5], CLASSES)]
            if len(starts):
                recs.append((x, labels[starts + 5], starts[:, None] + np.arange(WINDOW)))
        cached.append((templates, classes, scale, recs))

    records = []
    for sigma in levels:
        active_seeds = [seeds[0]] if sigma == 0.0 else seeds
        run_metrics = {m: [] for m in METHODS}
        for sd in active_seeds:
            rng = np.random.default_rng(sd)
            truths = []
            preds_all = []
            for templates, classes, scale, recs in cached:
                for x, truth, indices in recs:
                    z = x.copy()
                    if sigma > 0.0:
                        z[:, 3:] += rng.normal(size=z[:, 3:].shape) * scale[3:] * sigma
                    preds = classify(z[indices], templates, classes, scale, False)
                    truths.append(truth)
                    preds_all.append(preds)
            y_true = np.concatenate(truths)
            y_pred = np.vstack(preds_all)
            for mi, m in enumerate(METHODS):
                pred_labels = np.array([CLASSES[k] for k in y_pred[:, mi]])
                recalls = [np.mean(pred_labels[y_true == cls] == cls) for cls in CLASSES]
                run_metrics[m].append((np.mean(pred_labels == y_true), float(np.mean(recalls))))

        for m in METHODS:
            accs = [v[0] for v in run_metrics[m]]
            recs = [v[1] for v in run_metrics[m]]
            records.append([
                sigma, m, len(active_seeds),
                float(np.mean(accs)), float(np.std(accs)),
                float(np.mean(recs)), float(np.std(recs)),
            ])

    df = pd.DataFrame(records, columns=[
        "noise_sigma", "method", "runs",
        "acc_mean", "acc_std", "macro_recall_mean", "macro_recall_std",
    ])
    df.to_csv(out / "noise_sweep.csv", index=False)
    return df


def run_efficiency_benchmark(data: Path, out: Path, repeats: int = 5) -> pd.DataFrame:
    # 1. Dictionary build time & memory
    t0 = time.perf_counter()
    subj_dicts = []
    total_templates = 0
    total_bytes_f64 = 0
    for subject in sorted(data.glob("subject*")):
        templates, classes, scale = dictionary(subject)
        total_templates += len(templates)
        total_bytes_f64 += templates.nbytes + classes.nbytes + scale.nbytes
        windows_list = []
        for file in sorted((subject / "test").rglob("acc_front.csv")):
            x, labels = aligned(file.parent)
            starts = np.arange(3, len(x) - WINDOW + 1, 5)
            starts = starts[np.isin(labels[starts + 5], CLASSES)]
            if len(starts):
                windows_list.append(x[starts[:, None] + np.arange(WINDOW)])
        subj_dicts.append((templates, classes, scale, np.vstack(windows_list)))
    prep_ms = (time.perf_counter() - t0) * 1000.0
    n_windows = sum(len(w) for _, _, _, w in subj_dicts)

    def bench_mode(mode: str) -> float:
        prepared = []
        for templates, classes, scale, wins in subj_dicts:
            if mode == "acc":
                prepared.append((np.ascontiguousarray(templates[:, :, :3]), classes, scale[:3], np.ascontiguousarray(wins[:, :, :3])))
            elif mode == "gyro":
                prepared.append((np.ascontiguousarray(templates[:, :, 3:]), classes, scale[3:], np.ascontiguousarray(wins[:, :, 3:])))
            else:
                prepared.append((templates, classes, scale, wins))
        times = []
        for _ in range(repeats):
            ts = time.perf_counter()
            for tmpl, classes, scale, wins in prepared:
                for j in range(0, len(wins), 32):
                    batch = wins[j:j + 32]
                    delta = batch[:, None, :, :] - tmpl[None, :, :, :]
                    d = np.sqrt(np.sum(delta * delta, axis=2))
                    if mode in ("acc", "gyro"):
                        score = d.mean(axis=2)
                    elif mode == "raw":
                        score = 0.5 * (d[:, :, :3].mean(axis=2) + d[:, :, 3:].mean(axis=2))
                    else:
                        score = (d / scale).mean(axis=2)
                    _ = classes[score.argmin(axis=1)]
            times.append((time.perf_counter() - ts) * 1000.0)
        return float(np.median(times))

    rows = []
    for m, axes_cnt in [("acc", 3), ("gyro", 3), ("raw", 6), ("scaled", 6)]:
        total_ms = bench_mode(m)
        mem_kb = (total_bytes_f64 * (axes_cnt / 6.0)) / 1024.0
        rows.append([m, axes_cnt, total_templates, round(mem_kb, 1), n_windows,
                     round(total_ms, 2), round(total_ms * 1000.0 / n_windows, 2), round(prep_ms, 2)])

    eff = pd.DataFrame(rows, columns=[
        "method", "axes", "templates_total", "dict_memory_kb",
        "test_windows", "total_infer_ms", "per_window_us", "data_prep_ms",
    ])
    eff.to_csv(out / "efficiency.csv", index=False)
    return eff


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()
    v = run_voting_ablation(args.out)
    print("=== Voting Ablation ===")
    print(v.to_string(index=False))
    ns = run_noise_sweep(args.data, args.out)
    print("\n=== Multi-level Noise Sweep ===")
    print(ns.to_string(index=False))
    ef = run_efficiency_benchmark(args.data, args.out)
    print("\n=== Efficiency Benchmark ===")
    print(ef.to_string(index=False))
