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

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Songti SC", "STSong", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.65,
    "lines.linewidth": 1.2,
    "savefig.dpi": 350,
})

COLORS = ["#215278", "#a65628", "#2b7768", "#563f6c"]
CN_LABELS = ["坐姿", "站立", "步行", "上楼", "下楼"]
CN_MAP = dict(zip(CLASSES, CN_LABELS))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def examples():
    p = REPO / "demo/subject1"
    x, y = aligned(p / "test/Step1")
    t, c, _ = dictionary(p)
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 3.35), layout="constrained", sharex="col")
    start = next(i for i in range(len(x) - 39) if (y[i:i + 40] == "walk").all())
    win = next(i for i in range(20, len(x) - WINDOW) if (y[i - 20:i + WINDOW] == "walk").all())
    sample = x[win:win + WINDOW]
    best = np.linalg.norm(t - sample, axis=1).mean(axis=1).argmin()
    for k in range(2):
        a, b = axes[k]
        for j in range(3):
            a.plot(x[start:start + 40, 3 * k + j], color=COLORS[j], ls=["-", "--", ":"][j], label="xyz"[j])
        b.plot(sample[:, 3 * k], "o-", ms=3.2, color=COLORS[0], label="测试窗（步行）")
        b.plot(t[best, :, 3 * k], "s--", ms=3.2, color=COLORS[1], label=f"最近模板（{CN_MAP[CLASSES[c[best]]]}）")
        a.set_ylabel(["加速度 (g)", "角速度 (rad/s)"][k])
        b.set_ylabel(["x 轴加速度 (g)", "x 轴角速度 (rad/s)"][k])
        a.grid(axis="y", ls=":", lw=0.4, alpha=0.45)
        b.grid(axis="y", ls=":", lw=0.4, alpha=0.45)
    for i, title in enumerate(["(a) 连续 40 点步行信号片段", "(b) 测试窗口与最近训练模板"]):
        axes[0, i].set_title(title, fontsize=9.2, pad=4)
        axes[0, i].legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
        axes[0, i].margins(y=0.34)
    axes[1, 0].set_xlabel("对齐后采样点序号")
    axes[1, 1].set_xlabel("窗口内采样点序号")
    save(fig, "examples")


def confusion():
    fig, axs = plt.subplots(1, 2, figsize=(6.35, 2.75), layout="constrained", sharey=True)
    im = None
    for ax, m, title in zip(axs, ["raw", "scaled"], ["(a) 原尺度融合", "(b) 按轴缩放融合"]):
        cm = pd.read_csv(REPO / "results" / f"confusion_{m}.csv", index_col=0).loc[CLASSES, CLASSES]
        pct = 100 * cm.div(cm.sum(axis=1), axis=0)
        im = ax.imshow(pct, vmin=0, vmax=100, cmap="Blues")
        for i, j in np.ndindex(5, 5):
            val = pct.iloc[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.8,
                    color="white" if val > 55 else "#1a1a1a")
        ax.set(xticks=range(5), yticks=range(5), xticklabels=CN_LABELS, yticklabels=CN_LABELS, xlabel="预测活动")
        ax.set_title(title, fontsize=9.2, pad=5)
        ax.tick_params(length=0, pad=3)
    axs[0].set_ylabel("真实活动")
    cbar = fig.colorbar(im, ax=axs, shrink=0.88, pad=0.02, aspect=22)
    cbar.set_label("行百分比 (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)
    save(fig, "confusion")


def trends():
    results = REPO / "results"
    fig, ax = plt.subplots(figsize=(3.38, 2.25), layout="constrained")
    lag = pd.read_csv(results / "lag_sweep.csv")
    for i, (m, label) in enumerate([("raw", "原尺度融合"), ("scaled", "按轴缩放融合")]):
        ax.plot(lag.lag, lag[m] * 100, ["o-", "s--"][i], ms=3.8, color=COLORS[i], label=label)
    ax.set(xlabel="陀螺仪延迟（采样点数）", ylabel="宏平均召回率 (%)", xticks=lag.lag, ylim=(62, 75))
    ax.grid(axis="y", ls=":", lw=0.45, alpha=0.5)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    save(fig, "lag")

    fig, ax = plt.subplots(figsize=(3.38, 2.45), layout="constrained")
    speed = pd.read_csv(results / "by_step.csv").set_index("step").loc[[f"Step3/{s}_walk" for s in ("slow", "normal", "fast")]]
    styles = [("acc", "单加速度计", "o-"), ("gyro", "单陀螺仪", "s--"), ("raw", "原尺度融合", "^:"), ("scaled", "按轴缩放融合", "d-.")]
    for i, (m, label, st) in enumerate(styles):
        ax.plot(["慢走", "正常步速", "快走"], speed[m] * 100, st, ms=4.0, color=COLORS[i], label=label)
    ax.set(xlabel="步行速度条件", ylabel="步行召回率 (%)", ylim=(0, 106))
    ax.grid(axis="y", ls=":", lw=0.45, alpha=0.5)
    ax.legend(frameon=False, ncol=2, loc="lower center")
    save(fig, "speed")


def error_slices():
    frame = pd.read_csv(REPO / "results/predictions.csv")
    clean = frame[frame.condition == "clean"]
    walk = clean[(clean.truth == "walk") & clean.step.str.startswith("Step4/")]
    counts = pd.crosstab(walk.step, walk.raw).reindex(columns=CLASSES, fill_value=0)
    pct = 100 * counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(3.42, 2.42), layout="constrained")
    im = ax.imshow(pct, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    for i, j in np.ndindex(pct.shape):
        val = pct.iloc[i, j]
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.4,
                color="white" if val > 55 else "#1a1a1a")
    ax.set(
        xticks=range(5),
        xticklabels=CN_LABELS,
        yticks=range(4),
        yticklabels=["倒置 / 朝腿", "倒置 / 背腿", "正置 / 朝腿", "正置 / 背腿"],
        xlabel="预测活动",
        ylabel="机身 / 屏幕朝向",
    )
    ax.tick_params(length=0, pad=3)
    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.03, aspect=18)
    cbar.set_label("行百分比 (%)", fontsize=7.8)
    cbar.ax.tick_params(labelsize=7.2)
    save(fig, "pose")

    fig, ax = plt.subplots(figsize=(3.38, 2.35), layout="constrained")
    x = np.arange(len(CLASSES))
    for i, (m, label) in enumerate([("raw", "原尺度融合"), ("scaled", "按轴缩放融合")]):
        recall = frame[m].eq(frame.truth).groupby([frame.condition, frame.truth]).mean().unstack()
        diff = (recall.loc["delay"] - recall.loc["clean"]).reindex(CLASSES) * 100
        ax.bar(x + (i - 0.5) * 0.36, diff, 0.36, label=label, color=COLORS[i])
        if abs(diff.iloc[0]) < 0.05 and i == 0:
            ax.text(0, 0.25, "0.0", ha="center", va="bottom", fontsize=7.2, color="#444444")
    ax.axhline(0, color="black", lw=0.65)
    ax.grid(axis="y", ls=":", lw=0.45, alpha=0.5)
    ax.set(xticks=x, xticklabels=CN_LABELS, xlabel="活动类别", ylabel="召回率变化（百分点）", ylim=(-10.2, 1.2))
    ax.legend(frameon=False, loc="lower left", fontsize=7.8)
    save(fig, "change")


if __name__ == "__main__":
    examples()
    confusion()
    trends()
    error_slices()
