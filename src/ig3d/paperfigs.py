"""Simple, labelled figures for the paper, read from datasets/results."""
from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import DEFAULT  # noqa: E402
from .costfield import gt_field_from_features, volume_features  # noqa: E402
from .instructions import CLASSES  # noqa: E402
from .lidar import BORDER, REAL_CLASSES, SIZE, VOXEL_M, blocks_path, load_blocks  # noqa: E402
from .lidar import OUT as AUSTIN  # noqa: E402
from .metrics import behaviour  # noqa: E402
from .multiseed import load, matrix, stats  # noqa: E402
from .paths import FIGURES, RESULTS  # noqa: E402
from .planner_grid import GridGraph  # noqa: E402

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
ONE, TWO, THREE = "#eb6834", "#1baf7a", "#2a78d6"
NLI, EMBED = "#eda100", "#008300"
SYNTH, REAL = "#86b6ef", "#184f95"
TIER1 = ("open", "tight", "low", "high", "shortest")
TIER3 = ("climb_open", "descend_open")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5})


def style(ax, ylabel=None, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, length=0)
    if grid:
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)


def save(fig, name):
    out = FIGURES / (name + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def bars(ax, groups, series, width=0.26, fmt="{:.1f}", inside=False):
    """series: (label, colour, means, lows, highs), values in percent; lows/highs may be None."""
    x = np.arange(len(groups))
    n = len(series)
    tops = [np.asarray(hi if hi is not None else mean, float) for _, _, mean, lo, hi in series]
    peak = float(np.max(tops))
    for i, (label, colour, mean, lo, hi) in enumerate(series):
        pos = x + (i - (n - 1) / 2) * width
        mean = np.asarray(mean, float)
        ax.bar(pos, mean, width * 0.88, color=colour, label=label, zorder=2)
        top = mean
        if lo is not None:
            lo, hi = np.asarray(lo, float), np.asarray(hi, float)
            ax.errorbar(pos, mean, yerr=[np.maximum(mean - lo, 0), np.maximum(hi - mean, 0)],
                        fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.8, zorder=3)
            top = np.maximum(hi, mean)
        for p, m, t in zip(pos, mean, top):
            if inside:
                ax.text(p, m - 0.03 * peak, fmt.format(m), ha="center", va="top", fontsize=6.5,
                        color=INK, zorder=6)
            else:
                ax.text(p, t + 0.03 * peak, fmt.format(m), ha="center", va="bottom", fontsize=6.5,
                        color=INK2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, peak * 1.2)


def summary(rows_by_seed, classes):
    st = stats(matrix(rows_by_seed, "grid_regret", list(classes)))
    return 100 * st["mean"], 100 * st["lo"], 100 * st["hi"]


def ladder_summary(rows, classes, key):
    by = {}
    for r in rows:
        if r["cls"] in classes:
            by.setdefault(r["vol"], []).append(r[key])
    st = stats(np.array([[np.mean(by[v]) for v in sorted(by)]]))
    return 100 * st["mean"], 100 * st["lo"], 100 * st["hi"]


def austin_rows():
    data = {}
    for f in sorted(AUSTIN.glob("e1_held_*_s*.json")):
        d = json.loads(f.read_text())
        data.setdefault(d["name"], {})[d["seed"]] = d["rows"]
    return data


def top_legend(fig, axes, ncol, rect_top):
    handles, labels = {}, []
    for ax in np.atleast_1d(axes):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in handles:
                handles[lab] = h
                labels.append(lab)
    fig.legend([handles[lab] for lab in labels], labels, loc="upper center", ncol=ncol,
               frameon=False, bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, rect_top))


def fig_paths():
    blocks = json.loads(blocks_path().with_suffix(".json").read_text())["blocks"]
    vols = load_blocks()
    cfg = DEFAULT.cost
    for b in (b for b in blocks if b["used"]):
        v = vols[b["index"]]
        G = GridGraph(v.occ)
        f = volume_features(v, cfg)
        plans = {c: G.plan(gt_field_from_features(v.occ, f, c, cfg, 3), v.start, v.goal)[0]
                 for c in ("level", "shortest", "low", "high")}
        beh = {c: behaviour(v, p, f["edt"]) for c, p in plans.items()}
        if all(abs(beh[x][m] - beh[y][m]) > 1e-6
               for x, y, m in (("level", "shortest", "vertical"), ("low", "high", "mean_alt"))):
            break
    height = v.occ[:-1].sum(axis=0) * VOXEL_M
    inner = height[BORDER:-BORDER, BORDER:-BORDER]
    lo_m, hi_m = BORDER * VOXEL_M, (SIZE - BORDER) * VOXEL_M
    fig, (ax_top, ax_side) = plt.subplots(2, 1, figsize=(3.5, 5.0),
                                          gridspec_kw=dict(height_ratios=(2.2, 1)))
    im = ax_top.imshow(inner, origin="lower", cmap="Greys", vmin=0, vmax=float(inner.max()),
                       extent=(lo_m, hi_m, lo_m, hi_m), interpolation="nearest")
    cb = fig.colorbar(im, ax=ax_top, fraction=0.046, pad=0.03)
    cb.set_label("roof or ground height (m)", color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, length=0)
    specs = (("level", THREE, "-", "keep it level ({:.0f} m up and down)"),
             ("shortest", ONE, (0, (3, 1.5)), "shortest path ({:.0f} m up and down)"))
    for cls, colour, ls, label in specs:
        p = np.asarray(plans[cls], float)
        text = label.format(beh[cls]["vertical"] * VOXEL_M)
        ax_top.plot((p[:, 2] + 0.5) * VOXEL_M, (p[:, 1] + 0.5) * VOXEL_M, color=colour, lw=2,
                    ls=ls, label=text, zorder=3)
        dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(p[:, 1]), np.diff(p[:, 2])))]) * VOXEL_M
        if cls == "shortest":
            under = height[p[:, 1].astype(int), p[:, 2].astype(int)]
            ax_side.fill_between(dist, 0, under, step="mid", color="#d7d6cf", lw=0, zorder=1,
                                 label="roofs and ground under the shortest path")
        ax_side.plot(dist, (p[:, 0] + 0.5) * VOXEL_M, color=colour, lw=2, ls=ls, zorder=3)
    for pt, name, marker, size, dy in ((v.start, "start", "o", 5, -9), (v.goal, "goal", "*", 10, 2)):
        x, y = (pt[2] + 0.5) * VOXEL_M, (pt[1] + 0.5) * VOXEL_M
        ax_top.plot(x, y, marker, color=INK, ms=size, zorder=4)
        ax_top.annotate(name, (x, y), xytext=(6, dy), textcoords="offset points", color=INK,
                        fontsize=7.5)
    ax_top.set_title("Top view", color=INK, loc="left")
    ax_top.set_xlabel("east (m)", color=INK2)
    ax_top.set_ylabel("north (m)", color=INK2)
    style(ax_top, grid=False)
    ax_side.set_title("Side view: altitude along each path", color=INK, loc="left")
    ax_side.set_xlabel("distance along the path (m)", color=INK2)
    style(ax_side, "altitude (m)")
    ax_side.set_xlim(left=0)
    ax_side.set_ylim(bottom=0)
    top_legend(fig, (ax_top, ax_side), 1, 0.88)
    return save(fig, "paths_keep_level")


def fig_channels():
    synth = json.loads((RESULTS / "e2.json").read_text())["rows"]
    real = json.loads((AUSTIN / "ladder.json").read_text())
    groups = ["keep it\nlevel", "climb in\nthe open", "descend in\nthe open"]
    classes = (("level",), ("climb_open",), ("descend_open",))
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.8))
    for ax, rows, title in ((axes[0], synth, "Synthetic worlds (40 test worlds)"),
                            (axes[1], real, "Austin LiDAR (59 blocks)")):
        series = []
        for key, label, colour in (("regret_k1", "best 1-channel map", ONE),
                                   ("regret_k2", "best 2-channel map", TWO)):
            vals = [ladder_summary(rows, c, key) for c in classes]
            series.append((label, colour, *zip(*vals)))
        bars(ax, groups, series, width=0.36)
        ax.set_title(title, color=INK, loc="left")
        style(ax, "extra cost (%)")
        ax.set_ylim(0, 45)
    top_legend(fig, axes, 2, 0.94)
    return save(fig, "channels_needed")


def fig_learned():
    synth, real = load("held"), austin_rows()
    groups = ["other\ninstructions", "keep it\nlevel", "climb or descend\nin the open"]
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.9))
    panels = ((axes[0], synth, TIER1, "Synthetic worlds (30 test worlds)"),
              (axes[1], real, ("low", "high", "shortest"), "Austin LiDAR (59 blocks, no retraining)"))
    for ax, data, tier1, title in panels:
        series = []
        for name, label, colour in (("k1_onehot", "1 channel", ONE), ("k2_onehot", "2 channels", TWO),
                                    ("k3_onehot", "3 channels", THREE)):
            vals = [summary(data[name], c) for c in (tier1, ("level",), TIER3)]
            series.append((label, colour, *zip(*vals)))
        bars(ax, groups, series)
        ax.set_title(title, color=INK, loc="left")
        style(ax, "extra cost (%)")
    top_legend(fig, axes, 3, 0.94)
    return save(fig, "learned_channels")


def fig_input():
    synth, real = load("held"), austin_rows()
    methods = (("k3_onehot", "class\ngiven"), ("k3_nli", "NLI"), ("k3_nli_adapter", "NLI +\nadapter"),
               ("k3_embedding", "sentence\nembedding"))
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    series = []
    for data, classes, label, colour in ((synth, CLASSES, "synthetic worlds", SYNTH),
                                         (real, REAL_CLASSES, "Austin LiDAR", REAL)):
        vals = [summary(data[m], classes) for m, _ in methods]
        series.append((label, colour, *zip(*vals)))
    bars(ax, [name for _, name in methods], series, width=0.38)
    style(ax, "extra cost, all instructions (%)")
    top_legend(fig, ax, 2, 0.9)
    return save(fig, "instruction_input")


def fig_probe():
    e3 = json.loads((RESULTS / "e3.json").read_text())
    tasks = (("all 8\nclasses", e3["all8"], 12.5), ("wide vs\ntight", e3["pairs"]["open/tight"], 50.0),
             ("low vs\nhigh", e3["pairs"]["low/high"], 50.0),
             ("climb vs\ndescend", e3["pairs"]["climb_open/descend_open"], 50.0),
             ("level vs\nshortest", e3["pairs"]["level/shortest"], 50.0))
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    series = [("sentence embedding", EMBED, [100 * t[1]["embedding"] for t in tasks], None, None),
              ("NLI", NLI, [100 * t[1]["nli"] for t in tasks], None, None)]
    bars(ax, [t[0] for t in tasks], series, width=0.38, fmt="{:.0f}", inside=True)
    for i, (_, _, chance) in enumerate(tasks):
        ax.plot([i - 0.42, i + 0.42], [chance, chance], color=INK, lw=1.0, ls=(0, (2, 2)), zorder=5)
    ax.plot([], [], color=INK, lw=1.0, ls=(0, (2, 2)), label="chance")
    style(ax, "held-out accuracy (%)")
    ax.set_ylim(0, 104)
    ax.set_yticks(range(0, 101, 20))
    top_legend(fig, ax, 3, 0.9)
    return save(fig, "sentence_probe")


def main():
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    for fn in (fig_paths, fig_channels, fig_learned, fig_input, fig_probe):
        print("wrote " + str(fn()))


if __name__ == "__main__":
    main()
