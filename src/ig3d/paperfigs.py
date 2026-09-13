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
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from .config import DEFAULT  # noqa: E402
from .costfield import gt_field_from_features, volume_features  # noqa: E402
from .experiments import TEST_SEED  # noqa: E402
from .instructions import ALL_LABELS, CLASSES, TIER  # noqa: E402
from .lidar import BORDER, REAL_CLASSES, SIZE, VOXEL_M, blocks_path, load_blocks  # noqa: E402
from .lidar import OUT as AUSTIN  # noqa: E402
from .metrics import behaviour  # noqa: E402
from .multiseed import load, matrix, run_dir, stats  # noqa: E402
from .paths import FIGURES, RESULTS  # noqa: E402
from .planner_grid import GridGraph  # noqa: E402
from .volumes import generate  # noqa: E402

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
ONE, TWO, THREE = "#eb6834", "#1baf7a", "#2a78d6"
NLI, EMBED = "#eda100", "#008300"
SYNTH, REAL = "#86b6ef", "#184f95"
ONE_LIGHT, THREE_LIGHT = "#f6b89c", "#a3c4ee"
WALL, PILLAR = "#b9b8ae", "#e6e5de"
TIER1 = ("open", "tight", "low", "high", "shortest")
TIER3 = ("climb_open", "descend_open")
EXAMPLE_WORLD = 9
NAMES = {"open": "wide openings", "tight": "tight gaps", "low": "stay low", "high": "stay high",
         "shortest": "shortest", "level": "keep it level", "climb_open": "climb in the open",
         "descend_open": "descend in the open"}
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


def bars(ax, groups, series, width=0.26, fmt="{:.1f}", inside=False, label_size=6.5):
    """series: (label, colour, means, lows, highs[, hatch]), values in percent; lows/highs may be None."""
    x = np.arange(len(groups))
    n = len(series)
    tops = [np.asarray(s[4] if s[4] is not None else s[2], float) for s in series]
    bottoms = [np.asarray(s[3] if s[3] is not None else s[2], float) for s in series]
    peak = float(np.max(tops))
    floor = min(0.0, float(np.min(bottoms)))
    for i, (label, colour, mean, lo, hi, *hatch) in enumerate(series):
        pos = x + (i - (n - 1) / 2) * width
        mean = np.asarray(mean, float)
        ax.bar(pos, mean, width * 0.88, color=colour, label=label, zorder=2,
               hatch=hatch[0] if hatch else None, edgecolor="white", lw=0)
        top = mean
        if lo is not None:
            lo, hi = np.asarray(lo, float), np.asarray(hi, float)
            ax.errorbar(pos, mean, yerr=[np.maximum(mean - lo, 0), np.maximum(hi - mean, 0)],
                        fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.8, zorder=3)
            top = np.maximum(hi, mean)
        for p, m, t in zip(pos, mean, top):
            if inside:
                ax.text(p, m - 0.03 * peak, fmt.format(m), ha="center", va="top", fontsize=label_size,
                        color=INK, zorder=6)
            else:
                ax.text(p, t + 0.03 * peak, fmt.format(m), ha="center", va="bottom", fontsize=label_size,
                        color=INK2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(floor * 1.4, peak * 1.2)
    if floor < 0:
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)


def summary(rows_by_seed, classes, key="grid_regret"):
    st = stats(matrix(rows_by_seed, key, list(classes)))
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
    fig, (ax_top, ax_side) = plt.subplots(2, 1, figsize=(3.5, 4.4),
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


def test_world(index=EXAMPLE_WORLD):
    v = generate(index + 1, TEST_SEED, DEFAULT.volume)[index]
    return v, volume_features(v, DEFAULT.cost)


def side_view(ax, v):
    Z, Y, _ = v.shape
    ax.add_patch(Rectangle((0, 0), Y, Z, facecolor="white", edgecolor=AXIS, lw=0.8, zorder=0))
    solid = v.occ[BORDER:-BORDER].all(axis=0)
    for y0, y1 in v.walls:
        solid[y0:y1 + 1] = False
    solid[:BORDER] = solid[-BORDER:] = False
    for y in np.flatnonzero(solid[:, BORDER:-BORDER].any(axis=1)):
        ax.add_patch(Rectangle((y, 0), 1, Z, color=PILLAR, lw=0, zorder=0.5))
    for y0, y1 in v.walls:
        ax.add_patch(Rectangle((y0, 0), y1 - y0 + 1, Z, color=WALL, lw=0, zorder=1))
    for w in v.windows:
        z0, z1, y0, y1, _, _ = w.box
        ax.add_patch(Rectangle((y0 - 0.4, z0), y1 - y0 + 1.8, z1 - z0 + 1, facecolor="white",
                               edgecolor=INK2, lw=0.7, zorder=1.5,
                               ls="-" if w.kind == "wide" else (0, (1.5, 1))))
    ax.set_xlim(0, Y)
    ax.set_ylim(0, Z)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_route(ax, path, colour, ls="-", lw=1.6, label=None):
    p = np.asarray(path, float)
    ax.plot(p[:, 1] + 0.5, p[:, 0] + 0.5, color=colour, lw=lw, ls=ls, label=label, zorder=3,
            solid_capstyle="round")


def endpoints(ax, v):
    ax.plot(v.start[1] + 0.5, v.start[0] + 0.5, "o", color=INK, ms=3.5, zorder=4)
    ax.plot(v.goal[1] + 0.5, v.goal[0] + 0.5, "*", color=INK, ms=6.5, zorder=4)


def side_mean(channel, free):
    n = free.sum(axis=2)
    with np.errstate(all="ignore"):
        return np.where(n > 0, (channel * free).sum(axis=2) / np.maximum(n, 1), np.nan)


def fig_overview(model="k3_onehot", cls="climb_open"):
    from .device import get_device
    from .instructions import ALL_SENTENCES
    from .planner_prm import plan as prm_plan
    from .train import load_model, predict

    v, f = test_world()
    dev = get_device()
    net, meta = load_model(run_dir(model, 0) / "best.pt", dev)
    held = np.asarray(meta["held_sent"])
    si = int(held[ALL_LABELS[held] == CLASSES.index(cls)][0])
    field = predict(net, v.occ, np.asarray(meta["cond_table"], np.float32)[si], DEFAULT.cost, dev)
    path, _ = prm_plan(v.occ, field, v.start, v.goal, DEFAULT.prm, np.random.default_rng(0))
    short, _ = GridGraph(v.occ).plan(gt_field_from_features(v.occ, f, "shortest", DEFAULT.cost),
                                     v.start, v.goal)
    free = v.occ == 0
    W, H, s, y0 = 7.16, 1.36, 1.02, 0.05
    fig = plt.figure(figsize=(W, H))

    def panel(x, title):
        ax = fig.add_axes((x / W, y0 / H, s / W, s / H))
        ax.set_title(title, color=INK, fontsize=7.5, pad=3)
        return ax

    ax = panel(0.08, "occupancy (side view)")
    side_view(ax, v)
    endpoints(ax, v)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(WALL)
    names = ("travel cost $c$", "climb cost $a_\\mathrm{up}$", "descent cost $a_\\mathrm{dn}$")
    for i, name in enumerate(names):
        ax = panel(2.02 + i * (s + 0.08), name)
        im = ax.imshow(side_mean(field[i], free), origin="lower", cmap=cmap, vmin=0, vmax=1,
                       extent=(0, v.shape[1], 0, v.shape[0]), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(AXIS)
    cax = fig.add_axes(((2.02 + 3 * (s + 0.08) - 0.02) / W, y0 / H, 0.06 / W, s / H))
    cb = fig.colorbar(im, cax=cax, ticks=(0, 0.5, 1))
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, length=0, labelsize=6.5)
    ax = panel(6.1, "planned path")
    side_view(ax, v)
    draw_route(ax, short, INK2, ls=(0, (2.5, 1.5)), lw=1.1)
    draw_route(ax, path, THREE)
    endpoints(ax, v)
    words = ALL_SENTENCES[si].rstrip(".").split()
    wrapped = "“" + " ".join(words[:3]) + "\n" + " ".join(words[3:6]) + "\n" + " ".join(words[6:]) + ".”"
    fig.text(1.56 / W, (y0 + s) / H, wrapped, ha="center", va="top", fontsize=6.5, color=INK,
             style="italic", linespacing=1.25)
    for x0, x1, label in ((1.14, 1.98, "3D U-Net"), (5.62, 6.06, "PRM")):
        yc = (y0 + 0.3) / H
        fig.add_artist(FancyArrowPatch((x0 / W, yc), (x1 / W, yc), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=8, color=INK, lw=1.0))
        fig.text(0.5 * (x0 + x1) / W, yc + 0.06 / H, label, ha="center", va="bottom",
                 fontsize=7, color=INK, weight="bold")
    beh = behaviour(v, path, f["edt"]), behaviour(v, short, f["edt"])
    print("overview: sentence '{}', climb clearance planned {:.1f} vs shortest {:.1f}".format(
        ALL_SENTENCES[si], beh[0]["climb_clear"], beh[1]["climb_clear"]))
    return save(fig, "method_overview")


def fig_examples():
    v, f = test_world()
    G = GridGraph(v.occ)
    plans = {c: G.plan(gt_field_from_features(v.occ, f, c, DEFAULT.cost), v.start, v.goal)[0]
             for c in CLASSES}
    beh = {c: behaviour(v, p, f["edt"]) for c, p in plans.items()}
    Z = v.shape[0] - 1

    def narrow(c):
        return "{:.0f} of {:.0f} windows narrow".format(beh[c]["n_narrow"], beh[c]["n_windows"])

    def alt(c):
        return "mean altitude {:.0f}".format(beh[c]["mean_alt"] * Z)

    def vert(c):
        return "{:.0f} voxels up and down".format(beh[c]["vertical"])

    def clear(c):
        return "clearance while climbing {:.1f}".format(beh[c]["climb_clear"])

    panels = (("(a)", "open", "tight", narrow), ("(b)", "low", "high", alt),
              ("(c)", "level", "shortest", vert), ("(d)", "climb_open", "descend_open", clear))
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.12))
    for ax, (tag, a, b, fmt) in zip(axes, panels):
        side_view(ax, v)
        draw_route(ax, plans[b], ONE, ls=(0, (2.5, 1.2)), label="{}: {}".format(NAMES[b], fmt(b)))
        draw_route(ax, plans[a], THREE, label="{}: {}".format(NAMES[a], fmt(a)))
        endpoints(ax, v)
        ax.set_title("{} {}\nvs {}".format(tag, NAMES[a], NAMES[b]), color=INK, fontsize=7.5)
        h, lab = ax.get_legend_handles_labels()
        ax.legend(h[::-1], [t.replace(": ", ":\n") for t in lab[::-1]], loc="upper center",
                  bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=6.3, labelcolor=INK,
                  handlelength=2.2, borderaxespad=0)
    fig.tight_layout(rect=(0, 0, 1, 1), w_pad=0.6)
    fig.subplots_adjust(bottom=0.28, top=0.84)
    return save(fig, "example_paths")


def fig_per_class():
    held = json.loads((RESULTS / "multiseed" / "report.json").read_text())["held"]
    methods = (("k3_onehot", "class given", THREE), ("k3_nli", "NLI", NLI),
               ("k3_embedding", "sentence embedding", EMBED))
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    y = np.arange(len(CLASSES))
    h = 0.27
    for j, (m, label, colour) in enumerate(methods):
        g = held[m]["grid_regret"]
        mean = np.array([100 * g[c]["mean"] for c in CLASSES])
        lo = np.array([100 * g[c]["lo"] for c in CLASSES])
        hi = np.array([100 * g[c]["hi"] for c in CLASSES])
        pos = y + (j - 1) * h
        ax.barh(pos, mean, h * 0.88, color=colour, label=label, zorder=2)
        ax.errorbar(mean, pos, xerr=[np.maximum(mean - lo, 0), np.maximum(hi - mean, 0)],
                    fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5, zorder=3)
        for p, mv, hv in zip(pos, mean, hi):
            ax.text(max(hv, mv) + 0.8, p, "{:.1f}".format(mv), va="center", fontsize=6, color=INK2)
    for t in (4.5, 5.5):
        ax.axhline(t, color=AXIS, lw=0.8, ls=(0, (2, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels(["{} (tier {})".format(NAMES[c], TIER[c]) for c in CLASSES])
    ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("extra cost on new wording (%)", color=INK2)
    ax.set_xlim(0, 47)
    top_legend(fig, ax, 3, 0.93)
    return save(fig, "per_instruction")


def fig_planners():
    synth = load("held")
    groups = ["other\ninstructions", "keep it\nlevel", "climb or descend\nin the open"]
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    series = []
    for name, ch, dark, light in (("k1_onehot", "1 channel", ONE, ONE_LIGHT),
                                  ("k3_onehot", "3 channels", THREE, THREE_LIGHT)):
        for key, planner, colour, hatch in (("grid_regret", "exact", dark, None),
                                            ("prm_regret", "PRM", light, "////")):
            vals = [summary(synth[name], c, key) for c in (TIER1, ("level",), TIER3)]
            series.append(("{}, {}".format(ch, planner), colour, *zip(*vals), hatch))
    bars(ax, groups, series, width=0.21, label_size=5.6)
    style(ax, "extra cost (%)")
    top_legend(fig, ax, 2, 0.84)
    return save(fig, "planner_comparison")


def main():
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    for fn in (fig_overview, fig_examples, fig_per_class, fig_planners,
               fig_paths, fig_channels, fig_learned, fig_input, fig_probe):
        print("wrote " + str(fn()))


if __name__ == "__main__":
    main()
