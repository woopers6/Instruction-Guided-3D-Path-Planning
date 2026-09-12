"""Figures: side and top views of a volume with paths overlaid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .config import DEFAULT  # noqa: E402
from .costfield import gt_field_from_features, volume_features  # noqa: E402
from .instructions import CLASSES, TIER  # noqa: E402
from .metrics import PRIMARY, behaviour  # noqa: E402
from .planner_grid import GridGraph  # noqa: E402
from .volumes import generate  # noqa: E402

from .paths import FIGURES, RESULTS  # noqa: E402
WIN_COLOUR = {"narrow": "tab:red", "wide": "tab:blue"}
PAIR_FIGS = (("open", "tight"), ("low", "high"), ("level", "shortest"),
             ("climb_open", "descend_open"))
PATH_COLOURS = ("tab:green", "tab:purple", "tab:orange", "tab:brown")


def draw_volume(ax_side, ax_top, vol):
    Z, Y, X = vol.shape
    for ax, span in ((ax_side, Z), (ax_top, X)):
        ax.add_patch(Rectangle((-0.5, -0.5), Y, span, facecolor="white", edgecolor="0.3"))
        for (y0, y1) in vol.walls:
            ax.add_patch(Rectangle((y0 - 0.5, -0.5), y1 - y0 + 1, span, color="0.65", lw=0))
    for w in vol.windows:
        z0, z1, y0, y1, x0, x1 = w.box
        for ax, lo, hi in ((ax_side, z0, z1), (ax_top, x0, x1)):
            ax.add_patch(Rectangle((y0 - 0.5, lo - 0.5), y1 - y0 + 1, hi - lo + 1,
                                   facecolor="white", edgecolor=WIN_COLOUR[w.kind], lw=1.6))
    b = 1
    pillars = vol.occ[b:-b].all(axis=0).astype(float)
    for (y0, y1) in vol.walls:
        pillars[y0:y1 + 1] = 0
    pillars[:b] = pillars[-b:] = 0
    pillars[:, :b] = pillars[:, -b:] = 0
    ys, xs = np.nonzero(pillars)
    ax_top.scatter(ys, xs, s=14, marker="s", color="0.25", lw=0)
    ax_side.set_xlim(-1, Y)
    ax_side.set_ylim(-1, Z)
    ax_top.set_xlim(-1, Y)
    ax_top.set_ylim(-1, X)
    ax_side.set_ylabel("z  (up)")
    ax_top.set_ylabel("x  (across)")
    ax_top.set_xlabel("y  (start -> goal)")
    for ax in (ax_side, ax_top):
        ax.set_aspect("equal")


def draw_path(ax_side, ax_top, path, colour, label):
    p = np.asarray(path, float)
    ax_side.plot(p[:, 1], p[:, 0], color=colour, lw=2.0, label=label)
    ax_top.plot(p[:, 1], p[:, 2], color=colour, lw=2.0)


def figure(vol, paths: dict, out, title=""):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 9.0), sharex=True)
    draw_volume(a1, a2, vol)
    for (label, p), c in zip(paths.items(), PATH_COLOURS):
        draw_path(a1, a2, p, c, label)
    for ax, i in ((a1, 0), (a2, 2)):
        ax.plot(vol.start[1], vol.start[i], "ko", ms=6)
        ax.plot(vol.goal[1], vol.goal[i], "k*", ms=11)
    fig.suptitle(title, fontsize=9)
    a1.legend(loc="lower center", fontsize=8, ncol=1, bbox_to_anchor=(0.5, 1.01),
              frameon=False)
    a1.text(0.99, 0.01, "red = narrow window, blue = wide", transform=a1.transAxes,
            ha="right", va="bottom", fontsize=7, color="0.3")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def ladder(out_dir=None, n_search: int = 30, seed: int = 3000):
    """One figure per contrast pair, from the exact planner on the TRUE fields."""
    out_dir = Path(out_dir or FIGURES)
    vols = generate(n_search, seed, DEFAULT.volume)
    written = []
    for (a, b) in PAIR_FIGS:
        m = PRIMARY[a][0] if a != "level" else "vertical"
        if a == "climb_open":
            m = "climb_clear"
        for vi, v in enumerate(vols):
            f = volume_features(v, DEFAULT.cost)
            G = GridGraph(v.occ)
            pa, _ = G.plan(gt_field_from_features(v.occ, f, a, DEFAULT.cost), v.start, v.goal)
            pb, _ = G.plan(gt_field_from_features(v.occ, f, b, DEFAULT.cost), v.start, v.goal)
            ba, bb = behaviour(v, pa, f["edt"]), behaviour(v, pb, f["edt"])
            if np.isfinite(ba[m]) and np.isfinite(bb[m]) and abs(ba[m] - bb[m]) > 1e-6:
                title = ("oracle paths, exact planner, test volume {} (first of {} where "
                         "they differ)\n{}: {} {:.2f} vs {:.2f}").format(
                    vi, n_search, m, a + " / " + b, ba[m], bb[m])
                written.append(figure(v, {a: pa, b: pb}, out_dir / "oracle_{}_vs_{}.png".format(a, b),
                                      title))
                break
    return written


def demo(ckpt, text, out=None, seed: int = 3000, index: int = 0):
    """Free-text instruction -> predicted field -> PRM path, next to the shortest path."""
    from .device import get_device
    from .encoder import EmbeddingEncoder, NLIEncoder
    from .planner_prm import plan as prm_plan
    from .train import load_model, predict

    dev = get_device()
    model, meta = load_model(ckpt, dev)
    if meta["conditioning"] == "nli":
        cond = NLIEncoder(DEFAULT.model.nli_model)([text])[0]
    elif meta["conditioning"] == "embedding":
        enc = EmbeddingEncoder(DEFAULT.model.encoder_model, np.asarray(meta["train_sent"]),
                               meta["emb_dim"])
        cond = enc([text])[0]
    else:
        raise SystemExit("a one-hot model cannot read free text; use an nli or embedding checkpoint")
    v = generate(index + 1, seed, DEFAULT.volume)[index]
    from .costfield import CostConfig
    ccfg = CostConfig(**meta["config"]["cost"])
    field = predict(model, v.occ, cond, ccfg, dev)
    path, _ = prm_plan(v.occ, field, v.start, v.goal, DEFAULT.prm, np.random.default_rng(0))
    f = volume_features(v, ccfg)
    short, _ = GridGraph(v.occ).plan(gt_field_from_features(v.occ, f, "shortest", ccfg),
                                     v.start, v.goal)
    out = out or FIGURES / "demo.png"
    return figure(v, {'"{}"'.format(text): path, "shortest path": short}, out,
                  "{} model, PRM on the predicted field".format(meta["conditioning"]))


MODEL_STYLE = {
    "k3_onehot": ("K=3, one-hot (language given)", "#2a78d6", "o"),
    "k1_onehot": ("K=1 scalar, one-hot", "#eb6834", "s"),
    "k2_onehot": ("K=2, one-hot", "#1baf7a", "D"),
    "k3_nli": ("K=3, NLI", "#eda100", "^"),
    "k3_nli_adapter": ("K=3, NLI + adapter", "#e87ba4", "v"),
    "k3_embedding": ("K=3, frozen embedding (IG-PRM)", "#008300", "P"),
}
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def errorbars(out=None, key="grid_regret"):
    """Regret per class, mean of 3 seeds (marker) with 95% CI over test volumes (bar)."""
    rep = json.loads((RESULTS / "multiseed" / "report.json").read_text())
    held = rep["held"]
    panels = (("(a) expressiveness, language given", ("k1_onehot", "k2_onehot", "k3_onehot")),
              ("(b) conditioning, new wording", ("k3_onehot", "k3_nli", "k3_nli_adapter",
                                                 "k3_embedding")))
    rows = list(CLASSES) + ["MEAN"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.2), sharey=True)
    for ax, (title, models) in zip(axes, panels):
        models = [m for m in models if m in held]
        for j, m in enumerate(models):
            lab, col, mk = MODEL_STYLE[m]
            for i, cls in enumerate(rows):
                st = held[m][key][cls]
                y = i + (j - (len(models) - 1) / 2) * 0.19
                ax.plot([100 * st["lo"], 100 * st["hi"]], [y, y], color=col, lw=1.3,
                        solid_capstyle="round", zorder=2)
                ax.plot(100 * st["mean"], y, marker=mk, color=col, ms=4.8, mec="white",
                        mew=0.7, ls="", zorder=3, label=lab if i == 0 else None)
        ax.axhline(len(CLASSES) - 0.5, color=AXIS, lw=0.8)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xlim(left=0)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=7.5, length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.set_xlabel("regret vs optimal path, %", fontsize=7.5, color=INK2)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(["{}  (T{})".format(c.replace("_", " "), TIER[c]) for c in CLASSES]
                            + ["mean of 8"])
    axes[0].invert_yaxis()
    handles = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3, frameon=False,
               fontsize=7, labelcolor=INK, handletextpad=0.3, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.035))
    ref = held["k3_onehot"][key]["MEAN"]
    fig.text(0.01, 0.005, "marker: mean of {} training seed{}    bar: 95% bootstrap CI over "
             "{} unseen test volumes    exact grid planner on the predicted field".format(
                 ref["n_seeds"], "s" if ref["n_seeds"] > 1 else "", ref["n_vols"]),
             fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    out = Path(out or FIGURES / "multiseed_regret.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["ladder", "demo", "errorbars"])
    ap.add_argument("--ckpt")
    ap.add_argument("--text")
    ap.add_argument("--out")
    ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args(argv)
    if a.what == "ladder":
        for p in ladder(a.out):
            print("wrote " + str(p))
    elif a.what == "errorbars":
        print("wrote " + str(errorbars(a.out)))
    else:
        if not (a.ckpt and a.text):
            ap.error("demo needs --ckpt and --text")
        print("wrote " + str(demo(a.ckpt, a.text, a.out, index=a.index)))


if __name__ == "__main__":
    main()
