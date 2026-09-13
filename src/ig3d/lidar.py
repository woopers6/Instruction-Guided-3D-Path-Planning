"""Real environments: USGS 3DEP aerial LiDAR of downtown Austin, as 48^3 voxel blocks."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage

from .config import DEFAULT, CostConfig
from .costfield import gt_field_from_features, project_to_k, volume_features
from .instructions import ALL_LABELS, CLASSES
from .metrics import behaviour, regret
from .paths import CACHE, RESULTS
from .planner_grid import GridGraph, path_cost
from .volumes import Volume, connected

TILE = CACHE / "lidar" / "USGS_LPC_TX_Central_B1_2017_stratmap17_50cm_3097433c1_LAS_2019.laz"
KEEP = (2, 3, 4, 5, 6, 9, 10, 17)
REAL_CLASSES = ("low", "high", "shortest", "level", "climb_open", "descend_open")
VOXEL_M = 4.0
SIZE = 48
BORDER = 1
MIN_COVERAGE = 0.95
OUT = RESULTS / "austin"
BEHAVIOUR_KEYS = ("mean_alt", "vertical", "climb_clear", "descend_clear", "length")


def surface(path=TILE, voxel_m=VOXEL_M):
    """(top, covered, meta): highest kept return per column; empty columns take the nearest value."""
    import laspy

    pts = laspy.read(path)
    cls = np.asarray(pts.classification)
    keep = np.isin(cls, KEEP)
    x = np.asarray(pts.x)[keep]
    y = np.asarray(pts.y)[keep]
    z = np.asarray(pts.z)[keep].astype(np.float32)
    x0, y0 = float(x.min()), float(y.min())
    ix = ((x - x0) // voxel_m).astype(np.int64)
    iy = ((y - y0) // voxel_m).astype(np.int64)
    ny, nx = int(iy.max()) + 1, int(ix.max()) + 1
    top = np.full(ny * nx, -np.inf, np.float32)
    np.maximum.at(top, iy * nx + ix, z)
    top = top.reshape(ny, nx)
    covered = np.isfinite(top)
    _, (fy, fx) = ndimage.distance_transform_edt(~covered, return_indices=True)
    top = top[fy, fx]
    dropped = {int(c): int(n) for c, n in zip(*np.unique(cls[~keep], return_counts=True))}
    meta = dict(n_points=int(len(cls)), n_kept=int(keep.sum()), dropped_by_class=dropped,
                x0=x0, y0=y0, voxel_m=voxel_m, grid=[ny, nx], kept_classes=list(KEEP))
    return top, covered, meta


def block_occupancy(top_block, base, voxel_m=VOXEL_M, size=SIZE, border=BORDER):
    """(occ, h): solid from the floor up to each column's surface, plus a closed shell."""
    h = np.clip(np.ceil((top_block - base) / voxel_m), 1, size).astype(np.int64)
    occ = (np.arange(size)[:, None, None] < h[None]).astype(np.uint8)
    occ[:border] = occ[-border:] = 1
    occ[:, :border] = occ[:, -border:] = 1
    occ[:, :, :border] = occ[:, :, -border:] = 1
    return occ, h


def place_endpoints(occ, h, rng, tries=50, border=BORDER):
    """Start and goal at opposite y ends, same height, just above the taller surface; or None."""
    size = occ.shape[0]
    ys, yg = border + 2, size - border - 3
    for _ in range(tries):
        xs, xg = (int(v) for v in rng.integers(border + 2, size - border - 2, size=2))
        zs = int(max(h[ys, xs], h[yg, xg], border))
        if zs > size - border - 3:
            continue
        start, goal = (zs, ys, xs), (zs, yg, xg)
        if occ[start] == 0 and occ[goal] == 0 and connected(occ == 0, start, goal):
            return start, goal
    return None


def blocks_path(voxel_m=VOXEL_M):
    return CACHE / "lidar" / "austin_blocks_s{:g}.npz".format(voxel_m)


def build(path=TILE, voxel_m=VOXEL_M, seed=0):
    top, covered, meta = surface(path, voxel_m)
    ny, nx = top.shape
    vols, info = [], []
    for bi in range(ny // SIZE):
        for bj in range(nx // SIZE):
            sl = np.s_[bi * SIZE:(bi + 1) * SIZE, bj * SIZE:(bj + 1) * SIZE]
            rec = dict(block=[bi, bj], coverage=float(covered[sl].mean()))
            if rec["coverage"] < MIN_COVERAGE:
                info.append(dict(rec, used=False, reason="coverage"))
                continue
            t = top[sl]
            base = float(t.min()) - voxel_m
            occ, h = block_occupancy(t, base, voxel_m)
            ends = place_endpoints(occ, h, np.random.default_rng((seed, bi, bj)))
            if ends is None:
                info.append(dict(rec, used=False, reason="no valid start/goal"))
                continue
            vols.append(Volume(occ=occ, start=ends[0], goal=ends[1]))
            info.append(dict(rec, used=True, index=len(vols) - 1, base_m=base,
                             relief_m=float(t.max() - t.min()),
                             solid_frac=float(occ[1:-1, 1:-1, 1:-1].mean()), start_z=ends[0][0]))
    out = blocks_path(voxel_m)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, occ=np.stack([v.occ for v in vols]),
                        start=np.asarray([v.start for v in vols]),
                        goal=np.asarray([v.goal for v in vols]))
    out.with_suffix(".json").write_text(json.dumps(dict(meta=meta, blocks=info), indent=1))
    return vols, meta, info


def load_blocks(voxel_m=VOXEL_M):
    z = np.load(blocks_path(voxel_m))
    return [Volume(occ=o, start=tuple(int(a) for a in s), goal=tuple(int(a) for a in g))
            for o, s, g in zip(z["occ"], z["start"], z["goal"])]


def ladder(vols, cost_cfg: CostConfig = DEFAULT.cost):
    """Best K=1 and K=2 projections vs the true K=3 field, scored by regret. No network."""
    rows = []
    for vi, v in enumerate(vols):
        G = GridGraph(v.occ)
        f = volume_features(v, cost_cfg)
        p_short, _ = G.plan(gt_field_from_features(v.occ, f, "shortest", cost_cfg, 3), v.start, v.goal)
        b_short = behaviour(v, p_short, f["edt"])
        for cls in REAL_CLASSES:
            f3 = gt_field_from_features(v.occ, f, cls, cost_cfg, 3)
            p3, c3 = G.plan(f3, v.start, v.goal)
            b3 = behaviour(v, p3, f["edt"])
            rec = dict(vol=vi, cls=cls)
            for key in BEHAVIOUR_KEYS:
                rec["k3_" + key], rec["shortest_" + key] = b3[key], b_short[key]
            for k in (1, 2):
                pk, _ = G.plan(project_to_k(f3, k), v.start, v.goal)
                rec["regret_k%d" % k] = regret(path_cost(f3, pk), c3)
                bk = behaviour(v, pk, f["edt"])
                for key in ("vertical", "climb_clear"):
                    rec["k%d_%s" % (k, key)] = bk[key]
            rows.append(rec)
    return rows


def learned(ckpt, vols, sentences="held", seed=0):
    """Plan on a trained model's predicted field in every block; score under the true field."""
    from .device import get_device
    from .train import load_model, predict

    dev = get_device()
    model, meta = load_model(ckpt, dev)
    ccfg = CostConfig(**meta["config"]["cost"])
    table = np.asarray(meta["cond_table"], np.float32)
    pool = np.asarray(meta["held_sent"] if sentences == "held" else meta["train_sent"])
    rng = np.random.default_rng(seed)
    rows = []
    for vi, v in enumerate(vols):
        G = GridGraph(v.occ)
        f = volume_features(v, ccfg)
        for cls in REAL_CLASSES:
            si = int(rng.choice(pool[ALL_LABELS[pool] == CLASSES.index(cls)]))
            truth = gt_field_from_features(v.occ, f, cls, ccfg, 3)
            p_opt, c_opt = G.plan(truth, v.start, v.goal)
            pg, _ = G.plan(predict(model, v.occ, table[si], ccfg, dev), v.start, v.goal)
            bp, bo = behaviour(v, pg, f["edt"]), behaviour(v, p_opt, f["edt"])
            rec = dict(vol=vi, cls=cls, sent=si, grid_regret=regret(path_cost(truth, pg), c_opt))
            for key in BEHAVIOUR_KEYS:
                rec["pred_" + key], rec["oracle_" + key] = bp[key], bo[key]
            rows.append(rec)
    return rows


def _result_path(name, seed, sentences):
    return OUT / "e1_{}_{}_s{}.json".format(sentences, name, seed)


def eval_one(name, seed, sentences):
    from .multiseed import run_dir
    rows = learned(run_dir(name, seed) / "best.pt", load_blocks(), sentences, seed)
    OUT.mkdir(parents=True, exist_ok=True)
    _result_path(name, seed, sentences).write_text(
        json.dumps(dict(name=name, seed=seed, sentences=sentences, rows=rows), default=float))


def evaluate(names, seeds, workers, sentence_sets=("held",)):
    from .multiseed import _run, _stamp, run_dir
    jobs = [(n, s, t) for s in seeds for n in names for t in sentence_sets
            if (run_dir(n, s) / "last.pt").exists() and not _result_path(n, s, t).exists()]
    print("{} evaluating {} (model, seed) cells on Austin blocks with {} workers".format(
        _stamp(), len(jobs), workers), flush=True)

    def one(job):
        n, s, t = job
        cmd = [sys.executable, "-m", "ig3d.lidar", "eval-one", "--name", n, "--seed", str(s),
               "--sentences", t]
        return _run(cmd, OUT / "logs" / "{}_{}_s{}.log".format(t, n, s),
                    "austin {} seed {} {}".format(n, s, t))

    with ThreadPoolExecutor(workers) as ex:
        rcs = list(ex.map(one, jobs))
    print("{} AUSTIN_EVAL_DONE failures={}".format(_stamp(), sum(r != 0 for r in rcs)), flush=True)


def _informative(data):
    """Blocks whose true shortest path climbs at least once: the only ones where level/climb/descend can differ."""
    rows = next(iter(next(iter(data.values())).values()))
    return {r["vol"] for r in rows if r["cls"] == "shortest" and r["oracle_vertical"] > 0}


def _subset(data, keep):
    return {n: {s: [r for r in rows if r["vol"] in keep] for s, rows in by.items()}
            for n, by in data.items()}


def _report_lines(data, ms, title, sentences):
    from .multiseed import comparison, matrix, stats
    lines = ["## Austin LiDAR, {} phrasings, {} — grid regret %, mean ± sd over seeds "
             "[95% CI over blocks]".format(sentences, title), "",
             "| class | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
    for cls in list(REAL_CLASSES) + ["MEAN"]:
        cl = list(REAL_CLASSES) if cls == "MEAN" else [cls]
        cells = []
        for n in ms:
            st = stats(matrix(data[n], "grid_regret", cl))
            cells.append("{:5.1f} ± {:3.1f} [{:4.1f}, {:4.1f}]".format(
                100 * st["mean"], 100 * st["sd"], 100 * st["lo"], 100 * st["hi"]))
        lines.append("| {} | {} |".format("**mean**" if cls == "MEAN" else cls, " | ".join(cells)))
    claims = [("K=1 vs K=3 on level", "k1_onehot", "k3_onehot", ["level"]),
              ("K=1 vs K=3 on tier 3", "k1_onehot", "k3_onehot", ["climb_open", "descend_open"]),
              ("K=2 vs K=3 on tier 3", "k2_onehot", "k3_onehot", ["climb_open", "descend_open"]),
              ("NLI − one-hot (language gap)", "k3_nli", "k3_onehot", list(REAL_CLASSES)),
              ("embedding − NLI", "k3_embedding", "k3_nli", list(REAL_CLASSES)),
              ("adapter − NLI", "k3_nli_adapter", "k3_nli", list(REAL_CLASSES))]
    lines += ["", "| comparison (A − B, points) | A − B | holds in |", "|---|---|---|"]
    for name, a, b, cl in claims:
        if a in data and b in data:
            st = comparison(data, a, b, "grid_regret", cl)
            lines.append("| {} | {:+.1f} ± {:.1f} [{:+.1f}, {:+.1f}] | {}/{} seeds |".format(
                name, 100 * st["mean"], 100 * st["sd"], 100 * st["lo"], 100 * st["hi"],
                st["sign_holds"], st["n_seeds"]))
    return lines


def report(sentences="held"):
    from .multiseed import MODELS
    data = {}
    for f in sorted(OUT.glob("e1_{}_*_s*.json".format(sentences))):
        d = json.loads(f.read_text())
        data.setdefault(d["name"], {})[d["seed"]] = d["rows"]
    if not data:
        raise SystemExit("no Austin results for {} phrasings yet".format(sentences))
    ms = [n for n in MODELS if n in data]
    blocks = {r["vol"] for r in next(iter(data[ms[0]].values()))}
    keep = _informative(data)
    lines = _report_lines(data, ms, "all {} blocks".format(len(blocks)), sentences)
    if len(keep) >= 2:
        lines += [""] + _report_lines(
            _subset(data, keep), ms,
            "the {} blocks where the shortest path climbs (declared secondary analysis)".format(len(keep)),
            sentences)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _print_ladder(rows):
    keep = {r["vol"] for r in rows if r["shortest_vertical"] > 0}
    n = len({r["vol"] for r in rows})
    scopes = (("all {} blocks".format(n), rows),
              ("{} blocks where the shortest path climbs".format(len(keep)),
               [r for r in rows if r["vol"] in keep]))
    for title, scope in scopes:
        print(title)
        print("{:13s} {:>9s} {:>9s} {:>10s} {:>10s} {:>12s} {:>12s}".format(
            "class", "regretK1", "regretK2", "vert K3", "vert short", "climbclr K3", "climbclr K2"))
        for cls in REAL_CLASSES:
            rs = [r for r in scope if r["cls"] == cls]
            if not rs:
                continue
            print("{:13s} {:9.4f} {:9.4f} {:10.2f} {:10.2f} {:12.2f} {:12.2f}".format(
                cls, np.mean([r["regret_k1"] for r in rs]), np.mean([r["regret_k2"] for r in rs]),
                np.nanmean([r["k3_vertical"] for r in rs]),
                np.nanmean([r["shortest_vertical"] for r in rs]),
                np.nanmean([r["k3_climb_clear"] for r in rs]),
                np.nanmean([r["k2_climb_clear"] for r in rs])))


def figures():
    """Paper figures from the Austin data: tile map, example paths, no-network ladder, trained models."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    from .multiseed import matrix, stats
    from .paths import FIGURES
    from .viz import AXIS, INK, INK2, dot_whisker

    written = []
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"

    def save(fig, name):
        out = FIGURES / (name + ".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
        written.append(out)

    def tidy(ax, xlabel, ylabel):
        ax.tick_params(colors=INK2, labelsize=7, length=0)
        for s in ax.spines.values():
            s.set_color(AXIS)
        ax.set_xlabel(xlabel, fontsize=7.5, color=INK2)
        ax.set_ylabel(ylabel, fontsize=7.5, color=INK2)

    km = VOXEL_M / 1000.0
    top, covered, _ = surface()
    blocks = json.loads(blocks_path().with_suffix(".json").read_text())["blocks"]
    used = [b for b in blocks if b["used"]]
    rel = (top - np.percentile(top[covered], 1)).astype(float)
    rel[~covered] = np.nan
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#f0efec")
    fig, ax = plt.subplots(figsize=(3.5, 4.2))
    im = ax.imshow(rel, origin="lower", cmap=cmap, vmin=0, vmax=float(np.nanpercentile(rel, 99.9)),
                   extent=(0, top.shape[1] * km, 0, top.shape[0] * km), interpolation="nearest")
    side = SIZE * km
    for b in blocks:
        bi, bj = b["block"]
        ax.add_patch(Rectangle((bj * side, bi * side), side, side, fill=False,
                               lw=0.6 if b["used"] else 1.0, edgecolor=INK2 if b["used"] else INK,
                               ls="-" if b["used"] else (0, (2, 2))))
    tidy(ax, "east (km)", "north (km)")
    ax.set_title("Downtown Austin\nUSGS 3DEP aerial LiDAR", fontsize=8.5, color=INK, loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
    cb.set_label("surface height above lowest ground (m)", fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=6.5, colors=INK2, length=0)
    cb.outline.set_visible(False)
    ax.legend(handles=[Line2D([], [], color=INK2, lw=0.6,
                              label="{} blocks used, 192 m each".format(len(used))),
                       Line2D([], [], color=INK, lw=1.0, ls=(0, (2, 2)),
                              label="{} skipped, under 95% covered".format(len(blocks) - len(used)))],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False, fontsize=6.5,
              labelcolor=INK)
    fig.tight_layout()
    save(fig, "austin_tile")

    vols = load_blocks()
    cfg = DEFAULT.cost
    pairs = (("level", "shortest", "vertical"), ("low", "high", "mean_alt"))
    chosen = None
    for b in used:
        v = vols[b["index"]]
        G = GridGraph(v.occ)
        f = volume_features(v, cfg)
        plans = {c: G.plan(gt_field_from_features(v.occ, f, c, cfg, 3), v.start, v.goal)[0]
                 for c in ("level", "shortest", "low", "high")}
        beh = {c: behaviour(v, p, f["edt"]) for c, p in plans.items()}
        if all(abs(beh[x][m] - beh[y][m]) > 1e-6 for x, y, m in pairs):
            chosen = (b, v, plans, beh)
            break
    if chosen is None:
        raise SystemExit("no block where both contrasts differ")
    b, v, plans, beh = chosen
    surf = v.occ[:-1].sum(axis=0)
    names = {"level": '"keep it level"', "shortest": '"shortest path"', "low": '"stay low"',
             "high": '"stay high"'}
    lo_m, hi_m = BORDER * VOXEL_M, (SIZE - BORDER) * VOXEL_M
    def draw_pair(ax_top, ax_prof, x_cls, y_cls, m):
        ax_top.imshow(surf[BORDER:-BORDER, BORDER:-BORDER] * VOXEL_M, origin="lower", cmap="Greys",
                      vmin=0, vmax=float(surf[BORDER:-BORDER, BORDER:-BORDER].max() * VOXEL_M),
                      extent=(lo_m, hi_m, lo_m, hi_m), interpolation="nearest")
        for cls, colour, ls in ((x_cls, blue, "-"), (y_cls, orange, (0, (3, 1.5)))):
            p = np.asarray(plans[cls], float)
            ax_top.plot((p[:, 2] + 0.5) * VOXEL_M, (p[:, 1] + 0.5) * VOXEL_M, color=colour, lw=1.8,
                        ls=ls, label=names[cls])
            step = np.hypot(np.diff(p[:, 1]), np.diff(p[:, 2])) * VOXEL_M
            dist = np.concatenate([[0.0], np.cumsum(step)])
            under = surf[p[:, 1].astype(int), p[:, 2].astype(int)] * VOXEL_M
            ax_prof.fill_between(dist, 0, under, color=colour, alpha=0.15, lw=0, step="mid")
            ax_prof.plot(dist, (p[:, 0] + 0.5) * VOXEL_M, color=colour, lw=1.8, ls=ls)
        ax_top.plot((v.start[2] + 0.5) * VOXEL_M, (v.start[1] + 0.5) * VOXEL_M, "o", color=INK, ms=4)
        ax_top.plot((v.goal[2] + 0.5) * VOXEL_M, (v.goal[1] + 0.5) * VOXEL_M, "*", color=INK, ms=8)
        if m == "vertical":
            detail = "vertical travel {:.0f} m vs {:.0f} m".format(
                beh[x_cls][m] * VOXEL_M, beh[y_cls][m] * VOXEL_M)
        else:
            detail = "mean altitude {:.0f} m vs {:.0f} m".format(
                (beh[x_cls][m] * (SIZE - 1) + 0.5) * VOXEL_M,
                (beh[y_cls][m] * (SIZE - 1) + 0.5) * VOXEL_M)
        ax_top.set_title("{} vs {}\n{}".format(names[x_cls], names[y_cls], detail), fontsize=8,
                         color=INK, loc="left", pad=18)
        ax_top.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7,
                      labelcolor=INK, handlelength=2.2)
        tidy(ax_top, "east (m)", "north (m)")
        tidy(ax_prof, "distance along the path (m)", "altitude (m)")
        ax_prof.set_xlim(left=0)
        ax_prof.set_ylim(bottom=0)

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.4), gridspec_kw=dict(height_ratios=(2.3, 1)))
    for col, (x_cls, y_cls, m) in enumerate(pairs):
        draw_pair(axes[0, col], axes[1, col], x_cls, y_cls, m)
    fig.text(0.01, 0.022, "block ({}, {}): the first used block in scan order where both pairs of "
             "paths differ. Exact planner on the true cost fields, no network.".format(*b["block"]),
             fontsize=6.5, color=INK2)
    fig.text(0.01, 0.005, "Top: surface height, darker = taller; dot = start, star = goal. "
             "Bottom: altitude along each path, surface beneath it shaded in the same colour.",
             fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save(fig, "austin_paths")

    fig, (ax_top, ax_prof) = plt.subplots(2, 1, figsize=(3.5, 4.9),
                                          gridspec_kw=dict(height_ratios=(2.3, 1)))
    draw_pair(ax_top, ax_prof, *pairs[0])
    fig.tight_layout()
    save(fig, "austin_paths_level")

    rows = json.loads((OUT / "ladder.json").read_text())
    everyone = sorted({r["vol"] for r in rows})
    climbs = sorted({r["vol"] for r in rows if r["shortest_vertical"] > 0})
    held, style = {}, {}
    for tag, scope in (("all", set(everyone)), ("climb", set(climbs))):
        for k, label, colour, marker in (("k1", "best scalar map (K=1)", orange, "s"),
                                         ("k2", "best 2-channel map (K=2)", aqua, "D")):
            name = "{}_{}".format(k, tag)
            style[name] = (label, colour, marker)
            held[name] = {"regret": {}}
            for cls in list(REAL_CLASSES) + ["MEAN"]:
                cl = list(REAL_CLASSES) if cls == "MEAN" else [cls]
                by = {}
                for r in rows:
                    if r["cls"] in cl and r["vol"] in scope:
                        by.setdefault(r["vol"], []).append(r["regret_" + k])
                held[name]["regret"][cls] = stats(np.array([[np.mean(by[i]) for i in sorted(by)]]))
    panels = (("(a) all {} Austin blocks".format(len(everyone)), ("k1_all", "k2_all")),
              ("(b) the {} blocks where the shortest path climbs".format(len(climbs)),
               ("k1_climb", "k2_climb")))
    out = dot_whisker(held, REAL_CLASSES, panels,
                      "marker: mean over blocks   bar: 95% CI over blocks   best K=1/K=2 map scored "
                      "under the true K=3 field (which scores 0); no network",
                      FIGURES / "austin_ladder.png", key="regret", style=style,
                      figsize=(7.16, 3.3), bottom=0.12)
    written.append(out)

    data = {}
    for fp in sorted(OUT.glob("e1_held_*_s*.json")):
        d = json.loads(fp.read_text())
        data.setdefault(d["name"], {})[d["seed"]] = d["rows"]
    held = {m: {"grid_regret": {cls: stats(matrix(by, "grid_regret",
                                                  list(REAL_CLASSES) if cls == "MEAN" else [cls]))
                                for cls in list(REAL_CLASSES) + ["MEAN"]}}
            for m, by in data.items()}
    ref = held["k3_onehot"]["grid_regret"]["MEAN"]
    panels = (("(a) expressiveness, language given", ("k1_onehot", "k2_onehot", "k3_onehot")),
              ("(b) conditioning, new wording", ("k3_onehot", "k3_nli", "k3_nli_adapter",
                                                 "k3_embedding")))
    out = dot_whisker(held, REAL_CLASSES, panels,
                      "marker: mean of {} training seeds    bar: 95% bootstrap CI over {} Austin blocks"
                      "    trained on synthetic worlds only; exact grid planner".format(
                          ref["n_seeds"], ref["n_vols"]),
                      FIGURES / "austin_regret.png", figsize=(7.16, 3.9), bottom=0.13)
    written.append(out)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["build", "ladder", "eval", "eval-one", "report", "figures"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sentences", default="held", choices=["held", "train"])
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    if a.what == "figures":
        for p in figures():
            print("wrote " + str(p))
        return
    if a.what == "build":
        vols, meta, info = build()
        used = [b for b in info if b["used"]]
        print("points {:,} kept {:,}; grid {} columns at {} m".format(
            meta["n_points"], meta["n_kept"], meta["grid"], meta["voxel_m"]))
        print("blocks: {} used, {} skipped ({})".format(
            len(used), len(info) - len(used),
            ", ".join("{}={}".format(r, sum(b.get("reason") == r for b in info))
                      for r in ("coverage", "no valid start/goal"))))
        print("relief per block (m): median {:.0f}, max {:.0f}; solid fraction: median {:.2f}".format(
            np.median([b["relief_m"] for b in used]), max(b["relief_m"] for b in used),
            np.median([b["solid_frac"] for b in used])))
    elif a.what == "ladder":
        rows = ladder(load_blocks())
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "ladder.json").write_text(json.dumps(rows, default=float))
        _print_ladder(rows)
    elif a.what == "eval":
        from .multiseed import MODELS
        evaluate(a.models or list(MODELS), a.seeds, a.workers, (a.sentences,))
    elif a.what == "eval-one":
        eval_one(a.name, a.seed, a.sentences)
    else:
        report(a.sentences)


if __name__ == "__main__":
    main()
