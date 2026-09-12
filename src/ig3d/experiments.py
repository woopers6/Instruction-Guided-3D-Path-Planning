"""The paper's experiments."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from .config import DEFAULT, CostConfig
from .costfield import gt_field_from_features, polyline_cost, project_to_k, volume_features
from .instructions import ALL_LABELS, CLASSES, TIER
from .metrics import PRIMARY, behaviour, regret
from .planner_grid import GridGraph, path_cost
from .volumes import generate

from .paths import RESULTS
TEST_SEED = 3000

PAIRS = (("open", "tight", "narrow_frac"), ("low", "high", "mean_alt"),
         ("climb_open", "descend_open", "climb_clear"), ("level", "shortest", "vertical"))
BEHAVIOUR_KEYS = ("narrow_frac", "mean_alt", "vertical", "climb_clear", "descend_clear",
                  "length", "mean_clear")


def _feats(vols, cost_cfg):
    return [volume_features(v, cost_cfg) for v in vols]


def _uniform_field(v, cost_cfg):
    """Instruction-INDEPENDENT control: plain shortest-path cost."""
    f = np.full((1,) + v.shape, cost_cfg.base, np.float32)
    f[0][v.occ == 1] = cost_cfg.c_max
    return f


def _save(name, obj, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / (name + ".json")).write_text(json.dumps(obj, indent=2, default=float))
    print("\nwrote " + str(out / (name + ".json")))


def e0_degeneracy_gate(n: int = 20, cost_cfg: CostConfig = DEFAULT.cost) -> dict:
    """Is a signed vertical preference a real preference?"""
    vols = generate(n, TEST_SEED, DEFAULT.volume)
    base, span = cost_cfg.base, cost_cfg.span
    uni, spa = [], []
    for v in vols:
        G = GridGraph(v.occ)
        f = volume_features(v, cost_cfg)
        flds = []
        for s in (0.1, 0.5, 0.9):
            fl = np.zeros((3,) + v.shape, np.float32)
            fl[0], fl[1], fl[2] = base, 2 * span * s, 2 * span * (1 - s)
            fl[0][v.occ == 1] = cost_cfg.c_max
            flds.append(fl)
        plans = [G.plan(fl, v.start, v.goal) for fl in flds]
        uni.append(max(regret(path_cost(flds[j], plans[i][0]), plans[j][1])
                       for i in range(3) for j in range(3) if i != j))
        fc = gt_field_from_features(v.occ, f, "climb_open", cost_cfg)
        fd = gt_field_from_features(v.occ, f, "descend_open", cost_cfg)
        (pc, cc), (pd, cd) = G.plan(fc, v.start, v.goal), G.plan(fd, v.start, v.goal)
        spa.append(max(regret(path_cost(fd, pc), cd), regret(path_cost(fc, pd), cc)))
    spa = np.asarray(spa)
    res = dict(n=n, uniform_max_cross_regret=float(max(uni)),
               spatial_mean_cross_regret=float(spa.mean()),
               spatial_max_cross_regret=float(spa.max()),
               spatial_frac_volumes_where_it_matters=float(np.mean(spa > 1e-6)))
    print("E0 degeneracy gate, {} test volumes".format(n))
    print("  uniform up/down split, sum fixed : max cross-regret {:.2e}  -> {}".format(
        res["uniform_max_cross_regret"],
        "DEGENERATE (as derived)" if res["uniform_max_cross_regret"] < 1e-5 else "NOT degenerate"))
    print("  spatially varying (climb vs descend in the open): mean cross-regret {:.3f}, "
          "max {:.3f}, matters in {:.0%} of volumes".format(
              res["spatial_mean_cross_regret"], res["spatial_max_cross_regret"],
              res["spatial_frac_volumes_where_it_matters"]))
    return res


e0_degeneracy = e0_degeneracy_gate


def e1_learned(ckpts: dict, n: int = 40, sentences: str = "held", use_prm: bool = True,
               seed: int = 0) -> dict:
    """Predict fields on unseen test volumes and plan on them."""
    from .device import get_device
    from .planner_prm import plan as prm_plan
    from .train import load_model, predict

    dev = get_device()
    vols = generate(n, TEST_SEED, DEFAULT.volume)
    graphs = [GridGraph(v.occ) for v in vols]
    out = {}
    for name, path in ckpts.items():
        model, meta = load_model(path, dev)
        k = meta["k"]
        ccfg = CostConfig(**meta["config"]["cost"])
        feats = _feats(vols, ccfg)
        table = np.asarray(meta["cond_table"], np.float32)
        pool = np.asarray(meta["held_sent"] if sentences == "held" else meta["train_sent"])
        rng = np.random.default_rng(seed)
        rows, diag = [], {a + "/" + b: [] for a, b, _ in PAIRS}
        for vi, (v, G, f) in enumerate(zip(vols, graphs, feats)):
            bu = behaviour(v, G.plan(_uniform_field(v, ccfg), v.start, v.goal)[0], f["edt"])
            preds = {}
            for ci, cls in enumerate(CLASSES):
                si = int(rng.choice(pool[ALL_LABELS[pool] == ci]))
                truth = gt_field_from_features(v.occ, f, cls, ccfg, 3)
                p_opt, c_opt = G.plan(truth, v.start, v.goal)
                pred = predict(model, v.occ, table[si], ccfg, dev)
                preds[cls] = pred
                pg, _ = G.plan(pred, v.start, v.goal)
                bp, bo = behaviour(v, pg, f["edt"]), behaviour(v, p_opt, f["edt"])
                rec = dict(vol=vi, cls=cls, sent=si,
                           grid_regret=regret(path_cost(truth, pg), c_opt))
                for key in BEHAVIOUR_KEYS:
                    rec["pred_" + key], rec["oracle_" + key] = bp[key], bo[key]
                    rec["uniform_" + key] = bu[key]
                if use_prm:
                    pp, _ = prm_plan(v.occ, pred, v.start, v.goal, DEFAULT.prm,
                                     np.random.default_rng((seed, vi, ci)))
                    po, _ = prm_plan(v.occ, truth, v.start, v.goal, DEFAULT.prm,
                                     np.random.default_rng((seed, vi, ci)))
                    rec["prm_ok"] = pp is not None
                    if pp is not None and po is not None:
                        rec["prm_regret"] = polyline_cost(truth, pp) / polyline_cost(truth, po) - 1
                        rec["prm_primary"] = behaviour(v, pp, f["edt"])[PRIMARY[cls][0]]
                rows.append(rec)
            free = v.free
            for a, b, _ in PAIRS:
                ga = gt_field_from_features(v.occ, f, a, ccfg, k)
                gb = gt_field_from_features(v.occ, f, b, ccfg, k)
                diag[a + "/" + b].append((float(np.abs(preds[a] - preds[b])[:, free].mean()),
                                          float(np.abs(ga - gb)[:, free].mean())))
        out[name] = dict(k=k, conditioning=meta["conditioning"], sentences=sentences,
                         rows=rows, summary=_summarise_e1(rows, diag, use_prm))
        _print_e1(name, out[name], n)
    return out


def _summarise_e1(rows, diag, use_prm):
    s = dict(classes={}, pairs={}, conditioning_gap={})
    for cls in CLASSES:
        rs = [r for r in rows if r["cls"] == cls]
        m = PRIMARY[cls][0]
        s["classes"][cls] = dict(
            primary=m,
            pred=float(np.nanmean([r["pred_" + m] for r in rs])),
            oracle=float(np.nanmean([r["oracle_" + m] for r in rs])),
            uniform=float(np.nanmean([r["uniform_" + m] for r in rs])),
            grid_regret=float(np.mean([r["grid_regret"] for r in rs])),
            prm_regret=float(np.nanmean([r.get("prm_regret", np.nan) for r in rs]))
            if use_prm else None,
            prm_success=float(np.mean([r.get("prm_ok", True) for r in rs])))
    by = {(r["vol"], r["cls"]): r for r in rows}
    vols = sorted({r["vol"] for r in rows})
    for a, b, m in PAIRS:
        rec = {}
        for who in ("pred", "oracle", "uniform"):
            x = np.array([by[(i, a)][who + "_" + m] for i in vols], float)
            y = np.array([by[(i, b)][who + "_" + m] for i in vols], float)
            ok = ~np.isnan(x) & ~np.isnan(y)
            rec[who] = (float(x[ok].mean()), float(y[ok].mean())) if ok.any() else (np.nan, np.nan)
        s["pairs"][a + "/" + b] = dict(metric=m, **rec)
    for key, vals in diag.items():
        arr = np.asarray(vals)
        s["conditioning_gap"][key] = dict(pred=float(arr[:, 0].mean()), gt=float(arr[:, 1].mean()))
    return s


def _print_e1(name, res, n):
    s = res["summary"]
    print("\nE1 [{}]  K={}  conditioning={}  {} phrasings, {} test volumes".format(
        name, res["k"], res["conditioning"], res["sentences"], n))
    print("{:13s} {:>12s} {:>8s} {:>8s} {:>8s} {:>9s} {:>9s} {:>6s}".format(
        "class", "primary", "pred", "oracle", "uniform", "gridRegr", "prmRegr", "prmOK"))
    for cls, c in s["classes"].items():
        print("{:13s} {:>12s} {:8.3f} {:8.3f} {:8.3f} {:9.4f} {:9.4f} {:6.2f}".format(
            cls, c["primary"][:12], c["pred"], c["oracle"], c["uniform"], c["grid_regret"],
            c["prm_regret"] if c["prm_regret"] is not None else np.nan, c["prm_success"]))
    print("  contrast pairs (A vs B on the pair's metric):")
    for key, p in s["pairs"].items():
        print("    {:26s} {:12s} pred {:.3f}/{:.3f}  oracle {:.3f}/{:.3f}  uniform {:.3f}/{:.3f}".format(
            key, p["metric"], *p["pred"], *p["oracle"], *p["uniform"]))
    print("  conditioning diagnostic, mean |field_A - field_B|:")
    for key, g in s["conditioning_gap"].items():
        print("    {:26s} pred {:.4f}   ground truth {:.4f}   ratio {:.2f}".format(
            key, g["pred"], g["gt"], g["pred"] / g["gt"] if g["gt"] > 0 else np.nan))


def _altitude_band_scalar(v, cost_cfg, height_ref=8.0):
    """A scalar field that fakes 'keep it level' as 'stay near the START altitude'."""
    z = np.arange(v.shape[0], dtype=np.float32)[:, None, None]
    c = cost_cfg.base + cost_cfg.span * np.clip(np.abs(z - v.start[0]) / height_ref, 0, 1)
    c = np.broadcast_to(c, v.shape).astype(np.float32).copy()
    c[v.occ == 1] = cost_cfg.c_max
    return c[None]


def e2_ladder(n: int = 40, cost_cfg: CostConfig = DEFAULT.cost) -> dict:
    """Plan under the true K=3 field and its best K=2 and K=1 approximations."""
    vols = generate(n, TEST_SEED, DEFAULT.volume)
    rows = []
    for vi, v in enumerate(vols):
        G = GridGraph(v.occ)
        f = volume_features(v, cost_cfg)
        for cls in CLASSES:
            f3 = gt_field_from_features(v.occ, f, cls, cost_cfg, 3)
            p3, c3 = G.plan(f3, v.start, v.goal)
            rec = dict(vol=vi, cls=cls, **behaviour(v, p3, f["edt"]))
            for k in (1, 2):
                pk, _ = G.plan(project_to_k(f3, k), v.start, v.goal)
                rec["regret_k%d" % k] = regret(path_cost(f3, pk), c3)
                rec["primary_k%d" % k] = behaviour(v, pk, f["edt"])[PRIMARY[cls][0]]
            if cls == "level":
                pb, _ = G.plan(_altitude_band_scalar(v, cost_cfg), v.start, v.goal)
                rec["regret_start_aware_scalar"] = regret(path_cost(f3, pb), c3)
            rows.append(rec)

    by = {(r["vol"], r["cls"]): r for r in rows}
    summary = {}
    print("E2 representation ladder, {} test volumes, exact planner".format(n))
    print("{:13s} {:>4s} {:>13s} {:>8s} {:>8s} {:>7s} {:>7s} {:>9s} {:>9s}".format(
        "class", "tier", "primary", "K=3", "short.", "moved", "wrong", "regretK1", "regretK2"))
    for cls in CLASSES:
        m, sg = PRIMARY[cls]
        a = np.array([by[(i, cls)][m] for i in range(n)], float)
        s = np.array([by[(i, "shortest")][m] for i in range(n)], float)
        ok = ~np.isnan(a) & ~np.isnan(s)
        d = sg * (a[ok] - s[ok])
        r1 = float(np.mean([by[(i, cls)]["regret_k1"] for i in range(n)]))
        r2 = float(np.mean([by[(i, cls)]["regret_k2"] for i in range(n)]))
        summary[cls] = dict(tier=TIER[cls], primary=m, k3=float(np.nanmean(a)),
                            shortest=float(np.nanmean(s)),
                            frac_moved_right=float(np.mean(d > 1e-9)),
                            frac_moved_wrong=float(np.mean(d < -1e-9)),
                            regret_k1=r1, regret_k2=r2)
        print("{:13s} {:4d} {:>13s} {:8.3f} {:8.3f} {:7.2f} {:7.2f} {:9.4f} {:9.4f}".format(
            cls, TIER[cls], m, summary[cls]["k3"], summary[cls]["shortest"],
            summary[cls]["frac_moved_right"], summary[cls]["frac_moved_wrong"], r1, r2))
    ra = float(np.mean([by[(i, "level")]["regret_start_aware_scalar"] for i in range(n)]))
    summary["level_start_aware_scalar_regret"] = ra
    print("  level, scalar that KNOWS the start altitude (unavailable multi-query): "
          "regret {:.4f}".format(ra))
    x = np.array([by[(i, "climb_open")]["climb_clear"] for i in range(n)], float)
    y = np.array([by[(i, "descend_open")]["climb_clear"] for i in range(n)], float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    summary["tier3_discrimination"] = dict(
        k3_climb_clear_climb_open=float(x[ok].mean()),
        k3_climb_clear_descend_open=float(y[ok].mean()),
        k2_note="the K=2 projections of climb_open and descend_open are IDENTICAL fields, "
                "so K<=2 cannot distinguish the two instructions at all")
    print("  tier 3: climb clearance under K=3, climb_open {:.2f} vs descend_open {:.2f}; "
          "under K<=2 the two fields are identical by construction".format(
              x[ok].mean(), y[ok].mean()))
    return dict(summary=summary, rows=rows)


e1_expressiveness = e2_ladder


def sweep(n: int = 24, ratios=(2.0, 4.0, 8.0, 16.0), influences=(6.0, 12.0)) -> list:
    """Effect sizes versus the declared preference strength."""
    vols = generate(n, TEST_SEED, DEFAULT.volume)
    graphs = [GridGraph(v.occ) for v in vols]
    out = []
    print("ratio infl | o/t differ nf_open nf_tight len+open len+tight | "
          "vert_short vert_level lvl_rK1 | climb_rK2 climb_rK1")
    for infl in influences:
        feats = _feats(vols, dataclasses.replace(DEFAULT.cost, influence_r=infl))
        for ratio in ratios:
            cc = dataclasses.replace(DEFAULT.cost, influence_r=infl, detour_ratio=ratio)
            R = {k: [] for k in ("diff", "nfo", "nft", "lo", "lt", "vs", "vl", "lr1", "cr2", "cr1")}
            for v, G, f in zip(vols, graphs, feats):
                def bplan(cls):
                    return behaviour(v, G.plan(gt_field_from_features(v.occ, f, cls, cc),
                                               v.start, v.goal)[0], f["edt"])
                bs, bo, bt = bplan("shortest"), bplan("open"), bplan("tight")
                R["diff"].append(abs(bo["narrow_frac"] - bt["narrow_frac"]) > 1e-9)
                R["nfo"].append(bo["narrow_frac"])
                R["nft"].append(bt["narrow_frac"])
                R["lo"].append(bo["length"] / bs["length"] - 1)
                R["lt"].append(bt["length"] / bs["length"] - 1)
                fl = gt_field_from_features(v.occ, f, "level", cc)
                pl, cl = G.plan(fl, v.start, v.goal)
                R["vs"].append(bs["vertical"])
                R["vl"].append(behaviour(v, pl, f["edt"])["vertical"])
                R["lr1"].append(regret(path_cost(fl, G.plan(project_to_k(fl, 1), v.start,
                                                            v.goal)[0]), cl))
                fc = gt_field_from_features(v.occ, f, "climb_open", cc)
                _, c3 = G.plan(fc, v.start, v.goal)
                R["cr2"].append(regret(path_cost(fc, G.plan(project_to_k(fc, 2), v.start,
                                                            v.goal)[0]), c3))
                R["cr1"].append(regret(path_cost(fc, G.plan(project_to_k(fc, 1), v.start,
                                                            v.goal)[0]), c3))
            m = {k: float(np.nanmean(np.array(x, float))) for k, x in R.items()}
            m.update(detour_ratio=ratio, influence_r=infl)
            out.append(m)
            print("{:5.0f} {:4.0f} | {:10.2f} {:7.3f} {:8.3f} {:8.3f} {:9.3f} | {:10.1f} {:10.1f} "
                  "{:7.4f} | {:9.4f} {:9.4f}".format(ratio, infl, m["diff"], m["nfo"], m["nft"],
                                                     m["lo"], m["lt"], m["vs"], m["vl"], m["lr1"],
                                                     m["cr2"], m["cr1"]), flush=True)
    return out


def e3_conditioning(seeds=range(5)) -> dict:
    """Can each pathway tell the instruction classes apart on UNSEEN phrasings?"""
    from sklearn.linear_model import LogisticRegression

    from .encoder import NLI_AXES, EmbeddingEncoder, NLIEncoder
    from .instructions import split_indices

    raw = EmbeddingEncoder._raw(DEFAULT.model.encoder_model)
    nli_enc = NLIEncoder(DEFAULT.model.nli_model)
    nli = nli_enc.table
    nli_coef = NLIEncoder.signed_coefficients(nli_enc.probs)

    def probe(X, classes=None):
        accs = []
        for s in seeds:
            tr, he = split_indices(seed=s)
            if classes is not None:
                ids = [CLASSES.index(c) for c in classes]
                tr = tr[np.isin(ALL_LABELS[tr], ids)]
                he = he[np.isin(ALL_LABELS[he], ids)]
            clf = LogisticRegression(max_iter=5000).fit(X[tr], ALL_LABELS[tr])
            accs.append(clf.score(X[he], ALL_LABELS[he]))
        return float(np.mean(accs))

    res = dict(all8=dict(embedding=probe(raw), nli=probe(nli)), pairs={})
    print("E3 conditioning, held-out phrasings, {} splits".format(len(list(seeds))))
    print("  8-way (chance 0.125): embedding {:.3f}   nli {:.3f}".format(
        res["all8"]["embedding"], res["all8"]["nli"]))
    for a, b, _ in PAIRS:
        e, q = probe(raw, (a, b)), probe(nli, (a, b))
        res["pairs"][a + "/" + b] = dict(embedding=e, nli=q)
        print("  {:>10s} vs {:<13s} (chance 0.50): embedding {:.3f}   nli {:.3f}".format(a, b, e, q))
    leak = {c: nli_coef[ALL_LABELS == i].mean(0).tolist() for i, c in enumerate(CLASSES)}
    res["nli_mean_coef"] = dict(axes=list(NLI_AXES), per_class=leak)
    print("  mean NLI coefficient (rows: class, cols: " + ", ".join(NLI_AXES) + ")")
    for c, row in leak.items():
        print("    {:13s}".format(c) + "".join("{:8.2f}".format(x) for x in row))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["e0", "e1", "e2", "e3", "sweep"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--ckpt", action="append", default=[], help="name=path/to/best.pt")
    ap.add_argument("--sentences", default="held", choices=["held", "train"])
    ap.add_argument("--no-prm", action="store_true")
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args(argv)
    if a.exp == "e0":
        _save("e0", e0_degeneracy_gate(a.n or 20), a.out)
    elif a.exp == "e1":
        ck = dict(s.split("=", 1) for s in a.ckpt)
        if not ck:
            ap.error("e1 needs at least one --ckpt name=path")
        _save("e1_" + a.sentences, e1_learned(ck, a.n or 40, a.sentences, not a.no_prm), a.out)
    elif a.exp == "e2":
        _save("e2", e2_ladder(a.n or 40), a.out)
    elif a.exp == "sweep":
        _save("sweep", sweep(a.n or 24), a.out)
    elif a.exp == "e3":
        _save("e3", e3_conditioning(), a.out)


if __name__ == "__main__":
    main()
