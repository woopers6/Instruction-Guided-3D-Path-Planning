"""Multi-seed training, evaluation and error bars for E1."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .instructions import CLASSES

from .paths import MODELS as MODEL_DIR, SRC
from .paths import RESULTS as _RESULTS

RESULTS = _RESULTS / "multiseed"
N_TEST = 30

MODELS = {
    "k3_onehot": ["--k", "3", "--conditioning", "onehot"],
    "k1_onehot": ["--k", "1", "--conditioning", "onehot"],
    "k2_onehot": ["--k", "2", "--conditioning", "onehot"],
    "k3_embedding": ["--k", "3", "--conditioning", "embedding"],
    "k3_nli": ["--k", "3", "--conditioning", "nli"],
    "k3_nli_adapter": ["--k", "3", "--conditioning", "nli", "--adapter", "64"],
}
BASELINES = ("k3_onehot", "k1_onehot", "k2_onehot", "k3_embedding", "k3_nli")


def run_dir(name: str, seed: int) -> Path:
    return MODEL_DIR / name if seed == 0 else MODEL_DIR / "seeds" / "{}_s{}".format(name, seed)


def _stamp():
    return time.strftime("%H:%M:%S")


def _run(cmd, log: Path, tag: str):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("{} start {}".format(_stamp(), tag), flush=True)
    with open(log, "a") as fh:
        rc = subprocess.run(cmd, cwd=SRC, stdout=fh, stderr=subprocess.STDOUT).returncode
    print("{} done  {} exit={}".format(_stamp(), tag, rc), flush=True)
    return rc


def train(names, seeds, workers):
    jobs = [(n, s) for s in seeds for n in names
            if not (run_dir(n, s) / "last.pt").exists()]
    print("{} training {} models with {} workers".format(_stamp(), len(jobs), workers), flush=True)

    def one(job):
        n, s = job
        out = run_dir(n, s)
        cmd = [sys.executable, "-m", "ig3d.train", *MODELS[n], "--seed", str(s),
               "--out", str(out)]
        return _run(cmd, out / "train.log", "{} seed {}".format(n, s))

    with ThreadPoolExecutor(workers) as ex:
        rcs = list(ex.map(one, jobs))
    print("{} TRAIN_DONE failures={}".format(_stamp(), sum(r != 0 for r in rcs)), flush=True)


def _result_path(name, seed, sentences):
    return RESULTS / "e1_{}_{}_s{}.json".format(sentences, name, seed)


def eval_one(name, seed, sentences, n=N_TEST):
    from .experiments import e1_learned
    ck = run_dir(name, seed) / "best.pt"
    res = e1_learned({name: str(ck)}, n=n, sentences=sentences, use_prm=True, seed=seed)[name]
    res.update(name=name, seed=seed, n_volumes=n)
    RESULTS.mkdir(parents=True, exist_ok=True)
    _result_path(name, seed, sentences).write_text(json.dumps(res, default=float))


def evaluate(names, seeds, workers, sentence_sets=("held", "train")):
    jobs = [(n, s, t) for s in seeds for n in names for t in sentence_sets
            if (run_dir(n, s) / "last.pt").exists() and not _result_path(n, s, t).exists()]
    print("{} evaluating {} (model, seed, phrasing) cells with {} workers".format(
        _stamp(), len(jobs), workers), flush=True)

    def one(job):
        n, s, t = job
        cmd = [sys.executable, "-m", "ig3d.multiseed", "eval-one", "--name", n,
               "--seed", str(s), "--sentences", t]
        return _run(cmd, RESULTS / "logs" / "{}_{}_s{}.log".format(t, n, s),
                    "eval {} seed {} {}".format(n, s, t))

    with ThreadPoolExecutor(workers) as ex:
        rcs = list(ex.map(one, jobs))
    print("{} EVAL_DONE failures={}".format(_stamp(), sum(r != 0 for r in rcs)), flush=True)


def load(sentences):
    """name -> {seed: rows}"""
    data = {}
    for f in sorted(RESULTS.glob("e1_{}_*_s*.json".format(sentences))):
        d = json.loads(f.read_text())
        data.setdefault(d["name"], {})[d["seed"]] = d["rows"]
    return data


def matrix(rows_by_seed, key, classes, seeds=None):
    """(S, V) array: per seed and test volume, `key` averaged over `classes`."""
    seeds = sorted(rows_by_seed) if seeds is None else seeds
    out = []
    for s in seeds:
        by = {}
        for r in rows_by_seed[s]:
            if r["cls"] in classes:
                v = r.get(key)
                by.setdefault(r["vol"], []).append(np.nan if v is None else float(v))
        with np.errstate(all="ignore"):
            out.append([np.nanmean(by[v]) for v in sorted(by)])
    return np.asarray(out, float)


def stats(M, n_boot=10000, seed=0):
    """Mean over seeds and volumes, sd over seeds, bootstrap CI over volumes."""
    with np.errstate(all="ignore"):
        per_seed = np.nanmean(M, axis=1)
        v = np.nanmean(M, axis=0)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    mean = float(per_seed.mean())
    return dict(mean=mean,
                sd=float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan"),
                lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                per_seed=[float(x) for x in per_seed],
                sign_holds=int(np.sum(np.sign(per_seed) == np.sign(mean))),
                n_seeds=len(per_seed), n_vols=len(v))


def _pct(st):
    return "{:5.1f} ± {:3.1f} [{:4.1f}, {:4.1f}]".format(
        100 * st["mean"], 100 * st["sd"], 100 * st["lo"], 100 * st["hi"])


def _num(st, fmt="{:+.3f}"):
    return (fmt + " ± {:.3f} [" + fmt + ", " + fmt + "]").format(
        st["mean"], st["sd"], st["lo"], st["hi"])


PAIRS = (("open", "tight", "narrow_frac"), ("low", "high", "mean_alt"),
         ("climb_open", "descend_open", "climb_clear"), ("level", "shortest", "vertical"))


def comparison(data, a, b, key, classes):
    seeds = sorted(set(data[a]) & set(data[b]))
    D = matrix(data[a], key, classes, seeds) - matrix(data[b], key, classes, seeds)
    return stats(D)


def pair_gap(rows_by_seed, who, a, b, metric):
    """who_metric(A) - who_metric(B), per volume: the behavioural contrast."""
    return stats(matrix(rows_by_seed, who + "_" + metric, [a])
                 - matrix(rows_by_seed, who + "_" + metric, [b]))


def report(names=None):
    lines, out = [], {}
    for sentences in ("held", "train"):
        data = load(sentences)
        ms = [n for n in (names or MODELS) if n in data]
        if not ms:
            continue
        if sentences == "train":
            ms = [n for n in ms if "onehot" not in n]
        seeds_txt = ", ".join("{} {}".format(n, sorted(data[n])) for n in ms)
        lines += ["", "## E1, {} phrasings — seeds: {}".format(
            "held-out (new wording)" if sentences == "held" else "training (seen wording)",
            seeds_txt), ""]
        for key, title in (("grid_regret", "grid-planner regret, %"),
                           ("prm_regret", "PRM regret (full method), %; relative to a PRM "
                                          "on the true field, itself a sampled roadmap, so "
                                          "values can dip below 0")):
            lines += ["**{}** — mean ± sd over seeds [95% CI over test volumes]".format(title),
                      "", "| class | " + " | ".join(ms) + " |",
                      "|---|" + "---|" * len(ms)]
            for cls in list(CLASSES) + ["MEAN"]:
                cl = list(CLASSES) if cls == "MEAN" else [cls]
                cells = []
                for n in ms:
                    st = stats(matrix(data[n], key, cl))
                    out.setdefault(sentences, {}).setdefault(n, {}).setdefault(key, {})[cls] = st
                    cells.append(_pct(st))
                lines.append("| {} | {} |".format(
                    "**mean**" if cls == "MEAN" else cls, " | ".join(cells)))
            lines.append("")
        lines += ["**behavioural contrast** pred(A) − pred(B) on the pair's metric "
                  "(oracle for reference)", "", "| pair | metric | oracle | " +
                  " | ".join(ms) + " |", "|---|---|---|" + "---|" * len(ms)]
        for a, b, m in PAIRS:
            orc = pair_gap(data[ms[0]], "oracle", a, b, m)
            cells = []
            for n in ms:
                st = pair_gap(data[n], "pred", a, b, m)
                out[sentences][n].setdefault("pairs", {})[a + "/" + b] = st
                cells.append(_num(st))
            lines.append("| {}/{} | {} | {:+.3f} | {} |".format(a, b, m, orc["mean"],
                                                               " | ".join(cells)))
        lines.append("")

    held = load("held")
    tr = load("train")
    tier3 = ["climb_open", "descend_open"]
    claims = [
        ("learned scalar loses on 'level'", held, "k1_onehot", "k3_onehot", "grid_regret", ["level"]),
        ("learned scalar loses on tier 3", held, "k1_onehot", "k3_onehot", "grid_regret", tier3),
        ("K=2 vs K=3 on tier 3", held, "k2_onehot", "k3_onehot", "grid_regret", tier3),
        ("language gap, new wording (NLI − one-hot)", held, "k3_nli", "k3_onehot", "grid_regret", list(CLASSES)),
        ("language gap, seen wording (NLI − one-hot)", tr, "k3_nli", "k3_onehot", "grid_regret", list(CLASSES)),
        ("embedding − NLI, new wording", held, "k3_embedding", "k3_nli", "grid_regret", list(CLASSES)),
    ]
    for extra in (n for n in MODELS if n not in BASELINES):
        claims += [("{} − NLI, new wording".format(extra), held, extra, "k3_nli", "grid_regret", list(CLASSES)),
                   ("{} − NLI, seen wording".format(extra), tr, extra, "k3_nli", "grid_regret", list(CLASSES))]
    lines += ["", "## Paired comparisons (A − B, grid regret in %; negative = A better)", "",
              "| comparison | classes | A − B | holds in |", "|---|---|---|---|"]
    out["claims"] = {}
    for title, d, a, b, key, cl in claims:
        if a not in d or b not in d:
            continue
        st = comparison(d, a, b, key, cl)
        out["claims"][title] = st
        lines.append("| {} | {} | {} | {}/{} seeds |".format(
            title, ",".join(cl) if len(cl) < 8 else "all 8", _pct(st).replace("  ", " "),
            st["sign_holds"], st["n_seeds"]))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "report.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))
    print("\nwrote " + str(RESULTS / "report.md"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "eval", "eval-one", "report"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--name")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--sentences", choices=["held", "train"])
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    names = a.models or list(MODELS)
    if a.what == "train":
        train(names, a.seeds, a.workers)
    elif a.what == "eval":
        evaluate(names, a.seeds, a.workers)
    elif a.what == "eval-one":
        eval_one(a.name, a.seed, a.sentences)
    else:
        report(a.models)


if __name__ == "__main__":
    main()
