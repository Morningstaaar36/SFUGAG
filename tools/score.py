"""Step 8: one scorer for all three methods = CELL learn.py::evaluate_per_class, split 'all'.

usage: python tools/score.py [runs_dir] [--tags cell,soga,graphcta]
Reads each run's final output of the FULL algorithm:
  runs/cell/<ds>/seed_<s>/target_probs_run0.npy   P(y=1) per target node
  runs/soga/<ds>/seed_<s>/soga.npy                logits [N, 2]  -> softmax
  runs/graphcta/<ds>/seed_<s>/graphcta.npy        probabilities [N, 2]
Metrics on eval_mask (labelled target nodes), threshold 0.5, x100:
  ACC = balanced accuracy (mean per-class recall), AUC, dDP = |P(yhat=1|s=0)-P(yhat=1|s=1)|,
  dEO = |P(yhat=1|y=1,s=0)-P(yhat=1|y=1,s=1)|.
  pACC = plain accuracy, EOd = paper Eq. 4 equalized odds = mean of the gaps at y=1 and y=0.
A run with no output file is a crash; a run predicting one class is listed separately and
excluded from the mean.
A tag ending in _src (e.g. graphcta_src) scores <tag without _src>'s runs with source_only.npy,
the frozen source model before adaptation. Splits are read from $SPLITS (default: <repo>/splits).
"""
import glob, json, os, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = {"german": "german_1", "bailA": "bailA_1", "syn": "syn-1", "pokec": "pokec_n"}
OUTPUT = {"cell": "target_probs_run0.npy", "soga": "soga.npy", "graphcta": "graphcta.npy"}
SPLITS = os.environ.get("SPLITS", os.path.join(ROOT, "splits"))


def p1_of(method, arr):
    if method == "cell":
        return arr.reshape(-1)
    if method == "soga":                                   # logits -> softmax
        e = np.exp(arr - arr.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True))[:, 1]
    return arr[:, 1]                                       # graphcta: probabilities


def metrics(p1, split):
    m = split["eval_mask"]; y = split["y_raw"][m]; s = split["sens"][m]; p = p1[m]
    pred = (p > 0.5).astype(int)
    return dict(acc=100 * np.mean([np.mean(pred[y == c] == c) for c in (0, 1)]),
                auc=100 * roc_auc_score(y, p),
                pacc=100 * np.mean(pred == y),
                dp=100 * abs(pred[s == 0].mean() - pred[s == 1].mean()),
                eo=100 * abs(pred[(s == 0) & (y == 1)].mean() - pred[(s == 1) & (y == 1)].mean()),
                eod=100 * np.mean([abs(pred[(s == 0) & (y == c)].mean() - pred[(s == 1) & (y == c)].mean()) for c in (1, 0)]),
                one_class=bool(pred.min() == pred.max()))


def cell_printed(run):
    txt = open(os.path.join(run, "logs", "log.txt"), errors="ignore").read()
    vals = [float(re.search(rf"^{k}: ([-\d.na]+)", txt, re.M).group(1)) for k in ("Acc", "auc_roc", "parity", "equality")]
    return dict(zip(("acc", "auc", "dp", "eo"), vals))


def main():
    runs = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(ROOT, "runs")
    tags = sys.argv[sys.argv.index("--tags") + 1].split(",") if "--tags" in sys.argv else list(OUTPUT)
    rows, bad = [], []
    for tag in tags:
        src = tag.endswith("_src"); run_tag = tag[:-4] if src else tag
        method = next(m for m in OUTPUT if run_tag.startswith(m))
        for run in sorted(glob.glob(os.path.join(runs, run_tag, "*", "seed_*"))):
            ds = run.split(os.sep)[-2]; seed = int(run.split("_")[-1])
            split = np.load(os.path.join(SPLITS, f"seed_{seed}", TARGET[ds], "split.npz"))
            f = os.path.join(run, "source_only.npy" if src else OUTPUT[method])
            if not os.path.exists(os.path.join(run, "DONE")):          # still running or killed: not a result
                rows.append(dict(tag=tag, ds=ds, seed=seed, status="INCOMPLETE")); continue
            if not os.path.exists(f):
                rows.append(dict(tag=tag, ds=ds, seed=seed, status="CRASH")); continue
            r = dict(tag=tag, ds=ds, seed=seed, status="ok", **metrics(p1_of(method, np.load(f)), split))
            if r["one_class"]:
                r["status"] = "ONE-CLASS"
            if method == "cell" and not src:             # the scorer must reproduce CELL's own printout
                pr = cell_printed(run)
                r["matches_cell_print"] = all(abs(round(r[k], 2) - pr[k]) < 0.011 for k in pr)
                if not r["matches_cell_print"]:
                    bad.append(f"{run}: scorer {[round(r[k], 2) for k in pr]} != CELL print {list(pr.values())}")
            rows.append(r)
    K = ("acc", "auc", "dp", "eo", "pacc", "eod")
    print(f"{'tag':12s} {'dataset':7s} {'seed':5s} {'status':9s}   ACC     AUC     dDP     dEO    pACC     EOd")
    for r in rows:
        v = "" if r["status"] in ("CRASH", "INCOMPLETE") else "  ".join(f"{r[k]:6.2f}" for k in K)
        extra = "" if "matches_cell_print" not in r else f"   CELL-print match: {r['matches_cell_print']}"
        print(f"{r['tag']:12s} {r['ds']:7s} {r['seed']:<5d} {r['status']:9s} {v}{extra}")
    print("\nmean ± std over ok runs (crash / one-class runs counted, not averaged):")
    for tag in dict.fromkeys(r["tag"] for r in rows):
        for ds in dict.fromkeys(r["ds"] for r in rows if r["tag"] == tag):
            g = [r for r in rows if r["tag"] == tag and r["ds"] == ds]
            ok = [r for r in g if r["status"] == "ok"]
            ms = "  ".join(f"{np.mean([r[k] for r in ok]):6.2f}±{np.std([r[k] for r in ok]):5.2f}" for k in K) if ok else "-"
            print(f"  {tag:12s} {ds:7s} n_ok={len(ok)}/{len(g)} crash={sum(r['status'] == 'CRASH' for r in g)} "
                  f"one_class={sum(r['status'] == 'ONE-CLASS' for r in g)} incomplete={sum(r['status'] == 'INCOMPLETE' for r in g)}   {ms}")
    out = "scores.json" if tags == list(OUTPUT) else f"scores_{'_'.join(tags)}.json"   # never clobber the main file
    json.dump(rows, open(os.path.join(runs, out), "w"), indent=1)
    if bad:
        print("\nSCORER / CELL MISMATCH:\n  " + "\n  ".join(bad)); sys.exit(1)


if __name__ == "__main__":
    main()
