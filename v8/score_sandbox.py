"""Score the sandbox runs with the repository's shared scorer, and put the saved baselines next to them.

usage: python v8/score_sandbox.py [--ds pokec] [--tags t1,t2] [--verify]
  runs:  v8/runs/<tag>/<ds>/seed_<s>/pred.npy         P(y=1) after adaptation   -> row <tag>
                                                 /source_only.npy  P(y=1) of the frozen source model -> row <tag>_src
The metric function is imported from tools/score.py (not re-implemented), so ACC/AUC/dDP/dEO are defined
exactly as for the baselines. --verify re-scores the saved CELL predictions with it and checks that they
equal the numbers in runs/scores.json.
"""
import glob, importlib.util, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
_spec = importlib.util.spec_from_file_location("shared_score", os.path.join(REPO, "tools", "score.py"))
S = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(S)          # S.metrics, S.TARGET, S.SPLITS
K = ("acc", "auc", "dp", "eo", "pacc", "eod")
BASE_JSON = os.path.join(REPO, "runs", "scores_cell_soga_graphcta_graphcta_src_graphcta_fixed_graphcta_fixed_src.json")


def load_baselines():
    """All-tag baseline scores; falls back to runs/scores.json if that file is absent."""
    if os.path.exists(BASE_JSON):
        return json.load(open(BASE_JSON))
    print("(all-tag baseline file not found; using runs/scores.json: no graphcta_src rows)")
    return json.load(open(os.path.join(REPO, "runs", "scores.json")))


def score_file(f, ds, seed):
    split = np.load(os.path.join(S.SPLITS, f"seed_{seed}", S.TARGET[ds], "split.npz"))
    return S.metrics(np.load(f).reshape(-1), split)


def collect(ds, tags):
    rows = []
    for tag in tags:
        for run in sorted(glob.glob(os.path.join(HERE, "runs", tag, ds, "seed_*"))):
            seed = int(run.rsplit("_", 1)[1])
            for suffix, fname in (("_src", "source_only.npy"), ("", "pred.npy")):
                r = dict(tag=tag + suffix, ds=ds, seed=seed)
                f = os.path.join(run, fname)
                if not os.path.exists(os.path.join(run, "DONE")):
                    r["status"] = "INCOMPLETE"
                elif not os.path.exists(f):
                    r["status"] = "CRASH"
                else:
                    r.update(score_file(f, ds, seed)); r["status"] = "ONE-CLASS" if r["one_class"] else "ok"
                rows.append(r)
    return rows


def table(rows, title):
    print(f"\n== {title} ==")
    print(f"{'tag':28s} {'seed':5s} {'status':9s}   ACC     AUC     dDP     dEO    pACC     EOd")
    for r in rows:
        v = "  ".join(f"{r[k]:6.2f}" for k in K) if r["status"] in ("ok", "ONE-CLASS") else ""
        print(f"{r['tag']:28s} {r['seed']:<5d} {r['status']:9s} {v}")
    print("mean ± std over ok runs:")
    for tag in dict.fromkeys(r["tag"] for r in rows):
        g = [r for r in rows if r["tag"] == tag]; ok = [r for r in g if r["status"] == "ok"]
        ms = "  ".join(f"{np.mean([r[k] for r in ok]):6.2f}±{np.std([r[k] for r in ok]):5.2f}" for k in K) if ok else "-"
        print(f"  {tag:28s} n_ok={len(ok)}/{len(g)}  {ms}")


def main():
    a = sys.argv
    ds = a[a.index("--ds") + 1] if "--ds" in a else "pokec"
    if "--verify" in a:
        base = [r for r in json.load(open(os.path.join(REPO, "runs", "scores.json"))) if r["tag"] == "cell" and r["ds"] == ds]
        worst = 0.0
        for b in base:
            m = score_file(os.path.join(REPO, "runs", "cell", ds, f"seed_{b['seed']}", "target_probs_run0.npy"), ds, b["seed"])
            worst = max(worst, max(abs(m[k] - b[k]) for k in K))
        print(f"verify: re-scored {len(base)} CELL {ds} runs with the imported scorer; max |diff| vs runs/scores.json = {worst:.2e}")
        if worst > 1e-9:
            sys.exit("VERIFY FAILED")
    tags = a[a.index("--tags") + 1].split(",") if "--tags" in a else sorted(
        d for d in os.listdir(os.path.join(HERE, "runs")) if os.path.isdir(os.path.join(HERE, "runs", d)))
    rows = collect(ds, tags)
    seeds = sorted({r["seed"] for r in rows})
    base = [r for r in load_baselines() if r["ds"] == ds and r["tag"] in ("cell", "graphcta", "graphcta_src", "soga")]
    table([r for r in base if r["seed"] in seeds], f"saved baselines ({ds}, seeds {seeds}) from runs/")
    if rows:
        table(rows, f"sandbox runs ({ds})")
    json.dump(rows, open(os.path.join(HERE, "runs", f"scores_{ds}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
