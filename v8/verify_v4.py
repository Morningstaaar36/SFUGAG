"""Authenticity checks for the FairCTA v4 sweep (runs/faircta_v4/<ds>/seed_<s>/). Reads ground truth, own process.

usage: python v8/verify_v4.py [tag] [ds=reference_tag ...]   (default tag faircta_v4)
 1. leakage: files_opened.txt lists only the target's docs + edgelist of that seed and the run's own folder
 2. predictions: pred.npy / source_only.npy finite, in [0, 1], one value per target node
 3. source model: source_only.npy reproduces CELL's own logged pre-adaptation line (runs/cell/.../logs/log.txt)
 4. scorer: an independent sklearn re-implementation gives the same ACC / AUC / dDP / dEO as tools/score.py
 5. determinism: seed 1111 is bit-identical to an earlier run with the same flags (default for faircta_v4:
    pokec vs faircta_v4_faironly)
"""
import glob, importlib.util, json, os, re, sys
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
_spec = importlib.util.spec_from_file_location("shared_score", os.path.join(REPO, "tools", "score.py"))
S = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(S)
tag = sys.argv[1] if len(sys.argv) > 1 else "faircta_v4"
fails = []


def indep(p, split):
    m = split["eval_mask"]; y = split["y_raw"][m]; s = split["sens"][m]; pr = (p[m] > 0.5).astype(int)
    return dict(acc=100 * balanced_accuracy_score(y, pr), auc=100 * roc_auc_score(y, p[m]),
                dp=100 * abs(pr[s == 0].mean() - pr[s == 1].mean()),
                eo=100 * abs(pr[(s == 0) & (y == 1)].mean() - pr[(s == 1) & (y == 1)].mean()))


def cell_pre(ds, seed):
    txt = open(os.path.join(REPO, "runs", "cell", ds, f"seed_{seed}", "logs", "log.txt"), errors="ignore").read()
    line = next(l for l in txt.splitlines() if l.startswith("test on target data:"))
    return [float(v) for v in re.findall(r"\{'all': ([-\d.e]+)", line)]


runs = sorted(glob.glob(os.path.join(HERE, "runs", tag, "*", "seed_*")))
print(f"{len(runs)} runs of {tag}")
for run in runs:
    ds, seed = run.split(os.sep)[-2], int(run.rsplit("_", 1)[1])
    tgt = S.TARGET[ds]
    split = np.load(os.path.join(S.SPLITS, f"seed_{seed}", tgt, "split.npz"))
    n = json.load(open(os.path.join(REPO, "splits", f"seed_{seed}", tgt, "meta.json")))["num_nodes"]
    name = f"{ds}/seed_{seed}"
    if not os.path.exists(os.path.join(run, "DONE")):
        fails.append(f"{name}: not DONE"); continue
    # 1 leakage
    allowed = {os.path.join(REPO, "splits", f"seed_{seed}", tgt, "raw", f"{tgt}_{k}.txt") for k in ("docs", "edgelist")}
    opened = [l.strip() for l in open(os.path.join(run, "files_opened.txt")) if l.strip()]
    bad = [p for p in opened if p not in allowed and not p.startswith(os.path.realpath(run) + os.sep)]
    if bad or not allowed <= set(opened):
        fails.append(f"{name}: unexpected file access {bad or 'missing target inputs'}")
    # 2 predictions
    p, p0 = np.load(os.path.join(run, "pred.npy")), np.load(os.path.join(run, "source_only.npy"))
    for k, a in (("pred", p), ("source_only", p0)):
        if a.shape != (n,) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
            fails.append(f"{name}: {k} malformed shape={a.shape} min={a.min()} max={a.max()}")
    # 3 source model == CELL's pre-adaptation line
    m0 = S.metrics(p0, split); pre = cell_pre(ds, seed)
    d3 = max(abs(m0[k] - v) for k, v in zip(("acc", "auc", "dp", "eo"), pre))
    if d3 > 0.01:
        fails.append(f"{name}: source-only {[round(m0[k], 2) for k in ('acc', 'auc', 'dp', 'eo')]} != CELL pre {[round(v, 2) for v in pre]}")
    # 4 independent scorer
    m, mi = S.metrics(p, split), indep(p, split)
    d4 = max(abs(m[k] - mi[k]) for k in mi)
    if d4 > 1e-9:
        fails.append(f"{name}: scorer mismatch {d4}")
    flip = np.mean((p > 0.5) != (p0 > 0.5))
    print(f"{name:14s} leak-ok={not bad}  src-vs-CELL-pre max|d|={d3:.4f}  scorer max|d|={d4:.1e}  "
          f"flips vs source={100 * flip:4.1f}%  {tag}: {m['acc']:.2f}/{m['auc']:.2f}/{m['dp']:.2f}/{m['eo']:.2f}")
# 5 determinism: seed 1111 of each dataset vs an earlier run with the same flags (args: ds=reference_tag ...)
refs = dict(a.split("=") for a in sys.argv[2:]) if len(sys.argv) > 2 else (
    {"pokec": "faircta_v4_faironly"} if tag == "faircta_v4" else {})
for ds, ref in refs.items():
    a = os.path.join(HERE, "runs", tag, ds, "seed_1111", "pred.npy")
    b = os.path.join(HERE, "runs", ref, ds, "seed_1111", "pred.npy")
    if os.path.exists(a) and os.path.exists(b):
        same = np.array_equal(np.load(a), np.load(b))
        print(f"determinism: {tag}/{ds}/seed_1111 vs {ref}/{ds}/seed_1111 bit-identical = {same}")
        if not same:
            fails.append(f"{ds} seed_1111 not bit-identical to {ref} (max |d| {np.abs(np.load(a) - np.load(b)).max():.2e})")
print("\nALL CHECKS PASSED" if not fails else "\nFAILED:\n  " + "\n  ".join(fails))
sys.exit(1 if fails else 0)
