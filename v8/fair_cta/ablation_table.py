"""v8 ablation table: seed 1111 on all four datasets, plus a Bail seed-1114 column (the one case where v8's
certificates act). Scores saved predictions with the shared metric function (score_sandbox -> tools/score.py); re-runs
nothing, reads no label inside any adaptation process. A cell marked "=" is identical to v8 BY CONSTRUCTION: the
removed component cannot act there, e.g. the independence term on a dataset whose certificate found no group guess.
Writes runs/ablation_v8.txt.
usage: python ablation_table.py
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SB = os.path.dirname(HERE)
sys.path.insert(0, SB)
import score_sandbox as SS

K = ("acc", "auc", "dp", "eo")
CELLS = [("german", 1111), ("syn", 1111), ("bailA", 1111), ("pokec", 1111), ("bailA", 1114)]
V8 = "faircta_v8"
# (label, source per cell; None = identical to v8 by construction)
ROWS = [
    ("CELL source model (no adaptation)", {c: f"{V8}:source_only.npy" for c in CELLS}),
    ("source + median shift only", {c: f"{V8}:source_centered.npy" for c in CELLS}),
    ("no C1 bound, no C3", {("bailA", 1114): "abl8_point_norank"}),
    ("no C1 bound (point test), with C3", {("bailA", 1114): "abl8_point"}),
    ("no proxy screen at all (always xi)", {("bailA", 1111): "abl8_xi", ("bailA", 1114): "abl8_xi", ("pokec", 1111): "abl8_xi"}),
    ("no independence term (eta = 0)", {("german", 1111): "abl8_eta0", ("syn", 1111): "abl8_eta0"}),
    ("self-set strength (largest passing eta of 5..200)", {c: "faircta_v8_selfset" for c in CELLS}),
    ("no anchor", {c: "abl8_noanchor" for c in CELLS}),
    ("no median shift", {c: "abl8_nocenter" for c in CELLS}),
    ("v8 (full)", {c: V8 for c in CELLS}),
]


def score(src, ds, seed):
    tag, fname = (src.split(":", 1) + ["pred.npy"])[:2]
    f = os.path.join(SB, "runs", tag, ds, f"seed_{seed}", fname)
    if not os.path.exists(os.path.join(SB, "runs", tag, ds, f"seed_{seed}", "DONE")) or not os.path.exists(f):
        return None
    return SS.score_file(f, ds, seed)


S = lambda m: m["acc"] + m["auc"] - m["dp"] - m["eo"]
lines = ["v8 ablations. ACC / AUC / dDP / dEO (x100); S = ACC + AUC - dDP - dEO; '=' identical to v8 by construction",
         "| Row | German 1111 | Syn 1111 | Bail 1111 | Pokec 1111 | mean S (seed 1111) | Bail 1114 |",
         "|---|---|---|---|---|---|---|"]
for label, src in ROWS:
    cells, s1111 = [], []
    for c in CELLS:
        m = score(src.get(c, V8), *c)
        if m is None:
            cells.append("missing"); continue
        txt = " / ".join(f"{m[k]:.2f}" for k in K) + ("" if c in src else " (=)")
        cells.append(txt)
        if c[1] == 1111:
            s1111.append(S(m))
    ms = f"{np.mean(s1111):.2f}" if len(s1111) == 4 else "-"
    lines.append(f"| {label} | " + " | ".join(cells[:4]) + f" | {ms} | {cells[4]} |")
out = "\n".join(lines) + "\n"
print(out)
open(os.path.join(SB, "runs", "ablation_v8.txt"), "w").write(out)
