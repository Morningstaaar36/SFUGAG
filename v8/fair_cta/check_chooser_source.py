"""SOURCE-ONLY sanity check of the v7 group-guess chooser (fair_cta/chooser.py), before any v7 target run.

On each dataset's SOURCE graph the true group is known, so we can see whether the fixed, label-free rule picks a
guess that is right (partition accuracy >= 0.8) or correctly picks none. 4 graphs x 2 candidates = 8 cases.
The rule's thresholds are NOT fitted here; this only checks them. Nothing from this script is shipped or read by
adapt_v7.py.
Guard: every TARGET graph file, the raw datasets and the baseline runs are forbidden in this process.
usage: python check_chooser_source.py <check_dir> [seed]      (check_dir/<ds>/src_model/ holds CELL's checkpoints)
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import chooser

check_dir = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1111
ENC = {"german": "vanilla", "syn": "vanilla", "bailA": "GCN", "pokec": "GCN"}
forbidden = [re.compile(rf"/splits/seed_\d+/{re.escape(t)}/") for _, t in GRAPHS.values()] + \
            [re.compile(rf"{re.escape(REPO)}/dataset(_derived)?/"), re.compile(rf"{re.escape(REPO)}/runs/")]


def hook(event, a):
    if event != "open" or not a or isinstance(a[0], int):
        return
    try:
        p = os.path.abspath(os.fsdecode(a[0]))
    except (TypeError, ValueError):
        return
    for cand in (p, os.path.realpath(p)):
        if any(f.search(cand) for f in forbidden):
            raise PermissionError(f"SOURCE GUARD: tried to open target/raw file {cand}")


sys.addaudithook(hook)
torch.set_num_threads(8)
split_dir = os.path.join(SANDBOX, "splits", f"seed_{seed}")
out = {}
for ds in ("german", "syn", "bailA", "pokec"):
    src, _ = GRAPHS[ds]
    x, ei = load_graph_unlabelled(split_dir, src)
    s = np.load(os.path.join(split_dir, src, "split.npz"))["sens"]            # source group: for scoring only
    par = os.path.join(check_dir, ds, "src_model")
    with torch.no_grad():
        logit = load_cell_encoder(f"{par}/inter_train_{ENC[ds]}_{src}.pth", ENC[ds], x.size(1), "cpu").eval()(x, ei)[1].view(-1).double().numpy()
        q = torch.sigmoid(load_cell_encoder(f"{par}/sens_train_MLP_{src}.pth", "MLP", x.size(1), "cpu").eval()(x, ei)[1].view(-1)).double().numpy()
    e = ei.numpy()
    mode, soft, part, st = chooser.choose(q, (e[0], e[1]), len(q), logit)
    st["community"] = st.pop("community_record")                             # reported, not a candidate (v7.1)
    parts = {"xi": chooser.two_means(q), "community": chooser.two_means(chooser.smooth(q, (e[0], e[1]), len(q)))}
    for k, pt in parts.items():
        st[k]["acc_true"] = float(max(np.mean(pt == s), np.mean(pt != s)))
    valid = [k for k in ("xi",) if st[k]["acc_true"] >= 0.8]
    verdict = ("right" if (mode in valid or (mode == "none" and not valid)) else
               "missed (safe: groups off)" if mode == "none" else "WRONG")
    out[ds] = dict(st, valid=valid, verdict=verdict)
    for k in parts:
        c = st[k]
        print(f"{ds:6s} {src:9s} {k:9s} share={c['share']:.3f} D={c['D']:6.2f} corr={c['corr']:.3f} ok={str(c['ok']):5s} "
              f"| true acc={c['acc_true']:.3f}")
    print(f"{ds:6s} -> choice: {mode:9s} valid: {valid}  => {verdict}\n", flush=True)
json.dump(out, open(os.path.join(check_dir, "chooser_source_check.json"), "w"), indent=1)
