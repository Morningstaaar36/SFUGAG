"""Step 3: make one seed's split for one dataset pair with CELL's own code, and store it.

usage: python tools/make_splits.py <dataset> <inid> <outid> <seed> <splits_root>
   e.g. python tools/make_splits.py german _2 _1 1111 splits

Writes splits/seed_<seed>/<graph>/raw/<graph>_{edgelist,docs,labels}.txt  (SOGA + GraphCTA format)
       splits/seed_<seed>/<graph>/split.npz                              (masks, sens, y_raw)
       splits/seed_<seed>/<graph>/meta.json                              (counts + md5)
"""
import hashlib, json, os, sys

ds, inid, outid, seed, root = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), os.path.abspath(sys.argv[5])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "CELL_ICML_2026"))
logdir = os.path.join(root, "_logs"); os.makedirs(logdir, exist_ok=True)
# CELL's config.py parses sys.argv at import time; no --split_dir, so CELL's own loader makes the split
sys.argv = [sys.argv[0], "--dataset", ds, "--inid", inid, "--outid", outid, "--seed", str(seed),
            "--log_path", os.path.join(logdir, f"{ds}_seed{seed}.txt")]
import numpy as np
from config import args
from utils import seed_everything
from dataset import get_dataset

seed_everything(args.seed)                          # same order as CELL main.py: seed, source, target
graphs = [(inid, get_dataset(args, inid)), (outid, get_dataset(args, outid))]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


for gid, d in graphs:
    name = ("pokec" if "pokec" in ds else ds) + gid
    out = os.path.join(root, f"seed_{seed}", name)
    raw = os.path.join(out, "raw")
    os.makedirs(raw, exist_ok=True)
    ei = d.edge_index.numpy().T
    y = d.y.numpy().astype(np.int64)
    files = {
        f"raw/{name}_edgelist.txt": lambda p: np.savetxt(p, ei, fmt="%d", delimiter=","),
        f"raw/{name}_docs.txt": lambda p: np.savetxt(p, d.x.numpy(), fmt="%.9g", delimiter=","),
        f"raw/{name}_labels.txt": lambda p: np.savetxt(p, np.where(y < 0, 0, y), fmt="%d"),   # Pokec -1 -> filler 0
    }
    for rel, write in files.items():
        write(os.path.join(out, rel))
    np.savez(os.path.join(out, "split.npz"),
             train_mask=d.train_mask.numpy(), val_mask=d.val_mask.numpy(), test_mask=d.test_mask.numpy(),
             eval_mask=(d.train_mask | d.val_mask | d.test_mask).numpy(),   # = CELL's 'all' = labelled nodes
             sens=d.sens_labels.numpy().astype(np.int64), y_raw=y)
    meta = dict(dataset=ds, graph=name, seed=seed, num_nodes=int(d.x.shape[0]), num_features=int(d.x.shape[1]),
                num_edges_directed=int(ei.shape[0]), self_loops=int((ei[:, 0] == ei[:, 1]).sum()),
                labelled=int((y >= 0).sum()), train=int(d.train_mask.sum()), val=int(d.val_mask.sum()),
                test=int(d.test_mask.sum()), md5={rel: md5(os.path.join(out, rel)) for rel in files})
    meta["md5"]["split.npz"] = md5(os.path.join(out, "split.npz"))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    sys.stderr.write(f"[make_splits] seed {seed} {name}: n={meta['num_nodes']} f={meta['num_features']} "
                     f"E={meta['num_edges_directed']} labelled={meta['labelled']} "
                     f"train/val/test={meta['train']}/{meta['val']}/{meta['test']}\n")
