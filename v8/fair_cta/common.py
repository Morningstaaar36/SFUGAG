"""Shared pieces for the source-free FairCTA sandbox (GraphCTA adaptation loop on CELL's source model).

Leakage guard: `install_guard` adds a Python audit hook that
  * logs every file the process opens under the repository (written to files_opened.txt at exit), and
  * raises PermissionError if the process tries to open a forbidden file.
The adaptation process may open only the target graph's features + edges and the source *model* checkpoints:
target labels, target sensitive attributes, target split masks, any source-graph file and the raw datasets
are forbidden, so the adaptation code physically cannot read them.
"""
import atexit, os, re, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
SANDBOX = os.path.join(REPO, "v8")
_R = re.escape(REPO)
GRAPHS = {"pokec": ("pokec_z", "pokec_n"), "syn": ("syn-2", "syn-1"),
          "german": ("german_2", "german_1"), "bailA": ("bailA_2", "bailA_1")}


def install_guard(run_dir, target_graph, source_graph):
    """Forbid every file that would leak target labels / sensitive attributes or source data."""
    forbidden = [
        re.compile(rf"/{re.escape(target_graph)}/split\.npz$"),                 # target masks, y, sens
        re.compile(rf"/{re.escape(target_graph)}_labels\.txt$"),                 # target labels
        re.compile(rf"/{re.escape(target_graph)}/meta\.json$"),
        re.compile(rf"/splits/seed_\d+/{re.escape(source_graph)}/"),             # any source-graph file
        re.compile(rf"{_R}/dataset(_derived)?/"),                              # raw data (has y and s)
        re.compile(rf"{_R}/runs/"),                                            # baseline results
    ]
    opened = []

    def hook(event, args):
        if event != "open" or not args or isinstance(args[0], int):
            return
        try:
            p = os.path.abspath(os.fsdecode(args[0]))
        except (TypeError, ValueError):
            return
        rp = os.path.realpath(p)                     # splits/ is a symlink: check both spellings
        if not p.startswith(REPO + "/") and not rp.startswith(REPO + "/"):
            return
        for cand in (p, rp):
            for pat in forbidden:
                if pat.search(cand):
                    raise PermissionError(f"LEAKAGE GUARD: adaptation tried to open forbidden file {cand}")
        if not rp.endswith((".py", ".pyc", ".so")):
            opened.append(rp)

    sys.addaudithook(hook)

    def dump():
        with open(os.path.join(run_dir, "files_opened.txt"), "w") as f:
            f.write("\n".join(sorted(set(opened))) + "\n")
    atexit.register(dump)


def load_graph_unlabelled(split_dir, graph):
    """Features and edges only (exactly the tensors CELL's loader built; see tools/make_splits.py)."""
    raw = os.path.join(split_dir, graph, "raw")
    x = pd.read_csv(os.path.join(raw, f"{graph}_docs.txt"), header=None, dtype=np.float64).to_numpy()
    ei = pd.read_csv(os.path.join(raw, f"{graph}_edgelist.txt"), header=None, dtype=np.int64).to_numpy()
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(ei.T.copy())


# ---------------- CELL encoders (CELL_ICML_2026/models/encoder.py), same module names -> strict load ----------------
class _GNNBody(nn.Module):
    def __init__(self, in_dim, hid, n_layers, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.feat_proj = nn.Linear(in_dim, hid)
        self.gnn_layers = nn.ModuleList([GCNConv(hid, hid) for _ in range(n_layers)])

    def forward(self, x, edge_index, edge_weight=None):
        x = self.feat_proj(x)
        embs = [x]
        for i, gnn in enumerate(self.gnn_layers):
            x = F.relu(gnn(x, edge_index, edge_weight))
            if i != len(self.gnn_layers) - 1:
                x = self.dropout(x)
            embs.append(x)
        return torch.stack(embs, dim=1).mean(dim=1)


class _MLPBody(nn.Module):
    def __init__(self, in_dim, hid, dropout):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(in_dim, hid), nn.Dropout(p=dropout), nn.ReLU(), nn.Linear(hid, hid))

    def forward(self, x, edge_index=None, edge_weight=None):
        return self.lin(x)


class _VanillaBody(nn.Module):                     # CELL GCN_Body: one GCNConv, no activation (german, syn)
    def __init__(self, in_dim, hid):
        super().__init__()
        self.gc1 = GCNConv(in_dim, hid)

    def forward(self, x, edge_index, edge_weight=None):
        return self.gc1(x, edge_index, edge_weight)


class _Head(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.body = nn.Linear(hid, 1)

    def forward(self, z):
        return self.body(z)


class CellEncoder(nn.Module):
    """encoder_type 'GCN' (CELL inter_encoder on pokec, bailA), 'vanilla' (german, syn) or 'MLP' (sens_encoder)."""
    def __init__(self, encoder_type, in_dim, hid=64, n_layers=3, dropout=0.4):
        super().__init__()
        if encoder_type == "GCN":
            self.body = _GNNBody(in_dim, hid, n_layers, dropout)
        elif encoder_type == "vanilla":
            self.body = _VanillaBody(in_dim, hid)
        else:
            self.body = _MLPBody(in_dim, hid, dropout)
        self.fc = _Head(hid)

    def forward(self, x, edge_index, edge_weight=None):
        z = self.body(x, edge_index, edge_weight)
        return z, self.fc(z)


def load_cell_encoder(path, encoder_type, in_dim, device, hid=64, n_layers=3, dropout=0.4):
    enc = CellEncoder(encoder_type, in_dim, hid, n_layers, dropout)
    enc.load_state_dict(torch.load(path, map_location="cpu")["encoder"], strict=True)
    return enc.to(device)


def seed_all(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
