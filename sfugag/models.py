"""Source encoders (same layout as CELL's, so the source checkpoints load) and the file-access guard."""
import atexit, os, re, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def install_guard(run_dir, root, target_graph, source_graph):
    """Raise on any attempt to open target labels or sensitive attributes, source-graph files or raw data.
    Every other file opened under `root` is logged to files_opened.txt."""
    forbidden = [
        re.compile(rf"/{re.escape(target_graph)}/split\.npz$"),
        re.compile(rf"/{re.escape(target_graph)}_labels\.txt$"),
        re.compile(rf"/{re.escape(target_graph)}/meta\.json$"),
        re.compile(rf"/splits/seed_\d+/{re.escape(source_graph)}/"),
        re.compile(r"/dataset(_derived)?/"),
        re.compile(r"/runs/"),
    ]
    opened = []

    def hook(event, args):
        if event != "open" or not args or isinstance(args[0], int):
            return
        try:
            p = os.path.abspath(os.fsdecode(args[0]))
        except (TypeError, ValueError):
            return
        rp = os.path.realpath(p)
        if root not in p and root not in rp:
            return
        for cand in (p, rp):
            for pat in forbidden:
                if pat.search(cand):
                    raise PermissionError(f"guard: forbidden file {cand}")
        if not rp.endswith((".py", ".pyc", ".so")):
            opened.append(rp)

    sys.addaudithook(hook)

    def dump():
        with open(os.path.join(run_dir, "files_opened.txt"), "w") as f:
            f.write("\n".join(sorted(set(opened))) + "\n")
    atexit.register(dump)


def load_graph_unlabelled(split_dir, graph):
    """Node features and edge list of the target graph; nothing else is read."""
    raw = os.path.join(split_dir, graph, "raw")
    x = pd.read_csv(os.path.join(raw, f"{graph}_docs.txt"), header=None, dtype=np.float64).to_numpy()
    ei = pd.read_csv(os.path.join(raw, f"{graph}_edgelist.txt"), header=None, dtype=np.int64).to_numpy()
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(ei.T.copy())


class _GNNBody(nn.Module):
    """Projection + GCN layers with layer-mean readout (Bail, Pokec)."""
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
    """Feature-only sensitive encoder."""
    def __init__(self, in_dim, hid, dropout):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(in_dim, hid), nn.Dropout(p=dropout), nn.ReLU(), nn.Linear(hid, hid))

    def forward(self, x, edge_index=None, edge_weight=None):
        return self.lin(x)


class _VanillaBody(nn.Module):
    """One GCN layer, no activation (German, Syn)."""
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


class Encoder(nn.Module):
    """encoder_type: 'GCN', 'vanilla' (task encoders) or 'MLP' (sensitive encoder). Returns (embedding, logit)."""
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


def load_encoder(path, encoder_type, in_dim, device, hid=64, n_layers=3, dropout=0.4):
    enc = Encoder(encoder_type, in_dim, hid, n_layers, dropout)
    enc.load_state_dict(torch.load(path, map_location="cpu")["encoder"], strict=True)
    return enc.to(device)


def seed_all(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
