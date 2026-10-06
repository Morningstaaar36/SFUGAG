"""Target-side adaptation: anchored fine-tuning of the frozen source task encoder with a dependence penalty,
gated by an inferred-group check (C1), a rank check (C3) and an inferred-group parity check (C4).

Inputs: the target's node features and edges, and the two source-trained checkpoints (task, sensitive).
Output: pred.npy, the final probabilities. If no candidate passes, it is the median-centred source prediction.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from models import install_guard, load_graph_unlabelled, load_encoder, seed_all
from group_check import check_groups, two_means

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=1111)
ap.add_argument('--root', required=True, help='repository root, used by the file-access guard')
ap.add_argument('--split_dir', required=True)
ap.add_argument('--source', required=True); ap.add_argument('--target', required=True)
ap.add_argument('--run_dir', required=True)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--hidden_dim', type=int, default=64); ap.add_argument('--n_layers', type=int, default=3)
ap.add_argument('--dropout', type=float, default=0.4, help='dropout the checkpoints were trained with')
ap.add_argument('--inter_encoder', default='GCN', choices=['GCN', 'vanilla'])
ap.add_argument('--sens_encoder', default='MLP', choices=['MLP'])
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--weight_decay', type=float, default=1e-3)
ap.add_argument('--outer', type=int, default=333); ap.add_argument('--loop_model', type=int, default=3)
ap.add_argument('--momentum', type=float, default=0.9, help='update rate of the soft class weights')
ap.add_argument('--anchor', type=float, default=1.0)
ap.add_argument('--etas', default='50', help='penalty strength; a comma list fine-tunes once per value and keeps the largest that passes')
ap.add_argument('--s_rule', default='ci', choices=['ci', 'xi', 'none'], help="'xi' / 'none' force the group signal on / off (ablation)")
ap.add_argument('--rank_min', type=float, default=0.9, help='rank-check threshold; 0 disables it')
ap.add_argument('--fair_guard', type=int, default=1, help='inferred-group parity check on/off')
ap.add_argument('--center', type=int, default=1, help='1: median-centred logits; 0: raw logits (ablation)')
args = ap.parse_args()

install_guard(args.run_dir, args.root, args.target, args.source)
seed_all(args.seed)
dev = torch.device(args.device)
t0 = time.time()
x, edge_index = load_graph_unlabelled(args.split_dir, args.target)
x, edge_index = x.to(dev), edge_index.to(dev)
N, F_in = x.shape
src_dir = os.path.join(args.run_dir, "src_model")


def build():
    m = load_encoder(os.path.join(src_dir, f"inter_train_{args.inter_encoder}_{args.source}.pth"),
                     args.inter_encoder, F_in, dev, args.hidden_dim, args.n_layers, args.dropout)
    s = load_encoder(os.path.join(src_dir, f"sens_train_{args.sens_encoder}_{args.source}.pth"),
                     args.sens_encoder, F_in, dev, args.hidden_dim, args.n_layers, args.dropout)
    for prm in s.parameters():
        prm.requires_grad_(False)
    s.eval()
    return m, s


def probs2(p1):
    return torch.stack([1 - p1, p1], 1)


def out(lf):
    return torch.sigmoid(lf - lf.median()) if args.center else torch.sigmoid(lf)


model, xi = build()
with torch.no_grad():
    model.eval()
    _, logit0 = model(x, edge_index)
    l0 = logit0.view(-1)
    np.save(os.path.join(args.run_dir, "source_only.npy"), torch.sigmoid(l0).cpu().numpy())
    q_xi = torch.sigmoid(xi(x, edge_index)[1].view(-1))
    p_anchor = out(l0)
    p_src_out = p_anchor
np.save(os.path.join(args.run_dir, "source_centered.npy"), p_src_out.cpu().numpy())
l0_np = l0.double().cpu().numpy()

# C1: inferred-group check
qn = q_xi.double().cpu().numpy()
mode, soft, part, st = check_groups(qn, N, l0_np)
if args.s_rule in ("xi", "none"):
    mode = args.s_rule; st["mode"] = mode; st["forced"] = True
    if mode == "xi":
        soft, part = qn, two_means(qn)
    else:
        soft, part = np.full(N, 0.5), np.zeros(N, dtype=bool)
q = torch.tensor(soft, dtype=torch.float32, device=dev)
s_hat = torch.tensor(part, device=dev).long()
json.dump(st, open(os.path.join(args.run_dir, "s_choice.json"), "w"), indent=1)

g = s_hat.bool()
prox = lambda a: float(abs((a[g] > .5).float().mean() - (a[~g] > .5).float().mean()))
d0 = prox(p_src_out) if mode != "none" else None


def leak(z, mem_prob):
    """Class-conditional dependence: for each class c, the mean over embedding coordinates of the squared
    correlation between the normalised embedding and q, under soft class weights, weighted by the class share."""
    zc = F.normalize(z, dim=1)
    tot = zc.sum() * 0
    for c in (0, 1):
        a = mem_prob[:, c].detach(); wsum = a.sum().clamp_min(1e-8)
        dz = zc - (a.unsqueeze(1) * zc).sum(0) / wsum
        ds = q - (a * q).sum() / wsum
        cov = (a.unsqueeze(1) * dz * ds.unsqueeze(1)).sum(0) / wsum
        vz = (a.unsqueeze(1) * dz ** 2).sum(0) / wsum; vs = (a * ds ** 2).sum() / wsum
        tot = tot + (wsum / N) * (cov ** 2 / (vz * vs + 1e-8)).mean()
    return tot


def fine_tune(eta):
    """Fine-tune a fresh copy of the task checkpoint toward the centred source predictions."""
    seed_all(args.seed)
    m, _ = build()
    opt = torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mem_prob = probs2(p_anchor).detach().clone()
    P_SRC = mem_prob.clone()
    delta = torch.full_like(x, 1e-7)       # fixed offset, never updated; used from the second outer step on
    feat = x
    for it in range(args.outer):
        for _ in range(args.loop_model):
            m.train()
            opt.zero_grad()
            z, logit = m(feat, edge_index, None)
            prob = probs2(torch.sigmoid(logit.view(-1)))
            loss = args.anchor * (-(P_SRC * torch.log(prob + 1e-8)).sum(1)).mean() + eta * leak(z, mem_prob)
            loss.backward()
            opt.step()
            m.eval()
            with torch.no_grad():
                pe = probs2(torch.sigmoid(m(feat, edge_index, None)[1].view(-1)))
                mem_prob = (1 - args.momentum) * mem_prob + args.momentum * pe
        feat = (x + delta).detach()
    m.eval()
    with torch.no_grad():
        return m(x + delta, edge_index, None)[1].view(-1)


# candidates, then the rank check (C3) and the inferred-group parity check (C4)
etas = [float(e) for e in args.etas.split(",")] if mode != "none" else [0.0]
cands = []
for eta in etas:
    name = f"eta{eta:g}" if mode != "none" else "anchor"
    lf = fine_tune(eta)
    with torch.no_grad():
        p = out(lf)
    rho = float(spearmanr(lf.double().cpu().numpy(), l0_np).correlation)
    d1 = prox(p) if mode != "none" else None
    ok_rank = args.rank_min <= 0 or rho >= args.rank_min
    ok_fair = (not args.fair_guard) or mode == "none" or d1 <= d0
    cands.append(dict(eta=eta, name=name, rho=rho, proxy_dDP=d1, pass_rank=bool(ok_rank), pass_fair=bool(ok_fair), p=p))

passing = [c for c in cands if c["pass_rank"] and c["pass_fair"]]
chosen = max(passing, key=lambda c: c["eta"]) if passing else None
p_fin = chosen["p"] if chosen else p_src_out
np.save(os.path.join(args.run_dir, "pred.npy"), p_fin.cpu().numpy())
json.dump(dict(mode=mode, choice="adapted" if chosen else "centred_source", chosen=chosen["name"] if chosen else None,
               proxy_dDP_source=d0, candidates=[{k: v for k, v in c.items() if k != "p"} for c in cands]),
          open(os.path.join(args.run_dir, "decision.json"), "w"), indent=1)
