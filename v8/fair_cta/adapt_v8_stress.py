"""STRESS-TEST COPY of adapt_v8.py (E3b, final_experiments/PREREG_FINAL.md): identical except that xi's scores are
replaced by a random balanced bimodal proxy before C1 (see the E3b block). Never used for the main v8 results.

FairCTA v8, "certify, then adapt": strict source-free, fully unsupervised on the target, one method for every dataset.

Only CELL's two source checkpoints cross over. On the target (features + edges only), every decision is gated by a
label-free check:
  C1 group certificate  : xi's partition is used only if both groups hold >= 10% of the nodes, xi's posterior has two
                          clear bumps (Ashman's D >= 2), and the partition is significantly NOT label-driven: the upper
                          95% Fisher-z bound of |corr(partition, source logit)| < 0.3 (chooser.choose_ci)
  C2 fairness strength  : one global eta (50, as v7). --etas also accepts a grid: one fine-tune per eta, keeping the
                          LARGEST eta whose result passes C3 and C4 ("self-set strength"). That variant was tested on
                          the quick check (runs/faircta_v8_selfset) and dropped: every eta passed, so it always took the
                          largest (200), which helped Syn slightly and hurt German's dEO (REPORT section 10).
                          Without a certified group guess: one anchor-only fine-tune
  C3 rank trust region  : Spearman(adapted logits, source logits) over all target nodes >= --rank_min (0.9)
  C4 do-no-harm         : the disparity measured with the group guess over all target nodes must not rise
If nothing passes, the output is the source model at the balanced operating point (logits re-centred at the median).
The fine-tune is v7's: soft CE to the centred source predictions (anchor) + eta * class-conditional independence
between Z_c and the group signal. Each candidate restarts from the checkpoint with the same seed and the same setup
order as v7, so the eta = 50 candidate is bit-identical to a v7 run (a built-in correctness check).
Guard (common.install_guard): target labels / split / meta, any source-graph file, dataset/, baseline runs/ forbidden.
Outputs (--run_dir): pred.npy (final), pred_adapted.npy, cand_eta<e>.npy per candidate, source_only.npy,
source_centered.npy, s_hat.npy, s_part.npy, s_choice.json, decision.json, monitor_eta<e>.csv, files_opened.txt.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import chooser
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=1111)
ap.add_argument('--split_dir', required=True)
ap.add_argument('--source', required=True); ap.add_argument('--target', required=True)
ap.add_argument('--run_dir', required=True)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--hidden_dim', type=int, default=64); ap.add_argument('--n_layers', type=int, default=3)
ap.add_argument('--dropout', type=float, default=0.4, help="the checkpoint's own training dropout (CELL config.yaml)")
ap.add_argument('--inter_encoder', default='GCN', choices=['GCN', 'vanilla'], help="the checkpoint's architecture")
ap.add_argument('--sens_encoder', default='MLP', choices=['MLP'])
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--weight_decay', type=float, default=1e-3)
ap.add_argument('--outer', type=int, default=333); ap.add_argument('--loop_model', type=int, default=3)
ap.add_argument('--momentum', type=float, default=0.9, help='lagged teacher (GraphCTA memory momentum)')
ap.add_argument('--anchor', type=float, default=1.0)
ap.add_argument('--etas', default='50', help='C2 fairness strength(s); a comma list = grid, the largest passing wins')
ap.add_argument('--s_rule', default='ci', choices=['ci', 'point', 'xi', 'none'],
                help='ci = C1 (v8); point = v7 chooser; xi / none force the guess (ablations only)')
ap.add_argument('--rank_min', type=float, default=0.9, help='C3 threshold; 0 = off')
ap.add_argument('--fair_guard', type=int, default=1, help='C4 on/off')
ap.add_argument('--center', type=int, default=1, help='1: balanced operating point (median); 0: raw logits (ablation)')
args = ap.parse_args()

install_guard(args.run_dir, args.target, args.source)                      # before any data file is opened
seed_all(args.seed)
dev = torch.device(args.device)
t0 = time.time()
x, edge_index = load_graph_unlabelled(args.split_dir, args.target)
x, edge_index = x.to(dev), edge_index.to(dev)
N, F_in = x.shape
src_dir = os.path.join(args.run_dir, "src_model")


def build():
    """v7's setup order after seed_all: task encoder, then sensitive encoder (both draw RNG at construction)."""
    m = load_cell_encoder(os.path.join(src_dir, f"inter_train_{args.inter_encoder}_{args.source}.pth"),
                          args.inter_encoder, F_in, dev, args.hidden_dim, args.n_layers, args.dropout)
    s = load_cell_encoder(os.path.join(src_dir, f"sens_train_{args.sens_encoder}_{args.source}.pth"),
                          args.sens_encoder, F_in, dev, args.hidden_dim, args.n_layers, args.dropout)
    for prm in s.parameters():
        prm.requires_grad_(False)
    s.eval()
    return m, s


def probs2(p1):
    return torch.stack([1 - p1, p1], 1)


def out(lf):
    """Output stage, identical for every candidate and the fallback."""
    return torch.sigmoid(lf - lf.median()) if args.center else torch.sigmoid(lf)


model, xi = build()
print(f"target {args.target}: N={N} F={F_in} E={edge_index.size(1)} (load {time.time() - t0:.1f}s)", flush=True)
with torch.no_grad():
    model.eval()
    _, logit0 = model(x, edge_index)
    l0 = logit0.view(-1)
    np.save(os.path.join(args.run_dir, "source_only.npy"), torch.sigmoid(l0).cpu().numpy())
    q_xi = torch.sigmoid(xi(x, edge_index)[1].view(-1))
    # ---- E3b STRESS TEST (adapt_v8_stress.py only): replace xi's scores by a random, balanced, bimodal proxy that is
    # independent of the graph, the features and the model. A separate numpy RNG keeps torch's RNG stream as in v8.
    _rng = np.random.default_rng(args.seed + 424242)
    _grp = _rng.permutation(N) < N // 2
    _sc = np.where(_grp, _rng.normal(0.75, 0.05, N), _rng.normal(0.25, 0.05, N)).clip(0.01, 0.99)
    q_xi = torch.tensor(_sc, dtype=q_xi.dtype, device=q_xi.device)
    p_anchor = torch.sigmoid(l0 - l0.median()) if args.center else torch.sigmoid(l0)   # anchor target
    p_src_out = out(l0)                                                     # fallback output
np.save(os.path.join(args.run_dir, "source_centered.npy"), p_src_out.cpu().numpy())
l0_np = l0.double().cpu().numpy()

# ---- C1 group certificate ----
e_np = edge_index.cpu().numpy(); qn = q_xi.double().cpu().numpy()
if args.s_rule == "ci":
    mode, soft, part, st = chooser.choose_ci(qn, (e_np[0], e_np[1]), N, l0_np)
elif args.s_rule == "point":
    mode, soft, part, st = chooser.choose(qn, (e_np[0], e_np[1]), N, l0_np)
else:
    mode, soft, part, st = chooser.choose_ci(qn, (e_np[0], e_np[1]), N, l0_np)
    mode = args.s_rule; st["mode"] = mode; st["forced"] = True
    if mode == "xi":
        soft, part = qn, chooser.two_means(qn)
    else:
        soft, part = np.full(N, 0.5), np.zeros(N, dtype=bool)
q = torch.tensor(soft, dtype=torch.float32, device=dev)
s_hat = torch.tensor(part, device=dev).long()
json.dump(st, open(os.path.join(args.run_dir, "s_choice.json"), "w"), indent=1)
np.save(os.path.join(args.run_dir, "s_hat.npy"), q.cpu().numpy())
np.save(os.path.join(args.run_dir, "s_part.npy"), part.astype(np.int8))
print(f"C1 group guess: {mode}  xi {st['xi']}", flush=True)

g = s_hat.bool()
prox = lambda a: float(abs((a[g] > .5).float().mean() - (a[~g] > .5).float().mean()))
d0 = prox(p_src_out) if mode != "none" else None


def leak(z, mem_prob):
    """v7's class-conditional independence: within each class c (soft weights = lagged teacher), the mean over dims of
    the squared correlation between normalised Z_c and the group signal q, weighted by the class share."""
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


def fine_tune(eta, name):
    """v7's anchored fine-tune from a fresh copy of the checkpoint, with v7's seed and setup order."""
    seed_all(args.seed)
    m, _ = build()
    opt = torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mem_prob = probs2(p_anchor).detach().clone()
    P_SRC = mem_prob.clone()
    delta = torch.full_like(x, 1e-7)          # v5/v7 parity: GraphCTA's dX starts at 1e-7 and is never updated here
    feat = x
    mon = open(os.path.join(args.run_dir, f"monitor_{name}.csv"), "w")
    mon.write("it,loss,leak,pos_rate,flip_vs_anchor\n")
    for it in range(args.outer):
        for _ in range(args.loop_model):
            m.train()
            opt.zero_grad()
            z, logit = m(feat, edge_index, None)
            p1 = torch.sigmoid(logit.view(-1)); prob = probs2(p1)
            loss = args.anchor * (-(P_SRC * torch.log(prob + 1e-8)).sum(1)).mean()
            lk = leak(z, mem_prob)
            loss = loss + eta * lk
            loss.backward()
            opt.step()
            m.eval()
            with torch.no_grad():
                pe = probs2(torch.sigmoid(m(feat, edge_index, None)[1].view(-1)))
                mem_prob = (1 - args.momentum) * mem_prob + args.momentum * pe
        feat = (x + delta).detach()
        if it % 50 == 0 or it == args.outer - 1:
            with torch.no_grad():
                row = [it, loss.item(), lk.item(), (mem_prob[:, 1] > .5).float().mean().item(),
                       ((mem_prob[:, 1] > .5) != (p_anchor > .5)).float().mean().item()]
            mon.write(",".join(f"{v:.6g}" for v in row) + "\n"); mon.flush()
    m.eval()
    with torch.no_grad():
        return m(x + delta, edge_index, None)[1].view(-1)


# ---- C2 fairness strength, C3 rank trust region, C4 do-no-harm ----
etas = [float(e) for e in args.etas.split(",")] if mode != "none" else [0.0]
cands = []
for eta in etas:
    name = f"eta{eta:g}" if mode != "none" else "anchor"
    lf = fine_tune(eta, name)
    with torch.no_grad():
        p = out(lf)
    np.save(os.path.join(args.run_dir, f"cand_{name}.npy"), p.cpu().numpy())
    rho = float(spearmanr(lf.double().cpu().numpy(), l0_np).correlation)
    d1 = prox(p) if mode != "none" else None
    ok_rank = args.rank_min <= 0 or rho >= args.rank_min
    ok_fair = (not args.fair_guard) or mode == "none" or d1 <= d0
    cands.append(dict(eta=eta, name=name, rho=rho, proxy_dDP=d1, pass_rank=bool(ok_rank), pass_fair=bool(ok_fair), p=p))
    print(f"candidate {name}: rank rho {rho:.4f} (>= {args.rank_min}: {ok_rank}) | proxy dDP "
          f"{'-' if d1 is None else f'{100 * d0:.2f} -> {100 * d1:.2f}'} (ok {ok_fair}) ({time.time() - t0:.0f}s)", flush=True)

passing = [c for c in cands if c["pass_rank"] and c["pass_fair"]]
chosen = max(passing, key=lambda c: c["eta"]) if passing else None
np.save(os.path.join(args.run_dir, "pred_adapted.npy"),
        (chosen or max(cands, key=lambda c: c["eta"]))["p"].cpu().numpy())
p_fin = chosen["p"] if chosen else p_src_out
np.save(os.path.join(args.run_dir, "pred.npy"), p_fin.cpu().numpy())
why = (f"largest passing candidate: {chosen['name']}" if chosen else "no candidate passed C3/C4: source model at the "
       "balanced operating point")
json.dump(dict(mode=mode, choice="adapted" if chosen else "centred_source", chosen=chosen["name"] if chosen else None,
               why=why, proxy_dDP_source=d0,
               candidates=[{k: v for k, v in c.items() if k != "p"} for c in cands],
               rule=dict(etas=args.etas, rank_min=args.rank_min, fair_guard=args.fair_guard, s_rule=args.s_rule,
                         center=args.center, anchor=args.anchor)),
          open(os.path.join(args.run_dir, "decision.json"), "w"), indent=1)
print(f"decision: {'adapted' if chosen else 'centred_source'} ({why}); done in {time.time() - t0:.0f}s", flush=True)
