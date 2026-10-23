"""v7 label-free group-guess chooser (replaces v5/v6's source card). The same rule and constants for every dataset.

Inputs: the frozen sensitive head's posterior q (CELL's xi), the graph's edges, and the task model's own logits.
No label and no true group is read. The group guess is xi's posterior q, split by 1-D 2-means.
It is used only if it passes three fixed, conventional checks (thresholds set before any run, not fitted):
  share  : both groups hold at least MIN_SHARE of the nodes
  D      : its signal has two clearly separated bumps: Ashman's D >= MIN_D on a 2-component Gaussian mixture
           (D = sqrt(2) |mu1 - mu2| / sqrt(s1^2 + s2^2); D > 2 is the usual criterion for a clean separation)
  corr   : it does not follow the task model's own predictions: |corr(partition, task logit)| <= MAX_CORR
           (a partition that tracks the predictions is a label community, not a group)
Rule: xi if usable, else no group guess (every group term off).

v7.0 also had a second candidate, "community" (q smoothed STEPS times over the edges, then 2-means; adapt.py
--s_mode community). It was dropped in v7.1: on the SOURCE graphs no label-free test separates group communities
from other strong communities. bailA_2's community split crosses only 1.3% of edges (partition p-q 0.97) and passes
every check once the D test is made robust to outliers, yet race is randomly mixed there (true p-q 0.02) and the
split is at chance (0.54). community_stats() still reports it for the record.
"""
import numpy as np
from sklearn.mixture import GaussianMixture

MIN_SHARE, MIN_D, MAX_CORR, STEPS = 0.10, 2.0, 0.30, 40


def smooth(q, ei, n, steps=STEPS):
    u, v = ei
    deg = np.maximum(np.bincount(v, minlength=n), 1)
    h = q.astype(np.float64)
    for _ in range(steps):
        h = np.bincount(v, weights=h[u], minlength=n) / deg
    return h


def two_means(h, iters=50):
    """1-D 2-means, quartile init as in adapt.py; higher-mean cluster = 1."""
    c = np.percentile(h, [25, 75])
    if c[0] == c[1]:
        c = np.array([h.min(), h.max()])
    g = np.abs(h - c[1]) < np.abs(h - c[0])
    for _ in range(iters):
        g = np.abs(h - c[1]) < np.abs(h - c[0])
        if g.all() or not g.any():
            break
        c = np.array([h[~g].mean(), h[g].mean()])
    return g


def ashman_d(h):
    sd = h.std()
    if sd == 0:
        return 0.0
    z = ((h - h.mean()) / sd).reshape(-1, 1)
    gm = GaussianMixture(2, covariance_type="full", means_init=np.percentile(z, [25, 75]).reshape(-1, 1),
                         random_state=0, max_iter=500, tol=1e-6).fit(z)
    m, s2 = gm.means_.ravel(), gm.covariances_.ravel()
    return float(np.sqrt(2) * abs(m[0] - m[1]) / np.sqrt(s2[0] + s2[1]))


def stats(signal, part, logit):
    share = float(part.mean())
    corr = float(abs(np.corrcoef(part.astype(np.float64), logit.astype(np.float64))[0, 1])) if 0 < share < 1 else 1.0
    st = dict(share=share, D=ashman_d(signal), corr=corr)
    st["ok"] = bool(min(share, 1 - share) >= MIN_SHARE and st["D"] >= MIN_D and corr <= MAX_CORR)
    return st


def community_stats(q, ei, n, logit):
    """Record only (not a candidate since v7.1): the same checks on the community split."""
    h = smooth(q, ei, n)
    return stats(h, two_means(h), logit)


def choose(q, ei, n, logit):
    """q: xi posterior [n]; ei: (src, dst) numpy edge arrays; logit: task model logits [n].
    Returns mode ('xi' | 'none'), the soft group signal for the penalties, the hard partition, a log dict."""
    part = two_means(q)
    st = dict(xi=stats(q, part, logit), community_record=community_stats(q, ei, n, logit))
    mode = "xi" if st["xi"]["ok"] else "none"
    if mode == "xi":
        soft = q.astype(np.float64)
    else:
        soft, part = np.full(n, 0.5), np.zeros(n, dtype=bool)
    st.update(mode=mode, rule=dict(MIN_SHARE=MIN_SHARE, MIN_D=MIN_D, MAX_CORR=MAX_CORR, version="v7.1"))
    return mode, soft, part, st


# ---------------- v8: group certificate as a hypothesis test ----------------
Z95 = 1.959963984540054                      # two-sided 95% normal quantile


def corr_upper(r, n):
    """Upper end of the 95% Fisher-z confidence interval of a correlation r estimated from n nodes."""
    z = np.arctanh(min(abs(r), 1 - 1e-12))
    return float(np.tanh(z + Z95 / np.sqrt(max(n - 3, 1))))


def choose_ci(q, ei, n, logit):
    """v8 group certificate: v7's share and two-bump checks, plus a test that the partition is not label-driven:
    the upper 95% confidence bound of |corr(partition, task logit)| must be below MAX_CORR (v7 used the point
    estimate). Same return values as choose()."""
    part = two_means(q)
    st = dict(xi=stats(q, part, logit), community_record=community_stats(q, ei, n, logit))
    st["xi"]["corr_upper95"] = corr_upper(st["xi"]["corr"], n)
    st["xi"]["ok"] = bool(min(st["xi"]["share"], 1 - st["xi"]["share"]) >= MIN_SHARE and st["xi"]["D"] >= MIN_D
                          and st["xi"]["corr_upper95"] < MAX_CORR)
    mode = "xi" if st["xi"]["ok"] else "none"
    if mode == "xi":
        soft = q.astype(np.float64)
    else:
        soft, part = np.full(n, 0.5), np.zeros(n, dtype=bool)
    st.update(mode=mode, rule=dict(MIN_SHARE=MIN_SHARE, MIN_D=MIN_D, MAX_CORR=MAX_CORR, test="fisher-z 95% upper",
                                   version="v8"))
    return mode, soft, part, st
