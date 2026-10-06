"""Inferred-group check (C1): decides whether the sensitive predictor's scores q give a usable group partition."""
import numpy as np
from sklearn.mixture import GaussianMixture

MIN_SHARE, MIN_D, MAX_CORR = 0.10, 2.0, 0.30
Z95 = 1.959963984540054


def two_means(h, iters=50):
    """1-D 2-means, centres initialised at the 25th and 75th percentiles; the larger-centre cluster is group 1."""
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
    """Separation of a two-component Gaussian mixture fitted to the standardised scores."""
    sd = h.std()
    if sd == 0:
        return 0.0
    z = ((h - h.mean()) / sd).reshape(-1, 1)
    gm = GaussianMixture(2, covariance_type="full", means_init=np.percentile(z, [25, 75]).reshape(-1, 1),
                         random_state=0, max_iter=500, tol=1e-6).fit(z)
    m, s2 = gm.means_.ravel(), gm.covariances_.ravel()
    return float(np.sqrt(2) * abs(m[0] - m[1]) / np.sqrt(s2[0] + s2[1]))


def corr_upper(r, n):
    """Upper end of the 95% Fisher-z interval for a correlation r estimated from n nodes."""
    z = np.arctanh(min(abs(r), 1 - 1e-12))
    return float(np.tanh(z + Z95 / np.sqrt(max(n - 3, 1))))


def check_groups(q, n, logit):
    """q: sensitive-predictor probabilities [n]; logit: frozen task logits [n].
    Returns (mode, soft signal, hard partition, stats). mode is 'xi' if the check passes, else 'none'."""
    part = two_means(q)
    share = float(part.mean())
    corr = float(abs(np.corrcoef(part.astype(np.float64), logit.astype(np.float64))[0, 1])) if 0 < share < 1 else 1.0
    st = dict(share=share, D=ashman_d(q), corr=corr, corr_upper95=corr_upper(corr, n))
    st["ok"] = bool(min(share, 1 - share) >= MIN_SHARE and st["D"] >= MIN_D and st["corr_upper95"] < MAX_CORR)
    if st["ok"]:
        return "xi", q.astype(np.float64), part, dict(xi=st, mode="xi")
    return "none", np.full(n, 0.5), np.zeros(n, dtype=bool), dict(xi=st, mode="none")
