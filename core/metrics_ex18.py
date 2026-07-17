# metrics_ex18.py
# Core metrics for ex18 per SRS v0.6:
#   - OE_A, OE_B, OSS (Order Sensitivity Score)
#   - q_QQ residual
#   - 64-vertex robustness envelope over `other` mass allocations (Sec 6.2.1)
#   - TV / JS distances on order-aligned joints (Sec 6.3)
#   - Clopper-Pearson + Bonferroni simultaneous consistency check (Sec 4.6,
#     SRS v0.7; Wilson retained for comparison only after observed
#     simultaneous undercoverage)
# ASCII only. English comments per project policy.

import itertools
import numpy as np
from scipy.stats import norm

EPSILON_QQ = 0.02  # pre-registered (SRS 6.5.1)
EPSILON_OE = 0.05  # H-C decision only (SRS 6.5.1)


# ---------------------------------------------------------------------------
# Joint assembly (binary_logprob protocol)
# ---------------------------------------------------------------------------

def assemble_joint(p_first_y, p_second_y_given_first_y, p_second_y_given_first_n):
    """Assemble a 2x2 joint from binary-conditioned components.

    Returns dict keyed by (first, second) in {'y','n'}^2.
    p_first_y: P(first answer = Y) under binary conditioning.
    """
    pf = {"y": p_first_y, "n": 1.0 - p_first_y}
    ps = {
        "y": {"y": p_second_y_given_first_y, "n": 1.0 - p_second_y_given_first_y},
        "n": {"y": p_second_y_given_first_n, "n": 1.0 - p_second_y_given_first_n},
    }
    return {(a, b): pf[a] * ps[a][b] for a in "yn" for b in "yn"}


# ---------------------------------------------------------------------------
# Layer 1: order effect
# ---------------------------------------------------------------------------

def order_effects(joint_ab, joint_ba):
    """OE_A, OE_B, OSS from the two order-conditioned joints.

    joint_ab keys: (a, b) = (answer to A asked first, answer to B asked second)
    joint_ba keys: (b, a) = (answer to B asked first, answer to A asked second)
    """
    p_ab_by = joint_ab[("y", "y")] + joint_ab[("n", "y")]  # B marginal, B second
    p_ba_by = joint_ba[("y", "y")] + joint_ba[("y", "n")]  # B marginal, B first
    p_ab_ay = joint_ab[("y", "y")] + joint_ab[("y", "n")]  # A marginal, A first
    p_ba_ay = joint_ba[("y", "y")] + joint_ba[("n", "y")]  # A marginal, A second
    oe_b = p_ab_by - p_ba_by
    oe_a = p_ba_ay - p_ab_ay
    return {"OE_A": oe_a, "OE_B": oe_b, "OSS": abs(oe_a) + abs(oe_b)}


# ---------------------------------------------------------------------------
# Layer 2: QQ residual
# ---------------------------------------------------------------------------

def q_qq(joint_ab, joint_ba):
    """q_QQ = [pAB(Ay,Bn) + pAB(An,By)] - [pBA(By,An) + pBA(Bn,Ay)].

    joint_ba is keyed (b, a): first index is the B answer (asked first).
    pBA(By,An) -> joint_ba[('y','n')]; pBA(Bn,Ay) -> joint_ba[('n','y')].
    """
    lhs = joint_ab[("y", "n")] + joint_ab[("n", "y")]
    rhs = joint_ba[("y", "n")] + joint_ba[("n", "y")]
    return lhs - rhs


# ---------------------------------------------------------------------------
# 6.2.1 robustness envelope (multilinear, 2^6 vertex enumeration)
# ---------------------------------------------------------------------------

def _adjusted_joint(raw, t0, t_y, t_n):
    """Raw-level adjusted joint for one order.

    raw: dict with
      'p_first': (p_y, p_n, m0)           raw first-answer probs + other mass
      'p_second_given_y': (p_y, p_n, mY)  raw conditional probs + other mass
      'p_second_given_n': (p_y, p_n, mN)
    t0, t_y, t_n in [0, 1]: fraction of each other mass allocated to Y.
    """
    fy, fn, m0 = raw["p_first"]
    cy = raw["p_second_given_y"]
    cn = raw["p_second_given_n"]
    pf = {"y": fy + t0 * m0, "n": fn + (1.0 - t0) * m0}
    ps = {
        "y": {"y": cy[0] + t_y * cy[2], "n": cy[1] + (1.0 - t_y) * cy[2]},
        "n": {"y": cn[0] + t_n * cn[2], "n": cn[1] + (1.0 - t_n) * cn[2]},
    }
    return {(a, b): pf[a] * ps[a][b] for a in "yn" for b in "yn"}


def q_qq_envelope(raw_ab, raw_ba):
    """Exact [min, max] of q_QQ over the 2^6 vertices of the allocation box.

    q_QQ is multilinear in the six allocation variables, so extrema over the
    box are attained at vertices (coordinate-wise affine argument).
    """
    qs = []
    for verts in itertools.product((0.0, 1.0), repeat=6):
        t0a, tya, tna, t0b, tyb, tnb = verts
        j_ab = _adjusted_joint(raw_ab, t0a, tya, tna)
        j_ba = _adjusted_joint(raw_ba, t0b, tyb, tnb)
        qs.append(q_qq(j_ab, j_ba))
    return min(qs), max(qs)


def envelope_decision(q_min, q_max, eps=EPSILON_QQ, atol=1e-9):
    """SATISFIED / VIOLATED / INDETERMINATE per SRS 6.2.1.

    atol absorbs floating-point noise at the boundary (e.g. an exact
    [-0.02, 0.02] envelope computed as [-0.02000000000000024, ...]).
    """
    if q_min >= -eps - atol and q_max <= eps + atol:
        return "SATISFIED"
    if q_max < -eps - atol or q_min > eps + atol:
        return "VIOLATED"
    return "INDETERMINATE"


# ---------------------------------------------------------------------------
# Layer 3: distributional distances on aligned coordinates
# ---------------------------------------------------------------------------

def aligned_distances(joint_ab, joint_ba):
    """TV and JS divergence between p_AB(a,b) and p_BA(b,a)."""
    p = np.array([joint_ab[(a, b)] for a in "yn" for b in "yn"], dtype=float)
    q = np.array([joint_ba[(b, a)] for a in "yn" for b in "yn"], dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    tv = 0.5 * np.abs(p - q).sum()
    m = 0.5 * (p + q)

    def _kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return {"TV": float(tv), "JSD": js}


# ---------------------------------------------------------------------------
# 4.6 Wilson simultaneous sampling-consistency check
# ---------------------------------------------------------------------------

def wilson_interval(k, n, conf):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    z = norm.ppf(1.0 - (1.0 - conf) / 2.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def clopper_pearson_interval(k, n, conf):
    """Exact Clopper-Pearson interval for a binomial proportion."""
    from scipy.stats import beta
    alpha2 = (1.0 - conf) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(alpha2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha2, k + 1, n - k))
    return lo, hi


def consistency_check(p_logprob, counts, alpha=0.01, method="clopper_pearson"):
    """binary_logprob vs binary_rejection sampling-consistency check
    (SRS 4.6, F2/G2/G3; interval method corrected after calibration --
    Wilson showed simultaneous undercoverage (~0.98 observed vs 0.99
    nominal), exact Clopper-Pearson + Bonferroni restores the guarantee).

    p_logprob: dict {(a,b): prob} deterministic joint from logprobs.
    counts:    dict {(a,b): accepted count} from rejection sampling.
    n_rej is the number of ACCEPTED joint samples.
    Per-cell confidence 1 - alpha/4 (Bonferroni over 4 cells).
    """
    n_rej = sum(counts.values())
    conf = 1.0 - alpha / 4.0
    interval = (clopper_pearson_interval if method == "clopper_pearson"
                else wilson_interval)
    details = {}
    ok = True
    for cell in [("y", "y"), ("y", "n"), ("n", "y"), ("n", "n")]:
        lo, hi = interval(counts.get(cell, 0), n_rej, conf)
        inside = lo <= p_logprob[cell] <= hi
        details[cell] = {"ci": (lo, hi), "p": p_logprob[cell], "inside": inside}
        ok = ok and inside
    return ok, n_rej, details
