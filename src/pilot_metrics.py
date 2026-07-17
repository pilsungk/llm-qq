# pilot_metrics.py
# ex18 Minimal Pilot v1.2: joint assembly in token coordinates,
# canonicalization to semantic-option coordinates, order-matched pooling,
# registered robustness envelopes, conservative Gamma bounds, and the
# QQ / Gamma / saturation classifications.
#
# All statistical machinery reuses the validated read-only core
# (core/metrics_ex18.py). This module only:
#   (1) relabels token coordinates into semantic-option coordinates
#       (mandatory canonicalization BEFORE pooling, spec R1),
#   (2) pools order-matched mixtures with equal weight (Prop 6 / Q1),
#   (3) enumerates the 2^6 allocation vertices for the OE_A / OE_B / q_QQ
#       envelopes and combines the theory-note 4.3 conservative Gamma bounds.
#
# ASCII only. English comments.

import itertools

from core import metrics_ex18 as core_metrics

EPSILON_QQ = core_metrics.EPSILON_QQ  # 0.02, pre-registered
ATOL = 1e-9                           # envelope boundary tolerance (SRS 6.2.1)

CELLS = [("y", "y"), ("y", "n"), ("n", "y"), ("n", "n")]
TOK_CELLS = [("t1", "t1"), ("t1", "t2"), ("t2", "t1"), ("t2", "t2")]


# ---------------------------------------------------------------------------
# Token-coordinate joint assembly (before canonicalization)
# ---------------------------------------------------------------------------

def assemble_token_joint(pf_t1, ps_t1_given_t1, ps_t1_given_t2):
    """Assemble a 2x2 joint keyed by emit-token answers ('t1','t2').

    pf_t1            : binary-conditioned P(first answer = t1)
    ps_t1_given_t1   : binary-conditioned P(second = t1 | first = t1)
    ps_t1_given_t2   : binary-conditioned P(second = t1 | first = t2)

    Keys are (first_token_answer, second_token_answer). Reuses the validated
    core assembler (which uses 'y'/'n') and relabels 'y'->'t1', 'n'->'t2'.
    """
    j = core_metrics.assemble_joint(pf_t1, ps_t1_given_t1, ps_t1_given_t2)
    remap = {"y": "t1", "n": "t2"}
    return {(remap[a], remap[b]): v for (a, b), v in j.items()}


# ---------------------------------------------------------------------------
# Canonicalization: token coordinates -> semantic-option coordinates
# ---------------------------------------------------------------------------

def _tok_to_sem(swap):
    """Bijection emit-token -> semantic answer. semantic 'y' = option_1.
    map-1 / yesno: t1 (=A / Yes) is option_1 -> identity.
    map-2:        t1 (=A) is option_2 -> t1 maps to 'n', t2 maps to 'y'."""
    if swap:
        return {"t1": "n", "t2": "y"}
    return {"t1": "y", "t2": "n"}


def canonicalize_joint(joint_tok, swap):
    """Permute a token-coordinate joint into semantic-option coordinates.
    Applies the same relabeling to both the first and second answers."""
    m = _tok_to_sem(swap)
    return {(m[a], m[b]): v for (a, b), v in joint_tok.items()}


def canonicalize_raw(raw_tok, swap):
    """Permute a token-coordinate raw structure (with `other` mass per
    position) into semantic-option coordinates for the envelope routine.

    raw_tok keys:
      p_first          : (p_t1, p_t2, other)
      p_second_given_t1: (p_t1, p_t2, other)
      p_second_given_t2: (p_t1, p_t2, other)
    Returns a raw dict keyed p_first / p_second_given_y / p_second_given_n
    with each tuple ordered (p_semantic_y, p_semantic_n, other).
    """
    f = raw_tok["p_first"]
    c1 = raw_tok["p_second_given_t1"]
    c2 = raw_tok["p_second_given_t2"]
    if not swap:
        return {
            "p_first": (f[0], f[1], f[2]),
            "p_second_given_y": (c1[0], c1[1], c1[2]),
            "p_second_given_n": (c2[0], c2[1], c2[2]),
        }
    # swap: semantic y = t2. The "given semantic y" branch is the t2 branch.
    return {
        "p_first": (f[1], f[0], f[2]),
        "p_second_given_y": (c2[1], c2[0], c2[2]),
        "p_second_given_n": (c1[1], c1[0], c1[2]),
    }


# ---------------------------------------------------------------------------
# Order-matched pooling (equal weight)
# ---------------------------------------------------------------------------

def pool_joints(sem_joints):
    """Equal-weight mixture of semantic-coordinate joints (Prop 6, Q1).
    Every component must already be canonicalized to semantic coordinates."""
    w = 1.0 / len(sem_joints)
    return {c: w * sum(j[c] for j in sem_joints) for c in CELLS}


def pool_intervals(intervals):
    """Equal-weight mixture of per-mapping envelope intervals. Exact for the
    pooled envelope because q_QQ / OE are linear in the joints and the two
    mappings use independent allocation variables (Q1)."""
    w = 1.0 / len(intervals)
    lo = w * sum(iv[0] for iv in intervals)
    hi = w * sum(iv[1] for iv in intervals)
    return (lo, hi)


# ---------------------------------------------------------------------------
# Envelopes over the 2^6 allocation box (semantic coordinates)
# ---------------------------------------------------------------------------

def semantic_envelopes(raw_ab_sem, raw_ba_sem):
    """Exact [min,max] of q_QQ, OE_A, OE_B over the 64 allocation vertices.
    Each is multilinear in the six allocation variables, so extrema are at
    vertices. Reuses core._adjusted_joint / q_qq / order_effects."""
    qs, oea, oeb = [], [], []
    for verts in itertools.product((0.0, 1.0), repeat=6):
        t0a, tya, tna, t0b, tyb, tnb = verts
        j_ab = core_metrics._adjusted_joint(raw_ab_sem, t0a, tya, tna)
        j_ba = core_metrics._adjusted_joint(raw_ba_sem, t0b, tyb, tnb)
        qs.append(core_metrics.q_qq(j_ab, j_ba))
        oe = core_metrics.order_effects(j_ab, j_ba)
        oea.append(oe["OE_A"])
        oeb.append(oe["OE_B"])
    return (min(qs), max(qs)), (min(oea), max(oea)), (min(oeb), max(oeb))


# ---------------------------------------------------------------------------
# Conservative Gamma bounds (theory note 4.3)
# ---------------------------------------------------------------------------

def abs_bounds(interval):
    """(x_abs_min, x_abs_max) for an interval [lo, hi] (theory note 4.3)."""
    lo, hi = interval
    if lo <= 0.0 <= hi:
        amin = 0.0
    else:
        amin = min(abs(lo), abs(hi))
    amax = max(abs(lo), abs(hi))
    return amin, amax


def gamma_bounds(q_iv, oea_iv, oeb_iv):
    """Gamma_lower / Gamma_upper conservative certified bounds.
    Gamma = |q_QQ| - OSS; OSS = |OE_A| + |OE_B|."""
    q_amin, q_amax = abs_bounds(q_iv)
    a_amin, a_amax = abs_bounds(oea_iv)
    b_amin, b_amax = abs_bounds(oeb_iv)
    return {
        "gamma_lower": q_amin - a_amax - b_amax,
        "gamma_upper": q_amax - a_amin - b_amin,
        "q_abs": (q_amin, q_amax),
        "oea_abs": (a_amin, a_amax),
        "oeb_abs": (b_amin, b_amax),
    }


# ---------------------------------------------------------------------------
# Classifications
# ---------------------------------------------------------------------------

def classify_qq(q_iv, eps=EPSILON_QQ, atol=ATOL):
    """SATISFIED / VIOLATED / INDETERMINATE (registered envelope decision)."""
    return core_metrics.envelope_decision(q_iv[0], q_iv[1], eps=eps, atol=atol)


def classify_gamma(gamma_lower, gamma_upper):
    """CERTIFIED NONCONTEXTUAL / CONTEXTUAL / INDETERMINATE (theory 4.3)."""
    if gamma_upper <= 0.0:
        return "NONCONTEXTUAL"
    if gamma_lower > 0.0:
        return "CONTEXTUAL"
    return "INDETERMINATE"


def point_metrics(pooled_ab, pooled_ba):
    """Point OE_A / OE_B / OSS / q_QQ / Gamma from pooled semantic joints.
    Recomputed from the pooled joints, never averaged from mapping metrics."""
    oe = core_metrics.order_effects(pooled_ab, pooled_ba)
    q = core_metrics.q_qq(pooled_ab, pooled_ba)
    return {
        "OE_A": oe["OE_A"], "OE_B": oe["OE_B"], "OSS": oe["OSS"],
        "q_QQ": q, "Gamma_point": abs(q) - oe["OSS"],
    }


# ---------------------------------------------------------------------------
# Saturation and other-mass gate
# ---------------------------------------------------------------------------

def saturation(p_values, pos_thresh=0.99, frac_thresh=0.5):
    """An item is saturated if > frac_thresh of measured binary-conditioned
    positions have max(p, 1-p) > pos_thresh. p_values is the list of
    binary-conditioned probabilities (of one label) at each position."""
    sat = [max(p, 1.0 - p) > pos_thresh for p in p_values]
    n = len(p_values)
    frac = (sum(sat) / n) if n else 0.0
    return {
        "n_positions": n, "n_saturated": sum(sat),
        "frac_saturated": frac, "saturated": frac > frac_thresh,
    }


def other_mass_gate(masses, thr=0.01):
    """G2': PASS iff every measured position has other_mass < thr."""
    mx = max(masses) if masses else 0.0
    return {"passed": mx < thr, "max_other_mass": mx, "threshold": thr}


def joint_sums_to_one(joint, atol=1e-9):
    return abs(sum(joint.values()) - 1.0) <= atol
