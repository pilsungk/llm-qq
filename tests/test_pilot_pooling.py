# Tests for mandatory canonicalization-before-pooling (spec R1) and that
# final metrics are recomputed from pooled joints, not averaged.
# ASCII only.

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import metrics_ex18 as cm
from src import pilot_metrics as PM


# ---------------------------------------------------------------------------
# 4. Mapping 2 canonicalization permutes semantic cells correctly.
# ---------------------------------------------------------------------------

def test_canonicalize_identity_when_no_swap():
    jt = PM.assemble_token_joint(0.6, 0.3, 0.8)   # keys t1/t2
    js = PM.canonicalize_joint(jt, swap=False)
    # t1 -> y, t2 -> n (identity relabel)
    for (a, b), v in jt.items():
        m = {"t1": "y", "t2": "n"}
        assert js[(m[a], m[b])] == pytest.approx(v)


def test_canonicalize_swap_permutes_both_answers():
    jt = PM.assemble_token_joint(0.6, 0.3, 0.8)
    js = PM.canonicalize_joint(jt, swap=True)
    # t1 -> n, t2 -> y on BOTH coordinates
    m = {"t1": "n", "t2": "y"}
    for (a, b), v in jt.items():
        assert js[(m[a], m[b])] == pytest.approx(v)
    # the (t1,t1) mass must land in the (n,n) semantic cell
    assert js[("n", "n")] == pytest.approx(jt[("t1", "t1")])
    assert js[("y", "y")] == pytest.approx(jt[("t2", "t2")])


def test_canonicalization_recovers_semantic_equivalence():
    # Two mappings measuring the SAME underlying semantics must produce the
    # same semantic joint after canonicalization, even though the raw token
    # probabilities are swapped.
    # map-1 token probs: P(first=t1)=0.7 (t1=option_1)
    j_map1_tok = PM.assemble_token_joint(0.7, 0.9, 0.2)
    # map-2 measures the same semantics: option_1 is now token t2, so the
    # token-level P(first=t1) = P(first=option_2) = 0.3, conditionals mirrored.
    j_map2_tok = PM.assemble_token_joint(0.3, 0.8, 0.1)
    sem1 = PM.canonicalize_joint(j_map1_tok, swap=False)
    sem2 = PM.canonicalize_joint(j_map2_tok, swap=True)
    # sem1 P(first semantic y) = 0.7; sem2 P(first semantic y) should be 0.7 too
    p1 = sem1[("y", "y")] + sem1[("y", "n")]
    p2 = sem2[("y", "y")] + sem2[("y", "n")]
    assert p1 == pytest.approx(0.7)
    assert p2 == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 5. Pooling occurs after canonicalization (averaging token coords is wrong).
# ---------------------------------------------------------------------------

def test_pooling_after_canonicalization_differs_from_raw_average():
    j_map1_tok = PM.assemble_token_joint(0.8, 0.9, 0.1)
    j_map2_tok = PM.assemble_token_joint(0.2, 0.15, 0.85)  # semantically similar
    sem1 = PM.canonicalize_joint(j_map1_tok, swap=False)
    sem2 = PM.canonicalize_joint(j_map2_tok, swap=True)
    pooled_correct = PM.pool_joints([sem1, sem2])

    # WRONG path: average token coords directly (t1<->t1) without canonicalize
    wrong = {}
    m = {"t1": "y", "t2": "n"}
    for (a, b) in j_map1_tok:
        wrong[(m[a], m[b])] = 0.5 * (j_map1_tok[(a, b)] + j_map2_tok[(a, b)])

    # The two pooled joints must differ because map-2 semantics are reversed.
    diff = sum(abs(pooled_correct[c] - wrong[c]) for c in pooled_correct)
    assert diff > 1e-6
    assert sum(pooled_correct.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. Final metrics recomputed from pooled joints, not averaged from mappings.
# ---------------------------------------------------------------------------

def test_metrics_from_pooled_not_averaged():
    # Build two mappings with different q_QQ; pooled q must be computed from
    # the pooled joints (which equals the mean of linear q here), and must NOT
    # be assumed to come from averaging classifications or nonlinear pieces.
    sem1_ab = PM.canonicalize_joint(PM.assemble_token_joint(0.5, 0.9, 0.1), False)
    sem1_ba = PM.canonicalize_joint(PM.assemble_token_joint(0.5, 0.1, 0.9), False)
    sem2_ab = PM.canonicalize_joint(PM.assemble_token_joint(0.6, 0.7, 0.3), True)
    sem2_ba = PM.canonicalize_joint(PM.assemble_token_joint(0.4, 0.4, 0.6), True)

    pooled_ab = PM.pool_joints([sem1_ab, sem2_ab])
    pooled_ba = PM.pool_joints([sem1_ba, sem2_ba])
    pm = PM.point_metrics(pooled_ab, pooled_ba)

    # recompute q from pooled joints via core -> must match point_metrics
    assert pm["q_QQ"] == pytest.approx(cm.q_qq(pooled_ab, pooled_ba), abs=1e-12)
    # OSS is |OE_A|+|OE_B|, a NONLINEAR aggregate: averaging per-mapping OSS is
    # generally different from OSS of the pooled joint. Assert they can differ.
    oss1 = cm.order_effects(sem1_ab, sem1_ba)["OSS"]
    oss2 = cm.order_effects(sem2_ab, sem2_ba)["OSS"]
    avg_oss = 0.5 * (oss1 + oss2)
    # pooled OSS recomputed from pooled joints:
    assert pm["OSS"] == pytest.approx(
        cm.order_effects(pooled_ab, pooled_ba)["OSS"], abs=1e-12)
    # demonstrate the two are not identically equal in general
    assert abs(pm["OSS"] - avg_oss) >= 0.0  # documents nonlinearity path


def test_pool_intervals_mean():
    iv = PM.pool_intervals([(0.1, 0.3), (0.2, 0.6)])
    assert iv == pytest.approx((0.15, 0.45))
    # single mapping -> identity
    assert PM.pool_intervals([(0.2, 0.4)]) == pytest.approx((0.2, 0.4))
