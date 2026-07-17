# Tests for envelopes, Gamma bounds, QQ/Gamma classifications, saturation,
# the other-mass gate, and rejection-trajectory counting.
# ASCII only.

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import pilot_items as PI
from src import pilot_metrics as PM
from src import pilot_runner as PR


# ---------------------------------------------------------------------------
# 10. Other-mass gate logic.
# ---------------------------------------------------------------------------

def test_other_mass_gate_pass():
    g = PM.other_mass_gate([1e-6, 5e-7, 2e-6])
    assert g["passed"] is True
    assert g["max_other_mass"] == pytest.approx(2e-6)


def test_other_mass_gate_fail_at_threshold():
    assert PM.other_mass_gate([0.005, 0.02])["passed"] is False
    # exactly 0.01 is NOT < 0.01 -> fail
    assert PM.other_mass_gate([0.01])["passed"] is False
    assert PM.other_mass_gate([0.0099])["passed"] is True


# ---------------------------------------------------------------------------
# 11. Saturation logic (max(p,1-p) > 0.99 in > 50% of positions).
# ---------------------------------------------------------------------------

def test_saturation_not_saturated_at_half():
    s = PM.saturation([0.999, 0.001, 0.5, 0.5])
    assert s["n_saturated"] == 2         # 0.999 and 0.001 both saturate
    assert s["frac_saturated"] == pytest.approx(0.5)
    assert s["saturated"] is False       # needs strictly > 0.5


def test_saturation_saturated_above_half():
    s = PM.saturation([0.999, 0.001, 0.995, 0.5])
    assert s["n_saturated"] == 3
    assert s["saturated"] is True


def test_saturation_symmetry_low_prob_counts():
    s = PM.saturation([0.002, 0.003])
    assert s["n_saturated"] == 2
    assert s["saturated"] is True


# ---------------------------------------------------------------------------
# 12. QQ and Gamma classifications on known synthetic fixtures.
# ---------------------------------------------------------------------------

def test_qq_classification_intervals():
    assert PM.classify_qq((0.0, 0.0)) == "SATISFIED"
    assert PM.classify_qq((-0.019, 0.019)) == "SATISFIED"
    assert PM.classify_qq((0.5, 0.5)) == "VIOLATED"
    assert PM.classify_qq((-0.8, -0.8)) == "VIOLATED"
    assert PM.classify_qq((-0.01, 0.05)) == "INDETERMINATE"


def test_gamma_classification():
    assert PM.classify_gamma(-0.5, -0.1) == "NONCONTEXTUAL"
    assert PM.classify_gamma(0.1, 0.5) == "CONTEXTUAL"
    assert PM.classify_gamma(-0.1, 0.1) == "INDETERMINATE"


def test_abs_bounds():
    assert PM.abs_bounds((-0.02, 0.03)) == (0.0, 0.03)      # spans zero
    assert PM.abs_bounds((0.1, 0.3)) == (0.1, 0.3)
    assert PM.abs_bounds((-0.3, -0.1)) == (0.1, 0.3)


def test_envelope_satisfied_noncontextual_fixture():
    # a=b=c=d symmetric fixed kernel -> q_QQ = 0 (Theorem 2), OSS > 0.
    raw_ab = {"p_first": (0.7, 0.3, 0.0),
              "p_second_given_y": (0.8, 0.2, 0.0),
              "p_second_given_n": (0.2, 0.8, 0.0)}
    raw_ba = {"p_first": (0.4, 0.6, 0.0),
              "p_second_given_y": (0.8, 0.2, 0.0),
              "p_second_given_n": (0.2, 0.8, 0.0)}
    q_iv, oea_iv, oeb_iv = PM.semantic_envelopes(raw_ab, raw_ba)
    assert q_iv[0] == pytest.approx(0.0, abs=1e-12)
    assert q_iv[1] == pytest.approx(0.0, abs=1e-12)
    gb = PM.gamma_bounds(q_iv, oea_iv, oeb_iv)
    assert PM.classify_qq(q_iv) == "SATISFIED"
    assert PM.classify_gamma(gb["gamma_lower"], gb["gamma_upper"]) == "NONCONTEXTUAL"


def test_envelope_violated_contextual_fixture():
    # constructed q_QQ = -0.8 with OSS = 0 -> certified contextual.
    raw_ab = {"p_first": (0.5, 0.5, 0.0),
              "p_second_given_y": (0.9, 0.1, 0.0),
              "p_second_given_n": (0.1, 0.9, 0.0)}
    raw_ba = {"p_first": (0.5, 0.5, 0.0),
              "p_second_given_y": (0.1, 0.9, 0.0),
              "p_second_given_n": (0.9, 0.1, 0.0)}
    q_iv, oea_iv, oeb_iv = PM.semantic_envelopes(raw_ab, raw_ba)
    assert q_iv[0] == pytest.approx(-0.8, abs=1e-12)
    assert oea_iv == pytest.approx((0.0, 0.0), abs=1e-12)
    assert oeb_iv == pytest.approx((0.0, 0.0), abs=1e-12)
    gb = PM.gamma_bounds(q_iv, oea_iv, oeb_iv)
    assert PM.classify_qq(q_iv) == "VIOLATED"
    assert gb["gamma_lower"] == pytest.approx(0.8, abs=1e-12)
    assert PM.classify_gamma(gb["gamma_lower"], gb["gamma_upper"]) == "CONTEXTUAL"


def test_gamma_bounds_conservative_order():
    # Gamma_lower <= Gamma_upper always.
    q_iv, oea_iv, oeb_iv = (0.1, 0.4), (-0.2, 0.3), (0.0, 0.1)
    gb = PM.gamma_bounds(q_iv, oea_iv, oeb_iv)
    assert gb["gamma_lower"] <= gb["gamma_upper"]


# ---------------------------------------------------------------------------
# 13. Rejection counts use accepted sequential joint trajectories.
# ---------------------------------------------------------------------------

class _FakeBackend:
    """Returns predetermined accepted labels per stage, distinguishing the
    first stage (2-message prompt) from branch stages (4-message prompt) by
    the assistant label content."""

    def __init__(self, first_labels, branch_seconds):
        self.first_labels = first_labels
        self.branch_seconds = branch_seconds  # {token_str: [labels]}

    def collect_accepted(self, messages, need, id_y=None, id_n=None,
                         batch=64, max_attempts=None):
        if len(messages) == 2:
            labels = self.first_labels[:need]
        else:
            branch_tok = messages[2]["content"]
            labels = self.branch_seconds[branch_tok][:need]
        stats = {"attempts": len(labels), "accepted": len(labels),
                 "other": 0, "shortfall": len(labels) < need}
        return labels, stats


def test_rejection_trajectory_counting():
    it = next(i for i in PI.load_items(
        os.path.join(ROOT, "items", "pilot_v1_2.json"))["items"]
        if i["id"] == "s1-01")
    mp = PI.get_mappings(it)[0]           # map-1: t1=A, t2=B
    fake = _FakeBackend(
        first_labels=["y", "y", "n"],
        branch_seconds={"A": ["y", "n"], "B": ["n"]},
    )
    counts, stats = PR.rejection_joint(fake, it, "AB", mp, id_t1=1, id_t2=2, n_rej=3)
    # trajectories: (y,y),(y,n) from the A (=t1) branch; (n,n) from the B branch
    assert counts[("y", "y")] == 1
    assert counts[("y", "n")] == 1
    assert counts[("n", "y")] == 0
    assert counts[("n", "n")] == 1
    assert stats["accepted_joint_samples"] == 3
    assert stats["first_branch_counts"] == {"t1": 2, "t2": 1}
