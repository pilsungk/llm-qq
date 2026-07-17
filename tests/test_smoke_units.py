# Unit tests for ex18 R11 smoke-test helpers.
# Small and model-free: joint normalization and label-validation logic.
# ASCII only.

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.metrics_ex18 import assemble_joint
from src.logprob_backend import single_token_added, validate_label_pair


# ---------------------------------------------------------------------------
# Joints assembled from binary-conditioned components must sum to 1.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pf,pyy,pyn",
    [
        (0.5, 0.5, 0.5),
        (0.9, 0.1, 0.8),
        (0.0, 0.3, 0.7),
        (1.0, 1.0, 0.0),
        (0.42, 0.13, 0.99),
    ],
)
def test_joint_sums_to_one(pf, pyy, pyn):
    joint = assemble_joint(pf, pyy, pyn)
    assert len(joint) == 4
    total = sum(joint.values())
    assert total == pytest.approx(1.0, abs=1e-12)
    for v in joint.values():
        assert -1e-12 <= v <= 1.0 + 1e-12


def test_joint_marginals_match_inputs():
    # First-answer marginal must equal the supplied P(first=y) (no dropping
    # of the first-answer marginal -- estimand correctness).
    pf = 0.73
    joint = assemble_joint(pf, 0.2, 0.6)
    p_first_y = joint[("y", "y")] + joint[("y", "n")]
    assert p_first_y == pytest.approx(pf, abs=1e-12)


# ---------------------------------------------------------------------------
# single_token_added: exactly one extra token appended at the end.
# ---------------------------------------------------------------------------

def test_single_token_added_true():
    ok, added = single_token_added([1, 2, 3], [1, 2, 3, 9])
    assert ok is True
    assert added == 9


def test_single_token_added_multi_token():
    ok, added = single_token_added([1, 2, 3], [1, 2, 3, 9, 10])
    assert ok is False
    assert added is None


def test_single_token_added_prefix_mismatch():
    ok, added = single_token_added([1, 2, 3], [1, 2, 4, 9])
    assert ok is False
    assert added is None


# ---------------------------------------------------------------------------
# validate_label_pair: choose the single-token variant with a stub encoder.
# ---------------------------------------------------------------------------

def _make_encode(vocab):
    """Return an encode(text) that maps the rendered prefix to a fixed id list
    and each known suffix label to appended ids per `vocab`."""
    PREFIX = "PROMPT"
    base = [100, 101, 102]

    def encode(text):
        assert text.startswith(PREFIX)
        suffix = text[len(PREFIX):]
        return base + vocab.get(suffix, [7, 7])  # unknown -> 2 tokens

    return "PROMPT", encode


def test_validate_prefers_bare_when_single_token():
    rendered, encode = _make_encode({"": [], "Yes": [500], "No": [501]})
    res = validate_label_pair(encode, rendered)
    assert res["passed"] is True
    assert res["variant"] == "bare"
    assert res["token_ids"] == {"y": 500, "n": 501}


def test_validate_falls_back_to_leading_space():
    # Bare "Yes"/"No" are multi-token; leading-space variants are single-token.
    rendered, encode = _make_encode(
        {"": [], "Yes": [1, 2], "No": [3, 4], " Yes": [600], " No": [601]}
    )
    res = validate_label_pair(encode, rendered)
    assert res["passed"] is True
    assert res["variant"] == "leading_space"
    assert res["token_ids"] == {"y": 600, "n": 601}


def test_validate_fails_when_no_single_token_pair():
    rendered, encode = _make_encode(
        {"": [], "Yes": [1, 2], "No": [3], " Yes": [4, 5], " No": [6, 7]}
    )
    res = validate_label_pair(encode, rendered)
    assert res["passed"] is False
    assert res["variant"] is None
    assert res["token_ids"] is None
