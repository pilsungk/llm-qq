# Unit tests for ex18 Minimal Pilot v1.2 (message construction, label
# validation, joint normalization, key conventions, audit IO).
# Model-free. ASCII only.

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import pilot_items as PI
from src import pilot_metrics as PM

ITEMS = PI.load_items(os.path.join(ROOT, "items", "pilot_v1_2.json"))


def item(iid):
    return next(it for it in ITEMS["items"] if it["id"] == iid)


# ---------------------------------------------------------------------------
# Item set integrity: exactly the 18 frozen ids, s1-08 excluded.
# ---------------------------------------------------------------------------

def test_item_set_is_frozen_18():
    ids = [it["id"] for it in ITEMS["items"]]
    assert len(ids) == 18
    assert "s1-08" not in ids
    expected = {
        "s1-01", "s1-02", "s1-04", "s1-05", "s1-06", "s1-07",
        "s1-09", "s1-10", "s1-11", "s1-12",
        "s2-01", "s2-02", "s2-09",
        "s3-01", "s3-03", "s3-08", "s3-09", "s3-11",
    }
    assert set(ids) == expected


# ---------------------------------------------------------------------------
# 1 + 2. Joint normalization and AB/BA key conventions.
# ---------------------------------------------------------------------------

def test_token_joint_sums_to_one():
    j = PM.assemble_token_joint(0.6, 0.3, 0.8)
    assert set(j.keys()) == {(a, b) for a in ("t1", "t2") for b in ("t1", "t2")}
    assert sum(j.values()) == pytest.approx(1.0, abs=1e-12)


def test_pooled_joint_sums_to_one():
    j1 = PM.canonicalize_joint(PM.assemble_token_joint(0.6, 0.3, 0.8), False)
    j2 = PM.canonicalize_joint(PM.assemble_token_joint(0.4, 0.7, 0.2), True)
    pooled = PM.pool_joints([j1, j2])
    assert sum(pooled.values()) == pytest.approx(1.0, abs=1e-12)
    assert set(pooled.keys()) == {(a, b) for a in "yn" for b in "yn"}


def test_ab_ba_key_convention():
    # joint_ab keyed (a_answer, b_answer); joint_ba keyed (b_answer, a_answer).
    # order_effects/q_qq from core must consume these directly.
    from core import metrics_ex18 as cm
    jab = cm.assemble_joint(0.5, 0.8, 0.2)   # keys (a,b)
    jba = cm.assemble_joint(0.5, 0.8, 0.2)   # keys (b,a)
    oe = cm.order_effects(jab, jba)
    # symmetric inputs -> zero order effect
    assert oe["OE_A"] == pytest.approx(0.0, abs=1e-12)
    assert oe["OE_B"] == pytest.approx(0.0, abs=1e-12)
    assert cm.q_qq(jab, jba) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Canonical label validation: one-token vs multi-token detection.
# ---------------------------------------------------------------------------

def _stub_encode(vocab, base=(10, 11, 12), prefix="P"):
    def encode(text):
        assert text.startswith(prefix)
        suffix = text[len(prefix):]
        return list(base) + vocab.get(suffix, [9, 9])
    return encode


def test_validate_emit_tokens_bare_single():
    enc = _stub_encode({"": [], "A": [65], "B": [66]})
    res = PI.validate_emit_tokens(enc, "P", "A", "B")
    assert res["passed"] and res["variant"] == "bare"
    assert res["token_ids"] == {"t1": 65, "t2": 66}


def test_validate_emit_tokens_multi_token_fails_bare_then_space():
    # bare A/B are multi-token; leading-space single-token.
    enc = _stub_encode({"": [], "A": [1, 2], "B": [3, 4], " A": [70], " B": [71]})
    res = PI.validate_emit_tokens(enc, "P", "A", "B")
    assert res["passed"] and res["variant"] == "leading_space"
    assert res["token_ids"] == {"t1": 70, "t2": 71}


def test_validate_emit_tokens_no_valid_pair():
    enc = _stub_encode({"": [], "A": [1, 2], "B": [3], " A": [4, 5], " B": [6, 7]})
    res = PI.validate_emit_tokens(enc, "P", "A", "B")
    assert res["passed"] is False
    assert res["token_ids"] is None


# ---------------------------------------------------------------------------
# 7. Historical header: once, only before the first question.
# ---------------------------------------------------------------------------

def test_historical_header_first_turn_only():
    it = item("s1-05")
    mp = PI.get_mappings(it)[0]
    header = it["historical_header"]
    assert header
    first = PI.first_user_content(it, "AB", mp)
    second = PI.second_user_content(it, "AB", mp)
    assert first.count(header) == 1
    assert header not in second
    # symmetric in BA order
    first_ba = PI.first_user_content(it, "BA", mp)
    assert first_ba.count(header) == 1
    assert header not in PI.second_user_content(it, "BA", mp)


def test_non_header_item_has_no_header():
    it = item("s1-01")
    mp = PI.get_mappings(it)[0]
    # no historical framing sentence present
    assert "survey conducted" not in PI.first_user_content(it, "AB", mp)


# ---------------------------------------------------------------------------
# 8. Vignette: once, only in the first question turn.
# ---------------------------------------------------------------------------

def test_vignette_first_turn_only():
    it = item("s3-01")
    mp = PI.get_mappings(it)[0]
    vig = it["vignette"]
    assert vig
    assert PI.first_user_content(it, "AB", mp).count(vig) == 1
    assert vig not in PI.second_user_content(it, "AB", mp)
    assert PI.first_user_content(it, "BA", mp).count(vig) == 1
    assert vig not in PI.second_user_content(it, "BA", mp)


# ---------------------------------------------------------------------------
# 9. Same mapping used in AB and BA orders (mapping lines identical).
# ---------------------------------------------------------------------------

def test_same_mapping_lines_in_both_orders():
    it = item("s1-09")  # Approve/Disapprove
    for mp in PI.get_mappings(it):
        line = "A = %s\nB = %s" % (mp["prompt_A"], mp["prompt_B"])
        assert line in PI.first_user_content(it, "AB", mp)
        assert line in PI.first_user_content(it, "BA", mp)
        assert line in PI.second_user_content(it, "AB", mp)
        assert line in PI.second_user_content(it, "BA", mp)


def test_map1_map2_swap_options():
    it = item("s1-09")
    m1, m2 = PI.get_mappings(it)
    assert (m1["prompt_A"], m1["prompt_B"]) == (it["option_1"], it["option_2"])
    assert (m2["prompt_A"], m2["prompt_B"]) == (it["option_2"], it["option_1"])
    assert m1["swap_for_canonical"] is False
    assert m2["swap_for_canonical"] is True


def test_s3_single_yesno_mapping():
    it = item("s3-01")
    maps = PI.get_mappings(it)
    assert len(maps) == 1
    assert maps[0]["scheme"] == "YesNo"
    assert (maps[0]["t1"], maps[0]["t2"]) == ("Yes", "No")


def test_branch_message_uses_exact_label_token():
    it = item("s1-01")
    mp = PI.get_mappings(it)[0]
    msgs = PI.second_messages(it, "AB", mp, mp["t1"])
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == mp["t1"]
    assert msgs[3]["role"] == "user"


# ---------------------------------------------------------------------------
# 14. Audit JSON incremental write + reload.
# ---------------------------------------------------------------------------

def test_audit_json_incremental(tmp_path):
    path = tmp_path / "audit.json"
    rec = {"status": "started", "results": []}
    with open(path, "w") as f:
        json.dump(rec, f)
    # append incrementally
    for k in range(3):
        rec["results"].append({"id": "item-%d" % k})
        with open(path, "w") as f:
            json.dump(rec, f)
        reloaded = json.load(open(path))
        assert len(reloaded["results"]) == k + 1
    rec["status"] = "completed"
    with open(path, "w") as f:
        json.dump(rec, f)
    final = json.load(open(path))
    assert final["status"] == "completed"
    assert len(final["results"]) == 3
