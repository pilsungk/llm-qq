# Unit tests for ex18 Redesign Run R1 (Track P) item construction.
# Model-free. ASCII only.

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import pilot_items as PI
from src import r1_items as R1

PILOT = json.load(open(os.path.join(ROOT, "items", "pilot_v1_2.json")))
PILOT_BY_ID = {it["id"]: it for it in PILOT["items"]}


def test_r1_has_exactly_8_items_with_fixed_years():
    items = R1.load_r1_items()
    assert len(items) == 8
    got = {it["id"]: it["reference_year"] for it in items}
    assert got == {
        "s1-01": 1997, "s1-02": 1995, "s1-04": 1996, "s1-06": 1979,
        "s1-11": 2008, "s1-12": 2012, "s2-01": 1997, "s2-02": 1995,
    }


def test_r1_texts_and_options_verbatim():
    for it in R1.load_r1_items():
        base = PILOT_BY_ID[it["id"]]
        assert it["q_a"] == base["q_a"]
        assert it["q_b"] == base["q_b"]
        assert it["option_1"] == base["option_1"]
        assert it["option_2"] == base["option_2"]
        assert it["label_scheme"] == "AB"


def test_r1_system_prompt_is_t1p_with_year():
    for it in R1.load_r1_items():
        sp = it["system_prompt"]
        assert it["system_prompt_id"] == "T-1P"
        assert str(it["reference_year"]) in sp
        assert "average adult American survey respondent" in sp
        assert "exactly one letter: A or B" in sp
        # exact template match
        assert sp == R1.T1P_TEMPLATE.format(reference_year=it["reference_year"])


def test_r1_no_historical_header_anywhere():
    for it in R1.load_r1_items():
        assert it.get("historical_header") is None
        mp = PI.get_mappings(it)[0]
        first = PI.first_user_content(it, "AB", mp)
        second = PI.second_user_content(it, "AB", mp)
        # no framing header sentence in either turn
        assert "survey conducted" not in first
        assert "survey conducted" not in second


def test_r1_system_prompt_used_in_messages():
    it = R1.load_r1_items()[0]  # s1-01, 1997
    mp = PI.get_mappings(it)[0]
    msgs = PI.first_messages(it, "AB", mp)
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == it["system_prompt"]
    assert "1997" in msgs[0]["content"]
    # mapping lines still present in the user turn
    assert ("A = %s" % mp["prompt_A"]) in msgs[1]["content"]


def test_r1_s1_11_header_suppressed_even_though_pilot_had_one():
    # s1-11 has a historical_header in the pilot; R1 must suppress it.
    assert PILOT_BY_ID["s1-11"]["historical_header"]
    it = next(i for i in R1.load_r1_items() if i["id"] == "s1-11")
    assert it["historical_header"] is None
    mp = PI.get_mappings(it)[0]
    assert "Cold War" not in PI.first_user_content(it, "AB", mp)
    assert "2008" in it["system_prompt"]  # year lives in the persona prompt
