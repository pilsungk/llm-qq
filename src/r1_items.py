# r1_items.py
# ex18 Redesign Run R1 (Track P Minimal) item set.
#
# Governing spec: docs/ex18_redesign_run_spec_R1.md (single approved rerun).
# The ONLY change from the frozen v1.2 pilot is the system prompt: Track M
# (T-1) is replaced by Track P (T-1P, respondent simulation) with a per-item
# {reference_year} substitution. The historical framing header is NOT used
# (the persona already fixes the time point; avoid double context).
#
# Item texts and A/B option pairs are taken verbatim from
# items/pilot_v1_2.json -- no rewording. Restricted to the 8 pairs whose
# reference_year is fixed.
#
# ASCII only. English comments.

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT_ITEMS_PATH = os.path.join(ROOT, "items", "pilot_v1_2.json")

# (item_id, reference_year), reference_year per spec Section 2.
R1_ITEMS = [
    ("s1-01", 1997),
    ("s1-02", 1995),
    ("s1-04", 1996),
    ("s1-06", 1979),
    ("s1-11", 2008),
    ("s1-12", 2012),
    ("s2-01", 1997),
    ("s2-02", 1995),
]

SYSTEM_PROMPT_ID = "T-1P"

# Track P system prompt (pre-registered T-1P). {reference_year} substituted.
T1P_TEMPLATE = (
    "Answer each question as an average adult American survey respondent in "
    "{reference_year} would have answered. For each question, answer with "
    "exactly one letter: A or B. Do not explain."
)


def t1p_prompt(reference_year):
    return T1P_TEMPLATE.format(reference_year=reference_year)


def load_r1_items(pilot_path=PILOT_ITEMS_PATH):
    """Build the 8 Track P items from the frozen pilot JSON.

    Each item keeps its verbatim texts/options and label_scheme, gains the
    fixed reference_year, gets a per-item T-1P system prompt, and has the
    historical framing header suppressed (set to None).
    """
    with open(pilot_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_id = {it["id"]: it for it in data["items"]}
    items = []
    for iid, year in R1_ITEMS:
        base = dict(by_id[iid])  # verbatim copy (texts + option pairs)
        assert base["label_scheme"] == "AB", "R1 items are all A/B scheme"
        base["reference_year"] = year
        base["historical_header"] = None  # Track P persona fixes the period
        base["vignette"] = None
        base["system_prompt"] = t1p_prompt(year)
        base["system_prompt_id"] = SYSTEM_PROMPT_ID
        items.append(base)
    return items


def reference_years():
    return {iid: year for iid, year in R1_ITEMS}
