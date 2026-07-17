# pilot_items.py
# ex18 Minimal Pilot v1.2: item loading, label schemes, mapping generation,
# and multi-turn message construction.
#
# Frozen spec: ex18_minimal_pilot_spec_v1_2. Item text is verbatim from
# ex18_item_sets_draft_v0_6.md (stored in items/pilot_v1_2.json).
#
# Label schemes:
#   S1/S2: original response options mapped to neutral output tokens A/B,
#          executed under two counterbalancing mappings:
#            map-1: A = option_1, B = option_2
#            map-2: A = option_2, B = option_1
#          The same mapping is used for both AB and BA orders and for every
#          first-question and branch prompt.
#   S3   : canonical Yes/No labels, no A/B counterbalancing.
#
# Message rules (frozen):
#   - System prompt T-1 per scheme.
#   - S1/S2: every user turn includes the current mapping as plain text
#     "A = <option>" / "B = <option>" lines, then the question.
#   - Historical framing header: only s1-05 and s1-11, once per trajectory,
#     immediately before the first question, not repeated before the second.
#   - Vignette (S3): first user turn only, not repeated.
#   - Branch insertion: append an assistant message whose content is exactly
#     the accepted canonical label token, then the next user question.
#
# ASCII only. English comments. No hand-assembled chat-template strings: all
# rendering happens through tokenizer.apply_chat_template downstream.

import json

from src.logprob_backend import single_token_added

SYSTEM_PROMPT_AB = (
    "You are asked a series of judgment questions. For each question, answer "
    'with exactly one letter: "A" or "B". Do not explain.'
)
SYSTEM_PROMPT_YESNO = (
    "You are asked a series of judgment questions. For each question, answer "
    'with exactly one word: "Yes" or "No". Do not explain.'
)


# ---------------------------------------------------------------------------
# Item loading
# ---------------------------------------------------------------------------

def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def emit_tokens(item):
    """The two output token strings for an item's label scheme.
    t1 = the token for option_1 under map-1 (A) or the Yes label;
    t2 = the token for option_2 under map-1 (B) or the No label."""
    if item["label_scheme"] == "AB":
        return "A", "B"
    return "Yes", "No"


def get_mappings(item):
    """Return the list of label mappings to execute for an item.

    Each mapping dict fields:
      name              : "map-1" | "map-2" | "yesno"
      scheme            : "AB" | "YesNo"
      t1, t2            : emit token strings (fixed per scheme)
      swap_for_canonical: False if t1 == semantic option_1, True if t1 is
                          semantic option_2 (map-2). Used to canonicalize
                          token coordinates into semantic-option coordinates.
      prompt_A, prompt_B: plain-text semantic options shown in AB prompts.
    """
    t1, t2 = emit_tokens(item)
    if item["label_scheme"] == "YesNo":
        return [{
            "name": "yesno", "scheme": "YesNo", "t1": t1, "t2": t2,
            "swap_for_canonical": False, "prompt_A": None, "prompt_B": None,
        }]
    opt1, opt2 = item["option_1"], item["option_2"]
    return [
        {"name": "map-1", "scheme": "AB", "t1": t1, "t2": t2,
         "swap_for_canonical": False, "prompt_A": opt1, "prompt_B": opt2},
        {"name": "map-2", "scheme": "AB", "t1": t1, "t2": t2,
         "swap_for_canonical": True, "prompt_A": opt2, "prompt_B": opt1},
    ]


def system_prompt_for(item):
    # Per-item override (e.g. Track P T-1P with a substituted reference_year).
    # Falls back to the frozen T-1 scheme prompt when no override is present.
    if item.get("system_prompt"):
        return item["system_prompt"]
    return SYSTEM_PROMPT_AB if item["label_scheme"] == "AB" else SYSTEM_PROMPT_YESNO


# ---------------------------------------------------------------------------
# User-turn content construction
# ---------------------------------------------------------------------------

def _mapping_lines(mapping):
    return "A = %s\nB = %s" % (mapping["prompt_A"], mapping["prompt_B"])


def _question_text(item, order, which):
    """which in {'first','second'}. Order AB -> first=Q_A, second=Q_B."""
    if which == "first":
        return item["q_a"] if order == "AB" else item["q_b"]
    return item["q_b"] if order == "AB" else item["q_a"]


def first_user_content(item, order, mapping):
    """First user turn: [historical header?][vignette?][mapping lines?][Q]."""
    parts = []
    if item.get("historical_header"):
        parts.append(item["historical_header"])
    if item.get("vignette"):
        parts.append(item["vignette"])
    if mapping["scheme"] == "AB":
        parts.append(_mapping_lines(mapping))
    parts.append(_question_text(item, order, "first"))
    return "\n\n".join(parts)


def second_user_content(item, order, mapping):
    """Second user turn: [mapping lines?][Q]. No header, no vignette (they
    persist through history and must not be repeated)."""
    parts = []
    if mapping["scheme"] == "AB":
        parts.append(_mapping_lines(mapping))
    parts.append(_question_text(item, order, "second"))
    return "\n\n".join(parts)


def first_messages(item, order, mapping):
    return [
        {"role": "system", "content": system_prompt_for(item)},
        {"role": "user", "content": first_user_content(item, order, mapping)},
    ]


def second_messages(item, order, mapping, branch_token):
    """branch_token is the accepted first-answer token string (t1 or t2)."""
    return [
        {"role": "system", "content": system_prompt_for(item)},
        {"role": "user", "content": first_user_content(item, order, mapping)},
        {"role": "assistant", "content": branch_token},
        {"role": "user", "content": second_user_content(item, order, mapping)},
    ]


# ---------------------------------------------------------------------------
# Canonical single-token label validation (rendered-context)
# ---------------------------------------------------------------------------

def validate_emit_tokens(encode, rendered, t1, t2):
    """Validate that both emit tokens add exactly one token when appended to
    the rendered prompt. Try bare and leading-space forms; both members of a
    form must be single-token. Return the first valid form.

    encode(text) -> list[int] (no special tokens added).
    Returns {passed, variant, token_ids{t1,t2}, per_candidate}.
    """
    base_ids = encode(rendered)
    variants = [
        {"name": "bare", "t1": t1, "t2": t2},
        {"name": "leading_space", "t1": " " + t1, "t2": " " + t2},
    ]
    per_candidate = {}
    chosen = None
    for var in variants:
        ok_pair = True
        ids = {}
        for key in ("t1", "t2"):
            full_ids = encode(rendered + var[key])
            is_single, added = single_token_added(base_ids, full_ids)
            per_candidate["%s:%s" % (var["name"], key)] = {
                "label": var[key], "is_single": is_single, "added_id": added,
                "n_tokens_added": len(full_ids) - len(base_ids),
            }
            if is_single:
                ids[key] = added
            else:
                ok_pair = False
        if ok_pair and chosen is None:
            chosen = {"variant": var["name"], "token_ids": ids}
    return {
        "passed": chosen is not None,
        "variant": chosen["variant"] if chosen else None,
        "token_ids": chosen["token_ids"] if chosen else None,
        "per_candidate": per_candidate,
    }
