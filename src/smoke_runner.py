# smoke_runner.py
# ex18 R11 smoke test end-to-end driver.
#
# Goal is NOT research results but three pipeline health gates on a real LLM:
#   G1  canonical single-token label validation
#   G2  raw `other` mass at every measured logprob position (PASS if < 0.01)
#   G3  binary_logprob vs binary_rejection CP consistency check per item/order
#
# Protocols exercised: binary_logprob (primary) and binary_rejection (check).
# Multi-turn sequential protocol only (SRS 4.2): question -> answer appended to
# history -> next question, both orders (AB, BA). Never both questions in one
# prompt.
#
# ASCII only. English comments. No gradient; one forward at a time.

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

import torch
import transformers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.metrics_ex18 import (  # noqa: E402  (path set above)
    assemble_joint,
    consistency_check,
    order_effects,
    q_qq,
)
from src.logprob_backend import (  # noqa: E402
    CANONICAL_DISPLAY,
    DECODING,
    LogprobBackend,
)

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SEED = 20260717

# Track M, template T-1 (verbatim; do not edit).
SYSTEM_PROMPT = (
    "You are asked a series of judgment questions. For each question, answer "
    'with exactly one word: "Yes" or "No". Do not explain.'
)

CELLS = [("y", "y"), ("y", "n"), ("n", "y"), ("n", "n")]


# ---------------------------------------------------------------------------
# Item / conversation construction
# ---------------------------------------------------------------------------

def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_user_content(item, order):
    """First user turn text. The vignette (if any) is prepended to the FIRST
    user turn ONLY (SRS / item draft s3 rule)."""
    q_first = item["q_a"] if order == "AB" else item["q_b"]
    vignette = item.get("vignette")
    if vignette:
        return vignette + "\n\n" + q_first
    return q_first


def _second_user_content(item, order):
    return item["q_b"] if order == "AB" else item["q_a"]


def messages_first(item, order):
    """Conversation up to the first generation position."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _first_user_content(item, order)},
    ]


def messages_second(item, order, branch):
    """Conversation up to the second generation position, given the first
    answer branch in {'y','n'}. Branch insertion = append an assistant message
    whose content is exactly the label text, then the next user question."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _first_user_content(item, order)},
        {"role": "assistant", "content": CANONICAL_DISPLAY[branch]},
        {"role": "user", "content": _second_user_content(item, order)},
    ]


# ---------------------------------------------------------------------------
# binary_logprob per (item, order)
# ---------------------------------------------------------------------------

def run_logprob_order(backend, item, order):
    r_first = backend.position_readout(messages_first(item, order))
    r_y = backend.position_readout(messages_second(item, order, "y"))
    r_n = backend.position_readout(messages_second(item, order, "n"))

    joint = assemble_joint(
        p_first_y=r_first["p_yes_cond"],
        p_second_y_given_first_y=r_y["p_yes_cond"],
        p_second_y_given_first_n=r_n["p_yes_cond"],
    )
    positions = {"first": r_first, "second_given_y": r_y, "second_given_n": r_n}
    return joint, positions


# ---------------------------------------------------------------------------
# binary_rejection per (item, order): n_rej accepted JOINT samples
# ---------------------------------------------------------------------------

def run_rejection_order(backend, item, order, n_rej):
    """Stage-wise rejection sampling. Returns (counts, stats).

    counts: {(first,second): accepted count}, summing to the number of
    accepted joint samples (<= n_rej if a stage hits its attempt cap).
    """
    first_labels, s1 = backend.collect_accepted(messages_first(item, order), n_rej)
    n_used = len(first_labels)
    n_y = first_labels.count("y")
    n_n = first_labels.count("n")

    sec_y, sy = ([], {"attempts": 0, "accepted": 0, "other": 0, "shortfall": False})
    sec_n, sn = ([], {"attempts": 0, "accepted": 0, "other": 0, "shortfall": False})
    if n_y > 0:
        sec_y, sy = backend.collect_accepted(messages_second(item, order, "y"), n_y)
    if n_n > 0:
        sec_n, sn = backend.collect_accepted(messages_second(item, order, "n"), n_n)

    counts = {cell: 0 for cell in CELLS}
    iy = 0
    in_ = 0
    n_pairs = min(n_used, len(sec_y) + len(sec_n))
    for f in first_labels:
        if f == "y":
            if iy >= len(sec_y):
                break
            s = sec_y[iy]
            iy += 1
        else:
            if in_ >= len(sec_n):
                break
            s = sec_n[in_]
            in_ += 1
        counts[(f, s)] += 1

    stats = {
        "target_n_rej": n_rej,
        "accepted_joint_samples": sum(counts.values()),
        "stage_first": s1,
        "stage_second_given_y": sy,
        "stage_second_given_n": sn,
        "first_branch_counts": {"y": n_y, "n": n_n},
    }
    return counts, stats


# ---------------------------------------------------------------------------
# Consistency gate with the SRS G2 retry (double to 400, rerun once)
# ---------------------------------------------------------------------------

def _fmt_cell(cell):
    return "%s%s" % cell


def consistency_with_retry(backend, item, order, joint):
    counts, rstats = run_rejection_order(backend, item, order, n_rej=200)
    ok, n_rej, details = consistency_check(joint, counts)
    retry_status = "not_needed"
    if not ok:
        counts, rstats = run_rejection_order(backend, item, order, n_rej=400)
        ok, n_rej, details = consistency_check(joint, counts)
        retry_status = "retried_400_pass" if ok else "retried_400_FAIL_investigate"
    detail_ser = {
        _fmt_cell(cell): {
            "ci": list(details[cell]["ci"]),
            "p": details[cell]["p"],
            "inside": bool(details[cell]["inside"]),
        }
        for cell in CELLS
    }
    return {
        "ok": bool(ok),
        "n_rej": n_rej,
        "retry_status": retry_status,
        "counts": {_fmt_cell(c): counts[c] for c in CELLS},
        "details": detail_ser,
        "rejection_stats": rstats,
    }


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------

def build_metadata(backend, items_path, label_result, template_ok):
    return {
        "phase": "R11_smoke",
        "status": "started",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "model_id": backend.model_id,
        "model_revision": getattr(backend.model.config, "_name_or_path", None),
        "tokenizer_id": backend.model_id,
        "tokenizer_revision": None,
        "chat_template_sha256": backend.chat_template_sha256(),
        "chat_template_no_thinking_tags": template_ok,
        "quantization": "none",
        "torch_dtype": "bfloat16",
        "system_prompt": SYSTEM_PROMPT,
        "track": "M",
        "template_id": "T-1",
        "decoding": DECODING,
        "seed": SEED,
        "canonical_labels": {
            "passed": label_result["passed"],
            "variant": label_result["variant"],
            "token_ids": label_result["token_ids"],
            "display": CANONICAL_DISPLAY,
            "per_candidate": label_result["per_candidate"],
        },
        "event_definition": (
            "first generated token equals a canonical label token id; "
            "otherwise rejected and resampled (SRS 4.4.1 single-token model)"
        ),
        "control_arm": "none (Yes/No main, T-1 only)",
        "items_file": os.path.relpath(items_path, ROOT),
        "items_file_sha256": sha256_file(items_path),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "results": [],
    }


def write_record(record, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=True)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(record, g1, g2_max_other, g2_pass, g3_pass):
    print("")
    print("=" * 72)
    print("ex18 R11 SMOKE TEST SUMMARY")
    print("=" * 72)
    lab = record["canonical_labels"]
    print("G1  canonical single-token validation : %s" % ("PASS" if g1 else "FAIL"))
    if lab["passed"]:
        print("     variant=%s  token_ids=%s" % (lab["variant"], lab["token_ids"]))
    print("G2  max other mass over all positions : %.6e  (%s, thr < 1e-2)"
          % (g2_max_other, "PASS" if g2_pass else "FAIL"))
    print("G3  CP consistency (all item/order)   : %s" % ("PASS" if g3_pass else "FAIL"))
    print("-" * 72)

    for res in record["results"]:
        print("item %-6s [%s]" % (res["id"], res["stratum"]))
        for order in ("AB", "BA"):
            m = res["orders"][order]["metrics"]
            c = res["orders"][order]["consistency"]
            print("   %s  OE_A=%+.4f OE_B=%+.4f OSS=%.4f q_QQ=%+.4f | "
                  "consistency=%s (n_rej=%d, %s)"
                  % (order, m["OE_A"], m["OE_B"], m["OSS"], m["q_QQ"],
                     "PASS" if c["ok"] else "FAIL", c["n_rej"], c["retry_status"]))
        # per-position raw / other mass
        for order in ("AB", "BA"):
            pos = res["orders"][order]["positions"]
            masses = " ".join(
                "%s:raw=%.5f/other=%.2e" % (k, pos[k]["raw_mass"], pos[k]["other_mass"])
                for k in ("first", "second_given_y", "second_given_n")
            )
            print("     %s raw masses: %s" % (order, masses))
    print("-" * 72)

    # s1-04 refusal indicator: rejection accept vs attempt at the first stage
    for res in record["results"]:
        if res["id"] != "s1-04":
            continue
        print("REFUSAL INDICATOR (s1-04, sensitive_stress_test):")
        for order in ("AB", "BA"):
            st = res["orders"][order]["consistency"]["rejection_stats"]["stage_first"]
            ratio = (st["accepted"] / st["attempts"]) if st["attempts"] else float("nan")
            print("   %s first-stage: accepted=%d attempts=%d other=%d accept_ratio=%.4f"
                  % (order, st["accepted"], st["attempts"], st["other"], ratio))
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--items", default=os.path.join(ROOT, "items", "smoke.json"))
    args = ap.parse_args()

    torch.manual_seed(SEED)

    print("[load] %s (bfloat16, cuda)" % args.model)
    backend = LogprobBackend(args.model)

    # -- thinking-tag guard on the rendered template (non-thinking variant) --
    probe_msgs = messages_first(
        {"id": "_probe", "q_a": "Do you agree?", "q_b": "Do you agree?"}, "AB"
    )
    rendered_probe = backend.render(probe_msgs)
    template_ok = ("<think>" not in rendered_probe) and ("</think>" not in rendered_probe)
    if not template_ok:
        print("[FATAL] rendered chat template contains thinking tags; stopping.")
        print(rendered_probe)
        sys.exit(2)
    print("[ok] no thinking tags in rendered template")

    # -- G1: canonical single-token label validation ------------------------
    label_result = backend.validate_labels(probe_msgs)
    g1 = label_result["passed"]

    items = load_items(args.items)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    record_path = os.path.join(ROOT, "runs", "smoke_%s.json" % stamp)

    # Write metadata BEFORE any results (audit-record requirement).
    record = build_metadata(backend, args.items, label_result, template_ok)
    write_record(record, record_path)
    print("[audit] metadata written: %s" % record_path)

    if not g1:
        print("[FATAL] G1 canonical single-token validation FAILED; stopping.")
        print(json.dumps(label_result["per_candidate"], indent=2))
        record["status"] = "aborted_G1_fail"
        write_record(record, record_path)
        sys.exit(3)

    print("[ok] G1 labels: variant=%s token_ids=%s"
          % (label_result["variant"], label_result["token_ids"]))

    # -- Per item / order ---------------------------------------------------
    g2_max_other = 0.0
    g3_pass = True
    for item in items["items"]:
        print("[item] %s (%s)" % (item["id"], item["stratum"]))
        item_res = {"id": item["id"], "stratum": item["stratum"], "orders": {}}
        joints = {}
        positions_by_order = {}
        for order in ("AB", "BA"):
            print("   [logprob] order %s" % order)
            joint, positions = run_logprob_order(backend, item, order)
            joints[order] = joint
            positions_by_order[order] = positions
            for pos in positions.values():
                g2_max_other = max(g2_max_other, pos["other_mass"])

        metrics_ab = order_effects(joints["AB"], joints["BA"])
        q = q_qq(joints["AB"], joints["BA"])

        for order in ("AB", "BA"):
            print("   [rejection+consistency] order %s" % order)
            cons = consistency_with_retry(backend, item, order, joints[order])
            if not cons["ok"]:
                g3_pass = False
            item_res["orders"][order] = {
                "joint": {"%s%s" % k: v for k, v in joints[order].items()},
                "positions": {
                    name: {
                        "p_yes": p["p_yes"], "p_no": p["p_no"],
                        "raw_mass": p["raw_mass"], "other_mass": p["other_mass"],
                        "p_yes_cond": p["p_yes_cond"], "top5": p["top5"],
                    }
                    for name, p in positions_by_order[order].items()
                },
                "metrics": {
                    "OE_A": metrics_ab["OE_A"], "OE_B": metrics_ab["OE_B"],
                    "OSS": metrics_ab["OSS"], "q_QQ": q,
                },
                "consistency": cons,
            }
        record["results"].append(item_res)
        write_record(record, record_path)  # persist incrementally

    g2_pass = g2_max_other < 0.01

    record["status"] = "completed"
    record["gates"] = {
        "G1_label_single_token": g1,
        "G2_max_other_mass": g2_max_other,
        "G2_pass": g2_pass,
        "G3_consistency_pass": g3_pass,
    }
    write_record(record, record_path)

    print_summary(record, g1, g2_max_other, g2_pass, g3_pass)
    print("[audit] record: %s" % record_path)


if __name__ == "__main__":
    main()
