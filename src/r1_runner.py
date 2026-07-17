# r1_runner.py
# ex18 Redesign Run R1 (Track P Minimal) driver.
#
# Governing spec: docs/ex18_redesign_run_spec_R1.md. Single approved rerun.
# Reuses the frozen pilot infrastructure verbatim (measurement, canonicalize
# -> pool, envelopes, Gamma, gates, spot check). The ONLY design change is the
# Track P T-1P system prompt with a per-item reference_year (r1_items.py);
# the historical framing header is NOT applied.
#
# M1 only (Qwen3-4B-Instruct-2507, bf16). M2 remains execution-blocked and is
# recorded as such (unchanged from the pilot).
#
# ASCII only. English comments. No gradients (backend uses torch.no_grad()).

import datetime as _dt
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import pilot_items as PI  # noqa: E402
from src import pilot_metrics as PM  # noqa: E402
from src import pilot_runner as PR  # noqa: E402
from src import r1_items as R1  # noqa: E402
from src.logprob_backend import DECODING, LogprobBackend  # noqa: E402

SEED = 20260719
TRACK = "P"
SPOT_ITEMS = ("s1-01",)  # spec Section 3: s1-01 x 2 orders only
BASELINE_TRACK_M = os.path.join(
    ROOT, "runs", "pilot_v1_2_20260717_1247_qwen3_4b.json")
SPEC_PATH = os.path.join(ROOT, "docs", "ex18_redesign_run_spec_R1.md")
COND = PR.MODEL_CONDITIONS["M1"]


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    return {"python": SEED, "numpy": SEED, "torch": SEED}


def run():
    items = R1.load_r1_items()
    print("[load] %s (bf16) -- Track P (T-1P)" % COND["model_id"])
    backend = LogprobBackend(COND["model_id"])

    # thinking-tag guard under the Track P prompt
    probe_mp = PI.get_mappings(items[0])[0]
    rendered = backend.render(PI.first_messages(items[0], "AB", probe_mp))
    thinking = ("<think>" in rendered) or ("</think>" in rendered)
    if thinking:
        raise RuntimeError("STOP: rendered chat template contains thinking tags")
    print("[ok] no thinking tags in rendered Track P template")

    # G1 -- A/B single-token validation under the new system prompt
    g1_pass, per_scheme, scheme_ids = PR.validate_gate1(backend, items)
    print("[G1] pass=%s schemes=%s" % (g1_pass, list(per_scheme.keys())))
    if not g1_pass:
        raise RuntimeError(
            "STOP: G1 A/B single-token validation FAILED under Track P prompt")

    seeds = set_seeds()
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    record_path = os.path.join(
        ROOT, "runs", "redesign_R1_trackP_%s_%s.json" % (stamp, COND["slug"]))

    metadata = {
        "phase": "redesign_run_R1_trackP",
        "condition_key": "M1",
        "track": TRACK,
        "status": "started",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "spec_id": "ex18_redesign_run_spec_R1",
        "spec_sha256": PR.sha256_file(SPEC_PATH) if os.path.exists(SPEC_PATH) else None,
        "baseline_track_m_record": os.path.relpath(BASELINE_TRACK_M, ROOT),
        "items_file": "items/pilot_v1_2.json (subset via src/r1_items.py)",
        "items_file_sha256": PR.sha256_file(R1.PILOT_ITEMS_PATH),
        "model_id": COND["model_id"],
        "model_revision": getattr(backend.model.config, "_name_or_path", None),
        "tokenizer_id": COND["model_id"],
        "tokenizer_revision": None,
        "chat_template_sha256": backend.chat_template_sha256(),
        "chat_template_no_thinking_tags": (not thinking),
        "model_dtype": COND["dtype"],
        "quantization": None,
        "quantization_kind": None,
        "condition_reason": COND["reason"] + " | Track P redesign rerun R1",
        "cuda": PR.cuda_info(),
        "versions": PR.package_versions(),
        "system_prompt_id": R1.SYSTEM_PROMPT_ID,
        "system_prompt_template": R1.T1P_TEMPLATE,
        "per_item_system_prompts": {it["id"]: it["system_prompt"] for it in items},
        "reference_years": {it["id"]: it["reference_year"] for it in items},
        "historical_header_used": False,
        "template_id": "T-1P",
        "decoding": DECODING,
        "decoding_note": "binary_logprob deterministic next-token probabilities; "
                         "binary_rejection do_sample per DECODING.",
        "seeds": seeds,
        "canonical_labels": {
            scheme: {
                "passed": r["passed"], "variant": r["variant"],
                "token_ids": r["token_ids"], "per_candidate": r["per_candidate"],
            } for scheme, r in per_scheme.items()
        },
        "git": PR.git_info(),
        "gates": {"G1": g1_pass},
        "m2_status": "execution_blocked (HF access + bitsandbytes dependency + "
                     "sm_120 risk); recorded as in the v1.2 pilot",
        "results": [],
    }
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)
    print("[audit] metadata written: %s" % record_path)

    max_other = 0.0
    g3_present = False
    g3_final_pass = True
    for item in items:
        do_spot = item["id"] in SPOT_ITEMS
        print("   [item] %s (year=%s)%s"
              % (item["id"], item["reference_year"], " +spot" if do_spot else ""))
        item_result, item_max_other = PR.evaluate_item(
            backend, item, scheme_ids, do_spot)
        item_result["reference_year"] = item["reference_year"]
        item_result["track"] = TRACK
        max_other = max(max_other, item_max_other)
        if do_spot:
            g3_present = True
            for order in PR.ORDERS:
                if not item_result["spot_check"][order]["final_pass"]:
                    g3_final_pass = False
        metadata["results"].append(item_result)
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=True)

    g2 = PM.other_mass_gate([p["other_mass"]
                             for it in metadata["results"]
                             for p in it["positions"]])
    metadata["gates"] = {
        "G1": g1_pass,
        "G2_pass": g2["passed"], "G2_max_other_mass": g2["max_other_mass"],
        "G3_present": g3_present, "G3_pass": g3_final_pass,
    }
    metadata["status"] = "completed"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)

    print("[done] Track P R1  G1=%s G2=%s(max_other=%.2e) G3=%s"
          % (g1_pass, g2["passed"], g2["max_other_mass"], g3_final_pass))
    del backend
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record_path


if __name__ == "__main__":
    path = run()
    print("\n[audit record] " + path)
