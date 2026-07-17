# pilot_runner.py
# ex18 Minimal Pilot v1.2 driver (Track M / binary_logprob QQ audit).
#
# First-signal run. No controls, items, templates, metrics, or models beyond
# the frozen spec (ex18_minimal_pilot_spec_v1_2). Multi-turn sequential
# protocol only; all rendering via tokenizer.apply_chat_template.
#
# Per model-condition:
#   G1' canonical single-token label validation (A/B and Yes/No)
#   G3' binary_rejection consistency spot check (s1-01, s3-01, both orders)
#   binary_logprob over all 18 items x mappings x orders
#   G2' other-mass gate over every measured position
#   canonicalize -> pool -> registered envelopes, Gamma bounds, classifications
#
# ASCII only. English comments. No gradients (backend uses torch.no_grad()).

import argparse
import datetime as _dt
import hashlib
import json
import os
import random
import subprocess
import sys

import numpy as np
import torch
import transformers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import metrics_ex18 as core_metrics  # noqa: E402
from src import pilot_items as PI  # noqa: E402
from src import pilot_metrics as PM  # noqa: E402
from src.logprob_backend import DECODING, LogprobBackend  # noqa: E402

SEED = 20260718
SPEC_ID = "ex18_minimal_pilot_spec_v1_2"
SPEC_PATH = os.path.join(ROOT, "docs", "ex18_minimal_pilot_spec_v1_2_final.md")
TEMPLATE_ID = "T-1"
ORDERS = ("AB", "BA")
YN_CELLS = [("y", "y"), ("y", "n"), ("n", "y"), ("n", "n")]
SPOT_ITEMS = ("s1-01", "s3-01")
N_REJ = 200
N_REJ_RETRY = 400

MODEL_CONDITIONS = {
    "M1": {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "quant": None, "dtype": "bfloat16", "slug": "qwen3_4b",
        "reason": "primary model-condition (bf16 4B)",
    },
    "M2": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "quant": "8bit", "dtype": "int8", "slug": "llama31_8b",
        "reason": "frozen second model-condition (8-bit 8B)",
    },
    "M2_fallback": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "quant": "8bit", "dtype": "int8", "slug": "mistral7b_v03",
        "reason": "frozen fallback (8-bit) when M2 HF approval is unavailable",
    },
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    return {"python": SEED, "numpy": SEED, "torch": SEED}


def git_info():
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip())
        return {"commit": h, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def package_versions():
    try:
        import accelerate
        acc = accelerate.__version__
    except Exception:
        acc = None
    try:
        import bitsandbytes
        bnb = bitsandbytes.__version__
    except Exception:
        bnb = None
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": acc,
        "bitsandbytes": bnb,
    }


def cuda_info():
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_gb": round(props.total_memory / 1e9, 2),
    }


def build_quant_config(kind):
    if kind is None:
        return None, None
    if kind == "8bit":
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        return cfg, cfg.to_dict()
    raise ValueError("unknown quant kind: %s" % kind)


ZERO_STAGE = {"attempts": 0, "accepted": 0, "other": 0, "shortfall": False}


# ---------------------------------------------------------------------------
# binary_logprob measurement for one (item, mapping)
# ---------------------------------------------------------------------------

def measure_mapping(backend, item, mapping, id_t1, id_t2):
    """Measure both orders for one mapping. Returns a dict with per-order
    semantic point joints, canonicalized raw structures, per-position
    readouts, and the token-coordinate ('y'=t1) joint used for the G3 check.
    """
    t1s, t2s = mapping["t1"], mapping["t2"]
    swap = mapping["swap_for_canonical"]
    out = {"order": {}}
    for order in ORDERS:
        r_first = backend.position_readout(
            PI.first_messages(item, order, mapping), id_y=id_t1, id_n=id_t2)
        r_b1 = backend.position_readout(
            PI.second_messages(item, order, mapping, t1s), id_y=id_t1, id_n=id_t2)
        r_b2 = backend.position_readout(
            PI.second_messages(item, order, mapping, t2s), id_y=id_t1, id_n=id_t2)

        # token-coordinate joint keyed 'y'=t1 / 'n'=t2 (for G3 consistency)
        joint_yn_tok = core_metrics.assemble_joint(
            r_first["p_yes_cond"], r_b1["p_yes_cond"], r_b2["p_yes_cond"])
        # token joint keyed ('t1','t2') then canonicalized to semantic coords
        joint_tok = PM.assemble_token_joint(
            r_first["p_yes_cond"], r_b1["p_yes_cond"], r_b2["p_yes_cond"])
        joint_sem = PM.canonicalize_joint(joint_tok, swap)

        raw_tok = {
            "p_first": (r_first["p_yes"], r_first["p_no"], r_first["other_mass"]),
            "p_second_given_t1": (r_b1["p_yes"], r_b1["p_no"], r_b1["other_mass"]),
            "p_second_given_t2": (r_b2["p_yes"], r_b2["p_no"], r_b2["other_mass"]),
        }
        raw_sem = PM.canonicalize_raw(raw_tok, swap)

        out["order"][order] = {
            "readouts": {"first": r_first, "second_given_t1": r_b1,
                         "second_given_t2": r_b2},
            "joint_yn_tok": joint_yn_tok,
            "joint_sem": joint_sem,
            "raw_sem": raw_sem,
        }
    # per-mapping semantic joints
    sem_ab = out["order"]["AB"]["joint_sem"]
    sem_ba = out["order"]["BA"]["joint_sem"]
    q_iv, oea_iv, oeb_iv = PM.semantic_envelopes(
        out["order"]["AB"]["raw_sem"], out["order"]["BA"]["raw_sem"])
    pm = PM.point_metrics(sem_ab, sem_ba)
    gb = PM.gamma_bounds(q_iv, oea_iv, oeb_iv)
    out["diagnostic"] = {
        "joint_ab": sem_ab, "joint_ba": sem_ba,
        "point": pm,
        "envelope": {"q_QQ": list(q_iv), "OE_A": list(oea_iv), "OE_B": list(oeb_iv)},
        "gamma": gb,
        "qq_class": PM.classify_qq(q_iv),
        "gamma_class": PM.classify_gamma(gb["gamma_lower"], gb["gamma_upper"]),
    }
    out["envelopes"] = (q_iv, oea_iv, oeb_iv)
    return out


def _serialize_joint(joint):
    return {"%s%s" % k: v for k, v in joint.items()}


# ---------------------------------------------------------------------------
# binary_rejection spot check (full sequential trajectory sampling)
# ---------------------------------------------------------------------------

def rejection_joint(backend, item, order, mapping, id_t1, id_t2, n_rej):
    t1s, t2s = mapping["t1"], mapping["t2"]
    first_labels, s1 = backend.collect_accepted(
        PI.first_messages(item, order, mapping), n_rej, id_y=id_t1, id_n=id_t2)
    n_y = first_labels.count("y")
    n_n = first_labels.count("n")
    sec_y, sy = ([], dict(ZERO_STAGE))
    sec_n, sn = ([], dict(ZERO_STAGE))
    if n_y > 0:
        sec_y, sy = backend.collect_accepted(
            PI.second_messages(item, order, mapping, t1s), n_y,
            id_y=id_t1, id_n=id_t2)
    if n_n > 0:
        sec_n, sn = backend.collect_accepted(
            PI.second_messages(item, order, mapping, t2s), n_n,
            id_y=id_t1, id_n=id_t2)
    counts = {c: 0 for c in YN_CELLS}
    iy = in_ = 0
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
        "stage_first": s1, "stage_second_given_t1": sy, "stage_second_given_t2": sn,
        "first_branch_counts": {"t1": n_y, "t2": n_n},
    }
    return counts, stats


def consistency_spot_check(backend, item, mapping, id_t1, id_t2, joints_yn):
    """Run G3' for one item across both orders with the SRS 4.6 CP check and
    the single registered retry (double to 400)."""
    result = {}
    for order in ORDERS:
        counts, rstats = rejection_joint(
            backend, item, order, mapping, id_t1, id_t2, N_REJ)
        ok, n_rej, details = core_metrics.consistency_check(joints_yn[order], counts)
        retry = "not_needed"
        if not ok:
            counts, rstats = rejection_joint(
                backend, item, order, mapping, id_t1, id_t2, N_REJ_RETRY)
            ok, n_rej, details = core_metrics.consistency_check(
                joints_yn[order], counts)
            retry = "retried_400_pass" if ok else "retried_400_FAIL_investigate"
        result[order] = {
            "initial_pass": retry == "not_needed" and ok,
            "final_pass": bool(ok),
            "retry_status": retry,
            "n_rej": n_rej,
            "counts": {"%s%s" % c: counts[c] for c in YN_CELLS},
            "details": {"%s%s" % c: {
                "ci": list(details[c]["ci"]), "p": details[c]["p"],
                "inside": bool(details[c]["inside"])} for c in YN_CELLS},
            "rejection_stats": rstats,
        }
    return result


# ---------------------------------------------------------------------------
# Per-item full evaluation
# ---------------------------------------------------------------------------

def evaluate_item(backend, item, scheme_ids, do_spot):
    scheme = item["label_scheme"]
    id_t1 = scheme_ids[scheme]["t1"]
    id_t2 = scheme_ids[scheme]["t2"]
    mappings = PI.get_mappings(item)

    measured = {}
    for mp in mappings:
        measured[mp["name"]] = measure_mapping(backend, item, mp, id_t1, id_t2)

    # --- pooling (equal weight over mappings, semantic coordinates) ---
    sem_ab_list = [measured[mp["name"]]["order"]["AB"]["joint_sem"] for mp in mappings]
    sem_ba_list = [measured[mp["name"]]["order"]["BA"]["joint_sem"] for mp in mappings]
    pooled_ab = PM.pool_joints(sem_ab_list)
    pooled_ba = PM.pool_joints(sem_ba_list)

    # normalization stop-condition
    for j in sem_ab_list + sem_ba_list + [pooled_ab, pooled_ba]:
        if not PM.joint_sums_to_one(j):
            raise RuntimeError("joint does not normalize for %s" % item["id"])

    q_ivs = [measured[mp["name"]]["envelopes"][0] for mp in mappings]
    a_ivs = [measured[mp["name"]]["envelopes"][1] for mp in mappings]
    b_ivs = [measured[mp["name"]]["envelopes"][2] for mp in mappings]
    pooled_q = PM.pool_intervals(q_ivs)
    pooled_a = PM.pool_intervals(a_ivs)
    pooled_b = PM.pool_intervals(b_ivs)

    pooled_point = PM.point_metrics(pooled_ab, pooled_ba)
    pooled_gamma = PM.gamma_bounds(pooled_q, pooled_a, pooled_b)
    pooled = {
        "joint_ab": _serialize_joint(pooled_ab),
        "joint_ba": _serialize_joint(pooled_ba),
        "point": pooled_point,
        "envelope": {"q_QQ": list(pooled_q), "OE_A": list(pooled_a),
                     "OE_B": list(pooled_b)},
        "gamma": pooled_gamma,
        "qq_class": PM.classify_qq(pooled_q),
        "gamma_class": PM.classify_gamma(
            pooled_gamma["gamma_lower"], pooled_gamma["gamma_upper"]),
    }

    # --- saturation and other-mass positions ---
    p_values = []
    other_masses = []
    positions = []
    for mp in mappings:
        for order in ORDERS:
            for pos_name, r in measured[mp["name"]]["order"][order]["readouts"].items():
                p_values.append(r["p_yes_cond"])
                other_masses.append(r["other_mass"])
                positions.append({
                    "mapping": mp["name"], "order": order, "position": pos_name,
                    "raw_mass": r["raw_mass"], "other_mass": r["other_mass"],
                    "p_cond_t1": r["p_yes_cond"], "top5": r["top5"],
                })
    sat = PM.saturation(p_values)

    # --- per-mapping diagnostics (label-effect) ---
    mapping_diag = {}
    for mp in mappings:
        d = measured[mp["name"]]["diagnostic"]
        mapping_diag[mp["name"]] = {
            "joint_ab": _serialize_joint(d["joint_ab"]),
            "joint_ba": _serialize_joint(d["joint_ba"]),
            "point": d["point"], "envelope": d["envelope"],
            "gamma": d["gamma"], "qq_class": d["qq_class"],
            "gamma_class": d["gamma_class"],
        }

    item_result = {
        "id": item["id"], "stratum": item["stratum"],
        "label_scheme": scheme, "sensitive": item.get("sensitive", False),
        "monitor": item.get("monitor", False),
        "matched_s1": item.get("matched_s1"),
        "historical_header": item.get("historical_header"),
        "vignette": item.get("vignette"),
        "pooled": pooled, "mappings": mapping_diag,
        "saturation": sat, "positions": positions,
    }

    # --- G3 spot check (only for the registered spot items) ---
    if do_spot:
        spot_mapping = PI.get_mappings(item)[0]  # map-1 for AB, yesno for S3
        joints_yn = {
            order: measured[spot_mapping["name"]]["order"][order]["joint_yn_tok"]
            for order in ORDERS
        }
        item_result["spot_check"] = consistency_spot_check(
            backend, item, spot_mapping, id_t1, id_t2, joints_yn)
        item_result["spot_mapping"] = spot_mapping["name"]

    return item_result, max(other_masses)


# ---------------------------------------------------------------------------
# G1 validation for a model-condition
# ---------------------------------------------------------------------------

def validate_gate1(backend, items):
    """Validate canonical single-token labels for every scheme present.
    Returns (all_pass, per_scheme_results, scheme_ids)."""
    # representative rendered contexts per scheme
    probes = {}
    for it in items:
        scheme = it["label_scheme"]
        if scheme in probes:
            continue
        mp = PI.get_mappings(it)[0]
        probes[scheme] = (it, mp)
    per_scheme = {}
    scheme_ids = {}
    all_pass = True
    for scheme, (it, mp) in probes.items():
        rendered = backend.render(PI.first_messages(it, "AB", mp))
        res = PI.validate_emit_tokens(backend._encode, rendered, mp["t1"], mp["t2"])
        per_scheme[scheme] = res
        if res["passed"]:
            scheme_ids[scheme] = {"t1": res["token_ids"]["t1"],
                                  "t2": res["token_ids"]["t2"]}
        else:
            all_pass = False
    return all_pass, per_scheme, scheme_ids


# ---------------------------------------------------------------------------
# Run one model-condition end to end
# ---------------------------------------------------------------------------

def run_condition(cond_key, items_data, stamp, note=""):
    cond = MODEL_CONDITIONS[cond_key]
    items = items_data["items"]

    print("[load] %s (%s)" % (cond["model_id"], cond["quant"] or "bf16"))
    quant_cfg, quant_dict = build_quant_config(cond["quant"])
    backend = LogprobBackend(cond["model_id"], quantization_config=quant_cfg)

    # thinking-tag guard
    probe_item = items[0]
    probe_mp = PI.get_mappings(probe_item)[0]
    rendered_probe = backend.render(PI.first_messages(probe_item, "AB", probe_mp))
    thinking = ("<think>" in rendered_probe) or ("</think>" in rendered_probe)
    if thinking:
        raise RuntimeError("STOP: rendered chat template contains thinking tags")
    print("[ok] no thinking tags in rendered template")

    # G1
    g1_pass, per_scheme, scheme_ids = validate_gate1(backend, items)
    print("[G1] pass=%s schemes=%s" % (g1_pass, list(per_scheme.keys())))

    seeds = set_seeds()
    record_path = os.path.join(
        ROOT, "runs", "pilot_v1_2_%s_%s.json" % (stamp, cond["slug"]))

    metadata = {
        "phase": "minimal_pilot_v1_2",
        "condition_key": cond_key,
        "status": "started",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "spec_id": SPEC_ID,
        "spec_sha256": sha256_file(SPEC_PATH) if os.path.exists(SPEC_PATH) else None,
        "items_file": os.path.relpath(
            os.path.join(ROOT, "items", "pilot_v1_2.json"), ROOT),
        "items_file_sha256": sha256_file(os.path.join(ROOT, "items", "pilot_v1_2.json")),
        "model_id": cond["model_id"],
        "model_revision": getattr(backend.model.config, "_name_or_path", None),
        "tokenizer_id": cond["model_id"],
        "tokenizer_revision": None,
        "chat_template_sha256": backend.chat_template_sha256(),
        "chat_template_no_thinking_tags": (not thinking),
        "model_dtype": cond["dtype"],
        "quantization": quant_dict,
        "quantization_kind": cond["quant"],
        "condition_reason": cond["reason"] + ((" | " + note) if note else ""),
        "cuda": cuda_info(),
        "versions": package_versions(),
        "system_prompts": {"AB": PI.SYSTEM_PROMPT_AB, "YesNo": PI.SYSTEM_PROMPT_YESNO},
        "historical_headers": items_data.get("historical_headers", {}),
        "template_id": TEMPLATE_ID,
        "decoding": DECODING,
        "decoding_note": "binary_logprob: deterministic argmax-free next-token "
                         "probabilities; temperature/top_p metadata as above, "
                         "no sampling. binary_rejection: do_sample per DECODING.",
        "seeds": seeds,
        "canonical_labels": {
            scheme: {
                "passed": r["passed"], "variant": r["variant"],
                "token_ids": r["token_ids"], "per_candidate": r["per_candidate"],
            } for scheme, r in per_scheme.items()
        },
        "git": git_info(),
        "gates": {"G1": g1_pass},
        "results": [],
    }
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)
    print("[audit] metadata written: %s" % record_path)

    if not g1_pass:
        metadata["status"] = "aborted_G1_fail"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=True)
        raise RuntimeError("STOP: G1 canonical label validation failed for %s"
                           % cond_key)

    # per-item evaluation
    max_other = 0.0
    g3_final_pass = True
    g3_present = False
    for item in items:
        do_spot = item["id"] in SPOT_ITEMS
        print("   [item] %s (%s)%s" % (item["id"], item["stratum"],
                                       " +spot" if do_spot else ""))
        item_result, item_max_other = evaluate_item(
            backend, item, scheme_ids, do_spot)
        max_other = max(max_other, item_max_other)
        if do_spot:
            g3_present = True
            for order in ORDERS:
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

    print("[done] %s  G1=%s G2=%s(max_other=%.2e) G3=%s"
          % (cond_key, g1_pass, g2["passed"], g2["max_other_mass"], g3_final_pass))
    # release VRAM
    del backend
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="M1",
                    help="comma list of condition keys: M1,M2,M2_fallback")
    ap.add_argument("--note", default="", help="reason note for the audit record")
    args = ap.parse_args()

    items_data = PI.load_items(os.path.join(ROOT, "items", "pilot_v1_2.json"))
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")

    paths = []
    for cond_key in [c.strip() for c in args.models.split(",") if c.strip()]:
        path = run_condition(cond_key, items_data, stamp, note=args.note)
        paths.append(path)
    print("\n[audit records]")
    for p in paths:
        print("  " + p)


if __name__ == "__main__":
    main()
