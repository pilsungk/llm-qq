# pilot_reporting.py
# ex18 Minimal Pilot v1.2 report generator.
#
# Reads the per-condition audit JSON records and emits the frozen 14-section
# Markdown report. Headers/tables/math in English; prose in Korean per project
# convention. Compatibility language only; no quantum-cognition claims.
#
# ASCII only in source. (Report body itself contains Korean prose.)

import datetime as _dt
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OSS_EFFECT = 0.05          # OSS > 0.05 counts as an order-sensitive item
MEMO_THRESH = 0.3          # |OSS_S1 - OSS_S2| > 0.3 memorization alert


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def f4(x):
    try:
        return "%+.4f" % x
    except Exception:
        return str(x)


def fe(x):
    try:
        return "%.2e" % x
    except Exception:
        return str(x)


# ---------------------------------------------------------------------------
# Per-condition summary
# ---------------------------------------------------------------------------

def summarize(meta):
    items = []
    for it in meta["results"]:
        pooled = it["pooled"]
        scheme = it["label_scheme"]
        maps = it.get("mappings", {})
        map_contextual_all = all(
            m["gamma_class"] == "CONTEXTUAL" for m in maps.values()
        ) if maps else False
        certified_contextual = (
            pooled["gamma_class"] == "CONTEXTUAL"
            and (scheme == "YesNo" or map_contextual_all)
        )
        items.append({
            "id": it["id"], "stratum": it["stratum"], "scheme": scheme,
            "sensitive": it.get("sensitive", False),
            "monitor": it.get("monitor", False),
            "matched_s1": it.get("matched_s1"),
            "saturated": it["saturation"]["saturated"],
            "frac_saturated": it["saturation"]["frac_saturated"],
            "qq_class": pooled["qq_class"],
            "gamma_class": pooled["gamma_class"],
            "gamma_lower": pooled["gamma"]["gamma_lower"],
            "gamma_upper": pooled["gamma"]["gamma_upper"],
            "OSS": pooled["point"]["OSS"],
            "OE_A": pooled["point"]["OE_A"],
            "OE_B": pooled["point"]["OE_B"],
            "q_QQ": pooled["point"]["q_QQ"],
            "q_env": pooled["envelope"]["q_QQ"],
            "certified_contextual": certified_contextual,
            "map_names": list(maps.keys()),
            "maps": maps,
        })
    return {"key": meta["condition_key"], "model_id": meta["model_id"],
            "slug": meta.get("condition_key"), "meta": meta, "items": items}


def _by_id(summary):
    return {it["id"]: it for it in summary["items"]}


def counts_by(items, field, values):
    return {v: sum(1 for it in items if it[field] == v) for v in values}


# ---------------------------------------------------------------------------
# Go / No-Go evaluation (resource-allocation rules, not inference thresholds)
# ---------------------------------------------------------------------------

def go_no_go(summaries):
    n_models = len(summaries)
    out = {"n_models": n_models, "notes": [], "outcomes": {}}

    # per-model non-saturated sets
    per = []
    for s in summaries:
        nonsat = [it for it in s["items"] if not it["saturated"]]
        per.append({
            "key": s["key"],
            "nonsat": nonsat,
            "n_nonsat": len(nonsat),
            "violated": [it["id"] for it in nonsat if it["qq_class"] == "VIOLATED"],
            "satisfied": [it["id"] for it in nonsat if it["qq_class"] == "SATISFIED"],
            "indet": [it["id"] for it in nonsat if it["qq_class"] == "INDETERMINATE"],
            "contextual": [it["id"] for it in s["items"] if it["certified_contextual"]],
            "oss_gt": [it["id"] for it in nonsat if it["OSS"] > OSS_EFFECT],
        })

    # 1. immediate contextuality follow-up
    if n_models >= 2:
        common = set(per[0]["contextual"])
        for p in per[1:]:
            common &= set(p["contextual"])
        out["outcomes"]["immediate_contextuality_followup"] = {
            "triggered": len(common) > 0, "items": sorted(common)}
    else:
        cx = per[0]["contextual"]
        out["outcomes"]["immediate_contextuality_followup"] = {
            "triggered": False, "items": cx,
            "note": "NOT EVALUABLE: cross-model reproduction requires M2, which "
                    "was execution-blocked (no certified Gamma_lower > 0 in M1)"}

    # 2. systematic violation
    if n_models >= 2:
        trig = False
        detail = {}
        for i, p in enumerate(per):
            other = per[1 - i]
            if len(p["violated"]) >= 4:
                repro = set(p["violated"]) & set(other["violated"])
                if len(repro) >= 2:
                    trig = True
                detail[p["key"]] = {"violated": p["violated"],
                                    "reproduced_in_other": sorted(repro)}
        out["outcomes"]["systematic_violation"] = {"triggered": trig, "detail": detail}
    else:
        out["outcomes"]["systematic_violation"] = {
            "triggered": False,
            "note": "NOT EVALUABLE: requires two model-conditions; M2 execution-blocked",
            "violated_M1": per[0]["violated"]}

    # 3. broad satisfaction
    if n_models >= 2:
        ok = True
        detail = {}
        for p in per:
            frac = (len(p["satisfied"]) / p["n_nonsat"]) if p["n_nonsat"] else 0.0
            detail[p["key"]] = {"satisfied_frac": frac,
                                "n_oss_gt": len(p["oss_gt"])}
            if frac < 0.70 or len(p["oss_gt"]) < 5:
                ok = False
        out["outcomes"]["broad_satisfaction"] = {"triggered": ok, "detail": detail}
    else:
        p = per[0]
        frac = (len(p["satisfied"]) / p["n_nonsat"]) if p["n_nonsat"] else 0.0
        out["outcomes"]["broad_satisfaction"] = {
            "triggered": False,
            "note": "NOT EVALUABLE: requires both models; M2 execution-blocked",
            "M1_satisfied_frac": frac, "M1_n_oss_gt": len(p["oss_gt"])}

    # 4. redesign
    trig = False
    detail = {}
    for p in per:
        indet_rate = (len(p["indet"]) / p["n_nonsat"]) if p["n_nonsat"] else 1.0
        cond = (p["n_nonsat"] < 8) or (indet_rate > 0.30)
        detail[p["key"]] = {"n_nonsat": p["n_nonsat"], "indet_rate": indet_rate,
                            "flag": cond}
        trig = trig or cond
    out["outcomes"]["redesign"] = {"triggered": trig, "detail": detail}

    # 5. memorization alert (per model, matched 3 pairs)
    mem = {}
    for s in summaries:
        byid = _by_id(s)
        pairs = []
        n_big = 0
        for it in s["items"]:
            if it["matched_s1"] and it["matched_s1"] in byid:
                oss_s2 = it["OSS"]
                oss_s1 = byid[it["matched_s1"]]["OSS"]
                diff = abs(oss_s1 - oss_s2)
                pairs.append({"s2": it["id"], "s1": it["matched_s1"],
                              "oss_s1": oss_s1, "oss_s2": oss_s2, "abs_diff": diff})
                if diff > MEMO_THRESH:
                    n_big += 1
        mem[s["key"]] = {"triggered": n_big >= 2, "pairs": pairs}
    out["outcomes"]["memorization_alert"] = mem

    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _h(s):
    return "\n" + s + "\n"


def generate_report(audit_paths, out_path=None):
    metas = [load(p) for p in audit_paths]
    summaries = [summarize(m) for m in metas]
    gng = go_no_go(summaries)

    L = []
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L.append("# ex18 Minimal Pilot v1.2 Report")
    L.append("")
    L.append("- Generated: %s" % stamp)
    L.append("- Spec: `%s` (frozen)" % metas[0].get("spec_id"))
    L.append("- Model-conditions: %s" % ", ".join(s["key"] for s in summaries))
    L.append("- Protocol: binary_logprob (primary) + binary_rejection spot check")
    L.append("")
    L.append("본 보고서는 first-signal 파일럿 실행 레코드이다. QQ 만족은 투영측정 "
             "모델과의 **양립(compatibility)** 근거일 뿐 양자인지/양자성의 증명이 "
             "아니다. S1/S2(A/B label)와 S3(Yes/No)는 label 체계가 달라 지표를 "
             "**층간 직접 비교하지 않는다** (Section 14).")

    # 1. Execution summary
    L.append(_h("## 1. Execution Summary"))
    for s in summaries:
        g = s["meta"]["gates"]
        L.append("- **%s** (`%s`): status=%s | G1=%s G2=%s(max_other=%s) G3=%s"
                 % (s["key"], s["model_id"], s["meta"]["status"],
                    g.get("G1"), g.get("G2_pass"), fe(g.get("G2_max_other_mass", 0.0)),
                    g.get("G3_pass")))
    L.append("")
    L.append("실행 순서: M1 전체 -> sanity -> (M2 조건부). 각 조건은 독립 감사 "
             "레코드를 가진다.")
    if len(summaries) < 2:
        L.append("")
        L.append("**M2 미실행 (실행 차단)**: 두 번째 model-condition은 모델 접근 "
                 "불가(HF 승인/토큰 부재) + 금지된 신규 의존성(bitsandbytes) + "
                 "Blackwell sm_120 호환성 위험으로 실행 시점에 차단되었다. 이는 "
                 "M2 모델 실패가 아니라 환경/접근 차단이며, PI 결정에 따라 M1 단독 "
                 "으로 최종화한다 (Section 12/13).")

    # 2. Model-condition metadata
    L.append(_h("## 2. Model-Condition Metadata"))
    for s in summaries:
        m = s["meta"]
        L.append("### %s" % s["key"])
        L.append("| field | value |")
        L.append("|---|---|")
        L.append("| model_id | %s |" % m["model_id"])
        L.append("| model_revision | %s |" % m.get("model_revision"))
        L.append("| dtype | %s |" % m.get("model_dtype"))
        L.append("| quantization | %s |" % (m.get("quantization_kind") or "none"))
        L.append("| chat_template_sha256 | `%s` |" % m.get("chat_template_sha256"))
        L.append("| no_thinking_tags | %s |" % m.get("chat_template_no_thinking_tags"))
        L.append("| seeds | %s |" % m.get("seeds"))
        L.append("| decoding | %s |" % m.get("decoding"))
        cuda = m.get("cuda", {})
        L.append("| cuda | %s cc=%s %sGB |" % (cuda.get("name"),
                 cuda.get("capability"), cuda.get("total_memory_gb")))
        L.append("| versions | %s |" % m.get("versions"))
        L.append("| git | %s |" % m.get("git"))
        L.append("| condition_reason | %s |" % m.get("condition_reason"))
        labs = m.get("canonical_labels", {})
        for scheme, r in labs.items():
            L.append("| labels[%s] | variant=%s ids=%s |"
                     % (scheme, r.get("variant"), r.get("token_ids")))
        L.append("")

    # 3. Gate table
    L.append(_h("## 3. Gate Table (G1 / G2 / G3)"))
    L.append("| condition | G1 labels | G2 other-mass | G3 consistency |")
    L.append("|---|---|---|---|")
    for s in summaries:
        g = s["meta"]["gates"]
        L.append("| %s | %s | %s (max=%s, thr<1e-2) | %s |"
                 % (s["key"], "PASS" if g.get("G1") else "FAIL",
                    "PASS" if g.get("G2_pass") else "FAIL",
                    fe(g.get("G2_max_other_mass", 0.0)),
                    "PASS" if g.get("G3_pass") else ("N/A" if not g.get("G3_present")
                                                     else "FAIL")))
    L.append("")
    L.append("G3 주: 모델당 명목 union-bound 오경보 상한은 4셀 x 4검사 alpha=1% "
             "에서 **<= 4%**. 합성 검증에서 관찰된 약 2% (0.9951^4)는 명목 보증이 "
             "아니라 사전 합성 보정의 **경험적 비율**이다.")

    # 4. Per-item pooled results
    L.append(_h("## 4. Per-Item Pooled Results"))
    L.append("| id | stratum | OE_A | OE_B | OSS | q_QQ(point) | q_env[min,max] "
             "| QQ | Gamma[lo,hi] | Gamma class | sat |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        L.append("| **%s** | | | | | | | | | | |" % s["key"])
        for it in s["items"]:
            qenv = it["q_env"]
            L.append("| %s | %s | %s | %s | %.4f | %s | [%s, %s] | %s | [%s, %s] | %s | %s |"
                     % (it["id"], it["stratum"], f4(it["OE_A"]), f4(it["OE_B"]),
                        it["OSS"], f4(it["q_QQ"]), f4(qenv[0]), f4(qenv[1]),
                        it["qq_class"], f4(it["gamma_lower"]), f4(it["gamma_upper"]),
                        it["gamma_class"], "Y" if it["saturated"] else "n"))
    L.append("")

    # 5. Per-mapping diagnostic results
    L.append(_h("## 5. Per-Mapping Diagnostic Results (label-effect)"))
    L.append("A/B 문항만 매핑별 진단을 가진다 (S3는 Yes/No 단일).")
    for s in summaries:
        L.append("")
        L.append("### %s" % s["key"])
        L.append("| id | mapping | OSS | q_QQ | QQ | Gamma class |")
        L.append("|---|---|---|---|---|---|")
        for it in s["items"]:
            if not it["maps"]:
                continue
            for mname, m in it["maps"].items():
                L.append("| %s | %s | %.4f | %s | %s | %s |"
                         % (it["id"], mname, m["point"]["OSS"], f4(m["point"]["q_QQ"]),
                            m["qq_class"], m["gamma_class"]))
    L.append("")

    # 6. QQ classification counts
    L.append(_h("## 6. QQ Classification Counts (pooled)"))
    L.append("| condition | SATISFIED | VIOLATED | INDETERMINATE | (non-sat) S/V/I |")
    L.append("|---|---|---|---|---|")
    for s in summaries:
        c = counts_by(s["items"], "qq_class",
                      ["SATISFIED", "VIOLATED", "INDETERMINATE"])
        ns = [it for it in s["items"] if not it["saturated"]]
        cn = counts_by(ns, "qq_class", ["SATISFIED", "VIOLATED", "INDETERMINATE"])
        L.append("| %s | %d | %d | %d | %d/%d/%d |"
                 % (s["key"], c["SATISFIED"], c["VIOLATED"], c["INDETERMINATE"],
                    cn["SATISFIED"], cn["VIOLATED"], cn["INDETERMINATE"]))
    L.append("")

    # 7. Gamma classification counts
    L.append(_h("## 7. Gamma Classification Counts (pooled)"))
    L.append("| condition | NONCONTEXTUAL | CONTEXTUAL | INDETERMINATE | certified (pooled+maps) |")
    L.append("|---|---|---|---|---|")
    for s in summaries:
        c = counts_by(s["items"], "gamma_class",
                      ["NONCONTEXTUAL", "CONTEXTUAL", "INDETERMINATE"])
        cert = [it["id"] for it in s["items"] if it["certified_contextual"]]
        L.append("| %s | %d | %d | %d | %s |"
                 % (s["key"], c["NONCONTEXTUAL"], c["CONTEXTUAL"], c["INDETERMINATE"],
                    (", ".join(cert) if cert else "none")))
    L.append("")

    # 8. Saturation table
    L.append(_h("## 8. Saturation Table"))
    L.append("포화: 문항 측정 위치 중 max(p,1-p)>0.99 비율 > 50%.")
    L.append("| condition | saturated items | count |")
    L.append("|---|---|---|")
    for s in summaries:
        sat = [it["id"] for it in s["items"] if it["saturated"]]
        L.append("| %s | %s | %d |" % (s["key"], ", ".join(sat) if sat else "none",
                                       len(sat)))
    L.append("")

    # 9. Label-effect diagnostics
    L.append(_h("## 9. Label-Effect Diagnostics (map-1 vs map-2)"))
    L.append("| condition | id | dOSS | dq_QQ | q_QQ map-1 | q_QQ map-2 |")
    L.append("|---|---|---|---|---|---|")
    for s in summaries:
        for it in s["items"]:
            if not it["maps"] or len(it["maps"]) < 2:
                continue
            m1 = it["maps"].get("map-1"); m2 = it["maps"].get("map-2")
            doss = abs(m1["point"]["OSS"] - m2["point"]["OSS"])
            dq = abs(m1["point"]["q_QQ"] - m2["point"]["q_QQ"])
            L.append("| %s | %s | %.4f | %.4f | %s | %s |"
                     % (s["key"], it["id"], doss, dq, f4(m1["point"]["q_QQ"]),
                        f4(m2["point"]["q_QQ"])))
    L.append("")

    # 10. Matched S1/S2 comparison
    L.append(_h("## 10. S1/S2 Matched-Pair Descriptive Comparison"))
    L.append("| condition | S2 | S1 | OSS_S2 | OSS_S1 | |dOSS| | >0.3 |")
    L.append("|---|---|---|---|---|---|---|")
    for s in summaries:
        byid = _by_id(s)
        for it in s["items"]:
            if it["matched_s1"] and it["matched_s1"] in byid:
                a = it["OSS"]; b = byid[it["matched_s1"]]["OSS"]
                d = abs(a - b)
                L.append("| %s | %s | %s | %.4f | %.4f | %.4f | %s |"
                         % (s["key"], it["id"], it["matched_s1"], a, b, d,
                            "Y" if d > MEMO_THRESH else "n"))
    L.append("")
    L.append("주: 기술 비교이며 인간 벤치마크(QuestOrdData) 미수신 상태 — 절대 "
             "수준 해석 유보.")

    # 11. s2-02 monitoring
    L.append(_h("## 11. s2-02 Monitoring (Armey / Lott format validity)"))
    for s in summaries:
        it_raw = next((r for r in s["meta"]["results"] if r["id"] == "s2-02"), None)
        if it_raw is None:
            continue
        L.append("### %s" % s["key"])
        L.append("| mapping | order | position | raw_mass | other_mass | top1 |")
        L.append("|---|---|---|---|---|---|")
        for p in it_raw["positions"]:
            top1 = p["top5"][0] if p["top5"] else {}
            L.append("| %s | %s | %s | %.6f | %s | %s(%.3f) |"
                     % (p["mapping"], p["order"], p["position"], p["raw_mass"],
                        fe(p["other_mass"]),
                        (top1.get("string") or "").replace("|", "/"),
                        top1.get("prob", 0.0)))
        max_other = max(p["other_mass"] for p in it_raw["positions"])
        L.append("")
        L.append("판정: raw mass 및 top-5 기준 format validity %s (max other=%s). "
                 "인지도 열위로 인한 무효 응답 급증 징후는 %s."
                 % ("정상" if max_other < 0.01 else "주의",
                    fe(max_other),
                    "관찰되지 않음" if max_other < 0.01 else "관찰됨 -- 대체 인선 재검토 권고"))
    L.append("")

    # 12. Failures, retries, deviations
    L.append(_h("## 12. Failures, Retries, and Deviations"))
    for s in summaries:
        L.append("### %s" % s["key"])
        dev = []
        for r in s["meta"]["results"]:
            if "spot_check" in r:
                for order, sc in r["spot_check"].items():
                    if sc["retry_status"] != "not_needed":
                        dev.append("%s %s: %s (n_rej=%d)"
                                   % (r["id"], order, sc["retry_status"], sc["n_rej"]))
        if s["meta"]["gates"].get("G2_pass") is False:
            dev.append("G2 other-mass gate FAIL (max=%s)"
                       % fe(s["meta"]["gates"].get("G2_max_other_mass")))
        L.append("- " + ("; ".join(dev) if dev else "no retries or gate failures"))
    if len(summaries) < 2:
        L.append("- **Execution-blocking deviation (M2 not run)**: the second "
                 "model-condition was blocked at execution time by three "
                 "independent causes: (i) the gated models "
                 "`meta-llama/Llama-3.1-8B-Instruct` and the frozen fallback "
                 "`mistralai/Mistral-7B-Instruct-v0.3` were inaccessible (no "
                 "Hugging Face approval/token present); (ii) 8-bit loading "
                 "requires `bitsandbytes`, a new dependency prohibited without "
                 "prior approval (CLAUDE.md); (iii) unresolved Blackwell sm_120 "
                 "compatibility risk for bitsandbytes 8-bit. This is an "
                 "environment/access block, NOT an M2 model failure. Per PI "
                 "decision the run is finalized M1-only without modifying the "
                 "frozen design or adding conditions.")
    L.append("")

    # 13. Go/No-Go
    L.append(_h("## 13. Registered Go/No-Go Evaluation"))
    L.append("주: 아래는 사전 등록 추론 임계가 아니라 **후속 자원 배분 결정 "
             "규칙**이다.")
    o = gng["outcomes"]
    L.append("- **Immediate contextuality follow-up**: %s %s"
             % (o["immediate_contextuality_followup"]["triggered"],
                o["immediate_contextuality_followup"].get("items")
                or o["immediate_contextuality_followup"].get("note", "")))
    L.append("- **Systematic violation**: %s %s"
             % (o["systematic_violation"]["triggered"],
                o["systematic_violation"].get("detail")
                or o["systematic_violation"].get("note", "")))
    L.append("- **Broad satisfaction**: %s %s"
             % (o["broad_satisfaction"]["triggered"],
                o["broad_satisfaction"].get("detail")
                or o["broad_satisfaction"].get("note", "")))
    L.append("- **Redesign**: %s %s"
             % (o["redesign"]["triggered"], o["redesign"]["detail"]))
    L.append("- **Memorization alert**: %s"
             % {k: v["triggered"] for k, v in o["memorization_alert"].items()})
    L.append("")
    if len(summaries) < 2:
        red = o["redesign"]
        m1nonsat = list(red["detail"].values())[0]["n_nonsat"]
        L.append("### Applicable M1-only resource decision: **REDESIGN / NO EXPANSION**")
        L.append("")
        L.append("- 이 단일 조건 실행에서 적용되는 자원 배분 결정은 **REDESIGN / "
                 "NO EXPANSION**이다. 근거: 18문항 중 17문항이 포화(saturated)되어 "
                 "**비포화 문항은 %d개(< 8)**이며, 이는 프레임된 redesign 규칙을 "
                 "직접 발동한다. 문항·템플릿 개선 후 재실행이 필요하다 (확장 금지)."
                 % m1nonsat)
        L.append("- **Cross-model rules (immediate contextuality / systematic "
                 "violation / broad satisfaction)**: **NOT EVALUABLE** — 두 "
                 "model-condition 재현이 필요하나 M2가 실행 차단되었다.")
        L.append("- **T-2 미해당**: T-2 사전 고정 템플릿은 M1의 certified "
                 "contextuality 신호(Gamma_lower > 0)가 M2 재현에 실패했을 때만 "
                 "적용된다. M1은 certified Gamma_lower > 0을 산출하지 않았으므로 "
                 "(0 CONTEXTUAL), T-2는 후속 작업으로 등록되지 않는다.")
        L.append("- **암기 경보(memorization)**: 3개 matched S1/S2 쌍 기준 산출 "
                 "(Section 10). 단, 전 문항 포화로 OSS 해석이 제한적임을 병기한다.")

    # 14. Cross-stratum statement
    L.append(_h("## 14. Cross-Stratum Comparison Statement"))
    L.append("S1/S2 문항은 A/B 중립 토큰(원 옵션 매핑), S3 문항은 Yes/No 라벨을 "
             "사용한다. 라벨 체계가 다르므로 두 층의 지표(q_QQ, OSS, Gamma 등)를 "
             "**층간 직접 수치 비교하지 않는다**. 각 층 내부에서만 해석한다. QQ "
             "만족은 투영측정/대칭반복/order-matched mixture 등과의 양립 근거이며 "
             "기전 식별이나 양자성 주장이 아니다.")
    L.append("")

    text = "\n".join(L)
    if out_path is None:
        stamp2 = _dt.datetime.now().strftime("%Y-%m-%d_%H%M")
        out_path = os.path.join(
            ROOT, "runs", "%s_ex18_minimal_pilot_v1_2_report.md" % stamp2)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path, gng


# ---------------------------------------------------------------------------
# Redesign Run R1 report: Track M (baseline) vs Track P comparison
# ---------------------------------------------------------------------------

def generate_r1_report(p_audit_path, m_audit_path, out_path=None):
    """Track P R1 report with a per-item Track M vs Track P comparison and the
    spec Section 4 primary-outcome verdict (non-saturated >= 4/8)."""
    p_meta = load(p_audit_path)
    m_meta = load(m_audit_path)
    p_sum = summarize(p_meta)
    m_sum = summarize(m_meta)
    m_by = _by_id(m_sum)

    p_items = p_sum["items"]
    ids = [it["id"] for it in p_items]
    n_items = len(p_items)
    n_nonsat = sum(1 for it in p_items if not it["saturated"])
    success = n_nonsat >= 4

    L = []
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L.append("# ex18 Redesign Run R1 Report (Track P Minimal)")
    L.append("")
    L.append("- Generated: %s" % stamp)
    L.append("- Spec: `%s` (single approved rerun)" % p_meta.get("spec_id"))
    L.append("- Track: **P** (respondent simulation, system prompt id=%s)"
             % p_meta.get("system_prompt_id"))
    L.append("- Track M baseline: `%s`" % p_meta.get("baseline_track_m_record"))
    L.append("- Model-condition: M1 (`%s`, bf16); M2 execution-blocked."
             % p_meta.get("model_id"))
    L.append("")
    L.append("본 재실행은 **Track P 응답자 프레이밍이 출력 포화를 낮추는지**를 "
             "검정한다. 모델이 역사적 인간 판단을 재현하는지, 혹은 양자인지 여부를 "
             "검정하는 것이 **아니다**. QQ 만족은 투영측정 모델과의 양립 근거일 뿐 "
             "이다.")

    # 1. Execution summary + gates
    g = p_meta["gates"]
    L.append(_h("## 1. Execution Summary and Gates"))
    L.append("- status=%s | G1=%s G2=%s(max_other=%s) G3=%s"
             % (p_meta["status"], g.get("G1"), g.get("G2_pass"),
                fe(g.get("G2_max_other_mass", 0.0)), g.get("G3_pass")))
    L.append("")
    L.append("| gate | result |")
    L.append("|---|---|")
    L.append("| G1 A/B single-token (T-1P context) | %s |"
             % ("PASS" if g.get("G1") else "FAIL"))
    L.append("| G2 other-mass < 1e-2 all positions | %s (max=%s) |"
             % ("PASS" if g.get("G2_pass") else "FAIL",
                fe(g.get("G2_max_other_mass", 0.0))))
    L.append("| G3 CP consistency (s1-01 x 2 orders) | %s |"
             % ("PASS" if g.get("G3_pass") else
                ("N/A" if not g.get("G3_present") else "FAIL")))
    for it in p_meta["results"]:
        if "spot_check" in it:
            for order, sc in it["spot_check"].items():
                L.append("  - spot %s %s: final_pass=%s retry=%s n_rej=%d"
                         % (it["id"], order, sc["final_pass"],
                            sc["retry_status"], sc["n_rej"]))

    # 2. Metadata
    L.append(_h("## 2. Track P Metadata"))
    L.append("| field | value |")
    L.append("|---|---|")
    L.append("| system_prompt_id | %s |" % p_meta.get("system_prompt_id"))
    L.append("| system_prompt_template | %s |"
             % p_meta.get("system_prompt_template", "").replace("|", "/"))
    L.append("| historical_header_used | %s |" % p_meta.get("historical_header_used"))
    L.append("| reference_years | %s |" % p_meta.get("reference_years"))
    L.append("| seeds | %s |" % p_meta.get("seeds"))
    L.append("| chat_template_sha256 | `%s` |" % p_meta.get("chat_template_sha256"))
    L.append("| labels[AB] | variant=%s ids=%s |"
             % (p_meta["canonical_labels"]["AB"]["variant"],
                p_meta["canonical_labels"]["AB"]["token_ids"]))
    L.append("| m2_status | %s |" % p_meta.get("m2_status"))
    L.append("| git | %s |" % p_meta.get("git"))

    # 3. Per-item pooled results (Track P)
    L.append(_h("## 3. Per-Item Pooled Results (Track P)"))
    L.append("| id | year | OE_A | OE_B | OSS | q_QQ | q_env[min,max] | QQ | "
             "Gamma[lo,hi] | Gamma class | sat |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    ry = p_meta.get("reference_years", {})
    for it in p_items:
        qenv = it["q_env"]
        L.append("| %s | %s | %s | %s | %.4f | %s | [%s, %s] | %s | [%s, %s] | %s | %s |"
                 % (it["id"], ry.get(it["id"]), f4(it["OE_A"]), f4(it["OE_B"]),
                    it["OSS"], f4(it["q_QQ"]), f4(qenv[0]), f4(qenv[1]),
                    it["qq_class"], f4(it["gamma_lower"]), f4(it["gamma_upper"]),
                    it["gamma_class"], "Y" if it["saturated"] else "n"))

    # 4. Track M vs Track P comparison
    L.append(_h("## 4. Track M vs Track P Comparison (same 8 items)"))
    L.append("| id | sat_M | sat_P | q_QQ_M | q_QQ_P | OSS_M | OSS_P | "
             "Gamma_M | Gamma_P |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    n_sat_M = 0
    n_sat_P = 0
    for it in p_items:
        m = m_by.get(it["id"])
        if m is None:
            continue
        if m["saturated"]:
            n_sat_M += 1
        if it["saturated"]:
            n_sat_P += 1
        L.append("| %s | %s | %s | %s | %s | %.4f | %.4f | %s | %s |"
                 % (it["id"], "Y" if m["saturated"] else "n",
                    "Y" if it["saturated"] else "n",
                    f4(m["q_QQ"]), f4(it["q_QQ"]), m["OSS"], it["OSS"],
                    m["gamma_class"], it["gamma_class"]))
    L.append("")
    L.append("포화 요약: Track M %d/%d 포화 -> Track P %d/%d 포화 "
             "(비포화 Track P = %d/%d)."
             % (n_sat_M, n_items, n_sat_P, n_items, n_nonsat, n_items))

    # 5. Primary outcome verdict (spec Section 4)
    L.append(_h("## 5. Primary Outcome (spec Section 4)"))
    L.append("- 비포화 문항 (Track P): **%d / %d**" % (n_nonsat, n_items))
    if success:
        L.append("- **판정: Track P가 분포를 여는 데 성공 (비포화 >= 4/8).** "
                 "전체 지표 (q_QQ envelope, OSS, Gamma, label-effect)를 유의미한 "
                 "first-signal로 보고한다.")
    else:
        L.append("- **판정: 구성 개념 불변의 포화 (비포화 < 4/8).** 단일 모델의 "
                 "binary-conditioned 분포는 persona(응답자 모사) 조건에서도 "
                 "결정론적이라는 발견으로 보고한다.")
    L.append("- 어느 경우든 본 트랙은 이 보고서(불변 레코드)로 **완결**되며, "
             "후속 실행/확장은 없다 (spec Section 4).")

    # 6. Supplementary observations
    L.append(_h("## 6. Supplementary Observations (report-only)"))
    L.append("### Track M -> Track P shift (q_QQ, OSS)")
    L.append("| id | dq_QQ (P-M) | dOSS (P-M) |")
    L.append("|---|---|---|")
    for it in p_items:
        m = m_by.get(it["id"])
        if m is None:
            continue
        L.append("| %s | %s | %s |"
                 % (it["id"], f4(it["q_QQ"] - m["q_QQ"]), f4(it["OSS"] - m["OSS"])))
    L.append("")
    L.append("### Matched S1/S2 OSS divergence (Track P)")
    L.append("| S2 | S1 | OSS_S2 | OSS_S1 | |dOSS| |")
    L.append("|---|---|---|---|---|")
    p_by = _by_id(p_sum)
    for it in p_items:
        if it["matched_s1"] and it["matched_s1"] in p_by:
            a = it["OSS"]; b = p_by[it["matched_s1"]]["OSS"]
            L.append("| %s | %s | %.4f | %.4f | %.4f |"
                     % (it["id"], it["matched_s1"], a, b, abs(a - b)))
    L.append("")
    L.append("주: 인간 벤치마크(QuestOrdData) 미수신 -- 절대 수준·역사 재현 해석 "
             "유보. 방향 수준 정성 관찰만 가능.")

    # 7. s2-02 monitoring
    L.append(_h("## 7. s2-02 Monitoring (Track P)"))
    s2 = next((r for r in p_meta["results"] if r["id"] == "s2-02"), None)
    if s2 is not None:
        mo = max(p["other_mass"] for p in s2["positions"])
        L.append("| mapping | order | position | raw_mass | other_mass | top1 |")
        L.append("|---|---|---|---|---|---|")
        for p in s2["positions"]:
            t1 = p["top5"][0] if p["top5"] else {}
            L.append("| %s | %s | %s | %.6f | %s | %s(%.3f) |"
                     % (p["mapping"], p["order"], p["position"], p["raw_mass"],
                        fe(p["other_mass"]), (t1.get("string") or "").replace("|", "/"),
                        t1.get("prob", 0.0)))
        L.append("")
        L.append("판정: format validity %s (max other=%s)."
                 % ("정상" if mo < 0.01 else "주의", fe(mo)))

    # 8. Deviations / closure
    L.append(_h("## 8. Deviations and Closure"))
    L.append("- **M2 실행 차단 유지**: %s" % p_meta.get("m2_status"))
    L.append("- 본 실행은 승인된 단일 재설계 재실행이며, 결과와 무관하게 트랙을 "
             "완결한다. 프리프린트 조립로 이행 (spec Authorization/Section 4).")
    L.append("- S1/S2는 A/B 라벨 체계이며, 지표 해석은 층내로 한정한다. QQ 만족은 "
             "투영/대칭반복/order-matched mixture 등과의 양립 근거이며 기전 식별이나 "
             "양자성/역사 재현 주장이 아니다.")
    L.append("")

    text = "\n".join(L)
    if out_path is None:
        stamp2 = _dt.datetime.now().strftime("%Y-%m-%d_%H%M")
        out_path = os.path.join(
            ROOT, "runs", "%s_ex18_redesign_R1_trackP_report.md" % stamp2)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path, {"n_nonsat_P": n_nonsat, "n_items": n_items,
                      "success": success, "ids": ids}


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:]
    out, _ = generate_report(paths)
    print(out)
