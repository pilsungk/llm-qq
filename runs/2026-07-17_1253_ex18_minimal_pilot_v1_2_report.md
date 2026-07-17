# ex18 Minimal Pilot v1.2 Report

- Generated: 2026-07-17 12:53
- Spec: `ex18_minimal_pilot_spec_v1_2` (frozen)
- Model-conditions: M1
- Protocol: binary_logprob (primary) + binary_rejection spot check

본 보고서는 first-signal 파일럿 실행 레코드이다. QQ 만족은 투영측정 모델과의 **양립(compatibility)** 근거일 뿐 양자인지/양자성의 증명이 아니다. S1/S2(A/B label)와 S3(Yes/No)는 label 체계가 달라 지표를 **층간 직접 비교하지 않는다** (Section 14).

## 1. Execution Summary

- **M1** (`Qwen/Qwen3-4B-Instruct-2507`): status=completed | G1=True G2=True(max_other=2.22e-07) G3=True

실행 순서: M1 전체 -> sanity -> (M2 조건부). 각 조건은 독립 감사 레코드를 가진다.

**M2 미실행 (실행 차단)**: 두 번째 model-condition은 모델 접근 불가(HF 승인/토큰 부재) + 금지된 신규 의존성(bitsandbytes) + Blackwell sm_120 호환성 위험으로 실행 시점에 차단되었다. 이는 M2 모델 실패가 아니라 환경/접근 차단이며, PI 결정에 따라 M1 단독 으로 최종화한다 (Section 12/13).

## 2. Model-Condition Metadata

### M1
| field | value |
|---|---|
| model_id | Qwen/Qwen3-4B-Instruct-2507 |
| model_revision | Qwen/Qwen3-4B-Instruct-2507 |
| dtype | bfloat16 |
| quantization | none |
| chat_template_sha256 | `64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326` |
| no_thinking_tags | True |
| seeds | {'python': 20260718, 'numpy': 20260718, 'torch': 20260718} |
| decoding | {'temperature': 1.0, 'top_p': 1.0, 'top_k': 0, 'do_sample': True, 'max_new_tokens': 1} |
| cuda | NVIDIA GeForce RTX 5070 Ti cc=[12, 0] 16.61GB |
| versions | {'torch': '2.11.0+cu128', 'transformers': '5.14.1', 'accelerate': '1.14.0', 'bitsandbytes': None} |
| git | {'commit': None, 'dirty': None} |
| condition_reason | primary model-condition (bf16 4B) | M1 primary; M2 pending dependency/access decision |
| labels[AB] | variant=bare ids={'t1': 32, 't2': 33} |
| labels[YesNo] | variant=bare ids={'t1': 9454, 't2': 2753} |


## 3. Gate Table (G1 / G2 / G3)

| condition | G1 labels | G2 other-mass | G3 consistency |
|---|---|---|---|
| M1 | PASS | PASS (max=2.22e-07, thr<1e-2) | PASS |

G3 주: 모델당 명목 union-bound 오경보 상한은 4셀 x 4검사 alpha=1% 에서 **<= 4%**. 합성 검증에서 관찰된 약 2% (0.9951^4)는 명목 보증이 아니라 사전 합성 보정의 **경험적 비율**이다.

## 4. Per-Item Pooled Results

| id | stratum | OE_A | OE_B | OSS | q_QQ(point) | q_env[min,max] | QQ | Gamma[lo,hi] | Gamma class | sat |
|---|---|---|---|---|---|---|---|---|---|---|
| **M1** | | | | | | | | | | |
| s1-01 | S1 | -0.2800 | +0.4621 | 0.7421 | +0.1821 | [+0.1821, +0.1821] | VIOLATED | [-0.5600, -0.5600] | NONCONTEXTUAL | Y |
| s1-02 | S1 | +0.4825 | +0.4981 | 0.9806 | +0.9806 | [+0.9806, +0.9806] | VIOLATED | [-0.0000, +0.0000] | INDETERMINATE | Y |
| s1-04 | S1 | +0.2158 | -0.1900 | 0.4059 | +0.0015 | [+0.0015, +0.0015] | SATISFIED | [-0.4044, -0.4044] | NONCONTEXTUAL | Y |
| s1-05 | S1 | -0.0014 | -0.7313 | 0.7327 | -0.7300 | [-0.7300, -0.7300] | VIOLATED | [-0.0027, -0.0027] | NONCONTEXTUAL | Y |
| s1-06 | S1 | -0.0074 | +0.0016 | 0.0090 | -0.0058 | [-0.0058, -0.0058] | SATISFIED | [-0.0032, -0.0032] | NONCONTEXTUAL | Y |
| s1-07 | S1 | -0.3869 | +0.3869 | 0.7737 | -0.0000 | [-0.0000, -0.0000] | SATISFIED | [-0.7737, -0.7737] | NONCONTEXTUAL | Y |
| s1-09 | S1 | -0.1314 | -0.1779 | 0.3093 | -0.3090 | [-0.3090, -0.3090] | VIOLATED | [-0.0003, -0.0003] | NONCONTEXTUAL | Y |
| s1-10 | S1 | +0.0010 | +0.0238 | 0.0248 | -0.0229 | [-0.0229, -0.0229] | VIOLATED | [-0.0020, -0.0020] | NONCONTEXTUAL | Y |
| s1-11 | S1 | -0.3460 | -0.3949 | 0.7409 | -0.0489 | [-0.0489, -0.0489] | VIOLATED | [-0.6921, -0.6921] | NONCONTEXTUAL | Y |
| s1-12 | S1 | +0.0055 | +0.0000 | 0.0055 | +0.0055 | [+0.0055, +0.0055] | SATISFIED | [-0.0001, -0.0001] | NONCONTEXTUAL | Y |
| s2-01 | S2 | +0.4928 | -0.3111 | 0.8039 | +0.1817 | [+0.1817, +0.1817] | VIOLATED | [-0.6222, -0.6222] | NONCONTEXTUAL | Y |
| s2-02 | S2 | -0.0002 | +0.0001 | 0.0003 | +0.0000 | [+0.0000, +0.0000] | SATISFIED | [-0.0003, -0.0003] | NONCONTEXTUAL | Y |
| s2-09 | S2 | -0.0511 | -0.5442 | 0.5954 | +0.4676 | [+0.4676, +0.4676] | VIOLATED | [-0.1278, -0.1278] | NONCONTEXTUAL | Y |
| s3-01 | S3 | +0.0000 | -0.5622 | 0.5622 | +0.5622 | [+0.5622, +0.5622] | VIOLATED | [-0.0000, +0.0000] | INDETERMINATE | Y |
| s3-03 | S3 | -0.0067 | +0.4933 | 0.5000 | +0.5000 | [+0.5000, +0.5000] | VIOLATED | [-0.0000, -0.0000] | NONCONTEXTUAL | Y |
| s3-08 | S3 | +0.3775 | +0.9911 | 1.3686 | -0.6142 | [-0.6142, -0.6142] | VIOLATED | [-0.7544, -0.7544] | NONCONTEXTUAL | Y |
| s3-09 | S3 | -0.0440 | -0.7309 | 0.7749 | +0.6870 | [+0.6870, +0.6870] | VIOLATED | [-0.0879, -0.0879] | NONCONTEXTUAL | Y |
| s3-11 | S3 | -0.9241 | -0.0345 | 0.9586 | -0.8896 | [-0.8896, -0.8896] | VIOLATED | [-0.0690, -0.0690] | NONCONTEXTUAL | n |


## 5. Per-Mapping Diagnostic Results (label-effect)

A/B 문항만 매핑별 진단을 가진다 (S3는 Yes/No 단일).

### M1
| id | mapping | OSS | q_QQ | QQ | Gamma class |
|---|---|---|---|---|---|
| s1-01 | map-1 | 0.5600 | -0.5600 | VIOLATED | NONCONTEXTUAL |
| s1-01 | map-2 | 0.9242 | +0.9241 | VIOLATED | NONCONTEXTUAL |
| s1-02 | map-1 | 0.9627 | +0.9627 | VIOLATED | INDETERMINATE |
| s1-02 | map-2 | 0.9985 | +0.9985 | VIOLATED | INDETERMINATE |
| s1-04 | map-1 | 0.0058 | +0.0000 | SATISFIED | NONCONTEXTUAL |
| s1-04 | map-2 | 0.8175 | +0.0030 | SATISFIED | NONCONTEXTUAL |
| s1-05 | map-1 | 0.9629 | -0.9624 | VIOLATED | NONCONTEXTUAL |
| s1-05 | map-2 | 0.5025 | -0.4975 | VIOLATED | NONCONTEXTUAL |
| s1-06 | map-1 | 0.0141 | -0.0141 | SATISFIED | INDETERMINATE |
| s1-06 | map-2 | 0.0039 | +0.0024 | SATISFIED | NONCONTEXTUAL |
| s1-07 | map-1 | 1.3582 | -0.0000 | SATISFIED | NONCONTEXTUAL |
| s1-07 | map-2 | 0.1893 | +0.0000 | SATISFIED | NONCONTEXTUAL |
| s1-09 | map-1 | 0.6183 | -0.6182 | VIOLATED | NONCONTEXTUAL |
| s1-09 | map-2 | 0.0043 | +0.0002 | SATISFIED | NONCONTEXTUAL |
| s1-10 | map-1 | 0.0494 | -0.0455 | VIOLATED | NONCONTEXTUAL |
| s1-10 | map-2 | 0.0003 | -0.0002 | SATISFIED | NONCONTEXTUAL |
| s1-11 | map-1 | 1.4563 | -0.0983 | VIOLATED | NONCONTEXTUAL |
| s1-11 | map-2 | 0.0256 | +0.0005 | SATISFIED | NONCONTEXTUAL |
| s1-12 | map-1 | 0.0000 | +0.0000 | SATISFIED | INDETERMINATE |
| s1-12 | map-2 | 0.0111 | +0.0109 | SATISFIED | NONCONTEXTUAL |
| s2-01 | map-1 | 1.5619 | +0.3634 | VIOLATED | NONCONTEXTUAL |
| s2-01 | map-2 | 0.0459 | -0.0000 | SATISFIED | NONCONTEXTUAL |
| s2-02 | map-1 | 0.0006 | +0.0001 | SATISFIED | NONCONTEXTUAL |
| s2-02 | map-2 | 0.0000 | +0.0000 | SATISFIED | INDETERMINATE |
| s2-09 | map-1 | 0.2146 | -0.0239 | VIOLATED | NONCONTEXTUAL |
| s2-09 | map-2 | 0.9762 | +0.9590 | VIOLATED | NONCONTEXTUAL |
| s3-01 | yesno | 0.5622 | +0.5622 | VIOLATED | INDETERMINATE |
| s3-03 | yesno | 0.5000 | +0.5000 | VIOLATED | NONCONTEXTUAL |
| s3-08 | yesno | 1.3686 | -0.6142 | VIOLATED | NONCONTEXTUAL |
| s3-09 | yesno | 0.7749 | +0.6870 | VIOLATED | NONCONTEXTUAL |
| s3-11 | yesno | 0.9586 | -0.8896 | VIOLATED | NONCONTEXTUAL |


## 6. QQ Classification Counts (pooled)

| condition | SATISFIED | VIOLATED | INDETERMINATE | (non-sat) S/V/I |
|---|---|---|---|---|
| M1 | 5 | 13 | 0 | 0/1/0 |


## 7. Gamma Classification Counts (pooled)

| condition | NONCONTEXTUAL | CONTEXTUAL | INDETERMINATE | certified (pooled+maps) |
|---|---|---|---|---|
| M1 | 16 | 0 | 2 | none |


## 8. Saturation Table

포화: 문항 측정 위치 중 max(p,1-p)>0.99 비율 > 50%.
| condition | saturated items | count |
|---|---|---|
| M1 | s1-01, s1-02, s1-04, s1-05, s1-06, s1-07, s1-09, s1-10, s1-11, s1-12, s2-01, s2-02, s2-09, s3-01, s3-03, s3-08, s3-09 | 17 |


## 9. Label-Effect Diagnostics (map-1 vs map-2)

| condition | id | dOSS | dq_QQ | q_QQ map-1 | q_QQ map-2 |
|---|---|---|---|---|---|
| M1 | s1-01 | 0.3642 | 1.4841 | -0.5600 | +0.9241 |
| M1 | s1-02 | 0.0358 | 0.0358 | +0.9627 | +0.9985 |
| M1 | s1-04 | 0.8117 | 0.0030 | +0.0000 | +0.0030 |
| M1 | s1-05 | 0.4605 | 0.4649 | -0.9624 | -0.4975 |
| M1 | s1-06 | 0.0101 | 0.0165 | -0.0141 | +0.0024 |
| M1 | s1-07 | 1.1689 | 0.0000 | -0.0000 | +0.0000 |
| M1 | s1-09 | 0.6140 | 0.6185 | -0.6182 | +0.0002 |
| M1 | s1-10 | 0.0490 | 0.0453 | -0.0455 | -0.0002 |
| M1 | s1-11 | 1.4308 | 0.0988 | -0.0983 | +0.0005 |
| M1 | s1-12 | 0.0111 | 0.0109 | +0.0000 | +0.0109 |
| M1 | s2-01 | 1.5159 | 0.3635 | +0.3634 | -0.0000 |
| M1 | s2-02 | 0.0006 | 0.0001 | +0.0001 | +0.0000 |
| M1 | s2-09 | 0.7616 | 0.9829 | -0.0239 | +0.9590 |


## 10. S1/S2 Matched-Pair Descriptive Comparison

| condition | S2 | S1 | OSS_S2 | OSS_S1 | |dOSS| | >0.3 |
|---|---|---|---|---|---|---|
| M1 | s2-01 | s1-01 | 0.8039 | 0.7421 | 0.0618 | n |
| M1 | s2-02 | s1-02 | 0.0003 | 0.9806 | 0.9803 | Y |
| M1 | s2-09 | s1-09 | 0.5954 | 0.3093 | 0.2861 | n |

주: 기술 비교이며 인간 벤치마크(QuestOrdData) 미수신 상태 — 절대 수준 해석 유보.

## 11. s2-02 Monitoring (Armey / Lott format validity)

### M1
| mapping | order | position | raw_mass | other_mass | top1 |
|---|---|---|---|---|---|
| map-1 | AB | first | 1.000000 | -5.38e-08 | B(0.999) |
| map-1 | AB | second_given_t1 | 1.000000 | 0.00e+00 | A(0.924) |
| map-1 | AB | second_given_t2 | 1.000000 | -2.09e-14 | B(1.000) |
| map-1 | BA | first | 1.000000 | 5.45e-08 | B(0.999) |
| map-1 | BA | second_given_t1 | 1.000000 | -4.14e-08 | A(1.000) |
| map-1 | BA | second_given_t2 | 1.000000 | -4.19e-13 | B(1.000) |
| map-2 | AB | first | 1.000000 | 2.80e-09 | A(1.000) |
| map-2 | AB | second_given_t1 | 1.000000 | -9.35e-14 | A(1.000) |
| map-2 | AB | second_given_t2 | 1.000000 | 1.26e-08 | A(0.997) |
| map-2 | BA | first | 1.000000 | -1.67e-08 | A(1.000) |
| map-2 | BA | second_given_t1 | 1.000000 | -1.98e-13 | A(1.000) |
| map-2 | BA | second_given_t2 | 1.000000 | -2.14e-08 | B(1.000) |

판정: raw mass 및 top-5 기준 format validity 정상 (max other=5.45e-08). 인지도 열위로 인한 무효 응답 급증 징후는 관찰되지 않음.


## 12. Failures, Retries, and Deviations

### M1
- no retries or gate failures
- **Execution-blocking deviation (M2 not run)**: the second model-condition was blocked at execution time by three independent causes: (i) the gated models `meta-llama/Llama-3.1-8B-Instruct` and the frozen fallback `mistralai/Mistral-7B-Instruct-v0.3` were inaccessible (no Hugging Face approval/token present); (ii) 8-bit loading requires `bitsandbytes`, a new dependency prohibited without prior approval (CLAUDE.md); (iii) unresolved Blackwell sm_120 compatibility risk for bitsandbytes 8-bit. This is an environment/access block, NOT an M2 model failure. Per PI decision the run is finalized M1-only without modifying the frozen design or adding conditions.


## 13. Registered Go/No-Go Evaluation

주: 아래는 사전 등록 추론 임계가 아니라 **후속 자원 배분 결정 규칙**이다.
- **Immediate contextuality follow-up**: False NOT EVALUABLE: cross-model reproduction requires M2, which was execution-blocked (no certified Gamma_lower > 0 in M1)
- **Systematic violation**: False NOT EVALUABLE: requires two model-conditions; M2 execution-blocked
- **Broad satisfaction**: False NOT EVALUABLE: requires both models; M2 execution-blocked
- **Redesign**: True {'M1': {'n_nonsat': 1, 'indet_rate': 0.0, 'flag': True}}
- **Memorization alert**: {'M1': False}

### Applicable M1-only resource decision: **REDESIGN / NO EXPANSION**

- 이 단일 조건 실행에서 적용되는 자원 배분 결정은 **REDESIGN / NO EXPANSION**이다. 근거: 18문항 중 17문항이 포화(saturated)되어 **비포화 문항은 1개(< 8)**이며, 이는 프레임된 redesign 규칙을 직접 발동한다. 문항·템플릿 개선 후 재실행이 필요하다 (확장 금지).
- **Cross-model rules (immediate contextuality / systematic violation / broad satisfaction)**: **NOT EVALUABLE** — 두 model-condition 재현이 필요하나 M2가 실행 차단되었다.
- **T-2 미해당**: T-2 사전 고정 템플릿은 M1의 certified contextuality 신호(Gamma_lower > 0)가 M2 재현에 실패했을 때만 적용된다. M1은 certified Gamma_lower > 0을 산출하지 않았으므로 (0 CONTEXTUAL), T-2는 후속 작업으로 등록되지 않는다.
- **암기 경보(memorization)**: 3개 matched S1/S2 쌍 기준 산출 (Section 10). 단, 전 문항 포화로 OSS 해석이 제한적임을 병기한다.

## 14. Cross-Stratum Comparison Statement

S1/S2 문항은 A/B 중립 토큰(원 옵션 매핑), S3 문항은 Yes/No 라벨을 사용한다. 라벨 체계가 다르므로 두 층의 지표(q_QQ, OSS, Gamma 등)를 **층간 직접 수치 비교하지 않는다**. 각 층 내부에서만 해석한다. QQ 만족은 투영측정/대칭반복/order-matched mixture 등과의 양립 근거이며 기전 식별이나 양자성 주장이 아니다.
