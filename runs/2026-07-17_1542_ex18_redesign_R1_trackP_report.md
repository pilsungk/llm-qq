# ex18 Redesign Run R1 Report (Track P Minimal)

- Generated: 2026-07-17 15:42
- Spec: `ex18_redesign_run_spec_R1` (single approved rerun)
- Track: **P** (respondent simulation, system prompt id=T-1P)
- Track M baseline: `runs/pilot_v1_2_20260717_1247_qwen3_4b.json`
- Model-condition: M1 (`Qwen/Qwen3-4B-Instruct-2507`, bf16); M2 execution-blocked.

본 재실행은 **Track P 응답자 프레이밍이 출력 포화를 낮추는지**를 검정한다. 모델이 역사적 인간 판단을 재현하는지, 혹은 양자인지 여부를 검정하는 것이 **아니다**. QQ 만족은 투영측정 모델과의 양립 근거일 뿐 이다.

## 1. Execution Summary and Gates

- status=completed | G1=True G2=True(max_other=5.47e-08) G3=True

| gate | result |
|---|---|
| G1 A/B single-token (T-1P context) | PASS |
| G2 other-mass < 1e-2 all positions | PASS (max=5.47e-08) |
| G3 CP consistency (s1-01 x 2 orders) | PASS |
  - spot s1-01 AB: final_pass=True retry=not_needed n_rej=200
  - spot s1-01 BA: final_pass=True retry=not_needed n_rej=200

## 2. Track P Metadata

| field | value |
|---|---|
| system_prompt_id | T-1P |
| system_prompt_template | Answer each question as an average adult American survey respondent in {reference_year} would have answered. For each question, answer with exactly one letter: A or B. Do not explain. |
| historical_header_used | False |
| reference_years | {'s1-01': 1997, 's1-02': 1995, 's1-04': 1996, 's1-06': 1979, 's1-11': 2008, 's1-12': 2012, 's2-01': 1997, 's2-02': 1995} |
| seeds | {'python': 20260719, 'numpy': 20260719, 'torch': 20260719} |
| chat_template_sha256 | `64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326` |
| labels[AB] | variant=bare ids={'t1': 32, 't2': 33} |
| m2_status | execution_blocked (HF access + bitsandbytes dependency + sm_120 risk); recorded as in the v1.2 pilot |
| git | {'commit': None, 'dirty': None} |

## 3. Per-Item Pooled Results (Track P)

| id | year | OE_A | OE_B | OSS | q_QQ | q_env[min,max] | QQ | Gamma[lo,hi] | Gamma class | sat |
|---|---|---|---|---|---|---|---|---|---|---|
| s1-01 | 1997 | -0.5685 | +0.0000 | 0.5685 | -0.5685 | [-0.5685, -0.5685] | VIOLATED | [-0.0000, -0.0000] | NONCONTEXTUAL | Y |
| s1-02 | 1995 | +0.1126 | +0.0000 | 0.1126 | +0.1126 | [+0.1126, +0.1126] | VIOLATED | [-0.0000, +0.0000] | INDETERMINATE | Y |
| s1-04 | 1996 | +0.0637 | +0.5825 | 0.6461 | +0.6461 | [+0.6461, +0.6461] | VIOLATED | [-0.0000, -0.0000] | NONCONTEXTUAL | Y |
| s1-06 | 1979 | -0.9991 | -0.0000 | 0.9991 | -0.9991 | [-0.9991, -0.9991] | VIOLATED | [-0.0000, -0.0000] | NONCONTEXTUAL | Y |
| s1-11 | 2008 | -0.2842 | -0.6044 | 0.8886 | -0.2441 | [-0.2441, -0.2441] | VIOLATED | [-0.6445, -0.6445] | NONCONTEXTUAL | n |
| s1-12 | 2012 | +0.0001 | +0.0000 | 0.0001 | +0.0001 | [+0.0001, +0.0001] | SATISFIED | [-0.0000, +0.0000] | INDETERMINATE | Y |
| s2-01 | 1997 | +0.4937 | +0.5062 | 0.9998 | -0.9998 | [-0.9998, -0.9998] | VIOLATED | [-0.0000, +0.0000] | INDETERMINATE | Y |
| s2-02 | 1995 | -0.2420 | +0.2301 | 0.4721 | +0.0118 | [+0.0118, +0.0118] | SATISFIED | [-0.4602, -0.4602] | NONCONTEXTUAL | Y |

## 4. Track M vs Track P Comparison (same 8 items)

| id | sat_M | sat_P | q_QQ_M | q_QQ_P | OSS_M | OSS_P | Gamma_M | Gamma_P |
|---|---|---|---|---|---|---|---|---|
| s1-01 | Y | Y | +0.1821 | -0.5685 | 0.7421 | 0.5685 | NONCONTEXTUAL | NONCONTEXTUAL |
| s1-02 | Y | Y | +0.9806 | +0.1126 | 0.9806 | 0.1126 | INDETERMINATE | INDETERMINATE |
| s1-04 | Y | Y | +0.0015 | +0.6461 | 0.4059 | 0.6461 | NONCONTEXTUAL | NONCONTEXTUAL |
| s1-06 | Y | Y | -0.0058 | -0.9991 | 0.0090 | 0.9991 | NONCONTEXTUAL | NONCONTEXTUAL |
| s1-11 | Y | n | -0.0489 | -0.2441 | 0.7409 | 0.8886 | NONCONTEXTUAL | NONCONTEXTUAL |
| s1-12 | Y | Y | +0.0055 | +0.0001 | 0.0055 | 0.0001 | NONCONTEXTUAL | INDETERMINATE |
| s2-01 | Y | Y | +0.1817 | -0.9998 | 0.8039 | 0.9998 | NONCONTEXTUAL | INDETERMINATE |
| s2-02 | Y | Y | +0.0000 | +0.0118 | 0.0003 | 0.4721 | NONCONTEXTUAL | NONCONTEXTUAL |

포화 요약: Track M 8/8 포화 -> Track P 7/8 포화 (비포화 Track P = 1/8).

## 5. Primary Outcome (spec Section 4)

- 비포화 문항 (Track P): **1 / 8**
- **판정: 구성 개념 불변의 포화 (비포화 < 4/8).** 단일 모델의 binary-conditioned 분포는 persona(응답자 모사) 조건에서도 결정론적이라는 발견으로 보고한다.
- 어느 경우든 본 트랙은 이 보고서(불변 레코드)로 **완결**되며, 후속 실행/확장은 없다 (spec Section 4).

## 6. Supplementary Observations (report-only)

### Track M -> Track P shift (q_QQ, OSS)
| id | dq_QQ (P-M) | dOSS (P-M) |
|---|---|---|
| s1-01 | -0.7506 | -0.1736 |
| s1-02 | -0.8680 | -0.8680 |
| s1-04 | +0.6447 | +0.2403 |
| s1-06 | -0.9933 | +0.9902 |
| s1-11 | -0.1952 | +0.1476 |
| s1-12 | -0.0054 | -0.0055 |
| s2-01 | -1.1816 | +0.1959 |
| s2-02 | +0.0118 | +0.4718 |

### Matched S1/S2 OSS divergence (Track P)
| S2 | S1 | OSS_S2 | OSS_S1 | |dOSS| |
|---|---|---|---|---|
| s2-01 | s1-01 | 0.9998 | 0.5685 | 0.4313 |
| s2-02 | s1-02 | 0.4721 | 0.1126 | 0.3595 |

주: 인간 벤치마크(QuestOrdData) 미수신 -- 절대 수준·역사 재현 해석 유보. 방향 수준 정성 관찰만 가능.

## 7. s2-02 Monitoring (Track P)

| mapping | order | position | raw_mass | other_mass | top1 |
|---|---|---|---|---|---|
| map-1 | AB | first | 1.000000 | -3.49e-08 | A(0.996) |
| map-1 | AB | second_given_t1 | 1.000000 | -1.96e-08 | A(1.000) |
| map-1 | AB | second_given_t2 | 1.000000 | -6.91e-13 | B(1.000) |
| map-1 | BA | first | 1.000000 | -1.96e-08 | A(0.989) |
| map-1 | BA | second_given_t1 | 1.000000 | -1.25e-09 | A(1.000) |
| map-1 | BA | second_given_t2 | 1.000000 | -3.27e-13 | B(1.000) |
| map-2 | AB | first | 1.000000 | 0.00e+00 | B(0.500) |
| map-2 | AB | second_given_t1 | 1.000000 | -3.10e-12 | A(1.000) |
| map-2 | AB | second_given_t2 | 1.000000 | -7.45e-09 | B(0.953) |
| map-2 | BA | first | 1.000000 | -1.86e-08 | A(0.977) |
| map-2 | BA | second_given_t1 | 1.000000 | -4.19e-13 | A(1.000) |
| map-2 | BA | second_given_t2 | 1.000000 | -5.38e-08 | B(0.999) |

판정: format validity 정상 (max other=0.00e+00).

## 8. Deviations and Closure

- **M2 실행 차단 유지**: execution_blocked (HF access + bitsandbytes dependency + sm_120 risk); recorded as in the v1.2 pilot
- 본 실행은 승인된 단일 재설계 재실행이며, 결과와 무관하게 트랙을 완결한다. 프리프린트 조립로 이행 (spec Authorization/Section 4).
- S1/S2는 A/B 라벨 체계이며, 지표 해석은 층내로 한정한다. QQ 만족은 투영/대칭반복/order-matched mixture 등과의 양립 근거이며 기전 식별이나 양자성/역사 재현 주장이 아니다.
