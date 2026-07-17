# ex18 Minimal Pilot Execution Spec v1.2
## Track M / binary_logprob QQ Audit — First-Signal Run (**실행 기준본 — 동결**)

- **Document ID**: ex18_minimal_pilot_spec_v1_2
- **Date**: 2026-07-17 05:25 KST
- **Supersedes**: v1.1 (최종 정정 R1~R2 + 동결 전 정정 S1~S3 반영 — **동결 완료**, 이후 설계 확장 없이 실행)
- **Governing docs**: SRS v0.7 (사전 등록 판정 규칙 전부 적용),
  `ex18_item_sets_draft_v0_6.md` + v1.0 Annex A (개정 3건 유지),
  이론 노트 final, 스모크 보고서 2026-07-17_0249
- **Scope 원칙**: first-signal 파일럿. 본실험급 통제는 전부 이연 (Annex B).

---

## 0. Changes (v1.0 -> v1.2)

| ID | 변경 |
|---|---|
| Q1 | **Pooling 규칙 확정**: 지표 평균 금지. 주 결과 = 두 매핑 결합확률의 동일 가중 평균(pooled joints)에서 q_QQ/OSS/Gamma 재계산. 구현: q, OE_A, OE_B는 선형 -> pooled envelope = 매핑별 envelope의 interval 합산 (1/2)[min1+min2, max1+max2] (배분 변수 독립, 정확); |.|·Gamma는 이론 노트 3.5.1 절차. 매핑별 결과는 label-effect 진단 병기. 이론 근거: 동일 가중·순서 공통 혼합 = Prop 6 order-matched mixing (QQ 지위 보존 연산) |
| **R1** | **[v1.2] Canonicalization 선행**: pooling 전에 각 매핑의 측정 결과를 **원 의미 옵션 좌표**(예: Approve/Disapprove, Yes/No)로 역매핑한 뒤 결합확률을 평균한다. map-2의 (A,B) 토큰 좌표를 그대로 평균하면 반대 의미 셀이 혼합됨 — 금지. 역매핑은 셀 순열(선형)이므로 Q1의 interval 합산 규칙은 canonicalize 이후 좌표에서 동일 성립 |
| **R2** | **[v1.2] G3' 정의 명문화 + family 회귀**: binary_rejection은 primitive 조립이 아니라 **전체 순차 궤적 샘플링** (첫 답 샘플 -> 실제 답을 이력에 -> 둘째 답 샘플; 수락 (a,b) 쌍 = joint 표본 1개; 셀 카운트 = multinomial(n_rej, p_joint)). CP 검사는 SRS 4.6 등록 정의 그대로: **item-order당 4셀, alpha = 0.01** (합성 보정 0.9951로 검증된 세팅). v1.1 Q5의 모델당 16셀 family는 SRS 이탈이므로 철회. 모델당 4회 검사의 설계 오경보 ~2%는 G3 불통과 절차(확대 재실행 1회)가 흡수 |
| **S1** | **[동결 전 정정] G3' 오경보 수치**: 모델당 명목 상한 = union bound **<= 4%** (검사당 alpha=1% x 4회); 합성 검증에서 관찰된 경험적 비율은 약 2% (0.9951^4). "~2%" 단독 표기는 명목/경험 혼동 — 정정 |
| **S2** | **[동결 전 정정] 재현 조건 교체**: binary_logprob은 결정론적 -> 시드 변경은 독립 재현이 아님. 즉시 후속 조건 (iii) = **타 모델 재현; 불발 시 사전 고정 제2 템플릿 T-2로 후속 단계 확인**. T-2 사전 고정: "This is a survey. Respond to each item with exactly one letter: A or B." (S3: "... with exactly one word: Yes or No.") |
| **S3** | **[동결 전 정정] framing 헤더 위치**: 순차 실행당 **1회, 첫 질문 전에만** 제시 (이력 보존). 두 번째 질문 앞 재삽입 금지 — 질문 사이 문맥 삽입은 인접성 전제 훼손 (s1-03에서 문제 삼은 조건과 동일) |
| Q2 | **M2 사전 동결 + 프레이밍 교정**: 본 파일럿은 모델 비교가 아니라 **두 model-condition (bf16 4B / 8-bit 8B) 재현성 점검**. M2 = Llama-3.1-8B-Instruct 8-bit 동결; 폴백 발동은 "실행일까지 HF 승인 부재"(외부 사건)만, 폴백 = Mistral-7B-Instruct-v0.3 **8-bit** (조건 대칭) |
| Q3 | **역사 문항 처리**: s1-08 파일럿 제외 (현재시제 역사 전제 + 연도 미확정) -> 18쌍. s1-05, s1-11에 **historical framing 헤더** 부여 (Section 2.1) |
| Q4 | **Go/no-go 수치화** (Section 6) — 사전 등록 추론 임계가 아닌 자원 배분 결정 규칙임을 명시 |
| Q5 | (v1.2에서 R2로 대체 — 철회) |
| Q6 | **층간 수치 비교 금지**: S1/S2 (A/B label)와 S3 (Yes/No)는 label 체계가 다르므로 지표의 층간 직접 수치 비교를 하지 않는다 (층내 보고만) |

## 1. Configuration

| 항목 | 값 |
|---|---|
| Construct | Track M만 |
| Protocol | binary_logprob (주) + binary_rejection 스팟체크 |
| Template | T-1 단일 |
| Label | S1/S2: 원 옵션 -> A/B, counterbalancing 2매핑 (동일 매핑 AB/BA 공통); S3: Yes/No |
| Pooling | **canonicalize -> pool** (R1): 의미 옵션 좌표 역매핑 후 동일 가중 평균 -> 지표 재계산 (Q1) |
| M1 | Qwen/Qwen3-4B-Instruct-2507, bf16 |
| M2 (동결) | meta-llama/Llama-3.1-8B-Instruct, 8-bit (bitsandbytes; 양자화 감사 기록). 폴백(승인 부재 시만): mistralai/Mistral-7B-Instruct-v0.3, 8-bit |
| Decoding / Seed | T=1, top_p=1, top_k off / 20260718 |

## 2. Pilot Item Set (18쌍; 원문은 v0.6 마스터 + v1.0 Annex A)

- **S1 (10)**: s1-01, 02, 04, 05, 06, 07, 09, 10, 11, 12
  (s1-08 제외 — Q3. sensitive 필드는 기록만, 이중집계 이연)
- **S3 (5)**: s3-01, 03, 08, 09, 11
- **S2 (3)**: s2-01, s2-02 (모니터링 플래그), s2-09

### 2.1 Historical framing header (Q3)

대상: s1-05 (냉전기 어법), s1-11 ("the past presidential election"이 2026
현재 시점에서 오독됨). **순차 실행당 1회, 첫 질문 user 턴 앞에만** 1문장
헤더 (S3 정정 — 이력에 남으므로 재삽입 금지; 양 순서 대칭 적용):

```
"The following question is from a survey conducted {in 2008 | during the
Cold War era}. Answer the question as written, based on your own
assessment of that period."
```

- Track M 유지 (모델 자신의 평가; 응답자 모사 아님). 헤더 문구는 감사
  레코드에 기록. 기타 문항은 헤더 없음 (현재 시점 평가 가능 확인:
  s1-09 현직 의회, s1-10 과거시제, s1-01/02 인물 평가).

## 3. Run Matrix & Estimate

```
S1+S2 13쌍 x (3 forwards x 2 orders x 2 mappings) = 156
S3 5쌍 x (3 x 2 x 1) = 30   -> 186 forwards/model, 2 models ~ 370 forwards
```

RTX 5070 Ti 기준 분 단위. 실행 순서: M1 전체 -> sanity -> M2 전체.
감사 레코드: 스모크 스키마 + mapping, framing header, quantization 필드.

## 4. Gates

1. G1' — 모델별 A/B·Yes/No 단일 토큰 검증 (렌더 문맥)
2. G2' — other mass 전 위치 < 1e-2; 문항별 raw mass 기록
3. G3' — 스팟체크: 모델당 s1-01, s3-01 x 2순서, n_rej = 200 수락 joint
   표본 (**전체 순차 궤적 샘플링** — R2). CP 검사는 SRS 4.6 등록 정의:
   **item-order당 4셀 Bonferroni, alpha = 0.01**. 모델당 4회 검사의 설계
   오경보: 명목 상한 <= 4% (union bound), 합성 검증에서 관찰된 경험적
   비율은 약 2% — 불통과 절차 (확대 재실행 1회 -> 조사·중단)가 흡수
4. s2-02 모니터링 (raw mass·top-5); M2 양자화 경로는 G1'~G3' 필수

## 5. Metrics & Outputs

- 문항별 (pooled 주 + 매핑별 진단): q_QQ envelope 판정, OE_A/OE_B/OSS,
  Gamma 경계 판정, 포화율
- **포화 정의**: 문항의 측정 위치 중 max(p, 1-p) > 0.99인 비율 > 50%이면
  "포화 문항" (모델별)
- 매핑 간 차이 (label-effect), matched S1-S2 OSS 괴리 (3쌍)
- **층간 비교 금지 (Q6)**: S1/S2 vs S3 수치 비교 없음 — 층내 보고만
- 산출물: runs/ 감사 레코드 + 파일럿 보고서 (불변 레코드)

## 6. Go/No-Go 결정 규칙 (Q4 — 수치화)

주: 아래 수치는 사전 등록 추론 임계가 아니라 **후속 자원 배분용 결정
규칙**이다. 판정 자체는 SRS 사전 등록 규칙(envelope, eps)만 사용.

- **즉시 후속 (맥락성 신호)**: 어떤 문항에서 certified Gamma_lower > 0이
  (i) pooled 결과와 (ii) 두 매핑 각각에서 모두 성립하고, (iii) **타 모델에서
  같은 문항이 재현**될 때 (S2 정정: logprob은 결정론적이라 시드 재현 무효;
  타 모델 불발 시 사전 고정 T-2 템플릿 확인을 후속 단계 1차 작업으로)
- **체계적 위반**: 한 모델에서 비포화 문항 중 VIOLATED >= 4개이고 그중
  >= 2개가 타 모델에서 재현 -> 폐형식(polarity/position) 적합을 Phase 2로
- **광범위 만족**: 양 모델에서 비포화 문항의 SATISFIED 비율 >= 70%이고
  OSS > 0.05인 비포화 문항이 >= 5개 -> SI 상관 진단 corr(OE, p_A - p_B)을
  Phase 2로
- **재설계**: 모델별 비포화 문항 < 8개 또는 INDETERMINATE 비율 > 30%
  -> 문항·템플릿 개선 후 재실행 (확장 금지)
- **암기 경보**: matched 3쌍 중 >= 2쌍에서 |OSS_S1 - OSS_S2| > 0.3
  -> S2 확장을 후속 최우선으로

## 7. Deferred (Annex B, v1.0과 동일)

Track P, S2 전체, 민감도 이중집계, RMC, context_stress·designed context
manipulation (Phase 2 백로그 승인됨), arm 2, option-order reversal,
natural_generation 전면, Yes/No 재구성 민감도, sensitive_exploratory,
QuestOrdData 인간 비교 (수신 시 별도 노트), s1-08 (연도 고정 + framing
확정 후 복귀).

## Annex A — Item Set 개정 3건 (v1.0 Annex A 유지)

A-1 s3-05 -> 표준 exploratory QQ (사유 정정) / A-2 s2-06 재작성: 도서관
장서 제거 weak/strong 쌍 제안 + 배심원 면제 대안 (PI 택일 대기) /
A-3 option-order reversal 삭제. v0.7 통합은 QuestOrdData 수신 시 1회.
