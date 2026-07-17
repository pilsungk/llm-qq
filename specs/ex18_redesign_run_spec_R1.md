# ex18 Redesign Run Spec R1 — Track P Minimal (승인된 재설계 1회)
## Saturation-Targeted Rerun under the Frozen Redesign Rule

- **Document ID**: ex18_redesign_run_spec_R1
- **Date**: 2026-07-17 KST
- **Authorization**: v1.2 동결 규칙 "재설계: 문항·템플릿 개선 후 재실행
  (확장 금지)" + PI 결정 (B안). 본 실행 후 결과와 무관하게 **트랙 완결**
  (보고서 -> 프리프린트 조립). 추가 실행 없음.
- **Baseline**: M1 파일럿 레코드 `runs/pilot_v1_2_20260717_1247_qwen3_4b.json`
  + 보고서 2026-07-17_1253 (Track M 대조군으로 사용)

## 1. What Changes (단 하나)

- **시스템 프롬프트만 교체**: Track M -> **Track P** (응답자 모사).
  진단된 포화 원인 = 단일 정책 측정이므로, 모집단 모사가 비포화 분포를
  유도하는지가 본 실행의 1차 질문.

```
Track P system prompt (T-1P, 사전 고정):
"Answer each question as an average adult American survey respondent in
{reference_year} would have answered. For each question, answer with
exactly one letter: A or B. Do not explain."
```

- {reference_year}는 문항별 치환. **historical framing 헤더는 사용하지
  않음** — persona가 시점을 지정하므로 이중 문맥 방지 (인접성 유지).
- 그 외 전부 동일: 파이프라인, 매핑 counterbalancing + canonicalize->pool,
  디코딩(T=1, top_p=1), 지표·판정(SRS 규칙), 게이트.

## 2. Items (8쌍 — reference_year 확정분만)

| id | year | 비고 |
|---|---|---|
| s1-01 | 1997 | Clinton/Gore |
| s1-02 | 1995 | Gingrich/Dole |
| s1-04 | 1996 | (A few/Many), sensitive 기록 |
| s1-06 | 1979 | sensitive 기록 |
| s1-11 | 2008 | 캠페인 상호 비판 |
| s1-12 | 2012 | AA, sensitive 기록 |
| s2-01 | 1997 (상속) | matched s1-01 |
| s2-02 | 1995 (상속) | matched s1-02, 모니터링 유지 |

- 제외: s1-05/07 (연도 미확정), s1-09/10 (Pew 행 TBD), s2-09 (상속 불가),
  S3 (Track P 미적용 원칙 — v0.4 N2)
- Forwards: 8 x 12 = 96 (M1 전용; M2 실행 차단 상태 유지 기록)

## 3. Gates & Spot Check

- G1/G2 동일 (A/B 단일 토큰, other mass < 1e-2)
- G3: s1-01 x 2 orders, n_rej = 200, SRS 4.6 정의 (item-order당 4셀,
  alpha = 0.01). Track P 프롬프트 하에서 궤적 샘플링 재확인 목적
- Seed: 20260719

## 4. Primary Outcome & Closure Rule

- **1차 결과 = 포화율 변화** (Track M 동일 8쌍 대비):
  - 비포화 >= 4/8: Track P가 분포를 여는 데 성공 — 전체 지표(q_QQ
    envelope, OSS, Gamma, label-effect)를 유의미한 first-signal로 보고
  - 비포화 < 4/8: **"구성 개념 불변의 포화"** — 단일 모델의 binary-
    conditioned 분포는 persona 조건에서도 결정론적이라는 발견으로 보고
- 어느 경우든: 보고서 1편 (불변 레코드) 작성 -> **트랙 완결, 프리프린트
  조립로 이행**. 후속 실행·확장 없음.
- 보조 관찰 (보고만): Track M 대비 동일 문항의 q/OSS 변화, 인간 방향과의
  정성 비교 (s1-01 등 — QuestOrdData 미수신이므로 방향 수준만),
  matched 2쌍 S1-S2 괴리.

## 5. Audit Record Additions

- `track: "P"`, `reference_year` (문항별), `system_prompt_id: "T-1P"`,
  Track M baseline record 경로 참조. 나머지 필드 파일럿 스키마 동일.
