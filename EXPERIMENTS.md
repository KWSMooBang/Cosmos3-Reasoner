# EXPERIMENTS

| run_id | task | 설정 | 결과 | 판단 |
|---|---|---|---|---|
| t003 smoke / cosmos3_nano_nothink / MindCubeBench_tiny_raw_qa | t003 | limit20_seed0, TP=1 GPU1, 16384, server 68fd24a | overall 40.00 (among 38.46 / around 50.0 / rotation 33.33) | 파이프라인 OK. 점수는 20샘플 smoke 라 의미 없음 |
| t003 smoke / cosmos3_nano_nothink / VSI-Bench | t003 | limit20_seed0, 32 frames, server a606c14 | overall 70.31 | 파이프라인 OK (opencv backend 수정 후 32프레임 확인) |
| t003 smoke / cosmos3_nano_think / MindCubeBench_tiny_raw_qa | t003 | limit20_seed0, CosmosReason3Think, T=0, max_tokens 16384 | overall 30.00 | 5/20 truncated (</think> 미종료) → think 는 TP=2/65536·공식 샘플링 필요 |
| t003 smoke / cosmos3_nano_think / VSI-Bench | t003 | limit20_seed0, 32 frames | overall 34.17 | 4/20 truncated, 38 s/sample. 위와 같음 |

### t003 smoke 메모 (2026-11-25)
- results.tsv 원문 아님: worktree-exp-smoke 의 results/summary.csv 인용. GPU ≈1.0 h (GPU 1).
- vLLM 0.32 버그 2개를 serving 에서 수정 (브랜치 worktree-exp-smoke, 미병합):
  68fd24a --trust-request-mm-kwargs (없으면 mm_processor_kwargs 요청이 400 → 비디오 전부 실패, 점수 0),
  a606c14 video_backend=opencv (없으면 qwen3_vl loader 가 32프레임을 4프레임으로 줄임; Qwen3-VL 베이스라인에도 해당).
- 클라이언트가 temperature=0 을 보내 서버 기본값 T=0.7 을 덮어씀 (vLLM 로그 82건 확인). nothink 출력에 <think> 0/20.
- KV 20,736 tokens @TP=1 → think 4개 동시 생성 시 preempt·timeout. think 는 TP=2 권장.
- 판단: keep (파이프라인). next: smoke 브랜치 병합, think 설정(예산·샘플링) 결정, 평가 .venv 를 main checkout 에 재구성.

### t006 think 설정 (2026-11-25, TP=2 GPU 0,1, 65536, limit20_seed0) — 출처: worktree-exp-think results/summary_t006_tp2.csv
| run | MindCube | VSI | SpaceCast_CoT | truncated (MC/VSI/SC) |
|---|---|---|---|---|
| cosmos3_nano_think T=0 | 20.00 | 35.00 | 26.94 | 4/20, 6/20, 7/20 |
| cosmos3_nano_think_official (T0.6/p.95/k20) | 15.00 | 31.25 | – | 3/20, 7/20 |
| cosmos3_nano_nothink | 40.00 | 69.69 | 56.39 | 0 |
- 잘린 샘플은 모두 반복 루프 (max_tokens 16384 까지). 닫힌 trace 는 ≤1.4k(T=0)/≤3.1k(official) tokens → 이중 분포. 잘린 것 제외해도 think < nothink.
- TP=1 vs TP=2 nothink: MindCube 동일, VSI 4/20 수치 답 미세 차이 (bf16 TP 수치), 1문항 MRA 변화로 70.31→69.69. TP 간 ±1점 예상.
- 판단: think = 보조(ablation) 지표. 설정 T=0, max_tokens 8192, truncation rate 함께 보고. 공식 샘플링 discard. headline = nothink.
- GPU ≈3.0 h (약 1 h 는 L2 감시 누락으로 유휴).

### t007 ReVSI_64frame full (2026-11-25) — 출처: worktree-exp-revsi-full results/summary.csv
| run_id | task | 설정 | 결과 | 판단 |
|---|---|---|---|---|
| cosmos3_nano_nothink / ReVSI_64frame / full | t007 | 6,616문항, T=0, TP=1 GPU1, 16384, video max_pixels 20,971,520 (native 64×640×480), rev 45666784 | Overall 51.42 (공식 파서) / lenient 51.42 | keep — 논문 Qwen3-VL-8B 49.1 보다 +2.3, 32B 53.3 보다 −1.9 |
- 문항별 결과 보관: /outputs/Cosmos3-Reasoner/results/t007_ReVSI_64frame_cosmos3_nano_nothink_full/ (*_revsi_score.xlsx: 문항·정답·예측·score)
- summary.csv 원문: `cosmos3_nano_nothink,spatial,spatial,ReVSI_64frame,full,,,ok,51.42,51.42,2026-11-25T20:44:05,...`
- 과제별: Obj.Cnt 35.80 / Abs.Dist 50.38 / Obj.Size 69.01 / Room Size 55.47 / Rel.Dist 59.05 / Rel.Dir 42.88 / Route Plan 47.31.
- 파싱 fallback 0, unparsed 0, 빈 예측 0, API 실패 0, finish_reason 전부 stop. 평균 prompt 9,755 tok. 1.45 s/sample, 2 h 40 min, ≈2.9 GPU-h.
- 서버 기본 TP=1 video cap(12.58M px)이면 프레임이 512×384 로 줄어듦 → 이 run 은 cap 상향. serving 기본값·PROTOCOL 반영 필요.
- next: 논문 per-task Qwen3-VL 값 확보 후 비교, Qwen3-VL-8B 같은 설정 재현(베이스라인 검증), collect.py 가 summary.csv 를 덮어쓰는 문제 수정.

### t009 SpaceCastBench_CoT full (2026-11-26) — 출처: worktree-exp-spacecast-full results/summary.csv
| run_id | task | 설정 | 결과 | 판단 |
|---|---|---|---|---|
| cosmos3_nano_nothink / SpaceCastBench_CoT / full | t009 | 3,862문항, cot, T=0, TP=1 GPU1, 16384, rev 45666784 | Overall 31.80 (macro 16 types), micro 30.86 | keep — 논문 Qwen3-VL-4B 34.0 보다 −2.2, Qwen3.5-27B 39.3 보다 −7.5 |
- summary.csv 원문: `cosmos3_nano_nothink,spatial,spatial,SpaceCastBench_CoT,full,,,ok,31.80,,2026-11-26T01:17:09`
- L1 macro 38.34 / L2 30.31 / L3 22.18. 최저: Rotate Object Dir(L3) 12.11, Object Direction 15.94. 최고: Occlusion 66.06.
- n_invalid 6 (= finish_reason length 6, T=0 반복 루프), n_error 0, lenient 0. prompt mean 3.7k / max 9.9k tok. 1.02 s/sample, 65.6 min, ≈1.2 GPU-h.
- 스모크 56.39(limit20)·pilot 47.66/36.07(limit50) 은 소표본 노이즈였음 → 소표본 점수로 판단하지 말 것.
- 문항별 결과: /outputs/Cosmos3-Reasoner/results/t009_SpaceCastBench_CoT_cosmos3_nano_nothink_full/

### t013 순차 평가 (GPU 1, TP=1, cosmos3_nano_nothink T=0) — 출처: L2 [NOTE] (result 파일 인용은 [DONE] 때 갱신)
| run_id | 설정 | 결과 | 판단 |
|---|---|---|---|
| cosmos3_nano_nothink / PhysBench_val / full | 200문항, 이미지 cap 6.9MP | Overall 66.50 (property 54.05 / relationships 74.47 / scene 56.76 / dynamics 72.15), format_gap 0.0 | keep — 형식 패널티 없음 확인 |
| cosmos3_nano_nothink / VSTIBench / full | 5,736문항, 32 frames native | Overall 53.90 (paper Avg of 5; code9 63.93): CamObjAbs 50.43 / CamDisp 24.83 / CamMovDir 41.29 / ObjObjRelPos 74.65 / CamObjRelDist 78.29 | keep — 정오표 후 라벨이라 논문 수치(VLM-3R 58.8, GPT-4o 38.2)와 비교 불가. 0.70 GPU-h |
| cosmos3_nano_nothink / 3DSRBench_official / full | 5,157문항 (11,686행), CircularEval | Overall 53.42 (height 48.26 / location 68.23 / orientation 39.00 / multi_object 49.48; vanilla 55.24, flip_circ 47.43) | keep — Qwen2.5-VL-7B 48.4, Gemini-2.0-Flash 49.8, GPT-4o 44.2 보다 높음 (논문은 GPT 추출 fallback, 우리는 exact matching). 0.51 GPU-h |
