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
- summary.csv 원문: `cosmos3_nano_nothink,spatial,spatial,ReVSI_64frame,full,,,ok,51.42,51.42,2026-11-25T20:44:05,...`
- 과제별: Obj.Cnt 35.80 / Abs.Dist 50.38 / Obj.Size 69.01 / Room Size 55.47 / Rel.Dist 59.05 / Rel.Dir 42.88 / Route Plan 47.31.
- 파싱 fallback 0, unparsed 0, 빈 예측 0, API 실패 0, finish_reason 전부 stop. 평균 prompt 9,755 tok. 1.45 s/sample, 2 h 40 min, ≈2.9 GPU-h.
- 서버 기본 TP=1 video cap(12.58M px)이면 프레임이 512×384 로 줄어듦 → 이 run 은 cap 상향. serving 기본값·PROTOCOL 반영 필요.
- next: 논문 per-task Qwen3-VL 값 확보 후 비교, Qwen3-VL-8B 같은 설정 재현(베이스라인 검증), collect.py 가 summary.csv 를 덮어쓰는 문제 수정.
