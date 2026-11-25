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
