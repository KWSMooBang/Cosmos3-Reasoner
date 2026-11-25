# EXPERIMENTS

| run_id | task | 설정 | 결과 | 판단 |
|---|---|---|---|---|
| t003 smoke / cosmos3_nano_nothink / MindCubeBench_tiny_raw_qa | t003 | limit20_seed0, TP=1 GPU1, 16384, server 68fd24a | overall 40.00 (among 38.46 / around 50.0 / rotation 33.33) | 파이프라인 OK. 점수는 20샘플 smoke 라 의미 없음 |
| t003 smoke / cosmos3_nano_nothink / VSI-Bench | t003 | limit20_seed0, 32 frames, server a606c14 | overall 70.31 | 파이프라인 OK (opencv backend 수정 후 32프레임 확인) |
| t003 smoke / cosmos3_nano_think / MindCubeBench_tiny_raw_qa | t003 | limit20_seed0, CosmosReason3Think, T=0, max_tokens 16384 | overall 30.00 | 5/20 truncated (</think> 미종료) → think 는 TP=2/65536·공식 샘플링 필요 |
| t003 smoke / cosmos3_nano_think / VSI-Bench | t003 | limit20_seed0, 32 frames | overall 34.17 | 4/20 truncated, 38 s/sample. 위와 같음 |

| t006 smoke / cosmos3_nano_think / MindCubeBench_tiny_raw_qa | t006 | TP=2 GPU0,1, 65536, T=0, max_tokens 16384, api_nproc 4 | overall 20.00 | 4/20 TRUNCATED (all loops), closed traces ≤1.4k tok |
| t006 smoke / cosmos3_nano_think / VSI-Bench | t006 | 위와 같음, 32 frames | overall 35.00 | 6/20 TRUNCATED (all loops) |
| t006 smoke / cosmos3_nano_think_official / MindCubeBench_tiny_raw_qa | t006 | TP=2, T=0.6 top_p 0.95 top_k 20 seed 1 (payload 확인) | overall 15.00 | 3/20 TRUNCATED (loops) |
| t006 smoke / cosmos3_nano_think_official / VSI-Bench | t006 | 위와 같음 | overall 31.25 | 7/20 TRUNCATED (loops), 1275 s |
| t006 smoke / cosmos3_nano_nothink / MindCubeBench_tiny_raw_qa | t006 | TP=2 control | overall 40.00 | t003 TP=1 과 예측 20/20 동일 |
| t006 smoke / cosmos3_nano_nothink / VSI-Bench | t006 | TP=2 control | overall 69.69 | t003 70.31 대비: 4/20 수치 답 미세 차이 (TP 수치), 4690 abs_distance MRA .5→.3 |
| t006 spatial / cosmos3_nano_think / SpaceCastBench_CoT | t006 | limit20_seed0, TP=2, T=0 | overall 26.94 | 7/20 TRUNCATED (loops) |
| t006 spatial / cosmos3_nano_nothink / SpaceCastBench_CoT | t006 | limit20_seed0, TP=2 | overall 56.39 | 파이프라인 OK |
### t003 smoke 메모 (2026-11-25)
- results.tsv 원문 아님: worktree-exp-smoke 의 results/summary.csv 인용. GPU ≈1.0 h (GPU 1).
- vLLM 0.32 버그 2개를 serving 에서 수정 (브랜치 worktree-exp-smoke, 미병합):
  68fd24a --trust-request-mm-kwargs (없으면 mm_processor_kwargs 요청이 400 → 비디오 전부 실패, 점수 0),
  a606c14 video_backend=opencv (없으면 qwen3_vl loader 가 32프레임을 4프레임으로 줄임; Qwen3-VL 베이스라인에도 해당).
- 클라이언트가 temperature=0 을 보내 서버 기본값 T=0.7 을 덮어씀 (vLLM 로그 82건 확인). nothink 출력에 <think> 0/20.
- KV 20,736 tokens @TP=1 → think 4개 동시 생성 시 preempt·timeout. think 는 TP=2 권장.
- 판단: keep (파이프라인). next: smoke 브랜치 병합, think 설정(예산·샘플링) 결정, 평가 .venv 를 main checkout 에 재구성.

### t006 think 설정 메모 (2026-11-25)
- 결과: results/summary_t006_tp2.csv (TP=2 행; summary.csv 의 t003 TP=1 행과 키가 겹쳐 별도 파일). raw: worktree exp-think results/raw/ (+ _t006_dumps/ 요청별 원문).
- truncation 은 컨텍스트 artifact 가 아님: 65536 컨텍스트·완전한 16384 예산에서도 think T=0 10/40, official 10/40, SpaceCast 7/20 이 16384 토큰까지 반복 루프 (tail 의 고유 문장 비율 0.05–0.29, 같은 문장 9–34회). preemption 0.
- 길이 분포 이봉형: </think> 를 닫은 trace 는 모두 ≤3.1k 토큰 (T=0 ≤1.4k), 나머지는 전부 16384 루프.
- truncation 을 빼도 think < nothink (MindCube 비공백 정답률 25% vs 40%). official 샘플링은 루프를 줄이지 못하고 점수도 더 낮음.
- 클라이언트 추가: COSMOS_RAW_DUMP_DIR (기본 off) 로 요청별 latency/usage/finish_reason/보낸 샘플링 값/원문 저장.
- 판단: think 는 T=0, max_tokens 8192 권장 (smoke 결과 동일, 루프 비용 절반). nothink 가 기본 보고값. GPU ≈3.0 h (GPU 0,1 × 1.5 h, 그중 ~0.5 h 는 run3 종료 감지 누락으로 idle).
