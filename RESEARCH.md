# Cosmos3-Reasoner 평가 연구

## 목표
NVIDIA Cosmos 3 Reasoner(nvidia/Cosmos3-Nano, nvidia/Cosmos3-Super)를 3D spatial/perception·physics understanding
벤치마크로 평가하는 레포를 만들고, Qwen3-VL 베이스라인과 비교한다.
1단계(현재): 환경 세팅 + 레포 뼈대 + 스모크 테스트. 전체 벤치 실행은 하지 않는다.

## 환경 (2026-11-25 점검)
- GPU: RTX 4090 24GB × 8 (공유, 다른 연구가 일부 사용 중), driver 550.120 (CUDA ≤12.4), nvcc 12.4
- docker 없음 (nvidia-container-toolkit 도 없음) → NVIDIA 고정 이미지로 서빙 불가
- 디스크: /workspace 232G, /outputs 157G(98%), /data0 980G(쓰기 가능), HF_HOME=/cache/huggingface (347G)
- python3 3.10 (uv 0.12.23 으로 3.11 venv 생성), HF 토큰 파일 있음
- 모델 크기(HF API): Cosmos3-Nano 35.0GB (sha 45666784803b3a7ad577e0def405f9afcc3bddb2) → 4090 1장 불가, TP≥2
  Cosmos3-Super 132.7GB (sha 73d5746645961a0ec9607123a003bd9c1f8a0282) → 4090 8장 전부 필요. 받지 않음(사람 확인 전)
- LMUData: /data0/woosung/datasets/Cosmos3-Reasoner/LMUData

## 가설·백로그
- [x] t001 vlmevalkit-cosmos subtree + 평가 클라이언트 venv + configs/scripts/docs 뼈대 (main cf328c1)
- [x] t002 serving/ native vLLM 스크립트 (main 03cdb9c)
- [x] t003 스모크 테스트 (Nano, TP=1, GPU 1) — 파이프라인 OK, vLLM 0.32 서빙 버그 2개 수정 (main 10c75f5)
- [x] t006 think 설정: think 는 보조 지표 (T=0, max_tokens 8192, truncation rate 보고). headline = nothink. main .venv 재구성 완료
- [x] t004 ReVSI 구현 (main e5d00aa) — HF 3dlg-hcvc/ReVSI 4.9GB, 64 frames, MRA+acc. 논문 Qwen3-VL-8B 49.1
- [x] t005 SpaceCast-Bench 구현 (main 67817d2, 기본 모드 = CoT: 사람 결정, 공식 메인 실험과 동일) — 3,862 MCQ, 다중 시점 이미지 ≤8장, macro acc. 공식 평가기가 OpenAI 호환 API 지원
- [ ] t008 Beyond3D 세팅 (진행 중, 사람 지시 2026-11-25) — 데이터 준비 + VLMEvalKit 포팅: HD-EPIC 원본 영상 ~115GB 필요(공유 폴더 로그인 가능성), CC-BY-NC, 9,000 MCQ(8 유형, 148 영상),
      비디오를 query 시점까지 잘라 448x448 1fps + 타임스탬프 오버레이로 전처리. 공식 코드는 lmms-eval 포크
- [x] t007 ReVSI_64frame 전체 평가 (Nano nothink): Overall 51.42 (Qwen3-VL-8B 49.1 / 32B 53.3)
- [x] t009 SpaceCastBench_CoT 전체 평가 (Nano nothink): Overall 31.80 (Qwen3-VL-4B 34.0)
- [ ] t008 Beyond3D: 포팅 완료(exp/beyond3d), codex High 3건 수정 대기·병합 보류. → t014 (수정 + 보정 데이터 연결) 진행 중. 보정 데이터는 사람이 hd-epic/HD-EPIC_Intermediate_Data 로 업로드 중 (2026-11-26). TP=2/65536 필요 (prompt p50 22.6k). 영상 출처·보정 결정 대기
- [x] t010 QuantiPhy 구현 (main c958dda): val 158문항만 채점 가능 (test GT 비공개), 공식 parser ^2 버그 유지 + lenient
- [x] t011 PhysBench 구현 (main bc8c0b7): 10,002문항, TP=1 이미지 cap 6.9MP (359문항 영향), native TP=2 opt-in
- [x] t012 VSTI-Bench 구현 + 3DSRBench_official (upstream 키는 헤드라인 0.00 버그) — main 병합
- [x] t013 GPU 1 순차 전체 평가 (Nano nothink): PhysBench_val 66.50, VSTIBench 53.90, 3DSRBench_official 53.42, PhysBench 53.27, QuantiPhy(val) 41.15
- [ ] Qwen3-VL-8B/32B 베이스라인, 기술 보고서 Table 10 재현

## 결정
- 2026-11-25: 코드 작업은 L2 로. push·원격 생성은 사람 승인 후.
- 2026-11-25: 서빙은 docker 가 아니라 native vLLM(.venv-vllm)으로 한다 (이 세션이 이미 docker 안에서 돌기 때문). docker 스크립트는 참고용으로만 둔다.
  NVIDIA 고정 이미지와 vLLM 버전이 다를 수 있으니 그 차이를 PROTOCOL.md 에 기록한다. 1차 평가 대상은 Cosmos3-Nano.
- 2026-11-25: Nano 는 MoT. understanding 경로 ≈8.8B(≈17.5GB bf16)만 있으면 reasoner 로 쓸 수 있음 → 4090 1장(TP=1) 시도.
  나머지 ≈15GB 는 생성 전용(mlp_moe_gen, 다른 attention 세트, vae, sound_tokenizer). vLLM 이 und 만 로드하는지는 t002 가 확인.
  KV 여유 ≈3–4GB(≈20k tokens) → max-model-len 과 프레임/해상도 상한을 낮춰야 함. 안 되면 TP=2.
- 2026-11-25: native vLLM 0.32.0+cu129 사용 (Cosmos3 지원 v0.23.0+, cu12 wheel 최신). NVIDIA 이미지(vLLM bfee6a80, CUDA 13)는 driver 550 에서 불가.
  vLLM cosmos3.py 는 understanding 가중치만 로드 → Nano TP=1(≈17.6GB), 스모크는 max-model-len 16384. 보고서 대비 비교는 TP=2/65536.
- 2026-11-25: 스모크 테스트는 GPU 1 만 사용, Qwen 베이스라인 없이 Nano 만 (사람 지시). harness launch.sh 는 GPU 를 스스로 고르므로
  스모크는 CUDA_VISIBLE_DEVICES=1 로 직접 실행한다. t002 main 병합·push 03cdb9c.

- 2026-11-25: think 평가는 GPU 0,1 (TP=2, 65536) 사용 (사람 지시). t003·t004·t005 main 병합·push 67817d2.

- 2026-11-26: 데이터는 /data0 바로 아래가 아니라 /data0/woosung/datasets/Cosmos3-Reasoner/ 아래에 둔다 (사람 지시). LMUData·hd-epic 이동 완료.

- 2026-11-26: t010–t012 세팅이 끝나면, GPU 1장(TP=1, 16384)에 들어가는 새 벤치들을 GPU 1 에서 순차 전체 평가 (사람 지시).
  대상 후보: QuantiPhy, PhysBench, VSTIBench, 3DSRBench(검증 결과 반영). ReVSI·SpaceCast 는 완료, Beyond3D 는 TP=2 필요 + 데이터 대기라 제외.

- 2026-11-26: t006·t007·t009 main 병합 (사람 승인, 로컬). GPU 1 순차 평가 예산 4–10 GPU-h 승인.

## GPU 예산
- 미정. 사용: t003 ≈1.0 GPU-h. t006 ≈3.0, t007 ≈2.9, t009 ≈1.2, t013 ≈2.8 GPU-h (누적 ≈10.9). 전체 벤치 실행 전 사람과 정한다.
