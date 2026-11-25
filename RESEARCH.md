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
- LMUData: /data0/Cosmos3-Reasoner/LMUData

## 가설·백로그
- [x] t001 vlmevalkit-cosmos subtree + 평가 클라이언트 venv + configs/scripts/docs 뼈대 (main cf328c1)
- [ ] t002 serving/ native vLLM 스크립트 — 완료 보고 받음, codex 리뷰 수정 중
- [ ] t003 스모크 테스트 (Nano, TP=2) — docker 가 없어 보류. native vLLM 경로 승인 시 진행
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
- 2026-11-25: 스모크 테스트는 GPU 0,1 만 사용 (사람 지시). harness launch.sh 는 GPU 를 스스로 고르므로
  스모크는 CUDA_VISIBLE_DEVICES=0,1 로 직접 실행한다.

## GPU 예산
- 미정 (스모크 ≈ 2 GPU × 1h). 전체 벤치 실행 전 사람과 정한다.
