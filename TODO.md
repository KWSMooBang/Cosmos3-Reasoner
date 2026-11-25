# TODO

## Benchmarks that need implementation (stubs in the NVIDIA kit) or are missing

Stub modules ("dataset implementation not included in this public release"; list from
`python scripts/check_suites.py --list-stubs`). Relevant to this study:

| benchmark | stub class (module) | note |
|---|---|---|
| CV-Bench | `CVBench` (`cv_bench`) | needs implementation |
| BLINK depth / spatial only | `BlinkDepth` (`blink_depth`), `BlinkSpatial` (`blink_spatial`) | needs implementation; full `BLINK` (generic ImageMCQDataset, 14 subtasks) is in the spatial suite |
| SAT | `SATBench` (`sat_bench`) | needs implementation |
| Where2Place | `Where2Place` (`where2place`) | needs implementation |
| IntPhys2 | `IntPhys2` (`intphys2`) | needs implementation |
| VideoPhy2 | `VideoPhy2` (`videophy2`) | needs implementation |
| CausalVQA | `CausalVQA` (`causalvqa`) | needs implementation |
| MVPBench | `MVPBench` (`mvpbench`) | needs implementation |
| CosmosReason (Reason1 bench) | `CosmosReason` (`cosmos_reason` dataset) | needs implementation |
| WorldBench | `WorldBench` (`worldbench`) | needs implementation |
| MotionBench | `MotionBench` (`motionbench`) | needs implementation |
| NVIDIA's SPAR-Bench | `SPARBench`, `SPARBenchTiny` (`spar_bench`) | stub, but keys `SparBench`/`SparBench_tiny` are served by upstream `sparbench.SparBench` (used) |
| NVIDIA's RoboSpatialHome | `RoboSpatialHome` (`robospatial_home`) | stub, but key `RoboSpatialHome` is served by upstream `robospatialbench.RoboSpatialBench` (used) |
| NVIDIA's ERQA | `CosmosERQA` (`cosmos_erqa`) | stub; key `ERQA` served by `ERQABench` / `ERQADataset` (ERQABench used) |
| Omni3D detection | `Omni3DDetectionDataset` (`Omni3D/omni3d_dataset`) | stub; `Omni3DBench` (VQA) is implemented and used |
| CameraBench, CameraIntrinsic, AV 3D grounding, LocateAnything, ODinW13, … | see `--list-stubs` | not in scope; needs implementation if wanted |
| **PhysBench** | — | **missing** entirely (no class, no key) |

Implemented but not in a suite:
- `Physics` (`Physics_yale`): evaluate() calls `os.path.basename(os.environ.get('LOCAL_LLM'))`
  and crashes unless `LOCAL_LLM` is set, then requires an LLM judge. Add once a judge setup exists.
- `RefSpatial-Bench*` (`RefSpatialDataset`, no download URL) — `RefSpatial` (RefSpatialBench) used instead.
- `MindCubeBench_tiny_raw_qa`, `SparBench_tiny`: smoke only.

## Spec-vs-code differences (t001)

1. **SparBench / SparBench_tiny**: implemented (upstream `sparbench.SparBench`), although NVIDIA's own
   `spar_bench.py` is a stub. Smoke uses MindCubeBench_tiny_raw_qa (spec allowed either).
2. **RoboSpatialHome**: spec expected a stub; key is served by upstream `RoboSpatialBench` → added to spatial suite.
3. **Video frame settings live in the model config, not only the dataset config.** CosmosReason*
   is `VIDEO_LLM=True`; video datasets pass the raw video and the client samples frames from its own
   `nframes/fps/max_pixels/total_pixels`. Dataset-side `nframe` is ignored on that path. Data configs
   therefore carry a `video_sampling` block that `run_suite.py` merges into `model_conf`
   (`{class} | defaults | data.video_sampling | model.benchmarks[bench]`), an extension of run_all's compose.
4. **Video variant keys** (`VSI-Bench_32frame`, …) are in `supported_video_datasets`, not in
   `SUPPORTED_DATASETS`; suites use base keys (`VSI-Bench`) with explicit frame settings instead.
5. **No sample-limit option** in VLMEvalKit `run.py` → `scripts/vlmeval_run.py` wrapper (README "Sample limit").
6. **Thinking format**: `CosmosReason2Think` does not follow the official guide (different instruction,
   no system prompt, requires a leading `<think>`); added `CosmosReason3Think` (PROTOCOL.md §4).
   The Cosmos3 chat template has no `enable_thinking` switch, so `chat_template_kwargs.enable_thinking`
   (NVIDIA default) is a no-op.
7. **Official thinking sampling** also sets `presence_penalty 0.0, repetition_penalty 1.0` and the
   guide prescribes system prompt "You are a helpful assistant." (spec listed only T/top_p/top_k).
8. **Expected stubs list**: in addition to the spec list, NVIDIA's `CosmosERQA`, `Omni3DDetectionDataset`,
   `SPARBench(Tiny)` and `RoboSpatialHome` are stubs, but each key has a non-stub upstream implementation (see table).
9. **Official prompt guide location**: `cookbooks/cosmos3/reasoner/` no longer exists on NVIDIA/cosmos `main`;
   read at commit `1fe7e3be` (and its cosmos-framework mirror, identical content).
10. **Kit README pins Nano processor config at `e59a53c2`**, not the `45666784` revision we evaluate.
11. **`requirements.txt` quirk**: line `# acceleratepython-box` (missing newline) comments out `python-box`,
    but `python-box` is listed again on the next line, so the install is unaffected.
12. **Judge defaults**: VLMEvalKit's run.py would use `chatgpt-0125` for MCQ answer extraction, `gpt-4.1` for
    CoreCognition, `deepseek` (SiliconFlow) for SeePhys; PhyX requires `--judge-args '{"valid_type": ...}'`;
    QSpatial switches to an LLM path whenever any judge is passed. Choices in PROTOCOL.md §6.
13. Serving: native vLLM (no docker on host); NVIDIA image digest kept as reference only (human decision).

## Open items

- [ ] Fill PROTOCOL.md §1 with the native vLLM version/flags from t002 (and any 24 GB-GPU deviations: TP, max-model-len).
- [ ] Find Cosmos 3 technical report per-bench settings (frames, prompts, thinking on/off per bench) for Table 10
      reproduction; reconcile with §6 (esp. MMSIVideoBench 64 vs 300 frames, SiteBenchVideo 32).
- [ ] Pin Qwen3-VL-8B/32B-Instruct HF revisions when served.
- [ ] First smoke run (t003): confirm the `video_sampling` path yields the expected visual token counts
      (vLLM log / `execution_evidence_dir`), and that subset evaluation works for MindCube and VSI-Bench.
- [ ] Decide on judge for SeePhys (OpenAI key availability) and whether to add Physics_yale.
- [ ] Datasets will download to `$LMUData` / `$HF_HOME` on first run: check sizes before the full suites
      (VSI-Bench raw videos, MMSIVideoBench, STI-Bench, SiteBenchVideo, DSRBench).
