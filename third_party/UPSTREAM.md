# Vendored third-party code

## third_party/vlmevalkit-cosmos

| field | value |
|---|---|
| upstream repo | https://github.com/NVIDIA/cosmos-framework |
| upstream path | `.agents/skills/evaluation/cosmos3/reasoner/vlmevalkit/` |
| upstream commit | `a3fb34690e157972a9d04244c11b16022c9d6479` ("Fix wsm ftheta derivative (#426)") |
| commit date | 2026-11-20 21:13:45 -0800 (2026-11-21 05:13 UTC) |
| split commit | `2a776869b204650c07588f8263c29cf2d1e9dd6f` (`git subtree split`; 8 upstream commits touch the path) |
| method | `git subtree split -P <path> -b vlmevalkit-cosmos` in a scratch clone (took ~1 s), then `git subtree add --prefix third_party/vlmevalkit-cosmos <scratch> vlmevalkit-cosmos --squash` (squash commit `28adaf7`, merge `718d5f1`) |
| vendored on | 2026-11-25 |
| license | Apache-2.0 (see `vlmevalkit-cosmos/LICENSE`) |

### Updating

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git /tmp/cf && git -C /tmp/cf checkout <new-sha>
git -C /tmp/cf subtree split -P .agents/skills/evaluation/cosmos3/reasoner/vlmevalkit -b vlmevalkit-cosmos
git subtree pull --prefix third_party/vlmevalkit-cosmos /tmp/cf vlmevalkit-cosmos --squash
```
Then update this table.

### Local modifications (each in its own commit; see `git log -- third_party/vlmevalkit-cosmos`)

1. `vlmeval/api/cosmos_reason.py`, `vlmeval/api/__init__.py`: add `CosmosReason3Think`, a thinking
   wrapper that follows the official Cosmos 3 prompt guide (system prompt, `<think>` instruction,
   answer = text after the last `</think>`). Upstream classes are untouched. See PROTOCOL.md §4.
2. `vlmeval/api/cosmos_reason.py`: `CosmosReason3Think` also parses length-truncated replies (text after `</think>`,
   else empty prediction + `TRUNCATED` warning) instead of passing the raw reasoning to the scorer.
   See PROTOCOL.md §3.
3. `vlmeval/dataset/revsi.py` (new), `vlmeval/dataset/__init__.py` (one import + `VIDEO_DATASET += [ReVSI]`):
   ReVSI dataset (keys `ReVSI_{16,32,64}frame`, `ReVSI_tiny`), ported from the official evaluation code:
   open-compass/VLMEvalKit `vlmeval/dataset/revsi.py` @ `f5296f04f3d6b5d5664f55aa64014c9dc942ed15`,
   EvolvingLMMs-Lab/lmms-eval `lmms_eval/tasks/revsi/` @ `9134731f4f617e62ee80bcae9d0d5d9695ea55ef`,
   eamonn-zh/torchmetrics_ext `metrics/vqa/revsi.py` @ `3c18d782f51cbe31558ea7b53cbece5561b1721e`
   (referenced by 3dlg-hcvc/revsi @ `b26df855`). Data: HF `3dlg-hcvc/ReVSI` @ `345a811`. See PROTOCOL.md §6.1.
4. `vlmeval/dataset/physbench.py` (new), `vlmeval/dataset/utils/physbench_official.py` (new, verbatim copy of
   USC-GVL/PhysBench `eval/print_test_score.py` lines 1-78 @ `a7eadc4da554e20163303ec0c1e076a35a9f52e2`, Apache-2.0),
   `vlmeval/dataset/__init__.py` (one import + `VIDEO_DATASET += [PhysBench]`): PhysBench dataset (keys `PhysBench`,
   `PhysBench_val`). Prompt / 8-frame budget / interleaving from the official `eval/eval_utils/task_evaluator.py`
   and `eval/models/qa_model/imageqa_model.py` (GPT-4o client) at the same commit; diagnostic scorer from
   EvolvingLMMs-Lab/lmms-eval `lmms_eval/tasks/physbench/` @ `9134731f4f617e62ee80bcae9d0d5d9695ea55ef`.
   Data: HF `USC-PSI-Lab/PhysBench` @ `478fd93d` (apache-2.0, not gated); labels: official repo
   `eval/physbench/{test,val}_answer.json`. See PROTOCOL.md §6, PhysBench.
