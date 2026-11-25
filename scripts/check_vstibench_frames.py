#!/usr/bin/env python
"""Prove on the real VSTI-Bench files that the official 32-frame budget reaches the request (no GPU / server).

  1. Builds VSTIBench exactly as run.py does (data config -> vlmeval VSTIBench) from $LMUData.
  2. For --scenes sampled videos (seed 0) takes the message from build_prompt(video_llm=True), checks it is a
     video message whose only sampling key is nframes=32, and runs the client preprocessing
     (prepare_inputs -> qwen-vl-utils -> base64 mp4; endpoint stubbed) with the merged run_suite config,
     the bare model layer and conflicting model-level settings (nframes=8 / fps=2.0 / max_frames=4).
     The frame tensor that would be sent must have T == 32 and native 640x480. Each sent frame is matched
     (nearest neighbour on 32x24 thumbnails, window +-3 source frames) against the source video, and the
     offset to the official index np.linspace(0, n-1, 32, dtype=int) (VLM-3R vlm_3r.load_video) is reported:
     qwen-vl-utils uses torch.linspace(...).round(), so offsets of 0 or +1 source frame (1/24 s) are expected;
     anything else fails.
  3. Non-VIDEO_LLM path: save_video_frames returns 32 images = exactly the official indices.
  4. --dump-dir: writes one sent frame tensor (T,C,H,W uint8 .npy) + the longest prompt of every question
     type, for the token count with the Cosmos3-Nano HF processor (--tokens, run with .venv-vllm).
  5. Prints one numeric and one multiple-choice prompt.

    python scripts/check_vstibench_frames.py [--model cosmos3_nano_nothink] [--scenes 6] [--dump-dir DIR]
    .venv-vllm/bin/python scripts/check_vstibench_frames.py --tokens DIR      # processor token counts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SAMPLING_KEYS = {"nframes", "fps", "total_pixels", "max_pixels", "min_pixels", "max_frames"}
KEY = "VSTIBench"


def thumbs(frames, size=(32, 24)):
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(f).resize(size, Image.BILINEAR), dtype=np.float32).ravel()
                     for f in frames])


def tokens(dump_dir: Path, model_dir: str) -> None:
    """Count prompt tokens with the Cosmos3-Nano processor for the dumped request video + prompts."""
    import torch
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_dir)
    video = torch.from_numpy(np.load(dump_dir / "video.npy"))
    meta = json.loads((dump_dir / "meta.json").read_text())
    rows = []
    for qt, prompt in meta["prompts"].items():
        msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vmeta = dict(fps=meta["fps"], total_num_frames=int(video.shape[0]), frames_indices=list(range(video.shape[0])))
        enc = proc(text=[text], videos=[video], videos_kwargs=dict(do_sample_frames=False, video_metadata=[vmeta],
                                                                   do_resize=False), return_tensors="pt")
        n_vid = int((enc["input_ids"][0] == proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")).sum())
        rows.append((qt, int(enc["input_ids"].shape[1]), n_vid))
    print(f"video tensor {tuple(video.shape)} fps {meta['fps']:.4f}; Cosmos3-Nano processor ({model_dir}):")
    for qt, n, nv in rows:
        print(f"  {qt:28s} prompt tokens {n:5d} (video pad tokens {nv})")
    mx = max(r[1] for r in rows)
    print(f"max prompt {mx} tokens -> leaves {16384 - mx} of max-model-len 16384 (TP=1) for the reply")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="cosmos3_nano_nothink")
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--dump-dir")
    ap.add_argument("--tokens", help="dump dir to count tokens for (run with the vLLM venv)")
    ap.add_argument("--hf-model", default="nvidia/Cosmos3-Nano", help="HF id or local snapshot dir")
    ap.add_argument("--revision", default="45666784803b3a7ad577e0def405f9afcc3bddb2")
    args = ap.parse_args(argv)
    if args.tokens:
        from huggingface_hub import snapshot_download
        mdir = args.hf_model if os.path.isdir(args.hf_model) else snapshot_download(
            args.hf_model, revision=args.revision, allow_patterns=["*.json", "*.txt"])
        tokens(Path(args.tokens), mdir)
        return

    sys.path.insert(0, str(REPO / "scripts"))
    import run_suite as rs
    warnings.filterwarnings("ignore")
    import decord
    import vlmeval.api as A
    import vlmeval.api.cosmos_reason as cr
    import vlmeval.dataset as D
    from vlmeval.dataset.vstibench import official_frame_indices

    cr.CosmosReason._check_endpoint_health = lambda self: None
    cr.CosmosReason._resolve_model_name = lambda self, *a, **k: None
    captured = []
    orig_b64 = cr.tensor_video_to_base64

    def spy(video_tensor, fps, *a, **k):
        captured.append((video_tensor.detach().clone(), fps))
        return orig_b64(video_tensor, fps, *a, **k)
    cr.tensor_video_to_base64 = spy

    item = {"key": f"spatial/{KEY}", "domain": "spatial", "bench": KEY, "run": {}}
    cfg = rs.build_config(rs.load_model_layer(args.model), item, render=True)
    mconf = dict(next(iter(cfg["model"].values())))
    dconf = dict(cfg["data"][KEY])
    cls = getattr(D, dconf.pop("class"))
    dconf.pop("model_family", None)
    ds = cls(dataset=KEY, **dconf)
    n = ds.num_frames
    print(f"{KEY}: {len(ds)} questions, {ds.data['video'].nunique()} videos, frame budget {n}; model layer "
          f"{args.model}: client sampling { {k: mconf[k] for k in rs.VIDEO_SAMPLING_KEYS if mconf.get(k) is not None} }")
    print("question types:", ds.data["question_type"].value_counts().to_dict())

    klass = getattr(A, mconf.pop("class"))
    base = dict(mconf, use_video_cache=False, api_base="http://offline.invalid/v1/chat/completions")
    bare = {k: v for k, v in base.items() if k not in rs.VIDEO_SAMPLING_KEYS}
    models = {
        "run_suite config": klass(**base),
        "bare model layer": klass(**bare),
        "model-level nframes=8": klass(**dict(bare, nframes=8)),
        "model-level fps=2.0": klass(**dict(bare, fps=2.0)),
        "model-level max_frames=4": klass(**dict(bare, max_frames=4)),
    }

    fails, all_offsets = [], []
    rng = np.random.default_rng(0)
    first = ds.data.drop_duplicates("video")
    picks = first.iloc[rng.choice(len(first), size=min(args.scenes, len(first)), replace=False)]
    dumped = None
    for _, line in picks.iterrows():
        msg = ds.build_prompt(line, video_llm=True)
        vids = [m for m in msg if m["type"] == "video"]
        assert len(vids) == 1 and {k for k in vids[0] if k in SAMPLING_KEYS} == {"nframes"} \
            and vids[0]["nframes"] == n, vids
        path = vids[0]["value"]
        vr = decord.VideoReader(path)
        total = len(vr)
        off_idx = official_frame_indices(total)
        win = sorted({j for i in off_idx for j in range(max(0, i - 3), min(total, i + 4))})
        src = dict(zip(win, thumbs(vr.get_batch(win).asnumpy())))
        for mname, model in models.items():
            captured.clear()
            _, vkw = model.prepare_inputs(msg)
            (tensor, sent_fps), = captured
            sent = tensor.float().numpy()
            if sent.max() <= 1.0 + 1e-6:
                sent = sent * 255.0
            sent = np.clip(sent, 0, 255).astype(np.uint8)
            if dumped is None and args.dump_dir:
                dumped = (sent, sent_fps)
            a = thumbs(sent.transpose(0, 2, 3, 1))
            offs = []
            for t, i in enumerate(off_idx[:len(a)]):
                cands = [j for j in range(i - 3, i + 4) if j in src]
                offs.append(cands[int(np.argmin([((a[t] - src[j]) ** 2).mean() for j in cands]))] - i)
            ok = sent.shape[0] == n and sent.shape[2:] == (480, 640) and set(offs) <= {0, 1}
            all_offsets += offs
            print(f"[{'OK' if ok else 'FAIL'}] {os.path.relpath(path, ds.data_root)} ({total} frames) [{mname}]: "
                  f"request {sent.shape[0]} frames {sent.shape[3]}x{sent.shape[2]}, offset to official index "
                  f"{ {o: offs.count(o) for o in sorted(set(offs))} }; mm_processor_kwargs={vkw}")
            if not ok:
                fails.append(f"{path} [{mname}] offsets {offs}")

    line = picks.iloc[0]
    msg = ds.build_prompt(line, video_llm=False)
    imgs = [m["value"] for m in msg if m["type"] == "image"]
    vr = decord.VideoReader(ds.video_path(line["video"]))
    from PIL import Image
    exp = vr.get_batch(official_frame_indices(len(vr))).asnumpy()
    got = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in imgs])
    d = np.abs(got.astype(int) - exp.astype(int)).mean()
    ok = len(imgs) == n and d < 3.0
    print(f"[{'OK' if ok else 'FAIL'}] image-frame path: {len(imgs)} images for {line['video']}, mean |diff| to "
          f"official indices {d:.2f} (JPEG)")
    if not ok:
        fails.append("image path")

    prompts = {}
    for qt, sub in ds.data.groupby("question_type"):
        r = sub.loc[sub["question"].str.len().idxmax()]
        prompts[qt] = ds.build_prompt(r, video_llm=True)[-1]["value"]
    for qt in ("camera_displacement", "camera_obj_rel_dist_v1"):
        print(f"\n--- prompt ({qt}) ---\n{prompts[qt]}")
    if args.dump_dir:
        dd = Path(args.dump_dir)
        dd.mkdir(parents=True, exist_ok=True)
        np.save(dd / "video.npy", dumped[0])
        (dd / "meta.json").write_text(json.dumps({"fps": float(dumped[1]), "prompts": prompts}, indent=1))
        print(f"\ndumped request video {dumped[0].shape} + {len(prompts)} prompts to {dd}")

    print(f"\noffset histogram over all checks (sent frame vs official np.linspace(dtype=int) index): "
          f"{ {o: all_offsets.count(o) for o in sorted(set(all_offsets))} }")
    if fails:
        print("\nFAIL:\n  " + "\n  ".join(fails))
        raise SystemExit(1)
    print(f"OK: {KEY}: 32 frames reach the request at native 640x480 for every config; each frame is the official "
          f"frame or its next neighbour (rounding vs floor).")


if __name__ == "__main__":
    main()
