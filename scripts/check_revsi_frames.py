#!/usr/bin/env python
"""Prove on the real ReVSI files that the official frames reach the request (no GPU / server).

For a ReVSI key (default ReVSI_64frame, N = 64):
  1. Builds the dataset exactly as run.py does (data config -> vlmeval ReVSI) from $LMUData.
  2. For --scenes sampled questions (distinct scenes, seed 0) takes the message built by
     ReVSI.build_prompt(video_llm=True) and checks that it is a video message with no sampling
     keys pointing at <N>_frame/<scene>.mp4, and that this mp4 has exactly N frames.
  3. Builds the model from the merged run_suite config (endpoint checks stubbed) and runs the
     client preprocessing (prepare_inputs -> qwen-vl-utils -> base64 mp4), capturing the frame
     tensor that would be sent. Checks: T == N, and frame i of the request is frame i of the
     official mp4 (nearest-neighbour match over all N decoded frames, on 32x24 thumbnails).
  4. Provenance: for --provenance scenes, decodes the all-frame video from video.zip and checks
     that frame i of the N-frame mp4 is all-frame frame sampled_video_frame_idx[scene]['<N>-frame'][i]
     (nearest neighbour within +-5 frames).
  5. Non-VIDEO_LLM path: save_video_frames returns N images.
  6. Prints the prompt of one numeric and one multiple-choice question.

    python scripts/check_revsi_frames.py [--key ReVSI_64frame] [--model cosmos3_nano_nothink] [--scenes 5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import run_suite as rs  # noqa: E402

SAMPLING_KEYS = {"nframes", "fps", "total_pixels", "max_pixels", "min_pixels", "max_frames"}


def thumbs(frames, size=(32, 24)):
    """frames: iterable of HxWx3 uint8 arrays -> (n, 24*32*3) float thumbnails."""
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(f).resize(size, Image.BILINEAR), dtype=np.float32).ravel()
                     for f in frames])


def decode_all(path):
    import decord
    vr = decord.VideoReader(str(path))
    return vr.get_batch(list(range(len(vr)))).asnumpy()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", default="ReVSI_64frame")
    ap.add_argument("--model", default="cosmos3_nano_nothink")
    ap.add_argument("--scenes", type=int, default=5)
    ap.add_argument("--provenance", type=int, default=2)
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    import vlmeval.api as A
    import vlmeval.api.cosmos_reason as cr
    import vlmeval.dataset as D

    cr.CosmosReason._check_endpoint_health = lambda self: None
    cr.CosmosReason._resolve_model_name = lambda self, *a, **k: None
    captured = []
    orig_b64 = cr.tensor_video_to_base64

    def spy(video_tensor, fps, *a, **k):
        captured.append((video_tensor.detach().clone(), fps))
        return orig_b64(video_tensor, fps, *a, **k)
    cr.tensor_video_to_base64 = spy

    item = {"key": f"spatial/{args.key}", "domain": "spatial", "bench": args.key, "run": {}}
    cfg = rs.build_config(rs.load_model_layer(args.model), item, render=True)
    mconf = dict(next(iter(cfg["model"].values())))
    dconf = dict(cfg["data"][args.key])
    cls = getattr(D, dconf.pop("class"))
    dconf.pop("model_family", None)
    ds = cls(dataset=args.key, **dconf)
    n = ds.num_frames
    print(f"{args.key}: {len(ds)} questions, {ds.data['video'].nunique()} videos, frame budget N={n}; "
          f"model layer {args.model}: client sampling "
          f"{ {k: mconf[k] for k in rs.VIDEO_SAMPLING_KEYS if mconf.get(k) is not None} }")
    print("question types:", ds.data["question_type"].value_counts().to_dict())

    klass = getattr(A, mconf.pop("class"))
    mconf.update(use_video_cache=False, api_base="http://offline.invalid/v1/chat/completions")
    model = klass(**mconf)

    fails = []
    rng = np.random.default_rng(0)
    first = ds.data.drop_duplicates("video")
    picks = first.iloc[rng.choice(len(first), size=min(args.scenes, len(first)), replace=False)]
    for _, line in picks.iterrows():
        msg = ds.build_prompt(line, video_llm=True)
        vids = [m for m in msg if m["type"] == "video"]
        assert len(vids) == 1 and not (set(vids[0]) & SAMPLING_KEYS), vids
        path = vids[0]["value"]
        official = decode_all(path)
        captured.clear()
        _, vkw = model.prepare_inputs(msg)
        (tensor, sent_fps), = captured
        sent = tensor.float().numpy()
        if sent.max() <= 1.0 + 1e-6:
            sent = sent * 255.0
        sent = np.clip(sent, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)  # T,H,W,C
        a, b = thumbs(sent), thumbs(official)
        d = ((a[:, None, :] - b[None, :, :]) ** 2).mean(-1)
        match = d.argmin(1)
        ok = len(official) == n and sent.shape[0] == n and np.array_equal(match, np.arange(n))
        print(f"[{'OK' if ok else 'FAIL'}] {os.path.relpath(path, ds.data_root)}: mp4 has {len(official)} frames "
              f"({official.shape[2]}x{official.shape[1]}), request has {sent.shape[0]} frames "
              f"({sent.shape[2]}x{sent.shape[1]}), frame i of request == frame i of mp4 for "
              f"{int((match == np.arange(len(match))).sum())}/{len(match)}; mm_processor_kwargs={vkw}")
        if not ok:
            fails.append(path)

    # Provenance vs. sampled_video_frame_idx.json (frames of the all-frame video).
    root = Path(ds.data_root).parent
    idx_map = json.loads((root / "metadata" / "sampled_video_frame_idx.json").read_text())
    zpath = root / "video.zip"
    if args.provenance and zpath.exists():
        with zipfile.ZipFile(zpath) as zf:
            names = {Path(i.filename).parent.name + "/" + Path(i.filename).name: i.filename for i in zf.infolist()}
            for _, line in picks.head(args.provenance).iterrows():
                scene = str(line["scene_id"])
                member = names.get(f"all_frame/{scene}.mp4")
                if member is None:
                    print(f"[SKIP] provenance {scene}: no all_frame/{scene}.mp4 in zip")
                    continue
                with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
                    tmp = Path(td) / f"{scene}.mp4"
                    tmp.write_bytes(zf.read(member))
                    allf = decode_all(tmp)
                sel = idx_map[scene][f"{n}-frame"]
                sub = decode_all(ds.video_path(line["video"]))
                ta, ts = thumbs(allf), thumbs(sub)
                offs = []
                for i, j in enumerate(sel):
                    lo, hi = max(0, j - 5), min(len(allf), j + 6)
                    offs.append(int(np.argmin(((ta[lo:hi] - ts[i]) ** 2).mean(-1)) + lo - j))
                ok = len(sel) == n == len(sub) and all(o == 0 for o in offs)
                print(f"[{'OK' if ok else 'FAIL'}] provenance {scene}: {n}-frame mp4 frame i == all-frame frame "
                      f"sampled_video_frame_idx[i] for {sum(o == 0 for o in offs)}/{len(sel)} "
                      f"(all-frame video {len(allf)} frames; idx {sel[:3]}...{sel[-2:]})")
                if not ok:
                    fails.append(f"provenance {scene}: offsets {offs}")

    # Non-VIDEO_LLM path
    line = picks.iloc[0]
    msg = ds.build_prompt(line, video_llm=False)
    n_img = sum(m["type"] == "image" for m in msg)
    print(f"[{'OK' if n_img == n else 'FAIL'}] image-frame path: {n_img} images for {line['video']}")
    if n_img != n:
        fails.append(f"image path {n_img} != {n}")

    for qt in ("object_abs_distance", "object_rel_direction_forward_easy"):
        sub = ds.data[ds.data["question_type"] == qt]
        if len(sub):
            print(f"\n--- prompt ({qt}) ---\n{ds.build_prompt(sub.iloc[0], video_llm=True)[-1]['value']}")

    if fails:
        print("\nFAIL:\n  " + "\n  ".join(map(str, fails)))
        raise SystemExit(1)
    print(f"\nOK: {args.key}: the official {n} frames reach the request, in order.")


if __name__ == "__main__":
    main()
