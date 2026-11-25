#!/usr/bin/env python
"""Prove on the real QuantiPhy files that every frame reaches the request (no GPU / server).

Official protocol (github.com/Paulineli/QuantiPhy run_API_results.extract_frames): every frame of the
480p video. For each video of the key (default QuantiPhy = validation, all 24 videos):
  1. Builds the dataset as run.py does (data config -> vlmeval QuantiPhy) from $LMUData.
  2. Takes the message of QuantiPhy.build_prompt(video_llm=True) and checks: one system item (official
     SYSTEM_PROMPT), one video message whose only sampling key is nframes = 2*floor(n/2) (n = decord
     frame count of the mp4), then the official user text.
  3. Runs the client preprocessing (BaseAPI.preprocess_message_with_role -> prepare_inputs ->
     qwen-vl-utils -> base64 mp4; endpoint checks stubbed) with the merged run_suite model config and
     with conflicting model-level settings (nframes=8 / fps=1.0 / max_frames=4). Captures the frame
     tensor that would be sent and checks T == 2*floor(n/2), that request frame i is mp4 frame
     round(linspace(0, n-1, T))[i] (nearest-neighbour over all decoded frames on 32x24 thumbnails;
     ties with identical neighbours accepted), the frame size, and the fps sent to the server.
  4. Non-VIDEO_LLM path: save_video_frames returns n images (= the official request).
  5. Writes per-video (n, T, H, W, sent fps) to --out (JSON) for scripts/quantiphy_token_budget.py.

    python scripts/check_quantiphy_frames.py [--key QuantiPhy] [--model cosmos3_nano_nothink] [--videos 0]
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
sys.path.insert(0, str(REPO / "scripts"))
import run_suite as rs  # noqa: E402

SAMPLING_KEYS = {"nframes", "fps", "total_pixels", "max_pixels", "min_pixels", "max_frames"}
TWIN_MSE = 0.01  # 32x24 thumbnail MSE (0-255 scale) below which two mp4 frames count as the same image


def thumbs(frames, size=(32, 24)):
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(f).resize(size, Image.BILINEAR), dtype=np.float32).ravel()
                     for f in frames])


def decode_all(path):
    import decord
    vr = decord.VideoReader(str(path))
    return vr.get_batch(list(range(len(vr)))).asnumpy(), vr.get_avg_fps()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", default="QuantiPhy")
    ap.add_argument("--model", default="cosmos3_nano_nothink")
    ap.add_argument("--videos", type=int, default=0, help="check only this many videos (0 = all)")
    ap.add_argument("--quick", action="store_true",
                    help="merged run_suite config only (skip the conflicting model-level variants); "
                         "use for QuantiPhy_test (568 videos)")
    ap.add_argument("--out", default=None, help="JSON with per-video request geometry "
                    "(default: $LMUData/QuantiPhy/<key>_request_geometry.json)")
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    import torch
    import vlmeval.api as A
    import vlmeval.api.cosmos_reason as cr
    import vlmeval.dataset as D
    from vlmeval.dataset import quantiphy as qp

    print(f"vlmeval imported from {Path(D.__file__).parent.parent}")
    cr.CosmosReason._check_endpoint_health = lambda self: None
    cr.CosmosReason._resolve_model_name = lambda self, *a, **k: None
    captured = []
    orig_b64 = cr.tensor_video_to_base64

    def spy(video_tensor, fps, *a, **k):
        captured.append((video_tensor.detach().clone(), fps))
        return orig_b64(video_tensor, fps, *a, **k)
    cr.tensor_video_to_base64 = spy

    item = {"key": f"physics/{args.key}", "domain": "physics", "bench": args.key, "run": {}}
    cfg = rs.build_config(rs.load_model_layer(args.model), item, render=True)
    mconf = dict(next(iter(cfg["model"].values())))
    dconf = dict(cfg["data"][args.key])
    cls = getattr(D, dconf.pop("class"))
    dconf.pop("model_family", None)
    ds = cls(dataset=args.key, **dconf)
    print(f"{args.key}: {len(ds)} questions, {ds.data['video'].nunique()} videos; categories "
          f"{ds.data.apply(lambda r: r['inference_type'][0] + r['video_type'][1], axis=1).value_counts().to_dict()}")

    klass = getattr(A, mconf.pop("class"))
    base = dict(mconf, use_video_cache=False, api_base="http://offline.invalid/v1/chat/completions")
    bare = {k: v for k, v in base.items() if k not in rs.VIDEO_SAMPLING_KEYS}
    variants = {
        "run_suite config": base,
        "conflicting model-level nframes=8": dict(bare, nframes=8),
        "conflicting model-level fps=1.0": dict(bare, fps=1.0),
        "conflicting model-level max_frames=4": dict(bare, max_frames=4),
    }
    if args.quick:
        variants = {"run_suite config": base}

    fails, geom = [], {}
    first = ds.data.drop_duplicates("video")
    if args.videos:
        first = first.head(args.videos)
    for _, line in first.iterrows():
        msg = ds.build_prompt(line, video_llm=True)
        sysm = [m for m in msg if m.get("role") == "system"]
        vids = [m for m in msg if m["type"] == "video"]
        texts = [m for m in msg if m["type"] == "text" and m.get("role") != "system"]
        path = vids[0]["value"]
        frames, vfps = decode_all(path)
        n = len(frames)
        T_exp = n - n % 2
        ok_msg = (len(sysm) == 1 and sysm[0]["value"] == qp.SYSTEM_PROMPT and len(vids) == 1
                  and {k for k in vids[0] if k in SAMPLING_KEYS} == {"nframes"} and vids[0]["nframes"] == T_exp
                  and int(line["n_frames"]) == n and len(texts) == 1
                  and texts[0]["value"] == qp.build_user_text(line["question"], line["prior"], line["depth_info"],
                                                              line["video_type"])
                  and msg.index(vids[0]) < msg.index(texts[0]))
        if not ok_msg:
            fails.append(f"{line['video']}: message structure {msg}")
        exp_idx = torch.linspace(0, n - 1, T_exp).round().long().numpy()
        b = thumbs(frames)
        for vname, kw in variants.items():
            model = klass(**kw)
            captured.clear()
            m2 = model.preprocess_message_with_role([dict(x) for x in msg])
            payload, vkw = model.prepare_inputs(m2)
            (tensor, sent_fps), = captured
            sent = tensor.float().numpy()
            if sent.max() <= 1.0 + 1e-6:
                sent = sent * 255.0
            sent = np.clip(sent, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
            d = ((thumbs(sent)[:, None, :] - b[None, :, :]) ** 2).mean(-1)
            if len(sent) == T_exp:
                rows = np.arange(len(sent))
                nn = d.argmin(1)
                # Accept the expected frame if it is (one of) the nearest, or if it is indistinguishable from the
                # nearest mp4 frame itself (static scenes: consecutive frames differ by ~0.002 thumbnail MSE).
                twin = ((b[exp_idx] - b[nn]) ** 2).mean(-1) <= TWIN_MSE
                good = (d[rows, exp_idx] <= d[rows, nn] + 1e-3) | twin
            else:
                good = []
            sys_ok = payload[0]["role"] == "system" and qp.SYSTEM_PROMPT in payload[0]["content"]
            ok = len(sent) == T_exp and bool(np.all(good)) and sys_ok
            H, W = sent.shape[1:3]
            if vname == "run_suite config":
                geom[line["video"]] = dict(n=n, T=int(len(sent)), H=int(H), W=int(W), fps=float(sent_fps),
                                           video_fps=float(vfps), mm_processor_kwargs=vkw)
                print(f"[{'OK' if ok else 'FAIL'}] {line['video']:28s} mp4 {n:3d} frames {frames.shape[2]}x"
                      f"{frames.shape[1]} @ {vfps:.2f} fps -> request {len(sent):3d} frames {W}x{H} @ "
                      f"{sent_fps:.2f} fps; frame i == mp4 frame linspace[i] for {int(np.sum(good))}/{len(sent)}; "
                      f"system prompt {'present' if sys_ok else 'MISSING'}; mm_processor_kwargs={vkw}")
            elif not ok:
                print(f"[FAIL] {line['video']} [{vname}]: {len(sent)} frames")
            if not ok:
                fails.append(f"{line['video']} [{vname}]")
    if len(variants) > 1:
        print(f"[{'OK' if not fails else 'FAIL'}] conflicting model-level settings ({', '.join(list(variants)[1:])}) "
              "did not change any request")

    # payload of one request (nothink: one system message = official prompt)
    model = klass(**base)
    m2 = model.preprocess_message_with_role([dict(x) for x in ds.build_prompt(first.iloc[0], video_llm=True)])
    payload, _ = model.prepare_inputs(m2)
    shown = [{"role": p["role"], "content": p["content"] if isinstance(p["content"], str) else
              [c if c["type"] == "text" else {"type": c["type"], "url": "<base64 mp4>"} for c in p["content"]]}
             for p in payload]
    print("\n--- request messages (first question) ---\n" + json.dumps(shown, indent=1, ensure_ascii=False))

    # Non-VIDEO_LLM path
    line = first.iloc[0]
    msg = ds.build_prompt(line, video_llm=False)
    n_img = sum(m["type"] == "image" for m in msg)
    ok = n_img == int(line["n_frames"])
    print(f"[{'OK' if ok else 'FAIL'}] image-frame path: {n_img} images for {line['video']} "
          f"({line['n_frames']} frames)")
    if not ok:
        fails.append(f"image path {n_img}")

    n_all = np.array([g["n"] for g in geom.values()])
    t_all = np.array([g["T"] for g in geom.values()])
    print(f"\nframes per video: n min/p50/max {n_all.min()}/{int(np.median(n_all))}/{n_all.max()}; "
          f"sent {t_all.sum()}/{n_all.sum()} frames ({int((n_all % 2).sum())} odd-length videos lose 1 frame)")
    out = Path(args.out) if args.out else Path(os.environ["LMUData"]) / "QuantiPhy" / f"{args.key}_request_geometry.json"
    out.write_text(json.dumps(geom, indent=1))
    print(f"request geometry -> {out}")

    if fails:
        print("\nFAIL:\n  " + "\n  ".join(map(str, fails)))
        raise SystemExit(1)
    print(f"\nOK: {args.key}: every frame (even-rounded) of every checked video reaches the request, in order.")


if __name__ == "__main__":
    main()
