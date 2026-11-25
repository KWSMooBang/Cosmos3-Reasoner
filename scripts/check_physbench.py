#!/usr/bin/env python
"""PhysBench request checks on the real data (no GPU, no server; endpoint checks stubbed).

1. Static, all items: PhysBench.build_prompt() (VIDEO_LLM path) keeps the official order: text pieces
   and media alternate exactly as the `<video>` / `<image>` placeholders of the question, media =
   `file_name` in order, every video message pins nframes=8, joined text == question + END_PROMPT
   with the placeholders removed, no placeholder left. Image-frame path: 8 images per video.
2. Video metadata of every referenced mp4 (frame count >= 8, fps, size).
3. Dynamic, sampled items (all modes, includes the longest/largest videos): the model layer's client
   (CosmosReason2 from the merged run_suite config) runs prepare_inputs(); the request content keeps
   the order, each video carries its pinned frames (8; all-even for the 19 clips < 8 frames; spy on the encoder), plus
   mm_processor_kwargs. A conflicting model-level nframes=4 must not change the count.
4. Tokens: per-item prompt tokens with the Cosmos3-Nano processor geometry
   (patch 16, merge 2 -> 32x32 px per token; image min/max pixels from preprocessor_config.json;
   video frames resized by qwen-vl-utils on the client, 2 frames per temporal patch), text tokens from
   the Nano tokenizer. The analytic count is validated against the HF processor on sampled items.

    PYTHONPATH=third_party/vlmevalkit-cosmos python scripts/check_physbench.py [--key PhysBench] [--sample 40]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import run_suite as rs  # noqa: E402

NANO = glob.glob("/cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/45666784*")
NANO_REV = "45666784803b3a7ad577e0def405f9afcc3bddb2"


def img_tokens(h, w, min_px, max_px, factor=32):
    from qwen_vl_utils.vision_process import smart_resize
    rh, rw = smart_resize(h, w, factor=factor, min_pixels=min_px, max_pixels=max_px)
    return rh * rw // (factor * factor), (rh, rw)


def vid_geometry(h, w, nframes=8, factor=32):
    """qwen-vl-utils fetch_video resize with default limits (no sampling keys but nframes)."""
    from qwen_vl_utils import vision_process as vp
    min_px = vp.VIDEO_MIN_TOKEN_NUM * factor * factor
    total = vp.MODEL_SEQ_LEN * factor * factor * 0.9
    max_px = max(min(vp.VIDEO_MAX_TOKEN_NUM * factor * factor, total / nframes * vp.FRAME_FACTOR), int(min_px * 1.05))
    return vp.smart_resize(h, w, factor=factor, min_pixels=min_px, max_pixels=max_px)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", default="PhysBench")
    ap.add_argument("--model", default="cosmos3_nano_nothink")
    ap.add_argument("--sample", type=int, default=40, help="items for the dynamic client + HF processor check")
    ap.add_argument("--out", default=None, help="write per-item token counts (json)")
    args = ap.parse_args(argv)
    os.environ.setdefault("LMUData", "/data0/woosung/datasets/Cosmos3-Reasoner/LMUData")
    warnings.filterwarnings("ignore")
    import cv2
    from PIL import Image
    import vlmeval
    import vlmeval.api as A
    import vlmeval.api.cosmos_reason as cr
    from vlmeval.dataset import physbench as pb
    print("vlmeval from", vlmeval.__file__)

    ds = pb.PhysBench(args.key)
    data = ds.data
    fails = []
    print(f"{args.key}: {len(data)} items, root {ds.data_root}")

    # ---------------- 1. static order check ----------------
    seq_counter = Counter()
    for _, line in data.iterrows():
        msg = ds.build_prompt(line, video_llm=True)
        files = json.loads(line["file_name"])
        media = [m for m in msg if m["type"] in ("image", "video")]
        sent = iter(json.loads(line["images"]))
        want = [pb.media_path(ds.data_root, f) if f.lower().endswith(".mp4") else os.path.join(ds.data_root, next(sent))
                for f in files]
        if [m["value"] for m in media] != want:
            fails.append(f"{line['index']}: media order")
        for m in media:
            if m["type"] == "video" and (m.get("nframes") != pb.frame_budget(int(line["video_frames"]))
                                         or not m["value"].endswith(".mp4")):
                fails.append(f"{line['index']}: video msg {m}")
        text = "".join(m["value"] for m in msg if m["type"] == "text")
        if text != re.sub(r"<video>|<image>", "", line["question"] + pb.END_PROMPT) or "<image>" in text or "<video>" in text:
            fails.append(f"{line['index']}: text mismatch")
        # official placeholder order == message order
        order = [p for p in re.findall(r"<video>|<image>", line["question"])]
        if [("<video>" if m["type"] == "video" else "<image>") for m in media] != order:
            fails.append(f"{line['index']}: placeholder order")
        seq_counter[" ".join({"text": "T", "image": "I", "video": "V"}[m["type"]] for m in msg)] += 1
    print("message layouts (T=text, I=image, V=video):")
    for k, v in seq_counter.most_common(12):
        print(f"  {v:6d}  {k}")
    print(f"[{'OK' if not fails else 'FAIL'}] static order check over {len(data)} items")

    # ---------------- 1b. image cap: originals unless above the cap ----------------
    cap = ds.image_max_pixels
    n_orig = n_copy = 0
    for _, line in data.iterrows():
        imgs = [f for f in json.loads(line["file_name"]) if not f.lower().endswith(".mp4")]
        for f, sent in zip(imgs, json.loads(line["images"])):
            with Image.open(pb.media_path(ds.data_root, f)) as im:
                native = pb.image_tokens(im.height, im.width)[0]
            if cap is None or native * 1024 <= cap:
                n_orig += 1
                if sent != os.path.join("image", f):
                    fails.append(f"{line['index']}: {f} under the cap but not sent as the original")
            else:
                n_copy += 1
                with Image.open(os.path.join(ds.data_root, sent)) as im:
                    h, w = im.height, im.width
                if not ("image_maxpx" in sent and h % 32 == 0 and w % 32 == 0 and h * w <= cap
                        and pb.image_tokens(h, w)[0] == h * w // 1024):
                    fails.append(f"{line['index']}: capped copy {sent} {w}x{h}")
    print(f"[{'OK' if not fails else 'FAIL'}] image slots: {n_orig} originals, {n_copy} capped copies (cap {cap})")

    # ---------------- 2. video metadata ----------------
    vids = sorted({f for fl in data["file_name"] for f in json.loads(fl) if f.lower().endswith(".mp4")})
    meta = {}
    for v in vids:
        cap = cv2.VideoCapture(pb.media_path(ds.data_root, v))
        meta[v] = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS),
                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        cap.release()
    nf = np.array([m[0] for m in meta.values()])
    dec = {v: e["frames"] for v, e in pb.ensure_media(ds.data_root, vids).items()}  # decord counts (media manifest)
    short = sorted((v for v in vids if dec[v] < 8), key=lambda v: dec[v])
    print(f"videos: {len(vids)}; frames min/median/max {nf.min()}/{int(np.median(nf))}/{nf.max()}; "
          f"decord <8 frames: {len(short)} {[(v, dec[v], pb.frame_budget(dec[v])) for v in short]}; sizes {Counter((m[2], m[3]) for m in meta.values()).most_common(5)}")
    if any(pb.frame_budget(dec[v]) < 2 for v in vids):
        fails.append("video with < 2 frames")

    # ---------------- 4a. analytic tokens (all items) ----------------
    proc_cfg = json.load(open(os.path.join(NANO[0], "preprocessor_config.json")))
    min_px, max_px = proc_cfg["size"]["shortest_edge"], proc_cfg["size"]["longest_edge"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NANO[0])
    img_size = {}
    per_item = {}
    for _, line in data.iterrows():
        files = json.loads(line["file_name"])
        sent = iter(json.loads(line["images"]))  # the image files actually sent (capped copies where needed)
        t_img = t_vid = 0
        for f in files:
            p = None if f.lower().endswith(".mp4") else os.path.join(ds.data_root, next(sent))
            if f.lower().endswith(".mp4"):
                _, _, h, w = meta[f]
                nfr = int(line["video_nframes"])
                rh, rw = vid_geometry(h, w, nframes=nfr)
                t_vid += nfr // 2 * rh * rw // 1024
            else:
                if p not in img_size:
                    with Image.open(p) as im:
                        img_size[p] = (im.height, im.width)
                t_img += img_tokens(*img_size[p], min_px, max_px)[0]
        t_txt = len(tok(line["question"] + pb.END_PROMPT)["input_ids"])
        per_item[int(line["index"])] = dict(image=t_img, video=t_vid, text=t_txt, mode=line["mode"],
                                           total=t_img + t_vid + t_txt)
    tot = np.array([v["total"] for v in per_item.values()])
    print(f"\nanalytic prompt tokens (media + question text, excl. chat template / video timestamps): "
          f"mean {tot.mean():.0f}, p50 {np.percentile(tot, 50):.0f}, p90 {np.percentile(tot, 90):.0f}, "
          f"p99 {np.percentile(tot, 99):.0f}, max {tot.max()}")
    for mode in ("image-only", "image&video", "general"):
        t = np.array([v["total"] for v in per_item.values() if v["mode"] == mode])
        print(f"  {mode:12s} n={len(t):5d} mean {t.mean():7.0f} p99 {np.percentile(t, 99):7.0f} max {t.max():6d}")
    for thr in (8192, 16384, 32768):
        print(f"  items > {thr}: {(tot > thr).sum()}")
    isz = np.array([h * w for h, w in img_size.values()])
    n_cap = int((data["n_images_capped"] > 0).sum())
    print(f"image cap: {ds.image_max_pixels} px -> {n_cap} items / "
          f"{len({p for p in img_size if 'image_maxpx' in p})} unique images use a resized copy; the rest are the originals")
    print(f"images sent: {len(img_size)} unique; px min/median/max {isz.min()}/{int(np.median(isz))}/{isz.max()} "
          f"(server max {max_px} px = {max_px // 1024} tokens)")
    if args.out:
        json.dump(per_item, open(args.out, "w"))

    # ---------------- 3. dynamic client check + 4b. HF processor validation ----------------
    cr.CosmosReason._check_endpoint_health = lambda self: None
    cr.CosmosReason._resolve_model_name = lambda self, *a, **k: None
    seen = []
    orig = cr.tensor_video_to_base64

    def spy(video_tensor, fps, *a, **k):
        seen.append((video_tensor.clone(), fps))
        return orig(video_tensor, fps, *a, **k)
    cr.tensor_video_to_base64 = spy

    item = next(it for it in rs.select_benches(rs.load_suite(REPO / "configs/suites/physics.yaml"), None)
                if it["bench"] == "PhysBench")
    cfg = rs.build_config(rs.load_model_layer(args.model), item, render=True)
    mconf = dict(next(iter(cfg["model"].values())))
    klass = getattr(A, mconf.pop("class"))
    mconf.update(use_video_cache=False, api_base="http://offline.invalid/v1/chat/completions")
    model = klass(**mconf)
    mconf_conflict = dict(mconf, nframes=4)
    model_conflict = klass(**mconf_conflict)

    rng = random.Random(0)
    by_mode = {m: data[data["mode"] == m]["index"].tolist() for m in ("image-only", "image&video", "general")}
    pick = []
    for m, ids in by_mode.items():
        pick += rng.sample(ids, min(len(ids), args.sample // 3))
    biggest = sorted(per_item, key=lambda i: -per_item[i]["total"])[:4]
    capped_ids = data[data["n_images_capped"] > 0]["index"].tolist()
    pick += rng.sample(capped_ids, min(len(capped_ids), 8))
    longest_v = max(vids, key=lambda v: meta[v][0])
    pick += biggest + [int(data[data["video"] == longest_v]["index"].iloc[0])]
    pick += [int(i) for i in data[data["video"].isin(short)]["index"]]  # every short-clip item
    pick = list(dict.fromkeys(pick))

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(NANO[0])
    n_ok = 0
    diffs = []
    frames_seen = {}
    for idx in pick:
        line = data[data["index"] == idx].iloc[0]
        msg = ds.build_prompt(line, video_llm=True)
        for mdl, tag in ((model, "layer"), (model_conflict, "conflict nframes=4")):
            seen.clear()
            msgs, vkw = mdl.prepare_inputs(msg)
            content = msgs[-1]["content"]
            types_ = [c["type"] for c in content]
            want = [{"text": "text", "image": "image_url", "video": "video_url"}[m["type"]] for m in msg]
            pinned = [m["nframes"] for m in msg if m["type"] == "video"]
            frames = [t.shape[0] for t, _ in seen]
            ok = types_ == want and frames == pinned and all(n == 8 or n < 8 and n == pb.frame_budget(
                int(line["video_frames"])) for n in pinned)
            if not ok:
                fails.append(f"{idx} ({tag}): content {types_} vs {want}, frames {frames} pinned {pinned}")
            if tag == "layer":
                frames_seen[idx] = frames
        # HF processor: real prompt tokens for this item
        imgs, vids_in, vmeta, parts = [], [], [], []
        seen.clear()
        model.prepare_inputs(msg)  # the client's own 8-frame tensors (spy) feed the HF processor
        vi = 0
        for m in msg:
            if m["type"] == "text":
                parts.append({"type": "text", "text": m["value"]})
            elif m["type"] == "image":
                imgs.append(Image.open(m["value"]).convert("RGB"))
                parts.append({"type": "image"})
            else:
                t, fps = seen[vi]
                vi += 1
                vids_in.append(t.to(dtype=__import__("torch").uint8))
                vmeta.append(dict(fps=fps, total_num_frames=t.shape[0], frames_indices=list(range(t.shape[0]))))
                parts.append({"type": "video"})
        text = processor.apply_chat_template([{"role": "user", "content": parts}], tokenize=False,
                                             add_generation_prompt=True)
        kw = dict(text=[text], return_tensors="pt")
        if imgs:
            kw["images"] = imgs
        if vids_in:
            kw.update(videos=vids_in, video_metadata=vmeta, do_sample_frames=False)
        out = processor(**kw)
        n_hf = out["input_ids"].shape[1]
        ana = per_item[idx]
        n_img = int((out["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum())
        n_vid = int((out["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")).sum())
        if n_img != ana["image"] or n_vid != ana["video"]:
            fails.append(f"{idx}: HF image/video pads {n_img}/{n_vid} != analytic {ana['image']}/{ana['video']}")
        else:
            n_ok += 1
        diffs.append(n_hf - ana["total"])
        print(f"  idx {idx:5d} {line['mode']:11s} HF total {n_hf:6d} (image {n_img}, video {n_vid}); analytic "
              f"media+question {ana['total']}; overhead {n_hf - ana['total']}; vkw {vkw}")
    print(f"frames in the request per sampled item: {Counter(tuple(v) for v in frames_seen.values())}")
    print(f"[{'OK' if n_ok == len(pick) else 'FAIL'}] HF processor media tokens == analytic on {n_ok}/{len(pick)} items; "
          f"template/timestamp overhead {min(diffs)}..{max(diffs)} tokens")
    worst = max(per_item.values(), key=lambda v: v["total"])["total"] + max(diffs)
    print(f"largest prompt (analytic + max overhead): {worst} tokens")

    if fails:
        print("\nFAIL:\n  " + "\n  ".join(fails[:30]) + (f"\n  ... {len(fails)} total" if len(fails) > 30 else ""))
        raise SystemExit(1)
    print("\nOK: official order and the frame budget (8; all even frames for clips < 8) reach the request.")


if __name__ == "__main__":
    main()
