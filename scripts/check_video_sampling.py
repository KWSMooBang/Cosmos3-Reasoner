#!/usr/bin/env python
"""Offline proof that each video bench's frame settings actually reach the request (no GPU/server).

For every video bench in configs/suites/*.yaml (data config with `video_sampling` or nframe/fps):

1. Static: walk the dataset class's `build_prompt` (MRO) with `ast` and collect every
   `dict(type='video', ...)` it builds. `CosmosReason._get_video_processing_kwargs` ignores ALL
   model-level sampling attributes as soon as a video message carries ANY sampling key
   (nframes/fps/total_pixels/max_pixels/min_pixels/max_frames), so such a key would silently
   override our config. A bench may pin keys in the message only if its data config's
   `video_sampling` declares exactly those keys and no other model-level sampling key would be
   dropped (ReVSI pins nframes=N); the dynamic check then also sets a conflicting model-level value
   and verifies the pinned one wins. Any other sampling key in a video message fails. Benches that never build a video message (frames
   are sent as images, e.g. MMSIVideoBench) are sampled by the *dataset* (`nframe`/`fps`).
2. Dynamic, video-message benches: build the model from the merged run_suite config (endpoint
   checks stubbed), feed it a synthetic 40 s / 30 fps mp4 exactly as the dataset does
   (`{'type': 'video', 'value': path}`), run the client-side preprocessing
   (`prepare_inputs` -> qwen-vl-utils -> base64 mp4) and count the frames that would be sent,
   plus the `mm_processor_kwargs` of the request.
3. Dynamic, image-frame benches: run the dataset's own `save_video_frames` on a synthetic
   2-segment frame list (300 frames) and count the frames kept.

    python scripts/check_video_sampling.py [--model cosmos3_nano_nothink] [--seconds 40]
"""

from __future__ import annotations

import argparse
import ast
import inspect
import os
import sys
import tempfile
import textwrap
import types
import warnings
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import run_suite as rs  # noqa: E402

SAMPLING_KEYS = {"nframes", "fps", "total_pixels", "max_pixels", "min_pixels", "max_frames"}


def video_message_keys(cls) -> list[set[str]] | None:
    """Keyword sets of every dict(type='video', ...) / {'type': 'video', ...} in build_prompt."""
    fn = next((vars(k)["build_prompt"] for k in cls.__mro__ if "build_prompt" in vars(k)), None)
    if fn is None:
        return None
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dict":
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            if isinstance(kw.get("type"), ast.Constant) and kw["type"].value == "video":
                found.append(set(kw) | ({"**"} if any(k.arg is None for k in node.keywords) else set()))
        elif isinstance(node, ast.Dict):
            keys = [k.value if isinstance(k, ast.Constant) else "**" for k in node.keys]
            vals = dict(zip(keys, node.values))
            if isinstance(vals.get("type"), ast.Constant) and vals["type"].value == "video":
                found.append(set(keys))
    return found


def make_video(path: Path, seconds: int, fps: int = 30) -> None:
    import cv2
    import numpy as np
    w, h = 320, 240
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(seconds * fps):
        frame = np.full((h, w, 3), i % 256, dtype=np.uint8)
        cv2.putText(frame, str(i), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        out.write(frame)
    out.release()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="cosmos3_nano_nothink")
    ap.add_argument("--seconds", type=int, default=40, help="synthetic video length (30 fps)")
    args = ap.parse_args(argv)

    os.environ.setdefault("LMUData", str(REPO / ".lmudata_unused"))
    warnings.filterwarnings("ignore")
    import vlmeval.api as A
    import vlmeval.api.cosmos_reason as cr
    import vlmeval.dataset as D

    # No endpoint: stub the constructor's network checks.
    cr.CosmosReason._check_endpoint_health = lambda self: None
    cr.CosmosReason._resolve_model_name = lambda self, *a, **k: None
    seen_tensors = []
    orig_b64 = cr.tensor_video_to_base64

    def spy(video_tensor, fps, *a, **k):
        seen_tensors.append((tuple(video_tensor.shape), fps))
        return orig_b64(video_tensor, fps, *a, **k)
    cr.tensor_video_to_base64 = spy

    model_layer = rs.load_model_layer(args.model)
    tmp = Path(tempfile.mkdtemp(prefix="vidcheck_", dir=os.environ.get("TMPDIR")))
    video = tmp / "synthetic.mp4"
    make_video(video, args.seconds)

    items, seen = [], set()
    for spath in sorted((REPO / "configs" / "suites").glob("*.yaml")):
        for it in rs.select_benches(rs.load_suite(spath), None):
            dconf = rs.load_data_conf(it["domain"], it["bench"])
            if it["key"] not in seen and ("video_sampling" in dconf or "nframe" in dconf or "fps" in dconf):
                seen.add(it["key"])
                items.append((it, dconf))

    failures = []
    print(f"model layer: {args.model}; synthetic video: {args.seconds}s @ 30 fps = {args.seconds * 30} frames\n")
    for it, dconf in items:
        cfg = rs.build_config(model_layer, it, render=True)
        mconf = dict(next(iter(cfg["model"].values())))
        dsconf = next(iter(cfg["data"].values()))
        cls = getattr(D, dsconf["class"])
        vkeys = video_message_keys(cls)
        name = it["bench"]
        pinned, want = set(), {}
        for k in (vkeys or []):
            if "**" in k:
                failures.append(f"{name}: {cls.__name__}.build_prompt builds a video message with **kwargs; "
                                "cannot verify which sampling keys it carries")
            pinned |= k & SAMPLING_KEYS
        vs = dconf.get("video_sampling", {})
        if pinned:
            # The dataset pins sampling keys in the message -> the client ignores ALL model-level
            # sampling attributes. Allowed only if the data config's video_sampling declares exactly
            # those keys (same values by convention; real values are proven by a bench-specific check,
            # e.g. scripts/check_revsi_frames.py) and no other model-level key would be silently dropped.
            if pinned != set(vs):
                failures.append(f"{name}: build_prompt pins {sorted(pinned)} but data video_sampling declares "
                                f"{sorted(vs)}")
            dropped = {k: mconf[k] for k in rs.VIDEO_SAMPLING_KEYS
                       if mconf.get(k) is not None and (k not in pinned or mconf[k] != vs.get(k))}
            if dropped:
                failures.append(f"{name}: model-level {dropped} would be ignored (video message pins {sorted(pinned)})")
        if vkeys:  # raw video handed to the client
            want = {k: mconf[k] for k in rs.VIDEO_SAMPLING_KEYS if mconf.get(k) is not None}
            klass = getattr(A, mconf.pop("class"))
            mconf.update(use_video_cache=False, api_base="http://offline.invalid/v1/chat/completions")
            msg = dict(type="video", value=str(video))
            if pinned:
                want = {k: vs[k] for k in pinned}
                msg.update(want)
                # conflicting model-level values must lose against the pinned message keys
                if "nframes" in pinned:
                    mconf.update(nframes=7, fps=None)
                else:
                    mconf.update(fps=0.25, nframes=None)
            model = klass(**mconf)
            seen_tensors.clear()
            msgs, vkw = model.prepare_inputs([msg, dict(type="text", value="How many objects?")])
            (shape, sent_fps), = seen_tensors
            nf = shape[0]
            if "nframes" in want:
                expect = want["nframes"]
            else:  # qwen-vl-utils fps mode: round(duration * fps) to a multiple of 2, >= 4
                expect = max(4, round(args.seconds * want["fps"] / 2) * 2)
            ok = nf == expect
            src = "pinned in message (model-level conflict ignored)" if pinned else "client sampling"
            print(f"[{'OK' if ok else 'FAIL'}] {name:20s} video msg keys={sorted(vkeys[0])} -> {src} "
                  f"{want} -> {nf} frames sent (expected {expect}), frame {shape[2]}x{shape[3]}, "
                  f"mm_processor_kwargs={vkw}")
            if not ok:
                failures.append(f"{name}: {nf} frames sent, expected {expect}")
        else:  # frames sent as images; dataset-side sampling
            fake = types.SimpleNamespace(nframe=dsconf.get("nframe", 0), fps=dsconf.get("fps", -1))
            segs = [[f"/f/a_{i}.jpg" for i in range(120)], [f"/f/b_{i}.jpg" for i in range(180)]]
            kept = cls.save_video_frames(fake, segs, 30.0)
            nf = sum(len(s) for s in kept)
            expect = min(dsconf.get("nframe", 0), 300) if dsconf.get("nframe", 0) > 0 else None
            ok = expect is None or nf == expect
            ignored = sorted(k for k in rs.VIDEO_SAMPLING_KEYS if mconf.get(k) is not None)
            print(f"[{'OK' if ok else 'FAIL'}] {name:20s} no video msg (frames as images) -> dataset nframe="
                  f"{fake.nframe} fps={fake.fps} -> {nf}/300 frames kept"
                  + (f"; model-level {ignored} unused" if ignored else ""))
            if not ok:
                failures.append(f"{name}: dataset kept {nf} frames, expected {expect}")

    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        raise SystemExit(1)
    print(f"\nOK: {len(items)} video benches; configured frame settings reach the request.")


if __name__ == "__main__":
    main()
