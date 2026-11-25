#!/usr/bin/env python
"""QuantiPhy prompt-token budget with the Cosmos3-Nano processor (no GPU, no server).

Run with the vLLM env (its transformers is what the server uses), after scripts/check_quantiphy_frames.py
has written the per-video request geometry (frames T, size HxW, fps actually sent by the client):

    LMUData=... .venv-vllm/bin/python scripts/quantiphy_token_budget.py [--key QuantiPhy]

Per question the request is: system = official SYSTEM_PROMPT + "\\n" (BaseAPI appends it),
user = [video, official user text]. For every distinct video geometry the real processor
(AutoProcessor of the pinned Cosmos3-Nano snapshot, do_sample_frames=False, default do_resize as on
the server) is run on a zero tensor of the sent shape: this gives the exact video_grid_thw (incl. the
processor's own resize: Qwen3VLVideoProcessor caps T*H*W at size.longest_edge = 25,165,824 px) and the
exact prompt length. Text-only differences between questions are added from tokenizing the chat text.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SNAP = Path('/cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/45666784803b3a7ad577e0def405f9afcc3bddb2')


def literal(tree, name):
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], 'id', None) == name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--key', default='QuantiPhy')
    ap.add_argument('--reserve', type=int, default=512, help='output tokens to keep free (nothink replies are short)')
    ap.add_argument('--video-max-pixels', type=int, default=None,
                    help='server videos_kwargs.max_pixels (overrides size.longest_edge); 12582912 = serving/ TP=1 '
                         'default (TP1_MM_CAPS); unset = checkpoint default (TP>=2)')
    args = ap.parse_args(argv)
    import torch
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(str(SNAP))
    budget = args.video_max_pixels or proc.video_processor.size['longest_edge']
    size_kw = {}
    if args.video_max_pixels:  # same folding as vLLM Qwen2VLProcessingInfo._get_vision_size
        size_kw = {'size': {'longest_edge': args.video_max_pixels,
                            'shortest_edge': proc.video_processor.size['shortest_edge']}}
    tok = proc.tokenizer
    root = Path(os.environ['LMUData'])
    geom = json.loads((root / 'QuantiPhy' / f'{args.key}_request_geometry.json').read_text())
    with open(root / f'{args.key}.tsv', newline='') as f:
        data = list(csv.DictReader(f, delimiter='\t'))
    src = (REPO / 'third_party/vlmevalkit-cosmos/vlmeval/dataset/quantiphy.py').read_text()
    tree = ast.parse(src)
    system = literal(tree, 'SYSTEM_PROMPT') + '\n'
    post = literal(tree, 'POST_PROMPT')
    # build_user_text without importing vlmeval (this env lacks its deps): exec the two pure functions
    import math
    import types
    pd_shim = types.SimpleNamespace(isna=lambda x: x is None or (isinstance(x, float) and math.isnan(x)))
    ns = {'pd': pd_shim, 'math': math, 'POST_PROMPT': post}
    for fn in ('_isna', 'build_context_prefix', 'build_user_text'):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
        exec(compile(ast.Module([node], []), 'quantiphy.py', 'exec'), ns)

    def nan(x):
        return float('nan') if x == '' else x

    def chat(text):
        msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': system}]},
                {'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': text}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    vid_tok, vid_grid = {}, {}
    for v, g in geom.items():
        sub = [r for r in data if r['video'] == v]
        if not sub:
            continue
        text = chat(ns['build_user_text'](sub[0]['question'], nan(sub[0]['prior']), nan(sub[0]['depth_info']),
                                          sub[0]['video_type']))
        frames = torch.zeros((g['T'], 3, g['H'], g['W']), dtype=torch.uint8)
        out = proc(text=[text], videos=[frames],
                   video_metadata=[{'fps': g['fps'], 'total_num_frames': g['T'], 'frames_indices': list(range(g['T']))}],
                   do_sample_frames=False, return_tensors='pt', **size_kw)
        n_all = int(out['input_ids'].shape[1])
        n_text = len(tok(text)['input_ids']) - 1  # minus the single <|video_pad|> placeholder
        vid_tok[v] = n_all - n_text
        t, h, w = out['video_grid_thw'][0].tolist()
        vid_grid[v] = (t, h * 16, w * 16)

    tot, rows = [], []
    for r in data:
        text = chat(ns['build_user_text'](r['question'], nan(r['prior']), nan(r['depth_info']), r['video_type']))
        n = len(tok(text)['input_ids']) - 1 + vid_tok[r['video']]
        tot.append(n)
    tot = np.array(tot)
    print(f"{args.key}: {len(data)} questions, {len(vid_tok)} videos; processor {SNAP.name[:8]} "
          f"(video longest_edge {budget})")
    print('per video: frames sent -> server grid (T/2 x H x W after the processor resize) -> video tokens')
    for v in sorted(vid_tok, key=lambda k: -vid_tok[k]):
        g = geom[v]
        t, h, w = vid_grid[v]
        print(f"  {v.strip():30s} {g['T']:3d} x {g['W']}x{g['H']} -> grid t={t} {w}x{h} -> {vid_tok[v]:6d} tokens")
    q = np.percentile(tot, [0, 50, 90, 99, 100]).astype(int)
    print(f"\nprompt tokens per question: min {q[0]} p50 {q[1]} p90 {q[2]} p99 {q[3]} max {q[4]}")
    for L in (16384, 65536):
        print(f"  fits max-model-len {L} (prompt <= {L - args.reserve}): {(tot <= L - args.reserve).mean() * 100:.1f}%")
    down = [v for v in vid_grid if (vid_grid[v][1], vid_grid[v][2]) != (geom[v]['H'], geom[v]['W'])]
    nq = sum(1 for r in data if r['video'] in down)
    print(f"videos resized below the sent size by the server processor (T*H*W > {budget}): "
          f"{len(down)}/{len(vid_grid)} ({nq}/{len(data)} questions)")


if __name__ == '__main__':
    main()
