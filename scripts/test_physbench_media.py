#!/usr/bin/env python
"""PhysBench media-integrity test on the real data (no GPU; needs the extracted media + zips).

Damages two extracted files and checks that loading the dataset repairs them from the pinned zips:
  1. an image overwritten in place with the same size, original mtime restored: the stat check cannot
     see it, so a normal load keeps it and PHYSBENCH_VERIFY_MEDIA=1 must re-hash and repair it;
  2. a video truncated: the size check catches it on a normal load; the repaired file's sha256 and
     frame count equal the manifest again, and the TSV is unchanged (same media digest).
Also checks that a changed frame count in the manifest forces a TSV rebuild.

    PYTHONPATH=third_party/vlmevalkit-cosmos python scripts/test_physbench_media.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def main() -> None:
    os.environ.setdefault('LMUData', '/data0/woosung/datasets/Cosmos3-Reasoner/LMUData')
    from vlmeval.dataset import physbench as pb
    from vlmeval.smp import LMUDataRoot

    key = 'PhysBench_val'
    root = pb.physbench_root()
    ds = pb.PhysBench(key)
    tsv = os.path.join(LMUDataRoot(), f'{key}.tsv')
    tsv_md5 = md5(tsv)
    man_path = os.path.join(root, 'media_manifest.json')
    man = json.load(open(man_path))['files']
    files = [f for fl in ds.data['file_name'] for f in json.loads(fl)]
    img = next(f for f in files if not f.lower().endswith('.mp4'))
    vid = next(f for f in files if f.lower().endswith('.mp4'))
    fails = []

    # 1. silent same-size corruption of an image
    p = pb.media_path(root, img)
    st = os.stat(p)
    data = bytearray(open(p, 'rb').read())
    data[len(data) // 2] ^= 0xFF
    open(p, 'wb').write(bytes(data))
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    pb.PhysBench(key)
    if pb._sha256(p) == man[img]['sha256']:
        fails.append('stat-only load unexpectedly re-hashed')  # documents the cheap path
    os.environ['PHYSBENCH_VERIFY_MEDIA'] = '1'
    pb.PhysBench(key)
    del os.environ['PHYSBENCH_VERIFY_MEDIA']
    ok1 = pb._sha256(p) == man[img]['sha256']
    print(f"[{'OK' if ok1 else 'FAIL'}] same-size corruption of {img}: kept by the stat check, repaired with "
          f"PHYSBENCH_VERIFY_MEDIA=1")
    if not ok1:
        fails.append('image not repaired')

    # 2. truncated video
    p = pb.media_path(root, vid)
    with open(p, 'r+b') as f:
        f.truncate(os.path.getsize(p) // 2)
    pb.PhysBench(key)
    new = json.load(open(man_path))['files'][vid]
    ok2 = (pb._sha256(p) == man[vid]['sha256'] and new['frames'] == man[vid]['frames']
           and md5(tsv) == tsv_md5)
    print(f"[{'OK' if ok2 else 'FAIL'}] truncated {vid}: re-extracted, sha256 and frame count "
          f"({new['frames']}) as before, TSV unchanged")
    if not ok2:
        fails.append('video not repaired')

    # 3. a frame-count change must rebuild the TSV
    full = json.load(open(man_path))
    full['files'][vid]['frames'] += 2
    json.dump(full, open(man_path, 'w'))
    d = pb.PhysBench(key)
    row = d.data[d.data['video'] == vid].iloc[0]
    ok3 = md5(tsv) != tsv_md5 and int(row['video_frames']) == man[vid]['frames'] + 2
    full['files'][vid]['frames'] -= 2
    json.dump(full, open(man_path, 'w'))
    pb.PhysBench(key)
    ok3 = ok3 and md5(tsv) == tsv_md5
    print(f"[{'OK' if ok3 else 'FAIL'}] changed frame count in the manifest -> TSV rebuilt; restored -> "
          f"original TSV")
    if not ok3:
        fails.append('TSV not rebuilt on frame-count change')

    if fails:
        print('FAIL: ' + '; '.join(fails))
        sys.exit(1)
    print('OK: media manifest repairs damaged files and keeps the TSV in sync.')


if __name__ == '__main__':
    main()
