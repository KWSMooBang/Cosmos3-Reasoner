#!/usr/bin/env python
"""QuantiPhy cache-integrity tests (review fix 2), on a temporary copy of $LMUData/QuantiPhy (real cache untouched).

Validation key (copy of validation/, ~6 MB; repairs re-download single files from HF at the pinned revision):
  1. partial download: question table present, video folder present but 3 clips missing -> re-downloaded,
     TSV built; a second load re-downloads nothing.
  2. empty clip, truncated clip (wrong size) -> re-downloaded, TSV md5 unchanged.
  3. same-size corruption that keeps the mtime: survives the normal stat check, repaired with
     QUANTIPHY_VERIFY_MEDIA=1.
  4. question table changed -> re-downloaded (pinned sha256).
  5. TSV reuse: an edited TSV (md5 mismatch) is rebuilt; a TSV whose n_frames column disagrees with the verified
     media is rebuilt even when its manifest md5 was updated to match; a missing manifest -> rebuilt.
QuantiPhy_test transcodes (ensure_transcodes on 2 real source clips):
  6. made once, reused on the next call; deleted / truncated output -> re-made; same-size corruption with kept
     mtime -> re-made under QUANTIPHY_VERIFY_MEDIA=1; changed source sha256 -> re-made; frame count recorded == source.

    python scripts/test_quantiphy_media.py
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path

FAILS: list[str] = []


def check(ok, msg):
    print(f"[{'OK' if ok else 'FAIL'}] {msg}")
    if not ok:
        FAILS.append(msg)


def corrupt_same_size(p: Path):
    st = p.stat()
    b = bytearray(p.read_bytes())
    mid = len(b) // 2
    b[mid:mid + 64] = bytes(x ^ 0xFF for x in b[mid:mid + 64])
    p.write_bytes(bytes(b))
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))


def main() -> None:
    warnings.filterwarnings('ignore')
    real = Path(os.environ['LMUData'])
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as td:
        lmu = Path(td) / 'LMUData'
        (lmu / 'QuantiPhy').mkdir(parents=True)
        shutil.copytree(real / 'QuantiPhy' / 'validation', lmu / 'QuantiPhy' / 'validation',
                        ignore=shutil.ignore_patterns('.cache'))
        os.environ['LMUData'] = str(lmu)
        import vlmeval.dataset as D
        from vlmeval.dataset import quantiphy as qp
        from vlmeval.smp import LMUDataRoot
        assert LMUDataRoot() == str(lmu), LMUDataRoot()
        print(f'vlmeval imported from {Path(D.__file__).parent.parent}; temp LMUData {lmu}')
        val = lmu / 'QuantiPhy' / 'validation'
        vdir = val / 'validation_videos'
        tsv = lmu / 'QuantiPhy.tsv'
        man = lmu / 'QuantiPhy.manifest.json'
        (val / qp.VERIFIED).unlink(missing_ok=True)
        hfm = json.loads((val / qp.HF_MANIFEST).read_text()) if (val / qp.HF_MANIFEST).exists() else None

        # 1. partial download
        clips = sorted(p for p in vdir.iterdir() if p.suffix == '.mp4')
        for p in clips[:3]:
            p.unlink()
        ds = D.QuantiPhy('QuantiPhy')
        check(all(p.exists() and p.stat().st_size > 0 for p in clips[:3]) and len(ds) == 158 and tsv.exists(),
              'partial download (3 clips missing, table + folder present) -> clips re-downloaded, TSV built')
        hfm = json.loads((val / qp.HF_MANIFEST).read_text())
        check(all(qp._sha256(p) == hfm[f'validation_videos/{p.name}'][1] for p in clips),
              'every clip sha256 == HF LFS sha256 at the pinned revision')
        md5 = qp._file_md5(tsv)
        mt = {p: p.stat().st_mtime_ns for p in clips}
        D.QuantiPhy('QuantiPhy')
        check(all(p.stat().st_mtime_ns == mt[p] for p in clips) and qp._file_md5(tsv) == md5,
              'second load: nothing re-downloaded, TSV reused')

        # 2. empty / truncated clip
        clips[3].write_bytes(b'')
        clips[4].write_bytes(clips[4].read_bytes()[:1000])
        D.QuantiPhy('QuantiPhy')
        check(qp._sha256(clips[3]) == hfm[f'validation_videos/{clips[3].name}'][1]
              and qp._sha256(clips[4]) == hfm[f'validation_videos/{clips[4].name}'][1] and qp._file_md5(tsv) == md5,
              'empty clip + truncated clip -> re-downloaded, TSV md5 unchanged')

        # 3. same-size corruption, mtime kept
        corrupt_same_size(clips[5])
        D.QuantiPhy('QuantiPhy')
        survived = qp._sha256(clips[5]) != hfm[f'validation_videos/{clips[5].name}'][1]
        os.environ['QUANTIPHY_VERIFY_MEDIA'] = '1'
        D.QuantiPhy('QuantiPhy')
        del os.environ['QUANTIPHY_VERIFY_MEDIA']
        check(survived and qp._sha256(clips[5]) == hfm[f'validation_videos/{clips[5].name}'][1],
              'same-size corruption with kept mtime: passes the stat check, repaired with QUANTIPHY_VERIFY_MEDIA=1')

        # 4. question table changed
        csv = val / 'validation_dataset.csv'
        csv.write_text(csv.read_text().replace('wood block', 'WOOD BLOCK'))
        D.QuantiPhy('QuantiPhy')
        check(qp._sha256(csv) == qp.SPLIT_FILES['validation']['table_sha256'] and qp._file_md5(tsv) == md5,
              'edited question table -> re-downloaded (pinned sha256), TSV unchanged')

        # 5. TSV reuse checks
        txt = tsv.read_text()
        tsv.write_text(txt.replace('\t428\t', '\t427\t', 1))
        D.QuantiPhy('QuantiPhy')
        check(qp._file_md5(tsv) == md5, 'edited TSV (md5 mismatch) -> rebuilt')
        tsv.write_text(txt.replace('\t428\t', '\t427\t', 1))
        m = json.loads(man.read_text())
        m['tsv_md5'] = qp._file_md5(tsv)
        man.write_text(json.dumps(m))
        check(not qp.tsv_is_current(str(tsv), str(man), {k: v for k, v in
                                                          json.loads(json.dumps(_media(qp, val))).items()}),
              'TSV with matching manifest md5 but wrong n_frames column -> not current')
        D.QuantiPhy('QuantiPhy')
        check(qp._file_md5(tsv) == md5, '... and rebuilt on load (n_frames from the verified media)')
        man.unlink()
        D.QuantiPhy('QuantiPhy')
        check(man.exists() and qp._file_md5(tsv) == md5, 'missing TSV manifest -> rebuilt, same content')

        # 6. test-split transcodes
        src = real / 'QuantiPhy' / 'test'
        tsrc = lmu / 'QuantiPhy' / 'test'
        tsrc.mkdir()
        recs = {}
        thfm = json.loads((src / qp.HF_MANIFEST).read_text())
        for vid in ('simulation_0089', 'internet_0019'):
            shutil.copy2(src / f'{vid}.mp4', tsrc / f'{vid}.mp4')
            recs[vid] = (f'{vid}.mp4', thfm[f'{vid}.mp4'][1])
        dst = lmu / 'QuantiPhy' / 'test_480p'
        r1 = qp.ensure_transcodes(recs, str(tsrc), str(dst))
        import decord
        nsrc = {v: len(decord.VideoReader(str(tsrc / f'{v}.mp4'))) for v in recs}
        check(all(r1[v]['n_frames'] == nsrc[v] for v in recs), f'transcodes made, frame count == source {nsrc}')
        mt = {v: (dst / f'{v}.mp4').stat().st_mtime_ns for v in recs}
        qp.ensure_transcodes(recs, str(tsrc), str(dst))
        check(all((dst / f'{v}.mp4').stat().st_mtime_ns == mt[v] for v in recs), 'second call reuses both')
        a, b = list(recs)
        (dst / f'{a}.mp4').unlink()
        (dst / f'{b}.mp4').write_bytes((dst / f'{b}.mp4').read_bytes()[:500])
        r2 = qp.ensure_transcodes(recs, str(tsrc), str(dst))
        check(all(qp._sha256(dst / f'{v}.mp4') == r2[v]['sha256'] and r2[v]['n_frames'] == nsrc[v] for v in recs),
              'deleted + truncated transcode -> re-made')
        corrupt_same_size(dst / f'{a}.mp4')
        os.environ['QUANTIPHY_VERIFY_MEDIA'] = '1'
        r3 = qp.ensure_transcodes(recs, str(tsrc), str(dst))
        del os.environ['QUANTIPHY_VERIFY_MEDIA']
        check(qp._sha256(dst / f'{a}.mp4') == r3[a]['sha256'],
              'same-size corrupted transcode (mtime kept) -> re-made under QUANTIPHY_VERIFY_MEDIA=1')
        mt = (dst / f'{b}.mp4').stat().st_mtime_ns
        recs2 = dict(recs)
        recs2[b] = (recs[b][0], '0' * 64)
        r4 = qp.ensure_transcodes(recs2, str(tsrc), str(dst))
        check((dst / f'{b}.mp4').stat().st_mtime_ns != mt and r4[b]['src_sha256'] == '0' * 64,
              'changed source sha256 -> transcode re-made')

    if FAILS:
        print(f'\nFAIL ({len(FAILS)}):\n  ' + '\n  '.join(FAILS))
        raise SystemExit(1)
    print('\nOK: QuantiPhy cache integrity (re-download, re-transcode, TSV rebuild).')


def _media(qp, val):
    st = json.loads((val / qp.VERIFIED).read_text())
    return {rf: [r['sha256'], r['n_frames']] for rf, r in st.items() if rf.endswith('.mp4')}


if __name__ == '__main__':
    main()
