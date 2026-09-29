#!/usr/bin/env python
# unamblify — Copyright (c) 2026 Rob Ludwick.
# SPDX-License-Identifier: AGPL-3.0-only
"""Predicted mean opinion score of a checkpoint's eval clips.

Every metric the trainer writes is a hand-built proxy, and two of them
have now read "solved" while the ear said otherwise. This scores the
rendered clips with UTMOS22 (Saeki et al., the strong learner from the
VoiceMOS 2022 challenge; MIT, via the SpeechMOS torch.hub package), a
model trained on human naturalness ratings, and reports the mean per
signal kind and per mode:

    scripts/mos.py $UNAMBLIFY_DATA/runs/<run>/checkpoints/step-N/audio
    scripts/mos.py <audio dir> --json          # machine-readable
    scripts/mos.py <audio dir> --kinds out     # only the model output

Needs the project's torch venv (`just train-env`); the first call fetches
the weights (~100 MB) into torch's hub cache. Scores are 1–5; clean
studio speech scores ~4, a raw AMBE decode ~2.5–3. Like every learned
predictor it is a proxy too — but one trained on the thing we want,
rather than on a statistic we hope tracks it.
"""

import argparse
import collections
import json
import pathlib
import sys
import wave

import numpy as np

KINDS = ("clean", "degraded", "out")


def read_wav(path: pathlib.Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return x.astype(np.float32) / 32768.0, w.getframerate()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("audio_dir", type=pathlib.Path)
    ap.add_argument("--kinds", default=",".join(KINDS), help="comma-separated: clean,degraded,out")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch  # noqa: E402  (after argparse so --help needs no torch)

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor = predictor.to(args.device).eval()

    kinds = args.kinds.split(",")
    scores = collections.defaultdict(list)  # (kind, mode) -> [mos]
    per_clip = {}
    for path in sorted(args.audio_dir.glob("*.wav")):
        stem, kind = path.name[: -len(".wav")].rsplit(".", 1)
        if kind not in kinds:
            continue
        mode = stem.rsplit("@", 1)[1] if "@" in stem else "-"
        x, rate = read_wav(path)
        with torch.no_grad():
            mos = float(predictor(torch.from_numpy(x)[None].to(args.device), rate).item())
        scores[(kind, mode)].append(mos)
        per_clip[f"{stem}.{kind}"] = mos

    if not scores:
        print(f"no *.{{{args.kinds}}}.wav under {args.audio_dir}", file=sys.stderr)
        return 1
    modes = sorted({m for _, m in scores})
    summary = {
        kind: {
            "all": float(np.mean([v for (k, m), vs in scores.items() if k == kind for v in vs])),
            **{m: float(np.mean(scores[(kind, m)])) for m in modes if (kind, m) in scores},
        }
        for kind in kinds
        if any(k == kind for k, _ in scores)
    }
    if args.json:
        json.dump({"summary": summary, "clips": per_clip}, sys.stdout, indent=1)
        print()
        return 0
    width = max(len(m) for m in modes + ["all"])
    print(f"{'kind':10s} " + " ".join(f"{m:>{width}s}" for m in ["all"] + modes))
    for kind, row in summary.items():
        print(f"{kind:10s} " + " ".join(f"{row.get(m, float('nan')):{width}.2f}" for m in ["all"] + modes))
    n = sum(len(v) for v in scores.values())
    print(f"({n} clips, UTMOS22 strong; 1–5, higher is more natural)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
