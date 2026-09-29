#!/usr/bin/env python
"""Second decomposition: whose low band, whose high band. out = vocos(mel_low(A) ++ mel_high(B))."""
import sys, shutil, glob, os, importlib.util, wave
import numpy as np, torch, torchaudio
from vocos import Vocos
MOS_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mos.py")
spec = importlib.util.spec_from_file_location("mos", MOS_PY); mos = importlib.util.module_from_spec(spec); spec.loader.exec_module(mos)
def write16(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
src, out = sys.argv[1], sys.argv[2]
voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval(); rs = torchaudio.functional.resample
cut = 64
def mel(p):
    x, r = mos.read_wav(p); x = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]; return voc.feature_extractor(rs(x, r, 24000))
conds = {"model-low+clean-high": ("out", "clean"), "clean-low+model-high": ("clean", "out"), "clean-low+degraded-high": ("clean", "degraded"), "model-low+model-high": ("out", "out"), "clean-low+clean-high": ("clean", "clean")}
for s in conds: os.makedirs(f"{out}/{s}", exist_ok=True)
stems = sorted(f[:-len(".degraded.wav")] for f in glob.glob(f"{src}/*.degraded.wav"))
with torch.no_grad():
    for st in stems:
        name = os.path.basename(st); m = {k: mel(f"{st}.{k}.wav") for k in ("clean", "degraded", "out")}
        n = min(v.shape[-1] for v in m.values())
        for s, (lo, hi) in conds.items():
            h = m[hi][..., :n].clone(); h[:, :cut, :] = m[lo][:, :cut, :n]
            for k in ("clean", "degraded"):
                dst = f"{out}/{s}/{name}.{k}.wav"
                if not os.path.exists(dst): shutil.copy(f"{st}.{k}.wav", dst)
            write16(f"{out}/{s}/{name}.out.wav", rs(voc.decode(h), 24000, 16000)[0].numpy())
print("PROBE2_DONE")
