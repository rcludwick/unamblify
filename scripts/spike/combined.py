#!/usr/bin/env python
"""SPIKE: the whole streaming-shaped pipeline. codec output -> restorer -> mel -> CAUSAL vocoder -> wav.
combined.py <restorer-ckpt> <vocoder-ckpt|pretrained> <audio-dir> <out-dir>"""
import sys, os, glob, shutil, importlib.util
import numpy as np, torch, torchaudio, soundfile as sf
here = os.path.dirname(os.path.abspath(__file__))
def mod(n):
    sp = importlib.util.spec_from_file_location(n, f"{here}/{n}.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
R, V = mod("restorer"), mod("causal_vocoder"); rs = torchaudio.functional.resample
ck, net = R.load_restorer(sys.argv[1])
voc = V.load(sys.argv[2]); feats = R.Feats(); src, out = sys.argv[3], sys.argv[4]; os.makedirs(out, exist_ok=True)
with torch.no_grad():
    for p in sorted(glob.glob(f"{src}/*.degraded.wav")):
        stem = p[:-len(".degraded.wav")]; name = os.path.basename(stem); mode = name.rsplit("@", 1)[1]
        if os.environ.get("FAITHFUL", "1") == "1":     # the captured decode by key, not the trainer's 16 kHz rendering of it
            x8n, c16 = R.eval_input(name.rsplit("@", 1)[0], mode); x8 = torch.from_numpy(x8n)[None]
            sf.write(f"{out}/{name}.clean.wav", c16, 16000, subtype="PCM_16"); sf.write(f"{out}/{name}.degraded.wav", rs(x8, 8000, 16000)[0].numpy(), 16000, subtype="PCM_16")
        else:
            x, r = sf.read(p, dtype="float32"); x8 = rs(torch.from_numpy(x)[None], r, 8000)
            for k in ("clean", "degraded"): shutil.copy(f"{stem}.{k}.wav", f"{out}/{name}.{k}.wav")
        low, dm, _ = feats(x8, 8000); bb = R.eval_bits(name.rsplit("@", 1)[0], mode, dm.shape[-1]) if ck.get("bits") else None
        pred = net(low, dm, torch.tensor([ck["modes"].index(mode)]), bb)
        y = rs(voc.decode(pred), 24000, 16000)[0].numpy()
        sf.write(f"{out}/{name}.out.wav", np.clip(y, -1, 1), 16000, subtype="PCM_16")
print("RENDER_DONE")
