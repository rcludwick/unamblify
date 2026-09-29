#!/usr/bin/env python
"""How much future does the whole pipeline actually use? Cut the input off L ms after time T and see whether
the output BEFORE T changes. The smallest L with no change is the pipeline's true lookahead = its streaming latency."""
import sys, os, glob, importlib.util, numpy as np, torch, torchaudio, soundfile as sf
here = os.path.dirname(os.path.abspath(__file__))
def mod(n):
    sp = importlib.util.spec_from_file_location(n, f"{here}/{n}.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
R, V = mod("restorer"), mod("causal_vocoder"); rs = torchaudio.functional.resample
ck = torch.load(sys.argv[1], map_location="cpu"); net = R.Restorer(len(ck["modes"])); net.load_state_dict(ck["net"]); net.eval()
voc = V.load(sys.argv[2]); feats = R.Feats()
def run(x8, mode):
    with torch.no_grad():
        low, dm, _ = feats(x8, 8000); return voc.decode(net(low, dm, torch.tensor([mode])))[0].numpy()   # 24 kHz
p = sorted(glob.glob(sys.argv[3] + "/*@dstar.degraded.wav"))[0]; x, r = sf.read(p, dtype="float32"); x8 = rs(torch.from_numpy(x)[None], r, 8000)
mode = ck["modes"].index("dstar"); full = run(x8, mode); T = 2.0                       # seconds into the clip
ref = full[int((T - 0.5) * 24000):int(T * 24000)]; scale = np.abs(ref).max()
print(f"clip {os.path.basename(p)}; comparing the 0.5 s of output before T = {T} s; peak {scale:.3f}")
for L in (100, 200, 300, 350, 380, 400, 420, 450, 500):
    cut = run(x8[:, :int((T + L / 1000) * 8000)], mode)[int((T - 0.5) * 24000):int(T * 24000)]
    err = np.abs(cut - ref).max() / scale
    print(f"  future allowed {L:4d} ms -> max deviation {20 * np.log10(err + 1e-12):7.1f} dB re peak")
