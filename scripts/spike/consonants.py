#!/usr/bin/env python
"""Plosives and sibilants, measured against the clean recording at the moments the CLEAN one has them.
burst:    clean envelope jumps >= 18 dB inside 10 ms after >= 40 ms of quiet; report output peak - clean peak (dB) in the next 25 ms,
          and closure = how much louder the output is than clean in the 40 ms before (dB; positive = filled in).
sibilant: the 8 % of frames with the most clean 4-8 kHz energy; report output - clean level there in 4-8 kHz (dB), and the
          4-8k / 1-4k balance error (dB): negative = dull."""
import sys, glob, os, numpy as np, soundfile as sf
from scipy.signal import stft
def env(x, n=80):  # 5 ms RMS, hop 1 ms at 16 kHz
    e = np.sqrt(np.convolve(x * x, np.ones(n) / n, "same") + 1e-12); return 20 * np.log10(e[::16] + 1e-9)
def band(x, lo, hi):
    f, t, Z = stft(x, 16000, nperseg=512, noverlap=352); m = (f >= lo) & (f < hi); return 10 * np.log10((np.abs(Z[m]) ** 2).sum(0) + 1e-12)
def measure(clean, out):
    n = min(len(clean), len(out)); clean, out = clean[:n], out[:n]; ec, eo = env(clean), env(out); b, c = [], []
    for t in range(45, len(ec) - 30):
        if ec[t + 10] - ec[t] >= 18 and ec[t - 40:t].max() < ec[t + 10] - 15 and ec[t + 10] > -45 and (not b or t - last > 80):
            last = t; b.append(eo[t - 5:t + 30].max() - ec[t:t + 25].max()); c.append(eo[t - 40:t - 3].mean() - ec[t - 40:t - 3].mean())
    hc, ho, lc, lo_ = band(clean, 4000, 8000), band(out, 4000, 8000), band(clean, 1000, 4000), band(out, 1000, 4000)
    k = min(len(hc), len(ho)); hc, ho, lc, lo_ = hc[:k], ho[:k], lc[:k], lo_[:k]
    sel = (hc >= np.quantile(hc, 0.92)) & (hc > -60)
    return b, c, list(ho[sel] - hc[sel]), list((ho[sel] - lo_[sel]) - (hc[sel] - lc[sel]))
A = sys.argv[1]; systems = [a.split("=", 1) for a in sys.argv[2:]]
print(f"{'system':34s} {'mode':12s} bursts  burst dB  closure dB   s-level dB  s-balance dB")
for label, (d, kind) in [(l, v.rsplit(":", 1)) for l, v in systems]:
    for mode in ("dstar", "ysf-dmr", "codec2-3200"):
        B, C, S, T = [], [], [], []
        for p in sorted(glob.glob(f"{A}/*@{mode}.clean.wav")):
            o = f"{d}/{os.path.basename(p)[:-len('.clean.wav')]}.{kind}.wav"
            if not os.path.isfile(o): continue
            x, _ = sf.read(p, dtype="float32"); y, _ = sf.read(o, dtype="float32"); b, c, s, t = measure(x, y); B += b; C += c; S += s; T += t
        if B: print(f"{label:34s} {mode:12s} {len(B):5d}  {np.median(B):8.1f}  {np.median(C):10.1f}   {np.median(S):10.1f}  {np.median(T):12.1f}")
