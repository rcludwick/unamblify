#!/usr/bin/env python
"""Where does the restorer under-predict the high band? Mel-domain, no synthesiser: predicted minus clean
4-8 kHz band energy (dB), binned by how loud the CLEAN high band is in that frame."""
import sys, os, glob, importlib.util, numpy as np, torch, torchaudio, soundfile as sf
here = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location("restorer", f"{here}/restorer.py"); R = importlib.util.module_from_spec(sp); sp.loader.exec_module(R)
rs = torchaudio.functional.resample; ck = torch.load(sys.argv[1], map_location="cpu"); net = R.Restorer(len(ck["modes"])); net.load_state_dict(ck["net"]); net.eval(); feats = R.Feats()
db = lambda m: 10 / np.log(10) * torch.logsumexp(2 * m[:, 64:86], 1)[0].numpy()      # mel is log-magnitude: x2 for power
lowdb = lambda m: 10 / np.log(10) * torch.logsumexp(2 * m[:, 28:64], 1)[0].numpy()
print(f"{'mode':12s} {'clean-HF percentile':>20s}   pred-clean HF dB   codec 1-4k minus clean 1-4k dB   frames")
for mode in ("dstar", "ysf-dmr", "codec2-3200", "codec2-1600"):
    P_, C_, DL, CL = [], [], [], []
    for p in sorted(glob.glob(f"{sys.argv[2]}/*@{mode}.degraded.wav")):
        x, r = sf.read(p, dtype="float32"); c, rc = sf.read(p.replace(".degraded.", ".clean."), dtype="float32")
        with torch.no_grad():
            low, dm, tm = feats(rs(torch.from_numpy(x)[None], r, 8000), 8000, torch.from_numpy(c)[None], rc); pred = net(low, dm, torch.tensor([ck["modes"].index(mode)]))
        P_.append(db(pred)); C_.append(db(tm)); DL.append(lowdb(dm)); CL.append(lowdb(tm))
    P_, C_, DL, CL = map(np.concatenate, (P_, C_, DL, CL)); live = C_ > -60
    for lo, hi in ((50, 75), (75, 92), (92, 100)):
        a, b = np.percentile(C_[live], lo), np.percentile(C_[live], hi); sel = live & (C_ >= a) & (C_ <= b)
        print(f"{mode:12s} {f'{lo}-{hi}':>20s}   {np.median(P_[sel] - C_[sel]):16.1f}   {np.median(DL[sel] - CL[sel]):30.1f}   {sel.sum():6d}")
