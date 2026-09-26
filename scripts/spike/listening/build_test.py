#!/usr/bin/env python
"""Build the blind listening test: each utterance x mode through every system, loudness-matched, UTMOS-scored.
build_test.py <out-dir>   with RESTORER=<restorer.pt> SYNTH=<synthesiser.pt> and <out-dir>/../listening-set.json
(a list of {key, speaker, gender, corpus}); the v6 filter run id is FILTER_RUN (default the 2026-09-21 run)."""
import sys, os, json, subprocess, hashlib, importlib.util
import numpy as np, torch, torchaudio, soundfile as sf
ROOT = os.environ.get("UNAMBLIFY_DATA", "/Volumes/data/training_data/unamblify")
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = sys.argv[1]   # the spike scripts live one level up
def mod(n):
    sp = importlib.util.spec_from_file_location(n, f"{here}/{n}.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
R, V = mod("restorer"), mod("causal_vocoder"); rs = torchaudio.functional.resample
root = ROOT; exe = os.environ.get("UNAMBLIFY_BIN", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "target", "release", "unamblify"))
V6 = f"{root}/runs/" + os.environ.get("FILTER_RUN", "20260921-043805-mixed-50k-v6-perens-ll5")
ck = torch.load(os.environ["RESTORER"], map_location="cpu"); net = R.Restorer(len(ck["modes"])); net.load_state_dict(ck["net"]); net.eval()
voc = V.load(os.environ["SYNTH"]); feats = R.Feats()
predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
def first(*p): return next(x for x in p if os.path.isfile(x))
def norm(x, target=-26.0):
    x = np.asarray(x, dtype=np.float32); rms = np.sqrt(np.mean(x ** 2) + 1e-12); x = x * (10 ** (target / 20) / rms)
    return x / max(1.0, np.abs(x).max() / 0.99)
clips = []
for u in json.load(open(os.environ.get("LISTENING_SET", os.path.join(OUT, "listening-set.json")))):
    key = u["key"]; clean, r = sf.read(first(f"{root}/prepared/{key}.16k.flac", f"{root}/prepared/{key}.16k.wav"), dtype="float32")
    for mode in ("dstar", "ysf-dmr"):
        deg, dr = sf.read(first(f"{root}/captured/{mode}/{key}.flac", f"{root}/captured/{mode}/{key}.wav"), dtype="float32")
        tmp = f"{OUT}/tmp.wav"
        rr = subprocess.run([exe, "infer", "--run-dir", V6, "--step", "72000", "--key", key, "--mode", mode, "--out", tmp, "--data-root", root], capture_output=True, text=True)
        assert rr.returncode == 0, rr.stderr[-300:]
        filt, fr = sf.read(tmp, dtype="float32")
        with torch.no_grad():
            low, dm, _ = feats(torch.from_numpy(deg)[None], dr); new = rs(voc.decode(net(low, dm, torch.tensor([ck["modes"].index(mode)]))), 24000, 16000)[0].numpy()
        systems = {"clean": (clean, r), "codec": (rs(torch.from_numpy(deg)[None], dr, 16000)[0].numpy(), 16000), "filter": (filt, fr), "restorer": (new, 16000)}
        for name, (x, rate) in systems.items():
            if rate != 16000: x = rs(torch.from_numpy(np.asarray(x, dtype=np.float32))[None], rate, 16000)[0].numpy()
            x = norm(x); cid = hashlib.sha1(f"{key}|{mode}|{name}".encode()).hexdigest()[:10]
            sf.write(f"{OUT}/audio/{cid}.wav", x, 16000, subtype="PCM_16")
            with torch.no_grad(): p = float(predictor(torch.from_numpy(x)[None], 16000).item())
            clips.append({"id": cid, "key": key, "speaker": u["speaker"], "gender": u["gender"], "corpus": u["corpus"], "mode": mode, "system": name, "predicted": round(p, 3)})
        print(key, mode, "ok", flush=True)
for f in (f"{OUT}/tmp.wav", f"{OUT}/tmp.spec.json"):
    if os.path.exists(f): os.remove(f)
json.dump(clips, open(f"{OUT}/clips.json", "w"), indent=1); print("BUILD_DONE", len(clips))
