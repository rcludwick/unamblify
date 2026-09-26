#!/usr/bin/env python
"""SPIKE (throwaway): can the vocoder half stream? Take pretrained Vocos (288 ms of lookahead),
make every conv causal but for LOOK frames, fine-tune adversarially on clean studio speech only.
  causal_vocoder.py train <out-dir> [steps]      causal_vocoder.py render <ckpt|pretrained|causal-untrained> <audio-dir> <out-dir>"""
import sys, os, json, glob, time, shutil, wave, threading, queue
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchaudio, soundfile as sf
from vocos import Vocos
ROOT = os.environ.get("UNAMBLIFY_DATA", "/Volumes/data/training_data/unamblify"); STUDIO = {"libritts_r", "vctk", "ljspeech", "voicebank_demand"}
RIGHT = [1, 1, 1, 0, 0, 0, 0, 0, 0]          # embed conv + 8 blocks: 3 frames (32 ms) of lookahead; the iSTFT adds 2 more

def make_causal(voc):
    convs = [voc.backbone.embed] + [b.dwconv for b in voc.backbone.convnext]
    for conv, r in zip(convs, RIGHT):
        k = conv.kernel_size[0]; conv.padding = (0,)
        conv.register_forward_pre_hook(lambda m, a, k=k, r=r: (F.pad(a[0], (k - 1 - r, r)),))
    return voc

def files():
    out = []
    with open(f"{ROOT}/prepared/manifest.jsonl") as f:
        for line in f:
            if '"split":"train"' not in line or '"parent":"' in line: continue
            r = json.loads(line)
            if r["corpus"] in STUDIO and r["duration_s"] >= 1.2: out.append(r["key"])
    return out

def loader(keys, batch, n=16000, seed=1):
    q = queue.Queue(maxsize=8)
    def work(s):
        rng = np.random.default_rng(s)
        while True:
            xs = []
            while len(xs) < batch:
                k = keys[int(rng.integers(len(keys)))]
                p = next((f"{ROOT}/prepared/{k}.16k.{e}" for e in ("flac", "wav") if os.path.isfile(f"{ROOT}/prepared/{k}.16k.{e}")), None)
                if not p: continue
                x, r = sf.read(p, dtype="float32")
                if len(x) <= n: continue
                o = int(rng.integers(len(x) - n)); xs.append(x[o:o + n])
            q.put(torch.from_numpy(np.stack(xs)))
    for i in range(4): threading.Thread(target=work, args=(seed + i,), daemon=True).start()
    while True: yield q.get()

def train(out, steps, deadline=None):
    from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
    from vocos.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss
    os.makedirs(out, exist_ok=True); torch.manual_seed(1); DEV = "mps"
    G = make_causal(Vocos.from_pretrained("charactr/vocos-mel-24khz")).to(DEV).train()
    mpd, mrd = MultiPeriodDiscriminator().to(DEV), MultiResolutionDiscriminator().to(DEV)
    dl, gl, fml = DiscriminatorLoss(), GeneratorLoss(), FeatureMatchingLoss(); mell = MelSpecReconstructionLoss(sample_rate=24000).to(DEV)
    og = torch.optim.AdamW(list(G.backbone.parameters()) + list(G.head.parameters()), 5e-5, betas=(0.8, 0.9))
    od = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), 2e-4, betas=(0.8, 0.9))
    rs = torchaudio.transforms.Resample(16000, 24000).to(DEV); keys = files(); print("studio train utterances", len(keys), flush=True)
    it = loader(keys, 16, n=12000); t0 = time.time(); WARM = 1500     # discriminators alone first: they start from nothing, G does not
    for step in range(1, steps + 1):
        y = rs(next(it).to(DEV))
        with torch.no_grad(): mel = G.feature_extractor(y)
        yh = G.head(G.backbone(mel.clone()))[..., :y.shape[-1]]; y = y[..., :yh.shape[-1]]   # decode() is inference_mode
        rm, gm, _, _ = mpd(y=y, y_hat=yh.detach()); rr, gr, _, _ = mrd(y=y, y_hat=yh.detach())
        ld = dl(rm, gm)[0] / len(rm) + dl(rr, gr)[0] / len(rr)
        od.zero_grad(); ld.backward(); od.step()
        lm = mell(yh, y)
        if step > WARM:
            _, gm, fr, fg = mpd(y=y, y_hat=yh); _, gr, fr2, fg2 = mrd(y=y, y_hat=yh)
            lg = gl(gm)[0] / len(gm) + gl(gr)[0] / len(gr) + fml(fr, fg) / len(fr) + fml(fr2, fg2) / len(fr2) + 45 * lm
        else: lg = 45 * lm
        og.zero_grad(); lg.backward(); og.step()
        if step % 100 == 0: print(f"step {step} mel {lm.item():.4f} d {ld.item():.3f} {(time.time() - t0) / step:.2f} s/step", flush=True)
        late = deadline is not None and time.time() > deadline
        if step % 2500 == 0 or step == steps or late:
            torch.save({"backbone": {k: v.cpu() for k, v in G.backbone.state_dict().items()}, "head": {k: v.cpu() for k, v in G.head.state_dict().items()}, "step": step}, f"{out}/step-{step:06d}.pt"); shutil.copy(f"{out}/step-{step:06d}.pt", f"{out}/last.pt")
        if late: break
    print("TRAIN_DONE", flush=True)

def load(which):
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    if which == "pretrained": return voc.eval()
    ck = None if which == "causal-untrained" else torch.load(which, map_location="cpu")
    if ck is None or ck.get("causal", True): voc = make_causal(voc)
    if ck is not None: voc.backbone.load_state_dict(ck["backbone"]); voc.head.load_state_dict(ck["head"])
    return voc.eval()

def render(which, src, out):
    voc = load(which); os.makedirs(out, exist_ok=True); rs = torchaudio.functional.resample; seen = set()
    with torch.no_grad():
        for p in sorted(glob.glob(f"{src}/*.clean.wav")):
            stem = p[:-len(".clean.wav")]; name = os.path.basename(stem)
            x, r = sf.read(p, dtype="float32"); y = rs(voc.decode(voc.feature_extractor(rs(torch.from_numpy(x)[None], r, 24000))), 24000, 16000)[0].numpy()
            for k in ("clean", "degraded"): shutil.copy(f"{stem}.{k}.wav", f"{out}/{name}.{k}.wav")
            sf.write(f"{out}/{name}.out.wav", np.clip(y, -1, 1), 16000, subtype="PCM_16")
    print("RENDER_DONE")

if __name__ == "__main__":
    if sys.argv[1] == "train": train(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 30000, float(sys.argv[4]) if len(sys.argv) > 4 else None)
    else: render(sys.argv[2], sys.argv[3], sys.argv[4])
