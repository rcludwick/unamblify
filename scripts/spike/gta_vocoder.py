#!/usr/bin/env python
"""SPIKE (throwaway): teach the vocoder to make natural speech from the restorer's BLURRED mels.
The TTS recipe: fine-tune the vocoder on predicted spectra with waveform discriminators.
  gta_vocoder.py <restorer-ckpt> <vocoder: pretrained|ckpt> <out-dir> <deadline-epoch> [index.npy]"""
import sys, os, time, shutil, importlib.util
import numpy as np, torch, torchaudio
here = os.path.dirname(os.path.abspath(__file__))
def mod(n):
    sp = importlib.util.spec_from_file_location(n, f"{here}/{n}.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
R, V = mod("restorer"), mod("causal_vocoder")
from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss
ROOT = os.environ.get("UNAMBLIFY_DATA", "/Volumes/data/training_data/unamblify")
SHARDS = os.environ.get("SHARDS", ROOT + "/shards/mixed-50k-v4"); SEG = 64   # frames: 0.68 s at 24 kHz

def main(rck, vck, out, deadline, index=None):
    os.makedirs(out, exist_ok=True); torch.manual_seed(5); DEV = "mps"
    tr = R.Shards(SHARDS, "train")
    if index: tr.index = np.load(index)
    ck = torch.load(rck, map_location="cpu"); net = R.Restorer(len(tr.modes)); net.load_state_dict(ck["net"]); net.to(DEV).eval()
    G = V.load(vck).to(DEV).train(); causal = vck != "pretrained" and torch.load(vck, map_location="cpu").get("causal", True)
    feats = R.Feats(DEV); mpd, mrd = MultiPeriodDiscriminator().to(DEV), MultiResolutionDiscriminator().to(DEV)
    dl, gl, fml = DiscriminatorLoss(), GeneratorLoss(), FeatureMatchingLoss(); mell = MelSpecReconstructionLoss(sample_rate=24000).to(DEV)
    og = torch.optim.AdamW(list(G.backbone.parameters()) + list(G.head.parameters()), 5e-5, betas=(0.8, 0.9))
    od = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), 2e-4, betas=(0.8, 0.9))
    WARM = 1500
    if vck != "pretrained":
        prev = torch.load(vck, map_location="cpu")
        if "mpd" in prev: mpd.load_state_dict(prev["mpd"]); mrd.load_state_dict(prev["mrd"]); WARM = 0
    rs16 = torchaudio.transforms.Resample(16000, 24000).to(DEV); rng = np.random.default_rng(int(time.time()) % 100000); t0 = time.time(); step = 0
    print("restorer", rck, "vocoder", vck, "causal", causal, "examples", len(tr.index) if index else tr.total, flush=True)
    while True:
        step += 1
        c, d, m = tr.batch(16, rng)
        with torch.no_grad():
            low, dm, _ = feats(d, 8000); pred = net(low, dm, m.to(DEV)); c24 = rs16(c.to(DEV))
            a = int(rng.integers(2, pred.shape[-1] - SEG - 2)); mel = pred[..., a:a + SEG].clone()
        yh = G.head(G.backbone(mel)); y = c24[..., a * 256:a * 256 + yh.shape[-1]]; yh = yh[..., :y.shape[-1]]
        rm, gm, _, _ = mpd(y=y, y_hat=yh.detach()); rr, gr, _, _ = mrd(y=y, y_hat=yh.detach())
        ld = dl(rm, gm)[0] / len(rm) + dl(rr, gr)[0] / len(rr); od.zero_grad(); ld.backward(); od.step()
        lm = mell(yh, y)
        if step > WARM:
            _, gm, fr, fg = mpd(y=y, y_hat=yh); _, gr, fr2, fg2 = mrd(y=y, y_hat=yh)
            lg = gl(gm)[0] / len(gm) + gl(gr)[0] / len(gr) + fml(fr, fg) / len(fr) + fml(fr2, fg2) / len(fr2) + 45 * lm
        else: lg = 45 * lm
        og.zero_grad(); lg.backward(); og.step()
        if step % 100 == 0: print(f"step {step} mel {lm.item():.4f} d {ld.item():.3f} {(time.time() - t0) / step:.2f} s/step", flush=True)
        late = time.time() > deadline
        if step % 2500 == 0 or late:
            torch.save({"backbone": {k: v.cpu() for k, v in G.backbone.state_dict().items()}, "head": {k: v.cpu() for k, v in G.head.state_dict().items()}, "step": step, "causal": bool(causal), "mpd": {k: v.cpu() for k, v in mpd.state_dict().items()}, "mrd": {k: v.cpu() for k, v in mrd.state_dict().items()}}, f"{out}/step-{step:06d}.pt")
            shutil.copy(f"{out}/step-{step:06d}.pt", f"{out}/last.pt")
        if late: break
    print("TRAIN_DONE", flush=True)

if __name__ == "__main__": main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else None)
