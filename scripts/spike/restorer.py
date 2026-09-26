#!/usr/bin/env python
"""SPIKE (throwaway): can a causal mel restorer close the gap?
degraded 8 kHz -> [restorer, 85 ms lookahead] -> predicted clean wideband log-mel -> pretrained Vocos -> wav.
Train: restorer.py train <shard-dir> <out-dir> [steps]    Eval: restorer.py render <ckpt> <audio-dir> <out-dir>"""
import sys, os, json, glob, time, shutil, wave, importlib.util, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchaudio
from vocos import Vocos
MOS_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mos.py")
rs = torchaudio.functional.resample
NFFT, HOP, LOWBINS = 1024, 256, 171            # Vocos' grid at 24 kHz; 171 bins = 0..4 kHz
DIM, INTER = 384, 1152
# The codec's own channel frames as a second input (env BITS=1): 72 binary features per mel frame, the mode's frame
# padded to 9 bytes. A frame-based vocoder's bits say "unvoiced, this loud" outright where its decoded audio only hints.
FRAME_BYTES = {"dstar": 9, "ysf-dmr": 7, "codec2-3200": 8, "codec2-1600": 8}
FRAME_MS = {"dstar": 20, "ysf-dmr": 20, "codec2-3200": 20, "codec2-1600": 40}
MEL_MS = HOP / 24.0                                    # 10.667 ms per mel frame at 24 kHz

def bits_from_frames(frames_u8, mode_name, n_mel):
    """[n_frames*fb (+ zero pad)] uint8 -> [72, n_mel] float: each mel frame takes the channel frame it falls in."""
    fb, ms = FRAME_BYTES[mode_name], FRAME_MS[mode_name]
    n = len(frames_u8) // fb
    fr = np.zeros((max(n, 1), 9), dtype=np.uint8)
    if n: fr[:, :fb] = np.asarray(frames_u8[: n * fb], dtype=np.uint8).reshape(n, fb)
    bits = np.unpackbits(fr, axis=1).astype(np.float32)          # [n, 72]
    k = np.minimum((np.arange(n_mel) * MEL_MS / ms).astype(int), max(n - 1, 0))
    out = bits[k].T                                              # [72, n_mel]
    if n: out[:, (np.arange(n_mel) * MEL_MS / ms) >= n] = 0.0    # past the last frame (a tail): nothing was received
    return out

class Block(nn.Module):
    def __init__(s, right):
        super().__init__(); s.right = right
        s.dw = nn.Conv1d(DIM, DIM, 7, groups=DIM); s.norm = nn.LayerNorm(DIM)
        s.pw1 = nn.Linear(DIM, INTER); s.pw2 = nn.Linear(INTER, DIM); s.gamma = nn.Parameter(torch.full((DIM,), 1e-2))
    def forward(s, x):                           # [B,C,T]
        y = s.dw(F.pad(x, (6 - s.right, s.right))).transpose(1, 2)
        y = s.pw2(F.gelu(s.pw1(s.norm(y)))) * s.gamma
        return x + y.transpose(1, 2)

class Restorer(nn.Module):
    """Lookahead: conv_in 2 + two blocks x 3 = 8 frames (85 ms); everything after is causal."""
    def __init__(s, n_modes=4, bits=False):
        super().__init__()
        s.emb = nn.Embedding(n_modes, 16)
        s.conv_in = nn.Conv1d(LOWBINS + 100 + 16, DIM, 5)
        s.bits_in = nn.Conv1d(72, DIM, 5) if bits else None     # zero-initialised: a fine-tune starts as the net without it
        if bits: nn.init.zeros_(s.bits_in.weight); nn.init.zeros_(s.bits_in.bias)
        s.blocks = nn.ModuleList([Block(3), Block(3), Block(0), Block(0), Block(0), Block(0)])
        s.gru = nn.GRU(DIM, DIM, batch_first=True)
        s.norm = nn.LayerNorm(DIM); s.out = nn.Linear(DIM, 100)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
    def forward(s, low, degmel, mode, bits=None):   # [B,171,T], [B,100,T], [B], [B,72,T]
        e = s.emb(mode)[:, :, None].expand(-1, -1, low.shape[-1])
        x = s.conv_in(F.pad(torch.cat([low, degmel, e], 1), (2, 2)))
        if s.bits_in is not None and bits is not None: x = x + s.bits_in(F.pad(bits.to(x.dtype), (2, 2)))
        for b in s.blocks: x = b(x)
        h, _ = s.gru(x.transpose(1, 2))
        return degmel + s.out(s.norm(h + x.transpose(1, 2))).transpose(1, 2)   # residual on the codec's own mel

class Feats:
    def __init__(s, dev="cpu"):
        s.dev = dev; s.voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval().to(dev); s.win = torch.hann_window(NFFT).to(dev)
        s.rs = {r: torchaudio.transforms.Resample(r, 24000).to(dev) for r in (8000, 16000)}
    @torch.no_grad()
    def __call__(s, deg, deg_rate, clean=None, clean_rate=None):
        deg = deg.to(s.dev); d24 = s.rs[deg_rate](deg)
        mag = torch.stft(d24, NFFT, HOP, window=s.win, center=True, return_complex=True).abs()[:, :LOWBINS]
        low = torch.log(mag.clamp_min(1e-5)); dm = s.voc.feature_extractor(d24)
        tm = None
        if clean is not None:
            c24 = s.rs[clean_rate](clean.to(s.dev)); tm = s.voc.feature_extractor(c24)
            n = min(tm.shape[-1], dm.shape[-1]); tm, dm, low = tm[..., :n], dm[..., :n], low[..., :n]
        return low, dm, tm

class Shards:
    def __init__(s, d, split):
        idx = json.load(open(f"{d}/index.json")); L = idx["example_layout"]
        s.nc, s.nd = L["clean16_samples"], L["deg8_samples"]; s.n_mel = None
        s.fb = L["frames"] * L["frame_bytes"]; s.size = 4 * (s.nc + s.nd) + s.fb + L["flags_bytes"] + L.get("erasure_bytes", 0)
        s.maps = [(np.memmap(f"{d}/{f['file']}", dtype=np.uint8, mode="r"), f["examples"]) for f in json.load(open(f"{d}/files.json")) if f["split"] == split]
        s.cum = np.cumsum([n for _, n in s.maps]); s.total = int(s.cum[-1]); s.modes = idx["modes"]
    def get(s, i):
        fi = int(np.searchsorted(s.cum, i, side="right")); j = i - (int(s.cum[fi - 1]) if fi else 0)
        b = s.maps[fi][0][j * s.size:(j + 1) * s.size]
        clean = np.frombuffer(b[:4 * s.nc].tobytes(), dtype="<f4"); deg = np.frombuffer(b[4 * s.nc:4 * (s.nc + s.nd)].tobytes(), dtype="<f4")
        frames = np.array(b[4 * (s.nc + s.nd):4 * (s.nc + s.nd) + s.fb], dtype=np.uint8)
        return clean, deg, int(b[4 * (s.nc + s.nd) + s.fb]) >> 2, frames
    def batch(s, n, rng):
        pick = rng.integers(0, s.total, n) if getattr(s, "index", None) is None else s.index[rng.integers(0, len(s.index), n)]
        c, d, m, fr = zip(*(s.get(int(i)) for i in pick))
        s.last_frames = fr; s.last_modes = m
        return torch.from_numpy(np.stack(c)), torch.from_numpy(np.stack(d)), torch.tensor(m)
    def bits(s, n_mel):
        """The last batch's channel bits, [B,72,n_mel]."""
        return torch.from_numpy(np.stack([bits_from_frames(f, s.modes[m], n_mel) for f, m in zip(s.last_frames, s.last_modes)]))

def train(shard_dir, out, steps):
    os.makedirs(out, exist_ok=True); torch.set_num_threads(10); torch.manual_seed(1)
    DEV = "mps" if torch.backends.mps.is_available() else "cpu"
    BITS = bool(os.environ.get("BITS"))
    tr, dv = Shards(shard_dir, "train"), Shards(shard_dir, "dev"); feats = Feats(DEV); net = Restorer(len(tr.modes), bits=BITS).to(DEV)
    LR = float(os.environ.get("LR", "3e-4"))
    if os.environ.get("INDEX"): tr.index = np.load(os.environ["INDEX"]); print("restricted to", len(tr.index), "examples", flush=True)
    if os.environ.get("INIT"): print("init from", os.environ["INIT"], net.load_state_dict(torch.load(os.environ["INIT"], map_location="cpu")["net"], strict=False), flush=True)
    print("params", sum(p.numel() for p in net.parameters()), "train", tr.total, "dev", dv.total, flush=True)
    opt = torch.optim.AdamW(net.parameters(), LR, betas=(0.9, 0.98), weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.03)
    rng = np.random.default_rng(1); drng = np.random.default_rng(7)
    dev = []
    for _ in range(4):
        c, d, m = dv.batch(32, drng); low, dm, tm = feats(d, 8000, c, 16000); dev.append((low, dm, tm, m.to(DEV), dv.bits(dm.shape[-1]).to(DEV) if BITS else None))
    t0 = time.time()
    for step in range(1, steps + 1):
        c, d, m = tr.batch(24, rng); low, dm, tm = feats(d, 8000, c, 16000)
        pred = net(low, dm, m.to(DEV), tr.bits(dm.shape[-1]).to(DEV) if BITS else None); loss = (pred - tm).abs().mean()
        if os.environ.get("POWW"):    # mean-seeking term: L2 on compressed magnitude (mag^0.6 = power^0.3). Log-domain L1 finds the
            pw = float(os.environ["POWW"])   # median of log-energy, which for "a loud hiss or nothing" is far below the mean power.
            loss = loss + pw * ((torch.exp(0.6 * pred.clamp(-16, 6)) - torch.exp(0.6 * tm.clamp(-16, 6))) ** 2).mean()
        if os.environ.get("BANDW"):   # band-energy term: L1's cautious middle makes an "s" 7-17 dB too quiet; under-prediction costs double
            bw = float(os.environ["BANDW"])
            for lo_, hi_ in ((0, 28), (28, 64), (64, 100)):
                ep, et = torch.logsumexp(pred[:, lo_:hi_], 1), torch.logsumexp(tm[:, lo_:hi_], 1); d = ep - et
                live = (et > -9).float()                     # frames where the clean band is actually active
                loss = loss + bw * ((torch.where(d < 0, 2.0 * d * d, d * d)) * live).sum() / live.sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if step % 100 == 0: print(f"step {step} l1 {loss.item():.4f} {(time.time() - t0) / step:.2f} s/step", flush=True)
        if step % 2000 == 0 or step == steps:
            net.eval()
            with torch.no_grad():
                dl = float(np.mean([(net(l, dmm, mm, bb) - t).abs().mean().item() for l, dmm, t, mm, bb in dev]))
                base = float(np.mean([(dmm - t).abs().mean().item() for l, dmm, t, mm, bb in dev]))
            net.train(); print(f"DEV step {step} l1 {dl:.4f} (codec's own mel: {base:.4f})", flush=True)
            torch.save({"net": {k: v.cpu() for k, v in net.state_dict().items()}, "modes": tr.modes, "step": step, "bits": BITS}, f"{out}/step-{step:06d}.pt")
    print("TRAIN_DONE", flush=True)

def eval_bits(name, mode, n_mel, root=os.environ.get("UNAMBLIFY_DATA", "/Volumes/data/training_data/unamblify")):
    """Channel bits for an eval clip <key with / as _>@<mode>: the captured .ambe of that key; zeros past its end (the tail)."""
    keys = [l.strip() for l in open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "eval-clips.txt")) if l.strip() and not l.startswith("#")]
    key = next((k for k in keys if k.replace("/", "_") == name), None)
    if key is None: raise KeyError(f"{name}: not an eval clip")
    return torch.from_numpy(bits_from_frames(np.fromfile(f"{root}/captured/{mode}/{key}.ambe", dtype=np.uint8), mode, n_mel))[None]

_LAGS = {}
def eval_input(name, mode, root=os.environ.get("UNAMBLIFY_DATA", "/Volumes/data/training_data/unamblify")):
    """The faithful input for an eval clip: the captured 8 kHz decode of its key, the mode's canary lag undone, and the
    prepared 16 kHz clean cut to match. The trainer's <clip>.degraded.wav is that decode upsampled to 16 kHz through a
    half-band filter; resampled back to 8 kHz it is 10-30 dB down in the top 300 Hz of the band, which is where a
    narrowband codec keeps its fricative evidence, and the restorer read that as "no s" (experiment log #38)."""
    import soundfile as sf
    keys = [l.strip() for l in open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "eval-clips.txt")) if l.strip() and not l.startswith("#")]
    key = next((k for k in keys if k.replace("/", "_") == name), None)
    if key is None: raise KeyError(f"{name}: not an eval clip")
    if mode not in _LAGS: _LAGS[mode] = json.load(open(f"{root}/captured/{mode}/canary.json"))["lag_samples"]
    first = lambda *p: next(f for f in p if os.path.exists(f))
    y, r = sf.read(first(f"{root}/captured/{mode}/{key}.flac", f"{root}/captured/{mode}/{key}.wav"), dtype="float32"); assert r == 8000
    c, rc = sf.read(first(f"{root}/prepared/{key}.16k.flac", f"{root}/prepared/{key}.16k.wav"), dtype="float32"); assert rc == 16000
    x8 = y[_LAGS[mode]:]; n = min(len(x8), len(c) // 2)
    return np.ascontiguousarray(x8[:n]), np.ascontiguousarray(c[:2 * n])

def load_restorer(ckpt):
    ck = torch.load(ckpt, map_location="cpu"); net = Restorer(len(ck["modes"]), bits=ck.get("bits", False)); net.load_state_dict(ck["net"]); net.eval(); return ck, net

def render(ckpt, src, out):
    spec = importlib.util.spec_from_file_location("mos", MOS_PY); mos = importlib.util.module_from_spec(spec); spec.loader.exec_module(mos)
    ck, net = load_restorer(ckpt); feats = Feats(); os.makedirs(out, exist_ok=True)
    with torch.no_grad():
        for p in sorted(glob.glob(f"{src}/*.degraded.wav")):
            stem = p[:-len(".degraded.wav")]; name = os.path.basename(stem); mode = name.rsplit("@", 1)[1]
            if os.environ.get("FAITHFUL", "1") == "1":
                x8n, c16 = eval_input(name.rsplit("@", 1)[0], mode); x8 = torch.from_numpy(x8n)[None]
                sf_write = __import__("soundfile").write; sf_write(f"{out}/{name}.clean.wav", c16, 16000, subtype="PCM_16")
                sf_write(f"{out}/{name}.degraded.wav", rs(x8, 8000, 16000)[0].numpy(), 16000, subtype="PCM_16")
            else:
                x, r = mos.read_wav(p); x = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
                x8 = rs(x, r, 8000)                                   # the model hears narrowband only
            low, dm, _ = feats(x8, 8000); bb = eval_bits(name.rsplit("@", 1)[0], mode, dm.shape[-1]) if ck.get("bits") else None
            pred = net(low, dm, torch.tensor([ck["modes"].index(mode)]), bb)
            y = rs(feats.voc.decode(pred), 24000, 16000)[0].numpy()
            if os.environ.get("FAITHFUL", "1") != "1":
                for k in ("clean", "degraded"): shutil.copy(f"{stem}.{k}.wav", f"{out}/{name}.{k}.wav")
            pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
            with wave.open(f"{out}/{name}.out.wav", "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
    print("RENDER_DONE")

if __name__ == "__main__":
    if sys.argv[1] == "train": train(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 30000)
    else: render(sys.argv[2], sys.argv[3], sys.argv[4])
