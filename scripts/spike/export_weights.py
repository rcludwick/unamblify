#!/usr/bin/env python
"""Export the spike pipeline for the Rust runtime: one safetensors file + a JSON manifest + PyTorch references.

  export_weights.py <restorer.pt> <synth.pt> <out-dir>

Writes <out-dir>/pipeline.safetensors (every tensor the runtime needs, incl. the mel filterbank and the torchaudio
resampling kernel), <out-dir>/pipeline.json (shapes, pads, constants, mode order) and, for one D-STAR and one Codec 2
3200 eval clip on the faithful input, <out-dir>/references.safetensors holding <mode>.{x8,low,degmel,mel,wav24} so the
Rust side can be checked against PyTorch. Needs the .venv-mos environment (torch, torchaudio, vocos, safetensors)."""
import sys, os, json, importlib.util
import numpy as np, torch, torchaudio
from safetensors.torch import save_file
from vocos import Vocos
here = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location("restorer", f"{here}/restorer.py"); R = importlib.util.module_from_spec(sp); sp.loader.exec_module(R)

rck, sck, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
ck, net = R.load_restorer(rck)
if ck.get("bits"): raise SystemExit("a bits restorer is not exported (the runtime has no channel-bit input)")
voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()
s = torch.load(sck, map_location="cpu"); voc.backbone.load_state_dict(s["backbone"]); voc.head.load_state_dict(s["head"])
if s.get("causal"): raise SystemExit("a causal-converted synthesiser is not exported")

tensors = {}
for k, v in net.state_dict().items(): tensors[f"restorer.{k}"] = v.detach().float().contiguous()
for k, v in voc.backbone.state_dict().items(): tensors[f"synth.backbone.{k}"] = v.detach().float().contiguous()
for k, v in voc.head.state_dict().items(): tensors[f"synth.head.{k}"] = v.detach().float().contiguous()
fb = voc.feature_extractor.mel_spec.mel_scale.fb            # [513, 100]
tensors["features.mel_fbank"] = fb.detach().float().contiguous()
rs8 = torchaudio.transforms.Resample(8000, 24000)                # the spike's Feats: kernel built in float64, cast
kernel, width = rs8.kernel, rs8.width
tensors["features.resample_kernel"] = kernel.detach().float().contiguous()   # [3, 1, K]
tensors["features.window"] = torch.hann_window(1024).float()                  # the spike's low band: torch.stft's periodic Hann
tensors["features.mel_window"] = voc.feature_extractor.mel_spec.spectrogram.window.detach().float().contiguous()
# ^ Vocos's checkpoint carries its own window (one ulp from torch.hann_window); the mel is taken under it
save_file(tensors, f"{out}/pipeline.safetensors")

manifest = {
    "format": 1, "restorer_checkpoint": os.path.abspath(rck), "synth_checkpoint": os.path.abspath(sck),
    "modes": ck["modes"],
    "sample_rate": 24000, "input_rate": 8000, "n_fft": 1024, "hop": 256, "n_mels": 100, "low_bins": R.LOWBINS,
    "log_clamp": 1e-5, "mel_clamp": 1e-7, "mel_fmin": 0.0, "mel_fmax": 12000.0, "mel_scale": "htk", "mel_norm": None,
    "resample": {"orig_freq": 1, "new_freq": 3, "width": int(width), "lowpass_filter_width": 6, "rolloff": 0.99},
    "restorer": {"dim": R.DIM, "inter": R.INTER, "embed_dim": 16, "conv_in": {"kernel": 5, "left": 2, "right": 2},
                 "blocks": [{"kernel": 7, "left": 6 - b.right, "right": b.right} for b in net.blocks],
                 "ln_eps": 1e-5, "lookahead_frames": 2 + sum(b.right for b in net.blocks)},
    "synth": {"dim": 512, "inter": 1536, "layers": 8, "embed": {"kernel": 7, "left": 3, "right": 3},
              "block": {"kernel": 7, "left": 3, "right": 3}, "ln_eps": 1e-6, "mag_clip": 100.0,
              "istft": {"center": True, "n_fft": 1024, "hop": 256, "win_length": 1024}, "lookahead_frames": 3 + 8 * 3},
    "shapes": {k: list(v.shape) for k, v in tensors.items()},
}
json.dump(manifest, open(f"{out}/pipeline.json", "w"), indent=1)

# References on the faithful input: one safetensors file (tch reads it; its npy reader does not read numpy 2's headers).
refs = {}
keys = [l.strip() for l in open(os.path.join(os.path.dirname(os.path.dirname(here)), "configs", "eval-clips.txt")) if l.strip() and not l.startswith("#")]
feats = R.Feats()
for mode, key in (("dstar", keys[0]), ("codec2-3200", keys[0])):
    x8, c16 = R.eval_input(key.replace("/", "_"), mode)
    with torch.no_grad():
        low, dm, _ = feats(torch.from_numpy(x8)[None], 8000)
        mel = net(low, dm, torch.tensor([ck["modes"].index(mode)]))
        wav = voc.decode(mel)
    for name, t in (("x8", torch.from_numpy(x8)), ("low", low[0]), ("degmel", dm[0]), ("mel", mel[0]), ("wav24", wav[0])):
        refs[f"{mode}.{name}"] = t.detach().float().contiguous()
    print(mode, key, "x8", len(x8), "frames", mel.shape[-1], "wav24", wav.shape[-1])
save_file(refs, f"{out}/references.safetensors")
print("EXPORT_DONE", len(tensors), "tensors,", len(refs), "references")
