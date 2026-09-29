# License

unamblify is licensed under the **GNU Affero General Public License v3.0
only** (`AGPL-3.0-only`). See [`LICENSE`](https://github.com/rcludwick/unamblify/blob/main/LICENSE)
in the repository. This may change. The licence in the repository at any
given commit is the one that applies to that commit.

## Third-party code

The `codec2` crate is licensed `LGPL-2.1-only AND MIT`. It is the
pure-Rust Codec 2 vocoder used to capture the M17 modes
(`crates/unamblify-data`, pinned at 0.3.1). LGPL-2.1 is compatible with
AGPL-3.0: the combined work is distributed under the AGPL, and the
crate's own terms keep applying to the crate. So it is a normal
dependency here. It sits behind the `codec2` cargo feature of
`unamblify-data`, which is on by default. A build that must stay clear
of LGPL code can turn it off and lose only the `codec2-*` capture modes.
astar, whose default build avoids LGPL code, keeps the same crate behind
an opt-in feature. That is astar's choice, not a requirement of the
licence. The vendored ThumbDV driver (`vendor/ambe-thumbdv`) is
`MIT OR Apache-2.0`.

The vendored software D-STAR vocoder (`vendor/ham-digital-modes`) is
`LGPL-3.0-or-later`. It is the AMBE path of Bruce Perens'
`ham_digital_modes` from `hams_open`, and it is compatible with AGPL-3.0
on the same footing as `codec2` above. It sits behind the `perens` cargo
feature of `unamblify-data`, on by default. Turning it off loses only the
`dstar+perens` recode sibling, and the chip remains the only way to
capture D-STAR. Unlike `vendor/ambe-thumbdv`, this copy is **not**
verbatim. It is reduced to the D-STAR vocoder. Upstream's AMBE+2 (whose
patents have not expired), its fixed-point port, its own Codec 2 and the
unrelated digital modes are removed. `vendor/ham-digital-modes/VENDORED.md`
records the upstream revision, exactly what was dropped and how to
re-derive it.

## Training data

The model is trained on third-party speech corpora, each under its own
licence. This project redistributes none of them. The
[data sources](../research/data-sources.md) page records the licence and
attribution requirements for each, and the fetch script pulls them from
their original hosts.

## Trained weights

The first open weights are
[0.0.1-beta](https://github.com/rcludwick/unamblify/releases/tag/v0.0.1-beta),
released under the **GNU Affero General Public License v3.0 only**, the
same licence as the code. They are the restorer + waveform synthesiser
pipeline that `unamblify restore` runs.

The synthesiser is a fine-tuned derivative of Vocos
(`charactr/vocos-mel-24khz`, Copyright (c) 2023 Charactr Inc.), which is
MIT licensed. The release keeps its notice in `LICENSE-vocos`.

The weights also carry the attribution obligations of the corpora they
were trained on. The restorer used LibriTTS-R, VCTK, VoiceBank-DEMAND and
Hi-Fi TTS (all CC BY 4.0) and LJSpeech (public domain). The synthesiser's
first fine-tuning stage also drew on Mozilla Common Voice (CC0). The
release's `README.md` gives the full credits.
