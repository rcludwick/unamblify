# AMBE chip validation: findings so far

Real DVSI AMBE3003 hardware (a DVstick-33) is now reachable on the LAN via `AMBEServer3003`
(`192.168.10.189`, UDP ports 2460-2462, one emulated AMBE3000 channel per port), giving this
codebase's own from-spec P25 AMBE codec (`src/ambe/`) a real ground truth to validate against, and
the concrete configuration data needed to build a D-STAR mode. This document records what's been
confirmed so far, what's still open, and exactly how to reproduce or continue the validation.

## Executive summary (updated as of section 40) -- read this first

This document has grown to 40 sections across a long, multi-session investigation. This summary
exists so a reader (or a future session) doesn't have to read the whole thing to know where things
stand. Every claim below is sourced to its own section; treat this summary as an index and status
board, not a replacement for the underlying evidence.

**Chip modes with real, tested, chip-validated software** (all pass live against the real chip,
including real recorded speech, as of the latest re-run -- §23, §25, §29, §31, §35):
- **D-STAR** -- `ambe_dstar`, validated via `examples/ambe_chip_validate_dstar.rs`. Its own tone/DTMF
  frames (`ECMODE_IN`'s `TD_ENABLE` bit) are now also decoded and chip-validated (§40) --
  `ambe_dstar::decode::classify_b0`/`decode_tone`/`dtmf_digit_from_tone_index`, with its own
  dedicated pass/fail validator, `examples/ambe_chip_validate_dstar_tone.rs` (all 16 DTMF digits, all
  8 captured frames each, cross-checked against the independent `TONE_FRAME` ground-truth bit as well
  as `b0`).
- **AMBE+2 half-rate** (both FEC and No-FEC rates) -- `ambe_plus_2`, validated via
  `examples/ambe_chip_validate_ambe_plus_2.rs`. Ordinary speech, DTX-silence, and tone/DTMF frames
  (`ECMODE_IN`'s `TD_ENABLE` bit) are all now fully validated (§40) -- the chip genuinely detects a
  tone/DTMF digit but serializes it into the `Erasure` range instead of its own spec-defined `Tone`
  range (confirmed live, via forced generation, to be genuinely unreachable on this chip/rate, not
  merely undecoded), and the digit's own identity (and, for a plain single tone, its frequency) is
  fully recoverable via DVSI's own documented `TONE_IDX` field (`AMBE-3000R` manual Table 103/104,
  the same underlying identifiers as Annex J's own `f0`/`l1`/`l2` synthesis table) --
  `ambe_plus_2::decode::classify_b0`/`decode_tone_idx`/`dtmf_digit_from_tone_idx`, with its own
  dedicated pass/fail validator, `examples/ambe_chip_validate_ambe_plus_2_dtmf.rs` (all 16 DTMF
  digits plus 4 single tones, all 8 captured frames each, cross-checked against the independent
  `TONE_FRAME` ground-truth bit as well as `b0`). DVSI's own `Call Progress` tones were also tested
  (via forced generation): they read back as a second, distinct tone-bearing sub-code, `b0=122` (not
  `120`, `FrameKind::CallProgress`, kept separate from `FrameKind::Erasure`'s genuine `b0=121/123`) --
  see item 6 below for the full resolution, including the one real quirk this raised (a forced-DTMF
  rate-column mismatch, fully explained, not a lingering unknown).
- **RATET(27) P25 full-rate FEC/interleave layer, all 8 sub-blocks** -- `ambe::ratet27_wire_format` /
  `ambe::ratet27_fec`, validated via `examples/ambe_chip_validate_ratet27.rs` (§23, §29). Consolidated
  into one reusable entry point, `ambe::ratet27_frame::decode_frame`, tying the FEC layer together
  with the confirmed DTMF/DTX interpretations rather than leaving every caller re-derive the same
  8-block wire-bit extraction from scratch (§37).
- **RATET(27) DTMF encoding** -- `ambe::ratet27_dtmf`, fully decoded and validated (§25).
- **RATET(27) DTX-silence classification** -- `ambe::ratet27_dtx`, validated against real chip-
  reported `VOICE_ACTIVE` ground truth (not just stimulus inference), corrected once by its own test
  harness and once more when upgraded to ground truth (§31, §35).

**What "known and duplicated in software" does *not* yet cover -- the real, current gaps**:
1. **The semantic identity of most RATET(27) FEC sub-blocks.** `g0` alone is solidly confirmed as a
   gain/energy-related quantizer (§23, independently re-confirmed settling-independent in §33).
   `g1`/`g2` show real but weak, messy amplitude dependence that does *not* clean up with much
   longer settling (§33) -- likely gain-vector-related but not identified to a specific coefficient.
   `u4` shows a real, settling-independent dither signature -- a clean 2-state, exact-step-of-53
   pattern at fixed frequency/varying amplitude, but a messier, frequency-dependent multi-value
   pattern (not cleanly explained by `L_hat` alone) at fixed amplitude/varying frequency (§34) -- and
   a separate, fully-confirmed DTMF-column role (§25), but its *ordinary-voice-mode* semantic
   identity is still unknown. `u5` shows no clean signal under any test tried. `u6`'s
   pitch-candidate hypothesis was tested rigorously and explicitly **retracted** (§30) -- it is
   genuinely unstable across almost the entire tested frequency range. `c7`'s 2 of 7 bits are
   pitch-related (from an earlier session); the other 5 show real but complex, possibly
   waveform-shape-dependent signal (§24a). `g3`'s real 8-bit codeword space is fully implemented in
   software (§29) with a documented internal structure (steps of exactly 1024, independently
   replicated in §34), but *why* it's confined to 8 of its nominal 12 bits, and what it represents,
   remains open.
2. **A methodological complication discovered late (§32)**: `VOICE_ACTIVE` (and plausibly other
   readings) is adaptive/history-dependent, not a fixed per-frame function -- it responds to
   contrast with a slowly-adapting baseline. This was directly confirmed. Re-testing showed it does
   **not** universally explain earlier instabilities: `g0`'s own finding held up unchanged under 6x
   longer settling (§33), and `u4`'s frequency-dependent dither pattern likewise held up unchanged
   under 5x longer settling (§34), but `u6`'s instability persisted even under 500-frame settling
   (§30) -- so this is a real, partial explanation for *some* open items, not a master key to all of
   them. Separately, `g0`'s own DTX-silence classification (`DTX_SILENCE_G0`) was directly tested and
   found **not** history-dependent under 600 frames of sustained moderate noise (§35), unlike
   `VOICE_ACTIVE` itself.
3. **DTX's own "background noise level" claim** (DVSI's own manual wording) -- **now confirmed**
   (§35, upgrading §26's earlier inconclusive result): `g0` rises smoothly with actual noise level
   while the chip still judges the frame inactive (`VOICE_ACTIVE=0`), directly ground-truthed against
   the chip's own status flag rather than inferred from the stimulus alone.
4. **The exact quantizer formula** for `g0`'s gain values, and for `u4`'s frequency-dependent dither
   step, are not yet reduced to closed-form expressions matching a specific textbook equation -- and
   §34's own re-derivation found the naive "`L_hat`-dependent step size" hypothesis does not hold
   cleanly (identical `L_hat` values produced very different dither behavior at two frequencies), so
   this remains a genuinely open question, not just an unfinished derivation.
5. **RESOLVED (§39) -- a newly found, real `ECMODE_IN` feature at bit 8**: a systematic sweep of all
   16 `ECMODE_IN` bits (§36) found bit 8 measurably shifts `g1`/`g2`/`u4`/`u5`/`u6`/`c7` on identical
   content, and a follow-up characterization (§38) found it reliably produces `g0=1597` with no AGC-
   like transient, while also catching two of its own overclaims along the way (a tautological "boost
   curve," and an assumed-but-unconfirmed settling-shortfall explanation for a contaminated baseline
   measurement -- both corrected in §38, which also surfaced a genuine, separate finding: `g0` itself
   can be multi-modal for an unvarying pure tone over a long enough window, not just the
   already-documented blocks/frequencies). **Bit 8 is `CP_ENABLE` (Compand Enable), confirmed directly
   from DVSI's own primary-source manual (§39)**, fully explaining rather than merely correlating with
   every observed effect. `DCMODE_IN`/`DCMODE_OUT` are also now fully documented (§39) and confirmed
   out of scope (decode-direction controls this investigation's encode-direction methodology was never
   built to test). Does not resolve `g3`'s own plateau (§21) -- `g3` stayed within its already-known
   subspace under bit 8, and the manual itself contains no internal codec parameter documentation that
   could otherwise help.
6. **A real cross-mode `b0` labeling asymmetry (§40), with the detection question itself now
   resolved**: `ECMODE_IN`'s `TD_ENABLE` bit demonstrably acts on all three rates this document
   covers (RATET(27), D-STAR, AMBE+2 half-rate -- confirmed global), but the *serialized* `b0` value
   differs by rate for the same detected tone/DTMF stimulus: D-STAR reaches its own real `Tone` code
   (`b0=126`, matching mbelib's reference decoder exactly, with a fully decodable per-digit index),
   while AMBE+2 half-rate reaches `Erasure` (`b0=120`) instead of its own spec-defined `Tone` code
   (`b0=126/127`). **Checked directly against the chip's own independent ground-truth status flag**
   (`ECMODE_OUT`'s `TONE_FRAME` bit, DVSI's manual, read via `PKT_CHANFMT` the same way `VOICE_ACTIVE`
   was in §27/§28): `TONE_FRAME=1` for every one of AMBE+2 half-rate's 17 tone/DTMF stimuli, 8/8
   frames each, with D-STAR's own already-confirmed `Tone`/`b0=126` result serving as a positive
   control that the readback mechanism works correctly on this rate. **AMBE+2 half-rate's chip
   genuinely detects the tone -- this is not a detection failure -- it just doesn't serialize that
   detection into `b0`'s documented `Tone` range the way D-STAR or the spec text describe.**
   **RESOLVED in full, later in §40, for detected DTMF, detected single tones, and (via forced
   generation) call-progress tones and the `b0=126/127` question itself**: the detected tone's own
   identity *is* fully recoverable from the `Erasure` frame's other bits -- DVSI's own documented
   `TONE_IDX` field (`AMBE-3000R Vocoder Chip Users Manual` Table 103/104), found first by direct bit
   analysis of real captures and only afterward matched to the manual's own table. DTMF maps
   `0x80 | nibble`; a single tone maps `round(f0/31.25Hz)` -- both confirmed exactly against real
   captures (16 DTMF digits, 26 single-tone captures across 8 frequencies and 4 amplitudes) and
   chip-validated live, 160/160 (`examples/ambe_chip_validate_ambe_plus_2_dtmf.rs`). **Annex J's own
   `f0`/`l1`/`l2` table is the same underlying tone-identifier space, not a second, separate,
   still-mysterious one** -- `l1*f0`/`l2*f0` reconstruct the real DTMF/call-progress frequencies at
   the matching index, so Annex J is best understood as the decoder-side synthesis recipe for
   whichever identifier ends up in a frame. **`b0=126/127` was directly tested via forced generation
   (DVSI's own `TONE` field, bypassing the detector entirely, `examples/p25_ambe_plus_2_forced_tone_
   probe.rs`) and never observed** -- forced DTMF and forced single tones both read back `b0=120`,
   exactly like every detection-triggered capture; `FrameKind::Tone` is now understood to be
   unreachable on this chip/rate, not merely undecoded. The same forced-tone mechanism also found a
   second, genuinely distinct tone-bearing sub-code (`FrameKind::CallProgress`, not `Erasure`):
   forcing DVSI's own `Call Progress` tones (dial/ring/busy) reads back `b0=122`, and `decode_tone_idx`
   recovers `0xA0`/`0xA1`/`0xA2` **exactly as sent** -- no discrepancy at all for Call Progress, and
   none for single tones either.
   **Fully resolved: the one real quirk found was specific to forced DTMF, and only DTMF.** Forcing a
   DTMF `TONE_IDX` (e.g. `0x87`, `RATET(33)`'s own rate-33-61 code for '7') read back a *different*
   byte (`128`=`0x80`, that column's own code for '0'), which an earlier draft of this document
   wrongly generalized to Call Progress too, based on comparing a decimal readback against a
   hex-written input without converting -- re-checked directly in one base and there is no Call
   Progress discrepancy. The real DTMF quirk itself has a clean explanation: DVSI's Table 104
   documents *two* independent DTMF-to-`TONE_IDX` columns ("Rate Index 0-32" and "Rate Index 33-61",
   different non-monotonic nibble mappings), and at `RATET(33)` -- **the only rate tested; not yet
   confirmed for `RATET(34)` (AMBE+2 half-rate's own No-FEC rate) or any 0-32-group rate such as
   `RATET(27)`** -- the forced-generation `TONE` field is read by the encoder via the **0-32 column**,
   even though `33` is itself in the 33-61 group, while the encoder's own output (both a genuinely
   detected digit and a forced one's readback) is always reported via the 33-61 column at this rate.
   Checked exactly against all 16 forced DTMF digits: `readback = column_33_61[
   column_0_32[sent]]` matches every one of the 16 captures with zero exceptions -- not a guess, a
   confirmed formula. This does not affect the *detected*-DTMF finding's own live-validated
   correctness (128/128, in `TONE_IDX`'s own 33-61 numbering, confirmed directly) -- it only concerns
   how to interpret a byte fed into the forced-generation path. See `ambe_plus_2::decode::
   decode_tone_idx`'s own doc comment for the full derivation and a pinned regression test.

**What is deliberately out of scope**: DVSI's chip supports roughly 64 total `RATET` rate indices;
this investigation covers only the ones ham radio actually uses (D-STAR, P25 full-rate FEC, AMBE+2
half-rate FEC/No-FEC). The remaining ~58 rates are other DVSI-proprietary AMBE/AMBE+/AMBE+2
configurations not needed for any current ham use case, and were never attempted.

**The single most important methodological result, worth internalizing before continuing this
work** (§23): the exact bit-index-within-codeword permutation question that motivated much of the
early FEC-layer work is **provably unsolvable by black-box relational bit-flip testing alone**, no
matter how much data is gathered -- proven directly via a computational automorphism-invariance
argument, not just suspected. The fix that actually worked was **direct GF(2) rank/basis derivation
from real captured chip frames**, not more clever flip experiments. Any future semantic-layer work
should default to this same technique (capture many real frames, derive structure directly) rather
than re-attempting single-variable correlation sweeps, which this document's later sections (§30,
§33) show have a real, demonstrated failure mode: apparent correlations that don't survive proper
controls (same-condition pairs, long settling, varied stimulus content).

## 1. The chip's own rate table, confirmed against DVSI's own manual

DVSI's official "USB-3000 Manual" (downloaded from https://www.dvsinc.com/dlapps/appsoft.shtml)
gives the real, complete `RATET` index table for the AMBE-3000-generation chip (which is what's
inside a USB-3000/3003/3012 despite the table's own "AMBE-1000/2000/3000 Rates" section headers --
those are backward-compatible rate *definitions* the chip still implements, not evidence of which
chip generation is actually inside it; the manual states directly that USB-3000/3003 use the
AMBE-3000/AMBE+2 chip). Cross-checked index-for-index against `github.com/janakj/ambe`'s own
independently-sourced table -- exact agreement for indices 0-61 (janakj's table doesn't cover 62-63,
which are satellite-only rates).

**Index 27 -- 7200 total / 4400 speech / 2800 FEC bps -- is this codebase's own codec, exactly.**
`88 + 56 = 144` bits (this codec's own `VOICE_BITS`/`FEC_BITS`), and `4400 * 0.02 = 88`,
`2800 * 0.02 = 56` bits per 20ms frame -- an exact match, not a coincidence. Confirmed live: sending
`RATET(27)` to the real chip and encoding a test frame returns exactly 144 bits every time.

## 2. A real correction to this document's own prior citation: BABA vs. BABA-1

`AMBE_CODEC_AND_DSTAR_IMPLEMENTATION_PLAN.md` and this crate's own doc comments (`ambe/mod.rs`,
`ambe/pitch.rs`) cite "TIA-102.BABA" and describe it in prose as the "half-rate"/"AMBE-2000-era"
vocoder. Independently verified against the actual document titles (not assumed): **TIA-102.BABA
itself is titled "Project 25 IMBE Vocoder Description" -- the original, full-rate IMBE spec.**
**TIA-102.BABA-1** (a distinct addendum document, not the same as BABA) is titled "APCO Project 25
Half-Rate Vocoder Addendum." Given this codec's own frame structure (144 bits, four `[23,12]` Golay
plus three `[15,11]` Hamming codes, matching the well-documented classic P25 Phase 1 IMBE FEC layout)
and its exact numeric match to RATET index 27's "AMBE-2000 Rates"-labeled 7200/4400/2800 split, this
codec is almost certainly a real, correct implementation of the **original full-rate IMBE algorithm
(TIA-102.BABA itself)**, not the half-rate addendum the surrounding prose has been calling it. The
actual equations transcribed into `tables.rs`/`pitch.rs`/etc. should still be double-checked against
the real BABA PDF text directly (this correction is about the document's *identity*, not yet a
line-by-line re-verification of every transcribed table against it) -- but the prose describing this
codec as "half-rate"/"AMBE-2000" throughout the existing docs is now known to be imprecise and should
be corrected once that direct check is done.

## 3. Live validation harness: built, working, and already surfacing a real problem

`examples/ambe_chip_validate_p25.rs` configures the real chip for `RATET(27)`, feeds it a stationary
8kHz test tone 160 samples at a time over the real AMBEServer3003 UDP link, and runs the identical
signal through this crate's own encoder (bypassing the pitch tracker deliberately, using the tone's
exact analytically-known pitch, to isolate the quantization/DCT/FEC tables from any separate pitch-
tracker bug). Run it with `cargo run --example ambe_chip_validate_p25 -- 192.168.10.189 2460`.

**First real result, not yet resolved:**

- The real chip's own output is **bit-identical across all 40 test frames** once the signal is
  running. Originally read here as "expected and correct for a perfectly periodic input, since a
  well-behaved closed-loop encoder should converge to a fixed output" -- **that expectation itself
  was wrong, corrected after direct investigation below**, and in any case the chip's own
  convergence turns out to be weak evidence either way: item 2's own findings (already in this same
  document, below) show the chip's real output has no recognizable TIA-102.BABA FEC structure under
  eight independently-tried bit-layout hypotheses, strong evidence it's running a materially
  different vocoder generation (most likely DVSI's own AMBE+2) at this rate, not the published IMBE
  algorithm this module implements. A different algorithm's convergence behavior on this stimulus
  says nothing about whether TIA-102.BABA's own Eq. 54/62/63 are supposed to converge here.
- **This crate's own encoder does not converge to a fixed output for the same perfectly periodic
  input -- diagnosed directly, not just observed, and this is real DPCM limit-cycle behavior, not a
  bug.** Traced with a purely in-process reproduction (no chip needed): feeding the identical
  200Hz-tone PCM through `encode_frame`/`FrameState` repeatedly showed the frame-to-frame
  *unquantized* spectral amplitude estimate is genuinely bit-identical (confirmed to ~10 significant
  digits), so the oscillation is entirely internal to the prediction/quantization/reconstruction
  loop, not an artifact of the test harness's own windowing. The real mechanism: the gain vector's
  second-stage DCT coefficient `G_hat_2` (Annex F, 6 bits at this tone's `L_hat=18`) lands on a
  quantizer bin boundary, and `prediction::prediction_coefficient`'s own `rho ~= 0.49` feedback term
  (Eq. 54/77) amplifies each frame's quantization error by `1/(1-rho) ~= 1.96` before feeding it back
  into the next frame's residual -- the textbook condition for a closed-loop quantized DPCM predictor
  to settle into a stable period-2 limit cycle instead of a fixed point, confirmed exactly (bit-for-
  bit repeating every other frame from roughly frame 30 onward, out to at least 400 frames tested).
  This is structural, not a one-off coincidence of this exact PCM: the same kind of short, stable
  cycle (not divergence, and not exact convergence) appears across a wide tone-amplitude sweep
  (3000-12000 -- expected, since Eq. 54's own bias-correction term is proven elsewhere in this
  codebase to cancel a constant level exactly, so uniform amplitude scaling only shifts the coarse
  Annex E gain index `b2`, never the shape-encoding coefficients that actually drive the cycle) and
  for a harmonic-rich (sawtooth-like) stimulus too. A related, separate finding from the same
  investigation: block 0's own `C_1,2` coefficient (5 bits at `L=18`) saturates at its quantizer's
  maximum index for this stimulus, since a pure tone concentrates essentially all real energy on
  exactly one harmonic out of 18 -- an input far outside the dynamic range Annex G's step sizes are
  presumably tuned for, consistent with the reconstructed dominant-harmonic amplitude settling
  noticeably *below* the true input level rather than merely oscillating around it. One block
  (harmonics 13-15, the smallest higher-order bit budget among the unvoiced blocks) doesn't even
  settle into a short exact cycle within 400 frames, though its magnitude stays bounded in the
  expected noise-floor range rather than diverging. Net assessment: a maximally degenerate,
  perfectly-stationary single-tone stimulus is a real stress test this closed-loop predictive coder
  was very likely never designed or tuned against (real speech is never this stationary or this
  spectrally concentrated), and the oscillation is the generic, expected outcome of a quantized DPCM
  loop with `rho` not close to 0 -- not a transcription bug. See
  `src/ambe/mod.rs`'s own `encode_frame_stays_bounded_for_a_stationary_tone_even_though_it_does_not_
  converge` test for the permanent regression guard this became (bounded amplitude + a checked exact
  period-2 cycle for the dominant harmonic, not full-frame convergence or full-frame periodicity --
  the latter was tried and found unreliable, since the slowest-converging block's own cycle length
  varies with how long the run is allowed to settle).
- Bit agreement between the two streams across the offset search is essentially at chance
  (~49%, every tested frame-alignment offset from -4 to +4), which is far too low to be "close but a
  few table entries differ" -- this is more consistent with a real structural mismatch (bit-packing
  order, field order, or the modulation/scrambling step) than with a subtle numeric table error, and
  should be diagnosed *before* trusting any specific-table-value comparison. A concrete next
  diagnostic, not yet done: feed the chip's own real output frames into this crate's own
  `decode::DecoderState::decode_frame` and see whether the FEC (Golay/Hamming) checks pass at all --
  if they don't, that's strong evidence the bit-packing/field-order assumption
  (`pack_frame_msb_first` in the harness: c0..c7 concatenated MSB-first, widths
  `23,23,23,23,15,15,15,7`) is simply wrong and needs to be found empirically rather than assumed.

**Update, same session, after step 2 below**: tried this immediately (`examples/ambe_frame_diagnose.rs`,
also committed) -- unpacked the chip's real captured frame under five plausible bit/byte/field-order
conventions (MSB-first, LSB-first-per-byte, whole-stream-reversed, reversed field order, reversed byte
order) and ran each through this crate's own `decode_parameters`. **All five fail identically** (every
one triggers section 7.7's frame-repeat condition, meaning none produces a clean Golay decode of
`c_hat_0`). This rules out a simple bit-transposition bug as the explanation and points at something
more fundamental -- most likely that DVSI's chip, even at an aggregate bitrate matching RATET index
27 exactly, does not necessarily use the *same* Golay/Hamming split, bit-prioritization order, or
modulation PRN that TIA-102.BABA's own text describes; DVSI's own USB-3000 Manual never confirms the
chip's internal bitstream layout is identical to the published spec's example implementation, only
that the aggregate bit budget matches.

**A real, related discovery while chasing this**: G4KLX's AMBETools ships a full IMBE FEC codec
(`Common/IMBEFEC.cpp`) for converting to/from the P25 *over-the-air* dibit/`.dvtool` format --
real, working code showing an actual **144-bit interleave permutation** (`IMBE_INTERLEAVE[144]`) is
applied on top of the raw `c0..c7` concatenation before transmission, plus the same
content-dependent whitening/PRN construction this codec's own `modulation.rs` implements (seeded from
`c0`'s own 12 data bits via a linear congruential generator: `p = 16*c0; p = (173*p+13849) mod 65536`
-- worth directly comparing against `modulation::pseudo_random_sequence`'s own formula for an exact
match, not yet done). **This interleave is very likely an RF-modem/channel-symbol-level step applied
*downstream* of the vocoder chip, not something the AMBE3003 chip itself does to its own USB/serial
`CHANNEL` packets** -- AMBETools uses `CIMBEFEC` specifically for its `.dvtool`/over-the-air file
format conversion, separately from `DV3000SerialController`'s own direct chip I/O, which never
interleaves. So this probably isn't the missing piece for comparing against the chip's raw serial
output directly, but it's the right place to look once channel-frame-level work (a real, separate,
already-disclosed scope boundary per `mod.rs`'s own doc comment) is in scope.

**Update, same session, after the pitch-perturbation test below**: captured the chip's real output at
six nearby tones (150/175/200/225/250/300 Hz) and XORed each against the 200Hz baseline. The bit
positions that actually change for the four close frequencies (150/225/250, all within ~7-12 bits of
200's frame) cluster in a clean arithmetic sequence with **stride exactly 12** (e.g. `15, 27, 39, 51,
63, 75, 87, 99, 123, 135`) -- a real, structural signal, not noise, and consistent with some kind of
block-interleave with a 12-wide period (`144 / 12 = 12`). Tried deinterleaving the captured frame
through **eight** different concrete hypotheses before decoding: this crate's own real Annex H
dibit-interleave table, a real working reference implementation's own interleave table (AMBETools'
`IMBE_INTERLEAVE[144]`, tried both bit-numbering conventions), plus the five bit/byte/field-order
permutations from the previous update. **All eight still fail.** Checked precisely what "fail" means
here, since it matters: `golay_decode`'s own brute-force nearest-codeword search means `epsilon_0`
can never exceed 3 for *any* input at all -- Golay(23,12,7) is a **perfect code** (covering radius
equals packing radius, exactly 3), so literally every possible 23-bit pattern sits within distance 3
of some codeword. Every single one of the eight hypotheses above scored exactly `epsilon_0 = 3`, the
theoretical maximum -- meaning none of them found real structure; this isn't "close but not quite,"
it's indistinguishable from feeding the decoder pure noise.

**Honest assessment before continuing further**: the 300Hz frame's own huge bit-count jump (76 of
144 bits differ from 200Hz, versus single digits for the four closer frequencies) shows the chip's
real output *is* behaving like a genuine, well-structured vocoder -- pitch changes cause small,
localized differences until a harmonic-count-bucket boundary is crossed, then a large structural
change, exactly as this codec's own `L_hat`-dependent bit allocation would predict. So the chip is
producing real, meaningful, non-random output; the problem is specifically that this codec's own
assumed field/FEC layout (four Golay(23,12) + three Hamming(15,11) + 7 raw, in `c0..c7` order) has not
yet been matched to whatever layout DVSI's chip actually uses at this rate, despite the aggregate bit
budget being an exact, independently-confirmed match. The stride-12 clue is real and worth pursuing
further, but cracking DVSI's undocumented proprietary bit layout from here on is a genuine, patient
reverse-engineering project -- closer in kind to what the mbelib community did for D-STAR/DMR/NXDN
over real calendar time via captured-traffic analysis, not something a few more guesses will resolve.
A systematic, automated search (score every plausible field-order/bit-direction/rotation combination
against `epsilon_0` across *many* captured frames at once, to avoid a lucky-looking low score on just
one frame) is the right next tool if this is worth continuing, rather than further hand-picked
hypotheses.

**Update, after the D-STAR investigation resolved (see §5): the right methodology for this mystery is
now known, and one prior "real next step" is already confirmed done.**

The D-STAR investigation found and fixed two real framing bugs (a genuine wire-level block interleave
matching D-STAR's own over-the-air format, and a spare-bit position error) using a **falsification
test** -- does a hypothesis Golay-decode real, noise-free chip output with *zero* corrected errors on
essentially every captured frame -- rather than the correlation-based or single-frame-epsilon
comparisons tried above, both of which turned out to be too weak a signal (Golay(23,12,7)'s
perfect-code property means a single frame's low epsilon proves nothing, and this document's own §3
already found near-chance bit-agreement and correlation scores for every hypothesis tried so far). The
same falsification methodology, systematically automated across *many* real captured frames rather
than hand-picked hypotheses on one frame, is the right next tool here too -- item 1 below reflects
this update; the old single-hand-picked-hypothesis approach in the "Update" paragraphs above is
superseded.

**A second, related lesson from D-STAR, PARTIALLY RETRACTED after checking AMBETools more closely --
see the correction immediately below.** The D-STAR investigation's own first attempt to test
`szechyjs/dsd`'s real interleave table *also* "failed" outright, exactly like every P25 interleave
hypothesis tried above -- not because the interleave hypothesis was wrong, but because a *second*,
independent bug (the spare-bit position) was corrupting the test at the same time. This general
lesson (two independent bugs can mask each other in a falsification test) still stands. But the
specific idea below it -- that `AMBETools`' own `IMBE_INTERLEAVE[144]` might be the *real DVSI chip's*
P25 wire format, the same way `dsd`'s `dW`/`dX` turned out to be D-STAR's -- is now believed **wrong**,
per the correction below. AMBETools' `IMBEFEC.cpp::decode()` (fetched and read directly) does confirm
three OTHER structural details already independently confirmed correct in this crate (see item 3
below): whitening bits `23..137` (all of `c1..c6`, none of `c0` or the final 7 raw bits), data-bits-
first-then-parity-bits-second within each field, and the same `23,23,23,23,15,15,15,7`-bit block
sizes. Those three remain useful, real confirmations. Only the interleave-table claim is retracted.

**Correction: `IMBE_INTERLEAVE[144]`/`CIMBEFEC` is NOT the real DVSI chip's wire format --
it belongs to a completely unrelated, from-scratch open-source IMBE implementation.** Checking
`AMBE2WAV.cpp` (fetched from `g4klx/AMBETools` directly) shows `CIMBEFEC` is only ever constructed
inside a `#if !defined(HAVE_USB3000_P25)` / `else if (m_mode == MODE_P25)` branch, printing
`"Using open source IMBE vocoder by Pavel Yazev"` and calling `imbe_vocoder::imbe_decode` -- i.e. this
whole code path is the **software fallback used when no real DVSI USB-3000 P25 chip is present**.
When a real chip *is* available (`HAVE_USB3000_P25` defined), execution instead falls straight through
to `CDV3000SerialController::process()`, which talks to the chip's own `CHANNEL` packets directly and
never calls `CIMBEFEC` or touches `IMBE_INTERLEAVE[144]` at all. So the earlier "8 hypotheses including
AMBETools' own interleave table, both bit-number conventions, all failed identically" result (§3 above)
was testing an interleave that was never expected to apply to the real chip's own CHANNEL bytes in the
first place -- its failure is not evidence against a real DVSI-specific interleave existing, it's just
evidence against *this particular, unrelated* table. Unlike D-STAR (where `dsd`'s interleave table came
from a real, independent, working GMSK *demodulator* for the actual over-the-air signal the chip itself
also has to be compatible with), **no publicly available, independently-sourced reference for DVSI's
own real P25 chip wire format is currently known** -- this is a harder, more genuinely open reverse-
engineering problem than D-STAR's turned out to be, and should be approached accordingly: a systematic,
automated search over a principled space of transformations (block-order permutations, byte order,
per-byte bit direction), not a search for "the one real published table" the way D-STAR's search could
be, since none is known to exist publicly for this specific case.

**Real next steps, in order:**
1. **Build a proper falsification-test harness for P25**, mirroring
   `examples/ambe_chip_validate_dstar.rs`'s own approach: capture *many* real chip-encoded frames at
   several tones (not one hand-picked frame), and score hypotheses by how many of those frames
   Golay/Hamming-decode with zero corrected errors on *all four* Golay blocks simultaneously (not just
   `c0`) -- the real, decisive signal the earlier single-frame/correlation attempts above couldn't
   provide. Since no known-real reference interleave table exists for P25 (unlike D-STAR's `dsd`
   tables -- see the correction above), search a principled space of transformations rather than a
   short hand-picked list: byte order (normal/reversed), per-byte bit direction (MSB/LSB-first), and
   -- the new axis this correction motivates -- **permutations of the 8 blocks' own order** within the
   144-bit frame (`8! = 40320`, tractable to brute-force in a release build, especially with a
   syndrome-table Golay/Hamming decoder instead of brute-force nearest-codeword search for speed).
   This crate's own `c0..c7` data-vs-parity bit order within each field is already confirmed matching
   AMBETools (item 3 below), so that axis likely doesn't need to vary in the search.
2. ~~Diagnose and fix the 12-frame oscillation on a stationary input, independent of the chip
   comparison -- a closed-loop predictive encoder should converge to a fixed point for a genuinely
   stationary input, not cycle.~~ **Done -- diagnosed, and it's real DPCM limit-cycle behavior, not a
   bug to fix.** See the updated bullet earlier in this section for the full mechanism (a `rho ~= 0.49`
   prediction-feedback loop amplifying a `G_hat_2` quantizer-boundary error into a stable period-2
   cycle) and `src/ambe/mod.rs`'s own `encode_frame_stays_bounded_for_a_stationary_tone_even_though_
   it_does_not_converge` regression test. The original "period of exactly 12 frames" framing turned
   out to be an artifact of checking too early in the settling transient, not a real, exact period --
   direct measurement out to 400 frames shows most harmonics settle into a stable period-2 cycle,
   while at least one low-bit-budget unvoiced block never settles into a short exact period at all
   (though it stays bounded) -- so "does the 144-bit frame repeat with a short period" isn't itself a
   reliable invariant; "do reconstructed amplitudes stay bounded and sane" is the real, checked one.
3. ~~Directly compare `modulation::pseudo_random_sequence`'s own LCG formula against the
   `173*p + 13849 mod 65536` construction found in AMBETools' `IMBEFEC.cpp`~~ **Done, confirmed
   matching exactly, and while checking it two more structural pieces were confirmed matching too** --
   not the source of this mystery, any of them:
   - `src/ambe/modulation.rs`'s own `pseudo_random_sequence` already uses the identical
     `pr[0] = 16*u0; pr[n] = (173*pr[n-1] + 13849) % 65536` construction, byte-for-byte the same
     recurrence AMBETools' `IMBEFEC.cpp` and D-STAR's own whitening both use.
   - `src/ambe/mod.rs`'s own `c0..c6` assembly (`fec::golay_encode`/`fec::hamming_encode`) reuses the
     same "data bits high, parity bits low, MSB-first" systematic convention confirmed for D-STAR --
     matching AMBETools' own `encode()`, which reads each field's data bits first (positions `0..12`
     for a Golay block, `0..11` for a Hamming block) and appends parity second.
   - `modulation_vectors`' own whitening scope (`m_hat_1..m_hat_3`, the three Golay blocks after `c0`,
     plus `m_hat_4..m_hat_6`, all three Hamming blocks -- `m_hat_0`/`m_hat_7` never whitened) is
     `3*23 + 3*15 = 114` bits, exactly matching AMBETools' own `decode()` (`bit[i+23] ^= prn[i]` for
     `i` in `0..114`, i.e. bits `23..137`) bit-for-bit.

   With PRN, Golay/Hamming data-vs-parity convention, and whitening scope all independently confirmed
   to already match a real working reference, the wire-level interleave (item 1) is now the most
   likely remaining axis, by elimination -- not just "one hypothesis among several" the way it looked
   before these three were checked.
4. Extend the harness with a real decode-direction test (feed known bits to the chip's decoder, i.e.
   send a `CHANNEL` packet and read back the `SPEECH` response, compare PCM against this crate's own
   `decode_frame` output) once the encode-direction framing question above is resolved.

## 4. D-STAR: real, primary-source data now in hand, implementation not yet started

D-STAR's on-chip configuration is confirmed from **two independent primary sources**: DVSI's own
USB-3000 Manual (Table 30, "Custom Rate Control Words," explicitly labeled "interoperable with
D-STAR") and G4KLX's AMBETools source (`DV3000_REQ_DSTAR_FEC`) -- both give the exact same 6-word
RATEP rate-control-word: `0x0130 0x0763 0x4000 0x0000 0x0000 0x0048`, decoding (per DVSI's own manual)
to **3600 total / 2400 speech / 1200 FEC bps** -- i.e. a 72-bit, 9-byte frame every 20ms, not this
codec's own 144-bit frame. D-STAR's rate does not correspond to any single `RATET` table index; the
chip must be given these literal RCW words directly. A no-FEC D-STAR variant (`RATET` index 0, 2400
speech bits only, no FEC) is also documented and can be used to isolate FEC-layer bugs from
speech-quantization bugs during D-STAR development.

D-STAR's actual bit allocation (not documented in any TIA standard DVSI ships) was reverse-derived
from `mbelib` (github.com/szechyjs/mbelib) -- confirmed ISC-licensed for its D-STAR files
(`ambe3600x2400.c`/`ambe3600x2400_const.h`, permissive, not GPL as initially assumed), and per this
project's own established position (`docs/AMBE_TABLE_COPYRIGHTABILITY_ANALYSIS.md` in `hams_com`),
extracting the *numeric table values* for cross-reference is fine regardless -- new Rust code is
still written fresh, not adapted from mbelib's own C source.

**D-STAR's real 72-bit frame structure**, verified directly from mbelib's source:

- Four sub-blocks: `C0` (24 bits) + `C1` (23 bits) + `C2` (11 bits) + `C3` (14 bits) = 72 bits total.
- FEC: **only Golay(23,12)** is used (no Hamming, unlike this codec's own P25 FEC layout) -- one
  Golay(23,12) codeword each for `C0` (using 23 of its 24 bits, the 24th being an unverified spare/
  parity bit mbelib itself never actually checks) and `C1`. `C2` and `C3` are carried completely
  unprotected. Net: `12 + 12 = 24` data bits recovered from FEC-protected fields, `11 + 14 = 25` more
  carried raw, for **49 total decoded parameter bits** (72 - 49 = 23 bits of real FEC overhead).
- **A real quirk with no equivalent in this codec's own P25 path**: before decoding, `C1` is XORed
  with a pseudo-random sequence seeded from `C0`'s own already-corrected pitch bits (a
  content-dependent descrambling step, not the fixed/positional modulation this codec's own
  `modulation.rs` implements for P25) -- this needs its own from-spec (or, failing an available
  spec, carefully-referenced-and-independently-tested) implementation, not a copy of mbelib's own
  scrambling code.
- Parameter groups (bit widths, and which of `C0`-`C3` each comes from) are fully mapped out --
  pitch/tone index (7 bits), voicing pattern (4 bits), a gain term (6 bits), two PRBA
  (spectral-shape) gain groups (9 and 7 bits), and four higher-order-coefficient blocks (4 bits
  each) -- see the research transcript for the full bit-by-bit table if picking this up later; the
  actual lookup-table *contents* (pitch table, VUV pattern table, gain tables, PRBA tables, HOC
  tables) still need to be pulled from mbelib's source into this crate's own `tables.rs`-equivalent
  for D-STAR before any encoder/decoder can be written against them.

**Not yet done, and the largest remaining piece of this whole task**: writing the actual D-STAR
encoder/decoder module (a new `src/ambe_dstar/` or a D-STAR-specific path inside `src/ambe/`,
following this crate's own established pattern of one well-tested, doc-commented module per pipeline
stage), pulling in the real mbelib table *values* (not code), implementing the Golay-only FEC and the
C1 descrambling step, and validating it against the chip via the confirmed D-STAR RCW the same way
`ambe_chip_validate_p25.rs` already does for the P25 path.

## 5. D-STAR live validation: frame size, interleave, and framing confirmed against 320 real frames

Per Bruce's own direction ("do D-STAR first, then P25 -- maybe we will learn something from how
AMBE works for D-STAR"), the `ambe_dstar` module (see its own doc comment for the full frame
structure and mbelib provenance) was built and validated against the live chip before returning to
the P25 mystery above. Two real things were learned that generalize back to P25:

- **The same content-dependent PRN whitening construction (LCG: multiplier 173, increment 13849,
  modulus 65536) is used by both generations** -- confirmed independently in mbelib's real D-STAR
  source and in AMBETools' real P25 IMBE source (`Common/IMBEFEC.cpp`, found during the P25
  investigation above). This is genuine, useful cross-generation evidence for the DVSI-family
  whitening scheme as a whole, not something to re-derive per mode.
- **D-STAR's Golay(23,12,7) code is the exact same perfect code the P25 investigation already found
  ambiguous to validate with** (confirmed by reusing `super::ambe::fec::golay_encode`/`golay_decode`
  directly, without needing a second implementation) -- so the same "a low corrected-error count
  isn't by itself proof of correct framing" caveat applies here too, documented directly in
  `ambe_dstar::decode`'s own doc comment (after catching and fixing a first draft that had this
  backwards).

**Live validation, using the confirmed D-STAR RATEP configuration** (`0x0130 0x0763 0x4000 0x0000
0x0000 0x0048`, `examples/ambe_dstar_chip_check.rs`):

- Configuring the real chip with this exact RCW and encoding a test tone returns **exactly 72 bits
  (9 bytes)** every time -- confirms the frame *size* is right, live, not just on paper.
- **Real confound found and worked around**: the first perturbation tests used tone frequencies
  (150/175/225/300 Hz) whose period does not evenly divide the 160-sample frame, so consecutive
  "steady-state" frames of the same tone were not actually bit-identical (inter-frame phase drift).
  Fixed by restricting to frequencies whose period divides 160 exactly (50/100/200/250/400/500/800/
  1000 Hz) with frame-relative (not continuous) phase.
- **Second real confound found**: the chip's own D-STAR-mode encoder was found to **never converge to
  a truly fixed steady-state output** on a stationary pure-tone input -- 40 consecutive frames at a
  fixed 100 Hz tone never repeat exactly, oscillating quasi-periodically instead (this is plausibly
  the vocoder's pitch tracker hunting on a stimulus with no natural voice-like formant structure, not
  a bug in this investigation's own tooling). At 200 Hz and above the chip's output *does* converge
  to a fixed value after a short settling period (confirmed: standard deviation of decoded `f0`
  across 40 captured frames drops from the microvolt-noise floor of floating point at 200-1000 Hz,
  down from real double-digit-Hz variance at 50-100 Hz) -- worth knowing for anyone else probing this
  chip's D-STAR mode with synthetic test tones.
- **The real bit-order/framing bugs, found and fixed**: an initial correlation-based bit-order search
  (decoded `f0` vs. true test-tone frequency across byte-order x per-byte-bit-order x field-order
  hypotheses) never rose above noise-level correlation (~0.27-0.86 out of a possible 1.0, no clear
  winner) for any hypothesis -- misleading, because D-STAR's Golay(23,12,7) code is a genuine
  *perfect* code (covering radius = packing radius = 3), so a **falsification test** (does a
  hypothesis Golay-decode real, noise-free chip output with *zero* corrected errors on essentially
  every captured frame, not just "some" or "fewer than others") is the right diagnostic, not
  correlation. Two real, independent framing bugs were found and fixed this way, both now corrected
  in `ambe_dstar::interleave`/`ambe_dstar::decode`/`ambe_dstar::encode`:
  1. **The 9 raw CHAND bytes are not a simple `C0||C1||C2||C3` concatenation.** They carry the exact
     same block-interleave D-STAR uses over the air -- confirmed against `szechyjs/dsd`'s real,
     working GMSK demodulator (`include/dstar_const.h`'s `dW`/`dX` tables, fetched and verified
     directly, not transcribed from memory). This means DVSI's chip apparently transmits/receives a
     D-STAR CHAND frame pre-interleaved in exactly the RF transmission order, letting a repeater
     relay CHAND bits to/from RF with no separate interleave step of its own -- a genuinely
     interesting, previously-undocumented (in this codebase) design fact about the chip. Each byte's
     bits are read LSB-first.
  2. **`C0`'s spare bit is its own LSB, not its MSB.** This crate's first version of `parse_frame`
     masked off `C0`'s top bit as the spare and kept the bottom 23 bits as the codeword -- backwards
     from mbelib's real convention (`mbe_eccAmbe3600x2400C0`: `in[j] = ambe_fr[0][j+1]`, i.e.
     `ambe_fr[0][0]` is the spare and `ambe_fr[0][23..1]`, MSB-first, is the codeword), confirmed by
     fetching and reading mbelib's real `ambe3600x2400.c` directly. The wrong version's shifted-by-one
     codeword happened to still Golay-decode "successfully" (zero corrected errors) on some frames
     purely by coincidence -- to the *wrong* 12-bit data value -- which then poisoned every `C1`
     whitening seed downstream, making `C1` decode with the maximum possible corrected-error count
     (3) on literally every single frame, a strong enough signature to localize the bug via a direct,
     line-by-line cross-check against a from-scratch transliteration of mbelib's real C algorithm.
- **Final result, confirmed live against the real chip** (`examples/ambe_chip_validate_dstar.rs`,
  the permanent committed harness replacing the throwaway Python capture script): **320/320 real
  captured frames (50/100/200/250/400/500/800/1000 Hz, 40 frames each after a 10-frame settling
  discard) Golay-decode with zero corrected errors on both `C0` and `C1`** -- including the
  above-vocal-range 500/800/1000 Hz tones and the never-fully-converging 50 Hz tone (every individual
  frame it produces, even mid-oscillation, is still a well-formed, validly-encoded D-STAR AMBE
  frame -- the earlier "doesn't converge" observation was never evidence of corruption, just of the
  chip's pitch tracker legitimately changing its answer frame to frame). This is the real, decisive
  confirmation the perfect-code caveat above says a single frame's low error count alone can never
  give: 320 independent frames all landing on zero error is (1/2048)^320-level evidence against
  chance, not a coincidence.
- Semantic validation (does decoded `f0` actually track true input frequency, not just "does FEC
  validate") is the next real step now that framing is confirmed -- worth doing with a
  harmonic-rich stimulus (sawtooth/pulse train) rather than a pure sine, since AMBE's pitch estimator
  does harmonic matching and a single sinusoid is octave-ambiguous by construction; a pure tone can
  alias to a harmonic or subharmonic of its own true period even under fully correct framing.

## 6. Reproducing this work

- AMBEServer3003 runs as a systemd service on `pi500-1` (`192.168.10.189`), already configured and
  enabled at boot. **Never trigger this specific chip's UART BREAK/hard-reset** -- confirmed this
  session that it reliably locks up the chip's command pipeline until a full reboot; AMBEServer3003
  itself never does this, and neither should any future test code.
- `cargo run --example ambe_chip_validate_p25 -- 192.168.10.189 2460` from
  `hams_open/daemons/ham_digital_modes/` reproduces the P25-path finding above.
- `cargo run --release --example ambe_chip_validate_dstar -- 192.168.10.189:2460` runs the full,
  permanent live validation harness against the real chip: configures D-STAR RATEP, captures 40
  frames each at 8 test frequencies, and reports the Golay-decode exact-match rate for each (expect
  320/320 total). This is the committed replacement for the throwaway Python capture script this
  investigation started with.
- `cargo run --example ambe_dstar_chip_check -- <18-hex-char frame>` decodes one real captured 9-byte
  D-STAR frame (chip wire order) through `ambe_dstar::interleave` + `ambe_dstar::decode` and prints
  the recovered parameters -- useful for inspecting a single frame by hand.
- The mbelib tables themselves (pitch/VUV/gain/PRBA/HOC arrays) are already pulled into this repo
  directly, in `src/ambe_dstar/tables.rs`/`tables_prba.rs`/`tables_hoc.rs` -- generated
  programmatically from mbelib's own C header (not hand-transcribed) to avoid transcription error;
  this caught and fixed two real bugs during that process (see that commit's own message for detail).

## 7. P25 wire-format falsification-test harness: built, validated against known ground truth, and a genuine negative result against the real chip

Per section 3's "real next steps" item 1, a proper multi-frame falsification-test harness for the P25
wire-format mystery is now built: `examples/ambe_chip_validate_p25_wireformat.rs`. It captures many
real chip-encoded frames and scores candidate bit-order hypotheses by how many frames Golay-decode
with *zero* corrected errors on all four `c0..c3` blocks simultaneously -- the real, decisive signal
this document's own methodology section (end of §3) already established, rather than the
correlation/single-frame comparisons that misled the earlier attempts recorded in §3.

**What it tests, in three families:**

1. A cheap sliding-window scan (does a contiguous 23-bit or 15-bit window at some fixed offset land
   on a valid codeword across most captured frames?) across the 4 byte-order x per-byte-bit-direction
   combinations.
2. All `8! = 40320` permutations of which of the 8 fixed-size contiguous blocks (`c0..c3` at 23 bits,
   `c4..c6` at 15 bits, `c7` at 7 bits) occupies which position in the 144-bit frame, x the same 4
   byte/bit-direction combinations -- the harness this document's §3 asked for. Scoring correctly
   accounts for this codec's own confirmed content-dependent PRN whitening (§3 item 3): a candidate
   `c0` is checked for validity first (never whitened, so a valid `c0` directly gives real data via its
   own top 12 bits, no search needed), then `c1..c6` are dewhitened using this crate's own
   `ambe::modulation::modulation_vectors` (reused directly, not reimplemented) before being checked --
   both the dewhitened and raw (no-whitening-on-the-wire hypothesis) scores are tracked per hypothesis.
   Valid-codeword membership is checked via a precomputed 1-bit-per-codepoint bitmap (`golay_encode`
   over all 4096 data values, `hamming_encode` over all 2048), making each check O(1) rather than
   `golay_decode`'s own O(4096) brute-force search -- the full 40320-permutation x 4-byte/bit-hypothesis
   search runs in under 0.1 second in a release build against ~100-200 frames, nowhere near the "a few
   minutes" ceiling the task set.
3. A real, independently-sourced P25 Phase 1 IMBE OTA bit interleave: `szechyjs/dsd`'s
   `include/p25p1_const.h` (`iW`/`iX`/`iY`/`iZ`, ISC-licensed, fetched directly from GitHub -- numeric
   values transcribed for cross-reference only, no code copied, consistent with this project's
   established position on the mbelib tables in §4), applied via freshly-written Rust and cross-checked
   against `szechyjs/mbelib`'s real `imbe7200x4400.c` (`mbe_eccImbe7200x4400C0`/`Data`,
   `mbe_demodulateImbe7200x4400Data`) to confirm the resulting `imbe_fr[block][bitpos]` convention
   (bit weight `2^bitpos`), the data-first-MSB-first Golay/Hamming layout, and the PRN whitening
   formula/scope all match this crate's own `src/ambe/` exactly -- the same kind of primary-source
   cross-check that cracked D-STAR (§5), tried here because a real independent P25 demodulator has to
   be wire-compatible with the same over-the-air bitstream, and the AMBE3003's D-STAR CHAND bytes
   turned out to already be in real over-the-air order. Both dibit-bit-order conventions (which
   physical bit of each pair plays the demodulator's own "bit 1" role) are tried.

**The harness was validated against known ground truth before trusting any result against the chip**
(`--selftest`, no chip contact needed): it generates real frames from this crate's own from-spec
encoder (`ambe::encode_frame`, whose returned `c` is already fully modulated/whitened -- confirmed by
reading `ambe::mod::encode_code_vectors`) and packs them at the "natural" identity block order,
normal-byte, MSB-first convention. The permutation search correctly recovers this exact hypothesis
(`perm = [0,1,2,3,4,5,6,7]`, normal bytes, MSB-first) with **10/10 frames matching on all four
dewhitened Golay blocks** and, decisively, **10/10 on the full seven-block check** (four Golay +
three Hamming) -- while permutations that merely reorder the three same-sized Hamming blocks among
themselves still score 10/10 on the four-Golay-block metric (expected: that metric never touches
`c4..c6`) but collapse to near-zero on the seven-block metric, exactly the discriminating behavior
the two-tier scoring was designed to produce. This confirms the scoring pipeline (dewhitening via the
real `modulation_vectors` call, MSB-first bit extraction, membership-bitmap checks) has no
implementation bug that could hide a real positive.

**Live captures against the real chip**: two configurations, both via ordinary DVSI CONTROL/SPEECH UDP
packets (never touching the serial/USB layer, so the chip's UART-BREAK lockup hazard is not at risk):

- The task-confirmed P25 FEC RATEP rate-control-word (`0x0558 0x086B 0x1030 0x0000 0x0000 0x0190`):
  6 tones (100/150/200/300/400/600 Hz) x 2 amplitudes (2000/8000), continuous (not frame-relative)
  phase across frames for maximum bit diversity regardless of whether a tone's period evenly divides
  160 samples, 30 frames captured per combination after a 10-frame settling discard -- 360 frames
  captured, **125 unique** after dedup.
- RATET index 27 (this codec's own established default rate, same 144-bit frame, a different chip
  configuration path) as a secondary check on whether the two configs yield the same wire layout:
  3 tones x 1 amplitude, 90 frames captured, **31 unique**.

**Result: every hypothesis in every one of the three families scored at noise level for both
configurations.** The permutation search found **zero hypotheses** (out of 161280 checked per
configuration) reaching even the 2-frame noise-filter threshold on either the four-block or
seven-block metric. The dsd/mbelib OTA-interleave test scored **exactly 0 matching frames** for every
one of its 8 byte/bit/dibit-order combinations, on both configurations. The sliding-window scan found
no offset anywhere close to the ~100% match rate a true `c0` location should show (best Golay-window
hit was 3/125 frames, close to the 1/2048 per-frame chance rate x number of offsets tested, not a real
signal) -- consistent with, and independent confirmation of, the permutation search's own negative
result. RATEP(P25 FEC) and RATET(27) showed the same qualitative (negative) result, so this isn't
merely a property of one specific chip configuration.

**What this means, honestly**: given that the identical scoring pipeline decisively recovers the true
answer on synthetic ground truth, this is a real negative result, not a harness bug. It rules out,
for the real DVSI AMBE3003 chip's P25 `CHANNEL` bytes specifically:

- Any simple byte-order/per-byte-bit-direction combination with the 8 blocks left in their natural
  `c0..c7` concatenation order or any reordering of them as whole contiguous chunks (the full 8!
  search).
- The specific real P25 Phase 1 OTA bit-interleave `dsd`/`mbelib` use for actual RF demodulation, under
  either dibit-bit convention.

It does **not** rule out a genuine non-contiguous, bit-level interleave specific to DVSI's own chip
firmware that happens to differ from the standard's own over-the-air interleave -- which was always
the more likely outcome once the AMBETools `IMBE_INTERLEAVE[144]` table was ruled out as inapplicable
(see §3's correction) and no other independently-sourced reference for DVSI's *own* proprietary wire
format could be found. Unlike D-STAR, where the chip turned out to reuse the real over-the-air format
directly on its serial bytes, P25's chip apparently does not do the equivalent -- at least not via the
`dsd` reconstruction checked here, since confirmed against a stronger source below.

**Update, same session: the real, official standard's own interleave table was found, transcribed,
and tested -- still a clean negative.** Bruce pointed at two documents already in `hams_com`'s
`reference/ambe/`: `TIA-102.BABC_Vocoder_Reference_Test.pdf` (turned out to be an audio-quality
conformance *test procedure* manual, not bit-exact reference vectors -- not useful here) and
`TIA-102-BAAA-A_Project_25_FDMA_CAI.pdf`, the real Common Air Interface standard, which turned out to
contain exactly what was needed: **Table 5-1, "Interleaving Schedule for Voice Word"** -- the
standard's own authoritative voice-frame interleave (distinct from `dsd`'s third-party
reconstruction, and distinct from a separate, unrelated `7.2`-section data-channel interleaver in the
same document that a first pass mistakenly grabbed). Confirms this crate's own `c0..c7`
data-then-parity, MSB-first bit-numbering convention exactly (`c_0(22)` = Golay MSB, `c_X(14)` =
Hamming MSB, matching this crate's existing, independently-confirmed convention) and gives, for each
of the 72 transmitted dibit symbols, exactly which codeword bit is that symbol's Bit 1 and Bit 0.

Transcribed programmatically, not by eye, using **two independent extraction methods that agreed
exactly on all 72 symbols**: `pdftotext -layout` (regex-parsed after fixing a first attempt that
matched the table's own table-of-contents entry instead of its real location) and PyMuPDF's own
word-position extraction (grouping words by line, splitting columns by x-coordinate -- the same class
of technique this crate's own Annex G table used for a *different*, watermark-corrupted PDF, though
this document itself has no watermark or custom font encoding, confirmed via `pdfimages -list` and
`page.get_fonts()`). Both extractions were checked against the real structural invariant that each of
the 8 codewords' own bit-index set, collected across every appearance in the table, must be exactly
`0..width` with no gaps or duplicates -- true for all 8 blocks under both methods. The resulting
`apply_tia_interleave` function (in `examples/ambe_chip_validate_p25_wireformat.rs`) was itself
verified as a genuine bijection and a correct round trip (scatter a known `[c0..c7]` via the table's
own inverse, recover it exactly) before being trusted against real chip data.

**Result: this real, official, doubly-verified interleave table also scores at noise level against
the real chip**, on both RATEP(P25 FEC) and RATET(27) captures, across all 4 byte-order/bit-direction
combinations -- 0 matching frames out of 125 and 31 unique frames respectively, for every metric
(raw, dewhitened-4-block, and dewhitened-7-block). This is now a **much stronger** negative than the
`dsd`-based one: the standard's own text is unambiguous and directly authoritative, not a third
party's own reconstruction that could in principle have targeted a different protocol revision or
made its own transcription error. It also surfaces a real, relevant fact the document itself states
directly: "There are often other symbols interleaved within the voice frame" (frame sync, NAC, status
symbols/busy bits, encryption sync) -- real, non-audio information genuinely does get interleaved
into the *over-the-air* transmission, confirming a hypothesis Bruce raised directly. But the
`10 Annex for Transmit Bit Order`'s own per-symbol tables (covering the full Logical Link Data Unit,
not just the voice frame) describe those extra symbols as part of the larger over-the-air *frame*
structure, well beyond the 144 bits AMBEServer3003's `CHANNEL` packet always returns for this rate
(confirmed repeatedly, live, across every test in this document) -- strong evidence the DVSI chip's
own `CHANNEL` response is genuinely just the 144-bit voice codeword, stripped of that surrounding
RF-frame structure by the chip itself, not a case of non-audio bits silently making it into what this
harness assumed was pure vocoder data.

With four independently-sourced interleave/ordering hypotheses now tested and rejected (this crate's
own Annex H table, AMBETools' unrelated `IMBE_INTERLEAVE[144]`, `dsd`'s OTA reconstruction, and now
the real standard's own Table 5-1) plus the full `8!` contiguous-block-reordering search, the
conclusion stands even more firmly: DVSI's chip does not expose its P25 `CHANNEL` bytes in the
standard's own published bit order, unlike D-STAR. Cracking DVSI's actual proprietary P25 bit layout
from here, if still wanted, would need either a genuine from-scratch structural search (a much larger,
non-contiguous permutation space than the one searched here, likely intractable to brute force) or a
different empirical approach entirely (e.g. correlating specific known-content chip CONTROL/DATA
channel bits against expected vocoder parameter values one at a time, rather than guessing a global
bit-order transformation) -- a genuine, open reverse-engineering problem with no shortcut currently in
hand, exactly as this document's own honest assessment anticipated.

**Reproducing this work**: `cargo run --release --example ambe_chip_validate_p25_wireformat --
192.168.10.189 2460 --save <path>` captures fresh frames from the chip and runs all four hypothesis
families (sliding window, `8!` permutation search, `dsd`'s OTA interleave, and the real TIA-102.BAAA-A
Table 5-1) -- takes well under a minute total, dominated by the ~600 UDP round trips during capture,
not by the search itself. `--replay <path>` re-runs the analysis against a previously saved capture
without hitting the chip again. `--selftest` runs the ground-truth validation described above with no
network access at all (note: the TIA-interleave hypothesis scores 0 under `--selftest` too, since the
self-test's own synthetic ground truth is packed in this crate's plain contiguous `c0..c7` format, not
actually OTA-interleaved -- that's expected, not a self-test failure; `apply_tia_interleave`'s own
correctness was instead verified separately, via the bijection/round-trip check described above).

## 8. Also checked, per direct suggestions: offset/inversion, constant bits, and a real NOFEC mode

Three further, cheap-to-test hypotheses, checked directly against the real chip data before
concluding the wire-format mystery needs a genuinely open-ended search:

- **Simple offset or bit-inversion**: an exhaustive check of all 144 cyclic rotations x bit-complement
  (on/off) x the 4 byte-order/bit-direction combinations, against both the plain contiguous `c0..c7`
  layout and the real TIA-102.BAAA-A Table 5-1 interleave (§7), found **zero combinations with any
  match at all** -- not "a small improvement," literally every rotation scored the same as no
  rotation. Rules out a simple shift or global inversion combined with either candidate layout.
- **A large embedded non-audio header**: across 156 real captured frames (9 different tones/
  amplitudes), only **2 of 144** raw bit positions are constant across every frame, and which 2
  positions they are changes depending on byte/bit-order convention (i.e. they're not the same
  physical bits) -- nowhere near the many-bits-in-a-row pattern a genuine fixed sync/header field
  would produce. Real, though thin, evidence against a large fixed non-vocoder header living inside
  the 144-bit `CHANNEL` payload (the chip's response size is also always exactly 144 bits, matching
  the standard's own total frame size precisely -- not padded or truncated relative to it).
- **A real NOFEC mode, found and probed directly** (`examples/ambe_chip_probe_p25_nofec.rs`): DVSI's
  chip has a documented alternate rate configuration with no FEC at all -- confirmed via a real
  working reference (`DV3000_REQ_P25_NOFEC` in G4KLX AMBETools' `DV3000SerialController.cpp`) rather
  than guessed. Configuring the real chip this way and encoding returns **exactly 88 bits every
  time**, matching this crate's own `VOICE_BITS` exactly -- a real, independent confirmation of the
  voice/FEC bit split, live. This mode removes every Golay/Hamming/whitening/interleave ambiguity at
  a stroke: there is no FEC to protect, so (per the FDMA CAI document's own reasoning for *why*
  interleaving exists -- spreading burst errors across a *coded* word) there should be no reason to
  interleave a NOFEC frame either. The chip's NOFEC output also **converges to a perfectly stable,
  exactly-repeating value** for a settled stationary tone (confirmed across 200/400/500/1000Hz, no
  deviation across 10 frames after a 30-frame settle) -- cleaner than D-STAR's own low-frequency
  non-convergence.
  - **Bit-diffing the raw 88-bit NOFEC frames between different test frequencies** (the exact
    technique that found the original P25 FEC-mode "stride-12" clue in §3) shows only 3-5 of the 88
    bits change between any pair of the four frequencies tested, and -- a real, structural surprise --
    they are **not** clustered in the first 12 bits, where this crate's own `u0`-first, MSB-first,
    contiguous convention (matching TIA-102.BAAA-A's own stated field order and this crate's already-
    confirmed `c0..c7` numbering) would put the pitch parameter `u0`. Instead they land in what would
    be `u1`, `u2`, `u4`, `u5`, `u6`, and `u7` under that convention -- never `u0` or `u3`. Several
    fields shifting together, rather than one field varying smoothly across a 5x frequency range, is
    consistent with a harmonic-count-dependent bit-allocation boundary effect (changing pitch shifts
    `L_hat`, which shifts how many bits several *other* parameters get, per this codec's own real
    Annex-based bit allocation) -- a real, plausible mechanism, but not yet a settled "this specific
    bit range is pitch" conclusion. **Genuinely promising, not yet resolved**: NOFEC mode is a much
    cleaner signal than the FEC-mode wire-format search could ever be, and is the most promising
    concrete next step for continuing this investigation, rather than the intractable non-contiguous
    interleave search the FEC-mode results alone would otherwise motivate.

## 9. Breakthrough: NOFEC mode pitch field found -- it's `u2`, and it's Gray-coded

Directly prompted by two further questions -- "could they be using Gray coding?" and "look for bits
consistent across multiple frames" -- extended `examples/ambe_chip_probe_p25_nofec.rs` to capture 8
real frequencies (every one whose period exactly divides 160 samples: 50/100/200/250/400/500/800/
1000Hz, with 80 settling frames per tone) and, for each of the 8 raw fields `u0..u7` under this
crate's own MSB-first contiguous convention, check correlation against true frequency both as plain
binary and as Gray-decoded (standard Gray-to-binary conversion).

**`u2`, Gray-decoded, shows Spearman rank correlation = 0.976** (Pearson 0.58, lower because the
relationship is monotonic but not linear -- consistent with a logarithmic-style pitch quantizer,
matching this crate's own `src/ambe/`'s real `quantize_fundamental_frequency` formula, which is
itself logarithmic in frequency). Values across 50/100/200/250/400/500/800/1000Hz: `772, 3020, 3848,
4087, 3848, 4087, 4087, 4087` -- genuinely, robustly increasing with frequency (the plateau at 4087
for 250/500/800/1000Hz is consistent with a real quantizer ceiling: this crate's own `L_TABLE`-style
pitch tables also saturate above a maximum representable frequency). Reproduced identically (spearman
0.976 both times, values changing by less than 1% between runs) across two independent live captures.
**No other field, under either binary or Gray interpretation, comes remotely close** (the next-best
is `u1` binary at spearman -0.429, i.e. weak and the wrong sign). Plain binary `u2` itself shows
essentially zero correlation (spearman 0.048) -- the Gray-decoding step is what makes the signal
appear, a real, direct confirmation of Bruce's own Gray-coding hypothesis.

This overturns the working assumption (`u0` is pitch, per TIA-102.BAAA-A's own field labeling and
this crate's own `src/ambe/`) for whatever generation of AMBE DVSI's real chip is actually running --
consistent with the standing belief that the chip runs a different, proprietary generation
("AMBE+2") rather than the published open IMBE algorithm this crate implements from the TIA standard
text. Genuinely new, actionable information: DVSI's chip appears to (a) put its own pitch parameter
in the position this crate calls `u2`, not `u0`, and (b) Gray-code it, where the published IMBE
standard uses a plain quantizer index (`b_hat_0` in TIA-102.BAAA-A's own notation) with no Gray
coding mentioned anywhere in that document.

**50Hz and 100Hz remain unsettled** even after 80 settling frames (each frame in the capture
oscillates slightly, not truly converging the way 200Hz+ do) -- the same low-frequency non-
convergence pattern already documented for D-STAR (§5) and for this crate's own encoder's
degenerate-stimulus oscillation (§3). Their data points are real but noisier than the fully-converged
200-1000Hz points; the strong correlation already holds without needing them at all (recomputing
Spearman over just the 6 fully-converged points would only strengthen it further, not weaken it,
since they already sit at the low and high ends of the monotonic trend).

**Honest next steps, not yet done**: (1) find and Gray-decode the *other* real parameters (voicing,
gain, spectral shape) the same way, now that the general "check Gray coding, don't assume plain
binary" lesson has a concrete confirmed instance to generalize from; (2) determine the *exact* bit
width and position of the real pitch field within `u2`'s own 12-bit span (it may not be all 12 bits,
or may not align exactly with this crate's own field boundary -- worth checking with a systematic
bit-window slide inside and around `u2`, the same technique `ambe_chip_validate_p25_wireformat.rs`'s
sliding-window diagnostic already uses for FEC-mode Golay/Hamming windows); (3) revisit the FEC-mode
wire-format mystery (§3, §7) with this new information -- if DVSI's chip really does relabel/Gray-code
its own parameters relative to the published IMBE spec, the FEC-mode `c0..c7` assignment itself may
need the same kind of correction, not just a bit-order/interleave fix.

## 10. A real, independent cross-check: GopherTrunk's pure-Go IMBE/AMBE+2, and a clean negative for the textbook `b_hat_0` formula

Bruce raised a sharp, well-founded challenge to the whole "DVSI's chip diverges from the published
spec" line of reasoning: DVSI co-designed the underlying algorithm TIA standardized as IMBE, so a
wholesale divergence between the chip and what DVSI itself told APCO/TIA seemed implausible -- maybe
the chip is simply *configured* wrong, not running something exotic. Investigated directly rather than
assumed either way, using a real, independent, actively-developed open-source reference: **GopherTrunk**
(`github.com/MattCheramie/GopherTrunk`, Apache 2.0, a pure-Go SDR trunking-radio decoder with its own
from-scratch IMBE and AMBE+2 vocoder implementations, no DVSI/mbelib dependency).

**A striking, independent confirmation of this crate's own `bit_prioritization.rs`**: GopherTrunk's own
`internal/voice/imbe/doc.go` states directly, from its own from-scratch reading of TIA-102.BABA, that
"the `b_0` fundamental-frequency parameter lives at scattered positions `{0..5, 85, 86}`" within the
88-bit information vector -- i.e. **not** a simple contiguous first-12-bits field. This crate's own
`bit_prioritization.rs` (built independently, months earlier, from the same TIA-102.BABA text) already
implements exactly this: `prioritize_bits`'s own Step 1 places `b_hat_0`'s top 6 bits at the very front
(`u_hat_0`'s own top 6 of 12 bits) and Step 8 places its bottom 2 bits in the last 4 bits of the whole
88-bit stream (`u_hat_7`'s bits 1-2) -- and `extract_fundamental_frequency_quantizer(u)` already exists
as the direct, already-tested inverse: `((u[0] >> 6) << 2) | ((u[7] >> 1) & 0b11)`. Two independent
implementations of the same published spec landing on the identical scatter pattern is real, strong
evidence this crate's own *software* implementation of the published algorithm is correct -- the
mystery genuinely is about what the *chip* does, not a bug in this codebase's own reading of the
standard.

**Applying the real, correct formula to the real chip's NOFEC captures (§9) is a clean negative,
across every byte/bit-order hypothesis.** Naive `u0`-only extraction was always going to be wrong once
`bit_prioritization.rs`'s own scatter pattern was accounted for -- but running the *actual* formula
(`top 6 of u0`, `bits 1-2 of u7`, both binary and Gray-decoded) against the same 6 real, fully-converged
NOFEC frames (200/250/400/500/800/1000Hz) shows **no correlation with frequency under any of the 4
byte-order x bit-direction combinations** -- `b_hat_0` comes back completely constant under 2 of the 4,
and near-constant-with-noise under the other 2. This is a real, decisive result, not a step backward:
it means the chip's raw NOFEC bits are **not** simply "the textbook-prioritized IMBE information bits,
just in an unknown byte/bit order" -- ruling out the most natural remaining "maybe it's just configured
slightly wrong" explanation for NOFEC mode specifically. The `u2`-Gray-decoded correlation found in §9
(spearman 0.976) remains the strongest real, empirical signal so far, and it does *not* correspond to
where the textbook algorithm would put pitch -- consistent with the chip genuinely using a different
internal parameter layout, not a configuration mistake in this investigation's own test harness.

**A further real clue GopherTrunk surfaces, worth chasing next**: its separate `internal/voice/ambe2`
package (AMBE+2, used for P25 *Phase 2*, DMR, and NXDN -- a different, 49-bit-information, 2400 bps
frame, citing the exact same `szechyjs/mbelib` `ambe3600x2400.c` source this codebase's own
`ambe_dstar` module (§4-§8) was independently built from) documents `b_0`'s own AMBE+2-family scatter
and gain/PRBA/HOC structure as visibly different in *character* from IMBE's (matching this repository's
own D-STAR findings: Gray-coding-adjacent quantization, scattered small parameter fields, a
content-dependent PRN whitening keyed on `b_0`). Given (a) the `u2`-Gray finding just confirmed the real
P25 chip's NOFEC pitch parameter is genuinely Gray-coded (unlike textbook IMBE, which uses a plain
index) and (b) DVSI's own rate table groups the P25 rate used here under an "AMBE-2000/3000 Rates"
section header rather than an "IMBE" one (§1), the working hypothesis is now more concrete than
"probably AMBE+2" in the abstract: **the real DVSI P25 chip likely exposes an AMBE-family (not
textbook-IMBE) parameter layout even in its nominally "P25 IMBE" rate configurations**, and the D-STAR
generation's own real, source-verified quantizer/whitening conventions (`ambe_dstar/tables.rs`,
`ambe_dstar/whitening.rs`) -- not the published TIA-102.BABA IMBE algorithm this crate's `ambe/` module
implements -- may be the right family of hypotheses to test against the P25 chip's raw bits next,
rather than continuing to permute textbook-IMBE's own byte/bit order.

## 11. Resolution (partial, for a different rate): building real AMBE+2 half-rate and testing it against the chip's own documented "APCO Project 25 half-rate" configuration -- a genuine positive result

Bruce's own direct authorization, after reviewing this whole investigation: "Build the decoder, keep
it conditionally compiled out by default, with the explanation that it's kept compiled out until we
can clearly exercise the patents. Use it to test internally... And the encoder, please." A full
AMBE+2 half-rate encoder and decoder now exist at `src/ambe_plus_2/` (gated behind the `ambe_plus_2`
Cargo feature, off by default -- see `src/ambe/AMBE_PLUS_2_NOTES.md`'s own dated section for the
implementation details), built from the real TIA-102.BABA-1 addendum data already extracted in that
notes file plus mbelib's real `ambe3600x2450.c`/`ambe3600x2450_const.h` source for the procedural
details (bit scatter, FEC structure, whitening).

**A real, previously-unused fact found while preparing this build**: DVSI's own USB-3000 Manual
(already on hand from §1's own citation) names `PKT_RATET` Rate Index 33 **"APCO Project 25
half-rate with FEC (3600 bps)"** (control byte `0x21`) and Rate Index 34 **"...with No FEC (2450
bps)"** (`0x22`) -- in the manual's own words, not this investigation's inference, this is exactly
TIA-102.BABA-1's own half-rate addendum, i.e. AMBE+2. This is a **different, distinct rate** from
RATET 27 (this section's own predecessor sections 3, 7-10, which remains a genuine, unresolved
negative result for the full-rate/88-144-bit configuration) -- Rate 33/34's own 72-bit/49-bit frame
size genuinely matches the newly built codec's own frame size, where RATET 27's does not.

Configuring the real chip for RATET 33 and 34 and capturing live 8-test-tone data
(`examples/ambe_chip_validate_ambe_plus_2.rs`, the same settling-frame methodology as every other
harness in this crate) gave a real, decisive **positive** result on both rates:

- **RATET 34 (No FEC, 49 raw bits, no framing ambiguity at all)**: the same hypothesis-agnostic
  sliding 7-bit-window correlation scan that found §9's own Gray-coded `u2` pitch field, applied
  here, found bits `[29..36)`, **Gray-decoded**, with **Spearman rank correlation 0.976** against
  true frequency -- the identical magnitude to §9's own full-rate finding. A real, independent
  confirmation (different rate, different frame size, same chip) that this DVSI chip family's pitch
  parameter is genuinely Gray-coded as a general property, not an artifact specific to one rate.
- **RATET 33 (with FEC, 72 bits)**: two framing hypotheses were tried on the captured bytes -- a
  direct `C0||C1||C2||C3` concatenation (**0 of 8** captured frames Golay-decoded with zero
  corrected errors on both `C0` and `C1`) and TIA-102.BABA-1's own Annex H interleave, as
  implemented in `ambe_plus_2::interleave` (**8 of 8** frames, perfect). Golay(23,12) is a genuine
  perfect code, so a *wrong* framing hits zero corrected errors on a real 23-bit input only with
  probability 2^-11 per codeword; 8-for-8 across two independent codewords per frame, across 8
  different real captured tones, rules out coincidence. **The chip's own real wire format for this
  rate is exactly TIA-102.BABA-1's Annex H interleave** -- the same real finding this repository's
  own `ambe_dstar` investigation (§4-§8) made for D-STAR's wire format, now confirmed for a second,
  independent rate/generation. With the correct (deinterleaved) framing, the recovered `b0` pitch
  index tracks true frequency exactly as a real, working AMBE+2 encoder should: monotonically
  decreasing from 118 (50Hz) to 90 (100Hz) within AMBE's own designed vocal-pitch range, then
  saturating into the reserved 120-123 (erasure) code range for the 200-1000Hz test tones outside
  that designed range (120, 120, 120, 121, 121, 122 respectively) -- the same "quantizer ceiling
  saturation" shape already documented in §9 for full-rate NOFEC mode (there, a plateau at a fixed
  raw value; here, saturation into the reserved-code boundary itself), not a framing bug.

**Conclusion, reported honestly regardless of outcome, per this whole investigation's own
discipline**: this is a genuine **positive** result, not a negative one -- when the real DVSI chip
is explicitly configured for its own documented "APCO Project 25 half-rate" rate (33/34), it decodes
bit-for-bit through this freshly built, from-spec AMBE+2 half-rate codec, with a real wire-format
interleave exactly matching TIA-102.BABA-1's own Annex H, and a pitch parameter that tracks true
frequency exactly as a working vocoder should. **This does not, by itself, explain or resolve the
separate RATET(27) full-rate mystery** (§7-§10) -- that remains a genuine, still-open negative
result for a different rate with a different (88/144-bit) frame size; a working half-rate match does
not retroactively make the full-rate chip's own output "actually AMBE+2" too, and no attempt was
made here to re-decode RATET(27) data through this new half-rate codec (the frame sizes don't match,
as this section's own second paragraph notes). What this section *does* establish: the real chip
hardware genuinely implements standard, spec-compliant AMBE+2 half-rate when asked for it by its own
documented name, which is itself useful, real confirmation that this new codec's from-spec
implementation (tables, FEC framing, and bit scatter, all traced from mbelib's real source and
cross-checked against the TIA annexes) is correct against real, independent silicon -- not just
against mbelib's own software reimplementation of the same published spec.

**Honest next steps, not yet done**: (1) the exact bit width/position of the real Gray-coded pitch
field found here (bits `[29..36)`) has not been cross-checked against this codec's own `b1` field
position (`d[4..8)+d[35]` in the FEC-codeword-order hypothesis) -- the two don't obviously line up,
worth a real investigation rather than assuming either is simply "the" pitch field; (2) RATET 33's
own zero-error framing win was found by trying only the two most obvious hypotheses (direct vs.
Annex H) -- a real bit-order/byte-order sweep like §7's own systematic search was not performed
here, since Annex H already won cleanly on the first two tries; (3) Annex J's tone-frame mode
remains unimplemented (a disclosed stub, not silently skipped) -- if a captured frame's own `b0`
ever lands in 126-127 during future testing, that data is currently discarded rather than decoded.

**Independent re-verification, same session, with two more test frequencies (80Hz and 160Hz added
to the original 8)**: reproduced the whole result directly, including rebuilding the example and
re-running it live against the chip -- **10 of 10** frames now Golay-decode with zero errors under
Annex H framing (0 of 10 for direct concatenation), an even stronger margin. The two new points fill
in the trend cleanly and sharpen the picture of where the erasure boundary actually sits: `b0` goes
118 (50Hz, `f0~=66Hz`), 91 (80Hz, `f0~=100Hz`), 90 (100Hz, `f0~=101Hz`), 66 (160Hz, `f0~=146Hz`), then
120/120/120/121/121/122 (erasure) for 200/250/400/500/800/1000Hz. The real, useful new observation:
80Hz and 100Hz decode to nearly identical `f0` (~100-101Hz) despite differing true input frequencies
-- a real quantizer-neighborhood/pitch-tracking-confusion effect on a pure-tone stimulus (consistent
with this whole investigation's own repeated finding, first noted for the base-rate P25 encoder in
§3, that a bare sinusoid is a genuinely degenerate, out-of-design-envelope input for an AMBE-family
pitch tracker) rather than a framing error, since the *codeword-level* decode is already proven exact
by the 10/10 zero-Golay-error result independent of what semantic value the recovered bits happen to
mean. The erasure cutoff itself is clean and perfectly deterministic (every one of the 6 higher test
tones lands in 120-123 every time, not intermittently) -- consistent with a genuine chip-side
low-confidence/erasure declaration on non-voice-like pure-tone input outside its designed ~65-400Hz
working range (matching `W0_TABLE`'s own real endpoints, `b0=119` -> `f0~=65Hz` and `b0=0` ->
`f0~=400Hz`), not a remaining decode bug.

## 12. Real-speech validation: 1274/1274 (100%) Golay-clean frames on DVSI's own reference test speech

Bruce authorized using DVSI's own bundled USB-3000 software package directly (the manual was
downloaded from DVSI's own public download page, so a trade-secret claim would not be enforceable;
the specific patents named in that package's source headers were all issued in the 1990s/2001, whose
20-year terms have long since expired) -- "proceed with the implementation of all options, and make
the API operate as it does in the AMBE manual." That package's `usb3k-linux.tar.gz` (a Linux
reference client) bundles `in.dat`, a real ~25-second speech recording, and hardcodes `RATET(33)` --
the exact AMBE+2 half-rate FEC configuration this session already validated live with synthetic
tones (§11) -- as its own smoke-test rate.

**Built `examples/ambe_plus_2_dvsi_reference_replay.rs`**: configures the real chip for `RATET(33)`,
feeds it DVSI's own real reference speech frame by frame, and decodes every real encoded frame
through `ambe_plus_2::decode` with the Annex H framing already confirmed correct. **A real bug found
and fixed while building this**: an early version configured the chip with a RATEP custom word
copied from the wrong section of DVSI's manual (Figure 20's *full-rate* P25 example -- the same word
already used for the still-unresolved RATET-27 mystery, §§3, 7-10) instead of the simple `RATET(33)`
index already proven correct. This silently made the chip respond with 144-bit full-rate frames
instead of AMBE+2's own 72-bit ones -- caught by checking the actual returned bit count directly
rather than assuming the configuration took effect, once 0% Golay-clean on what should have been
working real speech looked wrong. Also found and abandoned: an attempt to replicate DVSI's own
pipelined (3-frame-lookahead) encode/decode protocol, traced directly from the bundled reference
client's real source, does not survive cleanly through AMBEServer3003's own UDP relay (responses
arrived out of the expected order) -- a real, disclosed limitation; a simple synchronous
per-frame protocol was used instead, which sacrifices an exact frame-aligned PCM comparison against
DVSI's own official reference output but is sufficient for what actually matters here (Golay
validity and semantic plausibility).

**Result: 1274 of 1274 real frames (100.0%) Golay-decode with zero corrected errors on both
protected codewords** -- every single frame of DVSI's own ~25-second reference speech recording,
not just synthetic test tones. All 1274 frames also classify as genuine `Speech` (zero
erasure/silence/tone) -- unlike the earlier synthetic-tone tests, where several out-of-vocal-range
pure tones legitimately triggered erasure (§11); real, continuously-spoken human speech content
never does. The recovered pitch trajectory shows real, smooth, speech-like variation (e.g. `100, 115,
107, 107, 107, 77, 75, 75, 75, 79, 79, 78, 79, 79, 77, 77, 75, 91, 91, 90, ...`) -- gradual drift
punctuated by occasional larger jumps consistent with real voiced/unvoiced or word-boundary
transitions, not random noise. This is a stronger, more decisive validation than the earlier
pure-tone tests: 1274 independent real frames all landing on zero Golay error, using DVSI's own
official reference test material (not this investigation's own synthetic stimuli), on real
continuous speech content the codec was actually designed for.

**Note on provenance, deliberately not changed by this authorization**: `in.dat`/`cmp.dat`
themselves, and the reference client's own confidential source code used to understand the
pipelined protocol, are kept local (this session's own scratchpad) rather than committed to this
public repository -- the patent-expiration reasoning above clears using this material for
understanding/testing, but redistributing DVSI's own bundled test audio and source code here is a
separate question this authorization did not address, so the more cautious default (already used
elsewhere in this crate for audio of uncertain redistribution status) stays in place for those
specific files. Only this crate's own, independently-written Rust code and this findings summary are
committed.

## 13. Returning to RATET(27) full-rate: a sliding Golay-window scan, and what it does/doesn't rule out

With AMBE+2 resolved, this session returned to the still-open RATET(27) full-rate mystery (§§3, 7-10)
armed with one new idea from the AMBE+2 work: what if full-rate IMBE's real wire format, like AMBE+2's,
differs from the textbook `BAAA-A` spec in ways this investigation hadn't tried?

**Sliding 23-bit Golay-window scan** (`examples/p25_ratet27_sliding_golay_scan.rs`): rather than
assuming the textbook `c0..c7` block positions, this scans every possible 23-bit starting position
across 26 unique real captured RATET(27)-FEC frames (8 test tones, settled), under all 4 byte-
order/bit-direction hypotheses, checking each window's Golay(23,12) validity rate. **Result: a clean
negative** -- no window position scores above 3.8% validity (best: 1/26) under any hypothesis, versus
the ~2048x-random-chance signal a real byte-aligned Golay codeword would produce.

**What this precisely does and doesn't rule out** (caught before over-reading the result): P25 IMBE's
real bit modulation XORs `c1-c6` with a PN sequence seeded from `u0` before transmission, and
interleaves all codewords' bits across the frame -- so even the textbook-correct format would show
*at most one* clean window (`c0`, the only unmodulated, unspread codeword), not four, and this scan
cannot see through either the PN whitening or an unknown interleave (which scatters each codeword's
23 bits to non-contiguous positions). What it does rule out: any format where a full 23-bit Golay
codeword sits contiguously, byte-aligned, unmodulated, anywhere in the raw 144 bits, under a simple
byte/bit-order transform. A genuine interleaved-and-whitened format is not excluded by this test.

**Also revisited**: the NOFEC-mode "pitch is Gray-coded at raw bits [29..36)" finding from §9 was
initially flagged as needing re-verification, on the concern that RATEP's NOFEC control word
(`RCW2=0x0000`) might select a different underlying vocoder rather than "the same codec, FEC off".
Checked directly against the manual: Table 8/9 explicitly label *both* the FEC and No-FEC RATEP
examples "APCO Project 25 full-rate" -- distinct from the separately-labeled "APCO Project 25
half-rate" AMBE+2 rates (Table 10/11) -- confirming they are the same underlying vocoder family, so
the §9 Gray-coding finding stands as evidence about full-rate IMBE's own real parameter layout.

## 14. A working oracle: the chip's own decoder finds two real unprotected wire bits by direct experiment

Given the sliding-scan negative couldn't see through PN-modulation or interleaving, the next approach
(suggested during a design review) uses the chip's own DECODER as a ground-truth oracle instead of
guessing a table: capture one real encoded frame `R`, flip one wire bit at a time, and check whether
the decoded output changes. A bit's membership in "some FEC-protected codeword, corrected on decode"
survives *any* interleave, PN-modulation, or byte/bit-order convention, since a single flipped wire
bit always flips exactly one bit of whatever codeword it maps to, however that mapping works --
sidestepping every open convention question from §13 at once.

**Two real dead ends found while building this, both left in
`examples/p25_ratet27_bitflip_oracle.rs`'s own doc comment as a durable warning**:
1. **Raw time-domain PCM comparison, using a voiced 200 Hz test tone for `R`, doesn't work** -- not
   because the decoder is non-deterministic, but because it correctly, continuously tracks pitch
   phase across frames for smooth voiced synthesis. Two consecutive decodes of the exact same
   unmodified frame produced a real measured RMS difference of ~8500 -- both were clearly the same
   frequency and amplitude, just at different phases. Comparing raw samples flagged all 144 bits as
   "changes decode".
2. **Switching `R` to digital silence** (suggested directly by Bruce, to sidestep phase entirely)
   does fix determinism -- residual RMS drops to ~2, a small comfort-noise/dither floor, not exact
   zero -- but then flipping *any* of the 144 bits shows zero effect. The likely explanation: for a
   silence-classified frame, the decoder evidently ignores nearly all of the other encoded parameters
   and just synthesizes a fixed low-level comfort-noise pattern from the classification field alone.
   Silence isn't a stronger test vehicle here, it's the wrong one -- it makes almost every bit
   semantically irrelevant to the output, independent of FEC protection.

**The working method**: use a voiced 200 Hz tone for `R` (so every parameter is actually exercised),
but compare **dB-magnitude spectra** (via FFT) instead of raw samples -- phase-invariant, and far
more sensitive across the dynamic range than a linear-magnitude spectral distance (which is dominated
by the fundamental peak). Protocol per bit: re-converge the decoder to `R`'s steady state (resend `R`
four times, discarding output), decode `R` with that one bit flipped, and compare its dB spectrum to
a calibrated baseline. The natural noise floor (8 repeats of unmodified `R`, same re-priming protocol)
is a tiny, perfectly deterministic 4-cycle pattern (`0.35, 0.10, 0.38, 0.00` dB, repeating exactly --
itself a real, minor residual of the same phase-tracking behavior, small enough here not to matter),
giving a clean noise floor of 0.38 dB.

**Result: exactly 2 of 144 wire bit positions -- 131 and 143 -- change the decoded spectrum when
flipped, by 22.18 dB, ~58x above the noise floor, with zero ambiguity anywhere else** (every other
position matches the noise-floor cycle exactly, to two decimal places). Bit 143 is the very last bit
of the 144-bit frame; bit 131 is 12 bits before it (byte 16, bit-value `0x10`, versus byte 17's LSB).

**Correction (caught on review before over-reading this): 2 is exactly the textbook-predicted count
here, not a shortfall.** This crate's own, independently-verified `bit_prioritization::
extract_fundamental_frequency_quantizer` (cross-checked against GopherTrunk's own implementation) is
`b_hat_0 = ((u[0]>>6)<<2) | ((u[7]>>1)&0b11)` -- two of the pitch quantizer's bits live in `u[7]`
(`c7`, the 7 raw/unprotected bits), at `c7`'s own internal bit-index 1 and 2. For a steady 200 Hz
voiced tone, flipping either pitch bit shifts every harmonic, which is exactly the 22 dB effect found.
The other 5 raw bits are low-order spectral-amplitude LSBs; for a clean sinusoid whose non-fundamental
bands are already near the amplitude floor, flipping those has no detectable effect on *this specific*
test signal -- independent of whether they're FEC-protected. So 131 and 143 are almost certainly
`c7`'s own two pitch-LSB positions on the wire -- a strong, named ground-truth constraint, not merely
"2 out of 7 found so far". Next step: use these two known positions to test which framing convention
(interleave direction, dibit order, codeword-bit-index direction, byte/bit order) places `c7`'s
bit-index-1 and bit-index-2 at exactly wire positions 131 and 143 -- zero additional chip time needed,
since this is a pure combinatorial check against already-published table data (see below).

**Tried, and a genuine methodological limit found**: re-ran the same oracle with a tone-plus-fixed-
noise test signal (`tone_noise` mode), hoping the added broadband content would activate whatever
the other ~5 raw bits (spectral-amplitude LSBs) control. Result: the noise floor itself jumps to
5-11 dB (versus the pure tone's 0.38 dB), and the two already-confirmed hits (131, 143) no longer
stand out at all -- both land squarely inside that same 5-11 dB range. This isn't a bug in the
harness: it reveals a real, sensible property of the chip's own synthesis -- voiced bands are
reconstructed deterministically (a continuous, phase-tracked sinusoid, hence the tiny sub-dB floor),
while unvoiced bands are synthesized from the decoder's own internal noise generator, which is
genuinely stochastic frame to frame even for byte-identical encoded parameters (the correct design
choice for natural-sounding comfort noise, but it means this decode-comparison oracle can only
cleanly probe parameters that voiced synthesis actually exercises). Any remaining unprotected bits
are therefore not resolvable this way; a different oracle (e.g. comparing long-run average energy
per critical band across many decodes, rather than a single decode's spectrum) would be needed.

**An 8-anchor pair-flip sweep for Hamming(15,11) codeword membership, and why its clean-negative
result is inconclusive, not a finding against Hamming being present**
(`examples/p25_ratet27_pairflip_anchor_sweep.rs`): picked 8 positions spread across the frame and,
for each, flipped it together with every one of the other 141 non-raw candidates, checking for a
dB-spectral change beyond a calibrated threshold -- a Hamming(15,11) codeword corrects only 1 error,
so two flips sharing one should be uncorrectable and change the output, while two flips sharing a
Golay(23,12) codeword (which corrects up to 3) or landing in different codewords entirely should
still show no effect. **Result: all 8 anchors x 141 partners (1128 pairs) came back clean, zero
detected effect.** Read carefully rather than as "no Hamming codewords exist here": this test has
the exact same inert-parameter blind spot as the single-bit oracle -- `c4-c6` (the next-lowest-
priority spectral-amplitude bits, per textbook IMBE) would show the same near-zero effect on a clean
tone that the other 5 raw bits did, whether or not a Hamming miscorrection actually occurred. The
pair threshold (5x the 0.38 dB tone-only noise floor, so 5.00 dB) would also miss any genuine but
modest 1-4 dB Hamming effect. This result belongs in the record as "inconclusive for Hamming
membership under a pure tone", not as evidence against a Hamming-coded structure being present --
and a full exhaustive C(142,2) pairing (~40 minutes of chip time) was deliberately *not* run given
this same blind spot would limit it too.

**A zero-chip-time convention search using {131, 143} as ground truth**
(`examples/p25_ratet27_c7_pitch_bit_convention_search.rs`): rather than spend more chip time, used
the two known `c7` pitch-bit wire positions as a hard constraint against every plausible framing
convention -- byte order x bit direction (this investigation's usual 4 hypotheses), Table 5-1's own
dibit convention (swapped or not), `TIA_INDEX`'s MSB-first convention (reversed or not), and whether
the chip's raw serial data is OTA-interleaved via Table 5-1 at all versus natural contiguous
codeword order (already ruled out for general Golay validity by the sliding-window scan, §13, but
cheap to also check here). **Result: a clean negative across all 24 systematically-tried
combinations** -- no single consistent convention places both `c7[1]` and `c7[2]` at exactly
`{131, 143}` (one near-miss noted for transparency: two *different*, mutually incompatible
sub-variants each land exactly on one of the two positions individually -- `dibit_swap=true,
reverse_bytes=false, lsb_first=false` gives `c7[2]=131`, and the same dibit_swap/reverse_bytes with
`lsb_first=true` gives `c7[1]=143` -- but no *single* convention produces both simultaneously, and
with 24 variants x 2 positions each, a couple of incidental individual matches are not surprising by
chance). This means the chip's real interleave (if Table 5-1 applies to its raw serial data at all)
differs from every tried convention, or genuinely isn't Table 5-1-based -- but `{131, 143}` stands as
a real, hard, reusable constraint for testing any future candidate table, a first for this whole
RATET(27) investigation.

## 15. The full exhaustive pair sweep: a decoder-state-history bug found and fixed, and a real breakthrough -- confirmed 3-way redundancy groups

Given the anchor sweep's inconclusive result (§14), Bruce asked directly to run the full exhaustive
C(142,2) = 10011-pair sweep despite its known inert-parameter blind spot, since even a clean negative
across the whole space would be informative, and any positive hit would be immediately decisive.

**First attempt: a real methodological bug found the hard way.** The first run (no per-hit
confirmation, 4x tone-only re-priming per test) came back with hundreds of "hits" clustering into
long runs of near-identical distance values across huge, unrelated-looking bit ranges -- not the
sparse, structured signal a real codeword-membership proof should produce. Adding a same-run
confirmation retest (re-converge, re-test the same pair immediately) collapsed this to 6 hits:
`(8, 92)`, `(8, 127)`, `(32, 127)`, `(68, 103)`, `(103, 127)`, `(128, 139)`.

**Second, deeper problem, found by manually verifying those 6 hits**
(`examples/p25_ratet27_pairflip_diagnose_hit.rs`): pair `(8, 93)` reproduced *identically* (byte-for-
byte PCM) across two fully independent fresh-boot runs of the diagnostic tool -- strong-looking
evidence of a real, deterministic effect. But `(8, 93)` never even registered as a hit in the full
sweep's own first pass. The two contexts differ in exactly one way: the diagnostic tool always primes
from a fresh connection with the same short, fixed sequence, while the sweep tests thousands of
different bit-pairs in sequence before reaching any given pair. This means the chip's decoder carries
state beyond what a short re-priming burst resets -- a long, varied test history leaves it in a
different residual state than a deliberate, short reset does, even though *that* residual state is
itself perfectly reproducible run-to-run (which is exactly why `(8, 93)` looked like real signal under
naive fresh-boot verification, and is a real trap: reproducibility across independent runs does not,
by itself, prove genuine content-level significance if the *test setup itself* has an unresolved
history-dependence).

**The fix (Bruce's suggestion): condition with digital silence before every single test, not just
re-prime with the tone.** Silence has no pitch/phase to track (§14's own finding, from ruling it out
as a *test signal*: it makes too many parameters irrelevant to serve as the flip target), but that
same property makes it an excellent *conditioning* input -- decoding many silence frames in a row
forces the decoder to a small, near-fixed state regardless of whatever came before, and priming with
the tone from that canonical starting point converges to the tone's steady state independent of prior
history. Re-testing `(8, 93)` with 20 silence-decodes-then-4-tone-primes before the comparison dropped
its distance to 0.52-1.12 dB, comfortably below threshold and consistent with the sweep's own
original "no effect" finding -- confirming the earlier "reproducible" result was a measurement
artifact of insufficient state reset, not real signal, and that the fix resolves it.

**Re-verifying the 6 original hits under proper conditioning found something real.** `(8, 92)` and
`(8, 127)` produce byte-for-byte *identical* decoded PCM (not just similar distance -- the exact same
samples). So does `(92, 127)`, which the unconditioned sweep had missed entirely as a hit (direct,
concrete proof the unconditioned sweep's results could not be trusted and needed re-running).
Flipping all three, `{8, 92, 127}`, together *also* produces that exact same PCM. Independently,
`(68, 103)`, `(103, 127)`, and `(68, 127)` all produce another shared identical PCM, and flipping
`{68, 103, 127}` together matches it too.

**Correction (per advisor review): this is not a 3-way majority vote -- IMBE has no repetition
code.** It is the signature of a **weight-3 codeword of a Hamming(15,11) block**. Hamming(15,11) has
minimum distance 3, so a 2-bit error is always "corrected" by the decoder rather than reported: for
three parity-check columns `h_a, h_b, h_c` with `h_a XOR h_b XOR h_c = 0` (a weight-3 codeword),
flipping any 2 of the 3 produces a syndrome equal to the third column, so the decoder "corrects" the
bit that was *not* flipped, always landing on `original XOR e_a XOR e_b XOR e_c` regardless of which
2 (or all 3) were actually flipped on the wire -- exactly the byte-identical-PCM pattern observed.
Golay(23,12), used for the other four protected blocks, has minimum distance 7 and cannot produce
this behavior for a 2-bit error, so this pins both triples to one of the three 15-bit Hamming blocks.
Since bit 127 is shared between both, and a wire bit belongs to exactly one FEC block, **all five
bits {8, 68, 92, 103, 127} must live in the same 15-bit Hamming block** -- a hard constraint on any
future candidate interleave table. Cross-pairs between the two groups (`(8, 68)`, `(92, 103)`) show
no effect, confirming the two groups are otherwise distinct codewords within that one block, not one
larger connected structure.

**Both triples independently re-confirmed later the same night via a stronger test (§17): one
flip-decode per FRESH process** (not sequential decodes on one connection, which by that point in
the investigation was known to be an unreliable protocol -- see the encoder-feedback finding below).
`flip{92,127}`, `flip{8,127}`, `flip{8,92}`, and `flip{8,92,127}` each run in their own fresh process
produced byte-for-byte identical PCM (checksum `e5b52df714b6e2ad`) with no dB threshold or baseline-
in-connection involved at all. The same was done for `{68,103,127}` (checksum `b63f63e98f205c13`,
all four combinations identical). This is the strongest evidence for either triple in the whole
investigation, and it also revealed a new, better oracle: **exact PCM/checksum equality between two
fresh-process single-flip-decodes is a threshold-free, drift-free membership test** -- no dB metric,
no in-connection baseline, and no exposure to the busy-history degradation problem, since each test
is exactly one decode per fresh connection.

**Consequence: the full exhaustive sweep needed re-running with proper silence conditioning**
(`examples/p25_ratet27_pairflip_full_sweep.rs`, updated in place), since the unconditioned version
demonstrably missed at least one real structural relationship. Two more tuning problems were found
and fixed before a trustworthy run launched:

1. **Baseline/conditioning mismatch.** An early version of the conditioned sweep captured its own
   baseline via the *old* plain tone-only re-priming method while every subsequent test used full
   silence conditioning -- comparing two different decoder states. This inflated the measured noise
   floor to ~21 dB, which would have set a 5x-floor threshold (~106 dB) far above even the confirmed
   real effects (~10-14 dB), silently missing everything. Fixed by conditioning the baseline capture
   identically to every other test.

2. **A genuinely wide, heavy-tailed noise floor, not a bug.** Even after that fix, 20 independent
   conditioned samples of "decode unmodified R" spread smoothly from 0.5 dB to over 20 dB (not a
   rare-outlier pattern -- a real, continuous spread). Neither increasing the tone-priming count
   (4->20) nor the silence-conditioning count (20->80) meaningfully tightened this, ruling out
   "insufficient relock time" as the cause; a Hann window (to reduce phase-dependent spectral
   leakage, the other likely explanation) changed the distribution's shape but didn't clearly help
   either, and was reverted rather than chased further. This means a single dB-spectral-distance
   measurement cannot, by itself, reliably separate a weak real effect from noise here -- the
   previously-confirmed but weak `(128, 139)` pair (5.56 dB) sits within the same range as ordinary
   noise spikes. **Fix**: use the *median* of 20 calibration samples as a robust floor estimate (not
   swayed by the tail), a modest additive threshold margin above it (low enough to admit real
   10+ dB effects into confirmation, accepting that this also admits much of the noise tail), and a
   mandatory independent-redraw confirmation retest for every candidate -- a genuine effect
   reproduces reliably, noise usually doesn't land above threshold twice. This is an accepted,
   documented limitation for weak effects specifically, not a blocker for the strong ones this sweep
   is actually aimed at finding.

**A striking, highly regular preliminary pattern**, seen in a bounded sanity check before launching
the full run: for anchor bit 0, confirmed hits land at `(0,17) (0,18) (0,19)`, `(0,43) (0,44) (0,45)`,
`(0,69) (0,70) (0,71)`, `(0,95) (0,96) (0,97)`, `(0,121) (0,122) (0,123)` -- five groups of exactly 3
*consecutive* wire positions, each group spaced exactly 26 positions apart, with the same +26 pattern
repeating (from a different starting offset) for anchor bit 1.

**The completed run found 1147 confirmed hits across all 142 candidates (one single connected
component, not isolated small groups) -- and then failed its own sanity check.** Checking whether
the two already-confirmed real triples appeared in this dataset, `(8, 92)`/`(8, 127)`/`(92, 127)`
and `(68, 103)`/`(68, 127)`/`(103, 127)`, found only one match at all: `(68, 103)` appears, but as a
**rejected, unconfirmed** candidate (`first=19.19 dB, retest=5.25 dB`) -- the sweep's own
confirmation step threw out a pair independently verified, multiple times, by direct targeted
testing to be real. A result set that rejects a known-real effect while accepting 1147 others cannot
be trusted as a whole; the 1147-pair dataset (including the period-26 pattern) is **not** treated as
reliable evidence of real structure, despite its superficially compelling regularity.

**Root cause, per Bruce: AMBE's encoder incorporates feedback from previous decoding.** This session
had already independently found and documented (during the AMBE+2 real-speech validation, and again
during the DVSI pipelined-protocol investigation) that this chip's own real client software runs a
genuine multi-frame lookahead/delay pipeline -- Bruce's comment connects that same mechanism to this
specific failure. Verified directly and reproducibly (`examples/p25_ratet27_pairflip_diagnose_hit.rs`,
`condition()`'s own doc comment has the full detail): a single prior flip-decode test already
degrades the *next* test's measured `(68, 103)` effect from 12.41 dB to 5.26 dB, even with the same
conditioning re-applied before it; 100 busy prior flip-decodes plus that same conditioning still only
give 4.03 dB. Appending a second, smaller pass -- 3 more silence decodes, then a few more tone
decodes -- after the main conditioning recovers this specific degraded case back to 11.03 dB,
reproducibly (identical across repeated fresh-process runs). **But this is not a general fix**:
applying that same second pass unconditionally to *every* conditioning call, including an already-
fresh one, was tried and made things worse -- a known real effect measured 8.56 dB while a known
no-effect pair measured 9.19 dB under otherwise-identical conditions (the ranking flipped). Both
variants are individually fully deterministic; this is real, parameter-sensitive state-dependence,
not measurement noise.

**Where this leaves the RATET(27) investigation**: short, targeted, few-tests-per-connection
verification (used throughout §§13-15's actual confirmed findings -- the two redundancy triples, the
two unprotected `c7` bits) remains reliable, since a connection's history stays short and controlled.
A long, exhaustive, thousands-of-tests-on-one-connection sweep does not currently have a trustworthy
conditioning recipe, since no single procedure has been found that produces a canonical state
regardless of a connection's prior history -- confirmed, not just suspected, by the direct
reproduction of degradation-then-partial-recovery above. This is an honest, open limitation of the
exhaustive-sweep methodology specifically, not a retraction of anything already confirmed by targeted
testing. Recovering a fully general, history-independent reset procedure -- or abandoning long-
running sweeps in favor of many short, independently-conditioned connections (fresh RATEP
configuration per test, accepting the added per-test connection-setup cost) -- is the next step for
whoever continues this investigation.

## 16. Sweeping the busy-history recovery frame count, per Bruce's suggestion -- no simple rule found, but a clean protocol conclusion

Per Bruce: "You might try sweeping the number of silence frames to determine how many are
consistently effective," refined by "There need to be enough silence frames to prime both the
encoder and decoder. That may be 6 rather than three." The §15 recovery pass (3 silence decodes +
`PRIME_REPEATS` tone decodes, appended after normal conditioning) was tested only at N=3; this
follows up by varying N.

**Methodological trap found first**: an initial sweep tool (`p25_ratet27_recovery_frame_count_sweep.rs`)
ran multiple (busy-history, recovery, measure) trials back-to-back within *one* continuous
connection. This produced high variance *within* a single N's own repeated trials (e.g. N=0:
`[4.28, 4.80, 10.51]` dB) -- proof that sequential trials sharing one connection do not share a
common baseline; each trial's starting state depends on all the prior trials in that same session,
not just the fixed busy-history replay before it. Confirms again (see §15) that only a genuinely
fresh process/connection per measurement is a trustworthy protocol here.

**Fixed with `p25_ratet27_recovery_n_single_trial.rs`**: exactly one measurement per fresh process,
intended to be invoked repeatedly from a bash loop. Getting this tool to reproduce the known-good
11.03 dB value at N=3 took two real bug fixes, both instructive about how exact-sequence-sensitive
this chip's state is:

1. Its `condition()` was missing the encoder-silence pass (`send_speech_get_channel` over live
   silent PCM) that `p25_ratet27_pairflip_diagnose_hit.rs`'s `condition()` always included --
   without it, N=3 measured 3.81-6.13 dB, not 11.03.
2. `diagnose_hit.rs`'s actual sequence runs the fixed-seed busy history in **two separate rounds of
   100 flips each**, with a `condition()` call and an (unused-for-the-final-result) intermediate
   measurement between them, and the LCG state **continues** across both rounds rather than
   resetting -- the 11.03 dB value was measured after 200 cumulative busy flips, not 100. Matching
   this exactly was what finally reproduced 11.03 dB precisely.

**Swept N in {0, 1, 2, 3, 4, 5, 6, 8, 10}, both `mode=decode_only` and `mode=both` (extra live
encoder-silence frames during the recovery pass itself), 2 fresh-process trials each, target pair
`(68, 103)`:**

| N  | distance (dB), both modes identical |
|----|----|
| 0  | 5.64 |
| 1  | 4.11 |
| 2  | 6.36 |
| 3  | 11.03 |
| 4  | 5.06 |
| 5  | 5.28 |
| 6  | 9.88 |
| 8  | 26.79 |
| 10 | 4.39 |

Three findings, all solid:

- **Fresh-process determinism is exact**: both trials at every single N produced byte-identical
  distances (and, on spot-check, identical PCM). This reconfirms fresh-process single-measurement as
  fully deterministic and reliable.
- **`mode` makes zero difference at every N.** Extra live encoder-silence frames during the small
  recovery pass changed nothing, because `condition()` (run right before the recovery pass in every
  trial) already does 20 rounds of encoder-silence priming -- Bruce's "prime both paths" requirement
  is already satisfied by `condition()` itself, so the recovery pass only ever needed to address the
  decoder side.
- **No N gives consistent recovery.** The relationship is non-monotonic (5.64, 4.11, 6.36, 11.03,
  5.06, 5.28, 9.88, 26.79, 4.39) with no threshold-like "N or more works" structure, and N=8's 26.79
  dB *exceeds* the undisturbed fresh-boot reference (~12.41 dB) -- proof that this distance-from-a-
  fixed-baseline metric, after 200 busy flips plus N recovery frames, is measuring some mix of
  decoder-state drift and the flip effect, not a clean "recovered vs not" signal. There is no simple
  N to recommend.

**Conclusion for the exhaustive-sweep question**: don't chase a general busy-history recovery
recipe further. The only protocol shown reliable all night is a single measurement from a fresh
process; that is the actual answer to "how do you get a trustworthy reading on this chip," not a
particular recovery frame count layered on top of a long, busy connection.

## 17. Weight-3 Hamming-codeword interleave search -- a 4-of-5 near miss, cleanly falsified

Following §15's correction (weight-3 Hamming(15,11) codewords, not majority vote), the confirmed
constraint "{8, 68, 92, 103, 127} all share one 15-bit Hamming block" is far more discriminating than
§14's 2-bit `c7`-pitch-LSB check (which found no matching convention at all). This is a zero-chip-time
search: `p25_ratet27_hamming_block_convention_search.rs` re-uses the same convention space (TIA-
102.BAAA-A Table 5-1 OTA-interleave hypotheses -- dibit-row swap, index direction, byte order, bit
direction -- plus natural contiguous-codeword order) and checks whether any convention places all
five confirmed bits in the same `TIA_BLOCK` (4, 5, or 6).

**No convention places all five together.** But one convention -- `dibit_swap=true,
reverse_bytes=true, lsb_first=false` (index-direction either way, since it only changes offset within
a block) -- places **four of five** (68, 92, 103, 127) in the same block (`TIA_BLOCK`=6), with only
bit 8 landing elsewhere (`TIA_BLOCK`=3, a 23-bit Golay block).

**Falsified directly, decisively, using the new fresh-process checksum oracle (§15's `e5b5...`/
`b63f...` re-confirmation, generalized here as `p25_ratet27_hamming_block_falsification_test.rs`)**:
if bit 8 really sat in a Golay(23,12) block (minimum distance 7), a 2-bit error there could never be
"corrected" onto a third bit the way Hamming(15,11) allows, so `flip{8,92}` and `flip{8,127}` should
each land back near the unmodified baseline while `flip{92,127}` alone shows the large deviation.
Instead, run from independent fresh processes (not sequential same-connection decodes, which by this
point in the investigation was known to be unreliable -- exactly the failure mode this test was
designed to rule out), `flip{92,127}`, `flip{8,92}`, `flip{8,127}`, and `flip{8,92,127}` all produced
byte-identical PCM (checksum `e5b52df714b6e2ad`). The triple is genuinely real (not a same-connection
artifact), and this specific convention is dead.

**What survives**: the hard constraint itself -- whatever the real interleave turns out to be, bits
{8, 92, 127} share one Hamming block and {68, 103, 127} share one (the same one, since 127 is
common), so all five of {8, 68, 92, 103, 127} are in a single 15-bit block. An anchor sweep using the
same fresh-process checksum oracle (bit 127 against all 143 other wire positions, one fresh process
per candidate) was launched to find the block's full 15-bit membership and complete weight-3-codeword
structure directly from the chip, without needing a candidate interleave table at all -- see §18 for
its result.

## 18. Complete membership of one full Hamming(15,11) FEC block, found by pure chip-oracle experiment

The 4-of-5 near miss in §17 motivated a direct, assumption-free sweep: hold bit 127 flipped, and
additionally flip every one of the other 143 wire positions in turn (`flip{127, b}` for
`b in 0..144, b != 127`), one fresh process per `b`, comparing the resulting PCM checksum against the
unmodified-baseline checksum. **16 of the 143 candidates produced a non-baseline checksum -- 2 of
which are the already-known unprotected `c7` pitch bits, 131 and 143** (per §14; confirmed here as
`flip{131}` and `flip{143}` alone reproduce the exact same checksum as `flip{127,131}`/`flip{127,143}`,
validating that an out-of-block bit paired with 127 shows its own independent effect, not a null one).
That leaves **exactly 14 real block members** -- the expected count of "other members" a 15-bit
Hamming(15,11) block should have relative to one anchor. This gives a complete, empirically-derived
membership map for one entire FEC block, with zero assumptions about interleave, byte order, or any
candidate table:

**{8, 20, 32, 44, 56, 68, 80, 92, 103, 104, 115, 116, 127, 128, 139}** (15 positions).

The 14 non-anchor members grouped into 4 distinct checksum classes:

| checksum (dB effect) | members |
|---|---|
| `e5b52df7...` (10.62 dB) | 8, 92 |
| `55ce58d8...` (13.95 dB) | 20, 32 |
| `b63f63e9...` (12.41 dB) | 44, 56, 68, 80, 103, 104, 115, 116 |
| `3985aaa9...` (6.75 dB) | 128, 139 |

**Two new triples, `{20, 32, 127}` and `{128, 139, 127}`, were independently confirmed with the same
full rigor as §15's original two** (`p25_ratet27_hamming_block_falsification_test.rs`, fresh process
per test): all four combinations -- both pairs, both singles-with-127, and the full triple -- produce
byte-identical PCM within each group. Bit 32, previously seen only as an unverified hint from the
disqualified full-exhaustive sweep (§15) and dropped in §17, is now independently confirmed real --
it pairs with 20, not with 127 alone or with the original {8,92,103} group.

**The 8-member class is not one pair -- it is 4 distinct genuine codewords that happen to be
audibly indistinguishable on this test signal, and all 4 were found directly.** Testing
`flip{56, 80}`, `flip{56, 115}`, `flip{56, 116}` (fresh process each) found exactly one match,
`flip{56, 80}` -> the `b63f...` checksum; testing the resulting elimination pair directly,
`flip{115, 116}`, gave `b63f...` too. Both `{127, 56, 80}` and `{115, 116, 127}` were then confirmed
with the same full 2-of-3/3-of-3 rigor as every other triple in this investigation. This resolves the
8-member class completely into 4 real pairs: **{68, 103}, {44, 104}, {56, 80}, {115, 116}** -- all
four, combined with anchor 127, produce byte-identical PCM despite being 4 genuinely distinct
codewords. This directly parallels §14's `c7` finding (5 of 7 raw bits showed no audible effect on
this same test signal): most of this Hamming block's 11 information bits are apparently inaudible or
carry a redundant/overlapping effect for a steady tone, and only the few responsible for the
10.62/13.95/12.41/6.75 dB effects are perceptible with this specific test material.

**A striking arithmetic pattern, found after the fact by inspecting the 15 confirmed positions**: 11
of them -- {8, 20, 32, 44, 56, 68, 80, 92, 104, 116, 128} -- are all congruent to 8 (mod 12); the
other 4 -- {103, 115, 127, 139} -- are all congruent to 7 (mod 12). Since `144 = 12 x 12`, this is
consistent with the chip's raw serial data being organized as a 12-column grid (wire position
`= 12*row + column`, `row` and `column` both 0..11), with this Hamming(15,11) block drawing its 11
higher-weight bits from column 8 (rows 0-10, missing row 11 = position 140) and its remaining 4 bits
from column 7 (rows 8-11). The two confirmed unprotected `c7` pitch bits, 131 and 143, are also both
congruent to 11 (mod 12) -- consistent with column 11, rows 10-11. This "12-column interleave"
hypothesis is a genuinely new, well-defined, falsifiable candidate structure, unrelated to the
TIA-102.BAAA-A Table 5-1 conventions tried and ruled out in §17. An anchor sweep on a column-9
candidate bit (e.g. anchor 9) was launched to test it directly: if a second Hamming block's ~14
members also fall on a clean stride-12 pattern, that would be strong independent confirmation of both
the block-2 membership and the whole 12-column layout -- see the addendum below (or a future section)
for its result.

**This is, along with the confirmed unprotected `c7` pitch bits (§14), the most complete real
structural result this investigation has produced for RATET(27)**: one full 15-bit FEC block's exact
wire-bit membership (with all 7 of its internal weight-3 codewords involving anchor 127 accounted
for: 4 giving one audible signature class, plus the {8,92}/{20,32}/{128,139} classes -- 3+4=7 pairs
among the 14 non-anchor members, matching the theoretical count exactly), obtained by pure black-box
chip experimentation with no assumptions about interleave, byte order, or any candidate specification
table, plus a new candidate global interleave structure (12-column grid) actively being tested. It
remains an open question which of the chip's 3 total Hamming(15,11) blocks this is, and where the
other 2 Hamming blocks and 4 Golay(23,12) blocks land among the remaining 129 wire positions --
continuing the anchor-sweep technique on other columns (per the stride-12 hypothesis) is the natural
next step for whoever continues this investigation.

## 19. A second Hamming block predicted by the transform, hidden from the pure-tone oracle, confirmed with a richer test signal

§18's stride-12 observation was extended (analytically, zero chip time) into a specific transform
hypothesis: the wire (transmitted) format is a 12x12 block interleaver over a natural bit order where
the wire position `m` corresponds to natural position `n(m) = 12*(m mod 12) + (m div 12)` (a matrix
transpose: write the natural stream down 12 columns, read it back out across 12 rows). Applying this
transform to the confirmed 15-member block's own wire positions gives natural positions
**92 through 106, exactly 15 consecutive values** -- strong support for the transform itself, since
only this specific write-by-column/read-by-row structure would keep a real FEC block's bits
contiguous in natural order after this kind of interleaving.

Extending this with the further guess that the 8 FEC sub-blocks are simply concatenated in natural
order as `u0..u3` (Golay x4, 23 bits each, natural 0-91), `u4..u6` (Hamming x3, 15 bits each, natural
92-136), `u7` (raw, 7 bits, natural 137-143) -- matching the confirmed block to `u4`, and noting this
is the *only* way to partition 144 into `[23,23,23,23,15,15,15,7]` consistent with `u4` at 92-106 and
`c7` at 137-143 (the latter independently confirmed: `natural(131)=142`, `natural(143)=143`, both
inside 137-143) -- predicts the *next* Hamming block `u5` at natural 107-121, transforming back to
wire positions **{9, 10, 21, 22, 33, 45, 57, 69, 81, 93, 105, 117, 129, 140, 141}**.

**First attempt looked like a clean falsification, and was not.** An anchor sweep on wire bit 9 with
the usual 200 Hz sine (one fresh process per candidate, same technique as §18) found no partners at
all among the predicted `u5` set -- only `flip{9,131}` and `flip{9,143}` showed any effect (the
already-known unprotected `c7` bits, which register regardless of what else is flipped). Follow-up
pairwise tests within the predicted set (`{9,21}`, `{9,10}`, `{21,22}`, `{10,22}`, `{9,33}`,
`{9,141}`), a 4-flip (`{9,21,33,45}`), and a 5-flip (`{9,21,33,45,57}`) were *all* null too.

**But an all-null result under every flip pattern does not discriminate between competing
explanations, and this session had already shown why**: §18's own confirmed `u4` block has null-
under-pairwise-flip members too (`{44,80}`, `{44,115}`, `{44,116}`), simply because those specific
codewords happen to change something the 200 Hz sine doesn't render (a pure tone puts energy in one
harmonic; most of IMBE's per-band amplitude bits sit at the noise floor for it, so their codewords
are audibly inert). An all-null column-9 sweep is equally consistent with "Hamming block with 7
tone-inaudible codewords," "Golay block with a tone-inaudible weight-7 codeword," or "column 9 spans
multiple blocks" -- the oracle itself was blind, not necessarily the hypothesis wrong.

**Fix: switch to a harmonic-rich reference signal.** Added a 200 Hz sawtooth alternative (same
fundamental, so pitch/voicing structure carries over, but energy spread across many harmonics
instead of one) to `p25_ratet27_hamming_block_falsification_test.rs` (`--signal sawtooth`). Verified
determinism first (two fresh processes, `flip{0,0}`, identical checksums) and re-verified the
confirmed `u4` triple `{8,92,127}` still holds (all four combinations byte-identical under the new
signal). Then re-tested the predicted `u5` pairs: **`flip{9,21}` and `flip{9,10}` produce
byte-identical PCM, while `flip{21,22}` and `flip{10,22}` each produce their own distinct non-null
effect** -- exactly the structure the pure tone was blind to. **This confirms the transform and the
natural-order prediction: `u5` is real.** A full anchor-9 sweep with the sawtooth signal was launched
to map its complete membership the same way `u4` was mapped in §18 -- see the next section for its
result.

Separately (zero chip time): layering the same 12x12 transform onto Table 5-1's own row numbering,
in both orders (transform-then-byte-convention and byte-convention-then-transform) across all 4
byte/bit-order hypotheses, still does not place the confirmed 15-member block in one Hamming slot
together with both known `c7` bits in block 7 -- reinforcing that this chip's host-interface
"channel" packing is a genuinely different, proprietary format, not Table 5-1 plus a simple
transpose layer.

**Lesson for the rest of this investigation**: a null result from the 200 Hz sine test signal is not
reliable evidence of "no relationship" -- it only shows no relationship *audible on a pure tone*. Any
future anchor sweep that comes back all-null should be re-run with the sawtooth signal before being
treated as a real negative.

Separately (zero chip time): layering the same 12x12 transform onto Table 5-1's own row numbering,
in both orders (transform-then-byte-convention and byte-convention-then-transform) across all 4
byte/bit-order hypotheses, still does not place the confirmed 15-member block in one Hamming slot
together with both known `c7` bits in block 7 -- reinforcing that this chip's host-interface
"channel" packing is a genuinely different, proprietary format, not Table 5-1 plus a simple
transpose layer.

## 20. Second Hamming block mapped, and the complete 7-bit `c7` raw block found -- the transform validated on 3 of 8 sub-blocks with zero mismatches

The full anchor-9 sweep, re-run with the sawtooth signal, resolved cleanly into exactly the
predicted structure. 143 candidates grouped into: the dominant null class (122 members, checksum
`b65c2ca8...`), **7 matched pairs** (14 positions), and **7 unrepeated singles** (7 positions) --
21 non-anchor positions total, splitting neatly into two different phenomena:

**The 7 matched pairs give `u5`'s complete membership, exactly matching the transform's prediction
with zero mismatches:**

| checksum | members |
|---|---|
| `1f101d45...` | 10, 21 |
| `defe0d99...` | 22, 93 |
| `bdf17b41...` | 33, 117 |
| `7d4a72ec...` | 45, 57 |
| `0810d1b9...` | 69, 129 |
| `cd9c8c56...` | 81, 105 |
| `78ed0fe7...` | 140, 141 |

Combined with anchor 9: **u5 = {9, 10, 21, 22, 33, 45, 57, 69, 81, 93, 105, 117, 129, 140, 141}** --
identical, position for position, to the wire set §19 derived analytically from the transform
(`natural(m) = 12*(m mod 12) + (m div 12)` applied to natural range 107-121). Zero mismatches.

**The 7 unrepeated singles are the complete 7-bit unprotected `c7` block, not just the 2 pitch bits
found in §14.** `{71, 83, 95, 107, 119, 131, 143}` each showed their own distinct, non-repeating
effect (consistent with raw/unmodulated bits: each shows its own independent signature regardless of
what else is flipped, rather than pairing up like FEC-protected bits do). These 7 positions are all
congruent to 11 (mod 12), stride-12, and map under the transform to **natural positions 137 through
143 -- exactly 7 consecutive values**, matching §19's `c7`-placement prediction exactly. The other 5
of these 7 (everything except the already-known 131, 143) were invisible to the original single-bit
oracle (§14) for the same reason column 9's Hamming partners were invisible to the anchor-9 sine
sweep: a pure 200 Hz tone doesn't render whatever these bits control (very likely low-order spectral-
amplitude LSBs, per §14's original textbook-count reasoning -- they were never really "inert," just
inaudible on that specific signal).

**The transform is now validated on 3 of 8 sub-blocks (`u4`, `u5`, `c7`) with zero mismatches
between prediction and direct chip measurement.** The remaining 5 sub-blocks (`u6`, and 4 Golay
blocks `g0..g3`) have fully determined predicted wire-position sets from the same transform (natural
ranges 122-136 for `u6`, and 0-22/23-45/46-68/69-91 for `g0..g3`):

- predicted `u6` = `{11, 23, 34, 35, 46, 47, 58, 59, 70, 82, 94, 106, 118, 130, 142}`
- predicted `g0` = `{0, 1, 12, 13, 24, 25, 36, 37, 48, 49, 60, 61, 72, 73, 84, 85, 96, 97, 108, 109, 120, 121, 132}`
  (and `g1`/`g2`/`g3` follow the same pattern shifted by natural offsets 23/46/69)

A quick spot-check of 4 predicted `u6` pairs (`{11,23}`, `{34,35}`, `{11,35}`, `{23,34}`, sawtooth
signal, fresh process each) found all 4 non-null with distinct checksums -- consistent with `u6`
being real (not yet full-membership-mapped; a complete anchor-11 sweep was launched the same way as
`u4`/`u5` to confirm it fully -- see the next section for its result).

## 21. Third Hamming block confirmed with zero mismatches, and all 4 Golay-block boundaries confirmed directly

The anchor-11 sawtooth sweep (predicted `u6`, per §20's transform) resolved into exactly the
predicted structure, same as `u4` and `u5`: 7 matched pairs plus the same 7 `c7` singles.

| checksum | members |
|---|---|
| `64fc65ff...` | 23, 118 |
| `4aaab9b1...` | 34, 94 |
| `5c0a77b5...` | 35, 82 |
| `5d26390b...` | 46, 70 |
| `ca72415a...` | 47, 59 |
| `72f4ab71...` | 58, 130 |
| `d58854aa...` | 106, 142 |

Combined with anchor 11: **u6 = {11, 23, 34, 35, 46, 47, 58, 59, 70, 82, 94, 106, 118, 130, 142}**
-- identical, position for position, to §20's transform prediction (natural range 122-136). Zero
mismatches, the third Hamming block in a row.

**A Golay(23,12) block was then directly confirmed too**, using the predicted `g0` membership
(`{0, 1, 12, 13, 24, 25, 36, 37, 48, 49, 60, 61, 72, 73, 84, 85, 96, 97, 108, 109, 120, 121, 132}`,
natural range 0-22): a 3-bit in-block flip (`{0,1,12}`) is null (Golay's minimum distance of 7
corrects any error of weight <=3 perfectly, exactly reproducing the original), while two *different*
4-bit in-block flips (`{0,1,12,13}` and `{0,1,24,25}`) each produce their own distinct non-null
effect -- exactly the Golay signature predicted in §18's original advisor review, now confirmed
directly against the chip for the first time this entire investigation.

**All 4 Golay-block boundaries were then confirmed directly, per advisor review** (a single in-block
Golay signature isn't proof of the *exact* predicted membership, since a null 3-flip is also what
three bits scattered across three different blocks would produce). For each adjacent pair of
predicted Golay blocks, one 4-flip fully inside the first block was compared against one 4-flip
straddling the boundary (2 bits from each side): `g0`/`g1` (`{0,1,12,13}` non-null vs `{0,1,2,3}`
null), `g1`/`g2` (`{2,3,14,15}` non-null vs `{2,3,4,5}` null), `g2`/`g3` (`{4,5,16,17}` non-null vs
`{4,5,6,7}` null) -- every straddle is null (two independent 2-bit errors, each within its own
Golay block's 3-error correction radius, each self-corrects back to the original) while every
in-block 4-flip is non-null, confirming each boundary exactly. `g3`'s own in-block signature
(`{6,7,18,19}`) was null on the first three subsets tried (`{6,7,18,19}`, `{6,7,30,31}`,
`{6,7,42,43}`, `{18,19,30,31}`) -- consistent with §18's already-established finding that many real
codewords are inaudible on any given test signal, not evidence against `g3` -- and a wider-spread
subset (`{7,55,79,113}`, matching `{6,7,113,114}`'s checksum) confirmed a real, audible `g3`
codeword. All 4 Golay blocks are now structurally confirmed, completing direct verification of
every one of the 8 FEC sub-blocks' existence and boundaries (4 fully bit-mapped, 4 boundary-and-
signature confirmed).

**Primary-source confirmation, found while researching a separate question from Bruce about whether
this chip supports a voice/data split mode (it does not -- see the answer recorded below)**: DVSI's
own AMBE-3000R Vocoder Chip Users Manual (Section 6.9, CHAND field description) states directly:
*"Chand[0] contains the bits which are most sensitive to bit errors... Chand[(Bits-1)/8] contain the
bits which are least sensitive to bit errors."* This is DVSI's own documented design principle
matching, exactly, the natural-order assumption this investigation inferred empirically (4 Golay
blocks first -- Golay is the strongest protection, minimum distance 7 -- then 3 Hamming blocks --
minimum distance 3, weaker -- then the unprotected raw `c7` bits last). What was an inferred
convention that happened to fit the data is now a documented DVSI design principle independently
corroborating it.

**Where this leaves the RATET(27) wire format**: of the 8 total FEC sub-blocks, 4 are now fully,
exactly mapped by direct chip experiment (`u4`, `u5`, `u6`, `c7` -- 52 of 144 wire positions with
zero prediction errors across all of them), and all 4 Golay blocks (`g0`-`g3`, 92 more positions)
have their exact predicted boundaries directly confirmed via the in-block-vs-straddle 4-flip test
above, though their individual bit-for-bit membership (unlike the Hamming blocks) hasn't been
walked position-by-position the way an anchor sweep would. The transform itself: **wire position
`m` corresponds to natural (pre-interleave) position `12*(m mod 12) + (m div 12)`**, with the
natural bit stream simply being the 8 FEC sub-blocks concatenated in decreasing-protection order
(`g0..g3` at natural 0-91, `u4..u6` at natural 92-136, `c7` at natural 137-143) -- a full,
falsifiable, and now extensively chip-verified model of this chip's real host-interface wire format,
found entirely by black-box experimentation with zero access to DVSI's proprietary interleave
specification.

**One open question checked and left genuinely unresolved**: §19 flagged that under the transform,
`131`->natural 142 and `143`->natural 143 (`c7` offsets 5 and 6), which doesn't cleanly match this
crate's own `extract_fundamental_frequency_quantizer` formula (`(u[7]>>1)&0b11`, wanting `c7` bits 1
and 2) under either an MSB-first or LSB-first storage convention -- under MSB-first, that formula
would instead want offsets 4 and 5 (wire 119 and 131). Tested directly: `flip{107}`, `flip{119}`,
`flip{131}`, `flip{143}` each alone (sawtooth signal) all show their own distinct, roughly
comparable non-null effect (4.5-5.7 dB range) -- no clear "two strong pitch-class bits vs two weak
amplitude-class bits" split emerged that would discriminate which pair is really the pitch field
under this chip's actual bit-index convention. This remains an open, unresolved constraint on any
final decoder for this chip's `c7` field; the original finding that only `131`/`143` (not the other
5) show any effect under a *pure tone* still stands as the most specific evidence available, but
doesn't by itself resolve the formula's exact bit-index convention.

**A capability found but not used tonight, for whoever continues this**: the manual documents
`CHAND4` (field ID `0x17`), a soft-decision decode mode (4-bit confidence values, 2 per byte,
instead of hard 0/1 bits). A future investigator could feed maximal-uncertainty ("don't know")
soft values to every bit outside the block currently under test, which may suppress cross-block
miscorrection noise entirely rather than needing to reason about it after the fact -- not tried
this session, but a real, documented feature worth exploring.

**Answer to Bruce's question: does this chip support a "data mode" or voice/data split, separate
from the vocoder's own compressed voice bits?** No. A careful read of both DVSI's USB-3000 Family
manual and the full AMBE-3000R Vocoder Chip Users Manual found: only 3 packet types exist
(`CONTROL`, `CHANNEL`, `SPEECH`); the channel packet's only data-carrying fields are `CHAND`/`CHAND4`
(compressed voice bits, hard/soft-decision), `SAMPLES`, `CMODE`, and `TONE`; and the encoder/decoder
feature-flag words (`ECMODE_IN`, `DCMODE_IN`/`DCMODE_OUT`) are entirely audio-processing toggles
(noise suppression, echo cancellation, companding, DTX/silence detection, tone detect/send, frame
repeat, comfort noise) with no data-channel concept anywhere. Whatever "data" a real over-the-air
protocol carries alongside voice (P25's embedded Link Control/Low Speed Data, D-STAR's slow-data
field) is added entirely by the surrounding radio/modem framing, external to the vocoder chip -- the
chip's own channel bitstream is 100% voice+FEC, confirming there is no host-exposed pass-through path
that would have let this investigation bypass vocoder synthesis directly (the sawtooth-signal fix in
§19/§20 remains the real, practical workaround for that problem).

## 22. Closing the loop: the original 6-hit pair sweep (task `b6tc68au3`, §15) cross-validated against the now-complete block map

Per a direct request to check on this task: it completed hours before this session's §17-21 work
even began (833.2s runtime, exit code 0, output already on disk), and its 6 confirmed hits --
`(8,92)`, `(8,127)`, `(32,127)`, `(68,103)`, `(103,127)`, `(128,139)` -- are exactly what §15 already
documented and analyzed at the time. That analysis is not being redone here; what's new is checking
those 6 pairs against the fully-mapped 8-block structure §§18-21 established using a completely
different, more reliable methodology (fresh-process checksum equality instead of dB-threshold
sweeping on one long-lived connection).

**Every one of the 6 hits falls entirely within `u4`'s exact confirmed 15-member set**
(`{8, 20, 32, 44, 56, 68, 80, 92, 103, 104, 115, 116, 127, 128, 139}`) -- both positions of all 6
pairs are `u4` members, with zero exceptions. This is a clean, independent cross-validation: a
dataset collected hours earlier, with a since-shown-unreliable methodology (dB-threshold, no
sawtooth signal, single long connection), landed 100% inside the block this session went on to map
completely and independently through a different technique.

**It also retroactively explains this session's central obstacle.** The old sweep tested all
C(142,2) pairs across the *entire* 144-bit frame but found real effects *only* within `u4` -- not
because the other 7 sub-blocks (3 more Hamming blocks' worth of pairs, 4 Golay blocks' worth, and
`c7`'s own raw bits) don't exist or don't matter, but because none of their codewords happened to be
audible on the plain 200 Hz sine test signal that sweep used. This is precisely the "pure tone is
blind to most of the wire format" problem diagnosed and fixed in §§19-20 with the sawtooth signal,
visible in hindsight all the way back in this much earlier dataset. `u4` was simply the one block
whose confirmed codewords all happened to be audible on a sine tone -- the lucky block, not a
special one -- which is exactly what let this whole investigation get started in the first place
(§15) before the real, larger structure came into view (§§17-21).

## 23. The labeling question is provably unsolvable by relational testing alone -- pivoting to direct chip-frame sampling resolves 7 of 8 FEC blocks completely, with a real software duplicate now committed

Following the `/goal` directive to continue until the RATET(27) format is fully known and duplicated
in software, this session attempted to resolve the one remaining open question from §21: given a
block's confirmed wire-bit *membership* (now complete for all 8 sub-blocks), what is the actual
bit-index-within-codeword *permutation* -- i.e. which physical wire bit is `fec.rs`'s codeword bit
14 vs. bit 3 vs. bit 0?

**This turned out to be mathematically unsolvable by any amount of black-box weight-3/weight-7
relational testing, proven directly rather than merely suspected.** A brute-force Python constraint
solver was built representing each wire position as an unknown column vector in GF(2)^4 (matching
`fec.rs`'s own `HAMMING_PARITY` column structure), with one constraint per confirmed weight-3
codeword (`column(a) XOR column(b) XOR column(c) = 0`). Using all 18 of `u4`'s confirmed triples
(three independent anchors: 127, 8, 92 -- the latter two swept fresh specifically to break the
degeneracy an earlier single-anchor attempt hit), the solver still found over 20,000 consistent
candidate permutations (search capped there for runtime, but the true count is almost certainly
much higher). Merging in `u5`'s and `u6`'s own 7 triples each (32 total constraints across three
independently-derived Hamming blocks) made no difference -- still capped at 20,001 solutions.

**The reason is structural, not a data shortage**: every constraint of the form `col(a) XOR col(b)
XOR col(c) = 0` is invariant under *any* linear automorphism `T` of GF(2)^4 applied simultaneously
to every column, since `T(col(a)) XOR T(col(b)) XOR T(col(c)) = T(0) = 0` for any linear `T`. The
Hamming(15,11) code's automorphism group has order `|GL(4,2)| = 20160`; the Golay(23,12) code's is
larger still (order roughly 10^7, related to the Mathieu group `M23`). No number of anchor bits or
flip-sweep triples can ever break this symmetry -- confirmed here computationally, not just argued
abstractly, and worth recording so no future session re-attempts the same approach expecting more
data to eventually resolve it.

**The fix: stop asking "which physical bit is codeword index N" and instead sample the code
directly.** The chip's own encoder emits real, valid codewords on every live frame. Two new
committed tools do this:

- `examples/p25_ratet27_capture_frames.rs` -- streams a wide variety of stimuli (9 voice-range
  frequencies as both sine and sawtooth, an amplitude ramp, silence, and 60+ pseudo-random LCG-noise
  frames at varying peak amplitude) through the real chip, dumping each response's raw 18-byte
  CHAND payload as a hex line. Includes a retry-with-backoff wrapper around the send/recv round
  trip, since this session had already hit several transient `WouldBlock` UDP timeouts under
  sustained chip load (previously fatal to long-running sweeps).
- `examples/p25_ratet27_capture_noise_burst.rs` -- a focused follow-up: more pseudo-random noise at
  8 fixed peak amplitudes (500 through 16000), for topping up coverage on whichever block needed
  more distinct samples.
- `examples/p25_ratet27_capture_real_speech.rs` -- streams real recorded speech (the already-cleared
  Open Speech Repository fixtures at `tests/fixtures/osr_speech/`, 8kHz mono, this codec's native
  rate) through the chip 160 samples at a time. Added specifically because synthetic noise alone
  plateaued on one block (see below) -- real speech has far richer, non-stationary spectral content
  than any synthetic signal this investigation had tried.

Across all three tools, **~2200 total captured frames, 1490 of them bit-for-bit distinct**, were
deinterleaved via the validated `natural_position` transform and split into the 8 sub-blocks.
Treating each block's own observed natural-order bit patterns as vectors in a GF(2) linear code and
computing GF(2) rank (Gaussian elimination) gives a direct, assumption-free answer to "is this
really an unwhitened FEC codeword, and if so, what's its generator matrix":

| block | rank found | expected (pure codeword) | verdict |
|---|---|---|---|
| `g0` | 12 | 12 | full rank -- pure Golay(23,12) codeword |
| `g1` | 12 | 12 | full rank -- pure Golay(23,12) codeword |
| `g2` | 12 | 12 | full rank -- pure Golay(23,12) codeword |
| `g3` | **8** | 12 | **short -- see below, unresolved** |
| `u4` | 11 | 11 | full rank -- pure Hamming(15,11) codeword |
| `u5` | 11 | 11 | full rank -- pure Hamming(15,11) codeword |
| `u6` | 11 | 11 | full rank -- pure Hamming(15,11) codeword |
| `c7` | 7 | 7 | full rank -- genuinely raw/unprotected, confirms DVSI's own description |

Full rank on 7 of 8 blocks is itself a real, standalone finding: it directly answers a question this
investigation had left open since §17 (whether the chip mixes any data-dependent whitening/PRN into
the wire bits, the way the textbook IMBE encoder's own `modulation` stage does) -- **it does not**,
for every block reaching full rank. The wire bits genuinely are the FEC codewords themselves.

**`g0`/`g1`/`g2`'s row-reduced generator basis turned out to be bit-for-bit identical to `fec.rs`'s
own systematic Golay(23,12) construction** -- same data/parity split (natural offsets 0-11 as free
data-bit positions, 12-22 as determined parity, an exact match in all 3 x 12 = 36 compared rows),
same `GOLAY_PARITY` values, same bit order (natural offset ascending = codeword bit descending,
MSB-first). This chip's real Golay code needs **no new implementation at all**: `fec.rs`'s existing
`golay_encode`/`golay_decode` apply directly to each Golay block's natural-order bits, verified with
a real round-trip test (`decode_block_round_trips_g0_with_the_real_golay_code`).

**`u4`/`u5`/`u6` all share one identical row-reduced generator basis** (one Hamming FEC routine used
three times, exactly as expected), but this basis's parity submatrix is a **different, though
equally valid, labeling** of the same 11 nonzero 4-bit column values `fec.rs`'s own
`HAMMING_PARITY` uses -- confirmed the columns are the same *set* (`chip_hamming_parity_uses_the_
same_15_nonzero_columns_as_fec_rs`), just assigned to different data-bit positions. This chip-real
table, `HAMMING_PARITY_CHIP = [0b1001, 0b1101, 0b1111, 0b1110, 0b0111, 0b1010, 0b0101, 0b1011,
0b1100, 0b0110, 0b0011]`, was **cross-validated against all 32 of this session's independently
gathered empirical weight-3 relationships** (18 from anchor 127, 7 from anchor 9, 7 from anchor 11 --
the very data the relational-testing approach above proved could never uniquely determine a
permutation) and every single one holds exactly, with zero mismatches. The permutation-degeneracy
result above and this validation are not in tension: many permutations satisfy those 32 relational
constraints, but the *one this session actually derived by sampling real codewords* is confirmed
consistent with all of them, which is the strongest evidence available that it's the chip's real
table (not merely "a" table that happens to work).

**New committed code**: `src/ambe/ratet27_wire_format.rs` (the validated 12x12 transform, the 8
sub-block boundaries, and `block_wire_members` computing each block's exact wire membership
programmatically from the transform rather than as separately hand-maintained lists -- regression-
tested against every directly chip-confirmed set from §§18-21 with zero mismatches) and
`src/ambe/ratet27_fec.rs` (the chip-real Golay reuse of `fec.rs`, the new `HAMMING_PARITY_CHIP`
table and its encode/decode functions, a unified `decode_block` entry point, and the full validation
test suite described above -- 19 tests total across both files, all passing). This is a genuine,
tested, chip-validated software duplicate of RATET(27)'s FEC layer for 7 of its 8 sub-blocks, not
just documentation of the format.

**`g3` remains open, and this is a real finding, not a sampling gap.** Despite the same ~2200-frame,
1490-distinct-frame capture spanning pure tones, 8 different noise amplitudes, and 16 seconds of
real recorded speech, `g3`'s observed wire bits plateau at exactly GF(2) rank 8 (not 12), with 4 of
its 23 natural-order bits (offsets 2-5, transforming to natural positions 71-74) staying **exactly
zero in every single one of the 1490 distinct captured frames** -- not merely rare, literally never
observed as 1. More than 100x the frame count that reached full rank on every other block failed to
move `g3` past rank 8, across wildly different stimulus types, which rules out "just needs more
samples" as the explanation. The most plausible reading is that `g3` carries a parameter tied to
some speech characteristic none of this investigation's stimuli (or, apparently, ordinary read-aloud
English sentences) ever produce -- deliberately extreme pitch, a specific voicing pattern, or very-
high-order spectral content are candidates, but this needs either different stimulus material or a
from-spec understanding of exactly which IMBE parameter lands in the highest-index Golay block to
know what to specifically provoke. `decode_block` deliberately panics if called on `g3` rather than
silently assuming it matches `g0`-`g2`'s already-confirmed generator.

**`g3`'s rank-8 plateau confirmed exhaustively, not just under the original stimulus set.** Three
follow-up capture rounds specifically targeted `g3`: two more real-speech recordings from different
OSR speakers (1000 more frames), and a dedicated exotic-stimulus tool
(`examples/p25_ratet27_capture_exotic_stimuli.rs`) covering 5 dual-tone/DTMF-style two-sinusoid
mixes, 5 fast intra-frame linear chirps sweeping the full pitch range in both directions, and 7
sine tones at and beyond the edges of AMBE's documented 57-444Hz pitch range (down to 30Hz, up to
800Hz). None of it moved the needle even slightly: **2687 total distinct captured frames (spanning
tones, ramps, 8 noise amplitudes, 3 real speakers' worth of recorded English speech, dual tones,
chirps, and extreme pitch) all land in the exact same GF(2) rank-8 subspace**, with the exact same 4
of 23 natural-order bits staying exactly zero throughout. This is about as thorough an audio-domain
stimulus search as is practical, and the finding held with zero exceptions across all of it -- strong
evidence that whatever `g3` encodes is not reachable by any single-frame audio stimulus at all, and
more likely depends on either multi-frame encoder history/adaptive state that builds up over many
frames of specific dynamics (not reachable in a short capture window), or an encoder feature/mode
this investigation's SPEECH-packet-only testing never engages (frame-repeat, DTX, or another
documented `ECMODE`/`DCMODE` flag). Resolving this now needs a from-spec understanding of exactly
which IMBE parameter and bit-history dependency lands in the highest-index Golay block, not more
undirected stimulus variety -- recorded here so a future session doesn't repeat the same broad
audio-stimulus search expecting a different result.

**Two scope notes for whoever continues this work, stated now rather than discovered later:**

- §9's finding that pitch lives in `u2`, Gray-coded, was derived through the *textbook* TIA-102
  Annex H deinterleave -- which this entire investigation (§§15-22 and this section) has since shown
  does not match this chip's real wire format. That specific claim needs re-deriving through the
  correct `ratet27_wire_format` deinterleave plus an empirical decode before anything semantic gets
  built on top of it; it should not be assumed to still hold.
- Byte-exact PCM reproduction of the chip's own *synthesis* output is almost certainly unreachable
  regardless of how completely the FEC/interleave layer above gets nailed down -- DVSI's manual
  documents proprietary post-processing (noise suppression, spectral enhancement variants, comfort
  noise) in the synthesis path that this crate's from-spec `src/ambe/` implementation was never
  going to reproduce bit-for-bit. The realistic, achievable validation bar for the semantic
  (voice-parameter) layer, once attempted, is parameter-level agreement -- decoded pitch/voicing/
  amplitude values matching what this crate's own encoder produces for the same input PCM -- not
  identical output samples. Recording this now so the eventual semantic-mapping work is scoped
  honestly from the start rather than discovering the ceiling at the end.

**A first, concrete probe at the semantic (u-vector) remapping question, negative but informative.**
`ambe::mod.rs`'s `encode_frame` was refactored (no behavior change -- verified by the full 190-test
`ambe` module suite passing unchanged) to expose its own pre-FEC `u_hat_0..u_hat_7` via a new public
`encode_prioritized_bits`, specifically to let real-chip validation work compare this crate's own
textbook `u_hat` semantics against the real chip's decoded data. The natural first hypothesis: since
`bit_prioritization`'s own block widths `[12,12,12,12,11,11,11,7]` exactly match the chip's real
`g0..g3`/`u4..u6`/`c7` sizes and natural order, maybe `u_hat_0` corresponds directly to `g0`'s
decoded data, `u_hat_1` to `g1`'s, and so on, with no further block-level relabeling needed.
`examples/ratet27_compare_textbook_u_vectors_to_chip.rs` tests this directly: for a known-frequency
pure sine, feed the *exact* pitch (not an estimate) through `encode_prioritized_bits` to get this
crate's own `u_hat_0`, and decode a real captured chip frame at the same frequency's `g0` block via
`ratet27_fec::decode_block`. **The top 6 bits did not match at any of 4 tested frequencies
(100/200/250/400 Hz)** -- and, notably, the chip's own decoded `g0` value saturated to the same
all-1s top-6-bits pattern (`0b111111`) at 200, 250, and 400 Hz while showing a distinct value at 100
Hz, which does not resemble the textbook fundamental-frequency quantizer's own smoothly-varying
`floor(4*pi/omega0 - 39)` output for those same frequencies. This rules out the simplest possible
`u0=g0` direct-correspondence hypothesis, at least for the fundamental-frequency field specifically,
and hints that the real chip's own pitch estimation/quantization convention may differ substantially
from the textbook's (not just its bit ordering) -- consistent with, and now a second independent
data point supporting, §9's earlier finding that the textbook's pitch-bit assumptions don't carry
over to this chip. This negative result is recorded so a future session doesn't re-attempt the same
simple hypothesis; the semantic-mapping question remains genuinely open and likely requires
understanding the chip's own pitch estimator (a substantial, separate reverse-engineering task) more
than further trial-and-error block-correspondence guessing.

**A second, broader semantic probe using real speech (not a single assumed-exact pitch), also
inconclusive but with one lead worth following up.** Since this chip's encoder is known to never
converge to a fixed steady state on a pure tone (`examples/ambe_chip_validate_dstar.rs`'s own doc
comment), the single-frequency `u0=g0` test above may simply have compared mismatched effective
pitches. `examples/ratet27_speech_u_vector_correlation.rs` runs real recorded speech through both
paths in lock-step, frame by frame -- this crate's own full encode pipeline (the same per-frame
pitch grid-search as `tests/ambe_real_speech_round_trip.rs`, with `FrameState` history properly
chained across frames) and the real chip on the identical PCM -- dumping each successfully-encoded
frame's `u_hat_0..u_hat_7` alongside the chip's own decoded `g0`/`g1`/`g2`/`u4`/`u5`/`u6`/`c7` for
600 real frames. Pearson correlation across all `(this crate's field, chip's field)` pairs found
**no strong correspondence anywhere** -- the largest is a moderate 0.359 between this crate's own
`u_hat_4` and the chip's decoded `u4`, with everything else under 0.2 in magnitude (several near
zero). This is genuinely inconclusive rather than a clean negative: raw-integer Pearson correlation
is a weak tool for finding a *bit-level* correspondence that might be Gray-coded (as already
established for D-STAR/AMBE+2's own pitch field, §9) or reordered -- a real relationship would show
near-zero linear correlation under either scrambling even if the mapping is deterministic. The
`u_hat_4`/chip-`u4` pair is the one lead worth another session's time (e.g. checking a per-bit XOR/
Gray-decode correlation rather than raw integer correlation); nothing else in this table showed
enough signal to prioritize. The full 600-frame capture (every field from both sides, per frame) is
committed at `docs/references/ratet27_captures/u_vector_speech_correlation_600frames.tsv` so a
future session can pick up exactly where this one left off (re-analyze with a different
correlation technique) instead of re-running the same real-speech capture against the chip from
scratch.

**A third semantic-layer lead, prompted by re-reading this document's own earlier §9**: that section
found NOFEC mode's pitch parameter sits in the raw stream's third 12-bit slice (`u2` under NOFEC's own
contiguous convention), Gray-coded, with a distinctive "increases then saturates at a quantizer
ceiling" shape across frequency. Re-running that exact shape-check against FEC mode's own `g0`
(natural offsets 0-22, the *first* Golay block, decoded via `golay_decode` -- the block already
proven bit-for-bit identical to `fec.rs`) across the 16 frequencies already captured this session
(30-800 Hz, one converged frame each) finds a real, but weaker and messier, echo of the same shape:
Spearman rank correlation with frequency is a strong 0.897 across all 16 points, and `g0`'s decoded
value visibly separates into "generally lower, noisier" below ~200 Hz and "clustered near a ceiling
around 4040-4047" above it -- the same qualitative saturation signature as NOFEC's `u2`. **This is a
real lead, not a confirmed mapping**: restricted to frequencies inside AMBE's own documented 57-444
Hz voice-pitch range, the values are not cleanly monotonic (57->2853, 80->3045, 100->3041,
125->3938, 160->2217, 200->4042, 250->4040, 320->3946, 400->4041, 444->4043) -- a real dip at 160 Hz
and an early plateau by 200 Hz rather than a smooth curve -- so this could equally reflect several
parameters simultaneously destabilizing outside/at the edge of the vocoder's normal operating
assumptions (most of the tested frequencies are below or at the edge of the documented voice range)
rather than `g0` specifically carrying pitch. `g1` and `g2` show much weaker correlations (Spearman
-0.35 and -0.49) over the same data, for comparison. The full 16-frequency dataset and the analysis
script are committed at `docs/references/ratet27_captures/analyze_g0_pitch_hypothesis.py` (reads
`captured_frames_all5`-equivalent capture data) so a future session can re-test this specific
hypothesis with denser, purely in-voice-range frequency sampling rather than re-deriving it from
scratch.

**Follow-up, upgraded from lead to confirmed finding: `g0` genuinely carries a real pitch-related
quantizer in FEC mode.** The messy result above used single sine-tone captures at scattered
frequencies (including several outside AMBE's own voice-pitch range); a focused follow-up
(`examples/p25_ratet27_capture_dense_pitch_sweep.rs`) instead swept 20 frequencies *strictly within*
the documented 57-444Hz range at 20Hz steps, using the **sawtooth** signal (already established
elsewhere in this investigation as far more reliable than sine for convergence), with 60 settling
frames and 8 captured frames per frequency. Result: **19 of the 20 frequencies gave a perfectly
stable, single repeated `g0` value across all 8 captured frames** (only 80Hz showed the same kind of
low-frequency bistable oscillation already documented elsewhere in this investigation for D-STAR
and this crate's own encoder), and those 19 stable values form a clean, almost perfectly monotonic
decreasing staircase as frequency increases: `60->3945, 100->3065, 120->2745, 140->2237 (=160),
180->1085, 200->1597 (=220), 240->1149 (=260), 280->765 (=300=320), 340->445 (=360=380),
400->125 (=420), 440->61`. **Spearman rank correlation with frequency: -0.952.** The value
*decreasing* with increasing frequency matches the textbook `quantize_fundamental_frequency`
formula's own sign convention (`floor(4*pi/omega0_hat - 39)`, which decreases as frequency/omega0
increases), and the coarsening step size at high frequency versus finer steps at low frequency (a
real signature of any quantizer that's uniform in *period*, i.e. `1/frequency`, rather than in
frequency itself) matches the qualitative shape a genuine AMBE-family pitch quantizer should have.
One real anomaly, disclosed rather than smoothed over: 180Hz's value (1085) dips below both its
neighbors (140/160's 2237 and 200/220's 1597), breaking strict monotonicity at that single point --
plausibly a quantizer-boundary interaction with the discrete `L_hat`/`K_hat` harmonics-count step
function rather than a measurement error (every other point was perfectly frame-stable), but not
yet explained. **This does not need Gray-decoding to show the relationship** (unlike NOFEC mode's
own `u2` field, per §9) -- plain binary `g0` already correlates cleanly, a real, disclosed structural
difference between how the two modes' pitch-related fields are quantized/coded, not an
inconsistency in this investigation's own analysis. Taken together with `g0`'s already-confirmed
bit-for-bit identity to `fec.rs`'s Golay code, this is the strongest, most concrete semantic-layer
result of the session: **RATET(27) FEC mode's pitch-related parameter lives in `g0`** (wire
positions `{0,1,12,13,24,25,36,37,48,49,60,61,72,73,84,85,96,97,108,109,120,121,132}`), plain
binary, monotonically decreasing with frequency. The exact quantizer formula (bin edges, whether it
matches `quantize_fundamental_frequency`'s literal constants or a chip-specific variant) remains
unfit -- a natural next step given this clean staircase data is now committed and reproducible.

**Correction, found by the very next follow-up test: `g0` is not simply "the pitch parameter" --
it responds to amplitude too, most likely making it a gain/energy-related quantizer rather than a
pure pitch quantizer.** A clean amplitude sweep (`examples/p25_ratet27_capture_amplitude_sweep.rs`:
a single fixed, well-converged frequency (200Hz sawtooth) at 16 different, precisely-known peak
amplitudes -- unlike pseudo-random noise, whose actual RMS is decorrelated from its stated peak,
making the earlier gain-correlation attempt against noise data too noisy to be useful) found `g0`
correlates with amplitude just as cleanly as it correlated with frequency above: **Spearman 1.000**,
a perfect monotonic staircase from 1045 (quietest) to 1597 (loudest, saturating). This raised a real
concern that the frequency-correlation finding above might have been an amplitude/RMS confound
rather than a genuine pitch effect: a discretely-sampled sawtooth's *actual* RMS is not perfectly
frequency-independent when the period is a large fraction of the 160-sample frame (incomplete-cycle
boundary effects at low frequencies), even though its *nominal peak* was held fixed.

**Directly tested and ruled out as a confound, but the underlying dual-dependency is real.**
`examples/p25_ratet27_capture_rms_normalized_pitch_sweep.rs` reruns the exact same 57-444Hz dense
sweep, but explicitly computes each frequency's actual buffer RMS and rescales to a fixed target
(confirmed: 3463.4-3463.6 across all 20 frequencies, genuinely constant, not just nominally so).
**Every single decoded `g0` value came back bit-for-bit identical to the original, non-normalized
sweep** -- the exact same near-perfect monotonic staircase, Spearman -0.952. This rules out RMS
confound as the explanation for the frequency correlation: `g0` really does depend on frequency
independently of amplitude, *and* (per the amplitude sweep above) really does depend on amplitude
independently of frequency, at the same time. This is not a contradiction -- it's exactly the
behavior a genuine **gain/energy quantizer** should have in any real vocoder: gain is computed from
a spectral-amplitude decomposition that itself depends on where the harmonics fall relative to the
estimated pitch, so a real gain parameter is expected to shift with pitch even at constant overall
signal RMS, not just with amplitude. The likelier reading, revising the framing above rather than
retracting the underlying data (both correlations are real and reproduced identically across
independent captures): **`g0` is this chip's real gain/energy-related quantizer** (plausibly
`b_hat_2`/`g_hat[0]` in this crate's own textbook terms, the first-stage DC/gain DCT coefficient,
rather than the fundamental-frequency quantizer `b_hat_0`) -- still a genuine, useful semantic
result (a real parameter's real location, confirmed bit-exact and reproducible), just not the
specific parameter first guessed. Disentangling gain from pitch fully would need a 2-D sweep
(varying both independently and checking whether `g0` is better explained by a formula combining
both, e.g. log-energy at the fundamental) -- a concrete, bounded next step, with all three
datasets (dense pitch sweep, amplitude sweep, RMS-normalized sweep) committed for it.

**A small, independent corroboration of the gain interpretation, and a real negative result for
`g3`'s prediction-residual hypothesis.** Comparing already-captured silence frames against 200Hz
voiced frames across all 4 Golay blocks: `g0`'s silence value (1025) sits almost exactly at the
bottom of the range the amplitude sweep independently established (quietest tested amplitude gave
1045) -- a real, unforced consistency check supporting `g0` as a genuine gain/energy quantizer
(silence naturally reads as "near-minimum energy"). `g0`, `g1`, and `g3` all show *zero* overlap
between their silence-frame and voiced-frame value sets (consistent with several parameters all
being energy-sensitive, not necessarily each independently encoding "voicing" as a dedicated
decision); `g2` shows partial overlap. Separately, `g3`'s own rank-8 plateau was tested against one
more concrete hypothesis: since this crate's own encode pipeline includes a real frame-to-frame
*prediction residual* stage (differential encoding against previous-frame history,
`src/ambe/prediction.rs`), maybe `g3` carries part of a similar residual that only shows real
variation under large frame-to-frame discontinuities -- untested by this session's earlier
stimuli (steady tones, per-frame-independent noise, and *smoothly*-varying real speech).
`examples/p25_ratet27_capture_abrupt_transitions.rs` fed 2400 frames abruptly alternating between
maximally different states every single frame (loud-high-pitch / silence / loud-low-pitch /
quiet-high-pitch / noise / quiet-low-pitch, cycling). **Result: zero new distinct `g3` values
appeared** -- the same 149 distinct values already on record, even under the most aggressive
frame-to-frame discontinuity this investigation has tried. This rules out the prediction-residual
hypothesis specifically (or at least this particular way of trying to trigger it), leaving `g3`'s
real cause still open per the exhaustive-stimulus note above.

**D-STAR and AMBE+2 half-rate status, checked against this session's broader `/goal` directive, and
freshly re-run live against the real chip this session (not just cited from an earlier session's
claim)**: `examples/ambe_chip_validate_dstar.rs`, re-run live: **PASS, 40/40 frames at every one of
8 tested frequencies (50-1000Hz) Golay-decode with zero corrected errors on both `C0` and `C1`**.
`examples/ambe_chip_validate_ambe_plus_2.rs`, re-run live (the first attempt hit this investigation's
already-known transient chip `WouldBlock` timeout, discussed throughout this document; a clean
retry succeeded): **RATET(33) (half-rate with FEC): 10/10 zero-error frames under the "Annex H
deinterleaved" framing hypothesis** (the other hypothesis tested, direct `C0||C1||C2||C3`
concatenation, correctly gets 0/10 -- confirming this rate genuinely does use the textbook Annex H
interleave, unlike RATET(27)'s full-rate mode this session spent most of its time on).
RATET(34) (half-rate, No FEC): the sliding-window correlation scan reproduces its own
previously-established result (best candidate `bits[27..34)`, Gray-decoded, `|spearman|=0.964`).
Both harnesses are real, existing, chip-validated code from earlier sessions, and this session
freshly confirmed both still pass against the live chip today -- closing that part of the broader
goal with current, not merely historical, evidence.

**RATET(27) now has the same kind of real PASS/FAIL chip-validation harness D-STAR and AMBE+2 half-
rate already had.** `examples/ambe_chip_validate_ratet27.rs` -- new this session -- captures live
chip frames across the same 8 frequencies as the D-STAR harness and decodes each through
`ratet27_wire_format`/`ratet27_fec` directly (no search, no hypothesis-scoring -- this session
already determined the real format), checking for zero corrected errors on all 7 resolved blocks
(`g0`, `g1`, `g2`, `u4`, `u5`, `u6`, `c7`; `g3` deliberately excluded). **Live result: PASS, 120/120
captured frames across all 8 frequencies, zero errors on every block.** This supersedes the older,
now-stale `ambe_chip_validate_p25_wireformat.rs` (a search harness built on the wrong assumption
that RATET(27) uses PRN whitening, since disproven by this session's GF(2) rank analysis) and is
the concrete "duplicated in software, validated against the chip" deliverable for RATET(27)'s FEC
layer this whole session's work has been building toward.

**Extended immediately with real recorded speech**, matching what a real deployment would actually
see rather than only synthetic tones (the same OSR speech fixtures used elsewhere in this
investigation, two different speakers, 400 frames -- 8 seconds -- each). **Live result: PASS, all
920 frames (120 synthetic-tone + 800 real-speech) decode with zero corrected errors on every one of
the 7 confirmed blocks.** This is a substantially stronger validation claim than synthetic tones
alone: real speech's non-stationary, wideband spectral content exercises far more of the FEC
codeword space than any fixed set of test tones could.

**The same real-speech improvement was applied to the D-STAR and AMBE+2 half-rate harnesses too,
for consistency across all three validated chip modes.** `ambe_chip_validate_dstar.rs`: **PASS,
800/800 real-speech frames** (2 recordings) Golay-decode with zero errors on both `C0`/`C1`, on top
of the existing 8-frequency synthetic-tone coverage. `ambe_chip_validate_ambe_plus_2.rs`: **PASS,
800/800 real-speech frames** zero-error under the confirmed "Annex H deinterleaved" framing for
RATET(33) -- this file also got a retry-on-`WouldBlock` wrapper added to its send/recv round trip
(it had none, and hit this investigation's well-documented transient chip timeout twice in three
runs while testing this exact change), matching the robustness already built into every other
capture tool this session wrote. All three chip modes -- D-STAR, AMBE+2 half-rate, and RATET(27) --
now have real, live, real-speech-validated PASS/FAIL harnesses on equal footing.

## 24a. `c7`'s 5 previously "inert" raw bits actually carry real signal -- refining, not overturning, an earlier session's finding

An earlier session's §14 found only 2 of `c7`'s 7 raw bits showed any audible effect on a plain 200Hz
sine test signal, and §18 (this session) repeated that observation on the same signal type. A
zero-chip-time re-check against this session's much larger, more varied captured dataset (2973
frames spanning tones, noise, real speech, dual-tones, and chirps) finds real per-bit variance on
**all 7** `c7` bits (43-60% ones each, none constant) and moderate frequency correlation on several
of them simultaneously (natural offsets 137/139/143 showing `|Spearman|` 0.37-0.46 on the mixed
frequency-labeled subset). **This refines rather than overturns the earlier finding**: the original
conclusion was scoped explicitly to "no effect on this specific test signal" (a plain sine), and a
richer stimulus set was always expected to reveal more, exactly as happened repeatedly elsewhere in
this investigation (the u5/u6 sine-blindness fix, section 19). A follow-up check against the clean,
single-waveform (sawtooth-only) dense pitch sweep found an even stronger single-bit correlation
(natural offset 138, `Spearman=0.880`) -- but with the opposite sign from the mixed-waveform
dataset's own reading of a *different* bit, which is itself informative: it suggests `c7`'s content
may be sensitive to waveform *shape* (sine vs. sawtooth), not purely fundamental frequency, since
mixing waveform types in the first dataset would scramble a shape-dependent signal in exactly this
way. **Real, useful expansion of known content** (5 more bits confirmed non-inert than previously
documented), but the exact relationship remains complex and not yet reduced to a specific formula --
left open for a future session with the two saved analysis scripts
(`analyze_c7_bits.py`, `analyze_c7_dense_pitch_sweep.py`) as a starting point.

## 24. First real signal on `g1`/`g2`'s semantic content: a moderate correlation with harmonic count, zero chip time

With `g0` now understood as a gain/energy quantizer (section 23), the next open semantic question is
what `g1` and `g2` (the second and third Golay blocks) carry. A zero-chip-time re-analysis of the
already-committed dense pitch sweep (`dense_pitch_sweep_57to444hz.tsv`, §23) against this crate's own
`vuv::harmonics_count` (`L_hat`) and `vuv::frequency_bands_count` (`K_hat`) -- both deterministic,
purely-pitch-derived integer step functions already implemented and tested in this codebase -- found
a real, if imperfect, correlation:

| block | Spearman vs `L_hat` | Spearman vs `K_hat` |
|---|---|---|
| `g1` | 0.558 | 0.508 |
| `g2` | 0.689 | 0.650 |

Neither is as clean as `g0`'s own -0.952 against frequency, but both are real, reproducible signals
(computed once, deterministically, from already-captured data -- see
`analyze_g1_g2_lhat_khat_correlation.py`). The raw values show a genuine staircase-like structure:
`g1` sits near a stable ~3410 for most of the mid-range (100-260Hz, `L_hat` 37 down to 13) then drops
sharply to ~1360 for the whole 320-440Hz range (`L_hat` 11 down to 8), with different values again at
the very lowest frequencies (60/80Hz, `L_hat` 61/46). `g2` shows a similar multi-stage pattern (values
cluster around 1506/2530-ish for `L_hat` in the low-to-mid teens/twenties, drop to ~480 for `L_hat`
around 11-12, drop again to ~34 for `L_hat` 9-10, and drop to ~2 for `L_hat` 8-9). **This is real,
new information about previously-unknown semantic content** (a moderate, genuine relationship to
harmonic count, not the null result this investigation would show if `g1`/`g2` were unrelated to
pitch structure at all) but does not yet pin down an exact formula or confirm `L_hat`/`K_hat` as the
literal encoded quantity rather than some other pitch-derived parameter that happens to correlate
with them (e.g. a genuine higher-order spectral/gain-vector coefficient, whose own natural range
also depends on `L_hat` per this crate's own `tables::block_lengths_for_l`). A natural next step,
requiring more chip time this session didn't spend: a systematic sweep specifically targeting the
`L_hat`/`K_hat` step-boundary frequencies (rather than the 20Hz-even grid used here, which mostly
missed them) to see whether `g1`/`g2` jump in lock-step with `L_hat`'s own exact transition points.

**That follow-up was run, and the result is genuinely nuanced -- disclosed as such rather than
forced into either a clean confirmation or a clean denial.** `examples/p25_ratet27_capture_lhat_
boundary_sweep.rs` tested 7 of `L_hat`'s own computed transition frequencies (100.7, 155.4, 202.6,
238.9, 271.2, 313.8, 340.5 Hz), each probed 3Hz below and 3Hz above the exact boundary (RMS-
normalized sawtooth, same technique as the earlier confound-controlled pitch sweep). **`g1` changed
value at every single one of the 7 boundaries tested** -- but this is weaker evidence for an
`L_hat`-specific link than it first appears, since a 6Hz-wide window with no same-`L_hat` control
pair cannot distinguish "`g1` jumps exactly at `L_hat` transitions" from "`g1` varies continuously
and finely with pitch" (the same character `g0`'s own gain quantizer already showed, per section 23)
-- both explanations predict a change across any 6Hz gap in this frequency range. **`g2`, by
contrast, stayed exactly stable across 6 of the 7 boundaries**, changing only at 340.5Hz -- this is
more consistent with `g2` tracking something genuinely coarser and step-like (plausibly `L_hat`/
`K_hat` or a quantity derived from them) than with continuous fine-grained pitch sensitivity,
though still short of proof without a proper same-`L_hat`, different-frequency control pair (e.g.
two frequencies several tens of Hz apart but on the same side of a boundary, which this specific
test didn't include). **Net honest read**: `g2` remains the more promising `L_hat`-linked candidate
of the two; `g1` is more likely a second continuously-varying, pitch-sensitive parameter (like
`g0`) than a discrete harmonic-count encoding. Full boundary-sweep data and analysis script
committed for a properly controlled follow-up.

**That properly controlled follow-up was run immediately, and it refutes the `g2=L_hat` reading
above -- corrected here rather than left standing.** `examples/p25_ratet27_capture_lhat_controlled_
test.rs` tests three matched-5Hz-spacing frequency triples around three different `L_hat`
boundaries, giving both a same-`L_hat` control pair and a crossing pair at identical spacing for
each: `(150,155,160)Hz` around the 155.4Hz boundary (`L_hat` 24->23), `(195,200,205)Hz` around
202.6Hz (`L_hat` 18->17), and `(266,271,276)Hz` around 271.2Hz (`L_hat` 13->12). **If `g2` genuinely
encoded `L_hat`, every same-`L_hat` pair should match exactly and only the crossing pairs should
differ. Instead, `g2` changed in 2 of the 3 same-`L_hat` pairs** (`195/200Hz`, both `L_hat`=18:
`g2` 1506->2529; `266/271Hz`, both `L_hat`=13: `g2` 2530->2518) -- changes of similar character to
the crossing pairs, not the "held exactly constant" behavior the hypothesis predicts. This
decisively rules out `g2` as a direct `L_hat`/`K_hat` encoding: the moderate correlation found in
this section's own opening analysis was almost certainly a **confound**, not a causal link --
`L_hat` and raw pitch are both monotonic-ish functions of frequency, so any genuinely continuous,
pitch-sensitive parameter (the same character already established for `g0`) will show *some*
correlation with `L_hat` as a side effect, without actually encoding it. **The corrected, more
likely reading, consistent with everything found this session**: `g0`, `g1`, and `g2` are probably
all continuously-varying gain/spectral-amplitude-related coefficients (this crate's own textbook
pipeline has a real 5-element `gain_vector`, `g_hat[0..5)`, feeding separate quantizers -- a
structurally plausible home for several distinct-but-correlated pitch-and-amplitude-sensitive
parameters), not a mix of a gain field and a discrete harmonic-count field. Disentangling which
specific coefficient each block carries -- if any single clean 1:1 correspondence exists at all --
remains open, and this specific `L_hat`-encoding hypothesis is now closed off rather than left
ambiguous. Full controlled-test data and script committed.

## 25. Primary-source-motivated ECMODE tests: `TD_ENABLE` has no effect on RATET(27)'s wire format; `TS_ENABLE` reveals a genuinely different, but currently uninformative, "tone frame" mode

A close re-read of DVSI's AMBE-3000R manual surfaced a previously untested, well-motivated
hypothesis for `g3`'s stubborn rank-8 plateau: `ECMODE_IN` bit 12 (`TD_ENABLE`, Tone Detect Enable)
is **enabled by default at reset**, and the manual states the encoder sets a `TONE_FRAME` status
flag "if the output frame contains either a single frequency tone, a DTMF tone, a KNOX tone, or a
call progress tone" -- raising the concern that every pure-tone/sawtooth stimulus this entire
investigation has used (the primary tool for mapping `g0`-`g3` bit-by-bit) could have been silently
triggering tone-detection-driven encoding differences the whole time, never tested or ruled out.

**Tested directly via `PKT_ECMODE` (field `0x05`, a 2-byte `ECMODE_IN` word, confirmed from the
manual's own Table 35).** `examples/p25_ratet27_capture_tone_detect_disabled.rs` sends
`ECMODE_IN=0x0000` (every feature including `TD_ENABLE` off) before the same dense 57-444Hz
sawtooth sweep used elsewhere. **Result: no detectable difference at all.** `g0` reads the exact
same value (1597) at 200Hz with `TD_ENABLE` on or off; merging this new data into the full 2840-
frame dataset left `g3`'s distinct-value count exactly unchanged (149, identical to before) --
every frame produced under `TD_ENABLE=0` matched a value already seen under the default
`TD_ENABLE=1`. This decisively rules out tone detection as an explanation for `g3`'s plateau, and
for any of this investigation's block-mapping results more generally: pure-tone stimuli behave
identically whether or not the chip's own tone-detection logic is active for this rate.

**A related but distinct flag, `TS_ENABLE` (bit 14, off by default), was also tested and behaves
very differently -- a genuine, newly-confirmed chip-mode discovery, though not directly useful for
`g3`.** DVSI's manual: "If TS_ENABLE=1, then the encoder produces a tone frame in place of the
frame that it would normally produce." `examples/p25_ratet27_capture_tone_send_forced.rs` sends
`ECMODE_IN=0x4000` and repeats the same 20-frequency sweep. **Every single frequency produced the
exact same fixed hex pattern** (`f08f00c08f00d00900500d00100408400400`), completely independent of
the actual input signal -- confirming the wire format genuinely does change under `TS_ENABLE=1` (a
real, distinct encoder mode exists and is reachable), but this specific forced-substitution mode
produces a constant placeholder rather than content reflecting the real input, at least for a plain
sawtooth stimulus. This is consistent with the chip's real tone-frame encoding needing input that
actually passes its own tone-classification logic (a genuine DTMF pair, KNOX tone, or call-progress
tone with correct timing/structure) rather than being reachable by simply forcing the flag alone --
a concrete, bounded lead for a future session (test real DTMF tone pairs, e.g. 697+1209Hz for "1",
with `TS_ENABLE` left at its own default so the chip's own detection decides when to substitute,
rather than forcing it unconditionally). Both datasets committed
(`tone_detect_disabled_sweep.tsv`, `tone_send_forced_sweep.tsv`).

**Direct follow-up: real ITU-T Q.23 DTMF digit tones, with `ECMODE_IN` left at its default (not
forced), letting the chip's own tone-classification logic decide.** `examples/p25_ratet27_capture_
real_dtmf.rs` fed all 16 real DTMF digit tone pairs (row 697/770/852/941Hz + column
1209/1336/1477/1633Hz, per ITU-T Q.23 -- the actual standard frequency pairs, not the arbitrary
dual-tone pairs `p25_ratet27_capture_exotic_stimuli.rs` tried earlier). **Result: none of the 16
digits produced the fixed placeholder pattern from the forced-`TS_ENABLE` test above, but the wire
output shows unmistakable, systematic row/column structure** -- e.g. digits sharing the row 697Hz
(`1`,`2`,`3`,`A`) all share one wire pattern prefix, digits sharing row 941Hz (`*`,`0`,`#`,`D`)
share a different one, with column identity determining the remaining bytes. This is qualitatively
different from ordinary voice-tone encoding (which showed no such simple prefix grouping anywhere
else in this investigation) and is real, positive evidence that the chip's built-in DTMF
classification is genuinely active and does encode DTMF content specially -- just not via the
constant placeholder the forced flag alone produced. Full characterization (whether this is a
distinct frame *format* entirely, or the normal FEC/interleave format carrying a much lower-entropy
DTMF-specific parameter set) is not yet done and is a concrete, bounded next step; the complete
16-digit dataset is committed (`real_dtmf_sweep.tsv`) so a future session can pick this up directly
rather than re-capturing it.

**That characterization was completed immediately, and it is a clean, complete, fully-solved
finding.** Decoding all 16 DTMF frames through the already-validated `ratet27_wire_format`/
`ratet27_fec` pipeline (`decode_dtmf.py`) resolves the entire structure at
a glance: **the normal FEC/interleave format is reused (not a distinct frame format), but with
almost everything zeroed except two fields that directly, cleanly encode the DTMF row and column
frequencies:**

| block | DTMF behavior |
|---|---|
| `g0` | Encodes the **row** frequency: exactly `4032, 4033, 4034, 4035` for rows 697/770/852/941Hz respectively -- a perfect, unbroken 4-step linear staircase, identical across all 4 digits sharing each row |
| `u4` | Encodes the **column** frequency: exactly `80, 208, 336, 464` for columns 1209/1336/1477/1633Hz -- also perfectly linear, with a constant step of exactly 128 (`2^7`) between adjacent columns |
| `g1`, `g2`, `g3`, `u5`, `u6`, `c7` | **All constant (`g1`=2944, `g2`=0, `g3`=0, `u5`=0, `u6`=0, `c7`=0) across every one of the 16 digits** -- these carry no DTMF-specific information at all; `g3` decodes with distance 0 to the all-zero Golay codeword, confirming `0` is a genuine, valid member of its codeword space (not itself informative about `g3`'s normal-voice-mode meaning, since DTMF mode simply never uses it) |

This is a complete, decisive resolution of the "what does the chip do with DTMF" question raised by
this section's own earlier tests -- not just "DTMF classification is active" (already shown above)
but the *exact* mechanism: reuse the same FEC blocks, park two of them (previously unassigned to any
confirmed semantic meaning) on simple linear row/column frequency codes, and zero the rest. This is
also indirect, real evidence for two things established more tentatively elsewhere in this
document: it confirms `g0` really is a frequency-related quantizer field (consistent with, though a
different specific mode from, the gain/pitch interpretation in section 23), and it gives `u4` a
second, cleanly-decoded, real-world semantic meaning distinct from its already-fully-solved Hamming
FEC role -- the chip clearly repurposes the same wire positions for different content depending on
what it classifies the input as. `decode_dtmf.py`'s own output and this table are fully
reproducible from the committed `real_dtmf_sweep.tsv` with zero chip time.

## 26. `DTX_ENABLE` genuinely changes silence-frame encoding, confirming DVSI's own "background noise level" claim; `g3` unaffected

Continuing the same primary-source-motivated ECMODE testing that found the DTMF encoding mode
(section 25), tested `DTX_ENABLE` (`ECMODE_IN` bit 11, Discontinuous Transmission / Voice Activity
Detection, hardware-pin-dependent default). DVSI's manual makes a specific, testable claim: with
VAD/DTX enabled, "the encoder will output a silence frame (in-band)... [which] contains information
regarding the level of background noise" for the decoder's Comfort Noise synthesis.

**Confirmed directly.** `examples/p25_ratet27_capture_dtx_silence.rs` captures pure digital silence
(and, as a control, a steady tone and two noise levels) with `ECMODE_IN`'s `DTX_ENABLE` explicitly
off (`0x0000`) and explicitly on (`0x0800`). For silence specifically, three blocks change from
variable-and-different to a **clean, near-constant value** under `DTX_ENABLE=1`:

| block | DTX off (3 sample frames) | DTX on (3 sample frames) |
|---|---|---|
| `g0` | `1025, 1025, 1025` | `3841, 3841, 3841` (constant, but a different constant) |
| `g2` | `2530, 2526, 2526` | `478, 478, 478` (tightly constant) |
| `c7` | `27, 95, 27` (varies) | `92, 92, 92` (constant) |

`g1` and `u4`-`u6` show comparable ranges either way; **`g3` continues oscillating between `1024`
and `2048` under both settings**, unaffected by `DTX_ENABLE` -- consistent with, and a further
independent confirmation of, section 23's own low-frequency/low-signal bistable-oscillation pattern
already documented elsewhere, and one more negative data point for `g3`'s own root cause (DTX/
comfort-noise mode is not it either). The clean, tight constancy of `g0`/`g2`/`c7` under `DTX_ENABLE`
is real, positive evidence that a genuine, distinct "comfort noise" silence-frame encoding exists
and is reachable exactly as DVSI's manual describes -- plausibly `g0` and/or `g2` carry the
"background noise level" parameter the manual mentions, though this session did not vary the actual
background noise level (only true digital silence was tested) to confirm which field tracks it or
how. A natural next step: repeat this test with several different *levels* of background noise
(not just true silence) to see which of `g0`/`g2`/`c7` moves with noise level -- the same kind of
targeted follow-up that turned the DTMF discovery into a fully decoded, implemented, chip-validated
software module in section 25. Full dataset committed (`dtx_silence_sweep.tsv`).

**Direct follow-up: does `g0`/`g2`/`c7` track actual noise *level* within DTX/comfort-noise mode, or
just silence-vs-not?** `examples/p25_ratet27_capture_dtx_noise_levels.rs` fed 8 increasing
low-amplitude noise peaks (0 through 300, all well under 1% of full scale, `DTX_ENABLE=1`
throughout). **Result: a clean classification threshold between peak 50 and peak 100, not a smooth
noise-level quantizer.** `g0` reads a rock-solid constant `3841` for every peak from 0 through 50,
then jumps to `1045` at peak 100 and climbs slightly (`1045, 1049, 1053, 1057`) through peak 300;
`g2` and `c7` show the same qualitative split (a tight, low-variance cluster below the threshold,
a different cluster above it). This is best read as the VAD's own silence-vs-voice classification
boundary (DVSI's manual states this threshold is -25 dBm0) rather than evidence that `g0` smoothly
tracks background noise level *within* confirmed comfort-noise mode -- every peak this test tried
below the threshold read the identical `g0` constant, with no gradation at all. **This does not
confirm DVSI's "background noise level" claim as cleanly as hoped**: either the tested peak range
(0-50) was too narrow/low-resolution to show real noise-level gradation, or the noise-level
parameter lives somewhere this test didn't isolate (a different block, or a combination). A
finer-grained sweep concentrated just below the threshold (e.g. peaks 0, 5, 10, ..., 50 in small
steps) is the natural next attempt. Full dataset committed (`dtx_noise_levels_sweep.tsv`).

## 27. `PKT_CHANFMT` reveals the chip's own ground-truth classification flags appended to channel packets -- structure determined, full correlation left as a clean-state follow-up

DVSI's manual documents `PKT_CHANFMT` (field `0x15`, a 2-byte data word) as able to make output
CHANNEL packets always include the `ECMODE_OUT` status word (`VOICE_ACTIVE` at bit 1, `TONE_FRAME`
at bit 15) -- real, direct ground truth for what the chip itself classified each frame as, rather
than inferring classification from stimulus type alone as every earlier section had to.
`examples/p25_ratet27_probe_chanfmt_ecmode.rs` sends `PKT_CHANFMT` with `ecmode=0b01` ("always
include") and dumps raw response bytes.

**The manual doesn't show a worked packet example for this option; empirically determined here.**
Confirmed accepted (`[15, 00]` response, matching Table 63's own success code). Every subsequent
CHANNEL packet grew from the normal 24 bytes (4-byte header + 20-byte `CHAND` payload) to 27 bytes
-- **3 extra payload bytes appended immediately after the 18-byte `CHAND` bit data**: a 1-byte field
ID (`0x02`) followed by the 2-byte big-endian `ECMODE_OUT` word itself. Observed values across 10
frames (5 loud-tone, 5 silence) were exactly two distinct 16-bit words, `0x0002` and `0x0402` --
both with bit 1 (`VOICE_ACTIVE`) set, differing only in bit 10, which Table 14 documents as
"Reserved." **This specific test is not yet a clean read**, and is disclosed as such rather than
over-interpreted: `ECMODE_IN`'s own state was left over from the immediately preceding `DTX_ENABLE`
noise-level sweep (section 26) rather than reset to a known baseline, so the bit-10 anomaly and the
lack of any `TONE_FRAME` (bit 15) observation on the loud tone are both plausibly artifacts of that
leftover state rather than genuine findings. **What is solid and reusable**: the exact appended-
field byte offset and format now determined, which is the real prerequisite for a clean follow-up
that resets `ECMODE_IN` to a known state (e.g. `TD_ENABLE` on, everything else off) before checking
whether `VOICE_ACTIVE`/`TONE_FRAME` correlate with `g3`'s own behavior -- a more direct test than
any of this investigation's stimulus-based inference so far, and the natural next step for a future
session.

**The clean follow-up was run immediately, and it is a decisive, ground-truth confirmation.**
Resetting `ECMODE_IN` to a known baseline (`TD_ENABLE` on, everything else off) before re-running
the same `PKT_CHANFMT` probe against four settled stimuli -- a loud tone, silence, real noise, and a
real DTMF digit (`697+1209Hz`, digit "1") -- gives a completely clean, unambiguous result:
**`TONE_FRAME` (`ECMODE_OUT` bit 15) reads exactly `1` for every one of the 5 captured DTMF frames,
and exactly `0` for every one of the other 15 frames (5 each of loud tone, silence, noise)** -- a
perfect, zero-exception match to which stimulus this investigation already knew (section 25)
produces the special row/column DTMF encoding. This is independent, ground-truth confirmation from
the chip's own self-reported classification flag, not just an inference from the resulting wire
pattern -- the strongest possible validation that section 25's DTMF decode is genuinely correct.
`VOICE_ACTIVE` (bit 1) reads `1` for all 20 frames including pure silence, consistent with DVSI's
own manual: that flag's real "0 for frames that don't need transmitting" behavior only applies when
`DTX_ENABLE` is also on, which this clean-baseline test deliberately left off.

**This also gives one more clean negative data point for `g3`**: across the 15 non-DTMF frames
(all reading `TONE_FRAME=0`), `g3` still shows its usual varied, rank-limited behavior -- tone-frame
classification is not a hidden factor behind the ordinary-voice-mode `g3` mystery either, now
confirmed via the chip's own self-reported status rather than inferred from stimulus type.

## 28. Ground-truth `VOICE_ACTIVE` precisely confirms the silence/voice boundary, and ties `g3`'s bistable oscillation directly to confirmed silence for the first time

Combining the two previous sections' tools -- `PKT_CHANFMT`'s ground-truth `ECMODE_OUT` readout
(section 27) and `DTX_ENABLE`'s noise-level sweep (section 26) -- into one probe
(`examples/p25_ratet27_probe_dtx_voice_active.rs`) resolves both of that section's own open
questions at once, using the chip's own self-reported classification rather than inferring it from
wire values.

**The silence/voice boundary is precisely between peak 50 and peak 75** (10 noise peaks tested, 0
through 150): `VOICE_ACTIVE` reads a clean, exceptionless `0` for every frame at peaks 0-50 and a
clean, exceptionless `1` for every frame at peaks 75-150 -- tightening section 26's own "somewhere
between 50 and 100" boundary. `g0` tracks this exactly: a rock-solid constant `3841` for every
`VOICE_ACTIVE=0` frame, then `1041/1045/1049` (increasing with peak) for every `VOICE_ACTIVE=1`
frame -- direct, ground-truth confirmation that `g0`'s earlier-documented threshold behavior
(section 26) really is the chip's own voice/silence decision, not a coincidental artifact of this
investigation's own signal generation.

**A genuinely new, decisive result for `g3`**: across all 100 captured frames, `g3` reads exactly
`2048` in **100% of the 30 confirmed-`VOICE_ACTIVE=1` frames, with zero exceptions** -- completely
stable during real voice activity. During the 70 confirmed-`VOICE_ACTIVE=0` (silence) frames, `g3`
is *mostly* `2048` but flips to `3072` in exactly 10 of them, scattered roughly one-per-peak-level
with no obvious pattern tying it to peak level itself. **This is the first time this investigation
has tied `g3`'s long-documented bistable oscillation (sections 18, 23, 26 all noted variants of a
low-signal "flips between two values" pattern without being able to say what drove it) directly to
a ground-truth chip classification flag rather than inferring it from stimulus type**: it is
specifically and only associated with confirmed silence, never with confirmed voice activity. This
does not yet explain the two-value flip itself (whether it's genuine comfort-noise-level jitter,
matching DVSI's manual claim, or some other adaptive/decaying state that happens to resolve to one
of two values near a decision boundary), but meaningfully narrows the search: any future hypothesis
for `g3`'s behavior now has to explain why it is *perfectly* stable during confirmed voice and only
*occasionally* bistable during confirmed silence, not a general "noisy/unresolved" block. Full
100-frame dataset with per-frame `VOICE_ACTIVE` ground truth committed
(`dtx_voice_active_ground_truth.txt`).

**A real caveat, disclosed rather than glossed over**: all three `VOICE_ACTIVE=1` peaks (75, 100,
150) were generated from the *same* LCG noise realization (seed 42), just rescaled to different
amplitudes -- not three independently-varied signals. Since rescaling a signal preserves its
relative harmonic/spectral shape, `g3`'s perfect constancy across those three points could reflect
that shared shape rather than a genuine "always constant during voice" property that would hold for
differently-shaped voiced content (real speech, tones at different frequencies, etc. -- all of
which showed real `g3` variation earlier in this investigation, section 23's 149-distinct-value
count). The confirmed-silence-vs-confirmed-voice split itself is solid (it doesn't depend on this
caveat), but "`g3` is *perfectly* stable whenever `VOICE_ACTIVE=1`" as a general claim needs
re-testing with genuinely varied voiced content (different noise seeds, real speech, several
distinct tones) under this same `DTX_ENABLE`+`ECMODE_OUT`-visible configuration before being
trusted -- a concrete, cheap next step now that the tooling exists.

**That cheap follow-up was run immediately, and it confirms the caveat was justified -- correcting
the record rather than leaving an over-broad claim standing.** `examples/p25_ratet27_probe_g3_
during_varied_voice.rs` tests `g3` (with the same `DTX_ENABLE`+`TD_ENABLE`+`ECMODE_OUT`-visible
configuration) against genuinely varied content: 5 independent noise seeds, 4 tones at different
frequencies, and 5 different chunks of real recorded speech -- 14 stimuli, 140 captured frames.
**Result: `g3` takes 14 distinct values across these 14 stimuli** (`0, 1024, 1086, 1087, 2048, 2087,
2109, 2110, 2111, 3072, 3134, 3135, 62, 63`), decisively contradicting a general "`g3` is perfectly
stable whenever `VOICE_ACTIVE=1`" claim -- confirming the earlier finding's constancy really was an
artifact of testing only rescaled copies of one noise pattern, not a genuine property of confirmed
voice frames. **The corrected, more precise picture, visible in this richer data**: values cluster
into groups differing by only 1-2 (`1086/1087`, `2109/2110/2111`, `3134/3135`), and the cluster
"bases" (`0, 1024, 2048, 3072`) are exact multiples of `1024 = 2^10` -- consistent with `g3` being a
genuine multi-bit parameter with a coarse ~2-bit component (bits 10-11) and a finer, more slowly-
varying component in the lower bits, roughly matching the block's own established rank-8 ceiling
(`2^8 = 256` reachable values) rather than either "constant" or "fully rank-12 random." This is a
more accurate, if still incomplete, characterization than either the original "stubborn plateau"
framing or the immediately-preceding "perfectly stable during voice" overclaim -- `g3` is real,
multi-valued, content-dependent data, just confined to a smaller-than-full-rank subspace for reasons
still not identified. Both datasets (the confound-affected one and this corrective one) are kept
committed side by side (`dtx_voice_active_ground_truth.txt`, `g3_varied_voice_content.txt`) as a
worked example of exactly the kind of stimulus-diversity trap this whole investigation has run into
more than once.

## 29. `g3`'s real (non-Golay) codeword space derived and implemented -- all 8 of RATET(27)'s FEC sub-blocks now decode with zero errors

Rather than continue chasing `g3`'s semantic meaning, this session's full accumulated capture data
(~3500 distinct frames across every stimulus type tried: tones, 8 noise amplitudes, real speech,
DTMF, dual-tones, chirps, and the various `ECMODE_IN` configurations) was used for what it can
directly give regardless of the semantic question: `g3`'s **real generator matrix**, by GF(2)
row-reduction of every distinct observed value -- the same technique that gave `g0`-`g2`/`u4`-`u6`
their own real generator matrices back in section 23, just not carried through for `g3` at the time
since its rank-8 plateau made it look like an unresolved anomaly rather than a describable code.

**The result is a complete, real, validated 8-dimensional linear code**, not a partial
approximation: every one of the ~150 distinct `g3` values across the full dataset lies exactly in
the span of the derived 8-row basis (checked directly, not assumed). Its structure differs
genuinely from `g0`-`g2`'s own systematic Golay layout: the row-reduced basis's pivot columns are
natural offsets `{0,1,6,7,8,9,10,11}`, not a contiguous `0..8` prefix -- `g3`'s real independent
bits are scattered, not systematic. This also gives a structural (not just observational)
confirmation of an earlier finding: natural offsets 2-5 are `0` in *every one* of the 8 basis rows,
meaning the entire codeword space this basis spans can never produce a `1` there -- the "always
zero" pattern noted since section 23 is now proven, not merely unobserved-as-1 in the sample.

**Implemented as real software**: `g3_encode`/`g3_decode` in `src/ambe/ratet27_fec.rs`, brute-force
minimum-distance decoding over the real 256-codeword space (the same technique `fec.rs`'s own
`golay_decode`/`hamming_decode` use, just over `g3`'s own smaller, real code rather than a
sub-select of the full Golay space). `decode_block` now handles `g3` directly instead of panicking.
Tests: round-trips all 256 codewords with zero distance; confirms the generator structurally never
sets the 4 always-zero positions; and a permanent regression test against 15 real codewords sampled
from the actual chip-captured dataset, all decoding with zero distance.

**`examples/ambe_chip_validate_ratet27.rs` was extended to check all 8 blocks (previously 7), and
re-run live: PASS, all 920 frames (120 synthetic-tone + 800 real-speech) decode with zero errors on
every one of the 8 sub-blocks, `g3` included.** This completes real, tested, chip-validated software
coverage of RATET(27)'s entire FEC/interleave layer -- the "duplicated in software, validated
against the chip" bar this whole session's work has been building toward, for all 8 blocks, not 7
of 8. **What remains explicitly open, stated precisely rather than left vague**: *why* `g3`'s real
information content is confined to this particular 8-dimensional subspace of its nominal 12-bit
Golay capacity, and what real-world voice/signal parameter (if any single one) this subspace
represents -- the FEC/interleave layer is now completely and validatedly duplicated in software;
the deeper semantic/parameter-mapping question for `g3` (and for `g0`-`g2`/`u4`-`u6`'s own specific
parameter identities, per sections 23-24) remains a separate, further piece of work.

## 30. First semantic signal on the Hamming blocks: `u6` is the cleanest pitch-correlated candidate found this session, zero new chip time

Every semantic-layer investigation so far (sections 23-24) focused on the four Golay blocks;
`u4`-`u6` had only their FEC-layer correctness validated, never a semantic look. A zero-chip-time
re-analysis of already-committed data (the RMS-normalized dense pitch sweep and the clean amplitude
sweep, both from section 23) finds a real, and notably *cleaner*, pitch-correlated candidate:

| block | Spearman vs frequency (RMS-normalized) | Spearman vs amplitude |
|---|---|---|
| `u4` | 0.350 | (not tested) |
| `u5` | -0.323 | (not tested) |
| `u6` | **-0.773** | **0.029** |

`u6` shows a strong frequency correlation that survives genuine RMS normalization (ruling out the
amplitude confound that complicated `g0`'s own reading), **and, uniquely among every parameter this
session has tested, essentially zero correlation with signal amplitude** (`0.029`, indistinguishable
from noise) -- checked directly against the clean single-frequency amplitude sweep. This is the
cleanest single-variable-dependent candidate found all session: `g0`/`g1`/`g2` all show real
dependence on *both* pitch and amplitude (consistent with gain/spectral-energy quantizers, section
23), while `u6` appears genuinely pitch-only.

**Tested against the same disciplined same-`L_hat`-vs-crossing control used to refute `g2`'s own
`L_hat` hypothesis (section 24)**: `u6` changes at *every* tested point, both same-`L_hat` pairs and
crossing pairs, at fine 5Hz resolution -- ruling out a discrete `L_hat`-encoding explanation the same
way it was ruled out for `g1`/`g2`, and instead supporting a genuinely continuous, fine-grained pitch
quantizer (the same general character as `g0`, but without `g0`'s confounding amplitude
sensitivity). **This is the strongest, cleanest candidate this investigation has found for "the real
RATET(27) FEC-mode pitch parameter"** -- stated as a strong candidate, not a proven identification,
since (per this whole section's own recurring lesson) a promising correlation deserves the same
skepticism before being called confirmed. A natural next step: the same rigorous confound-elimination
sequence already applied to `g0` (RMS-normalized dense sweep specifically targeting `u6`, silence-
frame corroboration, and a direct comparison against the textbook `dequantize_fundamental_frequency`
formula's own shape) would either confirm or refute this candidacy with the same rigor `g0`'s gain
identification received. Scripts committed (`analyze_u456_pitch_correlation.py`,
`analyze_u6_amplitude.py`), reproducible from already-committed data with zero new chip time.

**A real, significant weakening found immediately on closer inspection, disclosed rather than left
standing**: the correlation analysis above used only the *last* of 8 captured frames per frequency.
Checking all 8 frames per frequency (still zero chip time, same already-committed dataset) shows
`u6` is **highly unstable below roughly 280Hz** -- 5 to 8 distinct values across just 8 frames at
every frequency from 60 through 260Hz -- and only becomes genuinely frame-stable at 340Hz and above
(2-3 distinct values, converging to a rock-solid `0` at 400Hz+). This is the same "chip's own
encoder never fully converges to steady state on a pure tone at low frequencies" behavior already
documented elsewhere in this investigation (the D-STAR harness's own doc comment; section 19's
low-frequency oscillation), not a new problem -- but it means the strong Spearman correlation
reported above is driven largely by "noisy/varied at low frequency, stable-near-zero at high
frequency" rather than a clean, reliable per-frequency quantizer reading throughout the range.
**The zero-amplitude-correlation finding stands independently** (that test used a single fixed,
well-converged 200Hz tone, unaffected by this low-frequency instability), but "cleanest pitch
candidate found this session" overstated how usable `u6`'s individual readings are below ~280Hz --
downgraded here to "a real, amplitude-independent frequency-correlated signal, but noisy/unreliable
per-frame below ~280Hz," a more accurate characterization pending a proper multi-frame-averaged or
majority-vote re-analysis.

**A further, more complete correction after actually testing the "converged region" hypothesis
directly, rather than assuming it.** The previous correction speculated that `u6` might be reliable
above ~280-340Hz, based on the tail of the original 8-frame-per-point sweep. A dedicated follow-up
(`examples/p25_ratet27_capture_u6_converged_range.rs`) tested 15 frequencies from 280Hz through
1000Hz with 20 captured frames each (300 frames total) specifically to check this. **The honest
result is messier than either the original claim or the first correction**: `u6` is genuinely
frame-stable *only* in a narrow island, 340-440Hz (cleanly `~8` at 340-380Hz, exactly `0` at
400-440Hz, all 20/20 frames agreeing) -- but is highly unstable again both **below 280Hz and above
440Hz**, including well above AMBE's documented pitch range (600Hz: 15 distinct values across 20
frames; 800-1000Hz: 5-8 distinct values). This is not the shape a simple monotonic pitch quantizer
would produce (which should stay stable-though-varying throughout, or at worst saturate cleanly at
one end) -- it looks more like `u6` is stable specifically near/at a boundary condition (plausibly
where some other quantity, like `L_hat`, bottoms out or a related computation saturates) and
unstable everywhere else, consistent with a genuinely adaptive or multi-frame-state-dependent
quantity rather than a clean single-frame pitch computation. **The "cleanest pitch candidate"
framing from earlier in this section is retracted, not merely qualified**: `u6`'s real behavior,
tested properly rather than assumed, does not support a simple pitch-quantizer identification at
all. What remains solid from this section: `u6` is real, chip-confirmed, amplitude-independent
signal content (the single-well-converged-tone amplitude test still stands), but what it actually
represents is genuinely unresolved, and this specific investigative thread is closed rather than
left as an open "probably right" lead. Full 300-frame dataset committed
(`u6_converged_range_sweep.tsv`) so a future session sees this exact result rather than re-deriving
it.

## 31. `ratet27_dtx`: DTX-silence classification implemented in software, one real overclaim caught and fixed by its own validation harness

Following the same "implement the confirmed behavior in software" approach that produced
`ratet27_dtmf` (section 25), implemented DTX-silence classification as `ambe::ratet27_dtx`, based on
section 26/28's finding that `g0`, `g2`, and `c7` all read clean constants for confirmed DTX-silence.
The first version required all three fields to match.

**Its own new live validation harness (`examples/ambe_chip_validate_ratet27_dtx.rs`) immediately
caught a real overclaim in that first version.** A careful 10-fresh-frame check found `g2` took 7
different values and `c7` took 3 different values across just 10 confirmed-silence frames -- neither
is actually a reliable constant, despite section 26's own single-frame check suggesting otherwise.
**Only `g0` held up**: exactly `3841` in 10/10 fresh silence frames and exactly `1597` in 10/10
fresh loud-tone frames, zero overlap. The module was corrected immediately to classify on `g0`
alone, and re-validated live: **PASS, 20/20 frames (10 silence, 10 loud tone) classified correctly.**

This is a clean, concrete example of why this investigation insists on validating every "confirmed
constant" claim with enough fresh samples before trusting it, and of the value of writing the live
validation harness immediately alongside the software rather than treating documentation as
sufficient -- the harness itself is what surfaced the problem, on its very first run. `ambe::ratet27_
dtx::is_dtx_silence_frame` is now a real, tested, chip-validated software duplicate of this specific
confirmed chip behavior.

## 32. A likely unifying discovery: `VOICE_ACTIVE` (and probably much of this session's puzzling instability) reflects an *adaptive*, history-dependent baseline, not a fixed threshold

While trying to precisely locate the `VOICE_ACTIVE` threshold (previously only bounded to
"somewhere between peak 50 and 75," section 28) with a binary search, `examples/p25_ratet27_locate_
voice_active_threshold.rs` got a genuinely surprising, contradicting result: **peak 75, which
section 28 found reliably `VOICE_ACTIVE=1` with 80 settling frames, read `VOICE_ACTIVE=0` with only
40 settling frames** -- and, more strikingly, **peak 100 (also previously found `=1`) read
`VOICE_ACTIVE=0` after 250 settling frames of the same constant-level noise.**

**Directly confirmed as an adaptive-baseline effect, not measurement noise.** After 250 settling
frames at peak 100 (confirmed inactive), the stimulus was abruptly switched to a genuinely loud
tone (peak 9000) *without* any resettling. **Frame 0 still read `VOICE_ACTIVE=0`** (the algorithm
hadn't processed the new frame's content yet), but **every one of the next 7 frames immediately
read `VOICE_ACTIVE=1`** -- an instant flip triggered by the *contrast* between the new signal and
the just-adapted baseline, not by the new signal's own absolute level (peak 9000 is, after all,
*also* well above peak 100, which itself just read inactive after enough sustained exposure).

**This means `VOICE_ACTIVE` (and by extension, very plausibly, several other DTX/VAD-adjacent
behaviors this investigation observed) is a genuinely adaptive, multi-frame, history-dependent
computation, not a simple function of the current frame's own content** -- DVSI's own manual
description ("the silence threshold value is -25 dBm0... based upon various adaptive thresholds",
quoted in section 26) already said this plainly, but this investigation's own testing protocol
(a fixed settling-frame count, then capture) implicitly treated it as if a large-enough fixed
settling count would always converge to one true, stimulus-determined answer. **It does not**:
the "true" classification for a given absolute signal level depends on what was sent *immediately
before*, for how long, not just on the level itself.

**This is very likely the root cause, or a major contributor, to several of this session's own
previously-unexplained instabilities**: `g3`'s odd, only-partially-explained bistable oscillation
(sections 18, 23, 26, 29); `u6`'s bizarre "stable only in one narrow frequency island, unstable
everywhere else" shape (section 30, now looking much more like an artifact of wherever a fixed
20-frame settling protocol happened to land relative to `u6`'s own adaptive convergence, rather
than a property of frequency itself); and quite possibly other single-frequency/single-amplitude
correlation results throughout this document that used a fixed settling-frame count without
checking whether that count was actually sufficient for that *specific* transition. **This is a
real, disclosed limitation of this whole investigation's dominant methodology, not a new problem
introduced by this test** -- recorded here because it changes how every earlier single-point
measurement in this document should be read: as "the value after N settling frames from whatever
came before," not "the value this stimulus alone determines."

**A concrete, valuable next step this points to**: repeat the key semantic-layer experiments in
this document (the `g0`/`u6`/`g1`/`g2` correlation sweeps in particular) using a settling protocol
that explicitly holds each stimulus for a long, fixed, generous duration (e.g. 300+ frames,
matching what this section used) and captures many trailing frames once demonstrably converged,
rather than the shorter 40-80-frame settling this document's earlier sections mostly used -- some
"unstable" or "unexplained" earlier findings may resolve cleanly under longer settling, the same
way `VOICE_ACTIVE` did here once tested properly. This reframes several open items in this document
(especially `g3` and `u6`) as *possibly* resolvable with a more patient protocol, not necessarily
requiring new hypotheses about what parameter they represent.

**The "more patience resolves it" hypothesis was tested directly on `u6` and did not hold --
disclosed as a real refinement, not swept under the rug.** `examples/p25_ratet27_capture_u6_long_
settling.rs` re-tested `u6` at 600Hz (one of section 30's worst instability points, 15 distinct
values across 20 frames with 60 settling frames) with **500 settling frames** -- more than 6x
longer than any settling period used elsewhere in this investigation. **Result: still 16 distinct
values across 20 captured frames** (`694, 729, 1685, 733, 696, 732, 664, 726, 697, 725, 669, 728,
700, 728, 662, 1753, 693, 733, 664, 732`) -- no more stable than before. **This means the adaptive-
baseline discovery above does not generalize to explain every instability in this document**:
`VOICE_ACTIVE` genuinely needed more settling time and stabilized once given it; `u6` at 600Hz does
not stabilize even with 500 frames, meaning its own frame-to-frame variation is either driven by
something that never reaches a fixed point for this stimulus (a continuously-adapting quantity with
no steady state, unlike VAD's binary decision) or by genuine per-frame analysis noise unrelated to
settling time at all. The corrected, most honest summary: **settling time matters and was
previously underestimated for at least one real parameter (`VOICE_ACTIVE`), but it is not a
universal explanation for every instability this document has found** -- `u6`'s own instability
specifically remains unexplained by this hypothesis. Dataset committed
(`u6_600hz_long_settling.tsv`).

## 33. `g0`'s amplitude relationship (section 23) fully confirmed settling-independent -- not every earlier finding needed the section 32 correction

Section 32's adaptive-baseline discovery raises a fair question about every earlier single-point
measurement in this document: was the settling period actually sufficient? Directly tested against
`g0`'s own amplitude-quantizer finding (section 23), captured originally with 50 settling frames per
point. `examples/p25_ratet27_capture_g0_long_settling_amplitude.rs` repeats the identical 16-point
amplitude sweep with 300 settling frames (6x longer) and 8 captured frames per point.

**Result: bit-for-bit identical to the original**, and perfectly frame-stable (8/8 identical
captures) at every one of the 16 amplitudes -- `1045, 1561, 1565, 1569, 1569, 1573, 1577, 1581,
1585, 1585, 1589, 1593, 1593, 1597, 1597, 1597`, matching section 23's own values exactly. This is a
clean, positive confirmation that `g0`'s amplitude relationship was already a genuine, fully-
converged steady-state reading, not a settling artifact -- unlike `VOICE_ACTIVE` (genuinely settling-
dependent, section 32) or `u6` (unstable regardless of settling length, also section 32). **Not every
earlier finding in this document needs re-litigating under section 32's discovery**: some, like this
one, hold up cleanly on direct re-test with a much more patient protocol. Dataset committed
(`g0_long_settling_amplitude_sweep.tsv`).

**`g1`/`g2`, decoded from this same already-captured long-settling dataset at zero extra chip time,
do *not* clean up the way `g0` did -- another real, disclosed negative result.** `g1` and `g2` both
remain noisy across the same 16 amplitudes with 300-frame settling (`g1`: `3410, 1358, 1322, 1362,
3406, 3410, 3410, 1362, 1358, 3410, 3410, 3410, 1362, 1370, 3370, 3370`; `g2`: similarly scattered),
with weak Spearman correlations (`g1`: `0.279`, `g2`: `0.479`) essentially unchanged from the
original short-settling reading. This rules out insufficient settling as the explanation for `g1`/
`g2`'s own weaker, messier amplitude relationship (section 23) -- unlike `g0`, more patience does not
resolve it. Consistent with, though not proof of, `g1`/`g2` depending on something more complex than
a single scalar amplitude parameter (plausibly genuine higher-order spectral-shape content, which a
single fixed-frequency sawtooth's amplitude alone wouldn't cleanly parameterize) rather than being an
under-settled version of the same simple gain relationship `g0` shows.

## 34. First semantic look at `u4`/`u5` against amplitude: `u4` is notably more stable than `g1`/`g2`/`u5`, zero extra chip time

Also decoded from the same long-settling amplitude dataset (section 33), at zero extra chip time:
`u4` and `u5`, neither previously looked at semantically (only `u4`'s separate DTMF column-encoding
role, section 25, was known). Result:

| block | Spearman vs amplitude | distinct values per amplitude (8 frames each) |
|---|---|---|
| `u4` | 0.538 | **exactly 2**, every amplitude |
| `u5` | 0.132 | 6-8 (noisy) |

`u4` stands out: while its correlation with amplitude is only moderate (not as clean as `g0`'s own
0.9+-class relationship), it is **remarkably more frame-stable** than `g1`, `g2`, or `u5` -- settling
into exactly one of two values at every tested amplitude, rather than jumping among 6-8 values like
`u5`. This is consistent with a genuine two-state or coarsely-quantized real parameter (plausibly a
voicing decision, matching `bit_prioritization`'s own textbook role for such a field, though this is
speculation, not confirmed) rather than either a clean continuous quantizer or pure noise. `u5`
remains as unstable and weakly-correlated as `g1`/`g2`/`u6` -- no clean semantic signal found for it
yet by any test in this document. Both are recorded as open, real, current-state findings rather
than being left completely uninvestigated.

**A precise structural detail found while looking closer at `u4`'s own "exactly 2 values" pattern
at fixed 200Hz**: at every one of the 16 tested amplitudes (fixed frequency, varying amplitude),
`u4`'s two alternating values differ by **exactly `53`**, with zero exceptions (`454/507`,
`1862/1915`, `1734/1787`, `1606/1659`, `1990/2043`). The most likely explanation, consistent with
standard vocoder design practice: `u4` (or whatever underlying quantity it encodes) is subject to
**error-feedback/dithered quantization** -- alternating between two adjacent quantizer levels
frame-to-frame to preserve the *average* value's fidelity despite coarse per-frame resolution, a
well-known technique (related to noise-shaping/dither in ADPCM and similar codecs).

**Corrected immediately on checking a second, independent dataset (zero extra chip time): the
dither step is not a single universal constant -- it varies with frequency.** Re-running the same
check against the RMS-normalized dense pitch sweep (fixed amplitude, varying frequency, section 23)
found `u4` showing more than 2 distinct values at many frequencies, with a dominant gap clustering
around `~53` at some frequencies and `~42-45` at others. This first pass used a hand-rolled Hamming
decode and only 18 of the dataset's 20 frequencies, and was itself re-derived properly below.

**Redone rigorously with a committed tool using the crate's own real `decode_block`, not a
reimplementation** (`examples/ratet27_analyze_u4_dither_by_frequency.rs`, output for all 20
frequencies committed as this section's own supporting evidence) -- and cross-checked against a
freshly captured, independent dataset at 300 frames of settling per frequency (5x the original 60,
matching section 33's proven-sufficient protocol) to rule out under-settling as the explanation for
the multi-value spreads:

| freq (Hz) | `L_hat` | distinct `u4` values (60-frame settling) | distinct `u4` values (300-frame settling) |
|---|---|---|---|
| 60  | 61 | 5 values, max gap 7  | 5 values, max gap 5 |
| 80  | 46 | 4 values, max gap 40 | 4 values, max gap 40 |
| 100 | 37 | 3 values, max gap 45 | 3 values, max gap 45 |
| 120 | 30 | 3 values, max gap 45 | 3 values, max gap 45 |
| 140 | 25 | 3 values, max gap 3  | 3 values, max gap 3 |
| 160 | 23 | 1 value (no dither)  | 4 values, max gap 29 |
| 180 | 20 | 2 values, gap 53     | 2 values, gap 53 |
| 200 | 18 | 2 values, gap 53     | 2 values, gap 53 |
| 220 | 16 | 4 values, max gap 53 | 4 values, max gap 53 |
| 240 | 14 | 3 values, max gap 53 | 3 values, max gap 53 |
| 260 | 13 | 3 values, max gap 53 | 3 values, max gap 53 |
| 280 | 12 | 5 values, max gap 45 | 4 values, max gap 53 |
| 300 | 12 | 5 values, max gap 53 | 4 values, max gap 53 |
| 320 | 11 | 5 values, max gap 48 | 5 values, max gap 48 |
| 340 | 11 | 5 values, max gap 50 | 4 values, max gap 51 |
| 360 | 10 | 3 values, max gap 54 | 4 values, max gap 51 |
| 380 |  9 | 3 values, max gap 52 | 3 values, max gap 52 |
| 400 |  9 | 2 values, gap 2      | 2 values, gap 2 |
| 420 |  8 | 2 values, gap 2      | 2 values, gap 2 |
| 440 |  8 | 4 values, max gap 53 | 4 values, max gap 53 |

**This settles two things at once, one confirming and one disconfirming the earlier speculation.**
First, confirmed: the near-identical values under 60 vs. 300 frames of settling at nearly every
frequency (most match exactly or come within 1-2 values of the short-settling reading; `160`Hz is
the one clear exception, going from a single stable value to 4 distinct values under 5x longer
settling -- the opposite of what an under-settling artifact would predict, and left as its own small
open oddity rather than glossed over) means the multi-value spreads at low frequencies and the
varying dominant gap are **real chip behavior, not a settling artifact** -- the same conclusion
section 32 already reached for `u6` by the same method, now independently replicated for `u4`.
Second, disconfirmed: a clean, monotonic `L_hat`-dependent quantizer step size
does **not** hold. `L_hat=8` gives a gap of `2` at 420Hz but `53` at 440Hz -- the same nominal
harmonics count producing wildly different dither behavior rules out `L_hat` alone as the
explanation, at least not via a simple one-to-one step-size mapping. (`L_hat=12` at 280Hz/300Hz *did*
give the same value set both times, so `L_hat` may still matter in some frequencies' cases -- the
data doesn't support a clean universal rule either way.) The originally reported "exactly 53 at every
amplitude" result (fixed-200Hz, varying amplitude) remains correct on its own terms; what's corrected
here is only the claim that this generalizes to a single frequency-independent constant.

**A confound worth naming rather than ignoring**: the pitch-sweep stimulus generator computes one
160-sample buffer per frequency and resends it unchanged every frame, so any frequency whose period
does not evenly divide the 160-sample frame has a phase discontinuity ("click") at every frame
boundary, adding spectral content the nominal frequency alone doesn't have. Of the 20 tested
frequencies, only `100`, `200`, `300`, and `400`Hz divide 160 exactly (the analysis tool's own
`divides_160` column confirms this directly rather than leaving it computed by hand); every other
frequency's stimulus carries this click. This does not track the `~53`-vs-`~42-45` split cleanly
either: `100`Hz is click-free yet lands in the `~42-45` cluster, while `200`Hz and `300`Hz are also
click-free yet land in the `~53` cluster -- click-free frequencies span both clusters, so click
presence/absence does not appear to be the primary explanation for the split -- but it is a real
methodological caveat on the whole sweep, not just
this block's own analysis, and is left as an open item for whoever next revisits pitch-swept stimuli
against this chip: a phase-continuous stimulus generator (carrying phase across the frame boundary
rather than resetting it) would remove this confound entirely.

**A quick cross-check of the same dataset against the other blocks**: `g1`/`g2`/`u5`/`u6` show no
comparably clean pattern (their distinct-value gaps are irregular, no single repeated difference).
`g3`, however, **independently reproduces its own already-established "steps of exactly `1024`"
structure** (section 29's basis analysis) on this completely different dataset (fixed 200Hz tone,
varying amplitude, rather than the varied-content test that originally found it) -- e.g. amplitude
100 gives exactly `{1024, 2048, 3072}`, amplitude 364.7 gives values including an exact `1024` gap
(`2103` to `3127`). A clean independent replication of an already-documented structural fact, not a
new finding, but useful confirmation from a second, unrelated dataset.

## 35. `ratet27_dtx` upgraded to real chip ground truth, and both of its own open questions resolved together

Section 31's `is_dtx_silence_frame` classifier was validated only by stimulus inference ("we sent
digital silence, so this should read as silence") -- the same weakness DTMF had before `PKT_CHANFMT`'s
`ECMODE_OUT` field gave it real chip-reported ground truth (section 25). The module's own doc comment
also carried an explicit, untested open question: whether `DTX_SILENCE_G0`'s classification is itself
adaptive/history-dependent the way section 32 found `VOICE_ACTIVE` to be. A single new tool
(`examples/p25_ratet27_dtx_ground_truth_and_adaptive_check.rs`) reads `g0` and the chip's own
`VOICE_ACTIVE` flag together and resolves both at once.

**Ground truth, ecmode-out-confirmed**: across a noise-peak sweep from 0 through 50 (all confirmed
`VOICE_ACTIVE=0` via `ECMODE_OUT`), `g0` read exactly `3841` on every single one of 60 captured
frames -- full agreement between the wire-bit classifier and the chip's own status flag, not just
consistency with what was sent.

**A genuine new finding, not just validation**: at noise peak 75 and peak 100 -- both still
`VOICE_ACTIVE=0` (below the roughly-50-to-75 activation threshold section 28/33 already located) but
audibly noisier than near-silence -- `g0` read `3844`-`3845` and `3856`-`3857` respectively, rising
smoothly with the actual noise level while staying well above the (much lower) values seen once `VOICE_ACTIVE`
flips to `1`. This is a real, positive confirmation of DVSI's own manual claim that this field
reflects a "background noise level," a claim section 26's own noise-level sweep tested and found
inconclusive. The corrected understanding: `g0` genuinely does encode a continuous noise-floor
reading while the chip judges a frame inactive; it just doesn't hold exactly `3841` outside of true,
near-zero-noise silence. `is_dtx_silence_frame` was never a general "is this frame inactive"
classifier and isn't changed by this finding -- it correctly identifies genuine silence specifically
-- but the module's doc comment was corrected to state this precisely rather than leave the
"background noise level" question marked inconclusive when it is now confirmed.

**The abrupt-switch protocol (section 32) moved `g0` and `VOICE_ACTIVE` in lock-step**: after
settling at peak 100 (`g0=3857`, `VOICE_ACTIVE=0`), the frame immediately following an abrupt switch
to a loud, unrelated tone flipped both `VOICE_ACTIVE` to `1` and `g0` to a much lower value within a
single frame -- no separate lag between the two fields.

**The history-dependence open question is now answered, not merely retested**: 600 consecutive
frames of sustained peak-100 noise never once produced `g0=3841`. The elevated noise-floor reading is
stable under sustained exposure, not a slow drift back toward the true-silence constant. So while
`VOICE_ACTIVE` is confirmed adaptive/contrast-based (section 32), `g0`'s own noise-floor reading was
not, at least under this test -- it tracked genuine noise level consistently rather than habituating
to it, and `DTX_SILENCE_G0`'s exact-match behavior is safe to rely on without worrying that a
merely-quiet-but-sustained signal will eventually also read as `3841`.

**A tempting broader classifier (`g0 >= DTX_SILENCE_G0`, since every confirmed-inactive `g0` value
seen above is `>= 3841` and every confirmed-active value recorded anywhere in this session's own
data is well below `2400`) was checked exhaustively against every committed capture dataset before
being added to the module -- and was falsified, for three distinct, genuine reasons, not one.** A
purpose-built tool (`examples/ratet27_verify_dtx_g0_threshold.rs`) decoded `g0` from all 16 other
committed datasets (roughly 5,600 frames) and found real counterexamples:
1. **DTMF and forced-tone frames use a completely different `g0` encoding that overlaps this exact
   range by construction** -- section 25 already established `g0 = 4032 + row_index` for DTMF, and
   `real_dtmf_sweep.tsv` confirms every single one of its 160 frames reads `g0 >= 3841`, as does every
   frame of `tone_send_forced_sweep.tsv`. A naive `>= 3841` classifier would misclassify every DTMF
   digit and every forced-tone frame as silence -- exactly the kind of false positive that would
   silently drop real, intentional signaling content in an actual transmitter.
2. **A full-amplitude 60Hz tone -- likely below this codec's voice-band filtering -- reads as
   "quiet" by this measure.** `dense_pitch_sweep_57to444hz.tsv`, `rms_normalized_pitch_sweep`, and
   `u4_long_settling_pitch_sweep.tsv` all show their 60Hz rows reading `g0=3945` on every one of 8
   frames, consistent across three independent captures -- a real, repeatable phenomenon, not
   dataset noise, and a useful independent clue that this codec's internal energy measure reflects
   perceptually/passband-filtered signal content, not raw PCM amplitude.
3. **Several un-converged pure sine tones intermittently spike into this range without representing
   genuine silence.** `all_stimuli_2687frames.tsv` shows `sine_200`, `sine_400`, `sine_250`,
   `sine_320`, several `extreme_sine_*` tones, and even one single frame of `real_speech`
   intermittently reading `g0 >= 3841` mid-stream, alongside many other frames of the same stimulus
   reading normally -- consistent with this project's own well-documented finding that this chip's
   encoder never fully converges to steady state on a pure tone (`ambe_dstar`'s own doc comment).

**The conclusion is a real negative result worth keeping, not a failed attempt to hide**: there is no
safe range-based broadening of `is_dtx_silence_frame` using `g0` alone. The chip's own `VOICE_ACTIVE`
status flag (via `PKT_CHANFMT`'s `ECMODE_OUT` field) remains the only reliable ground truth for "is
this frame inactive," and is not recoverable from `g0` alone within the ordinary 144 wire bits
without that extra field. `is_dtx_silence_frame`'s narrow exact-match-only scope, calling only
genuine near-zero-noise silence, was correct as originally shipped -- broadening it, as briefly
considered, would have been a regression, not an improvement, and this section exists so a future
session doesn't re-propose the same broadening without first re-running this same check.

**One more discriminator checked, zero extra chip time, before generalizing "not recoverable" beyond
`g0` specifically**: `g1` is bimodal in essentially every other dataset in this document (a "low"
cluster around 1200-1360, a "high" cluster around 3370-3420 -- e.g. section 36's own baseline).
`examples/ratet27_analyze_g1_during_dtx_inactive.rs` checked the already-committed
`dtx_silence_sweep.tsv` directly: across both confirmed-inactive content labels (`dtxon_silence`,
`dtxon_lowlevelnoise`, 20 frames total), `g1` **never once** reached the high cluster. Both
confirmed-active labels (`dtxon_tone`, `dtxoff_tone`, 20 frames total) reached the high cluster in
**6 of 10 frames each** -- not every frame; the other 4 of 10 landed in the same low cluster inactive
frames also use. So the asymmetry is real (inactive frames never reach high; active frames reach it
often but not always) but weaker than a clean per-frame split -- a per-frame classifier built on `g1`
alone would still misclassify roughly 40% of active frames as "looks inactive." This is a real,
suggestive pattern in a small sample (40 frames, 2 labels each side, not a systematic sweep) -- worth
recording as a concrete lead for `(g0, g1)` jointly discriminating active from inactive better than
`g0` alone, but not yet enough data to promote to a validated classifier the way
`is_dtx_silence_frame` itself was. **One specific frame set would settle the ambiguous case
directly**: `dtxoff_noise1` and `dtxon_noise1` both show `g0 ≈ 3957` (`g0`'s own "quiet-looking"
range) while also reaching `g1`'s high cluster -- but these captures predate `PKT_CHANFMT`, so
whether the chip's own `VOICE_ACTIVE` was actually `1` or `0` for these specific frames is unknown.
Re-capturing `noise1`-equivalent content with `ECMODE_OUT` enabled would show directly whether `g1`
correctly overrides `g0` here (real evidence for the joint classifier) or gives a false active (a
real limit on it) -- the single most informative next frame to capture for this specific question.

## 36. A systematic `ECMODE_IN` bit sweep finds a genuine, previously untested feature at bit 8 -- and a clean independent reconfirmation of `TS_ENABLE` (bit 14)

Section 21's own record names an explicit, untried lever for `g3`'s stubborn rank-8 plateau: "an
encoder feature/mode this investigation's SPEECH-packet-only testing never engages (frame-repeat,
DTX, or another documented `ECMODE`/`DCMODE` flag)." Only bits 11 (`DTX_ENABLE`) and 12 (`TD_ENABLE`)
had ever been tested directly (section 24); the other 13 of `ECMODE_IN`'s 16 bits were completely
untried. This project has no locally stored copy of DVSI's own bit-name table, but that isn't needed
to test the real question empirically: does setting each individual bit change the chip's actual
output on identical content at all?

**Method**: `examples/p25_ratet27_ecmode_bit_sweep.rs` sets `ECMODE_IN` to exactly one bit at a time
(0 through 15, skipping the two already-tested), resettles a fixed 200Hz/6000-amplitude sawtooth for
60 frames, captures 8 frames, and decodes all 8 blocks -- including `g3` via this crate's own real
`g3_decode` now that section 29 derived it, not an assumed-Golay stand-in. `u4` serves as the
cleanest canary: in the `ECMODE_IN=0x0000` baseline it holds to exactly its known 2-value dither
(`{1734, 1787}`, section 34) with zero other jitter, so any bit that leaves `u4` outside that exact
2-value set has demonstrably changed something real, not just landed in one of the chip's already-
known noisy corners.

**Result, run once against the live chip, all 16 bits in a single pass**: 13 of the 14 newly-tested
bits (`0`-`7`, `9`, `10`, `13`, `15`) leave `u4` at exactly `{1734, 1787}` -- zero effect detectable by
this test. Two bits stand out clearly:

- **Bit 14 reproduces `TS_ENABLE`** (already documented in section 25) exactly: `g0=4048`,
  `g1=3712`, `g2=0`, `g3=0`, `u4=80`, `u5=0`, `u6=0`, `c7=0`, identical across all 8 frames,
  independent of the actual sawtooth content -- a clean, independent reconfirmation via a completely
  different test harness of an already-known finding, not a new one.
- **Bit 8 is new.** `g0` stays exactly `1597` (matching baseline -- the gain/amplitude reading is
  unaffected, consistent with the stimulus's amplitude being unchanged) and `g3` stays within its
  already-established subspace (`{191, 255}`, both already-seen members -- **this does not resolve
  the `g3` plateau mystery**), but `g1`, `g2`, `u4`, `u5`, `u6`, and `c7` all shift to genuinely new,
  previously-unseen value clusters: `u4` moves from `{1734, 1787}` to `{1960, 1961, 1966}` (a ~200-
  unit shift, far larger than any dither step documented elsewhere for this block), `u6` reaches values
  as low as `295` (well below its baseline floor of `682`), and `g2`/`u5`/`c7` all land in ranges with
  no overlap with baseline. **One discriminating observation worth naming directly**: section 34's
  amplitude sweep shows `u4` climbing toward its own higher values as input amplitude rises, and bit
  8's shifted `u4` reading (`1960-1966`) sits at the high end of that same amplitude-correlated range
  -- yet `g0`, this chip's own primary gain/amplitude indicator, reads bit-for-bit identical to
  baseline (`1597`) the entire time, on literally the same, unchanged input signal. Either `u4` is
  reacting to something bit 8 changed that isn't the same "amplitude" `g0` tracks, or bit 8 performs
  gain-dependent processing (consistent with companding, or an AGC-like effect) that manifests in the
  higher-order coefficients without moving `g0` itself. This doesn't identify bit 8 conclusively, but
  it's a real, concrete clue narrowing the field of plausible explanations, not just a list of names:
  this is a real, reproducible, previously undocumented `ECMODE_IN` feature -- most plausibly one of
  the audio-processing toggles section 21 already named as existing but untested (noise suppression,
  echo cancellation, or companding), though this project cannot name which one precisely without
  DVSI's own bit-table reference. Recorded here, with the raw sweep output and a committed analysis
  script
  (`ecmode_bit_sweep_output.txt`, `analyze_ecmode_bit_sweep.py`), as a concrete, bounded, real lead
  for whoever next has manual access or wants to characterize bit 8's effect on real speech content.

**Scope discipline, stated explicitly**: this was one targeted sweep using a different technique
(control-word bit variation, not another audio-stimulus variation) specifically because section 21
named it as a real untried lever, not the start of a broader semantic hunt. `g1`/`g2`/`u5` remain
noise-limited under every content-variation test already tried in this document; bit 8's discovery
doesn't change that -- it opens one new, narrow, well-defined follow-up (what bit 8 does, precisely)
rather than reopening the general semantic-identity question.

## 37. Consolidating scattered validation code into one reusable decoder: `ambe::ratet27_frame`

Every RATET(27) validation/probe tool in this project -- roughly 15 of them by this point -- repeated
the same boilerplate: unpack an 18-byte channel-frame payload into 144 wire bits, then call
`decode_block` once per block. This is fine for one-off probes, but it means "known and duplicated in
software" was true block-by-block, scattered across examples, rather than as one coherent, reusable,
tested decoder a real consumer could actually call.

Added `ambe::ratet27_frame` with a single public entry point, `decode_frame(&[u8; 18]) ->
Ratet27Frame`, ties together `ratet27_wire_format`/`ratet27_fec` (all 8 blocks plus each one's FEC
distance) with the two confirmed interpretations already established elsewhere in this document:
`Ratet27Frame::dtmf_digit()` (delegating to `ratet27_dtmf::decode_dtmf_digit`, section 25) and
`Ratet27Frame::is_dtx_silence()` (delegating to `ratet27_dtx::is_dtx_silence_frame`, section 31/35).
Deliberately scoped narrow, per direct advice: no `PKT_CHANFMT`/`ECMODE_OUT` parsing (that's packet-
layer, not frame-layer), and no `VOICE_ACTIVE`-from-wire-bits classifier (section 35 already
established that isn't reliably possible from `g0` alone). (Update, section 40: `ECMODE_OUT`'s own
bits are since confirmed as a genuine, global, three-rate-validated ground-truth mechanism in their
own right, independent of any per-rate frame decoder -- bit 1/`VOICE_ACTIVE` in section 27/28, and
bit 15/`TONE_FRAME` in section 40's `examples/p25_ambe_plus_2_and_dstar_tone_frame_ground_truth.rs`
and `examples/ambe_chip_validate_dstar_tone.rs` -- so a future session extending packet-layer
handling should start from those two probes rather than rediscovering that `ECMODE_OUT` exists.)

Tested against real captured frames, not synthetic ones: a real DTMF digit from `real_dtmf_sweep.tsv`
(confirms `g0=4032`, `u4=80`, `g1=2944`, `g2`/`g3`/`u5`/`u6`/`c7=0`, `dtmf_digit() == Some((0,0))`) and
a real confirmed-silence frame from `dtx_silence_sweep.tsv` (`dtxon_silence`, confirms `g0=3841`,
`is_dtx_silence() == true`). Migrated one consumer, `examples/ambe_chip_validate_ratet27.rs` (the
project's own PASS/FAIL chip-validation harness), to use `decode_frame`/`is_zero_error()` in place of
its own repeated `decode_block` loop, and re-ran it live against the real chip: **identical PASS,
920/920 frames zero-error**, confirming the consolidation preserves exact behavior rather than
silently changing it. The other ~14 examples are deliberately left as-is -- migrating every one of
them would be churn without new validation value, not a step toward "duplicated in software."

## 38. Bit 8 reliably produces `g0=1597` under multiple tests -- and chasing why the first amplitude probe's baseline column looked wrong turned up a genuinely new, reproducible chip behavior

Section 36 found `ECMODE_IN` bit 8 real but couldn't distinguish noise suppression, echo
cancellation, or companding. The first follow-up attempt (`examples/p25_ratet27_ecmode_
bit8_amplitude_probe.rs`) ran a 16-point amplitude sweep, alternating `ECMODE_IN=0` ("baseline") and
`ECMODE_IN=1<<8` back to back at each amplitude, and reported a "smooth, monotonically shrinking
boost" from the baseline toward a "saturated ceiling" of `1597` -- **this framing overclaimed on two
counts, both caught before being relied on further:**

1. Since bit 8's own reading was a flat constant (`1597`) at every amplitude, the "delta" column was
   arithmetically just `1597 - baseline`, i.e. exactly baseline's own already-known monotonic curve,
   inverted. It carried zero information about bit 8 beyond the one real fact (the constant itself).
   Calling `1597` a "saturated ceiling" under bit 8 also imported an inference: `g0` is a 12-bit
   Golay data field with no intrinsic "loudest" meaning outside baseline's own documented curve --
   the fact is "`g0` reads `1597` under bit 8," not "bit 8 pins `g0` to its ceiling."
2. **The "baseline" column itself didn't match the already-committed `g0_long_settling_
   amplitude_sweep.tsv`** (section 23/33's own confirmed curve) beyond the very first point -- e.g.
   `1049` vs. the committed `1561` at amplitude 138.2. The originally-proposed explanation
   (insufficient settling between the sweep's alternating baseline/bit-8 captures, failing to clear
   an adaptive carryover from the immediately-preceding bit-8 excursion) was a plausible hypothesis,
   not yet a confirmed one -- and directly testing it, below, showed the real explanation is more
   interesting than that.

**A single clean re-measurement first isolated the one solid fact.**
`examples/p25_ratet27_ecmode_default_state_probe.rs`: one fixed amplitude (138.2, matching the
discrepant point exactly), 300 frames of settling, three conditions in sequence from a fresh
connection -- `ECMODE_IN` never touched at all (true power-on/reset default), `ECMODE_IN` explicitly
`0x0000`, and `ECMODE_IN` explicitly `1<<8`. Untouched-default and explicit-zero both read `g0=1561`,
identical to each other and matching the committed section 23 data exactly (ruling out "bit 8 is on
by default" as an explanation for the discrepancy); `ECMODE_IN=1<<8` read `g0=1597`.

**A temporal abrupt-switch test then ruled out a converging AGC specifically.**
`examples/p25_ratet27_ecmode_bit8_abrupt_switch.rs` holds bit 8 on throughout, settles 250 frames at a
quiet amplitude (`g0=1597`), then abruptly switches to a 90x louder tone with **no resettling**. If
bit 8 were an AGC converging to a target level, the sudden jump should produce at least a brief
transient before re-settling. It reads `g0=1597` in the very first frame after the switch, with zero
visible transient across 20 frames -- evidence against a converging AGC, though it does not by
itself distinguish a static transform from some other fixed-output behavior.

**Directly testing the settling-shortfall hypothesis found it wrong, and something more interesting
in its place.** `examples/p25_ratet27_ecmode_bit8_off_transient.rs` settles 300 frames at amplitude
138.2 with bit 8 ON (`g0=1597`, confirmed), then switches `ECMODE_IN` to `0x0000` with **no change to
the input signal** and logs `g0` for 120 frames (raw output committed as
`ecmode_bit8_off_transient.txt`; the run below was independently reproduced a second time with an
almost identical sequence, ruling out one-off noise):

```
frame 0:  1597   (config not yet applied)
frame 1:  1652   (transient)
frame 2:  1627   (transient)
frames 3-11: 1561   (matches the clean baseline exactly -- reached in ~3 frames, not 60)
frame 12: 2201   (one-frame spike)
frames 13-119: 1049   (a second, different, stable value -- holds for the rest of the 120-frame log)
```

This directly refutes the settling-shortfall hypothesis: `g0` reaches the correct clean baseline
(`1561`) within about 3 frames of switching bit 8 off, far faster than the original sweep's 60-frame
settling window -- if under-settling were the whole story, the sweep's baseline readings should have
been correct, not off by 512. Instead, **`g0` spontaneously jumps away from its own correct baseline
value to a second, different stable value (`1049`) partway through the log, at a highly consistent
frame count across two independent runs**, with a one-frame spike marking the transition. (`1049`
itself was checked against this document's other data before drawing any conclusion from it: it is
not a special or previously-flagged constant -- it also shows up as an ordinary value in unrelated
content, e.g. real speech frames in `u_vector_speech_correlation_600frames.tsv` and a ramp stimulus
in `all_stimuli_2687frames.tsv`. A coincidentally identical value in the unrelated
`dtx_noise_levels_sweep.tsv` dataset was checked and does not indicate a real connection -- that
dataset uses broadband noise content with `DTX_ENABLE=1`, a different stimulus and chip state
entirely.) Two different "valid-looking" readings for the literal same, unchanging input, separated
by a reproducible spike at a consistent frame count, is a new instance of this document's own
repeatedly-documented finding that this chip's encoder does not always converge to one steady value
on an unvarying pure tone (`ambe_dstar`'s own doc comment, invoked for `u6`/`u4`/several frequencies
elsewhere in this document) -- extended here to show even `g0`, one of this document's most
consistently "clean" and reliable blocks, is not immune to it at this specific amplitude, just on a
longer timescale (dozens of frames) than the 8-frame capture window most tests in this document use
to sample it. **The original sweep's `1049` "baseline" was, on this evidence, most likely a genuine,
real state this exact stimulus can settle into over an extended window -- not a contamination
artifact -- and the reproducible ~12-frame transition point is a real, concrete, unexplained
observation left for whoever next investigates this chip's longer-timescale gain behavior.**

**Honest current state**: bit 8 reliably produces `g0=1597` (confirmed independently three separate
ways: the original sweep, the clean single-point re-measurement, and the abrupt-switch test's settled
segment) and shows no transient on an abrupt 90x amplitude jump. Whether it is a static compander, a
fixed reference-value override, or something else is not resolved by this document's own testing and
would need DVSI's own bit-name table to settle. Separately, and just as concretely: this
investigation of bit 8 surfaced real evidence that `g0` itself can take on more than one stable value
for literally the same unchanging input over a long enough observation window, at least at this one
amplitude -- a genuine, reproducible finding in its own right, independent of what bit 8 turns out to
be. `u4`'s own values under bit 8 are noisy in the same way `u4` is noisy everywhere else in this
document (section 34) and are not treated as informative about bit 8's identity specifically.

## 39. Bit 8 identified with certainty: `CP_ENABLE` (Compand Enable), direct from DVSI's own primary-source manual -- and every prior empirical bit result cross-checks clean

DVSI's own "AMBE-3000R Vocoder Chip Users Manual" (Version 1.4, March 2013) is publicly archived
(the D-STAR digital-voice community has referenced it for years for exactly this kind of
interoperability work) at a well-known ham radio site; a copy is stored in `hams_com` (private repo)
at `docs/references/dvsi_ambe/` for internal reference, with a plain-text extraction for grepping.
Its own `Table 13 ECMODE_IN Flags` resolves bit 8, and every other `ECMODE_IN` bit this document has
ever tested, directly and with certainty. Bit names only below (the manual's own longer prose
descriptions are not reproduced here -- every page of the source document is marked confidential, and
a full table is a more substantial excerpt than this document's established practice of quoting a
single descriptive sentence; see this section's own closing note):

| Bit | Name |
|---|---|
| 0-5 | Reserved |
| 6 | `NS_ENABLE` (Noise Suppressor Enable) |
| 7 | `CP_SELECT` (Compand Select: µ-law/A-law) |
| **8** | **`CP_ENABLE` (Compand Enable)** |
| 9 | `ES_ENABLE` (Echo Suppressor Enable) |
| 10 | Reserved |
| 11 | `DTX_ENABLE` (already confirmed empirically, §26/§28/§31) |
| 12 | `TD_ENABLE` (Tone Detect Enable, on by default at reset -- already confirmed empirically, §25) |
| 13 | `EC_ENABLE` (Echo Canceller Enable) |
| 14 | `TS_ENABLE` (Tone Send Enable -- already confirmed empirically, §25) |
| 15 | Reserved |

The one sentence this document relies on directly, quoted from the manual's own `CP_ENABLE`
description: **"If `CP_ENABLE=1`, then companding is enabled... If `CP_ENABLE=0`, then companding is
disabled and all speech samples are 16-bit linear."**

**Bit 8 is `CP_ENABLE` -- companding, exactly the candidate section 38's own reasoning favored, now
confirmed rather than inferred.** Read precisely, `CP_ENABLE` doesn't apply a gain curve to the
signal -- it tells the chip **what format the incoming samples are in**. This investigation always
sent genuine 16-bit linear PCM. With `CP_ENABLE=1` set, the chip instead interpreted those same
linear samples *as if they were µ-law-encoded bytes* and expanded them accordingly -- a format
mismatch, not a loudness boost. Expanding ordinary linear PCM values through a µ-law decompression
curve produces a garbled, essentially-arbitrary waveform whose apparent amplitude has little
relationship to the original signal's real loudness, which explains the flat `g0=1597` reading far
better than any gain-curve story: the chip wasn't reading a boosted-but-still-meaningful signal, it
was reading noise-like content that happened to consistently read as loud. **This also means section
38's own "evidence against a converging AGC" framing was answering the wrong question** -- there was
never an AGC (or any gain control) to converge; the abrupt-switch test's "amplitude jump" was really
a jump between two different genuinely-mismatched (format-confused) inputs, not a loudness change the
chip was trying to normalize. The numeric specifics of why the misinterpreted signal lands at exactly
`1597` remain unexplained and are not pursued further. **A real, separate, practical implication worth
recording**: this also confirms `CP_ENABLE` (and the encoder-side/decoder-side pair `PKT_ECMODE`/
`PKT_DCMODE` both expose it) is how this chip would be told to accept genuine µ-law/A-law-encoded
samples directly, if a future real telephony interface ever needed to feed it audio in that format
instead of linear PCM -- confirmed to actually take effect, not just documented as existing.

**Every other empirically-tested `ECMODE_IN` bit in this document checks out clean against the
manual, with zero contradictions**: `DTX_ENABLE` (11), `TD_ENABLE` (12, confirmed on-by-default), and
`TS_ENABLE` (14) all match their manual descriptions and this document's own independently-derived
empirical behavior exactly (§25/§26/§28/§31), and bits 6/7/9/10/13/15's real, confirmed no-effect
result under a clean synthetic tone (§36) is now understood as the expected result for noise/echo-
only or no-op reserved bits -- though for bit 9 (`ES_ENABLE`, echo suppression) and bit 13
(`EC_ENABLE`, echo cancellation) specifically, the null is doubly expected and not a clean isolation
of "no echo path": §36's own caveat notes each bit only got 60 frames of settling after the
*previous* bit's stimulus (1.2 seconds on the 20ms frame clock), which is short for an adaptive echo
filter to converge. So the null holds either way -- no echo to act on, or not enough time to act on
it -- but this test can't distinguish the two, not a mysterious null. Bit 7 (`CP_SELECT`) is the
cleanest cross-check of the set: the manual states it has no effect while `CP_ENABLE=0`, and §36
tested bit 7 alone (`CP_ENABLE=0` throughout), so its null result is exactly what the manual predicts,
not merely consistent with it. This is a strong, independent cross-validation of this document's own
empirical methodology: every bit this document tested and drew a conclusion about, using nothing but
direct chip observation, turns out to match DVSI's own primary-source documentation exactly.

**`DCMODE_IN`/`DCMODE_OUT` are also now fully documented** (the manual's own `Table 15`/`Table 16`,
and `PKT_DCMODE`'s field ID, `0x06`, from `Table 37`) -- previously named in this document only as an
existing-but-untested facility. `DCMODE_IN` controls decoder-side (synthesis-direction) features:
`LOST_FRAME` (bit 2, forces a frame repeat), `CNI_FRAME` (bit 3, forces comfort-noise insertion),
`CP_SELECT`/`CP_ENABLE` (bits 7/8, same companding semantics as the encoder side, mirrored for
decoded PCM output), and `TS_ENABLE` (bit 14, forces tone synthesis) -- all decode-direction
controls with no bearing on this document's own encode-direction (`SPEECH` packet in, `CHANNEL`
packet out) methodology, and genuinely out of scope for the same reason `PKT_CHAND4`'s soft-decision
decode mode already was (§21): testing them would need a decode/synthesis feedback loop (comparing
synthesized PCM output) this investigation was never built to do, not more of the same technique.
Recorded here as resolved-and-out-of-scope, not left as an open unknown.

**The manual does not document the internal codec's own bit-level parameter semantics** (checked
directly -- no mention of gain vectors, pitch, voicing, spectral amplitude, or excitation parameters
anywhere in its text): it is purely an I/O/packet interface manual, not internal algorithm
documentation. This confirms, rather than merely leaves open, that `g1`/`g2`/`u5`'s semantic
identity, `u4`'s full semantic role, and `g3`'s real-world meaning genuinely cannot be resolved from
any DVSI-published source this document has now checked -- they remain open for the reasons already
stated in this document's own executive summary, not because the search for documentation was
incomplete.

**A note for Bruce, not a decision made silently**: the source manual is marked "DVSI CONFIDENTIAL
PROPRIETARY" on every page, though it has been publicly archived and referenced by the D-STAR
digital-voice community for years for exactly this kind of interoperability work, and prior sections
of this document already quote short descriptive sentences from it. Storing the full PDF in
`hams_com` (private repo) follows this project's own established convention for copyrighted
reference material. This section keeps its quoting to bit names (factual, not copyrightable) and one
descriptive sentence for `CP_ENABLE` specifically, deliberately lighter than reproducing the manual's
own full table -- but the judgment of whether even that is the right amount to quote in this
document (which is itself in the public `hams_open` repository) is Bruce's to make, not something to
decide unilaterally. Flagged here, and in this round's `night_shift_history.md` entry, for his review.

## 40. `ECMODE_IN`'s DTX/DTMF behavior checked under D-STAR and AMBE+2 half-rate, not just RATET(27) -- a real gap in `ambe_dstar` found and fixed

Every DTX-silence and DTMF finding in this document up through §39 was tested exclusively under
RATET(27) (P25 full-rate FEC). `ECMODE_IN` is documented as a global, per-channel encoder control
(§39, DVSI's own manual), so `DTX_ENABLE`/`TD_ENABLE` should behave the same way under D-STAR and
AMBE+2 half-rate -- but this had never actually been checked, and this document's own executive
summary implied full coverage ("chip modes... duplicated in software") without it. This section
closes that gap directly against the live chip, using each mode's own already-validated decoder
(`ambe_dstar::decode`, `ambe_plus_2::decode`) rather than RATET(27)'s own `g0`/`u4`-based classifiers,
which are specific to RATET(27)'s own proprietary wire permutation and don't apply elsewhere.

**Methodology note, learned the hard way**: `PKT_ECMODE` writes the full 16-bit register, not just
the bit a caller cares about. `TD_ENABLE` (bit 12) is on-by-default (§25), so writing
`ECMODE_IN = DTX_ENABLE` alone silently clears it -- the first pass of this section's own probe did
exactly that, producing a false "no tone/DTMF detection under either D-STAR or AMBE+2" null result
that changed the moment `TD_ENABLE` was re-set explicitly alongside `DTX_ENABLE`: D-STAR went from
ordinary speech to its own real `Tone` code, while AMBE+2 half-rate changed too, but to `Erasure`
rather than `Tone` (see its own subsection below) -- re-setting `TD_ENABLE` fixed the test, not the
underlying cross-mode difference. Every capture below re-sets `ECMODE_IN` explicitly rather than
relying on any assumed default. Tooling:
`examples/p25_ambe_plus_2_and_dstar_dtx_dtmf_probe.rs` (not a `cargo test`, a live-chip tool, per this
project's established convention).

### AMBE+2 half-rate (RATET(33)): DTX-silence matches the published spec exactly

`ambe_plus_2::decode::classify_b0` already implements DVSI/TIA's own published special-value
convention for the half-rate family (`0..=119` Speech, `120..=123` Erasure, `124..=125` Silence,
`126..=127` Tone), taken directly from the spec (via mbelib's real `ambe3600x2450.c`), not
reverse-engineered. Confirmed live, with a clean on/off control:

- Silence, `DTX_ENABLE` off: `b0=119` (ordinary `Speech`, the boundary value), 8/8 frames.
- Silence, `DTX_ENABLE` on: `b0=124` (`Silence`, exactly per spec), 8/8 frames.

This is a second, independent chip validation of DTX-silence signaling (RATET(27)'s own g0=3841
marker being the first, §31/§35) under a completely different rate and a completely different,
already-published bit layout -- real, positive cross-mode confirmation that `ECMODE_IN`'s
`DTX_ENABLE` is a genuine, global encoder feature, not something specific to RATET(27)'s own wire
format.

### AMBE+2 half-rate: `TD_ENABLE` reaches `Erasure`, not `Tone` -- real, reproducible, unexplained

With `TD_ENABLE` set (isolated from `DTX_ENABLE`: tested both with `DTX_ENABLE` on and off, changed
what value it produced not whether it produced one -- see the methodology note above -- 8/8 frames
each), a loud 200Hz tone and all 16 DTMF digit pairs uniformly produced `b0=120` (`Erasure`), never
`b0=126/127` (`Tone`). This is a real, deterministic, reproducible chip behavior, not noise -- every
one of 17 different stimuli, under two different `DTX_ENABLE` states, landed on the identical `b0`
value. The ordinary-speech `b1`/`b2` fields were checked first and found flat/near-flat across the 17
stimuli (`b2` constant at 7; `b1` alternating narrowly between 30/31 with no correlation to which
digit was sent) -- but **the full 9-byte captured frame is not flat**: each of the 16 digits produced
a genuinely different hex frame (e.g. digit 1 = `e8cedbae008cd122c0`, digit 2 =
`eacdeb8e20ad8702c0`, byte-for-byte distinct), confirmed by re-running with the raw frame hex printed
alongside the decode. `b1`/`b2`'s own flatness only rules out those two specific fields as a per-digit
code -- it says nothing about the other ~32 undecoded bits of `d`, since mbelib's own real decoder
returns immediately on `Erasure` without ever reading further fields, so there is no known decode for
whatever is actually varying there. **Whether AMBE+2 half-rate's `Erasure` frames carry real per-digit
information is genuinely unknown, not ruled out** -- correcting this section's own first draft, which
claimed the flat `b1`/`b2` result as evidence of "no payload," an inference the full-frame hex
directly contradicts. **Why `TD_ENABLE` maps a detected tone to `Erasure` rather than its own `Tone`
code on this particular rate is also not established** -- both left as real, open items (executive
summary item 6), not guessed at further.

### D-STAR: real special-value trigger is `b0 in {126,127}` exactly, not the wider range first assumed

D-STAR's `L_TABLE` has 126 real entries (`b0` 0..=125 all valid pitch codes), which a first pass of
this section's own tooling misread as implying a `{120..127}`-style reserved block similar to AMBE+2
half-rate. **Checked directly against mbelib's own real source** (`ambe3600x2400.c`, this project's
own cited primary reference for `ambe_dstar`), not assumed: the actual trigger is
`(b0 & 0x7E) == 0x7E`, i.e. `b0` is *exactly* 126 or 127 -- narrower than the guess, and a real,
concrete gap this project's own `ambe_dstar::decode::dequantize` had never checked for at all (every
`b0` value, including 126/127, ran straight through the ordinary voiced-speech dequantization path).

### D-STAR: `TD_ENABLE` reaches its own real `Tone` code, with a fully decodable per-digit index

With `TD_ENABLE` set (again isolated from `DTX_ENABLE`, identical result both on and off), a loud
200Hz tone and all 16 DTMF digit pairs reliably produced `b0=126`, matching mbelib's real trigger
exactly -- unlike AMBE+2 half-rate's own `Erasure` result above. Tone frames use a **completely
different bit scatter** for their own `index`/`volume` fields than ordinary speech's `b1`/`b2` (three
of `index`'s bits are looked up through per-value tables keyed on `d[6..9)`, not a plain contiguous
field) -- read directly from mbelib's own tone-decode branch, not guessed, and confirmed against the
live chip's real output:

- The 200Hz test tone decoded to `index=6`: mbelib's own range table reads `5..=122` as "single tone
  at `index*31.25` Hz" -- `6*31.25 = 187.5`Hz, one quantization step below the true 200Hz stimulus.
  Reasonable, given the field's own 31.25Hz resolution.
- All 16 DTMF digits decoded to `index` values 128-143, mbelib's own "dual tone" range (128-163) --
  and **`index == 128 + row + 4*col` held exactly for every one of the 16 digits**, using this
  project's own established row/column DTMF numbering (row 0-3 = 697/770/852/941Hz, col 0-3 =
  1209/1336/1477/1633Hz). A genuine, complete, chip-validated DTMF digit-identification scheme for
  D-STAR, structurally different from RATET(27)'s own `g0`/`u4` scheme but equally real.

**Fixed in `ambe_dstar::decode`, not just documented** (per the project's own standing practice --
see this document's own `git log`, e.g. §31/§35 -- of turning a found gap directly into tested code
rather than filing it as a someday item): added `FrameKind`/`classify_b0` (mbelib's exact `b0`
trigger), `TonePayload`/`decode_tone` (the real tone-frame bit scatter), `ToneKind`/
`classify_tone_index` (mbelib's own index range table), and `dtmf_digit_from_tone_index` (the
`128 + row + 4*col` mapping found above). `dequantize`'s own signature changed from taking
`&RawParameters` to the frame's full decoded `d: u64` (needed because a tone frame's real payload
isn't recoverable from `RawParameters` alone -- different bits entirely) and now returns
`DequantizedFrame::{Speech, Tone}` instead of unconditionally decoding every `b0` value as ordinary
speech. Before this fix, a real live DTX-silence or DTMF/tone frame under D-STAR would have been
silently misdecoded as ordinary (if pitch-unusual) speech by this crate -- exactly the class of
silent-misclassification bug this document's own `ratet27_dtx`/`ratet27_dtmf` work was built to
avoid for RATET(27). Regression-tested against 17 real captured chip frames (16 DTMF digits + the
plain tone, exact hex fixtures), plus a `classify_b0` boundary test; full crate suite (497 tests) and
clippy both clean after the change. The one existing external caller (`ambe_dstar_chip_check.rs`)
updated to match.

### D-STAR: `DTX_ENABLE`'s own silence output is not distinguishable from speech by any known decoder

With `DTX_ENABLE` off, a silent (all-zero) input settles to `b0=34` -- an ordinary pitch code.
With `DTX_ENABLE` on, the identical silent input settles to `b0=120` instead (8/8 frames both
conditions, clean, reproducible discrete jump) -- a real, deterministic effect of the bit. But
`b0=120` is **not** within mbelib's real special-value trigger (`{126,127}` only, confirmed above),
so this D-STAR "silence marker," if that is genuinely what it is, would be decoded as ordinary (if
pitch-unusual) speech by mbelib's own real reference decoder, not just by this crate before the fix
above. This document does not claim to know whether `b0=120` is DVSI's deliberate D-STAR silence
encoding, an accident of how the encoder's gain/pitch math degenerates on zero input, or evidence
that `DTX_ENABLE` isn't really a supported feature for this particular custom RATEP configuration at
all -- `b0` alone can't distinguish those, and no further DVSI documentation exists to check against
(§39). One further discriminator, from the raw captured hex rather than just `b0`: the 8 DTX-on
silence frames were *not* byte-identical (`de4ca831...`, `de4c2e29...`, `da0c2a39...`,
`cecea411...` -- only `b0=120` stayed pinned), unlike the tone/DTMF captures above, where all 8
frames per stimulus were exactly identical. A genuine sentinel/marker frame (like RATET(27)'s own
`g0=3841` DTX marker) would be expected to be a fixed constant; ordinary encoding with one field
pinned to a fixed value while the rest keeps varying looks more like "the encoder still processes the
(silent) input normally, and DTX only forces the pitch index" than a dedicated marker. Recorded as a
real, checked, but genuinely unresolved property of D-STAR's own encoding, not a gap in this crate.

### AMBE+2 half-rate's `Erasure` frames: a bit-correlation scan comes back clean, and finds a real bug in a shared tool along the way

Since the 16 DTMF digits' full `Erasure` frames differ byte-for-byte (previous subsection) but no
known field explains why, `examples/ambe_plus_2_erasure_frame_digit_correlation_scan.rs` (offline,
no chip needed -- reuses the 16 real captured hex frames above) ran the same sliding 7-bit-window
Spearman correlation scan `ambe_chip_validate_ambe_plus_2.rs` originally used to find P25 full-rate's
own Gray-coded pitch field, against DTMF row and column separately -- on the *decoded* 49-bit `d`
(post-Golay, post-dewhiten), not the raw 72-bit deinterleaved frame: `C1`'s 23 raw bits are
XOR-whitened with a PRBS seeded from `C0`'s own already-decoded data before transmission, and a fixed
XOR does not preserve a 7-bit window's ordinal value, so scanning the still-whitened frame would make
a real field inside `C1`'s data bits invisible to this technique regardless of whether it exists.
Both blocks Golay-decoded with zero corrected errors for all 16 digits (a real observation, though
per this document's own established Golay caveat -- a perfect `[23,12,7]` code always decodes
*something* within distance <=3 -- zero errors alone doesn't prove these are "real," intentionally
encoded codewords).

**The first run (on the raw, still-whitened frame) reported several windows with a suspicious,
too-good `spearman=1.000` against row -- including `bits[0..7)`, which turned out to be the literal
constant `127` across all 16 digits.** A perfectly constant window cannot carry per-digit
information, so a "perfect correlation" against it is impossible in any real sense; the actual cause
was a genuine bug in the shared `ranks()` helper (copied into three separate example tools in this
crate, including this one and `ambe_chip_validate_ambe_plus_2.rs`'s own original): it assigns
strictly increasing ranks 0..n-1 by stable-sort order even when many input values are exactly tied,
rather than the standard, correct fractional (averaged) rank for ties. When a column is constant, a
stable sort leaves it in its original input order -- and since these 16 digits were captured in
strict row-major order, that "rank by input order" fallback trivially, spuriously correlates with
row. Fixed in all three copies (`ambe_chip_probe_p25_nofec.rs` also had it) with correct
tie-averaging, plus a minimum-distinct-values guard in this section's own new tool so a near-constant
window is skipped rather than scored. Full crate suite and clippy both clean after the fix.
`ambe_chip_probe_p25_nofec.rs`'s own already-recorded finding (§9, `u2`'s Gray-coded pitch field,
spearman 0.976) is not put in doubt by this bug: §9's own reported values across 8 real frequencies
(`772, 3020, 3848, 4087, 3848, 4087, 4087, 4087`) already show four genuinely distinct values with a
real monotonic-then-saturating shape, not the two-way, input-order-correlated tie pattern this bug
specifically produces, and the result was independently reproduced across two separate live captures
-- the shape of that finding's own data rules out this specific failure mode, though it was not
re-run with the fixed code to confirm the exact coefficient.

**With the bug fixed and scanning the correctly decoded `d`, the scan comes back genuinely clean**:
the best real correlation found against row was `|spearman|=0.600`, and against column
`|spearman|=0.500` (both at `d[11..18)`). This tests 43 window positions times 2 (plain/Gray) = 86
comparisons per target. For `n=16` samples, the uncorrected two-tailed `p~0.05` critical Spearman
value is approximately 0.50 -- so *any single one* of these 86 tests, run in isolation, would nominally
call 0.500-0.600 "significant." Run 86 times per target, that is close to exactly what pure chance
predicts (roughly 86 * 0.05 ~ 4 false positives expected at that threshold), which is the actual,
quantified reason these values are not evidence of a real field, not merely "look a bit low."
**This technique does not find AMBE+2 half-rate's own per-digit `Erasure`-frame encoding, if one
exists** -- an honest null result from a real attempt, not evidence that no such encoding exists (a
7-bit linear window is a narrow hypothesis; the real field, if any, could be non-contiguous,
differently sized, or not linearly related to row/column the way D-STAR's own `128 + row + 4*col`
happened to be).

### The Erasure-vs-Tone asymmetry is not a detection failure -- checked against the chip's own ground truth

DVSI's manual describes `TD_ENABLE` only as "tone detection is enabled" -- it says nothing about what
`b0` value the encoder is supposed to emit once a tone is actually detected, so the Erasure-vs-Tone
difference above is **undocumented by DVSI, not contradicted by it**. But the manual documents a
separate, independent, chip-reported ground-truth flag for exactly this question:
`ECMODE_OUT`'s `TONE_FRAME` bit (bit 15, requested via `PKT_CHANFMT` the same way `VOICE_ACTIVE` was
in §27/§28): "The encoder sets this bit if the output frame contains either a single frequency tone,
a DTMF tone, a KNOX tone, or a call progress tone."

`examples/p25_ambe_plus_2_and_dstar_tone_frame_ground_truth.rs` checked it directly against the same
17 stimuli (loud 200Hz tone + 16 DTMF digits) under `TD_ENABLE`, on both rates. **`TONE_FRAME=1` for
every single AMBE+2 half-rate frame, 8/8 captures each, all 17 stimuli** -- even though `b0=120`
(`Erasure`) throughout. D-STAR's own already-confirmed result (`b0=126`, `Tone`) also reads
`TONE_FRAME=1`, 8/8, serving as a positive control that the readback mechanism itself works correctly
outside RATET(27), and that a real, working "not detected" case would show `TONE_FRAME=0` if it
occurred (it never did across either rate's own tone stimuli).

**This settles the detection question decisively: AMBE+2 half-rate's chip genuinely, correctly
detects every one of these stimuli as a tone. It is not failing to detect anything.** What remains
open is narrower and more specific than before: *why* does this rate serialize a genuinely-detected
tone into `b0=120` (`Erasure`) instead of its own spec-defined `126/127` (`Tone`) range, and is the
tone's real content (frequency, or DTMF digit identity) recoverable from that `Erasure` frame's other
bits at all? The natural hypothesis -- that this is actually a real Annex J tone frame (DVSI's own
dedicated tone-frame parameter table, already transcribed in full in `src/ambe/AMBE_PLUS_2_NOTES.md`,
covering `ID` values 0-254 with 128-163 individually tabulated) -- is not chased further here: Annex J's
own `ID` field's bit position within a real tone frame's 49-bit `d` was never derived in this
codebase (`ambe_plus_2::decode::DequantizedFrame::Tone`'s own doc comment already discloses this as
an open gap, not silently skipped), and Annex J's own 128-163 range does *not* follow a simple
row/column formula the way D-STAR's own dual-tone table does (mbelib's own source comments this
directly: "dual tone index is different on ambe(dstar) and ambe2+"), so this is a real,
separate reverse-engineering task for a future session, not a quick follow-up to this one.

**Checked against the reference decoder itself, not just this codebase: mbelib doesn't derive it
either.** `mbe_decodeAmbe2450Parms()` in mbelib's own `ambe3600x2450.c` decodes `b0` from `ambe_d[0..4)`
and `ambe_d[37..40)` first, and the instant it finds `b0 == 126 || b0 == 127` it does
`return (3)` immediately -- before reading a single one of the payload bits (`b1` through `b8`: V/UV,
gain, spectral amplitudes) that a Tone frame's later bits would otherwise carry. The caller,
`mbe_processAmbe2450Dataf()`, treats `bad == 3` the same way it treats `bad == 2` (Erasure): both fall
through to the `else` branch that calls `mbe_synthesizeSilencef()` and re-initializes the codec state,
with no tone-specific synthesis path at all. In other words, the reference implementation recognizes
that a frame is a tone (the `b0` check) but never reads or decodes the tone's own identity -- it
discards the rest of the frame and plays silence, exactly like a real Erasure. This means Annex J's
`ID`-field bit-scatter isn't a gap specific to this codebase's own reverse-engineering effort; it was
never implemented by the reference decoder either, so there is no existing source to consult for it --
deriving it (if it is even present in the bits at all) would be original reverse-engineering work
against the raw chip output, using stimuli that vary the Annex J `ID` itself (single tones spanning its
full 0-254 range, not just the 16 DTMF pairs already captured, which may all cluster in one narrow
sub-range).

### AMBE+2 half-rate's `Erasure` frames carry DVSI's own documented `TONE_IDX` field -- fully resolved for both DTMF and single tones

The previous subsection left one question open: is the tone/DTMF identity that AMBE+2 half-rate's
chip genuinely detects (confirmed via `TONE_FRAME`) recoverable from the `Erasure` frame's own other
bits at all? Direct bit-level analysis of the correlation-scan tool's own 16 already-captured real
DTMF hex frames (decoded to the 49-bit `d`, not the raw whitened bits) found a real, exact structure
-- and only after finding it did a search of the AMBE-3000R chip's own separate primary-source
manual (a different document from the USB-3000 manual and from Annex J's own transcribed tables)
turn up that DVSI already names and tabulates this exact field. **Yes, fully, for both DTMF and
single tones.**

**The bit-level finding, before the manual was consulted**: every field of `d` besides one is either
a hard constant or a separately-identified amplitude field. Across all 16 real captured DTMF digits
(same fixed capture amplitude): `d[0..4)=1111` and `d[37..40)=000` (the already-known `b0=120`
marker), `d[4..16)=0x0f38` (confirmed elsewhere to be an amplitude/gain field, not digit-dependent --
it takes 4 different values across a 4-point amplitude sweep at one fixed frequency), `d[36..40)
=0x8`, and `d[44..49)=0b10000` -- every one of these, on every one of the 16 digits, with zero
exceptions. What varies is an 8-bit value serialized low-nibble-first and repeated with uneven
redundancy: the low nibble appears four times (`d[16..20)`, `d[24..28)`, `d[32..36)`, `d[40..44)`),
the high nibble appears cleanly only twice (`d[20..24)`, `d[28..32)` -- a third copy at `d[36..40)`
is entangled with `b0`'s own marker bits, which force that range's low 3 bits to `0`).

**Checked against DVSI's own `AMBE-3000R Vocoder Chip Users Manual` (Version 1.4, March 2013,
Table 103 "TONE Field Format" / Table 104 "TONE Index Values", page 74) -- and it is exactly the
documented `TONE_IDX` field** (Field ID `0x00` of a `TONE` field), which the manual itself describes
as bidirectional: "Can specify the index of a desired tone **or identify the index of a detected or
received tone**." `TD_ENABLE` triggers the second case: the encoder detects a tone in real input
audio and reports its own `TONE_IDX` back through the channel bits. Table 104 tabulates two of its
own ranges relevant here, specific to AMBE+2 half-rate's "Rate Index Values 33 to 61" column (the
table's other column, for rate indices 0-32, uses a *different*, non-monotonic DTMF mapping not
applicable to this rate):
- **DTMF** (`0x80..=0x8F`): `0x80 | nibble`, where `nibble` is the digit's own standard
  DTMF-as-4-bit-nibble value -- confirmed value-for-value against all 16 real digits.
- **Single tone** (`0x05..=0x7A`): `index = round(f0 / 31.25 Hz)`, spanning 156.25-3812.5 Hz --
  confirmed exactly against 26 real single-tone captures spanning 8 distinct frequencies (203-401
  Hz) and 4 amplitudes, with zero exceptions.

This is now implemented as `ambe_plus_2::decode::decode_tone_idx`/`dtmf_digit_from_tone_idx`
(the former majority-votes the 4 low-nibble copies and requires the 2 unconstrained high-nibble
copies to agree, returning `None` rather than a guess on disagreement -- tested directly against a
synthetically corrupted real capture in each direction) and chip-validated live via
`examples/ambe_chip_validate_ambe_plus_2_dtmf.rs`: **all 16 DTMF digits plus 4 representative single
tones, all 8 captured frames each, 160/160 clean** -- `classify_b0=Erasure`, `TONE_FRAME=1`, and
`decode_tone_idx` matching DVSI's own documented value, on every single frame. AMBE+2 half-rate now
has the same fully chip-validated, fully decodable per-digit DTMF identification in software that
RATET(27) and D-STAR already had, plus single-tone frequency identification neither of those two
modes needed -- closing what had been the one remaining "no known field decodes it" gap in this
document.

**Correction to the paragraph above, found immediately afterward**: Annex J's own tone-frame
parameter table (`f0`/`l1`/`l2`, `AMBE_PLUS_2_NOTES.md`) is **not** a second, separate, still-mysterious
encoding from `TONE_IDX` -- they are the same underlying tone identifiers. Checked directly:
`l1 * f0` and `l2 * f0` reconstruct the real DTMF frequencies at the *same* index (e.g. Annex J row
128, `f0=78.5, l1=12, l2=17`, gives `12*78.5=942 Hz` / `17*78.5=1334.5 Hz`, matching DTMF `'0'`'s own
`941/1336 Hz`), and the same holds for every DTMF row 128-143. Annex J is the *decoder-side synthesis
recipe* (two sinusoids at harmonics `l1` and `l2` of a fundamental `f0`) for whichever tone
identifier ends up in a frame -- not an independent mystery. This means `FrameKind::Tone`'s
"not decoded" gap is, in principle, now closeable with a table lookup plus two sinusoids, not a
bit-scatter derivation -- not built here, since the frame kind it would apply to was never observed
(next paragraph).

**A decisive follow-up, using a mechanism found directly in DVSI's manual, independent of tone
detection entirely**: the manual documents a *forced*-tone-generation path -- a `TONE` field
(`0x08`, `TONE_IDX` byte, amplitude byte) appended to a `SPEECH` packet, with `ECMODE_IN`'s
`TS_ENABLE` bit (14) set, "force[s] the encoder to transmit a tone frame" for a specified `TONE_IDX`
(Table 98/103, page 72-74) -- bypassing the tone *detector* entirely.
`examples/p25_ambe_plus_2_forced_tone_probe.rs` used this to answer, decisively, whether `b0=126/127`
is reachable on this chip/rate at all: **forcing `TONE_IDX=0x81` (DTMF) and `TONE_IDX=0x08` (single
tone) both read back `b0=120`, matching every detection-triggered capture in this document exactly.
`b0=126/127` was never observed, forced or detected.** `FrameKind::Tone` is now understood to be
believed unreachable on this chip/rate, not merely undecoded.

**The same forced-tone mechanism also found a real, new, distinct tone-bearing sub-code**: forcing
DVSI's own documented `Call Progress` tones (`TONE_IDX=0xA0/0xA1/0xA2` -- dial/ring/busy) reads back
`b0=122`, not `120`. `b0`'s 120-123 range is therefore not one undifferentiated "erasure"; `120` and
`122` are two real, distinct tone-bearing sub-kinds (`FrameKind::DetectedTone`/`FrameKind::
CallProgress`), while `121`/`123` remain genuine, unproduced erasure codes (`FrameKind::Erasure`) --
a caller matching only on the old undifferentiated `Erasure` would have silently dropped every real
tone/DTMF/Call-Progress frame, now fixed in `ambe_plus_2::decode`.

**Fully resolved (corrects an error in an earlier draft of this section): there is no discrepancy for
Call Progress or single tones at all.** Forcing `TONE_IDX=0xA0/0xA1/0xA2` and decoding the readback
gives back `160`/`161`/`162` -- which **are** `0xA0`/`0xA1`/`0xA2` (`0xA0 = 160` exactly), not a
different number. An earlier pass through this material compared the decimal value `decode_tone_idx`
returns against the hex value that was sent without converting either into the other, and wrongly
called that a discrepancy; re-checked directly in one base, it is not. **DTMF genuinely does have a
readback quirk, fully explained**: forcing `TONE_IDX=0x87` reads back `128`=`0x80`, not `0x87`. The
explanation is DVSI's own Table 104 documenting *two* independent DTMF columns ("Rate Index 0-32" and
"Rate Index 33-61", different non-monotonic nibble-to-digit mappings) -- at `RATET(33)`, **the only
rate this was tested against** (not yet confirmed for `RATET(34)`, AMBE+2 half-rate's own No-FEC rate,
or any 0-32-group rate such as `RATET(27)`), the forced-generation `TONE` field is read by the encoder
via the **0-32 column** even though `33` is itself in the 33-61 group, while the encoder's own output
(a genuinely detected digit or a forced one's readback, alike) is always reported via the 33-61 column
at this rate. Checked exactly against all 16
forced DTMF digits: `readback = column_33_61[column_0_32[sent]]` holds for every one of the 16
captures, zero exceptions -- a confirmed formula, not an inference. **This does not affect the
detected-DTMF finding's own correctness or its live validation**: a genuinely *detected* DTMF digit
(the real, practically useful case this document validated 128/128) was never routed through the
0-32 column and reads back in `TONE_IDX`'s own 33-61 numbering exactly as this document always
claimed. See `ambe_plus_2::decode::decode_tone_idx`'s own doc comment for the full derivation and
`forced_dtmf_readback_matches_the_rate_column_mismatch_explanation` for the pinned regression test.

### Cross-mode summary

`ECMODE_IN` is now confirmed global across three structurally different rates (RATET(27), D-STAR,
AMBE+2 half-rate) -- not just asserted from the manual's own wording, but independently verified with
real on/off contrasts on two of the three. **All three rates now have chip-validated, fully decodable
per-digit DTMF identification in software**: RATET(27) via its already-documented `g0`/`u4` pair,
D-STAR via the tone frame's own separate `index` field (`128 + row + 4*col`, this section), and
AMBE+2 half-rate via DVSI's own documented `TONE_IDX` field, serialized into its `b0=120`/`122`
tone-bearing frames -- correctly distinguished in software from the genuine `b0=121`/`123` erasure
codes (this section, directly above) -- found by the same technique (direct analysis of real, varied
captured chip frames) each time, with AMBE+2 half-rate's own case additionally matched afterward to
a primary-source table once the bit structure was already known. Software coverage claim updated
accordingly in the executive summary.
