# `sibilant_*`: do the sibilants survive?

**Two columns, both *output minus clean* at the clean's own sibilant
frames, so 0 is "as bright as the recording". `sibilant_level_db` is the
4–8 kHz level (negative: the "s" is quieter). `sibilant_balance_db` is
the high-to-mid balance, (4–8 kHz) − (1–4 kHz) (negative: duller even
where it is as loud). Standalone only so far: `unamblify measure --dir D`.**

!!! info "Provenance"
    Added 2026-09-24, after the "s" was heard going dull in the
    restorer pipeline before any column showed it (experiment log #36).

## What it measures

An "s", "sh" or "f" is broadband noise above 4 kHz. No narrowband codec
carries that band, so every post-filter has to invent it. A model
trained by regression that cannot tell whether a frame holds a loud hiss
or nothing predicts the middle. The middle of "loud or absent" is
"quiet": the consonant is there, but dull. Whole-clip spectral metrics
average it away, and a listener hears it at once.

## How it is computed

`unamblify_audio::sibilant_excess(out, clean)`, at 16 kHz only:

1. A 32 ms / 10 ms STFT of both signals. Per frame, the energy in
   1–4 kHz and in 4–8 kHz, in dB.
2. **Sibilant frames are found in the clean.** Of its frames above a
   floor, they are the loudest 8 % by 4–8 kHz energy.
3. At those frames, output minus clean: the high-band level and the
   balance (high minus mid). These are averaged over the frames, and
   `measure` weights clips by their frame count.

The band-swap probes of experiment #33 said the restorer's high band was
already good on average. This is the check that the average is not made
of a good vowel tail and a missing "s".

## How to read it

Measured 2026-09-24 on the 80 eval clips (`unamblify measure`), level /
balance in dB at the clean's sibilant frames:

| System | D-STAR | YSF/DMR | Codec 2 3200 | Codec 2 1600 |
|---|---|---|---|---|
| raw codec | −63 / −63 | −63 / −63 | −63 / −59 | −63 / −59 |
| filter model (v6 at 72 k) | −9.9 / −9.7 | −10.4 / −9.9 | −11.4 / −9.3 | −12.1 / −10.3 |
| restorer, L1 only | −9.5 / −7.3 | −9.7 / −7.5 | −19.5 / −16.7 | −20.2 / −17.1 |
| restorer + synthesiser (#35) | **−8.4 / −7.0** | **−8.4 / −6.9** | **−19.9 / −18.6** | **−20.3 / −18.6** |

!!! warning "Corrected 2026-09-25 (experiment #38)"
    The table above was measured on the trainer's 16 kHz rendering of the
    codec output, resampled back to 8 kHz, which is 10–30 dB down in the
    top 300 Hz of the band. The restorer read that as "no s". On the
    captured decode itself (`eval_input`), the pipeline measures
    **−4.7 / −4.6 on D-STAR, −4.4 / −4.3 on YSF/DMR, −4.6 / −4.4 on
    Codec 2 3200 and −4.3 / −3.9 on 1600**: a softer "s" on every mode,
    no longer a missing consonant anywhere.

The codec's −63 means "nothing above 4 kHz". The codec is narrowband, and
the whole band is the post-filter's to invent. As a rule of thumb, −3 dB
is audible as a softer "s", −6 dB as a lisp and −15 dB as a missing
consonant. Read from the table, the pipeline lisped on AMBE and dropped
the consonant on Codec 2 (experiment log #37). On the faithful input
(the correction above) it is about −4.5 on every mode: a softer "s",
short of a lisp.
