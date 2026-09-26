# Reading the numbers

!!! info "Provenance"
    Written 2026-09-12. The metric keys and where they are written are in
    [Training](../design/training.md). The volatility measurement is from
    run `20260912-022411-generalist-large-full-ll5`, 54 evaluations.

Every 500 steps the trainer scores the model on a fixed list of held-out
clips and writes a row per metric. This page explains these rows
and indicates the threshold for statistical significance between measurements.

## The columns

Each column has its own page in [Metrics](../metrics/index.md) covering its
definition, computation, target values, and baseline noise level:

| Column | Description |
|---|---|
| [`eval/lsd`](../metrics/lsd.md) | Spectral distance from clean in dB. The primary evaluation metric. |
| [`eval/mel`](../metrics/mel-l1.md) | Spectral distance across 80 perceptual bands. |
| [`eval/sisdr`](../metrics/si-sdr.md) | Waveform distortion. Values are strongly negative because the codec discards phase. |
| [`eval/hnr`](../metrics/hnr.md) | Buzziness in dB above clean. The target is 0. A raw decode averages +10. |
| [`eval/periodicity`](../metrics/periodicity.md) | An alternative buzz metric used primarily for observation. |
| [`eval/babble`](../metrics/babble.md) | Measures speech generation into silence after transmission ends. |
| [condition columns](../metrics/conditions.md) | LSD under receive noise, dropped frames, and at the edges. Includes `@<mode>` per vocoder. |
| [loss terms](../metrics/loss-terms.md) | The specific loss terms minimized by the optimizer. |

## Statistical significance

The following variance was measured across 54 evaluations of a converged run on identical inputs:

| Column | Median | Range | Std dev |
|---|---|---|---|
| `eval/lsd` | 11.06 | 10.89 – 11.42 | 0.12 |
| `eval/mel` | 5.60 | 5.40 – 5.92 | 0.12 |
| `eval/lsd_last1s` | 11.11 | 10.92 – 11.53 | 0.14 |
| `eval/lsd_first1s` | 10.85 | 10.65 – 11.36 | 0.17 |
| `eval/hnr` | 3.66 | 2.47 – 4.94 | 0.48 |
| `eval/lsd_rx` | 15.29 | 14.16 – 17.89 | 0.67 |
| `eval/sisdr` | −19.93 | −22.43 – −18.67 | 0.84 |
| `eval/babble` | −78.93 | −95.48 – −62.75 | 7.47 |

(Based on 41 evaluations of a converged run past step 20,000. `eval/hnr` is from 39
evaluations of the naturalness run past step 8,000.)

Guidelines for interpretation:

- A gap of 0.05 in `eval/lsd` is within the noise floor. Models within this margin should be considered tied.
- A gap of 0.25 or more indicates a significant difference. This exceeds one standard deviation and appears consistently across modes.
- For `eval/lsd_rx`, `eval/hnr`, and `eval/babble`, single evaluation values are unreliable. Use the median of multiple recent evaluations rather than the final checkpoint value.

This significance threshold applies to the [capacity result](what-the-experiments-showed.md#capacity). A model with three times the parameter count scored 0.04 better (within the noise margin). A model with one-seventh of the parameters scored 0.26 worse (a significant difference).

## Evaluation considerations

Checkpoint selection based on a single metric can be misleading. In the referenced run, step 32,500 had a near-optimal `eval/lsd` of 10.99 but the worst `eval/lsd_rx` of 17.89. Since the trainer selects the best checkpoint using `eval/lsd` alone, the auto-selected checkpoint may perform poorly under noise. Review multiple metrics before deployment.

Changing the evaluation dataset breaks comparability. As the YSF/DMR capture expanded to more speakers, the number of evaluation clips increased from 19 to 66. While per-mode metrics for unchanged modes remained comparable, the overall mean shifted due to the different mix of clips. Ensure the clip count matches when comparing two runs. The trainer logs this count at startup.

## Metric limitations

These metrics measure distance between spectra or waveforms. They do not account for intelligibility, speaker identity, listening fatigue, or artifact severity. Filter configurations typically scored between 10.7 and 11.4, a spread marginally larger than the measurement noise. This indicates that standard distance metrics may not capture further improvements.

The project currently uses a learned [MOS predictor](../metrics/mos.md) to compare architectures. Close variants are evaluated through blind listening tests. Under this system, the filter peaked near 2.1 predicted MOS. The combined restorer and waveform synthesizer pipeline scores 3.57 on a standard input (compared to codec at 2.22 and clean audio at 4.14). In blind listening tests, the pipeline scored 4.6 out of 5.
