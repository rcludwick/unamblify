# What the experiments showed

!!! info "Provenance"
    Measured 2026-09-11 and 2026-09-12 on an Apple Silicon Mac (MPS).
    Every run is under `$UNAMBLIFY_DATA/runs/`, and the decisions they
    settled are recorded in [Design](../design/index.md). All figures are
    `eval/lsd` at each run's own best checkpoint. Read
    [Reading the numbers](reading-the-numbers.md) first, particularly the
    section about significance thresholds for score differences.

!!! note "2026-09-25"
    This page covers the filter model (candidate 1) up to 2026-09-19.
    The restorer + waveform synthesiser pipeline that replaced it
    ([candidate 3](../design/training.md#candidate-3-concretely)) scores
    3.57 predicted MOS on a faithful input, compared to 2.22 for the codec on
    the same clips and 4.14 for the recording. A blind listening test put
    it at 4.6 out of 5. These experiments are #33–#38 in the
    [experiment log](experiment-log.md).

## 1. One model for every mode beats one model per mode

A model trained to undo D-STAR might be expected to outperform one handling three other codecs, but measurements show otherwise.

| Mode | Generalist (2.1 M) | Trained on that mode alone |
|---|---|---|
| YSF/DMR | 10.86 | — |
| Codec 2 3200 | 10.97 | 10.90 |
| D-STAR | 11.06 | 11.02 |
| Codec 2 1600 | 11.22 | 11.09 |

The two models are within 0.13 across all modes. The generalist saw roughly a fifth as much data per mode because the balanced shard set capped every mode at the smallest one. When trained on the larger set, the same architecture outperformed all four single-mode models.

The four codecs degrade speech in [similar ways](what-the-codec-destroys.md), making restoration a shared task. The [mode embedding](the-post-filter.md#mode-embedding) provides the necessary per-codec specialisation using a small parameter addition.

## 2. Specialising a generalist did not improve performance

A trained generalist was fine-tuned on D-STAR alone to test if specialising improves performance.

It converged to 11.06, which matches the generalist's D-STAR score, and did not exceed it at any evaluation step. The only metric that changed was the noisy column.

This suggests the model is not capacity-starved per mode. Dedicating the network to one codec did not improve performance. As a result, a single model is used instead of a family of models. This simplifies astar by requiring only one weights file and avoiding mode-dependent loading.

!!! warning "What this experiment did not test"
    The specialist fine-tuned on the same shards the generalist had
    already trained on, at a tenth the learning rate. A specialist on
    genuinely new data, at a higher learning rate, or with a smaller
    profile remains untested. A lite specialist per mode may yield
    different results, as a smaller model might lack capacity for four codecs.

## 3. Capacity: the full profile sits at the knee { #capacity }

Three sizes of the same architecture were tested using the same data and 40 000 steps:

| Model | Parameters | Mean | D-STAR | YSF/DMR | C2-3200 | C2-1600 |
|---|---|---|---|---|---|---|
| lite | 315 K | 11.15 | 11.33 | 11.12 | 11.03 | 11.12 |
| full ×1 | 2.1 M | 10.89 | 11.08 | 10.74 | 10.80 | 10.90 |
| full ×1.75 | 6.4 M | 10.85 | 11.02 | 10.73 | 10.74 | 10.89 |

Tripling the parameters improves the score by 0.04, which is within measurement noise. Shrinking the model sevenfold degrades the score by 0.26, which is outside measurement noise.

`unamblify bench` shows the 6.4 M model uses 0.8 % of the 20 ms frame budget on one M-series core, indicating the capacity is affordable. However, the larger size provides negligible improvement and increases the weights file from 8 MB to 24 MB in astar.

## 4. Data scale improves performance

Growing the training set from 11 112 to 141 930 examples with no change to the model improved the mean by 0.20 and Codec 2 1600 by 0.32. This is a larger gain than tripling the model size and does not increase inference cost.

| Change | Mean LSD gained | Cost at inference |
|---|---|---|
| 13× more training data | 0.20 | none |
| 3× more parameters | 0.04 | 3× memory and file size |

This result indicates that data scale is highly effective. [Capture throughput](../research/hardware-throughput.md) is prioritised because AMBE corpora are the primary constraint.

!!! note "A caveat on the mix"
    The large set is 84 % Codec 2 by example count. The Codec 2 captures
    include long LibriTTS utterances that yield about two crops each,
    while the AMBE captures are mostly short reads. D-STAR received five times
    more data but only one batch in six. These two effects
    cancelled out: D-STAR ended at 11.08 compared to 11.06 before. To improve a
    specific mode, its share of the batches matters as much as its absolute quantity.

## 5. The robotic quality is measurable, and the first fix was a mean, not a match

A listener reported that the high end sounded restored but the voice sounded robotic. This was measured on the eval clips as harmonic peaks against the valleys between them on voiced frames, counting every harmonic once:

| Signal | D-STAR | excess over clean (mean) | per-frame \|excess\| |
|---|---|---|---|
| Clean speech | 9.7 dB | — | — |
| After the codec | 13.1 dB | +3.4 | 4.1 |
| Filter-only model | 11.3 dB | +1.5 | 3.0 |
| With the noise head and `hnr_w` | 8.9 dB | −0.9 | 2.6 |

The codec leaves the output too periodic. A filter-only model recovers about 40 % of that because filtering a periodic signal leaves it periodic. Adding an [aperiodic excitation path](the-post-filter.md) and a loss that targets the target's ratio moved the mean past zero at no LSD cost (converged medians 11.00 against 11.01).

Matching the mean ratio required several adjustments. Initial loss weights were too low (0.08 % of the total) and had no effect. Increasing the weight caused the model to optimize the proxy metric while degrading actual performance. This was resolved by measuring the ratio directly. Additionally, an accurate mean can obscure per-frame errors. The metric now counts every harmonic once in both the loss and the metric. `eval/hnr_abs` reports the per-frame error alongside the mean. The noise path is modulated by the periodic path's envelope with a learned depth to produce breath-like rather than hiss-like noise.

## 6. Stop consonants require structural changes

Measured on 73 plosive bursts in the D-STAR eval clips, inside the codec's band (2–3.8 kHz, 1 ms resolution):

| Signal | burst peak | closure depth | rise time | burst length |
|---|---|---|---|---|
| Clean | 0 | 43.5 dB | 2.9 ms | 10 ms |
| Codec | −12.8 dB | 31.7 dB | 13.7 ms | 30 ms |
| Filter-only model | −13.5 dB | 28.8 dB | 13.8 ms | 29 ms |
| Noise-head model | −13.3 dB | 28.0 dB | 14.0 ms | 29 ms |

The codec smears bursts to three times their length at a third of their level. Both models passed that through unchanged and filled the closure before it by a further 3 dB. Every per-frame gain was held constant for 80 samples, meaning nothing shorter than 10 ms could be formed. The finest loss window was 32 ms. At that scale, a 5 ms burst is a fraction of one frame and an L1 loss is minimised by a smooth output. LSD cannot detect these changes. The response included the [`eval/plosive_*`](../metrics/plosive.md) columns, the [transient term](../metrics/loss-terms.md), interpolated frame gains and the noise path's 1 ms gains. These were insufficient for the filter. Run #37 in the [experiment log](experiment-log.md#naturalness) measured filter v6 as worse than the codec on bursts. The restorer pipeline was the first system to improve on this.

## 7. Predicted MOS scores

[UTMOS22](../metrics/mos.md) is a predictor trained on human naturalness ratings. On the noise-head run's best checkpoint:

| Kind | all | D-STAR | YSF/DMR | Codec 2 3200 | Codec 2 1600 |
|---|---|---|---|---|---|
| clean | 4.05 | 4.05 | 4.05 | 4.05 | 4.05 |
| codec decode | 1.78 | 1.88 | 2.16 | 1.76 | 1.34 |
| model output | 2.08 | 2.20 | 2.28 | 2.22 | 1.66 |

The model shows a modest improvement over the codec decode but remains below the clean signal. UTMOS22 provides a more direct measure of perceptual quality compared to spectral metrics and ranks the modes consistently with human listeners.

## 8. Most of the AMBE chip time was held out

LibriTTS-R's `dev-clean` and `test-clean` were captured through the chip first, and the initial split rule held every reader of them out. On 2026-09-12, D-STAR had 12.0 captured hours in `train` against 20.8 in `dev` and `test`, and YSF/DMR had 17.8 against 21.2. Evaluation uses 17 clips per mode. The rule now hashes those readers like the VCTK ones, keeping the eval clips' readers. `unamblify prepare --resplit` reapplies it without touching audio, and the shard set was rebuilt. This moved 8 228 utterances to `train`. The same set was 84 % Codec 2 by example, so an AMBE mode got one gradient step in six. `[data] mode_weights` now sets the share directly.

## 9. Performance floor

Four single-mode models, two generalists, a specialist, three model sizes and two lookahead variants all score between 10.7 and 11.4. The spread is barely larger than the run-to-run measurement noise.

This narrow spread indicates a performance floor imposed by the task rather than model configuration. This aligns with [the theory](what-the-codec-destroys.md#performance-limitations): the codec transmits a fixed amount of information and the model reconstructs plausible speech consistent with it. Additional parameters and training steps do not add constraints beyond what the bits provide. The band-swap probe (#33 in the [experiment log](experiment-log.md)) later indicated that much of the remaining gap was due to the filter architecture rather than the bit rate.

## 10. Data scaling limits (2026-09-19) { #not-the-road }

Measured a week later on a 200 000-utterance set (50 000 per capture set) and evaluated by [predicted MOS](../metrics/mos.md).

Doubling the training data at the original step count reduced the predicted MOS to 1.91 compared to the baseline 2.08. Concurrently, `eval/lsd` fell from 14.97 to 11.74 (log #25), indicating the run had stopped before convergence. Doubling the steps and tripling `hnr_w` on the same set reached a MOS of 2.12 at step 76 000, compared to a degraded 1.79 and a clean 4.10 (#26). The final checkpoint scored 2.09 and the best by `eval/hnr_abs` scored 2.07. Runs now keep every checkpoint to allow scoring across multiple steps.

This adds context to section 4. Data scale provided improvements initially, but returns diminished at larger sizes. A subsequent review (#28, [training](../design/training.md#review-2026-09-19)) identified architectural limitations rather than data constraints. The signal path lacked temporal shaping and the adversarial training phase was inactive (`gan = true` had no effect despite discriminators being present). The review also identified missing training conditions like frame loss and distorted microphone input. These data issues were quantified (#29, #30) and addressed via dataset modifications before evaluating architectural changes. The subsequent band-swap probe (#33) led to a new architecture: the restorer and waveform synthesiser pipeline used in runs #34–#38.

## Where the remaining headroom is

!!! note "2026-09-20"
    The list below is the 2026-09-12 ranking. Section 10 reorders it. Item 3
    has flattened, and item 4 is misdescribed: the discriminators are
    written and shape-tested, but the adversarial phase is not wired
    into the trainer.

Ranked by the size of the gap:

1. **Noisy condition performance.** `eval/lsd_rx` is approximately 4 dB worse than the clean column. The training objective does not constrain noisy and clean behaviour together, causing noise robustness to vary during training. Potential mitigations include raising `rx_share` or adding a consistency term between clean and noisy passes.
2. **Evaluation metrics.** With model scores compressed within a 0.7 dB band, LSD does not distinguish model differences effectively. As noted in section 7, all models are within a third of a point of the raw decode on a five-point scale. Future runs are evaluated using [predicted MOS](../metrics/mos.md), `eval/hnr_abs`, plosive metrics, and listening tests.
3. **AMBE data scaling.** Additional data provided measurable improvements initially. The YSF/DMR capture was in its early stages during these experiments.
4. **Adversarial training.** The adversarial path is implemented but inactive. While GANs typically improve perceptual sharpness, they can introduce artifacts. This approach requires evaluation by listening rather than relying solely on LSD.
