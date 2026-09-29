# Theory

This section covers the underlying concepts and design rationale. The [Research](../research/index.md) pages contain sourced facts and the [Design](../design/index.md) pages document the implementation. This section explains the concepts necessary to understand the system.

| Page | Description |
|---|---|
| [What the codec destroys](what-the-codec-destroys.md) | Why 2 400 bit/s speech sounds synthetic, which specific losses cause each artifact, and what is recoverable. |
| [The post-filter](the-post-filter.md) | Why the first design filters the decoded audio instead of resynthesizing speech, and the purpose of each network block. |
| [How it learns](how-it-learns.md) | The source of training pairs, the function of the loss, and the reason for intentionally degrading training data. |
| [Reading the numbers](reading-the-numbers.md) | Which metrics are reliable and the threshold for significant model differences. Detailed metric pages are in [Metrics](../metrics/index.md). |
| [What the experiments showed](what-the-experiments-showed.md) | Measured results including mode performance, capacity curve flattening, data vs. model size scaling, and performance limits. |
| [Experiment log](experiment-log.md) | A table of experiments detailing outcomes and open questions. |

## The one-paragraph version

A digital-voice codec discards approximately 98% of the data and reconstructs speech using a low-resolution model. This produces intelligible but synthetic audio where pitch is quantized, inter-harmonic fine structure is removed, phase is approximated, and frequencies above 3.7 kHz are omitted. unamblify operates after the decoder to restore statistically predictable components. The initial design used a small network to apply a time-varying filter to the decoded audio and extended it to 16 kHz. This reshaped the input signal directly to maintain real-time performance and reached a predicted MOS near 2.1. The current design reconstructs a wideband spectrogram from the decoded audio and processes it with a waveform synthesizer ([candidate 3](../design/training.md#candidate-3-concretely)). On identical test clips it achieves a predicted MOS of 3.57 compared to the codec's 2.22 and the original recording's 4.14, and scored 4.6 out of 5 in a blind listening test.
