# Research

Underlying facts and sources for the design.

| Page | Question it answers |
|---|---|
| [AMBE primer](ambe-primer.md) | Details what the vocoder retains and discards and which losses cause robotic artifacts. Explains AMBE-3000 hardware operation per mode. Details M17 payload (Codec 2) and framing. |
| [Data sources](data-sources.md) | Details speech corpora selection, licensing and acquisition methods. |
| [Corpus inventory](corpus-inventory.md) | Lists current prepared and captured drive contents and requirements for publishing AMBE pairs. |
| [Hardware throughput](hardware-throughput.md) | Details daily speech processing capacity of a ThumbDV and implications for corpus size and capture order. |
| [Prior art](prior-art.md) | Reviews neural post-filtering and bandwidth extension of low-rate codecs, successful approaches and available Rust ML tooling for AMD GPU training. |

For the reasoning behind these facts, see [Theory](../theory/index.md). It covers the relationship between codec loss and artifacts, network handling and measurement interpretation.

Research pages include dates and source citations. Findings that alter decisions are noted in the relevant design pages.
