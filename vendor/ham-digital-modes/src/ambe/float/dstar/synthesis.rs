// SPDX-License-Identifier: LGPL-3.0-or-later
//! D-STAR bits-to-PCM decoding: `parse_frame` (FEC + de-whitening) -> `dequantize` -> the shared
//! [`super::super::mbe_synthesis::MbeSynthesizer`]. Tone frames (DTMF / single tone) are recognized
//! by `dequantize` and synthesized as sinusoids via [`ToneSynthesizer`].

use super::decode::{
    classify_tone_index, dequantize, dtmf_digit_from_tone_index, parse_frame, DStarDecoderState,
    DequantizedFrame, ToneKind,
};
use crate::ambe::float::mbe_synthesis::MbeSynthesizer;
use crate::ambe::float::ratet27::unvoiced_synthesis::N;
use crate::ambe::float::tone_synthesis::{dstar_tone_amplitude, ToneSynthesizer};

pub struct DStarSynthesisDecoder {
    dequant: DStarDecoderState,
    synth: MbeSynthesizer,
    tone: ToneSynthesizer,
}

impl DStarSynthesisDecoder {
    pub fn new() -> Self {
        Self {
            dequant: DStarDecoderState::initial(),
            synth: MbeSynthesizer::new(),
            tone: ToneSynthesizer::new(),
        }
    }

    /// Decodes one logical 72-bit frame (see `interleave::wire_bytes_to_frame`) into 20 ms of PCM.
    pub fn decode_frame(&mut self, logical_frame: u128) -> Option<[f64; N]> {
        let parsed = parse_frame(logical_frame);
        match dequantize(parsed.d, &mut self.dequant) {
            DequantizedFrame::Speech(p) => {
                self.tone.reset();
                self.synth.synthesize_speech(
                    p.w0,
                    &p.voiced,
                    &p.ml,
                    parsed.epsilon_c0,
                    parsed.epsilon_c1,
                )
            }
            DequantizedFrame::Tone(t) => match classify_tone_index(t.index) {
                ToneKind::Single { hz } => {
                    Some(self.tone.synthesize(&[hz], dstar_tone_amplitude(t.volume)))
                }
                ToneKind::Dual => match dtmf_digit_from_tone_index(t.index) {
                    Some((row, col)) => {
                        Some(self.tone.dtmf(row, col, dstar_tone_amplitude(t.volume)))
                    }
                    // Dual-tone codes 144..=163: meaning unidentified, so emit silence.
                    None => Some([0.0; N]),
                },
                ToneKind::Invalid => Some([0.0; N]),
            },
        }
    }
}

impl Default for DStarSynthesisDecoder {
    fn default() -> Self {
        Self::new()
    }
}
