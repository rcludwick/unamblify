// SPDX-License-Identifier: LGPL-3.0-or-later
//! Mode-independent MBE frame synthesis shared by D-STAR and AMBE+2 half-rate: turns one frame's
//! already-dequantized `(w0, per-harmonic voicing, Ml)` into 20 ms of PCM by reusing
//! [`super::ratet27::synthesis::SynthesisState`] (spectral enhancement, V/UV and amplitude
//! smoothing, then voiced + unvoiced synthesis, Eq. 105-142).
//!
//! mbelib runs one shared `mbe_synthesizeSpeechf` for P25, D-STAR and AMBE+2, and its
//! `mbe_spectralAmpEnhance` is the same Eq. 105-110 enhancement `ratet27::enhancement` implements
//! (0.96*pi constant, harmonics with `8*l <= L` left unweighted, weights clamped to `[0.5, 1.2]`,
//! energy renormalized), so the parameter set is identical across modes and nothing here is
//! mode-specific. Two deliberate differences from mbelib remain: this crate's unvoiced half is the
//! spec's DFT construction rather than mbelib's multisine mix, and `SynthesisState` also applies the
//! spec's V/UV and amplitude smoothing, which mbelib omits.
//!
//! FEC error statistics feed the same smoothing thresholds as RATET(27), using the two Golay blocks
//! these two modes actually carry (`epsilon_c0`, `epsilon_c1`) and zero for the vectors they lack.

use super::ratet27::error_estimation::{estimate_errors, FrameErrors};
use super::ratet27::synthesis::SynthesisState;
use super::ratet27::unvoiced_synthesis::N;

pub struct MbeSynthesizer {
    synthesis: SynthesisState,
    error_rate_prev: f64,
}

impl MbeSynthesizer {
    pub fn new() -> Self {
        Self {
            synthesis: SynthesisState::new(),
            error_rate_prev: 0.0,
        }
    }

    fn errors_for(&mut self, epsilon_c0: u32, epsilon_c1: u32) -> FrameErrors {
        let errors = estimate_errors(
            &[epsilon_c0, epsilon_c1, 0, 0, 0, 0, 0],
            self.error_rate_prev,
        );
        self.error_rate_prev = errors.rate;
        errors
    }

    /// Synthesizes one speech frame. `voiced` and `ml` are both 1-indexed by harmonic (index 0 is
    /// unused padding), exactly as `dstar::decode::DStarParameters` and
    /// `ambe_plus_2::decode::Parameters` carry them. Returns `None` on a length mismatch.
    pub fn synthesize_speech(
        &mut self,
        w0: f64,
        voiced: &[bool],
        ml: &[f64],
        epsilon_c0: u32,
        epsilon_c1: u32,
    ) -> Option<[f64; N]> {
        if voiced.len() != ml.len() || voiced.len() < 2 {
            return None;
        }
        let errors = self.errors_for(epsilon_c0, epsilon_c1);
        self.synthesis
            .synthesize_frame(&ml[1..], w0, &voiced[1..], &errors)
    }

    /// Repeats the previous frame's parameters (an erasure); `None` before any real frame has run.
    pub fn synthesize_repeat(&mut self) -> Option<[f64; N]> {
        self.synthesis.synthesize_repeated_frame()
    }

    /// A silence frame: all zeros, like mbelib's `mbe_synthesizeSilencef`.
    pub fn synthesize_silence(&self) -> [f64; N] {
        [0.0; N]
    }
}

impl Default for MbeSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}
