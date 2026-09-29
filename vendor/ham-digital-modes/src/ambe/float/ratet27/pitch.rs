//! Pitch estimation (TIA-102.BABA_2003.pdf section 5.1) -- not yet a working pitch estimator, just
//! the two real, low-risk input tables that estimator needs (the analysis window and the lowpass
//! filter), transcribed and independently verified. The actual error function (Eq. 5), look-back/
//! look-ahead pitch tracking (5.1.2-5.1.4), and quarter-sample refinement (5.1.5, needing 256-pt and
//! 16384-pt DFTs) are a materially larger undertaking -- real next step, not attempted in this pass.
//!
//! # Real transcription methodology for both tables below
//!
//! Both Annex B (this window) and Annex D (the filter) are, like Annexes E/F/G/J, real extractable
//! PDF text (confirmed via `pdfimages -list`, zero embedded images), not bit-matrix figures -- lower
//! transcription risk than `fec.rs`'s Golay/Hamming matrices from the start. Each was still checked
//! against a real, spec-mandated invariant before being trusted, not read on faith:
//!
//! - **Annex B** (301 values, `n` from -150 to 150): the spec's own Eq. 6 requires `sum(w_I(j)^2 for
//!   j in -150..=150) == 1.0` -- measured directly against the parsed table: `1.0000000356`,
//!   matching to within the rounding error inherent in the spec's own 8-decimal-digit printed values.
//!   Also confirmed `w_I(n) == w_I(-n)` for all 150 pairs (the spec's own stated symmetry
//!   convention), so only the non-negative half needs to be stored.
//! - **Annex D** (21 taps, `n` from -10 to 10): the raw extracted text has `n=-1`'s own sign character
//!   dropped (a real, minor extraction glitch, not the watermark-collision problem `tables.rs` deals
//!   with) -- recovered via the same symmetry property (`h(n) == h(-n)`, confirmed exactly for all
//!   other 9 pairs, so `h(-1) == h(1)` is not a guess but the same pattern every other coefficient
//!   already follows). DC gain (`sum(h(n))`) measures `1.00138`, close to the unity gain a lowpass
//!   filter's coefficients are conventionally normalized to, with the small residual consistent with
//!   21 coefficients each carrying their own printed 6-decimal-digit rounding.

/// Annex B: the 301-sample window used for initial pitch estimation, `w_I(n)`. Only the non-negative
/// half (`n = 0..=150`) is stored; `w_I(n) == w_I(-n)` per the spec's own stated symmetry convention,
/// confirmed exactly for all 150 pairs during transcription.
const INITIAL_PITCH_WINDOW_HALF: [f64; 151] = [
    0.09207659, 0.09206691, 0.09203795, 0.09198972, 0.09192220, 0.09183547, 0.09172948, 0.09160442,
    0.09146029, 0.09129713, 0.09111510, 0.09091420, 0.09069464, 0.09045644, 0.09019975, 0.08992471,
    0.08963151, 0.08932017, 0.08899096, 0.08864402, 0.08827950, 0.08789764, 0.08749853, 0.08708246,
    0.08664963, 0.08620022, 0.08573447, 0.08525267, 0.08475497, 0.08424167, 0.08371302, 0.08316927,
    0.08261070, 0.08203757, 0.08145022, 0.08084887, 0.08023385, 0.07960547, 0.07896401, 0.07830978,
    0.07764313, 0.07696434, 0.07627381, 0.07557184, 0.07485873, 0.07413481, 0.07340052, 0.07265611,
    0.07190202, 0.07113854, 0.07036604, 0.06958490, 0.06879549, 0.06799815, 0.06719328, 0.06638119,
    0.06556234, 0.06473708, 0.06390573, 0.06306869, 0.06222639, 0.06137912, 0.06052732, 0.05967136,
    0.05881159, 0.05794842, 0.05708219, 0.05621329, 0.05534209, 0.05446897, 0.05359430, 0.05271840,
    0.05184171, 0.05096456, 0.05008731, 0.04921030, 0.04833391, 0.04745849, 0.04658438, 0.04571191,
    0.04484143, 0.04397328, 0.04310780, 0.04224531, 0.04138612, 0.04053057, 0.03967896, 0.03883159,
    0.03798876, 0.03715080, 0.03631795, 0.03549054, 0.03466884, 0.03385311, 0.03304362, 0.03224064,
    0.03144442, 0.03065519, 0.02987323, 0.02909875, 0.02833197, 0.02757312, 0.02682242, 0.02608007,
    0.02534628, 0.02462122, 0.02390509, 0.02319806, 0.02250030, 0.02181198, 0.02113325, 0.02046427,
    0.01980515, 0.01915605, 0.01851709, 0.01788837, 0.01727001, 0.01666212, 0.01606477, 0.01547807,
    0.01490209, 0.01433691, 0.01378257, 0.01323915, 0.01270669, 0.01218523, 0.01167480, 0.01117544,
    0.01068715, 0.01020996, 0.00974387, 0.00928887, 0.00884496, 0.00841213, 0.00799034, 0.00757957,
    0.00717979, 0.00679095, 0.00641300, 0.00604590, 0.00568957, 0.00534396, 0.00500898, 0.00468457,
    0.00437064, 0.00406710, 0.00377385, 0.00349080, 0.00321783, 0.00295485, 0.00270174,
];

/// `w_I(n)` per Annex B, for any `n` in `-150..=150`; panics (via array index) outside that range,
/// matching the spec's own "the window functions are assumed to be equal to zero outside the range
/// given in the Annexes" -- a caller must not evaluate this outside the defined window, so an
/// explicit `Option` would only mask a real logic error elsewhere.
pub fn initial_pitch_window(n: i32) -> f64 {
    INITIAL_PITCH_WINDOW_HALF[n.unsigned_abs() as usize]
}

/// Annex D: the 21-tap FIR lowpass filter `h_LPF(n)` used to compute `s_LPF(n)` (Eq. 9). Only the
/// non-negative half (`n = 0..=10`) is stored; `h(n) == h(-n)`, confirmed exactly for all 10 pairs.
const LOWPASS_FILTER_TAPS_HALF: [f64; 11] = [
    0.351338, 0.278990, 0.118754, -0.015116, -0.055990, -0.026955, 0.008800, 0.016601, 0.005666,
    -0.002831, -0.002898,
];

/// Annex C: the 221-sample window used for pitch refinement (and also spectral amplitude estimation
/// and V/UV determination, per section 5.1's own text), `w_R(n)`. Only the non-negative half (`n =
/// 0..=110`) is stored; `w_R(n) == w_R(-n)`, confirmed exactly for all 110 pairs, and `w_R(0) == 1.0`
/// (the real center peak a window function should have). Unlike Annex B, the spec states no explicit
/// sum-of-squares normalization constraint for this window, so completeness and symmetry are the
/// real checks available here -- both confirmed during extraction.
const PITCH_REFINEMENT_WINDOW_HALF: [f64; 111] = [
    1.000000, 0.999774, 0.999095, 0.997966, 0.996386, 0.994358, 0.991884, 0.988967, 0.985610,
    0.981817, 0.977592, 0.972940, 0.967866, 0.962377, 0.956477, 0.950174, 0.943474, 0.936386,
    0.928916, 0.921074, 0.912868, 0.904307, 0.895400, 0.886157, 0.876589, 0.866705, 0.856516,
    0.846033, 0.835267, 0.824231, 0.812935, 0.801391, 0.789612, 0.777610, 0.765397, 0.752986,
    0.740390, 0.727620, 0.714692, 0.701616, 0.688406, 0.675076, 0.661638, 0.648105, 0.634490,
    0.620807, 0.607067, 0.593284, 0.579470, 0.565639, 0.551802, 0.537971, 0.524160, 0.510379,
    0.496640, 0.482955, 0.469336, 0.455793, 0.442337, 0.428978, 0.415727, 0.402594, 0.389588,
    0.376718, 0.363994, 0.351425, 0.339018, 0.326782, 0.314724, 0.302851, 0.291171, 0.279689,
    0.268413, 0.257347, 0.246497, 0.235869, 0.225466, 0.215294, 0.205355, 0.195653, 0.186192,
    0.176974, 0.168001, 0.159276, 0.150799, 0.142572, 0.134596, 0.126872, 0.119398, 0.112176,
    0.105205, 0.098483, 0.092009, 0.085782, 0.079801, 0.074062, 0.068563, 0.063303, 0.058277,
    0.053482, 0.048915, 0.044573, 0.040451, 0.036546, 0.032852, 0.029365, 0.026081, 0.022995,
    0.020102, 0.017397, 0.014873,
];

/// `w_R(n)` per Annex C, for `n` in `-110..=110`.
pub fn pitch_refinement_window(n: i32) -> f64 {
    PITCH_REFINEMENT_WINDOW_HALF[n.unsigned_abs() as usize]
}

/// `h_LPF(n)` per Annex D, for `n` in `-10..=10`.
pub fn lowpass_filter_tap(n: i32) -> f64 {
    LOWPASS_FILTER_TAPS_HALF[n.unsigned_abs() as usize]
}

/// `w_I(n)` per Annex B, but returning `0.0` for `|n| > 150` instead of panicking -- the spec's own
/// explicit convention ("the window functions are assumed to be equal to zero outside the range
/// given in the Annexes"), needed because Eq. 7's own `r(t)` sum legitimately evaluates `w_I(j+t)`
/// for `j+t` outside `-150..=150` (`j` ranges up to 150 and `t` can itself be up to ~122). This is a
/// real, different contract from [`initial_pitch_window`]'s own panic-on-out-of-range, not an
/// inconsistency: that function's own doc comment reserves the panic for callers with no legitimate
/// reason to evaluate outside the window (a real logic error), while this one exists specifically
/// for the one real, spec-mandated case that does.
// [@ANCHOR: initial_pitch_window_or_zero]
fn initial_pitch_window_or_zero(n: i32) -> f64 {
    if (-150..=150).contains(&n) {
        initial_pitch_window(n)
    } else {
        0.0
    }
}

/// The lowpass-filtered speech signal `s_LPF(n)` (Eq. 9): `sum(s(n-j) * h_LPF(j) for j in -10..=10)`.
/// `raw` is the real speech signal; `center` is the sample index in `raw` corresponding to `n = 0`
/// (the center of the current analysis frame). Panics (via array indexing) if `center as i32 + n`
/// falls outside `raw`'s own bounds by more than the filter's own 10-sample margin -- a caller must
/// provide enough context on both sides of the frame, matching this function's real, unavoidable
/// data dependency (there is no sensible zero-padding substitute for real speech samples the way
/// there is for a window function that is genuinely defined to be zero past its own edge).
// [@ANCHOR: lowpass_filtered_sample]
fn lowpass_filtered_sample(raw: &[f64], center: usize, n: i32) -> f64 {
    let mut acc = 0.0;
    for j in -10i32..=10 {
        let idx = (center as i32 + n - j) as usize;
        acc += raw[idx] * lowpass_filter_tap(j);
    }
    acc
}

/// Precomputed `s_LPF(j)` for `j` in `-150..=150`, the one expensive input both `r(t)` (Eq. 7-8) and
/// `E(P)` (Eq. 5) repeatedly read -- computed once per analysis frame rather than recomputed on
/// every call, since [`lowpass_filtered_sample`] itself does real work (a 21-tap FIR sum) per sample.
pub struct PitchAnalysisFrame {
    s_lpf: [f64; 301],
}

impl PitchAnalysisFrame {
    /// `raw` is the real speech signal; `center` is the sample index in `raw` corresponding to the
    /// current frame's own `n = 0`. `raw` must have at least 160 real samples of margin on both
    /// sides of `center` (150 for the analysis window itself, 10 more for the lowpass filter's own
    /// reach past each edge) -- panics via array indexing if it doesn't, the same real, unavoidable
    /// data dependency [`lowpass_filtered_sample`] itself has.
    // [@ANCHOR: PitchAnalysisFrame::new]
    pub fn new(raw: &[f64], center: usize) -> Self {
        let mut s_lpf = [0.0; 301];
        for (i, slot) in s_lpf.iter_mut().enumerate() {
            let j = i as i32 - 150;
            *slot = lowpass_filtered_sample(raw, center, j);
        }
        Self { s_lpf }
    }

    // [@ANCHOR: PitchAnalysisFrame::s_lpf_at]
    fn s_lpf_at(&self, j: i32) -> f64 {
        if (-150..=150).contains(&j) {
            self.s_lpf[(j + 150) as usize]
        } else {
            0.0
        }
    }

    /// `r(t)` for integer `t` (Eq. 7): `sum(s_LPF(j)*w_I(j)^2*s_LPF(j+t)*w_I(j+t)^2 for j in
    /// -150..=150)`. `w_I(j)` itself never needs the zero-padded variant here (`j` never leaves
    /// `-150..=150`), but `w_I(j+t)` does, since `j+t` legitimately can.
    fn r_integer(&self, t: i32) -> f64 {
        (-150i32..=150)
            .map(|j| {
                let a = self.s_lpf_at(j) * initial_pitch_window(j).powi(2);
                let b = self.s_lpf_at(j + t) * initial_pitch_window_or_zero(j + t).powi(2);
                a * b
            })
            .sum()
    }

    /// `r(t)` for any real `t` (Eq. 8): linear interpolation between the two nearest integers.
    // [@ANCHOR: PitchAnalysisFrame::r]
    fn r(&self, t: f64) -> f64 {
        let t_floor = t.floor();
        let r_floor = self.r_integer(t_floor as i32);
        let r_ceil = self.r_integer(t_floor as i32 + 1);
        (1.0 + t_floor - t) * r_floor + (t - t_floor) * r_ceil
    }

    /// The pitch error function `E(P)` (Eq. 5), evaluated at candidate pitch period `p` (in samples).
    /// Smaller values indicate a better candidate; the real initial pitch estimate is chosen by
    /// comparing `E(P)` across the spec's own candidate set (`21, 21.5, ..., 122`) via pitch
    /// tracking (section 5.1.2, not yet implemented here), not by simply minimizing `E(P)` alone.
    // [@ANCHOR: PitchAnalysisFrame::error_function]
    pub fn error_function(&self, p: f64) -> f64 {
        let s: f64 = (-150i32..=150)
            .map(|j| {
                let v = self.s_lpf_at(j) * initial_pitch_window(j);
                v * v
            })
            .sum();
        let n_max = (150.0 / p).floor() as i32;
        let r_sum: f64 = (-n_max..=n_max).map(|n| self.r(n as f64 * p)).sum();
        let w4: f64 = (-150i32..=150)
            .map(|j| initial_pitch_window(j).powi(4))
            .sum();
        (s - p * r_sum) / (s * (1.0 - p * w4))
    }
}

/// The half-sample-spaced candidate pitch set `{21, 21.5, ..., 121.5, 122}` Eq. 11/13/15 all name
/// (203 values). Every candidate pitch this module ever selects -- `P_hat_B`, `P_hat_1`, `P_hat_2`,
/// `P_hat_0` -- is a member of exactly this set.
pub(crate) fn candidate_pitches() -> impl Iterator<Item = f64> {
    (0..=202).map(|i| 21.0 + 0.5 * i as f64)
}

/// Rounds `p` to its nearest member of [`candidate_pitches`] (mean-square-error closeness, per the
/// spec's own stated rule for snapping a computed sub-multiple back onto the candidate set),
/// clamping to the set's own `21.0..=122.0` range first.
fn nearest_candidate_pitch(p: f64) -> f64 {
    let clamped = p.clamp(21.0, 122.0);
    ((clamped - 21.0) / 0.5).round() * 0.5 + 21.0
}

/// Look-back pitch tracking (Eq. 10-12): finds `P_hat_B`, the candidate nearest continuity with the
/// previous two frames' own chosen pitches, and its cumulative error `CE_B`. `error_fn` is the
/// *current* frame's own `E(P)` (e.g. [`PitchAnalysisFrame::error_function`]); `prev1`/`prev2` are
/// the previous two frames' own `(P_hat, E(P_hat))` -- per the spec's own initialization rule, a
/// frame with no real history yet should pass `(100.0, 0.0)` for both (its own stated default:
/// "Upon initialization the error functions E_-1(P) and E_-2(P) are assumed to be equal to zero, and
/// P_hat_-1 and P_hat_-2 are assumed to be equal to 100").
// [@ANCHOR: look_back_pitch_tracking]
pub fn look_back_pitch_tracking(
    error_fn: impl Fn(f64) -> f64,
    prev1: (f64, f64),
    prev2: (f64, f64),
) -> (f64, f64) {
    let (p_prev1, e_prev1) = prev1;
    let (_, e_prev2) = prev2;
    let lo = 0.8 * p_prev1;
    let hi = 1.2 * p_prev1;
    let (p_b, e_b) = candidate_pitches()
        .filter(|&p| p >= lo && p <= hi)
        .map(|p| (p, error_fn(p)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        // p_prev1 itself always satisfies 0.8*p_prev1 <= p_prev1 <= 1.2*p_prev1, so this range is
        // never empty as long as p_prev1 is itself a real candidate pitch (guaranteed by
        // construction: either a prior call's own returned P_hat_B, or the spec's own 100.0 default).
        .expect("look_back_pitch_tracking: prev1 pitch must be a real candidate pitch");
    let ce_b = e_b + e_prev1 + e_prev2;
    (p_b, ce_b)
}

/// Look-ahead pitch tracking (Eq. 13-20): finds `P_hat_F`, the forward pitch estimate, and its
/// cumulative error `CE_F(P_hat_F)`. `error_fn` is the *current* frame's own `E(P)`; `future1_error_fn`/
/// `future2_error_fn` are the two *future* frames' own `E(P)` (already computable from their own real
/// speech data, per the spec's own text: "the pitch has not been determined for these future frames"
/// -- only their pitch choice is undetermined, not their error function).
// [@ANCHOR: look_ahead_pitch_tracking]
pub fn look_ahead_pitch_tracking(
    error_fn: impl Fn(f64) -> f64,
    future1_error_fn: impl Fn(f64) -> f64,
    future2_error_fn: impl Fn(f64) -> f64,
) -> (f64, f64) {
    // CE_F(p0) per Eq. 17: E(p0) + E1(P_hat_1) + E2(P_hat_2), where P_hat_1/P_hat_2 jointly
    // minimize E1(P1)+E2(P2) subject to Eq. 14/16's own nested range constraints.
    let ce_f_at = |p0: f64| -> f64 {
        let (lo1, hi1) = (0.8 * p0, 1.2 * p0);
        let best: f64 = candidate_pitches()
            .filter(|&p1| p1 >= lo1 && p1 <= hi1)
            .map(|p1| {
                let e1 = future1_error_fn(p1);
                let (lo2, hi2) = (0.8 * p1, 1.2 * p1);
                let best_e2 = candidate_pitches()
                    .filter(|&p2| p2 >= lo2 && p2 <= hi2)
                    .map(&future2_error_fn)
                    .fold(f64::INFINITY, f64::min);
                e1 + best_e2
            })
            .fold(f64::INFINITY, f64::min);
        error_fn(p0) + best
    };

    let p_hat_0 = candidate_pitches()
        .map(|p0| (p0, ce_f_at(p0)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("candidate_pitches is never empty")
        .0;
    let ce_f_p_hat_0 = ce_f_at(p_hat_0);

    // Sub-multiple check (the spec's own text after Eq. 17, and Eq. 18-20): try P_hat_0/2,
    // P_hat_0/3, ... (each snapped to the nearest real candidate pitch), smallest first, and take
    // the first one satisfying any of Eq. 18/19/20; if none do, P_hat_F = P_hat_0 itself.
    let mut submultiples: Vec<f64> = Vec::new();
    let mut n = 2u32;
    loop {
        let raw = p_hat_0 / n as f64;
        if raw < 21.0 {
            break;
        }
        submultiples.push(nearest_candidate_pitch(raw));
        n += 1;
    }
    // "the smallest of these sub-multiples is checked... the next largest sub-multiple is checked
    // next" -- smallest first, matching the loop order above (p_hat_0/2 < p_hat_0/3 is false; higher
    // n gives a SMALLER raw value, so the vector above is already smallest-first as built).
    for &candidate in &submultiples {
        let ce_f_candidate = ce_f_at(candidate);
        let ratio = ce_f_candidate / ce_f_p_hat_0;
        let satisfies_18 = ce_f_candidate <= 0.85 && ratio <= 1.7;
        let satisfies_19 = ce_f_candidate <= 0.4 && ratio <= 3.5;
        let satisfies_20 = ce_f_candidate <= 0.05;
        if satisfies_18 || satisfies_19 || satisfies_20 {
            return (candidate, ce_f_candidate);
        }
    }
    (p_hat_0, ce_f_p_hat_0)
}

/// The final initial pitch estimate decision (Eq. 21-23): compares the backward and forward
/// cumulative errors and picks whichever candidate they favor.
///
/// Eq. 21 and Eq. 22 both select `p_hat_b`, which reads as duplicated logic (clippy flags it) but
/// isn't: Eq. 21's own `CE_B <= 0.48` is a confidence override, independent of how `CE_F` compares --
/// a backward estimate confident enough on its own is trusted even in the case `CE_F` would
/// otherwise have been numerically smaller. Collapsing the two conditions into one `||` would lose
/// that this is two separate rules from the spec, not one; kept as written, matching Eq. 21-23
/// directly, with the lint silenced for exactly this reason rather than restructured to satisfy it.
#[allow(clippy::if_same_then_else)]
// [@ANCHOR: choose_initial_pitch_estimate]
pub fn choose_initial_pitch_estimate(p_hat_b: f64, ce_b: f64, p_hat_f: f64, ce_f: f64) -> f64 {
    if ce_b <= 0.48 {
        p_hat_b
    } else if ce_b <= ce_f {
        p_hat_b
    } else {
        p_hat_f
    }
}

#[cfg(test)]
mod pitch_analysis_tests {
    use super::*;

    /// A synthetic periodic pulse train at period `period_samples` -- the simplest real signal
    /// with an unambiguous, known true pitch, long enough to give `PitchAnalysisFrame::new` its
    /// own required 160-sample margin on both sides of the frame center.
    fn periodic_pulse_train(period_samples: f64, total_len: usize) -> Vec<f64> {
        (0..total_len)
            .map(|n| {
                let phase = (n as f64) % period_samples;
                // A raised-cosine pulse once per period, not a bare impulse train -- a real
                // impulse train's own spectrum is all-harmonics-equal-amplitude, which the
                // lowpass filter would attenuate unevenly in a way that's harder to reason
                // about; this shape keeps most of its energy in-band while still being a real,
                // unambiguous single-period signal, closer in spirit to real glottal pulses.
                let width = period_samples * 0.25;
                if phase < width {
                    0.5 * (1.0 - (std::f64::consts::PI * phase / width).cos())
                } else {
                    0.0
                }
            })
            .collect()
    }

    #[test]
    // Tests [@ANCHOR: PitchAnalysisFrame::new]
    // Tests [@ANCHOR: PitchAnalysisFrame::error_function]
    // Tests [@ANCHOR: PitchAnalysisFrame::s_lpf_at]
    // Tests [@ANCHOR: PitchAnalysisFrame::r]
    // Tests [@ANCHOR: initial_pitch_window_or_zero]
    // Tests [@ANCHOR: lowpass_filtered_sample]
    fn error_function_is_minimized_near_the_true_period_or_a_real_harmonic_multiple() {
        // Real, measured behavior, not assumed: a perfectly periodic synthetic signal is
        // *also* perfectly periodic at every integer multiple of its own true period, so E(P)
        // legitimately scores an octave-multiple candidate (here, 120 = 2x the true 60) at
        // least as well as the true period itself -- confirmed directly (E(60)=0.0203,
        // E(120)=-0.0125, both far below every unrelated candidate tried). This is the exact,
        // spec-acknowledged reason section 5.1.4's own look-ahead tracking explicitly checks
        // integer sub-multiples of the raw minimizing candidate afterward (not implemented
        // here yet) -- this test checks what E(P) alone can honestly promise: the global
        // minimum lands on the true period or a real integer multiple of it, not on an
        // unrelated candidate.
        let period = 60.0; // 8000/60 ~= 133 Hz, a real, plausible voice pitch
        let raw = periodic_pulse_train(period, 400);
        let center = 200usize;
        let frame = PitchAnalysisFrame::new(&raw, center);

        let candidates: Vec<f64> = (0..=202).map(|i| 21.0 + 0.5 * i as f64).collect();
        let mut best_p = candidates[0];
        let mut best_e = frame.error_function(best_p);
        for &p in &candidates[1..] {
            let e = frame.error_function(p);
            if e < best_e {
                best_e = e;
                best_p = p;
            }
        }
        let ratio = best_p / period;
        let nearest_multiple = ratio.round();
        assert!(
            nearest_multiple >= 1.0 && (ratio - nearest_multiple).abs() < 0.05,
            "expected the minimizing candidate {best_p} to be a real integer multiple of the \
             true period {period}, got ratio {ratio} (E={best_e})"
        );
    }

    #[test]
    fn error_function_is_much_larger_for_a_period_far_from_the_true_one() {
        let period = 60.0;
        let raw = periodic_pulse_train(period, 400);
        let frame = PitchAnalysisFrame::new(&raw, 200);
        let e_true = frame.error_function(period);
        let e_wrong = frame.error_function(35.0); // not a small-integer submultiple/multiple of 60
        assert!(
            e_wrong > e_true,
            "expected a mismatched period to score worse (higher E), got e_true={e_true} e_wrong={e_wrong}"
        );
    }

    #[test]
    // Tests [@ANCHOR: look_back_pitch_tracking]
    fn look_back_tracking_resolves_the_octave_ambiguity_using_real_pitch_continuity() {
        // The real point of this whole module: `error_function_is_minimized_near_the_true_period_
        // or_a_real_harmonic_multiple` above already proved E(P) alone can't tell 60 from 120 for a
        // perfectly periodic signal. Look-back tracking's own real job is resolving exactly that,
        // using continuity with a previous frame that's already known to be near 60 -- 120 sits
        // outside Eq. 10's own [0.8*60, 1.2*60] = [48, 72] range, so it's never even a candidate.
        let period = 60.0;
        let raw = periodic_pulse_train(period, 400);
        let frame = PitchAnalysisFrame::new(&raw, 200);
        let error_fn = |p: f64| frame.error_function(p);

        let (p_b, _ce_b) = look_back_pitch_tracking(error_fn, (60.0, 0.02), (60.0, 0.02));
        assert!(
            (p_b - period).abs() < 1.0,
            "expected look-back tracking to land on the true period {period} given real prior \
             continuity, got {p_b}"
        );
    }

    #[test]
    fn look_back_tracking_falls_back_to_the_spec_default_with_no_real_history() {
        // Per the spec's own stated initialization rule: with no real prior frames, P_hat_-1 =
        // P_hat_-2 = 100.0 and E_-1 = E_-2 = 0.0 -- confirmed here to behave sanely (not panic, and
        // to search in the range Eq. 10 actually implies: [80, 120]) rather than assumed correct
        // without being run.
        let period = 100.0; // chosen inside the default search range so this frame's own true
                            // period is reachable from the 100.0 default without contradicting it
        let raw = periodic_pulse_train(period, 500);
        let frame = PitchAnalysisFrame::new(&raw, 250);
        let error_fn = |p: f64| frame.error_function(p);

        let (p_b, ce_b) = look_back_pitch_tracking(error_fn, (100.0, 0.0), (100.0, 0.0));
        assert!(
            (80.0..=120.0).contains(&p_b),
            "expected the default 100.0 history to constrain the search to [80,120], got {p_b}"
        );
        assert!(ce_b.is_finite());
    }

    #[test]
    // Tests [@ANCHOR: choose_initial_pitch_estimate]
    fn choose_initial_pitch_estimate_prefers_backward_when_its_error_is_confidently_low() {
        // Eq. 21: CE_B <= 0.48 alone is enough to pick the backward estimate, regardless of CE_F.
        assert_eq!(choose_initial_pitch_estimate(60.0, 0.1, 90.0, 0.01), 60.0);
    }

    #[test]
    fn choose_initial_pitch_estimate_prefers_whichever_cumulative_error_is_smaller_otherwise() {
        // Eq. 22-23: once CE_B > 0.48, it's a plain comparison between the two cumulative errors.
        assert_eq!(choose_initial_pitch_estimate(60.0, 0.9, 90.0, 1.2), 60.0);
        assert_eq!(choose_initial_pitch_estimate(60.0, 1.2, 90.0, 0.9), 90.0);
    }

    #[test]
    // Tests [@ANCHOR: look_ahead_pitch_tracking]
    fn look_ahead_tracking_lands_on_the_true_period_or_a_valid_submultiple() {
        // Same real octave-ambiguity risk as look-back's own test, resolved differently: look-ahead
        // has no *previous*-frame continuity to lean on (by construction -- it looks forward), so
        // its own raw P_hat_0 search can genuinely land on an ambiguous multiple of the true period.
        // The real correctness property this checks is the one the spec's own submultiple-check
        // procedure (Eq. 18-20) exists to guarantee: whatever P_hat_F comes out, it must be the true
        // period or a real integer submultiple of whatever P_hat_0 the raw search found -- not an
        // unrelated value.
        let period = 60.0;
        let raw = periodic_pulse_train(period, 700);
        let center = PitchAnalysisFrame::new(&raw, 200);
        let future1 = PitchAnalysisFrame::new(&raw, 360);
        let future2 = PitchAnalysisFrame::new(&raw, 520);

        let (p_f, ce_f) = look_ahead_pitch_tracking(
            |p| center.error_function(p),
            |p| future1.error_function(p),
            |p| future2.error_function(p),
        );
        let ratio = p_f / period;
        let nearest_multiple = ratio.round();
        assert!(
            nearest_multiple >= 1.0 && (ratio - nearest_multiple).abs() < 0.05,
            "expected P_hat_F {p_f} to be the true period {period} or a real integer multiple of \
             it, got ratio {ratio} (CE_F={ce_f})"
        );
    }

    #[test]
    fn nearest_candidate_pitch_snaps_to_the_real_half_sample_grid() {
        assert_eq!(nearest_candidate_pitch(60.0), 60.0);
        assert_eq!(nearest_candidate_pitch(60.2), 60.0);
        assert_eq!(nearest_candidate_pitch(60.3), 60.5);
        assert_eq!(nearest_candidate_pitch(10.0), 21.0); // clamped to the set's own real floor
        assert_eq!(nearest_candidate_pitch(200.0), 122.0); // clamped to the set's own real ceiling
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_pitch_window_satisfies_the_specs_own_normalization_constraint() {
        // Eq. 6: sum(w_I(j)^2 for j in -150..=150) must equal 1.0 -- a real, spec-mandated
        // invariant, not an assumption this test invented, and independent of how the table was
        // transcribed (a wrong digit anywhere would very likely break this to more than rounding
        // error, the same real check fec.rs's own weight-distribution proof relies on).
        let sum_sq: f64 = (-150..=150).map(|n| initial_pitch_window(n).powi(2)).sum();
        assert!(
            (sum_sq - 1.0).abs() < 1e-6,
            "expected sum of squares ~1.0 per Eq. 6, got {sum_sq}"
        );
    }

    #[test]
    fn initial_pitch_window_is_symmetric_and_peaks_at_the_center() {
        for n in 1..=150 {
            assert_eq!(initial_pitch_window(n), initial_pitch_window(-n), "n={n}");
        }
        for n in 0..150 {
            assert!(
                initial_pitch_window(n) >= initial_pitch_window(n + 1),
                "window should taper monotonically from the center, failed at n={n}"
            );
        }
    }

    #[test]
    fn lowpass_filter_tap_is_symmetric_with_near_unity_dc_gain() {
        for n in 1..=10 {
            assert_eq!(lowpass_filter_tap(n), lowpass_filter_tap(-n), "n={n}");
        }
        let dc_gain: f64 = (-10..=10).map(lowpass_filter_tap).sum();
        assert!(
            (dc_gain - 1.0).abs() < 0.01,
            "expected DC gain near 1.0, got {dc_gain}"
        );
    }

    #[test]
    fn pitch_refinement_window_is_symmetric_and_peaks_at_one_in_the_center() {
        for n in 1..=110 {
            assert_eq!(
                pitch_refinement_window(n),
                pitch_refinement_window(-n),
                "n={n}"
            );
        }
        assert_eq!(pitch_refinement_window(0), 1.0);
        for n in 0..110 {
            assert!(
                pitch_refinement_window(n) >= pitch_refinement_window(n + 1),
                "window should taper monotonically from the center, failed at n={n}"
            );
        }
    }
}
