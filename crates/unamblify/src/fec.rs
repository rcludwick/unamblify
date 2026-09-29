// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The forward error correction YSF DN puts around AMBE+2 voice bits.
//!
//! The chip is told 2450 bit/s with no FEC for YSF (`chip::ratep_ysf`), so
//! unlike DMR the protection is the host's job. That makes it something
//! this crate can model in software, which is what `augment --kind ber`
//! needs in order to corrupt a `ysf-dmr` capture the way the air does.
//!
//! The 49 voice bits sit in the codec's logical order: twelve protected by
//! an extended Golay (24, 12) word, twelve protected by a Golay (23, 12)
//! word, and twenty-five protected by nothing at all. V/D mode 2 protects
//! a different set: the first twenty-seven bits are sent three times and
//! voted, and the remaining twenty-two go bare.
//!
//! Two consequences decide what a flipped bit sounds like, and both are
//! reproduced here:
//!
//! * The second word is whitened by a sequence keyed on the *decoded*
//!   first word ([`whiten`]). Mis-correct the first word and every bit of
//!   the second is lost with it.
//! * A first word further than three bits from any codeword cannot be
//!   repaired, and since it holds the whitening key the whole frame goes,
//!   including the twenty-five unprotected bits. A receiver substitutes
//!   its mute codeword, which is this crate's [`crate::channel::
//!   YSF_DMR_MUTE_FRAME`].

use std::sync::OnceLock;

/// Generator polynomial of the (23, 12) cyclic Golay code, low bit first:
/// `x^11 + x^10 + x^6 + x^5 + x^4 + x^2 + 1`.
const GENERATOR: u32 = 0xC75;

/// Encode twelve data bits into a 24-bit extended Golay codeword: data in
/// bits 23..12, eleven parity bits in 12..1, overall even parity in bit 0.
#[must_use]
pub fn golay24_encode(data: u16) -> u32 {
    let data = u32::from(data) & 0xFFF;
    let mut rem = data << 11;
    for i in (11..23).rev() {
        if rem >> i & 1 == 1 {
            rem ^= GENERATOR << (i - 11);
        }
    }
    let word23 = (data << 11) | (rem & 0x7FF);
    (word23 << 1) | (word23.count_ones() & 1)
}

/// Every codeword, indexed by its data word. Built once.
fn codewords() -> &'static [u32; 4096] {
    static TABLE: OnceLock<[u32; 4096]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0u32; 4096];
        for (data, slot) in table.iter_mut().enumerate() {
            *slot = golay24_encode(u16::try_from(data).unwrap_or(0));
        }
        table
    })
}

/// Decode a 24-bit codeword, correcting up to three bit errors.
///
/// `None` when the word is further than three bits from every codeword.
/// The minimum distance is 8, so a word within three of one codeword is
/// within three of no other and the first hit is the only hit.
#[must_use]
pub fn golay24_decode(received: u32) -> Option<u16> {
    let received = received & 0x00FF_FFFF;
    for (data, &codeword) in codewords().iter().enumerate() {
        if (codeword ^ received).count_ones() <= 3 {
            return u16::try_from(data).ok();
        }
    }
    None
}

/// Decode a 23-bit Golay (23, 12) word: the (24, 12) code with its overall
/// parity bit dropped.
///
/// This never gives up. The (23, 12) code has minimum distance 7, and past
/// what it can correct the nearest word is still the best available guess,
/// which is what the reference receivers take.
#[must_use]
pub fn golay23_decode(received: u32) -> u16 {
    let received = received & 0x007F_FFFF;
    let mut best = 0u16;
    let mut best_distance = u32::MAX;
    for (data, &codeword) in codewords().iter().enumerate() {
        let distance = ((codeword >> 1) ^ received).count_ones();
        if distance < best_distance {
            best_distance = distance;
            best = u16::try_from(data).unwrap_or(0);
        }
    }
    best
}

/// The whitening sequence that scrambles the second protected word, keyed
/// on the twelve bits the first word carries.
///
/// A linear congruential generator, `x <- 173x + 13849 (mod 2^16)` from a
/// seed of `16 · data`, taking the top bit of each of twenty-four steps.
/// Callers shift right by one to whiten a 23-bit word.
#[must_use]
pub fn whiten(data: u16) -> u32 {
    let mut x = u32::from(data) * 16;
    let mut word = 0u32;
    for _ in 0..24 {
        x = (173 * x + 13849) % 65536;
        word = (word << 1) | (x >> 15);
    }
    word
}

/// Majority vote over three copies of a bit, as V/D mode 2 sends the first
/// twenty-seven voice bits.
#[must_use]
pub fn majority3(a: bool, b: bool, c: bool) -> bool {
    // Counting the votes says what this means, and it is how the
    // reference receiver decides.
    u8::from(a) + u8::from(b) + u8::from(c) >= 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_codeword_round_trips() {
        for data in 0..4096u16 {
            assert_eq!(golay24_decode(golay24_encode(data)), Some(data), "{data}");
        }
    }

    #[test]
    fn the_code_corrects_three_errors_and_refuses_four() {
        // Exhaustive over error patterns for a spread of data words.
        for data in [0u16, 1, 0x555, 0xABC, 0xFFF] {
            let word = golay24_encode(data);
            for i in 0..24 {
                assert_eq!(golay24_decode(word ^ (1 << i)), Some(data), "1 bit");
                for j in (i + 1)..24 {
                    assert_eq!(
                        golay24_decode(word ^ (1 << i) ^ (1 << j)),
                        Some(data),
                        "2 bits"
                    );
                    for k in (j + 1)..24 {
                        let hit = word ^ (1 << i) ^ (1 << j) ^ (1 << k);
                        assert_eq!(golay24_decode(hit), Some(data), "3 bits");
                    }
                }
            }
        }
        // Four errors are detectable but not correctable. The nearest
        // codeword is then either absent or the wrong one; what must not
        // happen is a confident correct answer.
        let word = golay24_encode(0xABC);
        let four = word ^ 0b1111;
        assert_ne!(
            golay24_decode(four),
            Some(0xABC),
            "four errors must not correct"
        );
    }

    #[test]
    fn minimum_distance_is_eight() {
        let table = codewords();
        for a in [0usize, 1, 100, 2000, 4095] {
            for b in 0..4096usize {
                if a == b {
                    continue;
                }
                assert!((table[a] ^ table[b]).count_ones() >= 8, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn the_twenty_three_bit_code_always_answers() {
        for data in [0u16, 7, 0x333, 0xFFF] {
            let word = golay24_encode(data) >> 1;
            assert_eq!(golay23_decode(word), data, "clean");
            for i in 0..23 {
                assert_eq!(golay23_decode(word ^ (1 << i)), data, "one error");
            }
        }
    }

    /// The whitening key is the decoded first word, so a single bit of
    /// difference in the key has to change much of the sequence. That is
    /// what makes a mis-corrected first word destroy the second.
    #[test]
    fn whitening_is_deterministic_and_key_sensitive() {
        assert_eq!(whiten(0x123), whiten(0x123));
        let spread = (whiten(0x123) ^ whiten(0x122)).count_ones();
        assert!(spread >= 6, "one key bit changed only {spread} of 24 bits");
    }

    #[test]
    fn majority_takes_two_of_three() {
        assert!(majority3(true, true, false));
        assert!(majority3(false, true, true));
        assert!(!majority3(true, false, false));
        assert!(!majority3(false, false, false));
        assert!(majority3(true, true, true));
    }
}
