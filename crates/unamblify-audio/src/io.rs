// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! File readers and the one writer. Readers fold every channel layout to
//! mono by averaging channels; sample formats are scaled to `[-1, 1]`.

use std::fs::File;
use std::path::Path;

use flacenc::component::BitRepr;
use flacenc::error::Verify;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::{AudioError, Result};

/// Decoded mono audio.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    /// Samples in `[-1, 1]`.
    pub samples: Vec<f32>,
    /// Sample rate, Hz.
    pub rate: u32,
}

impl Audio {
    /// Length in seconds.
    #[must_use]
    pub fn duration_s(&self) -> f64 {
        self.samples.len() as f64 / f64::from(self.rate)
    }
}

/// Read by extension: `.wav` via hound, `.flac` and `.mp3` via symphonia.
pub fn read(path: impl AsRef<Path>) -> Result<Audio> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("wav") => read_wav(path),
        Some("flac") => read_flac(path),
        Some("mp3") => read_mp3(path),
        _ => Err(AudioError::Unsupported(format!(
            "{}: not .wav, .flac or .mp3",
            path.display()
        ))),
    }
}

/// Read a WAV file (s16 / s24 / s32 integer or f32) to mono.
pub fn read_wav(path: impl AsRef<Path>) -> Result<Audio> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (u32::from(spec.bits_per_sample) - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<std::result::Result<_, _>>()?
        }
    };
    Ok(Audio {
        samples: downmix(&interleaved, channels),
        rate: spec.sample_rate,
    })
}

/// Header facts of a WAV file, without reading its samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavInfo {
    /// Samples per channel.
    pub samples: usize,
    /// Sample rate, Hz.
    pub rate: u32,
    /// Channels.
    pub channels: u16,
}

impl WavInfo {
    /// Length in seconds.
    #[must_use]
    pub fn duration_s(&self) -> f64 {
        self.samples as f64 / f64::from(self.rate)
    }
}

/// Read a WAV file's header.
pub fn wav_info(path: impl AsRef<Path>) -> Result<WavInfo> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    Ok(WavInfo {
        samples: reader.duration() as usize,
        rate: spec.sample_rate,
        channels: spec.channels,
    })
}

/// Read `n` samples per channel of a WAV file starting at sample `start`
/// (clamped to the file's end), downmixed to mono — for a window out of a
/// five-minute noise clip without decoding the whole file.
pub fn read_wav_range(path: impl AsRef<Path>, start: usize, n: usize) -> Result<Audio> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    let total = reader.duration() as usize;
    let start = start.min(total);
    let n = n.min(total - start);
    reader.seek(u32::try_from(start).map_err(|_| {
        AudioError::Unsupported(format!(
            "seek to sample {start}: past a 32-bit sample index"
        ))
    })?)?;
    let want = n * channels;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .take(want)
            .collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (u32::from(spec.bits_per_sample) - 1)) as f32;
            reader
                .samples::<i32>()
                .take(want)
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<std::result::Result<_, _>>()?
        }
    };
    Ok(Audio {
        samples: downmix(&interleaved, channels),
        rate: spec.sample_rate,
    })
}

/// Read a FLAC file to mono.
pub fn read_flac(path: impl AsRef<Path>) -> Result<Audio> {
    read_symphonia(path.as_ref(), "flac")
}

/// Read an MP3 file to mono, at the file's real sample rate.
pub fn read_mp3(path: impl AsRef<Path>) -> Result<Audio> {
    read_symphonia(path.as_ref(), "mp3")
}

/// Decode a symphonia-supported file (`.flac` or `.mp3`) to mono, folding
/// every channel by averaging and returning the file's real sample rate.
/// `ext` is the probe hint and names the format in any error.
fn read_symphonia(path: &Path, ext: &str) -> Result<Audio> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let mut hint = Hint::new();
    hint.with_extension(ext);
    let probed = match symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    ) {
        Ok(probed) => probed,
        // A FLAC that holds no audio frames (an empty capture, as produced by
        // `write_flac_s16` on an empty slice): symphonia reads the first frame
        // while probing and hits end-of-stream, so it never opens the file.
        // Recover the sample rate straight from STREAMINFO and return zero
        // samples.
        Err(SymphoniaError::IoError(e))
            if ext == "flac" && e.kind() == std::io::ErrorKind::UnexpectedEof =>
        {
            return Ok(Audio {
                samples: Vec::new(),
                rate: flac_streaminfo_rate(path)?,
            });
        }
        Err(e) => return Err(e.into()),
    };
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AudioError::Unsupported(format!("{ext}: no decodable track")))?;
    let track_id = track.id;
    let rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| AudioError::Unsupported(format!("{ext}: no sample rate")))?;
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut mono = Vec::new();
    let mut buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let audio = decoder.decode(&packet)?;
        let spec = *audio.spec();
        let channels = spec.channels.count().max(1);
        let needed = audio.frames() * channels;
        let sb = match buf.as_mut() {
            Some(sb) if sb.capacity() >= needed => sb,
            _ => buf.insert(SampleBuffer::<f32>::new(audio.capacity() as u64, spec)),
        };
        sb.copy_interleaved_ref(audio);
        push_downmixed(&mut mono, sb.samples(), channels);
    }
    Ok(Audio {
        samples: mono,
        rate,
    })
}

/// Read the sample rate from a FLAC `STREAMINFO` block without decoding audio.
///
/// Used only to recover the rate of a frameless (empty) FLAC, which symphonia
/// cannot open. The layout is fixed: `fLaC` (4 bytes), the first metadata block
/// header (4 bytes), then the 34-byte `STREAMINFO` payload whose 20-bit sample
/// rate begins 10 bytes in — absolute offset 18.
fn flac_streaminfo_rate(path: &Path) -> Result<u32> {
    let bytes = std::fs::read(path)?;
    let off = 4 + 4 + 10;
    match (bytes.get(off), bytes.get(off + 1), bytes.get(off + 2)) {
        (Some(&b0), Some(&b1), Some(&b2)) => {
            Ok((u32::from(b0) << 12) | (u32::from(b1) << 4) | (u32::from(b2) >> 4))
        }
        _ => Err(AudioError::Unsupported(
            "flac: truncated STREAMINFO".to_owned(),
        )),
    }
}

/// Write mono samples as a 16-bit PCM WAV, clipping to `[-1, 1]`.
pub fn write_wav_s16(path: impl AsRef<Path>, samples: &[f32], rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample(to_s16(s))?;
    }
    writer.finalize()?;
    Ok(())
}

/// Write mono samples as a lossless 16-bit PCM FLAC, clipping to `[-1, 1]`.
///
/// Uses the pure-Rust `flacenc` encoder — no C dependency and no borrowed
/// non-`Send` state — so this is safe to call from a spawned background
/// thread. The encode is lossless: the samples are quantised to s16 exactly
/// as [`write_wav_s16`] does (via [`to_s16`]), and decoding the result with
/// [`read_flac`] yields byte-identical s16 samples at the same sample rate.
///
/// # Errors
///
/// Returns [`AudioError::FlacEncode`] if the encoder rejects its configuration
/// or fails to serialise the stream, and [`AudioError::Io`] if the file cannot
/// be written.
pub fn write_flac_s16(path: impl AsRef<Path>, samples: &[f32], rate: u32) -> Result<()> {
    let pcm: Vec<i32> = samples.iter().map(|&s| i32::from(to_s16(s))).collect();
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|(_, e)| AudioError::FlacEncode(format!("config: {e:?}")))?;
    let block_size = config.block_size;
    let source = flacenc::source::MemSource::from_samples(&pcm, 1, 16, rate as usize);
    let mut stream = flacenc::encode_with_fixed_block_size(&config, source, block_size)
        .map_err(|e| AudioError::FlacEncode(format!("{e:?}")))?;
    // symphonia (our `read_flac`) infers the fixed-blocksize blocking strategy
    // from `min_block_size == max_block_size` in STREAMINFO, but flacenc lowers
    // the minimum to the size of the last (partial) frame, which trips that
    // heuristic and makes symphonia misparse every stream whose length is not a
    // whole multiple of the block size. Force the two equal — spec-compliant and
    // exactly what the reference `flac` encoder writes for a fixed-blocksize
    // stream — so any length round-trips. A frame is still allowed to be shorter
    // than the declared block size; only the last one ever is.
    stream
        .stream_info_mut()
        .set_block_sizes(block_size, block_size)
        .map_err(|e| AudioError::FlacEncode(format!("block sizes: {e:?}")))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| AudioError::FlacEncode(format!("write: {e:?}")))?;
    std::fs::write(path, sink.as_slice())?;
    Ok(())
}

/// Clip and round one sample to s16.
#[must_use]
pub fn to_s16(s: f32) -> i16 {
    (s.clamp(-1.0, 1.0) * 32_767.0).round() as i16
}

fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(interleaved.len() / channels);
    push_downmixed(&mut out, interleaved, channels);
    out
}

fn push_downmixed(out: &mut Vec<f32>, interleaved: &[f32], channels: usize) {
    if channels == 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    let inv = 1.0 / channels as f32;
    out.extend(
        interleaved
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() * inv),
    );
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::testutil::sine;

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("unamblify-audio-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn wav_s16_round_trip() {
        let x = sine(440.0, 16_000, 1600, 0.5);
        let p = tmp("rt.wav");
        write_wav_s16(&p, &x, 16_000).unwrap();
        let a = read(&p).unwrap();
        assert_eq!(a.rate, 16_000);
        assert_eq!(a.samples.len(), x.len());
        let max_err = a
            .samples
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 1.0 / 32_000.0, "{max_err}");
        assert!((a.duration_s() - 0.1).abs() < 1e-9);
    }

    /// The s16 value symphonia's decoder reconstructs from a decoded f32
    /// sample (it scales i16 by `1 / 32768`); comparing these integers is the
    /// byte-identical / lossless check.
    fn decoded_s16(got: f32) -> i16 {
        (got * 32_768.0).round() as i16
    }

    #[test]
    fn flac_s16_is_lossless() {
        // A few thousand full-scale samples: a sine plus deterministic noise,
        // pushed past full scale so the s16 quantiser both saturates and clips.
        let n = 5_000;
        let mut x = Vec::with_capacity(n);
        let mut state: u32 = 0x1234_5678;
        for i in 0..n {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 8) as f32 / f32::from(u16::MAX) - 0.5; // ~[-0.5, 0.5)
            let tone = (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 16_000.0).sin();
            x.push((tone * 0.9 + noise * 0.8).clamp(-1.5, 1.5));
        }
        let p = tmp("lossless.flac");
        write_flac_s16(&p, &x, 16_000).unwrap();

        let a = read_flac(&p).unwrap();
        assert_eq!(a.rate, 16_000);
        assert_eq!(a.samples.len(), x.len());
        // Byte-identical: every decoded s16 equals the s16 the writer stored.
        for (i, (&got, &src)) in a.samples.iter().zip(&x).enumerate() {
            assert_eq!(decoded_s16(got), to_s16(src), "sample {i}");
        }
    }

    #[test]
    fn flac_handles_empty_and_tiny_buffers() {
        // Empty buffer: no frames, so symphonia falls back to STREAMINFO.
        let p = tmp("empty.flac");
        write_flac_s16(&p, &[], 8_000).unwrap();
        let a = read_flac(&p).unwrap();
        assert_eq!(a.rate, 8_000);
        assert_eq!(a.samples.len(), 0);

        // A handful of full-scale samples, far shorter than one FLAC block.
        let tiny = [1.0f32, -1.0, 0.5, -0.5, 0.0];
        let p = tmp("tiny.flac");
        write_flac_s16(&p, &tiny, 48_000).unwrap();
        let a = read_flac(&p).unwrap();
        assert_eq!(a.rate, 48_000);
        assert_eq!(a.samples.len(), tiny.len());
        for (&got, &src) in a.samples.iter().zip(&tiny) {
            assert_eq!(decoded_s16(got), to_s16(src));
        }
    }

    #[test]
    fn stereo_and_24_bit_and_float_are_downmixed_and_scaled() {
        let p = tmp("stereo24.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&p, spec).unwrap();
        // L = +0.5, R = -0.25 → mono 0.125.
        for _ in 0..100 {
            w.write_sample((0.5 * 8_388_607.0) as i32).unwrap();
            w.write_sample((-0.25 * 8_388_607.0) as i32).unwrap();
        }
        w.finalize().unwrap();
        let a = read_wav(&p).unwrap();
        assert_eq!(a.rate, 48_000);
        assert_eq!(a.samples.len(), 100);
        assert!((a.samples[0] - 0.125).abs() < 1e-5, "{}", a.samples[0]);

        let p = tmp("f32.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 8_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(&p, spec).unwrap();
        w.write_sample(0.75f32).unwrap();
        w.finalize().unwrap();
        assert_eq!(read_wav(&p).unwrap().samples, vec![0.75]);
    }

    #[test]
    fn a_window_reads_the_same_samples_as_the_whole_file() {
        let x = sine(440.0, 16_000, 4_000, 0.5);
        let p = tmp("window.wav");
        write_wav_s16(&p, &x, 16_000).unwrap();
        let info = wav_info(&p).unwrap();
        assert_eq!(
            info,
            WavInfo {
                samples: 4_000,
                rate: 16_000,
                channels: 1
            }
        );
        assert!((info.duration_s() - 0.25).abs() < 1e-9);
        let whole = read_wav(&p).unwrap();
        let w = read_wav_range(&p, 1_000, 500).unwrap();
        assert_eq!(w.rate, 16_000);
        assert_eq!(w.samples, whole.samples[1_000..1_500]);
        // Clamped at the end; past the end is empty.
        assert_eq!(read_wav_range(&p, 3_900, 500).unwrap().samples.len(), 100);
        assert!(read_wav_range(&p, 9_000, 10).unwrap().samples.is_empty());
        // Stereo: downmixed, per-channel sample indices.
        let p2 = tmp("window-stereo.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 8_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut wr = hound::WavWriter::create(&p2, spec).unwrap();
        for i in 0..100i16 {
            wr.write_sample(i * 100).unwrap();
            wr.write_sample(-i * 100).unwrap();
        }
        wr.finalize().unwrap();
        assert_eq!(wav_info(&p2).unwrap().channels, 2);
        assert_eq!(wav_info(&p2).unwrap().samples, 100);
        let w = read_wav_range(&p2, 10, 5).unwrap();
        assert_eq!(w.samples.len(), 5);
        assert!(w.samples.iter().all(|&v| v == 0.0), "L + R cancel");
    }

    #[test]
    fn unknown_extension_is_refused() {
        assert!(matches!(
            read("/nonexistent/x.ogg"),
            Err(AudioError::Unsupported(_))
        ));
        assert!(read("/nonexistent/x.flac").is_err());
    }

    #[test]
    fn read_routes_mp3_to_the_mp3_decoder() {
        // A `.mp3` path routes to the symphonia decode path, not the
        // "unsupported extension" branch: a missing file yields an I/O
        // error, never `Unsupported`.
        assert!(matches!(read("/nonexistent/x.mp3"), Err(AudioError::Io(_))));
    }

    #[test]
    fn mp3_fixture_decodes_to_mono_at_its_real_rate() {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/sine440-mono-16k.mp3");
        let a = read(&p).unwrap();
        assert_eq!(a.rate, 16_000, "the file's real sample rate");
        // ~0.3 s of audio (MP3 encoder/decoder delay shifts the exact count).
        assert!(a.samples.len() > 3_000, "{}", a.samples.len());
        assert!(a.duration_s() > 0.2, "{}", a.duration_s());
        // A 440 Hz tone is well inside [-1, 1] and not all-zero.
        let peak = a.samples.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(peak > 0.05 && peak <= 1.0, "{peak}");
    }

    #[test]
    fn s16_clips() {
        assert_eq!(to_s16(2.0), 32_767);
        assert_eq!(to_s16(-2.0), -32_767);
        assert_eq!(to_s16(0.0), 0);
    }
}
