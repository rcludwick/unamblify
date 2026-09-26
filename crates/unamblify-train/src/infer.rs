// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! `unamblify infer --run-dir D --step N --key K [--mode M] [--out P]`:
//! one whole prepared utterance through one checkpoint, on the CPU, for
//! the dashboard's Samples page. The pair is read exactly as the pipeline
//! loader reads it (`prepared/<key>.8k.wav` is the reference length,
//! `captured/<mode>/<key>.wav` for `--mode` — the first of the run's
//! `[data] modes` by default — the recorded canary lag applied), the
//! model runs over the whole signal at once told that mode
//! ([`crate::eval::forward_whole`]), and the 16 kHz output lands at
//! `<run>/samples/step-N/<clip>.out.wav` with its `spec.json` beside it
//! (one file per clip and step, whichever mode rendered last).

use std::path::{Path, PathBuf};

use anyhow::Context;
use tch::Device;
use tch::nn::VarStore;
use unamblify::run::{checkpoint_dir, sample_out_path, spec_path_for};
use unamblify::{RunConfig, VocoderMode};
use unamblify_audio::{Spec, write_wav_s16};

use crate::checkpoint;
use crate::data::pipeline::{join_manifests, load_utterance, read_lag};
use crate::eval::forward_whole;
use crate::model::Net;

/// What [`run_one`] produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inferred {
    /// The 16 kHz s16 WAV.
    pub out_wav: PathBuf,
    /// Its `spec.json`.
    pub spec: PathBuf,
    /// The mode the run trains on (the chip output that was used).
    pub mode: VocoderMode,
    /// Output samples at 16 kHz (twice the 8 kHz input).
    pub samples: usize,
}

/// Render `key` through checkpoint `step` of the run at `run_dir` into
/// the run's `samples/` directory. Returns the output WAV path.
pub fn run_one(run_dir: &Path, step: u64, key: &str, data_root: &Path) -> anyhow::Result<PathBuf> {
    Ok(run_one_to(run_dir, step, key, data_root, None)?.out_wav)
}

/// [`run_one`] with an explicit output path (`<out>.spec.json` — the
/// `.wav` replaced — is written beside it), in the run's first mode.
pub fn run_one_to(
    run_dir: &Path,
    step: u64,
    key: &str,
    data_root: &Path,
    out: Option<&Path>,
) -> anyhow::Result<Inferred> {
    run_one_in(run_dir, step, key, None, data_root, out)
}

/// [`run_one_to`] in `mode` (one of the run's `[data] modes`; the first
/// when `None`).
pub fn run_one_in(
    run_dir: &Path,
    step: u64,
    key: &str,
    mode: Option<VocoderMode>,
    data_root: &Path,
    out: Option<&Path>,
) -> anyhow::Result<Inferred> {
    unamblify::key::validate(key)?;
    let cfg_path = run_dir.join("config.toml");
    let cfg_text =
        std::fs::read_to_string(&cfg_path).with_context(|| cfg_path.display().to_string())?;
    let cfg = RunConfig::from_toml(&cfg_text).with_context(|| cfg_path.display().to_string())?;
    let ckpt = checkpoint_dir(run_dir, step);
    anyhow::ensure!(
        ckpt.join("model.safetensors").is_file(),
        "no checkpoint at step {step} under {} ({})",
        run_dir.display(),
        ckpt.display()
    );
    let modes = cfg.data.modes()?;
    let mode = mode.unwrap_or(modes[0]);
    let mode_idx = modes
        .iter()
        .position(|&m| m == mode)
        .and_then(|i| u8::try_from(i).ok())
        .with_context(|| {
            format!(
                "{mode} is not one of the run's modes ({})",
                modes
                    .iter()
                    .map(|m| m.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })?;

    // The pair, as the loaders see it.
    let items = join_manifests(data_root, mode)?;
    let item = items
        .iter()
        .find(|i| i.key == key)
        .with_context(|| format!("{key}: not in both manifests for {mode}"))?;
    let lag = read_lag(data_root, mode)?;
    let utt = load_utterance(data_root, item, lag)?;
    let len8 = utt.deg8.len();

    let mut vs = VarStore::new(Device::Cpu);
    let embed = cfg.model.mode_embed.then_some(modes.len());
    let net = Net::build(&vs.root(), &crate::model::net_opts(&cfg.model, embed));
    let meta = checkpoint::load(&ckpt, &mut vs, None)?;
    anyhow::ensure!(
        meta.step == step,
        "{}: meta.json says step {}, not {step}",
        ckpt.display(),
        meta.step
    );

    let mut out16 = forward_whole(&net, &utt.deg8, mode_idx, Device::Cpu)?;
    out16.truncate(2 * len8);

    let out_wav = out.map_or_else(|| sample_out_path(run_dir, step, key), Path::to_path_buf);
    if let Some(parent) = out_wav.parent() {
        std::fs::create_dir_all(parent).with_context(|| parent.display().to_string())?;
    }
    // Write into siblings and rename, so a poll never sees a torn file.
    let tmp_wav = out_wav.with_extension("wav.tmp");
    write_wav_s16(&tmp_wav, &out16, 16_000).with_context(|| tmp_wav.display().to_string())?;
    let spec = spec_path_for(&out_wav);
    let tmp_spec = spec.with_extension("json.tmp");
    std::fs::write(
        &tmp_spec,
        serde_json::to_string(&Spec::new().with("out", &out16))?,
    )
    .with_context(|| tmp_spec.display().to_string())?;
    std::fs::rename(&tmp_spec, &spec)?;
    std::fs::rename(&tmp_wav, &out_wav)?;
    Ok(Inferred {
        out_wav,
        spec,
        mode,
        samples: out16.len(),
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use unamblify::{CanaryRecord, CaptureRow, Split, UtteranceRow};
    use unamblify_audio::resample;

    /// A synthetic data root with one prepared + captured utterance for
    /// `mode`, the capture lagged by `lag` samples as a real chip's is.
    pub(crate) fn synthetic_root(root: &Path, key: &str, mode: VocoderMode, lag: i32) -> usize {
        let utt = crate::data::synthetic::utterance(7, 1.0).unwrap();
        let clean8 = resample(&utt.clean16, 16_000, 8_000).unwrap();
        let prepared = root.join("prepared");
        let captured = root.join("captured").join(mode.as_str());
        std::fs::create_dir_all(prepared.join(key).parent().unwrap()).unwrap();
        std::fs::create_dir_all(captured.join(key).parent().unwrap()).unwrap();
        write_wav_s16(
            prepared.join(format!("{key}.16k.wav")),
            &utt.clean16,
            16_000,
        )
        .unwrap();
        write_wav_s16(prepared.join(format!("{key}.8k.wav")), &clean8, 8_000).unwrap();
        // The chip's output arrives `lag` samples late.
        let mut late = vec![0.0f32; usize::try_from(lag).unwrap()];
        late.extend_from_slice(&utt.deg8);
        late.truncate(utt.deg8.len());
        write_wav_s16(captured.join(format!("{key}.wav")), &late, 8_000).unwrap();
        let row = UtteranceRow {
            key: key.to_owned(),
            corpus: unamblify::key::corpus_of(key).unwrap().to_owned(),
            speaker: "s0".to_owned(),
            gender: None,
            split: Split::Dev,
            duration_s: 1.0,
            src_rate: 16_000,
            src_path: "raw/x.wav".to_owned(),
            licence: "CC0".to_owned(),
            rms_dbfs_in: -20.0,
            gain_db: 0.0,
            trim_lead_s: 0.0,
            trim_tail_s: 0.0,
            sha256_16k: "a".to_owned(),
            sha256_8k: "b".to_owned(),
            prepared_at: "2026-09-10T00:00:00Z".to_owned(),
            parent: None,
            aug: None,
        };
        std::fs::write(
            prepared.join("manifest.jsonl"),
            format!("{}\n", serde_json::to_string(&row).unwrap()),
        )
        .unwrap();
        let cap = CaptureRow {
            key: key.to_owned(),
            mode,
            frames: u32::try_from(utt.deg8.len() / mode.frame_samples()).unwrap(),
            port: "/dev/sim".to_owned(),
            prodid: "SIM".to_owned(),
            version: "0".to_owned(),
            encode_ms: 1,
            decode_ms: 1,
            roundtrip_ms: None,
            sha256_ambe: "c".to_owned(),
            sha256_wav: "d".to_owned(),
            captured_at: "2026-09-10T00:00:00Z".to_owned(),
            attempts: 1,
            warm_state: None,
            aug: None,
        };
        std::fs::write(
            captured.join("manifest.jsonl"),
            format!("{}\n", serde_json::to_string(&cap).unwrap()),
        )
        .unwrap();
        let canary = CanaryRecord {
            mode,
            clip: "canary/x.wav".to_owned(),
            frames_sha256: "e".to_owned(),
            frames_first_16: "f".to_owned(),
            lag_samples: lag,
            prodid: "SIM".to_owned(),
            version: "0".to_owned(),
            recorded_at: "2026-09-10T00:00:00Z".to_owned(),
            warm_up_frames: 20,
        };
        std::fs::write(
            captured.join("canary.json"),
            serde_json::to_string(&canary).unwrap(),
        )
        .unwrap();
        utt.deg8.len()
    }

    #[test]
    fn infer_renders_a_whole_utterance_through_a_smoke_checkpoint() {
        let tmp = tempfile::tempdir().unwrap();
        let run = tmp.path().join("run");
        let outcome = crate::smoke(Some(&run)).unwrap();
        assert_eq!(outcome.step, 20);
        let root = tmp.path().join("data");
        let key = "synthetic/dev/utt_00";
        let len8 = synthetic_root(&root, key, VocoderMode::Dstar, 5);

        let out = run_one(&run, 20, key, &root).unwrap();
        assert_eq!(
            out,
            run.join("samples/step-000020/synthetic_dev_utt_00.out.wav")
        );
        let wav = unamblify_audio::read_wav(&out).unwrap();
        assert_eq!(wav.rate, 16_000);
        assert_eq!(wav.samples.len(), 2 * len8);
        assert!(wav.samples.iter().any(|&v| v != 0.0), "silent output");
        let spec: Spec = serde_json::from_str(
            &std::fs::read_to_string(
                run.join("samples/step-000020/synthetic_dev_utt_00.out.spec.json"),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(spec.frames, 1 + 2 * len8 / spec.hop);
        assert_eq!(spec.mats["out"].len(), spec.frames);

        // An explicit --out, a missing checkpoint, an unknown key.
        let custom = tmp.path().join("elsewhere/x.wav");
        let r = run_one_to(&run, 10, key, &root, Some(&custom)).unwrap();
        assert_eq!(r.out_wav, custom);
        assert_eq!(r.spec, tmp.path().join("elsewhere/x.spec.json"));
        assert_eq!(r.samples, 2 * len8);
        assert!(r.spec.is_file());
        assert!(run_one(&run, 15, key, &root).is_err());
        let err = run_one(&run, 20, "synthetic/dev/nope", &root)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not in both manifests"), "{err}");
        assert!(run_one(&run, 20, "../etc", &root).is_err());
        // A mode the run does not train on is refused; its own is fine.
        let err = run_one_in(&run, 20, key, Some(VocoderMode::YsfDmr), &root, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not one of the run's modes"), "{err}");
        let r = run_one_in(
            &run,
            20,
            key,
            Some(VocoderMode::Dstar),
            &root,
            Some(&custom),
        )
        .unwrap();
        assert_eq!(r.mode, VocoderMode::Dstar);
    }
}
