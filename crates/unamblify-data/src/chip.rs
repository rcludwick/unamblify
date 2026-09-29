// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! The AMBE-3000 control side (spec §3): per-mode init over any
//! `ambe_thumbdv::Transport`, a 300 ms `transact` with status checking, the
//! rate words for the three modes, the generic channel-in builder, and the
//! serial-port rules copied from astar — the FTDI VID/PID intersection that
//! refuses a `--port` the scan did not find, and the raw non-blocking open
//! that tells *busy* from *absent*.
//!
//! The vendored `ThumbDv` driver is D-STAR only and hides its transport, so
//! the pipelined runner ([`crate::pipeline`]) needs the transport back;
//! [`Chip`] therefore re-implements the nine init steps generically over
//! the mode and exposes `transport_mut()`.

use std::fmt;
use std::io;
use std::time::{Duration, Instant};

use ambe_thumbdv::packet::{
    dcmode_off, ecmode_off, gain_zero, init_encdec, parse_response, prodid_query, ratep_dstar,
    reset, verstring_query,
};
use ambe_thumbdv::{Deframer, Response, SerialTransport, Transport};
use unamblify::VocoderMode;

use crate::{DataError, Result};

/// The request → response deadline for one control transaction.
pub const TRANSACT_DEADLINE: Duration = Duration::from_millis(300);

/// How long each poll of the initial drain listens for stale bytes.
pub const DRAIN_TIMEOUT: Duration = Duration::from_millis(50);

/// How long the initial drain keeps reading in total. The interleaved
/// runner can leave up to `2 * INTERLEAVE_DEPTH` replies owed — about
/// 2 kB, more than one 1 kB read — so the drain loops until a poll comes
/// back empty rather than reading once; this bounds a babbling chip.
pub const DRAIN_DEADLINE: Duration = Duration::from_millis(500);

/// Baud rate of the ThumbDV / DVstick 30.
pub const BAUD: u32 = 460_800;

/// Frames of the canary the warm-up pushes through the encoder and
/// discards: the chip's first ~10 frames after init are rubbish (AMBE+2
/// pitch lock), so twice that.
pub const WARM_UP_FRAMES: usize = 20;

/// Control-protocol errors.
#[derive(Debug, thiserror::Error)]
pub enum ChipError {
    /// Transport I/O.
    #[error("io: {0}")]
    Io(#[from] io::Error),
    /// No complete response inside [`TRANSACT_DEADLINE`].
    #[error("timeout: no response to {0} within 300 ms")]
    Timeout(&'static str),
    /// The device is not an AMBE-3000.
    #[error("wrong device: PRODID {0:?} does not start with AMBE3000")]
    WrongDevice(String),
    /// A response that was not what the step expected.
    #[error("protocol: {0}")]
    Protocol(String),
    /// A status reply with a non-zero status byte.
    #[error("status: field 0x{field:02X} returned 0x{status:02X}")]
    Status {
        /// The field the status is for.
        field: u8,
        /// The non-zero status.
        status: u8,
    },
    /// A mode the chip does not produce (the Codec 2 modes run in software).
    #[error("{0} is not an AMBE-3000 mode; it is captured in software, not through the chip")]
    NotAChipMode(VocoderMode),
}

/// DVSI packet framing: start byte, big-endian length, packet type,
/// fields. The vendored crate keeps its builder private.
#[must_use]
pub fn dvsi_packet(ptype: u8, fields: &[u8]) -> Vec<u8> {
    let mut p = vec![0x61];
    let len = u16::try_from(fields.len()).unwrap_or(u16::MAX);
    p.extend_from_slice(&len.to_be_bytes());
    p.push(ptype);
    p.extend_from_slice(fields);
    p
}

/// RATEP for YSF DN (and NXDN): AMBE+2 2450 bit/s of voice, no FEC.
/// Wire bytes `61 00 0D 00 0A 04 31 07 54 00 00 00 00 00 00 70 31`.
#[must_use]
pub fn ratep_dn() -> Vec<u8> {
    dvsi_packet(
        0,
        &[
            0x0A, 0x04, 0x31, 0x07, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x31,
        ],
    )
}

/// RATEP for DMR: AMBE+2 2450 + 1150 bit/s of FEC, 72-bit frames.
/// Wire bytes `61 00 0D 00 0A 04 31 07 54 24 00 00 00 00 00 6F 48`.
///
/// Not a capture mode: DMR's 49 voice bits are YSF DN's (the `ysf-dmr`
/// capture serves both), and its Golay FEC only matters under bit
/// errors. The word is kept, pinned by a test, for the day the `ber`
/// augmentation synthesises DMR framing from the 49 bits and needs the
/// chip to decode it through its FEC.
#[must_use]
pub fn ratep_dmr() -> Vec<u8> {
    dvsi_packet(
        0,
        &[
            0x0A, 0x04, 0x31, 0x07, 0x54, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6F, 0x48,
        ],
    )
}

/// The RATEP packet for a chip mode; `None` for the software modes.
#[must_use]
pub fn ratep(mode: VocoderMode) -> Option<Vec<u8>> {
    match mode {
        VocoderMode::Dstar => Some(ratep_dstar()),
        VocoderMode::YsfDmr => Some(ratep_dn()),
        VocoderMode::Codec2_3200 | VocoderMode::Codec2_1600 => None,
    }
}

/// Generic channel-in packet: `0x61, BE(2 + len), 0x01, 0x01, bits, data`.
/// The bytes go on the wire exactly as given — for DN that is the chip's
/// own bit order, which is also what it emitted when encoding.
#[must_use]
pub fn channel_in_bits(bits: u8, data: &[u8]) -> Vec<u8> {
    let mut fields = Vec::with_capacity(2 + data.len());
    fields.push(0x01);
    fields.push(bits);
    fields.extend_from_slice(data);
    dvsi_packet(1, &fields)
}

/// The channel-in packet for one frame in `mode`: `mode.channel_bits()`
/// bits over `mode.frame_bytes()` bytes.
#[must_use]
pub fn channel_in_mode(mode: VocoderMode, frame: &[u8]) -> Vec<u8> {
    #[allow(clippy::cast_possible_truncation)]
    let bits = mode.channel_bits() as u8;
    channel_in_bits(bits, &frame[..mode.frame_bytes().min(frame.len())])
}

/// What the codec said about itself at init: the chip's PRODID and
/// VERSTRING, or `codec2` and the crate version for the software modes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChipInfo {
    /// PRODID reply (`AMBE3000F`), or `codec2`.
    pub prodid: String,
    /// VERSTRING reply, or the `codec2` crate version.
    pub version: String,
}

/// One initialised AMBE-3000 on a transport, configured for one mode.
pub struct Chip<T: Transport> {
    transport: T,
    mode: VocoderMode,
    info: ChipInfo,
    inits: u32,
}

impl<T: Transport> fmt::Debug for Chip<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Chip")
            .field("mode", &self.mode)
            .field("info", &self.info)
            .field("inits", &self.inits)
            .finish_non_exhaustive()
    }
}

impl<T: Transport> Chip<T> {
    /// Steps 1–9: drain 50 ms, `reset` → Ready, `prodid` starts with
    /// `AMBE3000`, `verstring`, RATEP for `mode`, `init_encdec`,
    /// `ecmode_off`, `dcmode_off`, `gain_zero`; each a 300 ms transact
    /// with a status check.
    pub fn init(transport: T, mode: VocoderMode) -> std::result::Result<Self, ChipError> {
        if ratep(mode).is_none() {
            return Err(ChipError::NotAChipMode(mode));
        }
        let mut chip = Self {
            transport,
            mode,
            info: ChipInfo {
                prodid: String::new(),
                version: String::new(),
            },
            inits: 0,
        };
        chip.reinit()?;
        Ok(chip)
    }

    /// The full init sequence again (steps 1–9), on the same transport.
    /// Used after any failure: reset, then everything.
    pub fn reinit(&mut self) -> std::result::Result<(), ChipError> {
        self.drain();

        match self.transact(&reset(), "reset")? {
            Response::Ready => {}
            other => {
                return Err(ChipError::Protocol(format!(
                    "expected Ready after reset, got {other:?}"
                )));
            }
        }
        let prodid = match self.transact(&prodid_query(), "prodid")? {
            Response::ProdId(s) => s,
            other => {
                return Err(ChipError::Protocol(format!(
                    "expected ProdId, got {other:?}"
                )));
            }
        };
        if !prodid.starts_with("AMBE3000") {
            return Err(ChipError::WrongDevice(prodid));
        }
        let version = match self.transact(&verstring_query(), "verstring")? {
            Response::Version(s) => s,
            other => {
                return Err(ChipError::Protocol(format!(
                    "expected Version, got {other:?}"
                )));
            }
        };
        let ratep = ratep(self.mode).ok_or(ChipError::NotAChipMode(self.mode))?;
        let r = self.transact(&ratep, "ratep")?;
        check_status(&r, 0x0A)?;
        let r = self.transact(&init_encdec(), "init_encdec")?;
        check_status(&r, 0x0B)?;
        let r = self.transact(&ecmode_off(), "ecmode_off")?;
        check_status(&r, 0x05)?;
        let r = self.transact(&dcmode_off(), "dcmode_off")?;
        check_status(&r, 0x06)?;
        let r = self.transact(&gain_zero(), "gain_zero")?;
        check_status(&r, 0x4B)?;
        self.info = ChipInfo { prodid, version };
        self.inits += 1;
        Ok(())
    }

    /// Read and discard whatever the chip still owes, until a poll comes
    /// back empty or [`DRAIN_DEADLINE`] passes. A failed utterance can
    /// leave several replies in flight, and any of them reaching the
    /// `reset` transact would look like a protocol error.
    fn drain(&mut self) {
        let deadline = Instant::now() + DRAIN_DEADLINE;
        let mut buf = [0u8; 1024];
        loop {
            match self.transport.recv_some(&mut buf, DRAIN_TIMEOUT) {
                Ok(0) | Err(_) => return,
                Ok(_) => {}
            }
            if Instant::now() >= deadline {
                return;
            }
        }
    }

    /// Send one request and wait up to [`TRANSACT_DEADLINE`] for one
    /// response. `what` names the step in the timeout error.
    pub fn transact(
        &mut self,
        req: &[u8],
        what: &'static str,
    ) -> std::result::Result<Response, ChipError> {
        transact(&mut self.transport, req, what)
    }

    /// PRODID / VERSTRING.
    #[must_use]
    pub fn info(&self) -> &ChipInfo {
        &self.info
    }

    /// The mode this chip was initialised for.
    #[must_use]
    pub fn mode(&self) -> VocoderMode {
        self.mode
    }

    /// How many times the init sequence has run (1 after `init`).
    #[must_use]
    pub fn inits(&self) -> u32 {
        self.inits
    }

    /// The transport, for the pipelined runner.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Give the transport back.
    pub fn into_transport(self) -> T {
        self.transport
    }
}

/// Send one request on `transport` and wait up to [`TRANSACT_DEADLINE`]
/// for one deframed response.
pub fn transact<T: Transport>(
    transport: &mut T,
    req: &[u8],
    what: &'static str,
) -> std::result::Result<Response, ChipError> {
    transport.send(req)?;
    let deadline = Instant::now() + TRANSACT_DEADLINE;
    let mut deframer = Deframer::new();
    let mut buf = [0u8; 1024];
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(ChipError::Timeout(what));
        }
        let n = transport.recv_some(&mut buf, remaining)?;
        if n > 0 {
            deframer.push(&buf[..n]);
            if let Some(pkt) = deframer.next_packet() {
                return parse_response(&pkt).map_err(|e| ChipError::Protocol(e.to_string()));
            }
        }
    }
}

/// A status reply must be for `expected_field` with status 0.
pub fn check_status(resp: &Response, expected_field: u8) -> std::result::Result<(), ChipError> {
    match resp {
        Response::Status { field, status } => {
            if *status != 0 {
                return Err(ChipError::Status {
                    field: *field,
                    status: *status,
                });
            }
            if *field != expected_field {
                return Err(ChipError::Protocol(format!(
                    "expected status for field 0x{expected_field:02X}, got 0x{field:02X}"
                )));
            }
            Ok(())
        }
        other => Err(ChipError::Protocol(format!(
            "expected Status for field 0x{expected_field:02X}, got {other:?}"
        ))),
    }
}

// ── Port selection (astar's rules, verbatim in spirit) ─────────────────

/// The refusal named in the error when a `--port` is not in the scan.
pub const PORT_RULE: &str = "FTDI 0x0403:0x6015 / \"ThumbDV\" scan";

/// Intersect the requested ports with the VID/PID scan, exactly like
/// astar's `thumbdv_candidate_ports_from`: the request NARROWS the scan
/// and never widens it. No request → every scanned port. A requested
/// path the scan did not find is refused with an error naming the rule
/// — pointing an opener at a USB radio interface's port would assert RTS
/// and key a transmitter.
pub fn candidate_ports_from(requested: &[String], scanned: &[String]) -> Result<Vec<String>> {
    if requested.is_empty() {
        return Ok(scanned.to_vec());
    }
    let mut out = Vec::with_capacity(requested.len());
    for p in requested {
        if !scanned.iter().any(|s| s == p) {
            return Err(DataError::PortRefused(format!(
                "--port {p} is not a ThumbDV: no serial port with that path matched the \
                 {PORT_RULE} ({scanned:?}). Refusing to open it — pointing this at a USB radio \
                 interface's port would assert RTS and key a transmitter."
            )));
        }
        if !out.contains(p) {
            out.push(p.clone());
        }
    }
    Ok(out)
}

/// The real scan, intersected with `requested`. Never widened.
pub fn select_ports(requested: &[String]) -> Result<Vec<String>> {
    let scanned = SerialTransport::candidate_ports();
    let ports = candidate_ports_from(requested, &scanned)?;
    if ports.is_empty() {
        return Err(DataError::PortAbsent);
    }
    Ok(ports)
}

/// What a trial open of a port found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortState {
    /// Opened (and was closed again immediately).
    Free,
    /// `EBUSY` / `EACCES`: something else has it.
    Busy {
        /// What `lsof` said about the holder, if it could.
        holder: Option<String>,
    },
    /// The path does not exist.
    Absent,
    /// Some other open failure.
    Other(String),
}

/// One raw `O_NONBLOCK | O_NOCTTY` open of `port`, closed at once
/// (astar's `classify_thumbdv_failure` / `trial_open`). Raw rather than
/// `SerialTransport::open` because `serialport` remaps the `TIOCEXCL`
/// `EBUSY` to its own `NoDevice`, which then reaches us as `Other`; std's
/// `OpenOptions` keeps `ResourceBusy` / `PermissionDenied` intact. The
/// non-blocking flags keep a dongle with no carrier from hanging the open.
#[must_use]
pub fn probe_port(port: &str) -> PortState {
    classify_open(port, trial_open(port))
}

/// The classification, with the open result injected for tests.
#[must_use]
pub fn classify_open(port: &str, result: io::Result<()>) -> PortState {
    match result {
        Ok(()) => PortState::Free,
        Err(e) => match e.kind() {
            io::ErrorKind::ResourceBusy | io::ErrorKind::PermissionDenied => PortState::Busy {
                holder: lsof_holder(port),
            },
            io::ErrorKind::NotFound => PortState::Absent,
            _ => PortState::Other(e.to_string()),
        },
    }
}

fn trial_open(port: &str) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Apple/BSD: O_NONBLOCK 0x0004, O_NOCTTY 0x20000.
        // Linux (asm-generic): O_NONBLOCK 0o4000, O_NOCTTY 0o400.
        #[cfg(any(target_vendor = "apple", target_os = "freebsd", target_os = "netbsd"))]
        const FLAGS: i32 = 0x0004 | 0x0002_0000;
        #[cfg(not(any(target_vendor = "apple", target_os = "freebsd", target_os = "netbsd")))]
        const FLAGS: i32 = 0o4000 | 0o400;
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(FLAGS)
            .open(port)
            .map(drop)
    }
    #[cfg(not(unix))]
    {
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(port)
            .map(drop)
    }
}

/// `lsof -t -- <port>` plus the command names, when `lsof` is available.
fn lsof_holder(port: &str) -> Option<String> {
    let out = std::process::Command::new("lsof")
        .args(["-F", "pc", "--", port])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let (mut pid, mut parts) = (None::<String>, Vec::new());
    for line in text.lines() {
        if let Some(p) = line.strip_prefix('p') {
            pid = Some(p.to_owned());
        } else if let Some(c) = line.strip_prefix('c')
            && let Some(p) = pid.take()
        {
            parts.push(format!("{c} (pid {p})"));
        }
    }
    (!parts.is_empty()).then(|| parts.join(", "))
}

/// Open `port` for capture: the busy probe first, then the serial port at
/// [`BAUD`].
pub fn open_serial(port: &str) -> Result<SerialTransport> {
    match probe_port(port) {
        PortState::Free => {}
        PortState::Busy { holder } => {
            return Err(DataError::PortBusy {
                port: port.to_owned(),
                holder: holder.map(|h| format!(" ({h})")).unwrap_or_default(),
            });
        }
        PortState::Absent => return Err(DataError::PortAbsent),
        PortState::Other(msg) => {
            return Err(DataError::Invalid(format!("{port}: cannot open: {msg}")));
        }
    }
    SerialTransport::open(port, BAUD).map_err(|e| DataError::io(port, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{hex, scripted_init};
    use ambe_thumbdv::MockTransport;

    #[test]
    fn rate_words_are_pinned_to_the_reference_bytes() {
        assert_eq!(
            ratep(VocoderMode::Dstar).unwrap(),
            hex("61 00 0D 00 0A 01 30 07 63 40 00 00 00 00 00 00 48")
        );
        assert_eq!(
            ratep(VocoderMode::YsfDmr).unwrap(),
            hex("61 00 0D 00 0A 04 31 07 54 00 00 00 00 00 00 70 31")
        );
        // DMR is no capture mode; its word stays pinned for the ber framing.
        assert_eq!(
            ratep_dmr(),
            hex("61 00 0D 00 0A 04 31 07 54 24 00 00 00 00 00 6F 48")
        );
        assert!(ratep(VocoderMode::Codec2_3200).is_none());
        assert!(ratep(VocoderMode::Codec2_1600).is_none());
    }

    #[test]
    fn a_software_mode_never_touches_the_transport() {
        // An empty mock: any request would fail it.
        let m = MockTransport::new();
        assert!(matches!(
            Chip::init(m, VocoderMode::Codec2_1600),
            Err(ChipError::NotAChipMode(VocoderMode::Codec2_1600))
        ));
    }

    #[test]
    fn channel_in_bits_matches_the_reference_layouts() {
        let dn = channel_in_bits(0x31, &[0xAA; 7]);
        assert_eq!(&dn[..6], &hex("61 00 09 01 01 31")[..]);
        assert_eq!(&dn[6..], &[0xAA; 7][..]);
        assert_eq!(dn.len(), 13);
        let full = channel_in_bits(0x48, &[0x11; 9]);
        assert_eq!(full, ambe_thumbdv::channel_in(&[0x11; 9]));
        assert_eq!(channel_in_mode(VocoderMode::YsfDmr, &[0xAA; 7]), dn);
        assert_eq!(channel_in_mode(VocoderMode::Dstar, &[0x11; 9]), full);
    }

    #[test]
    fn init_runs_the_nine_steps_for_every_chip_mode() {
        for mode in VocoderMode::ALL.into_iter().filter(|m| !m.is_software()) {
            let chip = Chip::init(scripted_init(mode), mode).unwrap();
            assert_eq!(chip.info().prodid, "AMBE3000F");
            assert_eq!(
                chip.info().version,
                "V121.E100.XXXX.C110.G514.R014.A0030608.C0020208"
            );
            assert_eq!(chip.mode(), mode);
            assert_eq!(chip.inits(), 1);
            assert!(chip.into_transport().done(), "{mode}: script not consumed");
        }
    }

    #[test]
    fn init_refuses_a_non_ambe3000_and_a_bad_status() {
        let mut m = MockTransport::new();
        m.expect(hex("61 00 01 00 33"), vec![hex("61 00 01 00 39")]);
        let mut prodid = hex("61 00 0B 00 30");
        prodid.extend_from_slice(b"AMBE2000X\0");
        m.expect(hex("61 00 01 00 30"), vec![prodid]);
        assert!(matches!(
            Chip::init(m, VocoderMode::Dstar),
            Err(ChipError::WrongDevice(s)) if s == "AMBE2000X"
        ));

        let mut m = MockTransport::new();
        m.expect(hex("61 00 01 00 33"), vec![hex("61 00 01 00 39")]);
        let mut prodid = hex("61 00 0B 00 30");
        prodid.extend_from_slice(b"AMBE3000F\0");
        m.expect(hex("61 00 01 00 30"), vec![prodid]);
        m.expect(hex("61 00 01 00 31"), vec![hex("61 00 03 00 31 56 00")]);
        m.expect(
            ratep(VocoderMode::YsfDmr).unwrap(),
            vec![hex("61 00 02 00 0A 05")],
        );
        assert!(matches!(
            Chip::init(m, VocoderMode::YsfDmr),
            Err(ChipError::Status {
                field: 0x0A,
                status: 5
            })
        ));
    }

    #[test]
    fn transact_times_out_on_silence() {
        let mut m = MockTransport::new();
        m.expect(hex("61 00 01 00 33"), vec![]);
        let t = Instant::now();
        assert!(matches!(
            Chip::init(m, VocoderMode::Dstar),
            Err(ChipError::Timeout("reset"))
        ));
        let dt = t.elapsed();
        assert!(dt >= Duration::from_millis(290), "{dt:?}");
        assert!(dt < Duration::from_millis(1500), "{dt:?}");
    }

    #[test]
    fn port_intersection_narrows_and_never_widens() {
        let scanned = vec![
            "/dev/cu.usbserial-A".to_owned(),
            "/dev/cu.usbserial-B".to_owned(),
        ];
        assert_eq!(candidate_ports_from(&[], &scanned).unwrap(), scanned);
        assert_eq!(
            candidate_ports_from(&["/dev/cu.usbserial-B".to_owned()], &scanned).unwrap(),
            vec!["/dev/cu.usbserial-B".to_owned()]
        );
        let err =
            candidate_ports_from(&["/dev/cu.usbmodem-RADIO".to_owned()], &scanned).unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, DataError::PortRefused(_)));
        assert!(msg.contains("is not a ThumbDV"), "{msg}");
        assert!(msg.contains(PORT_RULE), "{msg}");
        assert!(msg.contains("key a transmitter"), "{msg}");
        assert!(candidate_ports_from(&["/dev/x".to_owned()], &[]).is_err());
    }

    #[test]
    fn open_classification_reads_busy_absent_and_other() {
        assert_eq!(classify_open("/dev/none", Ok(())), PortState::Free);
        assert!(matches!(
            classify_open(
                "/dev/none",
                Err(io::Error::from(io::ErrorKind::ResourceBusy))
            ),
            PortState::Busy { .. }
        ));
        assert!(matches!(
            classify_open(
                "/dev/none",
                Err(io::Error::from(io::ErrorKind::PermissionDenied))
            ),
            PortState::Busy { .. }
        ));
        assert_eq!(
            classify_open("/dev/none", Err(io::Error::from(io::ErrorKind::NotFound))),
            PortState::Absent
        );
        assert!(matches!(
            classify_open("/dev/none", Err(io::Error::other("x"))),
            PortState::Other(_)
        ));
        // A path that does not exist is `Absent` through the real probe;
        // no serial device is ever opened by this test.
        assert_eq!(
            probe_port("/nonexistent/unamblify-no-such-port"),
            PortState::Absent
        );
    }
}
