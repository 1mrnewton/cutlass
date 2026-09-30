//! Audio playback: device output, timeline mixing, and the playback master
//! clock (Phase 3 of the desktop port).
//!
//! Three players, all decoupled by lock-free state:
//!
//! - The **UI thread** owns transport intent: play/seek bump an *epoch* and
//!   reset the shared clock atomically, then notify the mixer. It also reads
//!   the clock (`AudioHandle::current_tick`) from the playback timer.
//! - The **mixer thread** owns an [`ExportAudioMixer`] over a project
//!   snapshot — the same streamed mixer export uses, so what you hear is
//!   what you ship. It pulls interleaved stereo blocks, tagged with the
//!   epoch they were mixed for, through a bounded channel sized to ~100ms.
//! - The **device callback** pops blocks, drops any with a stale epoch
//!   (seek flush without locks), spreads stereo onto the device's channel
//!   count, and advances `frames_played` — *consumed frames are the clock*,
//!   so A/V sync is device-true regardless of buffer depth.
//!
//! Edits never touch a live mixer (it owns a snapshot): the engine worker
//! publishes a fresh project clone with every projection republish, and the
//! mixer reopens at its current position — the mobile reader semantics
//! (reopen on seek and on revision change) instead of main's per-span reader
//! surgery. Retimed clips are varispeed-resampled and denoise-flagged clips
//! run through RNNoise in the shared [`ExportAudioMixer`]; reversed clips
//! still play silent (forward-only decoders).
//!
//! The clock counts only real consumed frames: an underrun stalls the
//! playhead briefly instead of letting video run away from audio. Silence
//! (no clips under the playhead) still produces zero-blocks, so the audio
//! device paces silent timelines too; the wall-clock transport is only a
//! fallback for machines with no output device (and for shuttle speeds,
//! which play muted — see `transport.rs`).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use cutlass_models::{ClipId, ClipParam, ParamValue, Project};
use cutlass_render::{EXPORT_AUDIO_RATE, ExportAudioMixer, ParamOverrides};
use tracing::{info, warn};

/// Interleaved channel count of every mixed block (the export mixer's).
const CHANNELS: usize = cutlass_render::EXPORT_AUDIO_CHANNELS as usize;
/// Frames per mixed block. 1024 @ 48kHz ≈ 21ms.
const BLOCK_FRAMES: usize = 1024;
/// Blocks in flight device-ward. 6 × 21ms ≈ 128ms of buffered audio — also
/// the worst-case latency for a mid-playback edit to become audible.
const BLOCK_CAPACITY: usize = 6;
/// Blocks one scrub gesture emits before the mixer goes idle. 4 × 21ms ≈
/// 85ms — long enough to *hear* the content under the playhead, short enough
/// that the next drag step retriggers a fresh burst (CapCut's scrub-audio
/// feel). Each step bumps the epoch, so the callback drops the previous
/// burst's tail and the newest position wins.
const SCRUB_BURST_BLOCKS: usize = 4;

enum AudioMsg {
    /// A fresh project snapshot (published on every projection republish).
    /// The mixer reopens over it at its current position, so mid-playback
    /// edits become audible within the buffered ~100ms.
    Snapshot(Box<Project>),
    /// Start (or re-anchor) producing from `tick`. `epoch` was assigned by
    /// the UI thread *before* send; blocks tagged older are already dead.
    Play {
        tick: i64,
        epoch: u64,
    },
    Pause,
    /// Emit one short audio burst from `tick` (scrub audio), then idle.
    /// `epoch` (assigned UI-side) supersedes any earlier play/scrub burst so
    /// only the newest scrub position is heard.
    Scrub {
        tick: i64,
        epoch: u64,
    },
    /// Live volume/pan override during an inspector drag. Applied on the next
    /// mix block without reopening source readers (unlike [`Snapshot`]).
    ParamOverride {
        clip: ClipId,
        param: ClipParam,
        value: ParamValue,
    },
    /// Drop one live volume/pan override after that param is committed.
    ClearParamOverride {
        clip: ClipId,
        param: ClipParam,
    },
    /// Drop every live param override for `clip` (release with no net change).
    ClearParamOverrides {
        clip: ClipId,
    },
    /// Session replacement (new / open / template): drop every mirrored
    /// volume/pan override. Clip ids are project-local, so leftovers must
    /// not survive onto the next project's mixer.
    ClearAllParamOverrides,
}

/// One mixed block on its way to the device.
struct AudioBlock {
    epoch: u64,
    /// Interleaved stereo, `BLOCK_FRAMES * CHANNELS` long.
    samples: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Shared clock state
// ---------------------------------------------------------------------------

struct AudioShared {
    /// Bumped by every play/seek/scrub; the callback discards stale-tagged
    /// blocks (the lock-free flush).
    epoch: AtomicU64,
    /// Timeline tick at the current epoch's origin.
    anchor_tick: AtomicI64,
    /// Device frames of the current epoch consumed by the callback.
    frames_played: AtomicU64,
    playing: AtomicBool,
    /// Scrub-audio gate: the callback plays queued bursts while set, but —
    /// unlike `playing` — never advances `frames_played`, so a scrub burst
    /// is heard without moving the master clock (the playhead is driven by
    /// the drag, not the audio).
    scrubbing: AtomicBool,
    underruns: AtomicU64,
}

impl AudioShared {
    fn new() -> Self {
        Self {
            epoch: AtomicU64::new(0),
            anchor_tick: AtomicI64::new(0),
            frames_played: AtomicU64::new(0),
            playing: AtomicBool::new(false),
            scrubbing: AtomicBool::new(false),
            underruns: AtomicU64::new(0),
        }
    }
}

/// Cloneable, `Send` interface to the audio system: transport control for
/// the UI thread, snapshot publishing for the engine worker, and the master
/// clock for the playback timer.
#[derive(Clone)]
pub struct AudioHandle {
    shared: Arc<AudioShared>,
    tx: Option<Sender<AudioMsg>>,
    sample_rate: u32,
}

impl AudioHandle {
    /// Whether a real output device is pacing playback. False ⇒ the UI must
    /// fall back to the wall-clock transport.
    pub fn active(&self) -> bool {
        self.tx.is_some()
    }

    /// Start playing from `tick` (also the mid-playback seek: every seek is
    /// a re-anchored play). Clock state flips on the caller's thread, so a
    /// `current_tick` immediately after returns `tick` — no mixer round-trip.
    pub fn play(&self, tick: i64) {
        let epoch = self.shared.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        self.shared.anchor_tick.store(tick, Ordering::Release);
        self.shared.frames_played.store(0, Ordering::Release);
        self.shared.scrubbing.store(false, Ordering::Release);
        self.shared.playing.store(true, Ordering::Release);
        if let Some(tx) = &self.tx {
            let _ = tx.send(AudioMsg::Play { tick, epoch });
        }
    }

    pub fn pause(&self) {
        self.shared.playing.store(false, Ordering::Release);
        self.shared.scrubbing.store(false, Ordering::Release);
        if let Some(tx) = &self.tx {
            let _ = tx.send(AudioMsg::Pause);
        }
    }

    /// Play a short audio burst from `tick`: the UI calls this as the
    /// playhead is dragged while paused, so the user hears the content under
    /// the playhead. Bumps the epoch so the burst supersedes the prior one;
    /// never touches the master clock (a scrub must not move the playhead,
    /// the drag already does). No-op without an output device.
    pub fn scrub(&self, tick: i64) {
        if self.tx.is_none() {
            return;
        }
        let epoch = self.shared.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        self.shared.playing.store(false, Ordering::Release);
        self.shared.scrubbing.store(true, Ordering::Release);
        if let Some(tx) = &self.tx {
            let _ = tx.send(AudioMsg::Scrub { tick, epoch });
        }
    }

    /// Publish a fresh project snapshot (call on every projection republish).
    pub fn publish_snapshot(&self, project: Project) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(AudioMsg::Snapshot(Box::new(project)));
        }
    }

    /// Live volume/pan preview during an inspector drag. No-ops for other
    /// params and when audio output is unavailable.
    pub fn set_param_override(&self, clip: ClipId, param: ClipParam, value: ParamValue) {
        if !matches!(param, ClipParam::Volume | ClipParam::Pan) {
            return;
        }
        if let Some(tx) = &self.tx {
            let _ = tx.send(AudioMsg::ParamOverride { clip, param, value });
        }
    }

    /// Drop one live volume/pan override after that param is committed.
    pub fn clear_param_override(&self, clip: ClipId, param: ClipParam) {
        if !matches!(param, ClipParam::Volume | ClipParam::Pan) {
            return;
        }
        if let Some(tx) = &self.tx {
            let _ = tx.send(AudioMsg::ClearParamOverride { clip, param });
        }
    }

    /// Drop every live param override for `clip` (release with no net change).
    pub fn clear_param_overrides(&self, clip: ClipId) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(AudioMsg::ClearParamOverrides { clip });
        }
    }

    /// Drop every mirrored volume/pan override (session replacement).
    pub fn clear_all_param_overrides(&self) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(AudioMsg::ClearAllParamOverrides);
        }
    }

    /// Playhead tick by the audio clock: anchor + consumed device frames at
    /// the sequence rate (exact i128, floored).
    pub fn current_tick(&self, fps_num: i32, fps_den: i32) -> i64 {
        let anchor = self.shared.anchor_tick.load(Ordering::Acquire);
        if fps_num <= 0 || fps_den <= 0 || self.sample_rate == 0 {
            return anchor;
        }
        let frames = self.shared.frames_played.load(Ordering::Acquire);
        let ticks = i128::from(frames) * i128::from(fps_num)
            / (i128::from(self.sample_rate) * i128::from(fps_den));
        (i128::from(anchor) + ticks).clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
    }
}

// ---------------------------------------------------------------------------
// System: device stream + mixer thread
// ---------------------------------------------------------------------------

/// Owns the cpal stream (`!Send` — lives on the main thread) and the mixer
/// thread. Dropping it stops audio; hand out [`AudioHandle`]s freely.
pub struct AudioSystem {
    handle: AudioHandle,
    _stream: Option<cpal::Stream>,
}

impl AudioSystem {
    /// Bring up the default output device at the mixer's 48 kHz. A machine
    /// without one (or without 48 kHz f32 support) degrades to a disabled
    /// system whose handle reports `active() == false`.
    pub fn start() -> Self {
        match Self::try_start() {
            Ok(system) => system,
            Err(e) => {
                warn!("audio output unavailable, playback clock falls back to wall time: {e}");
                Self {
                    handle: AudioHandle {
                        shared: Arc::new(AudioShared::new()),
                        tx: None,
                        sample_rate: 0,
                    },
                    _stream: None,
                }
            }
        }
    }

    pub fn handle(&self) -> AudioHandle {
        self.handle.clone()
    }

    fn try_start() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or("no default output device")?;
        // The stream must run at the export mixer's fixed rate — the OS
        // resamples to hardware rate underneath (CoreAudio does this
        // transparently). Prefer the default config when it already matches.
        let default_config = device.default_output_config().map_err(|e| e.to_string())?;
        let config = if default_config.sample_format() == cpal::SampleFormat::F32
            && default_config.sample_rate() == EXPORT_AUDIO_RATE
        {
            default_config
        } else {
            device
                .supported_output_configs()
                .map_err(|e| e.to_string())?
                .filter(|range| range.sample_format() == cpal::SampleFormat::F32)
                .find_map(|range| range.try_with_sample_rate(EXPORT_AUDIO_RATE))
                .ok_or(format!("no f32 output config at {EXPORT_AUDIO_RATE} Hz"))?
        };
        let stream_config: cpal::StreamConfig = config.into();
        let sample_rate = stream_config.sample_rate;
        let device_channels = stream_config.channels as usize;

        let shared = Arc::new(AudioShared::new());
        let (msg_tx, msg_rx) = unbounded::<AudioMsg>();
        let (block_tx, block_rx) = bounded::<AudioBlock>(BLOCK_CAPACITY);

        let cb_shared = Arc::clone(&shared);
        let mut sink = CallbackSink::new(block_rx);
        let stream = device
            .build_output_stream(
                stream_config,
                move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    sink.fill(out, device_channels, &cb_shared);
                },
                |e| warn!("audio stream error: {e}"),
                None,
            )
            .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;

        std::thread::Builder::new()
            .name("cutlass-audio-mixer".into())
            .spawn(move || mixer_loop(msg_rx, block_tx))
            .map_err(|e| e.to_string())?;

        info!(sample_rate, device_channels, "audio output ready");
        Ok(Self {
            handle: AudioHandle {
                shared,
                tx: Some(msg_tx),
                sample_rate,
            },
            _stream: Some(stream),
        })
    }
}

// ---------------------------------------------------------------------------
// Device callback
// ---------------------------------------------------------------------------

/// Callback-side state: the block being consumed plus its read cursor.
struct CallbackSink {
    rx: Receiver<AudioBlock>,
    current: Option<AudioBlock>,
    cursor: usize,
}

impl CallbackSink {
    fn new(rx: Receiver<AudioBlock>) -> Self {
        Self {
            rx,
            current: None,
            cursor: 0,
        }
    }

    /// Real-time path: no locks, no blocking waits; allocation only when a
    /// stale block is dropped (acceptable for a desktop editor's callback).
    fn fill(&mut self, out: &mut [f32], device_channels: usize, shared: &AudioShared) {
        out.fill(0.0);
        let playing = shared.playing.load(Ordering::Acquire);
        // Scrub bursts play through the same ring but don't drive the clock —
        // only real playback advances `frames_played`.
        let scrubbing = !playing && shared.scrubbing.load(Ordering::Acquire);
        if !playing && !scrubbing {
            self.current = None;
            return;
        }
        let epoch = shared.epoch.load(Ordering::Acquire);
        let want_frames = out.len() / device_channels.max(1);
        let mut filled = 0usize;

        while filled < want_frames {
            let block = match &self.current {
                Some(b) if b.epoch == epoch && self.cursor < b.samples.len() => {
                    self.current.as_ref().expect("just matched")
                }
                _ => {
                    self.current = None;
                    match self.rx.try_recv() {
                        Ok(b) if b.epoch == epoch => {
                            self.cursor = 0;
                            self.current.insert(b)
                        }
                        Ok(_stale) => continue, // pre-seek block: drop, next
                        Err(_) => break,        // ring empty: underrun
                    }
                }
            };

            let frames_left = (block.samples.len() - self.cursor) / CHANNELS;
            let take = frames_left.min(want_frames - filled);
            for i in 0..take {
                let src = self.cursor + i * CHANNELS;
                let (l, r) = (block.samples[src], block.samples[src + 1]);
                let dst = (filled + i) * device_channels;
                match device_channels {
                    0 => {}
                    1 => out[dst] = 0.5 * (l + r),
                    _ => {
                        out[dst] = l;
                        out[dst + 1] = r;
                        // extra channels stay silent
                    }
                }
            }
            self.cursor += take * CHANNELS;
            filled += take;
        }

        // Only real playback owns the clock; a scrub burst draining to silence
        // is expected, not an underrun.
        if playing {
            if filled > 0 {
                shared
                    .frames_played
                    .fetch_add(filled as u64, Ordering::AcqRel);
            }
            if filled < want_frames {
                shared.underruns.fetch_add(1, Ordering::AcqRel);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mixer thread
// ---------------------------------------------------------------------------

fn mixer_loop(msg_rx: Receiver<AudioMsg>, block_tx: Sender<AudioBlock>) {
    // The latest project snapshot; the mixer below is always derived from it.
    let mut project: Option<Project> = None;
    // Streamed mixer over the snapshot. `None` while the timeline is silent
    // or after a decode failure (silence until the next reopen).
    let mut mixer: Option<ExportAudioMixer> = None;
    // Session-only volume/pan overrides survive mixer reopen (snapshot /
    // seek) so a mid-drag project republish doesn't drop the live gain.
    let mut param_overrides = ParamOverrides::new();
    let mut fps = (0i32, 0i32);
    let mut epoch = 0u64;
    let mut playing = false;
    // Scrub-burst state: how many burst blocks remain to emit before the
    // mixer idles. Reset on each `Scrub`, zeroed by play/pause.
    let mut scrub_blocks_left = 0usize;
    // Timeline sample-frame position the next block mixes from.
    let mut write_frame = 0i64;

    loop {
        // Heartbeat doubles as the pacing valve: when the block channel is
        // full we sleep here instead of spinning.
        let mut reopen = false;
        match msg_rx.recv_timeout(Duration::from_millis(4)) {
            Ok(msg) => {
                let mut latest = Some(msg);
                while let Some(msg) = latest.take() {
                    match msg {
                        AudioMsg::Snapshot(snapshot) => {
                            let new_fps = {
                                let rate = snapshot.timeline().frame_rate;
                                (rate.num, rate.den)
                            };
                            if playing && fps != new_fps && fps.0 > 0 {
                                // Rate change mid-play (project swap): keep
                                // the time, not the frame count.
                                let tick = frames_to_ticks(write_frame, fps);
                                write_frame = ticks_to_frames(tick, new_fps);
                            }
                            fps = new_fps;
                            project = Some(*snapshot);
                            reopen = true;
                        }
                        AudioMsg::Play { tick, epoch: e } => {
                            epoch = e;
                            playing = true;
                            scrub_blocks_left = 0;
                            write_frame = ticks_to_frames(tick, fps);
                            reopen = true;
                        }
                        AudioMsg::Pause => {
                            playing = false;
                            scrub_blocks_left = 0;
                        }
                        AudioMsg::Scrub { tick, epoch: e } => {
                            epoch = e;
                            playing = false;
                            scrub_blocks_left = SCRUB_BURST_BLOCKS;
                            write_frame = ticks_to_frames(tick, fps);
                            reopen = true;
                        }
                        AudioMsg::ParamOverride { clip, param, value } => {
                            param_overrides.set(clip, param, value);
                            if let Some(m) = &mut mixer {
                                m.set_param_override(clip, param, value);
                            }
                        }
                        AudioMsg::ClearParamOverride { clip, param } => {
                            param_overrides.clear_param(clip, param);
                            if let Some(m) = &mut mixer {
                                m.clear_param_override(clip, param);
                            }
                        }
                        AudioMsg::ClearParamOverrides { clip } => {
                            param_overrides.clear_clip(clip);
                            if let Some(m) = &mut mixer {
                                m.clear_param_overrides(clip);
                            }
                        }
                        AudioMsg::ClearAllParamOverrides => {
                            param_overrides.clear();
                            if let Some(m) = &mut mixer {
                                m.set_param_overrides(ParamOverrides::new());
                            }
                        }
                    }
                    latest = msg_rx.try_recv().ok();
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return,
        }

        // Reopen on every snapshot and seek (the mobile reader semantics): a
        // fresh mixer re-derives its spans from the snapshot and re-seeks
        // its source readers lazily on the next mix. Cheap — readers open on
        // first overlap only. Re-apply live volume/pan overrides so a drag
        // survives the reopen.
        if reopen {
            mixer = project.as_ref().and_then(ExportAudioMixer::for_project);
            if let Some(m) = &mut mixer {
                m.set_param_overrides(param_overrides.clone());
            }
        }

        // Idle unless playing or mid-scrub-burst.
        if !playing && scrub_blocks_left == 0 {
            continue;
        }

        // Fill whatever room the device side has left, but never starve the
        // message queue: a pending play/seek/scrub/snapshot outranks the next
        // block. A scrub burst stops after its fixed block count (then idles).
        while !block_tx.is_full() && msg_rx.is_empty() {
            if !playing && scrub_blocks_left == 0 {
                break;
            }
            let mut samples = vec![0f32; BLOCK_FRAMES * CHANNELS];
            if let Some(m) = &mut mixer
                && let Err(e) = m.mix_into(write_frame, &mut samples)
            {
                // Fail-soft in preview (export is the fail-loud path):
                // log, go silent, and let the next snapshot/seek retry.
                warn!("audio mix failed, muting until the next edit/seek: {e}");
                mixer = None;
                samples.fill(0.0);
            }
            if block_tx.send(AudioBlock { epoch, samples }).is_err() {
                return; // device side gone
            }
            write_frame += BLOCK_FRAMES as i64;
            if !playing {
                scrub_blocks_left -= 1;
            }
        }
    }
}

/// `value` ticks at `rate` fps → sample frames at the mixer rate (exact
/// i128, floored).
fn ticks_to_frames(value: i64, rate: (i32, i32)) -> i64 {
    let (num, den) = rate;
    if num <= 0 || den <= 0 {
        return 0;
    }
    let frames =
        i128::from(value) * i128::from(den) * i128::from(EXPORT_AUDIO_RATE) / i128::from(num);
    frames.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

/// Inverse of [`ticks_to_frames`] (floored).
fn frames_to_ticks(frames: i64, rate: (i32, i32)) -> i64 {
    let (num, den) = rate;
    if num <= 0 || den <= 0 {
        return 0;
    }
    let ticks =
        i128::from(frames) * i128::from(num) / (i128::from(den) * i128::from(EXPORT_AUDIO_RATE));
    ticks.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use cutlass_models::{Clip, MediaSource, Rational, TimeRange, TrackKind};

    #[test]
    fn tick_frame_conversion_is_exact_for_integer_rates() {
        // 24 ticks (1s at 24fps) = 48000 frames at 48kHz.
        assert_eq!(ticks_to_frames(24, (24, 1)), 48_000);
        assert_eq!(frames_to_ticks(48_000, (24, 1)), 24);
        // One tick = 2000 frames.
        assert_eq!(ticks_to_frames(1, (24, 1)), 2_000);
    }

    #[test]
    fn tick_frame_conversion_handles_ntsc() {
        // 30000 ticks at 30000/1001 fps = 1001 seconds = 1001 · 48000 frames.
        assert_eq!(ticks_to_frames(30_000, (30_000, 1_001)), 1_001 * 48_000);
        assert_eq!(frames_to_ticks(1_001 * 48_000, (30_000, 1_001)), 30_000);
    }

    #[test]
    fn invalid_rates_collapse_to_zero() {
        assert_eq!(ticks_to_frames(100, (0, 1)), 0);
        assert_eq!(frames_to_ticks(100, (24, 0)), 0);
    }

    #[test]
    fn clock_reports_anchor_plus_consumed_frames() {
        let handle = AudioHandle {
            shared: Arc::new(AudioShared::new()),
            tx: None,
            sample_rate: 48_000,
        };
        handle.play(100);
        assert_eq!(handle.current_tick(24, 1), 100);
        // 1.5s of consumed audio = 36 ticks at 24fps.
        handle.shared.frames_played.store(72_000, Ordering::Release);
        assert_eq!(handle.current_tick(24, 1), 136);
    }

    #[test]
    fn scrub_burst_plays_without_advancing_the_clock() {
        // A scrub-tagged block is audible through the callback, but a scrub
        // never advances `frames_played` (the drag drives the playhead, not
        // the audio clock).
        let shared = AudioShared::new();
        shared.scrubbing.store(true, Ordering::Release);
        let (tx, rx) = bounded::<AudioBlock>(BLOCK_CAPACITY);
        tx.send(AudioBlock {
            epoch: 0, // matches the fresh AudioShared epoch
            samples: vec![0.5; BLOCK_FRAMES * CHANNELS],
        })
        .unwrap();

        let mut sink = CallbackSink::new(rx);
        let mut out = vec![0f32; 256 * 2];
        sink.fill(&mut out, 2, &shared);
        assert!(out.iter().any(|&s| s != 0.0), "scrub burst is audible");
        assert_eq!(
            shared.frames_played.load(Ordering::Acquire),
            0,
            "a scrub burst must not move the master clock"
        );
    }

    #[test]
    fn callback_drops_stale_epoch_blocks() {
        // A seek bumps the epoch UI-side; blocks mixed for the old epoch
        // must be discarded, not played (the lock-free flush).
        let shared = AudioShared::new();
        shared.playing.store(true, Ordering::Release);
        shared.epoch.store(5, Ordering::Release);
        let (tx, rx) = bounded::<AudioBlock>(BLOCK_CAPACITY);
        tx.send(AudioBlock {
            epoch: 4,
            samples: vec![0.9; BLOCK_FRAMES * CHANNELS],
        })
        .unwrap();
        tx.send(AudioBlock {
            epoch: 5,
            samples: vec![0.25; BLOCK_FRAMES * CHANNELS],
        })
        .unwrap();

        let mut sink = CallbackSink::new(rx);
        let mut out = vec![0f32; 128 * 2];
        sink.fill(&mut out, 2, &shared);
        assert!(
            out.iter().all(|&s| s == 0.25),
            "only current-epoch samples reach the device"
        );
    }

    /// A media asset with an audio stream, if the local asset pool has one
    /// (same optional-fixture convention as the engine's tests).
    fn audio_asset() -> Option<PathBuf> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../local-assets/assets");
        std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| {
                p.extension().is_some_and(|e| e == "mp3" || e == "m4a")
                    || (p.extension().is_some_and(|e| e == "mp4")
                        && cutlass_decoder::open_audio_reader(p, 48_000, 2).is_ok())
            })
    }

    /// One audio clip over `path`, timeline ticks `[0, 48)` at 24fps.
    fn project_with_clip(path: &std::path::Path) -> (Project, ClipId) {
        let mut project = Project::new("audio-test", Rational::FPS_24);
        let media = project.add_media(MediaSource::new(path, 0, 0, Rational::FPS_24, 480, true));
        let track = project.add_track(TrackKind::Audio, "A1");
        let range = |start: i64, dur: i64| TimeRange::at_rate(start, dur, Rational::FPS_24);
        let clip = Clip::from_media(media, range(0, 48), range(0, 48));
        let clip_id = clip.id;
        project
            .timeline_mut()
            .add_clip(track, clip)
            .expect("clip placement is valid");
        (project, clip_id)
    }

    #[test]
    fn mixer_thread_emits_a_finite_scrub_burst_then_idles() {
        let Some(path) = audio_asset() else {
            return;
        };
        let (msg_tx, msg_rx) = unbounded::<AudioMsg>();
        let (block_tx, block_rx) = bounded::<AudioBlock>(BLOCK_CAPACITY);
        let mixer = std::thread::spawn(move || mixer_loop(msg_rx, block_tx));

        // A clip over the scrub position so the burst carries real audio.
        let (project, _) = project_with_clip(&path);
        msg_tx.send(AudioMsg::Snapshot(Box::new(project))).unwrap();
        msg_tx.send(AudioMsg::Scrub { tick: 0, epoch: 7 }).unwrap();

        // Exactly SCRUB_BURST_BLOCKS blocks, all tagged with the scrub epoch.
        let mut heard = false;
        for _ in 0..SCRUB_BURST_BLOCKS {
            let block = block_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("scrub burst produces");
            assert_eq!(block.epoch, 7);
            heard |= block.samples.iter().any(|&s| s != 0.0);
        }
        assert!(heard, "scrub burst is audible over the clip");

        // Then the mixer idles: no further blocks without a new message.
        assert!(
            block_rx.recv_timeout(Duration::from_millis(200)).is_err(),
            "the burst is finite — the mixer stops producing after it"
        );

        drop(msg_tx);
        while block_rx.try_recv().is_ok() {}
        mixer.join().expect("mixer thread exits cleanly");
    }

    #[test]
    fn session_reset_drops_mirrored_param_overrides() {
        // A volume=0 override must not survive ClearAllParamOverrides across a
        // Snapshot reopen (new/open/template) — otherwise the next project's
        // clip with the same raw id stays muted.
        let Some(path) = audio_asset() else {
            return;
        };
        let (msg_tx, msg_rx) = unbounded::<AudioMsg>();
        let (block_tx, block_rx) = bounded::<AudioBlock>(BLOCK_CAPACITY);
        let mixer = std::thread::spawn(move || mixer_loop(msg_rx, block_tx));

        let (project, clip) = project_with_clip(&path);
        msg_tx
            .send(AudioMsg::Snapshot(Box::new(project.clone())))
            .unwrap();
        msg_tx
            .send(AudioMsg::ParamOverride {
                clip,
                param: ClipParam::Volume,
                value: ParamValue::Scalar(0.0),
            })
            .unwrap();
        msg_tx.send(AudioMsg::Scrub { tick: 0, epoch: 1 }).unwrap();

        for _ in 0..SCRUB_BURST_BLOCKS {
            let block = block_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("muted scrub burst");
            assert!(
                block.samples.iter().all(|&s| s == 0.0),
                "volume override 0 must silence the burst"
            );
        }

        // Session replacement: clear the mirror, then reopen over a snapshot.
        msg_tx.send(AudioMsg::ClearAllParamOverrides).unwrap();
        msg_tx.send(AudioMsg::Snapshot(Box::new(project))).unwrap();
        msg_tx.send(AudioMsg::Scrub { tick: 0, epoch: 2 }).unwrap();

        let mut heard = false;
        for _ in 0..SCRUB_BURST_BLOCKS {
            let block = block_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("post-reset scrub burst");
            assert_eq!(block.epoch, 2);
            heard |= block.samples.iter().any(|&s| s != 0.0);
        }
        assert!(
            heard,
            "after ClearAllParamOverrides a Snapshot reopen must not re-mute"
        );

        drop(msg_tx);
        while block_rx.try_recv().is_ok() {}
        mixer.join().expect("mixer thread exits cleanly");
    }

    #[test]
    fn mixer_thread_keeps_pacing_a_silent_timeline() {
        // No audible clips ⇒ no ExportAudioMixer, but playback must still
        // emit zero-blocks so the device keeps consuming and the master
        // clock advances over silence.
        let (msg_tx, msg_rx) = unbounded::<AudioMsg>();
        let (block_tx, block_rx) = bounded::<AudioBlock>(BLOCK_CAPACITY);
        let mixer = std::thread::spawn(move || mixer_loop(msg_rx, block_tx));

        msg_tx
            .send(AudioMsg::Snapshot(Box::new(Project::new(
                "silent",
                Rational::FPS_24,
            ))))
            .unwrap();
        msg_tx.send(AudioMsg::Play { tick: 0, epoch: 3 }).unwrap();

        let block = block_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("silent playback still produces blocks");
        assert_eq!(block.epoch, 3);
        assert!(block.samples.iter().all(|&s| s == 0.0));

        drop(msg_tx);
        while block_rx.try_recv().is_ok() {}
        mixer.join().expect("mixer thread exits cleanly");
    }
}
