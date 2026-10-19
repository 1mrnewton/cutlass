//! Apple audio decode: `AVAssetReader` over the audio track, delivering
//! interleaved `f32` resampled to a caller-chosen rate / channel count.
//!
//! AVFoundation does the heavy lifting — request LPCM float output and it
//! demuxes, decodes (VideoToolbox/AudioToolbox), resamples, and down/up-mixes
//! to the asked-for rate and channels. We pull `CMSampleBuffer`s, copy their
//! `CMBlockBuffer` bytes as `f32`, and hand them out behind
//! [`cutlass_core::AudioReader`]. Seeking rebuilds the reader at a new start
//! time (forward-only, same as the video backend).

use std::path::Path;
use std::ptr::NonNull;

use objc2::AllocAnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_av_foundation::{
    AVAssetReader, AVAssetReaderStatus, AVAssetReaderTrackOutput, AVAssetTrack, AVMediaTypeAudio,
    AVURLAsset,
};
use objc2_core_media::{CMTime, CMTimeFlags, CMTimeRange, kCMTimePositiveInfinity};
use objc2_foundation::{NSMutableDictionary, NSNumber, NSString, NSURL};

use cutlass_core::{AudioReader, DecodeError, Rational, RationalTime};

use crate::apple::rational_to_cmtime;

// CoreAudio `AudioFormatID` FourCC for uncompressed linear PCM (`'lpcm'`).
const K_AUDIO_FORMAT_LINEAR_PCM: u32 = u32::from_be_bytes(*b"lpcm");

// AVFoundation audio-settings dictionary keys. The pinned `objc2-av-foundation`
// 0.3.2 ships an empty `AVAudioSettings` binding (the header didn't translate),
// so we link the framework's exported `NSString *` constants directly — the
// same shape the crate uses for the video-settings keys.
#[link(name = "AVFoundation", kind = "framework")]
unsafe extern "C" {
    static AVFormatIDKey: Option<&'static NSString>;
    static AVSampleRateKey: Option<&'static NSString>;
    static AVNumberOfChannelsKey: Option<&'static NSString>;
    static AVLinearPCMBitDepthKey: Option<&'static NSString>;
    static AVLinearPCMIsFloatKey: Option<&'static NSString>;
    static AVLinearPCMIsBigEndianKey: Option<&'static NSString>;
    static AVLinearPCMIsNonInterleaved: Option<&'static NSString>;
}

/// An audio source decoded to interleaved `f32` at a fixed output rate/channels.
pub struct AvfAudioReader {
    asset: Retained<AVURLAsset>,
    track: Retained<AVAssetTrack>,
    reader: Retained<AVAssetReader>,
    output: Retained<AVAssetReaderTrackOutput>,
    out_rate: u32,
    channels: usize,
    started: bool,
    /// Decoded interleaved samples pulled but not yet handed out.
    pending: Vec<f32>,
    pending_cursor: usize,
    /// Output-frame position of the next sample [`read`](AudioReader::read)
    /// emits, anchored from the first decoded buffer's PTS.
    position: Option<i64>,
    eos: bool,
}

// SAFETY: the AVFoundation objects are owned and touched by a single thread at a
// time — the export/mix worker this reader is parked on. We never share `&self`
// across threads, satisfying `AudioReader: Send` without `Sync`.
unsafe impl Send for AvfAudioReader {}

impl AvfAudioReader {
    /// Open the first audio track of `path`, decoding to interleaved `f32` at
    /// `out_rate` Hz and `channels` channels.
    pub fn open(path: &Path, out_rate: u32, channels: u16) -> Result<Self, DecodeError> {
        if out_rate == 0 || channels == 0 {
            return Err(DecodeError::unsupported("zero audio rate or channels"));
        }
        let path_str = path
            .to_str()
            .ok_or_else(|| DecodeError::unsupported("path is not valid UTF-8"))?;

        let url = NSURL::fileURLWithPath(&NSString::from_str(path_str));
        let asset = unsafe { AVURLAsset::initWithURL_options(AVURLAsset::alloc(), &url, None) };

        let track = first_audio_track(&asset)
            .ok_or_else(|| DecodeError::unsupported("no audio track found"))?;

        let (reader, output) = build_audio_reader(&asset, &track, out_rate, channels as usize, None)?;

        Ok(Self {
            asset,
            track,
            reader,
            output,
            out_rate,
            channels: channels as usize,
            started: false,
            pending: Vec::new(),
            pending_cursor: 0,
            position: None,
            eos: false,
        })
    }

    /// Pull the next sample buffer into `pending`, anchoring `position` on the
    /// first buffer. Returns `false` at end of stream.
    fn fill_pending(&mut self) -> Result<bool, DecodeError> {
        if !self.started {
            if !unsafe { self.reader.startReading() } {
                return Err(self.reader_error("startReading failed"));
            }
            self.started = true;
        }

        loop {
            let Some(sample) = (unsafe { self.output.copyNextSampleBuffer() }) else {
                self.eos = true;
                return match unsafe { self.reader.status() } {
                    AVAssetReaderStatus::Failed => Err(self.reader_error("read failed")),
                    _ => Ok(false),
                };
            };

            if unsafe { sample.num_samples() } == 0 {
                continue; // marker-only buffer
            }

            if self.position.is_none() {
                let pts = unsafe { sample.presentation_time_stamp() };
                self.position = Some(pts_to_out_frame(pts, self.out_rate));
            }

            let Some(block) = (unsafe { sample.data_buffer() }) else {
                continue;
            };
            let len = unsafe { block.data_length() };
            if len == 0 {
                continue;
            }
            let mut bytes = vec![0u8; len];
            let status = unsafe {
                block.copy_data_bytes(
                    0,
                    len,
                    NonNull::new(bytes.as_mut_ptr().cast()).expect("non-null dest"),
                )
            };
            if status != 0 {
                return Err(DecodeError::Decode(format!(
                    "CMBlockBufferCopyDataBytes failed ({status})"
                )));
            }

            // LPCM float output is native-endian interleaved f32.
            self.pending = bytes
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            self.pending_cursor = 0;
            return Ok(true);
        }
    }

    fn reader_error(&self, context: &str) -> DecodeError {
        let detail = unsafe { self.reader.error() }
            .map(|e| e.localizedDescription().to_string())
            .unwrap_or_else(|| context.to_string());
        DecodeError::Decode(format!("{context}: {detail}"))
    }
}

impl AudioReader for AvfAudioReader {
    fn read(&mut self, out: &mut [f32]) -> Result<usize, DecodeError> {
        let channels = self.channels;
        debug_assert_eq!(out.len() % channels, 0, "buffer not a frame multiple");
        let want_frames = out.len() / channels;
        let mut produced = 0;

        while produced < want_frames {
            if self.pending_cursor >= self.pending.len() {
                if self.eos {
                    break;
                }
                if !self.fill_pending()? {
                    break;
                }
                continue;
            }
            let avail_frames = (self.pending.len() - self.pending_cursor) / channels;
            let take = (want_frames - produced).min(avail_frames);
            if take == 0 {
                // A partial trailing frame should never occur; guard anyway.
                self.pending_cursor = self.pending.len();
                continue;
            }
            let src = &self.pending[self.pending_cursor..self.pending_cursor + take * channels];
            out[produced * channels..(produced + take) * channels].copy_from_slice(src);
            self.pending_cursor += take * channels;
            produced += take;
        }

        if let Some(pos) = self.position.as_mut() {
            *pos += produced as i64;
        }
        Ok(produced)
    }

    fn seek_to_frame(&mut self, frame: i64) -> Result<(), DecodeError> {
        if self.position == Some(frame) {
            return Ok(());
        }
        // Start time expressed at the output rate, so `frame` maps exactly.
        let start = rational_to_cmtime(RationalTime::new(
            frame.max(0),
            Rational::new(self.out_rate as i32, 1),
        ));
        let (reader, output) =
            build_audio_reader(&self.asset, &self.track, self.out_rate, self.channels, Some(start))?;
        self.reader = reader;
        self.output = output;
        self.started = false;
        self.pending.clear();
        self.pending_cursor = 0;
        self.position = None;
        self.eos = false;
        Ok(())
    }

    fn position(&self) -> Option<i64> {
        self.position
    }
}

/// First audio track of an asset, or `None` if it has none.
fn first_audio_track(asset: &AVURLAsset) -> Option<Retained<AVAssetTrack>> {
    let media_audio = unsafe { AVMediaTypeAudio }?;
    #[allow(deprecated)]
    let tracks = unsafe { asset.tracksWithMediaType(media_audio) };
    tracks.firstObject()
}

/// Build a reader + LPCM track output, optionally starting at `start`.
fn build_audio_reader(
    asset: &AVURLAsset,
    track: &AVAssetTrack,
    out_rate: u32,
    channels: usize,
    start: Option<CMTime>,
) -> Result<(Retained<AVAssetReader>, Retained<AVAssetReaderTrackOutput>), DecodeError> {
    let reader = unsafe { AVAssetReader::initWithAsset_error(AVAssetReader::alloc(), asset) }
        .map_err(|e| DecodeError::Open(e.localizedDescription().to_string()))?;

    let settings = audio_output_settings(out_rate, channels)?;
    let output = unsafe {
        AVAssetReaderTrackOutput::initWithTrack_outputSettings(
            AVAssetReaderTrackOutput::alloc(),
            track,
            Some(&settings),
        )
    };

    unsafe {
        if let Some(start) = start {
            reader.setTimeRange(CMTimeRange {
                start,
                duration: kCMTimePositiveInfinity,
            });
        }
        if !reader.canAddOutput(&output) {
            return Err(DecodeError::Open("reader cannot add audio output".into()));
        }
        reader.addOutput(&output);
    }
    Ok((reader, output))
}

/// LPCM float32 output settings at `out_rate` / `channels`, interleaved.
fn audio_output_settings(
    out_rate: u32,
    channels: usize,
) -> Result<Retained<NSMutableDictionary<NSString, AnyObject>>, DecodeError> {
    let settings = NSMutableDictionary::<NSString, AnyObject>::new();

    let set = |key: Option<&'static NSString>, value: &AnyObject| -> Result<(), DecodeError> {
        let key = key.ok_or_else(|| DecodeError::unsupported("missing AV audio settings key"))?;
        unsafe { settings.setObject_forKey(value, ProtocolObject::from_ref(key)) };
        Ok(())
    };

    let format = NSNumber::numberWithUnsignedInt(K_AUDIO_FORMAT_LINEAR_PCM);
    let rate = NSNumber::numberWithDouble(f64::from(out_rate));
    let chans = NSNumber::numberWithInt(channels as i32);
    let depth = NSNumber::numberWithInt(32);
    let is_float = NSNumber::numberWithBool(true);
    let is_big_endian = NSNumber::numberWithBool(false);
    let is_non_interleaved = NSNumber::numberWithBool(false);

    unsafe {
        set(AVFormatIDKey, &format)?;
        set(AVSampleRateKey, &rate)?;
        set(AVNumberOfChannelsKey, &chans)?;
        set(AVLinearPCMBitDepthKey, &depth)?;
        set(AVLinearPCMIsFloatKey, &is_float)?;
        set(AVLinearPCMIsBigEndianKey, &is_big_endian)?;
        set(AVLinearPCMIsNonInterleaved, &is_non_interleaved)?;
    }
    Ok(settings)
}

/// Convert a sample buffer PTS to an output-frame index (round to nearest).
fn pts_to_out_frame(pts: CMTime, out_rate: u32) -> i64 {
    if pts.timescale <= 0 || !pts.flags.contains(CMTimeFlags::Valid) || pts.value <= 0 {
        return 0;
    }
    let num = i128::from(pts.value) * i128::from(out_rate);
    let den = i128::from(pts.timescale);
    ((num + den / 2) / den) as i64
}
