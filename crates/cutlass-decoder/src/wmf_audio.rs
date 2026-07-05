//! Windows audio decode: a Media Foundation **Source Reader** over the first
//! audio track, delivering interleaved `f32` at a caller-chosen rate and
//! channel count (the mixer's format) behind [`cutlass_core::AudioReader`].
//!
//! ## Format negotiation
//!
//! The reader is created with `MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING`
//! — despite the name, that attribute is what allows the source reader to
//! insert the Audio Resampler DSP, so we first ask for **Float32 PCM at the
//! exact output rate/channels** and usually get it (the passthrough path:
//! samples are copied straight out). When a source can't be brought to that
//! format we fall back to *any* Float32/PCM16 layout the reader offers and do
//! the remaining work in Rust: channel mixing at ingest and streaming linear
//! resampling on read — the same policy the Android backend applies.
//!
//! ## Position model
//!
//! Positions are **output sample frames** since the start of the source
//! (`frame / out_rate` seconds). Sample timestamps anchor the mapping: after a
//! seek, the reader lands at/before the target and [`read`](AudioReader::read)
//! discards forward to it exactly; a source whose audio starts late shows up
//! as a `position()` ahead of the seek target (the mixers pad the lead with
//! silence). Timestamp gaps inside the stream render as silence, keeping A/V
//! sync rather than compacting time.

use std::path::Path;

use windows::Win32::Media::MediaFoundation::{
    IMFAttributes, IMFMediaBuffer, IMFSourceReader, MF_MT_ALL_SAMPLES_INDEPENDENT,
    MF_MT_AUDIO_AVG_BYTES_PER_SECOND, MF_MT_AUDIO_BITS_PER_SAMPLE, MF_MT_AUDIO_BLOCK_ALIGNMENT,
    MF_MT_AUDIO_NUM_CHANNELS, MF_MT_AUDIO_SAMPLES_PER_SECOND, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE,
    MF_SOURCE_READER_ALL_STREAMS, MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING,
    MF_SOURCE_READER_FIRST_AUDIO_STREAM, MF_SOURCE_READERF_ENDOFSTREAM, MFAudioFormat_Float,
    MFAudioFormat_PCM, MFCreateAttributes, MFCreateMediaType, MFMediaType_Audio,
};
use windows::Win32::System::Com::StructuredStorage::PROPVARIANT;
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx};
use windows::core::GUID;

use cutlass_core::{AudioReader, DecodeError};

use crate::wmf::{
    HNS_PER_SEC, MAX_EMPTY_READS, decode_err, ensure_mf_platform, open_err, open_source_reader,
    probe_duration, seek_err,
};

/// How the negotiated source-side samples are encoded in the media buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampleFormat {
    F32,
    I16,
}

/// An audio source decoded to interleaved `f32` at a fixed output
/// rate/channels, streamed sample-by-sample (playback pulls this live).
pub struct WmfAudioReader {
    reader: IMFSourceReader,
    /// The "first audio stream" selector, reused for every reader call.
    stream: u32,
    out_rate: u32,
    out_channels: usize,
    /// Rate/format the reader actually delivers. `src_rate == out_rate` is the
    /// passthrough fast path; otherwise `read` linearly resamples.
    src_rate: u32,
    src_channels: usize,
    src_format: SampleFormat,
    /// Decoded samples not yet handed out: interleaved `f32`, already mixed to
    /// `out_channels`, still at `src_rate`. Frame `cursor` is absolute source
    /// frame `cursor_src`.
    pending: Vec<f32>,
    cursor: usize,
    cursor_src: i64,
    /// Last consumed source frame, kept for interpolation across pulls.
    carry: Vec<f32>,
    carry_src: Option<i64>,
    /// Output frame the next `read` emits; `None` until a decode anchors it.
    next_out: Option<i64>,
    /// Source length in output frames, when the container reports one. Seeks
    /// at/past it skip the (failing) `SetCurrentPosition` and pin EOS instead.
    duration_frames: Option<i64>,
    eos: bool,
}

// SAFETY: the Source Reader and its COM children are owned and touched by one
// thread at a time — the mixer/export worker this reader is parked on. Opened
// in the MTA (see `open`), so the COM objects may legally travel between
// worker threads; `&self` is never shared across threads.
unsafe impl Send for WmfAudioReader {}

impl WmfAudioReader {
    /// Open the first audio track of `path`, decoding to interleaved `f32` at
    /// `out_rate` Hz and `channels` channels.
    pub fn open(path: &Path, out_rate: u32, channels: u16) -> Result<Self, DecodeError> {
        if out_rate == 0 || channels == 0 {
            return Err(DecodeError::unsupported("zero audio rate or channels"));
        }
        ensure_mf_platform();
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        }

        // ADVANCED_VIDEO_PROCESSING lets the reader chain the resampler DSP
        // behind the decoder, enabling rate/channel conversion to our format.
        let attributes = reader_attributes().map_err(open_err)?;
        let reader = open_source_reader(path, Some(&attributes))?;
        let stream = MF_SOURCE_READER_FIRST_AUDIO_STREAM.0 as u32;

        unsafe {
            reader
                .SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS.0 as u32, false)
                .map_err(open_err)?;
            // Selecting the audio stream doubles as the "has audio?" check.
            reader
                .SetStreamSelection(stream, true)
                .map_err(|_| DecodeError::unsupported("no audio track found"))?;
        }

        let (src_rate, src_channels, src_format) =
            negotiate_output(&reader, stream, out_rate, channels)?;

        // `seconds * out_rate`, floored — a frame at/past this is out of range.
        let duration_frames = probe_duration(&reader, Some(path))
            .map(|d| (i128::from(d.value) * i128::from(out_rate) / i128::from(HNS_PER_SEC)) as i64);

        Ok(Self {
            reader,
            stream,
            out_rate,
            out_channels: channels as usize,
            src_rate,
            src_channels,
            src_format,
            pending: Vec::new(),
            cursor: 0,
            cursor_src: 0,
            carry: Vec::new(),
            carry_src: None,
            next_out: None,
            duration_frames,
            eos: false,
        })
    }

    fn passthrough(&self) -> bool {
        self.src_rate == self.out_rate
    }

    fn avail_frames(&self) -> usize {
        self.pending.len() / self.out_channels - self.cursor
    }

    /// Pull the next non-empty sample into `pending` (compacting the consumed
    /// prefix first). Returns `false` at end of stream.
    fn fill_pending(&mut self) -> Result<bool, DecodeError> {
        if self.eos {
            return Ok(false);
        }
        if self.cursor > 0 {
            self.pending.drain(..self.cursor * self.out_channels);
            self.cursor = 0;
        }

        for _ in 0..MAX_EMPTY_READS {
            let mut stream_flags: u32 = 0;
            let mut timestamp: i64 = 0;
            let mut sample = None;
            unsafe {
                self.reader.ReadSample(
                    self.stream,
                    0,
                    None,
                    Some(&mut stream_flags),
                    Some(&mut timestamp),
                    Some(&mut sample),
                )
            }
            .map_err(decode_err)?;

            if stream_flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0 {
                self.eos = true;
            }
            let Some(sample) = sample else {
                if self.eos {
                    return Ok(false);
                }
                continue; // stream tick / gap
            };

            let buffer = unsafe { sample.ConvertToContiguousBuffer() }.map_err(decode_err)?;
            if self.append_buffer(&buffer, timestamp)? {
                return Ok(true);
            }
            if self.eos {
                return Ok(false);
            }
        }
        Err(DecodeError::Decode(
            "source reader produced no audio within the read budget".into(),
        ))
    }

    /// Decode `buffer` into `pending` (mixed to `out_channels`), anchoring
    /// `cursor_src` from `timestamp` when the queue was empty. Returns whether
    /// any frames were appended.
    fn append_buffer(
        &mut self,
        buffer: &IMFMediaBuffer,
        timestamp: i64,
    ) -> Result<bool, DecodeError> {
        let mut base: *mut u8 = core::ptr::null_mut();
        let mut max_len: u32 = 0;
        let mut cur_len: u32 = 0;
        unsafe { buffer.Lock(&mut base, Some(&mut max_len), Some(&mut cur_len)) }
            .map_err(decode_err)?;
        let result = (|| -> Result<bool, DecodeError> {
            if base.is_null() {
                return Err(DecodeError::Decode("locked audio buffer is null".into()));
            }
            // SAFETY: the buffer holds `cur_len` valid bytes while locked.
            let bytes = unsafe { core::slice::from_raw_parts(base, cur_len as usize) };

            let bytes_per_sample = match self.src_format {
                SampleFormat::F32 => 4,
                SampleFormat::I16 => 2,
            };
            let frame_bytes = bytes_per_sample * self.src_channels;
            let frames = bytes.len() / frame_bytes;
            if frames == 0 {
                return Ok(false);
            }
            if self.avail_frames() == 0 {
                // Empty queue: this sample's timestamp is the new anchor.
                self.cursor_src = src_frame_of_hns(timestamp, self.src_rate);
                self.cursor = 0;
                self.pending.clear();
            }

            self.pending.reserve(frames * self.out_channels);
            let mut src_frame = vec![0.0f32; self.src_channels];
            let mut out_frame = vec![0.0f32; self.out_channels];
            for i in 0..frames {
                let frame = &bytes[i * frame_bytes..(i + 1) * frame_bytes];
                match self.src_format {
                    SampleFormat::F32 => {
                        for (c, chunk) in frame.chunks_exact(4).enumerate() {
                            src_frame[c] =
                                f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        }
                    }
                    SampleFormat::I16 => {
                        for (c, chunk) in frame.chunks_exact(2).enumerate() {
                            src_frame[c] =
                                f32::from(i16::from_ne_bytes([chunk[0], chunk[1]])) / 32768.0;
                        }
                    }
                }
                mix_frame(&src_frame, &mut out_frame);
                self.pending.extend_from_slice(&out_frame);
            }
            Ok(true)
        })();
        unsafe { buffer.Unlock() }.map_err(decode_err)?;
        result
    }

    /// Consume the pending head frame, remembering it as the interpolation
    /// left-neighbor.
    fn consume_into_carry(&mut self) {
        let ch = self.out_channels;
        let start = self.cursor * ch;
        self.carry.clear();
        self.carry
            .extend_from_slice(&self.pending[start..start + ch]);
        self.carry_src = Some(self.cursor_src);
        self.cursor += 1;
        self.cursor_src += 1;
    }

    /// Establish `next_out` from the stream's first decoded timestamp.
    /// `target` is a seek destination: emit from `max(target, anchored)` so a
    /// late-starting stream is reported (mixers pad the lead) rather than
    /// silently shifted.
    fn anchor(&mut self, target: Option<i64>) -> Result<(), DecodeError> {
        if self.avail_frames() == 0 && !self.eos {
            self.fill_pending()?;
        }
        let anchored = (self.avail_frames() > 0)
            .then(|| out_frame_of_src(self.cursor_src, self.src_rate, self.out_rate));
        self.next_out = Some(match (target, anchored) {
            (Some(t), Some(a)) => t.max(a),
            (Some(t), None) => t,
            (None, Some(a)) => a,
            (None, None) => 0,
        });
        Ok(())
    }

    /// `read` when the reader already delivers `out_rate` (the common case):
    /// bulk copies, with timestamp gaps rendered as silence.
    fn read_passthrough(
        &mut self,
        out: &mut [f32],
        mut pos: i64,
        want: usize,
    ) -> Result<(usize, i64), DecodeError> {
        let ch = self.out_channels;
        let mut produced = 0;
        while produced < want {
            if self.avail_frames() == 0 && (self.eos || !self.fill_pending()?) {
                break;
            }
            if self.cursor_src < pos {
                // Stale frames from before the emit position (post-seek): drop.
                let drop = ((pos - self.cursor_src) as usize).min(self.avail_frames());
                self.cursor += drop;
                self.cursor_src += drop as i64;
                continue;
            }
            if self.cursor_src > pos {
                // Timestamp gap: keep time, emit silence.
                let fill = ((self.cursor_src - pos) as usize).min(want - produced);
                out[produced * ch..(produced + fill) * ch].fill(0.0);
                produced += fill;
                pos += fill as i64;
                continue;
            }
            let take = self.avail_frames().min(want - produced);
            let start = self.cursor * ch;
            out[produced * ch..(produced + take) * ch]
                .copy_from_slice(&self.pending[start..start + take * ch]);
            self.cursor += take;
            self.cursor_src += take as i64;
            produced += take;
            pos += take as i64;
        }
        Ok((produced, pos))
    }

    /// `read` when rates differ: streaming linear interpolation between the
    /// two nearest source frames of each output position.
    fn read_resampled(
        &mut self,
        out: &mut [f32],
        mut pos: i64,
        want: usize,
    ) -> Result<(usize, i64), DecodeError> {
        /// Where one interpolation endpoint comes from.
        enum Tap {
            Pending(usize),
            Carry,
            Silence,
        }
        let ch = self.out_channels;
        let mut produced = 0;
        while produced < want {
            let (floor, frac) = src_pos_of(pos, self.src_rate, self.out_rate);
            // Pull until the right-neighbor frame (floor + 1) is in the queue.
            while !self.eos && self.cursor_src + self.avail_frames() as i64 <= floor + 1 {
                if !self.fill_pending()? {
                    break;
                }
            }
            while self.cursor_src < floor && self.avail_frames() > 0 {
                self.consume_into_carry();
            }
            if self.avail_frames() == 0 {
                break; // end of stream
            }
            let (a, b) = if self.cursor_src == floor {
                let b = if self.avail_frames() >= 2 {
                    Tap::Pending(self.cursor + 1)
                } else {
                    Tap::Pending(self.cursor) // EOS: hold the last frame
                };
                (Tap::Pending(self.cursor), b)
            } else if self.cursor_src == floor + 1 {
                // Left neighbor was consumed (or never existed for a
                // late-starting stream): use the carry when it matches.
                match self.carry_src {
                    Some(cs) if cs == floor => (Tap::Carry, Tap::Pending(self.cursor)),
                    _ => (Tap::Pending(self.cursor), Tap::Pending(self.cursor)),
                }
            } else {
                // Both neighbors precede the queued data: a timestamp gap.
                (Tap::Silence, Tap::Silence)
            };
            for c in 0..ch {
                let tap = |t: &Tap| -> f32 {
                    match t {
                        Tap::Pending(frame) => self.pending[frame * ch + c],
                        Tap::Carry => self.carry[c],
                        Tap::Silence => 0.0,
                    }
                };
                let (a, b) = (tap(&a), tap(&b));
                out[produced * ch + c] = a + (b - a) * frac;
            }
            produced += 1;
            pos += 1;
        }
        Ok((produced, pos))
    }
}

impl AudioReader for WmfAudioReader {
    fn read(&mut self, out: &mut [f32]) -> Result<usize, DecodeError> {
        let ch = self.out_channels;
        debug_assert_eq!(out.len() % ch, 0, "buffer not a frame multiple");
        let want = out.len() / ch;
        if self.next_out.is_none() {
            self.anchor(None)?;
        }
        let pos = self.next_out.expect("anchored above");
        let (produced, pos) = if self.passthrough() {
            self.read_passthrough(out, pos, want)?
        } else {
            self.read_resampled(out, pos, want)?
        };
        self.next_out = Some(pos);
        Ok(produced)
    }

    fn seek_to_frame(&mut self, frame: i64) -> Result<(), DecodeError> {
        let frame = frame.max(0);
        if self.next_out == Some(frame) {
            return Ok(());
        }
        // A short hop forward decodes-and-discards through the absolute
        // position mapping in `read` — cheaper than a container seek that
        // lands on a packet boundary and re-primes the decoder.
        if let Some(current) = self.next_out
            && frame > current
            && frame - current <= i64::from(self.out_rate)
        {
            self.next_out = Some(frame);
            return Ok(());
        }
        // At or past the end: some sources fail `SetCurrentPosition` there
        // (`MF_E_INVALID_POSITION`); the contract is simply "subsequent reads
        // return 0 frames", so pin EOS without touching the reader.
        if let Some(len) = self.duration_frames
            && frame >= len
        {
            self.pending.clear();
            self.cursor = 0;
            self.eos = true;
            self.next_out = Some(frame);
            return Ok(());
        }

        // `GUID_NULL` selects the 100-ns time format (as the video path does).
        let hns = i128::from(frame) * i128::from(HNS_PER_SEC) / i128::from(self.out_rate);
        let position = PROPVARIANT::from(hns as i64);
        let time_format = GUID::from_u128(0);
        unsafe { self.reader.SetCurrentPosition(&time_format, &position) }.map_err(seek_err)?;

        self.pending.clear();
        self.cursor = 0;
        self.cursor_src = 0;
        self.carry.clear();
        self.carry_src = None;
        self.eos = false;
        self.next_out = None;
        // Anchor eagerly so `position()` right after the seek is exact — the
        // mixers read it to align a late-starting stream.
        self.anchor(Some(frame))
    }

    fn position(&self) -> Option<i64> {
        self.next_out
    }
}

/// Reader-creation attributes: enable the converter/resampler pipeline.
fn reader_attributes() -> Result<IMFAttributes, windows::core::Error> {
    let mut attributes: Option<IMFAttributes> = None;
    unsafe { MFCreateAttributes(&mut attributes, 1) }?;
    let attributes = attributes.expect("MFCreateAttributes succeeded");
    unsafe {
        attributes.SetUINT32(&MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, 1)?;
    }
    Ok(attributes)
}

/// Negotiate the reader's audio output type, preferring exact-format
/// passthrough. Returns the (rate, channels, sample format) it will deliver.
fn negotiate_output(
    reader: &IMFSourceReader,
    stream: u32,
    out_rate: u32,
    channels: u16,
) -> Result<(u32, usize, SampleFormat), DecodeError> {
    // 1. Float32 at exactly the mixer's rate/channels (resampler inserted).
    let exact = exact_float_type(out_rate, channels).map_err(open_err)?;
    if unsafe { reader.SetCurrentMediaType(stream, None, &exact) }.is_ok() {
        return read_back_output(reader, stream);
    }

    // 2. Any Float32 layout (rate/channels chosen by the source).
    if let Ok(partial) = partial_type(&MFAudioFormat_Float)
        && unsafe { reader.SetCurrentMediaType(stream, None, &partial) }.is_ok()
    {
        return read_back_output(reader, stream);
    }

    // 3. Any 16-bit PCM layout (converted to f32 at ingest).
    if let Ok(partial) = partial_type(&MFAudioFormat_PCM)
        && unsafe { reader.SetCurrentMediaType(stream, None, &partial) }.is_ok()
    {
        return read_back_output(reader, stream);
    }

    Err(DecodeError::unsupported(
        "could not negotiate PCM output for the audio track",
    ))
}

/// A complete Float32 PCM media type at `rate`/`channels`.
fn exact_float_type(
    rate: u32,
    channels: u16,
) -> Result<windows::Win32::Media::MediaFoundation::IMFMediaType, windows::core::Error> {
    let block_align = u32::from(channels) * 4;
    let media_type = unsafe { MFCreateMediaType() }?;
    unsafe {
        media_type.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)?;
        media_type.SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_Float)?;
        media_type.SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, rate)?;
        media_type.SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, u32::from(channels))?;
        media_type.SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, 32)?;
        media_type.SetUINT32(&MF_MT_AUDIO_BLOCK_ALIGNMENT, block_align)?;
        media_type.SetUINT32(&MF_MT_AUDIO_AVG_BYTES_PER_SECOND, rate * block_align)?;
        media_type.SetUINT32(&MF_MT_ALL_SAMPLES_INDEPENDENT, 1)?;
    }
    Ok(media_type)
}

/// A partial media type (major + subtype); the reader completes the layout.
fn partial_type(
    subtype: &GUID,
) -> Result<windows::Win32::Media::MediaFoundation::IMFMediaType, windows::core::Error> {
    let media_type = unsafe { MFCreateMediaType() }?;
    unsafe {
        media_type.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)?;
        media_type.SetGUID(&MF_MT_SUBTYPE, subtype)?;
    }
    Ok(media_type)
}

/// Read the negotiated output layout back from the reader.
fn read_back_output(
    reader: &IMFSourceReader,
    stream: u32,
) -> Result<(u32, usize, SampleFormat), DecodeError> {
    let media_type = unsafe { reader.GetCurrentMediaType(stream) }.map_err(open_err)?;
    let rate =
        unsafe { media_type.GetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND) }.map_err(open_err)?;
    let channels = unsafe { media_type.GetUINT32(&MF_MT_AUDIO_NUM_CHANNELS) }.map_err(open_err)?;
    if rate == 0 || channels == 0 {
        return Err(DecodeError::Open(
            "audio output type reports zero rate or channels".into(),
        ));
    }
    let subtype = unsafe { media_type.GetGUID(&MF_MT_SUBTYPE) }.map_err(open_err)?;
    let format = if subtype == MFAudioFormat_Float {
        SampleFormat::F32
    } else if subtype == MFAudioFormat_PCM {
        let bits = unsafe { media_type.GetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE) }.unwrap_or(16);
        if bits != 16 {
            return Err(DecodeError::Unsupported(format!(
                "unsupported PCM bit depth: {bits}"
            )));
        }
        SampleFormat::I16
    } else {
        return Err(DecodeError::Unsupported(format!(
            "unexpected audio output subtype: {subtype:?}"
        )));
    };
    Ok((rate, channels as usize, format))
}

/// Mix one interleaved source frame into `dst.len()` output channels:
/// identity, mono fan-out, average-downmix to mono, or positional mapping
/// (extra source channels dropped). The resampler DSP handles the standard
/// matrices on the passthrough path; this covers the Rust fallback.
fn mix_frame(src: &[f32], dst: &mut [f32]) {
    if src.len() == dst.len() {
        dst.copy_from_slice(src);
    } else if src.len() == 1 {
        dst.fill(src[0]);
    } else if dst.len() == 1 {
        dst[0] = src.iter().sum::<f32>() / src.len() as f32;
    } else {
        for (c, value) in dst.iter_mut().enumerate() {
            *value = src[c.min(src.len() - 1)];
        }
    }
}

/// Source-frame index of a 100-ns timestamp (rounded to nearest).
fn src_frame_of_hns(hns: i64, src_rate: u32) -> i64 {
    let num = i128::from(hns.max(0)) * i128::from(src_rate);
    let den = i128::from(HNS_PER_SEC);
    ((num + den / 2) / den) as i64
}

/// Output-frame index equivalent to source frame `s` (rounded to nearest).
fn out_frame_of_src(s: i64, src_rate: u32, out_rate: u32) -> i64 {
    let num = i128::from(s) * i128::from(out_rate);
    let den = i128::from(src_rate);
    ((num + den / 2) / den) as i64
}

/// Source position of output frame `pos`: `(floor, frac)` with
/// `source = floor + frac` and `frac ∈ [0, 1)`. Exact in `i128`.
fn src_pos_of(pos: i64, src_rate: u32, out_rate: u32) -> (i64, f32) {
    let num = i128::from(pos) * i128::from(src_rate);
    let den = i128::from(out_rate);
    let floor = num.div_euclid(den);
    let frac = num.rem_euclid(den) as f64 / den as f64;
    (floor as i64, frac as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_frame_covers_the_channel_layouts() {
        // Identity.
        let mut out = [0.0; 2];
        mix_frame(&[0.25, -0.5], &mut out);
        assert_eq!(out, [0.25, -0.5]);
        // Mono fan-out.
        let mut out = [0.0; 2];
        mix_frame(&[0.75], &mut out);
        assert_eq!(out, [0.75, 0.75]);
        // Downmix to mono averages.
        let mut out = [0.0; 1];
        mix_frame(&[1.0, 0.0], &mut out);
        assert_eq!(out, [0.5]);
        // Positional map: extra source channels dropped, missing ones cloned.
        let mut out = [0.0; 3];
        mix_frame(&[0.1, 0.2], &mut out);
        assert_eq!(out, [0.1, 0.2, 0.2]);
    }

    #[test]
    fn src_pos_is_exact_for_equal_rates() {
        for pos in [0, 1, 47_999, 48_000] {
            let (floor, frac) = src_pos_of(pos, 48_000, 48_000);
            assert_eq!(floor, pos);
            assert_eq!(frac, 0.0);
        }
    }

    #[test]
    fn src_pos_tracks_ratio_for_unequal_rates() {
        // 44.1k source read at 48k out: output frame 48000 is source 44100.
        let (floor, frac) = src_pos_of(48_000, 44_100, 48_000);
        assert_eq!(floor, 44_100);
        assert_eq!(frac, 0.0);
        // One output frame is 0.91875 source frames.
        let (floor, frac) = src_pos_of(1, 44_100, 48_000);
        assert_eq!(floor, 0);
        assert!((frac - 0.918_75).abs() < 1e-6);
    }

    #[test]
    fn frame_conversions_round_to_nearest() {
        // 1 second of 100-ns ticks at 44.1 kHz.
        assert_eq!(src_frame_of_hns(10_000_000, 44_100), 44_100);
        // Half a sample rounds up.
        assert_eq!(src_frame_of_hns(113, 44_100), 0);
        assert_eq!(src_frame_of_hns(114, 44_100), 1);
        // Source->output round trip at the common 44.1k -> 48k ratio.
        assert_eq!(out_frame_of_src(44_100, 44_100, 48_000), 48_000);
        assert_eq!(out_frame_of_src(22_050, 44_100, 48_000), 24_000);
    }
}
