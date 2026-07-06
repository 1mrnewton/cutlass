//! Windows backend: Media Foundation `IMFSinkWriter` (H.264 encode + mp4 mux,
//! AAC audio) behind [`cutlass_core::VideoEncoder`].
//!
//! Pushed frames arrive as packed RGBA CPU planes (what the renderer's export
//! loop produces). Each is converted to NV12 on the CPU — the one input
//! format every H.264 encoder MFT accepts, hardware or software, so encode
//! never hinges on a driver's RGB conversion — wrapped in an `IMFSample`,
//! and handed to the sink writer, which drives the encoder MFT and muxes to
//! the output `.mp4`. Audio blocks are converted `f32` → 16-bit PCM and AAC
//! is encoded by the writer's own MFT.
//!
//! Mirrors the decoder's Media Foundation patterns (`cutlass-decoder`'s
//! `wmf.rs`): one-time `MFStartup`, per-thread MTA `CoInitializeEx`, and
//! `windows` crate COM ergonomics.

use std::path::Path;
use std::sync::Once;

use windows::Win32::Media::MediaFoundation::{
    IMFMediaType, IMFSinkWriter, MF_MT_ALL_SAMPLES_INDEPENDENT, MF_MT_AUDIO_AVG_BYTES_PER_SECOND,
    MF_MT_AUDIO_BITS_PER_SAMPLE, MF_MT_AUDIO_BLOCK_ALIGNMENT, MF_MT_AUDIO_NUM_CHANNELS,
    MF_MT_AUDIO_SAMPLES_PER_SECOND, MF_MT_AVG_BITRATE, MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE,
    MF_MT_INTERLACE_MODE, MF_MT_MAJOR_TYPE, MF_MT_MAX_KEYFRAME_SPACING, MF_MT_SUBTYPE,
    MF_MT_TRANSFER_FUNCTION, MF_MT_VIDEO_NOMINAL_RANGE, MF_MT_VIDEO_PRIMARIES, MF_MT_YUV_MATRIX,
    MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, MF_SINK_WRITER_DISABLE_THROTTLING, MF_VERSION,
    MFAudioFormat_AAC, MFAudioFormat_PCM, MFCreateAttributes, MFCreateMediaType,
    MFCreateMemoryBuffer, MFCreateSample, MFCreateSinkWriterFromURL, MFMediaType_Audio,
    MFMediaType_Video, MFNominalRange_16_235, MFSTARTUP_FULL, MFStartup, MFVideoFormat_H264,
    MFVideoFormat_NV12, MFVideoInterlace_Progressive, MFVideoPrimaries_BT709,
    MFVideoPrimaries_SMPTE170M, MFVideoTransFunc_709, MFVideoTransferMatrix_BT601,
    MFVideoTransferMatrix_BT709,
};
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx};
use windows::core::HSTRING;

use cutlass_core::{
    AudioEncoderConfig, EncodeError, EncoderConfig, PixelFormat, Rational, RationalTime,
    VideoEncoder, VideoFrame,
};

use crate::yuv::{YuvMatrix, matrix_for, rgba_to_nv12};

/// Media Foundation's timestamp tick rate: 100-nanosecond units per second.
const HNS_PER_SEC: i64 = 10_000_000;

/// AAC bytes-per-second for the muxed audio track. Media Foundation's AAC
/// encoder accepts exactly 12000/16000/20000/24000; 16000 B/s = 128 kbps,
/// matching the Apple backend's `AAC_BIT_RATE`.
const AAC_BYTES_PER_SECOND: u32 = 16_000;

/// H.264 target bitrate for `width`×`height` at `frame_rate`: ~0.1 bit per
/// pixel per frame — delivery-grade for 1080p30 (≈6 Mbps) and 4K30
/// (≈25 Mbps), modest for the small short-GOP preview proxies (≈1.5 Mbps at
/// 960×540). Clamped to sane codec bounds.
fn h264_bitrate(width: u32, height: u32, frame_rate: Rational) -> u32 {
    let fps = if frame_rate.is_valid() {
        f64::from(frame_rate.num) / f64::from(frame_rate.den)
    } else {
        30.0
    };
    let bits = f64::from(width) * f64::from(height) * fps * 0.1;
    bits.clamp(1_000_000.0, 60_000_000.0) as u32
}

/// An H.264 (+ optional AAC) → mp4 encoder backed by Media Foundation's sink
/// writer. The writer owns interleaving and throttling internally, so pushes
/// are plain blocking calls — no readiness pump like AVFoundation needs.
pub struct WmfEncoder {
    writer: IMFSinkWriter,
    video_stream: u32,
    /// Audio stream index, present iff `config.audio` was `Some`.
    audio_stream: Option<u32>,
    size: (u32, u32),
    frame_rate: Rational,
    /// RGB→YUV coefficients for the NV12 conversion, from `config.color` —
    /// must match what the stream is tagged as (and what decoders assume for
    /// untagged HD) or every round-trip shifts colors.
    matrix: YuvMatrix,
    audio: Option<AudioEncoderConfig>,
    finished: bool,
}

// SAFETY: the sink writer and its COM children are owned and touched by a
// single thread at a time — the export worker the engine parks this encoder
// on. We open in the multithreaded apartment (see `open`) so the COM objects
// may legally travel between MTA worker threads, and we never share `&self`
// across threads, which satisfies `VideoEncoder: Send` without `Sync`.
unsafe impl Send for WmfEncoder {}

impl WmfEncoder {
    /// Create an mp4 sink writer at `path` for `config`'s size and frame
    /// rate. Overwrites any existing file.
    pub fn open(path: &Path, config: EncoderConfig) -> Result<Self, EncodeError> {
        let (width, height) = config.size;
        if width == 0 || height == 0 {
            return Err(EncodeError::unsupported("zero output dimensions"));
        }
        if width % 2 != 0 || height % 2 != 0 {
            // H.264 4:2:0 needs even dimensions; the export path evens them
            // upstream (`ExportSettings::evened`).
            return Err(EncodeError::unsupported(format!(
                "H.264 output size must be even, got {width}x{height}"
            )));
        }
        if !config.frame_rate.is_valid() {
            return Err(EncodeError::unsupported("invalid frame rate"));
        }
        let matrix = matrix_for(&config.color);

        ensure_mf_platform();
        // The writer instantiates COM objects on the calling thread; join the
        // MTA (best-effort, mirroring the decoder) so it can live on any
        // worker thread.
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        }

        let _ = std::fs::remove_file(path);

        // Writer attributes: allow hardware encoder MFTs (QuickSync/NVENC/
        // AMF) with Microsoft's software encoder as automatic fallback, and
        // don't throttle exports to real time.
        let mut attrs = None;
        unsafe { MFCreateAttributes(&mut attrs, 2) }.map_err(start_err)?;
        let attrs = attrs.ok_or_else(|| EncodeError::Start("MFCreateAttributes".into()))?;
        unsafe {
            attrs
                .SetUINT32(&MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, 1)
                .map_err(start_err)?;
            attrs
                .SetUINT32(&MF_SINK_WRITER_DISABLE_THROTTLING, 1)
                .map_err(start_err)?;
        }

        let url = HSTRING::from(path);
        let writer = unsafe { MFCreateSinkWriterFromURL(&url, None, &attrs) }.map_err(start_err)?;

        // Video output: H.264 at the target size/rate. `MF_MT_AVG_BITRATE`
        // is required by the encoder MFT; keyframe spacing is optional and
        // only proxies set it.
        let video_out = unsafe { MFCreateMediaType() }.map_err(start_err)?;
        unsafe {
            video_out
                .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
                .map_err(start_err)?;
            video_out
                .SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_H264)
                .map_err(start_err)?;
            video_out
                .SetUINT32(
                    &MF_MT_AVG_BITRATE,
                    h264_bitrate(width, height, config.frame_rate),
                )
                .map_err(start_err)?;
            video_out
                .SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)
                .map_err(start_err)?;
            video_out
                .SetUINT64(&MF_MT_FRAME_SIZE, pack_u32_pair(width, height))
                .map_err(start_err)?;
            video_out
                .SetUINT64(
                    &MF_MT_FRAME_RATE,
                    pack_u32_pair(config.frame_rate.num as u32, config.frame_rate.den as u32),
                )
                .map_err(start_err)?;
            if let Some(interval) = config.keyframe_interval {
                video_out
                    .SetUINT32(&MF_MT_MAX_KEYFRAME_SPACING, interval.max(1))
                    .map_err(start_err)?;
            }
            set_color_tags(&video_out, matrix).map_err(start_err)?;
        }
        let video_stream = unsafe { writer.AddStream(&video_out) }.map_err(start_err)?;

        // Video input: NV12 frames we convert on the CPU.
        let video_in = unsafe { MFCreateMediaType() }.map_err(start_err)?;
        unsafe {
            video_in
                .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
                .map_err(start_err)?;
            video_in
                .SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)
                .map_err(start_err)?;
            video_in
                .SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)
                .map_err(start_err)?;
            video_in
                .SetUINT64(&MF_MT_FRAME_SIZE, pack_u32_pair(width, height))
                .map_err(start_err)?;
            video_in
                .SetUINT64(
                    &MF_MT_FRAME_RATE,
                    pack_u32_pair(config.frame_rate.num as u32, config.frame_rate.den as u32),
                )
                .map_err(start_err)?;
            set_color_tags(&video_in, matrix).map_err(start_err)?;
            writer
                .SetInputMediaType(video_stream, &video_in, None)
                .map_err(start_err)?;
        }

        // Optional AAC audio track, fed 16-bit PCM.
        let audio_stream = match config.audio {
            Some(audio) => Some(add_audio_stream(&writer, audio)?),
            None => None,
        };

        unsafe { writer.BeginWriting() }.map_err(start_err)?;

        Ok(Self {
            writer,
            video_stream,
            audio_stream,
            size: (width, height),
            frame_rate: config.frame_rate,
            matrix,
            audio: config.audio,
            finished: false,
        })
    }

    /// Wrap `bytes` in an `IMFSample` stamped `pts_hns`/`duration_hns` and
    /// hand it to stream `stream`.
    fn write_sample(
        &self,
        stream: u32,
        bytes: &[u8],
        pts_hns: i64,
        duration_hns: i64,
    ) -> Result<(), EncodeError> {
        let len = u32::try_from(bytes.len())
            .map_err(|_| EncodeError::Encode("sample exceeds u32 bytes".into()))?;
        let buffer = unsafe { MFCreateMemoryBuffer(len) }.map_err(encode_err)?;

        let mut base: *mut u8 = core::ptr::null_mut();
        let mut max_len: u32 = 0;
        unsafe { buffer.Lock(&mut base, Some(&mut max_len), None) }.map_err(encode_err)?;
        if base.is_null() || max_len < len {
            unsafe {
                let _ = buffer.Unlock();
            }
            return Err(EncodeError::Encode("media buffer lock failed".into()));
        }
        // SAFETY: the buffer holds `max_len ≥ len` writable bytes while locked.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), base, bytes.len());
            buffer.Unlock().map_err(encode_err)?;
            buffer.SetCurrentLength(len).map_err(encode_err)?;
        }

        let sample = unsafe { MFCreateSample() }.map_err(encode_err)?;
        unsafe {
            sample.AddBuffer(&buffer).map_err(encode_err)?;
            sample.SetSampleTime(pts_hns).map_err(encode_err)?;
            sample.SetSampleDuration(duration_hns).map_err(encode_err)?;
            self.writer
                .WriteSample(stream, &sample)
                .map_err(encode_err)?;
        }
        Ok(())
    }
}

impl VideoEncoder for WmfEncoder {
    fn push(&mut self, frame: &VideoFrame) -> Result<(), EncodeError> {
        let (width, height) = self.size;
        if (frame.width(), frame.height()) != (width, height) {
            return Err(EncodeError::Encode(format!(
                "frame {}x{} does not match output {width}x{height}",
                frame.width(),
                frame.height()
            )));
        }
        if frame.format != PixelFormat::Rgba8 {
            return Err(EncodeError::unsupported(format!(
                "encoder expects Rgba8 frames, got {:?}",
                frame.format
            )));
        }
        let cpu = frame
            .cpu()
            .ok_or_else(|| EncodeError::unsupported("encoder needs CPU frames"))?;
        let plane = cpu
            .planes
            .first()
            .ok_or_else(|| EncodeError::Encode("frame has no plane".into()))?;
        if plane.data.len() < plane.stride * height as usize {
            return Err(EncodeError::Encode("frame plane is too small".into()));
        }

        let nv12 = rgba_to_nv12(&plane.data, width, height, plane.stride, self.matrix);
        let pts = to_hns(frame.pts);
        let duration = frame_duration_hns(self.frame_rate);
        self.write_sample(self.video_stream, &nv12, pts, duration)
    }

    fn push_audio(&mut self, samples: &[f32], pts: RationalTime) -> Result<(), EncodeError> {
        let (Some(stream), Some(cfg)) = (self.audio_stream, self.audio) else {
            return Err(EncodeError::unsupported("this encoder has no audio track"));
        };
        if samples.is_empty() {
            return Ok(());
        }
        let channels = usize::from(cfg.channels);
        if channels == 0 || samples.len() % channels != 0 {
            return Err(EncodeError::Encode(
                "audio block length is not a frame multiple".into(),
            ));
        }

        // f32 → interleaved 16-bit PCM, the AAC encoder MFT's input format.
        let mut pcm = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            pcm.extend_from_slice(&v.to_le_bytes());
        }

        let frames = (samples.len() / channels) as i64;
        let duration = HNS_PER_SEC * frames / i64::from(cfg.sample_rate.max(1));
        self.write_sample(stream, &pcm, to_hns(pts), duration)
    }

    fn finish(&mut self) -> Result<(), EncodeError> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        unsafe { self.writer.Finalize() }
            .map_err(|e| EncodeError::Encode(format!("Finalize failed: {e}")))
    }
}

/// Configure the writer's AAC output stream and its 16-bit PCM input type,
/// returning the stream index.
fn add_audio_stream(writer: &IMFSinkWriter, audio: AudioEncoderConfig) -> Result<u32, EncodeError> {
    if audio.channels == 0 || audio.sample_rate == 0 {
        return Err(EncodeError::unsupported("invalid audio config"));
    }
    let audio_out = unsafe { MFCreateMediaType() }.map_err(start_err)?;
    unsafe {
        audio_out
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)
            .map_err(start_err)?;
        audio_out
            .SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_AAC)
            .map_err(start_err)?;
        audio_out
            .SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, audio.sample_rate)
            .map_err(start_err)?;
        audio_out
            .SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, u32::from(audio.channels))
            .map_err(start_err)?;
        audio_out
            .SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, 16)
            .map_err(start_err)?;
        audio_out
            .SetUINT32(&MF_MT_AUDIO_AVG_BYTES_PER_SECOND, AAC_BYTES_PER_SECOND)
            .map_err(start_err)?;
    }
    let stream = unsafe { writer.AddStream(&audio_out) }.map_err(start_err)?;

    let block_align = u32::from(audio.channels) * 2;
    let audio_in = unsafe { MFCreateMediaType() }.map_err(start_err)?;
    unsafe {
        audio_in
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)
            .map_err(start_err)?;
        audio_in
            .SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_PCM)
            .map_err(start_err)?;
        audio_in
            .SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, audio.sample_rate)
            .map_err(start_err)?;
        audio_in
            .SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, u32::from(audio.channels))
            .map_err(start_err)?;
        audio_in
            .SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, 16)
            .map_err(start_err)?;
        audio_in
            .SetUINT32(&MF_MT_AUDIO_BLOCK_ALIGNMENT, block_align)
            .map_err(start_err)?;
        audio_in
            .SetUINT32(
                &MF_MT_AUDIO_AVG_BYTES_PER_SECOND,
                audio.sample_rate * block_align,
            )
            .map_err(start_err)?;
        audio_in
            .SetUINT32(&MF_MT_ALL_SAMPLES_INDEPENDENT, 1)
            .map_err(start_err)?;
        writer
            .SetInputMediaType(stream, &audio_in, None)
            .map_err(start_err)?;
    }
    Ok(stream)
}

/// Tag `media_type` with the colorimetry matching our CPU RGB→YUV conversion
/// (limited range, BT.709 transfer, matrix/primaries per `matrix`), so the
/// H.264 stream signals in VUI what the samples actually contain and decoders
/// don't have to guess.
fn set_color_tags(media_type: &IMFMediaType, matrix: YuvMatrix) -> windows::core::Result<()> {
    let (mf_matrix, mf_primaries) = match matrix {
        YuvMatrix::Bt709 => (MFVideoTransferMatrix_BT709, MFVideoPrimaries_BT709),
        YuvMatrix::Bt601 => (MFVideoTransferMatrix_BT601, MFVideoPrimaries_SMPTE170M),
    };
    unsafe {
        media_type.SetUINT32(&MF_MT_YUV_MATRIX, mf_matrix.0 as u32)?;
        media_type.SetUINT32(&MF_MT_VIDEO_PRIMARIES, mf_primaries.0 as u32)?;
        media_type.SetUINT32(&MF_MT_TRANSFER_FUNCTION, MFVideoTransFunc_709.0 as u32)?;
        media_type.SetUINT32(&MF_MT_VIDEO_NOMINAL_RANGE, MFNominalRange_16_235.0 as u32)?;
    }
    Ok(())
}

/// Start the Media Foundation platform once per process; mirrors the
/// decoder's policy (no matching `MFShutdown` — the platform lives for the
/// process).
fn ensure_mf_platform() {
    static MF_PLATFORM: Once = Once::new();
    MF_PLATFORM.call_once(|| unsafe {
        let _ = MFStartup(MF_VERSION, MFSTARTUP_FULL);
    });
}

/// Two `u32`s packed high/low into a `u64` — Media Foundation's encoding for
/// `MF_MT_FRAME_SIZE` (width, height) and `MF_MT_FRAME_RATE` (num, den).
fn pack_u32_pair(high: u32, low: u32) -> u64 {
    (u64::from(high) << 32) | u64::from(low)
}

/// [`RationalTime`] → 100 ns units, exact in `i128` before the cast back.
fn to_hns(t: RationalTime) -> i64 {
    if t.rate.num <= 0 || t.rate.den <= 0 {
        return 0;
    }
    let hns = i128::from(t.value) * i128::from(HNS_PER_SEC) * i128::from(t.rate.den)
        / i128::from(t.rate.num);
    hns as i64
}

/// One frame period at `rate`, in 100 ns units.
fn frame_duration_hns(rate: Rational) -> i64 {
    if rate.num <= 0 || rate.den <= 0 {
        return 0;
    }
    HNS_PER_SEC * i64::from(rate.den) / i64::from(rate.num)
}

fn start_err(e: windows::core::Error) -> EncodeError {
    EncodeError::Start(e.to_string())
}

fn encode_err(e: windows::core::Error) -> EncodeError {
    EncodeError::Encode(e.to_string())
}
