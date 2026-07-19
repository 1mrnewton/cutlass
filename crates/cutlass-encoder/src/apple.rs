//! Apple backend: `AVAssetWriter` (VideoToolbox H.264 encode + mp4 mux) behind
//! [`cutlass_core::VideoEncoder`].
//!
//! Pushed frames arrive as packed RGBA/BGRA CPU planes (what the renderer's
//! export loop produces). Each is copied into a `CVPixelBuffer` (32-bit BGRA,
//! VideoToolbox's preferred input), appended to an
//! `AVAssetWriterInputPixelBufferAdaptor` at its presentation time, and the
//! writer encodes + muxes to the output `.mp4`.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::path::Path;
use std::ptr::{self, NonNull};

use objc2::AllocAnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_av_foundation::{
    AVAssetWriter, AVAssetWriterInput, AVAssetWriterInputPixelBufferAdaptor, AVAssetWriterStatus,
    AVFileTypeMPEG4, AVMediaTypeAudio, AVMediaTypeVideo, AVVideoCodecKey, AVVideoCodecTypeH264,
    AVVideoHeightKey, AVVideoWidthKey,
};
use objc2_core_foundation::CFRetained;
use objc2_core_media::{CMBlockBuffer, CMFormatDescription, CMSampleBuffer, CMTime, CMTimeFlags};
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferCreate, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow,
    CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
};
use objc2_foundation::{NSMutableDictionary, NSNumber, NSString, NSURL};

use cutlass_core::{
    AudioEncoderConfig, EncodeError, EncoderConfig, PixelFormat, RationalTime, VideoEncoder,
    VideoFrame,
};

/// CoreVideo FourCC for 32-bit BGRA, the pixel-buffer format we feed the writer.
const FOURCC_BGRA: u32 = u32::from_be_bytes(*b"BGRA");

/// AAC bitrate for the muxed audio track (128 kbps stereo — transparent enough
/// for an export MVP).
const AAC_BIT_RATE: i32 = 128_000;

/// An H.264 (+ optional AAC) → mp4 encoder backed by AVFoundation /
/// VideoToolbox.
///
/// With both a video and an audio input, `AVAssetWriter` gates each input's
/// readiness on its ideal interleaving pattern: it will hold the video input
/// not-ready while it waits for audio it hasn't received (and vice versa), and
/// appending to a not-ready input raises an ObjC exception. Since both inputs
/// are driven synchronously from one thread, blocking on readiness would
/// deadlock — so pushes land in per-track queues and [`Self::pump`] appends
/// to whichever input is ready.
pub struct AvfEncoder {
    writer: Retained<AVAssetWriter>,
    input: Retained<AVAssetWriterInput>,
    adaptor: Retained<AVAssetWriterInputPixelBufferAdaptor>,
    size: (u32, u32),
    /// Audio writer input + config, present iff `config.audio` was `Some`.
    audio_input: Option<Retained<AVAssetWriterInput>>,
    audio: Option<AudioEncoderConfig>,
    /// LPCM source format description for [`push_audio`](Self::push_audio),
    /// built once on first use (the writer encodes this to AAC).
    audio_format: Option<CFRetained<CMFormatDescription>>,
    /// Frames/blocks pushed but not yet accepted by their writer input.
    pending_video: VecDeque<(CFRetained<CVPixelBuffer>, CMTime)>,
    pending_audio: VecDeque<CFRetained<CMSampleBuffer>>,
    started: bool,
    finished: bool,
}

// SAFETY: the AVFoundation objects are owned and touched by a single thread at a
// time — the export worker the engine parks this encoder on. We never share
// `&self` across threads, satisfying `VideoEncoder: Send` without `Sync`.
unsafe impl Send for AvfEncoder {}

impl AvfEncoder {
    /// Create an mp4 writer at `path` for `config`'s size and frame rate.
    /// Overwrites any existing file (`AVAssetWriter` refuses to open one).
    pub fn open(path: &Path, config: EncoderConfig) -> Result<Self, EncodeError> {
        let (width, height) = config.size;
        if width == 0 || height == 0 {
            return Err(EncodeError::unsupported("zero output dimensions"));
        }
        let path_str = path
            .to_str()
            .ok_or_else(|| EncodeError::Start("path is not valid UTF-8".into()))?;
        let _ = std::fs::remove_file(path);

        let url = NSURL::fileURLWithPath(&NSString::from_str(path_str));
        let file_type = unsafe { AVFileTypeMPEG4 }
            .ok_or_else(|| EncodeError::Start("AVFileTypeMPEG4 unavailable".into()))?;
        let writer = unsafe {
            AVAssetWriter::initWithURL_fileType_error(AVAssetWriter::alloc(), &url, file_type)
        }
        .map_err(|e| EncodeError::Start(e.localizedDescription().to_string()))?;

        // Output settings: { codec: H.264, width, height }.
        let settings = NSMutableDictionary::<NSString, AnyObject>::new();
        let codec_key = unsafe { AVVideoCodecKey }.ok_or_else(missing("AVVideoCodecKey"))?;
        let codec_h264 =
            unsafe { AVVideoCodecTypeH264 }.ok_or_else(missing("AVVideoCodecTypeH264"))?;
        let width_key = unsafe { AVVideoWidthKey }.ok_or_else(missing("AVVideoWidthKey"))?;
        let height_key = unsafe { AVVideoHeightKey }.ok_or_else(missing("AVVideoHeightKey"))?;
        let width_num = NSNumber::numberWithInt(width as i32);
        let height_num = NSNumber::numberWithInt(height as i32);
        unsafe {
            settings.setObject_forKey(codec_h264, ProtocolObject::from_ref(codec_key));
            settings.setObject_forKey(&width_num, ProtocolObject::from_ref(width_key));
            settings.setObject_forKey(&height_num, ProtocolObject::from_ref(height_key));
        }

        let media_video = unsafe { AVMediaTypeVideo }.ok_or_else(missing("AVMediaTypeVideo"))?;
        let input = unsafe {
            AVAssetWriterInput::initWithMediaType_outputSettings(
                AVAssetWriterInput::alloc(),
                media_video,
                Some(&settings),
            )
        };
        // Export, not capture: let the writer pull as fast as we can supply.
        unsafe { input.setExpectsMediaDataInRealTime(false) };

        let adaptor = unsafe {
            AVAssetWriterInputPixelBufferAdaptor::initWithAssetWriterInput_sourcePixelBufferAttributes(
                AVAssetWriterInputPixelBufferAdaptor::alloc(),
                &input,
                None,
            )
        };

        if !unsafe { writer.canAddInput(&input) } {
            return Err(EncodeError::Start("writer rejects the video input".into()));
        }
        unsafe { writer.addInput(&input) };

        // Optional AAC audio track.
        let audio_input = match config.audio {
            Some(audio) => Some(build_audio_input(&writer, audio)?),
            None => None,
        };

        Ok(Self {
            writer,
            input,
            adaptor,
            size: (width, height),
            audio_input,
            audio: config.audio,
            audio_format: None,
            pending_video: VecDeque::new(),
            pending_audio: VecDeque::new(),
            started: false,
            finished: false,
        })
    }

    /// LPCM source format description for the pushed audio, built once.
    fn ensure_audio_format(
        &mut self,
        cfg: AudioEncoderConfig,
    ) -> Result<*mut CMFormatDescription, EncodeError> {
        if self.audio_format.is_none() {
            let bytes_per_frame = u32::from(cfg.channels) * 4;
            let asbd = AudioStreamBasicDescription {
                m_sample_rate: f64::from(cfg.sample_rate),
                m_format_id: K_AUDIO_FORMAT_LINEAR_PCM,
                m_format_flags: K_AUDIO_FORMAT_FLAG_IS_FLOAT | K_AUDIO_FORMAT_FLAG_IS_PACKED,
                m_bytes_per_packet: bytes_per_frame,
                m_frames_per_packet: 1,
                m_bytes_per_frame: bytes_per_frame,
                m_channels_per_frame: u32::from(cfg.channels),
                m_bits_per_channel: 32,
                m_reserved: 0,
            };
            let mut out: *mut CMFormatDescription = ptr::null_mut();
            let status = unsafe {
                CMAudioFormatDescriptionCreate(
                    ptr::null(),
                    &asbd,
                    0,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    &mut out,
                )
            };
            let out = NonNull::new(out).filter(|_| status == 0).ok_or_else(|| {
                EncodeError::Encode(format!("CMAudioFormatDescriptionCreate failed ({status})"))
            })?;
            self.audio_format = Some(unsafe { CFRetained::from_raw(out) });
        }
        Ok(CFRetained::as_ptr(self.audio_format.as_ref().unwrap()).as_ptr())
    }

    /// Begin a writing session on first use.
    fn ensure_started(&mut self) -> Result<(), EncodeError> {
        if self.started {
            return Ok(());
        }
        if !unsafe { self.writer.startWriting() } {
            return Err(self.writer_error("startWriting failed"));
        }
        unsafe { self.writer.startSessionAtSourceTime(cm_zero()) };
        self.started = true;
        Ok(())
    }

    /// Copy a packed RGBA/BGRA frame into a fresh BGRA `CVPixelBuffer`.
    fn make_pixel_buffer(
        &self,
        frame: &VideoFrame,
    ) -> Result<CFRetained<CVPixelBuffer>, EncodeError> {
        let (width, height) = self.size;
        if (frame.width(), frame.height()) != (width, height) {
            return Err(EncodeError::Encode(format!(
                "frame {}x{} does not match output {width}x{height}",
                frame.width(),
                frame.height()
            )));
        }
        let cpu = frame
            .cpu()
            .ok_or_else(|| EncodeError::unsupported("encoder needs CPU frames"))?;
        let plane = cpu
            .planes
            .first()
            .ok_or_else(|| EncodeError::Encode("frame has no plane".into()))?;
        let swizzle = match frame.format {
            PixelFormat::Rgba8 => true,
            PixelFormat::Bgra8 => false,
            other => {
                return Err(EncodeError::unsupported(format!(
                    "encoder expects Rgba8/Bgra8 frames, got {other:?}"
                )));
            }
        };
        let src_stride = plane.stride;
        if plane.data.len() < src_stride * height as usize {
            return Err(EncodeError::Encode("frame plane is too small".into()));
        }

        let mut out: *mut CVPixelBuffer = std::ptr::null_mut();
        let status = unsafe {
            CVPixelBufferCreate(
                None,
                width as usize,
                height as usize,
                FOURCC_BGRA,
                None,
                NonNull::from(&mut out),
            )
        };
        if status != 0 {
            return Err(EncodeError::Encode(format!(
                "CVPixelBufferCreate failed ({status})"
            )));
        }
        let pb = unsafe {
            CFRetained::from_raw(
                NonNull::new(out).ok_or_else(|| EncodeError::Encode("null pixel buffer".into()))?,
            )
        };

        unsafe {
            if CVPixelBufferLockBaseAddress(&pb, CVPixelBufferLockFlags(0)) != 0 {
                return Err(EncodeError::Encode(
                    "CVPixelBufferLockBaseAddress failed".into(),
                ));
            }
            let dst_stride = CVPixelBufferGetBytesPerRow(&pb);
            let base = CVPixelBufferGetBaseAddress(&pb) as *mut u8;
            if base.is_null() {
                CVPixelBufferUnlockBaseAddress(&pb, CVPixelBufferLockFlags(0));
                return Err(EncodeError::Encode("null pixel buffer base address".into()));
            }
            let src = plane.data.as_ptr();
            for y in 0..height as usize {
                let src_row = src.add(y * src_stride);
                let dst_row = base.add(y * dst_stride);
                for x in 0..width as usize {
                    let s = src_row.add(x * 4);
                    let d = dst_row.add(x * 4);
                    if swizzle {
                        *d = *s.add(2); // B <- R
                        *d.add(1) = *s.add(1); // G
                        *d.add(2) = *s; // R <- B
                        *d.add(3) = *s.add(3); // A
                    } else {
                        *d = *s;
                        *d.add(1) = *s.add(1);
                        *d.add(2) = *s.add(2);
                        *d.add(3) = *s.add(3);
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(&pb, CVPixelBufferLockFlags(0));
        }
        Ok(pb)
    }

    fn writer_error(&self, context: &str) -> EncodeError {
        let detail = unsafe { self.writer.error() }
            .map(|e| e.localizedDescription().to_string())
            .unwrap_or_else(|| context.to_string());
        EncodeError::Encode(format!("{context}: {detail}"))
    }

    /// Append queued media to every input that reports ready. Returns whether
    /// anything was accepted.
    fn pump_once(&mut self) -> Result<bool, EncodeError> {
        let mut progressed = false;
        while !self.pending_video.is_empty() && unsafe { self.input.isReadyForMoreMediaData() } {
            let (pb, pts) = self.pending_video.front().expect("non-empty");
            let adaptor = &self.adaptor;
            let ok = objc2::exception::catch(std::panic::AssertUnwindSafe(|| unsafe {
                adaptor.appendPixelBuffer_withPresentationTime(pb, *pts)
            }))
            .map_err(|ex| EncodeError::Encode(format!("appendPixelBuffer raised: {ex:?}")))?;
            if !ok {
                return Err(self.writer_error("appendPixelBuffer failed"));
            }
            self.pending_video.pop_front();
            progressed = true;
        }
        if let Some(audio_input) = &self.audio_input {
            while !self.pending_audio.is_empty() && unsafe { audio_input.isReadyForMoreMediaData() }
            {
                let sbuf = self.pending_audio.front().expect("non-empty");
                let ok = objc2::exception::catch(std::panic::AssertUnwindSafe(|| unsafe {
                    audio_input.appendSampleBuffer(sbuf)
                }))
                .map_err(|ex| EncodeError::Encode(format!("appendSampleBuffer raised: {ex:?}")))?;
                if !ok {
                    return Err(self.writer_error("appendSampleBuffer (audio) failed"));
                }
                self.pending_audio.pop_front();
                progressed = true;
            }
        }
        Ok(progressed)
    }

    /// Pump until the backlog is at most `max_backlog` items, sleeping between
    /// attempts. Errors if the writer fails or nothing moves for ~10 s.
    fn pump_until(&mut self, max_backlog: usize) -> Result<(), EncodeError> {
        let mut stalled_ms = 0u32;
        loop {
            let progressed = self.pump_once()?;
            if self.pending_video.len() + self.pending_audio.len() <= max_backlog {
                return Ok(());
            }
            if unsafe { self.writer.status() } == AVAssetWriterStatus::Failed {
                return Err(self.writer_error("writer failed"));
            }
            if progressed {
                stalled_ms = 0;
            } else {
                if stalled_ms >= 10_000 {
                    return Err(EncodeError::Encode(
                        "encoder made no progress for 10s".into(),
                    ));
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
                stalled_ms += 1;
            }
        }
    }
}

/// Hard cap on queued-but-unaccepted items (video frames + audio blocks): a
/// safety valve so a wedged writer surfaces an error instead of exhausting
/// memory. The writer's interleaving window keeps the steady-state backlog to
/// roughly a second of media, far below this.
const MAX_PENDING: usize = 600;

impl VideoEncoder for AvfEncoder {
    fn push(&mut self, frame: &VideoFrame) -> Result<(), EncodeError> {
        self.ensure_started()?;
        let pb = self.make_pixel_buffer(frame)?;
        let pts = rational_to_cmtime(frame.pts);
        self.pending_video.push_back((pb, pts));
        // Opportunistic drain only: blocking here could deadlock, because the
        // writer may be waiting for audio that the caller pushes after us.
        self.pump_once()?;
        if self.pending_video.len() + self.pending_audio.len() > MAX_PENDING {
            self.pump_until(MAX_PENDING)?;
        }
        Ok(())
    }

    fn push_audio(&mut self, samples: &[f32], pts: RationalTime) -> Result<(), EncodeError> {
        let cfg = self
            .audio
            .ok_or_else(|| EncodeError::unsupported("this encoder has no audio track"))?;
        if self.audio_input.is_none() {
            return Err(EncodeError::unsupported("this encoder has no audio track"));
        }
        if samples.is_empty() {
            return Ok(());
        }
        let channels = usize::from(cfg.channels);
        if channels == 0 || !samples.len().is_multiple_of(channels) {
            return Err(EncodeError::Encode(
                "audio block length is not a frame multiple".into(),
            ));
        }
        self.ensure_started()?;
        let format = self.ensure_audio_format(cfg)?;
        let num_frames = samples.len() / channels;
        let bytes_per_frame = channels * 4;
        let byte_len = samples.len() * 4;

        // Block buffer: allocate, then copy the interleaved f32 in.
        let mut block_ptr: *mut CMBlockBuffer = ptr::null_mut();
        let status = unsafe {
            CMBlockBufferCreateWithMemoryBlock(
                ptr::null(),
                ptr::null_mut(),
                byte_len,
                ptr::null(),
                ptr::null(),
                0,
                byte_len,
                K_CM_BLOCK_BUFFER_ASSURE_MEMORY_NOW,
                &mut block_ptr,
            )
        };
        let block = NonNull::new(block_ptr)
            .filter(|_| status == 0)
            .ok_or_else(|| EncodeError::Encode(format!("CMBlockBufferCreate failed ({status})")))?;
        let block: CFRetained<CMBlockBuffer> = unsafe { CFRetained::from_raw(block) };
        let status = unsafe {
            CMBlockBufferReplaceDataBytes(samples.as_ptr().cast(), block_ptr, 0, byte_len)
        };
        if status != 0 {
            return Err(EncodeError::Encode(format!(
                "CMBlockBufferReplaceDataBytes failed ({status})"
            )));
        }

        let timing = CMSampleTimingInfoRaw {
            duration: CMTime {
                value: 1,
                timescale: cfg.sample_rate as i32,
                flags: CMTimeFlags::Valid,
                epoch: 0,
            },
            presentation_time_stamp: rational_to_cmtime(pts),
            decode_time_stamp: cm_invalid(),
        };
        let sample_size = bytes_per_frame;
        let mut sbuf_ptr: *mut CMSampleBuffer = ptr::null_mut();
        let status = unsafe {
            CMSampleBufferCreate(
                ptr::null(),
                block_ptr,
                1,
                ptr::null(),
                ptr::null_mut(),
                format,
                num_frames as isize,
                1,
                &timing,
                1,
                &sample_size,
                &mut sbuf_ptr,
            )
        };
        let sbuf = NonNull::new(sbuf_ptr)
            .filter(|_| status == 0)
            .ok_or_else(|| {
                EncodeError::Encode(format!("CMSampleBufferCreate failed ({status})"))
            })?;
        let sbuf: CFRetained<CMSampleBuffer> = unsafe { CFRetained::from_raw(sbuf) };
        drop(block); // the sample buffer retains it now

        self.pending_audio.push_back(sbuf);
        self.pump_once()?;
        if self.pending_video.len() + self.pending_audio.len() > MAX_PENDING {
            self.pump_until(MAX_PENDING)?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), EncodeError> {
        if self.finished {
            return Ok(());
        }
        // Start a (possibly empty) session so finishWriting yields a valid file
        // even when no frames were pushed.
        self.ensure_started()?;
        // Drain both queues, marking each input finished the moment its queue
        // empties: the writer can hold one input not-ready while it waits for
        // more data on the *other*, and only end-of-stream releases it.
        let mut video_done = false;
        let mut audio_done = false;
        let mut stalled_ms = 0u32;
        while !(video_done && audio_done) {
            let progressed = self.pump_once()?;
            if !video_done && self.pending_video.is_empty() {
                unsafe { self.input.markAsFinished() };
                video_done = true;
            }
            if !audio_done && self.pending_audio.is_empty() {
                if let Some(audio_input) = &self.audio_input {
                    unsafe { audio_input.markAsFinished() };
                }
                audio_done = true;
            }
            if video_done && audio_done {
                break;
            }
            if unsafe { self.writer.status() } == AVAssetWriterStatus::Failed {
                return Err(self.writer_error("writer failed"));
            }
            if progressed {
                stalled_ms = 0;
            } else {
                if stalled_ms >= 10_000 {
                    return Err(EncodeError::Encode(
                        "encoder made no progress draining for 10s".into(),
                    ));
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
                stalled_ms += 1;
            }
        }
        #[allow(deprecated)] // synchronous finishWriting is fine on an export worker.
        let ok = unsafe { self.writer.finishWriting() };
        self.finished = true;
        if !ok || unsafe { self.writer.status() } != AVAssetWriterStatus::Completed {
            return Err(self.writer_error("finishWriting failed"));
        }
        Ok(())
    }
}

/// Build the AAC audio writer input for `audio` and add it to `writer`.
fn build_audio_input(
    writer: &AVAssetWriter,
    audio: AudioEncoderConfig,
) -> Result<Retained<AVAssetWriterInput>, EncodeError> {
    let settings = NSMutableDictionary::<NSString, AnyObject>::new();
    let set = |key: Option<&'static NSString>, value: &AnyObject, name: &'static str| {
        let key = key.ok_or_else(missing(name))?;
        unsafe { settings.setObject_forKey(value, ProtocolObject::from_ref(key)) };
        Ok::<(), EncodeError>(())
    };

    let format = NSNumber::numberWithUnsignedInt(K_AUDIO_FORMAT_MPEG4_AAC);
    let rate = NSNumber::numberWithDouble(f64::from(audio.sample_rate));
    let chans = NSNumber::numberWithInt(i32::from(audio.channels));
    let bitrate = NSNumber::numberWithInt(AAC_BIT_RATE);
    unsafe {
        set(AVFormatIDKey, &format, "AVFormatIDKey")?;
        set(AVSampleRateKey, &rate, "AVSampleRateKey")?;
        set(AVNumberOfChannelsKey, &chans, "AVNumberOfChannelsKey")?;
        set(AVEncoderBitRateKey, &bitrate, "AVEncoderBitRateKey")?;
    }

    let media_audio = unsafe { AVMediaTypeAudio }.ok_or_else(missing("AVMediaTypeAudio"))?;
    let input = unsafe {
        AVAssetWriterInput::initWithMediaType_outputSettings(
            AVAssetWriterInput::alloc(),
            media_audio,
            Some(&settings),
        )
    };
    unsafe { input.setExpectsMediaDataInRealTime(false) };
    if !unsafe { writer.canAddInput(&input) } {
        return Err(EncodeError::Start("writer rejects the audio input".into()));
    }
    unsafe { writer.addInput(&input) };
    Ok(input)
}

/// Build a `missing constant` error closure for `ok_or_else`.
fn missing(name: &'static str) -> impl Fn() -> EncodeError {
    move || EncodeError::Start(format!("{name} unavailable"))
}

/// An invalid `CMTime` (no `Valid` flag) — used for `decodeTimeStamp`, which
/// LPCM audio doesn't carry.
fn cm_invalid() -> CMTime {
    CMTime {
        value: 0,
        timescale: 0,
        flags: CMTimeFlags(0),
        epoch: 0,
    }
}

// CoreAudio `AudioFormatID` FourCCs and LPCM format flags.
const K_AUDIO_FORMAT_LINEAR_PCM: u32 = u32::from_be_bytes(*b"lpcm");
const K_AUDIO_FORMAT_MPEG4_AAC: u32 = u32::from_be_bytes(*b"aac ");
const K_AUDIO_FORMAT_FLAG_IS_FLOAT: u32 = 1 << 0;
const K_AUDIO_FORMAT_FLAG_IS_PACKED: u32 = 1 << 3;
const K_CM_BLOCK_BUFFER_ASSURE_MEMORY_NOW: u32 = 1 << 0;

/// CoreAudio stream description (matches the C `AudioStreamBasicDescription`
/// layout). Declared locally because the pinned `objc2-core-audio-types`
/// binding for it isn't reachable via the obfuscated CoreMedia wrappers.
#[repr(C)]
#[derive(Clone, Copy)]
struct AudioStreamBasicDescription {
    m_sample_rate: f64,
    m_format_id: u32,
    m_format_flags: u32,
    m_bytes_per_packet: u32,
    m_frames_per_packet: u32,
    m_bytes_per_frame: u32,
    m_channels_per_frame: u32,
    m_bits_per_channel: u32,
    m_reserved: u32,
}

/// CoreMedia per-sample timing (matches the C `CMSampleTimingInfo` layout).
#[repr(C)]
#[derive(Clone, Copy)]
struct CMSampleTimingInfoRaw {
    duration: CMTime,
    presentation_time_stamp: CMTime,
    decode_time_stamp: CMTime,
}

// The 0.3.2 `objc2-core-media` bindings obfuscate the audio
// sample/format-description constructors (e.g. `CMAudioFormatDescriptionCreate`
// translated to a method named `ln`), so we link the framework's C entry points
// directly. Their object outputs are wrapped back into the crate's CF types for
// retain/release.
#[link(name = "CoreMedia", kind = "framework")]
unsafe extern "C-unwind" {
    fn CMAudioFormatDescriptionCreate(
        allocator: *const c_void,
        asbd: *const AudioStreamBasicDescription,
        layout_size: usize,
        layout: *const c_void,
        magic_cookie_size: usize,
        magic_cookie: *const c_void,
        extensions: *const c_void,
        format_description_out: *mut *mut CMFormatDescription,
    ) -> i32;

    fn CMBlockBufferCreateWithMemoryBlock(
        structure_allocator: *const c_void,
        memory_block: *mut c_void,
        block_length: usize,
        block_allocator: *const c_void,
        custom_block_source: *const c_void,
        offset_to_data: usize,
        data_length: usize,
        flags: u32,
        block_buffer_out: *mut *mut CMBlockBuffer,
    ) -> i32;

    fn CMBlockBufferReplaceDataBytes(
        source_bytes: *const c_void,
        destination_buffer: *mut CMBlockBuffer,
        offset_into_destination: usize,
        data_length: usize,
    ) -> i32;

    fn CMSampleBufferCreate(
        allocator: *const c_void,
        data_buffer: *mut CMBlockBuffer,
        data_ready: u8,
        make_data_ready_callback: *const c_void,
        make_data_ready_refcon: *mut c_void,
        format_description: *mut CMFormatDescription,
        num_samples: isize,
        num_sample_timing_entries: isize,
        sample_timing_array: *const CMSampleTimingInfoRaw,
        num_sample_size_entries: isize,
        sample_size_array: *const usize,
        sample_buffer_out: *mut *mut CMSampleBuffer,
    ) -> i32;
}

// AVFoundation audio-settings dictionary keys (the pinned binding ships an empty
// `AVAudioSettings`); link the framework's `NSString *` constants directly.
#[link(name = "AVFoundation", kind = "framework")]
unsafe extern "C" {
    static AVFormatIDKey: Option<&'static NSString>;
    static AVSampleRateKey: Option<&'static NSString>;
    static AVNumberOfChannelsKey: Option<&'static NSString>;
    static AVEncoderBitRateKey: Option<&'static NSString>;
}

/// `CMTime` for source-time zero.
fn cm_zero() -> CMTime {
    CMTime {
        value: 0,
        timescale: 1,
        flags: CMTimeFlags::Valid,
        epoch: 0,
    }
}

/// [`RationalTime`] → `CMTime` (seconds preserved as value·den / num).
fn rational_to_cmtime(t: RationalTime) -> CMTime {
    CMTime {
        value: t.value * i64::from(t.rate.den.max(1)),
        timescale: t.rate.num.max(1),
        flags: CMTimeFlags::Valid,
        epoch: 0,
    }
}
