//! Windows video decoding via Media Foundation's **Source Reader**.
//!
//! [`IMFSourceReader`] is the Windows analogue of Apple's `AVAssetReader`: one
//! object that demuxes the container *and* drives the hardware (or software)
//! decoder, so we feed it a URL and pull decoded samples. Control stays in Rust
//! behind [`cutlass_core::VideoDecoder`]; only the codec call is native.
//!
//! ## What this backend does today
//!
//! - Attaches the process's **D3D11 device manager** so the reader loads the
//!   GPU vendor's hardware decoder MFT (DXVA); machines where that fails
//!   (VMs, RDP, driver trouble) fall back to Microsoft's software decoders
//!   transparently ([`crate::wmf_d3d`]).
//! - Forces an **uncompressed NV12** output type, which makes the reader insert
//!   a decoder (+ a converter when the codec is 10-bit).
//! - **CPU output**: reads each sample's [`IMFMediaBuffer`] stride-aware (via
//!   [`IMF2DBuffer`] when available, falling back to a flat `Lock`) into
//!   [`FrameData::Cpu`] — for DXGI-backed samples the lock performs the
//!   staging readback internally.
//! - **GPU output** ([`crate::OutputMode::Gpu`], best effort): exports the
//!   sample's decoder-owned texture slice through a shared NV12 texture pool +
//!   shared fence ([`crate::wmf_gpu`]) as [`FrameData::Gpu`] — no CPU pixel
//!   copy. Sources/machines that can't (software path, pre-fence OS) degrade
//!   to CPU planes rather than failing.
//! - Probes size / rotation / frame-rate / colorimetry and the clean-aperture
//!   crop (`MF_MT_MINIMUM_DISPLAY_APERTURE` — hardware decoders report
//!   macroblock-padded coded sizes) from the negotiated output
//!   [`IMFMediaType`], duration from the presentation descriptor
//!   (`MF_PD_DURATION`) with a fragmented-MP4 fallback
//!   ([`crate::mp4_duration`]), and seeks with `SetCurrentPosition` (100-ns
//!   units).
//! - Answers [`VideoDecoder::frame_at`] with the shared roll-forward policy
//!   ([`crate::seek`]) so sequential playback / forward scrubbing never pays a
//!   seek-plus-GOP-re-decode per frame.
//!
//! ## 10-bit / HDR (P010)
//!
//! P010 passthrough is **documented-but-off**: the compositor has no 16-bit
//! YUV sampling path yet, so the reader keeps forcing NV12 and 10-bit sources
//! decode through MF's converter (a quality, not correctness, limitation).
//! When the compositor grows P010 planes this backend only needs the forced
//! subtype relaxed.
//!
//! ## Measured (4K H.264 3840×2160@29.97, Intel UHD 630, Win11)
//!
//! `decode_bench` / `composite_bench` means over 150 frames:
//!
//! | path | per frame | eff fps |
//! |---|---|---|
//! | software decode → CPU planes | 12.2 ms | 71 |
//! | hardware decode → CPU readback | 19.4 ms | 49 |
//! | hardware decode → GPU surface (zero-copy) | 5.0 ms | 199 |
//! | decode + composite + readback, CPU path | 41.1 ms | 23 |
//! | decode + composite + readback, zero-copy | 12.9 ms | 74 |
//!
//! Hardware decode with CPU *readback* loses to software on mean latency
//! (the 12 MB/frame NV12 copy over PCIe dominates) but wins on p95 and CPU
//! load; the zero-copy path removes the readback and the re-upload, which is
//! where the 3×+ end-to-end win comes from. Sequential `frame_at` overhead on
//! top of raw decode is ~0.9×–1.1×; seek-per-frame is ~80× slower than the
//! roll-forward path it replaced (see [`crate::seek`]).
//!
//! **`MF_LOW_LATENCY` was measured and rejected.** Setting it on the reader
//! (same clip/machine, `decode_bench` over 150 frames) improved only the
//! first frame after a cold open (CPU 109→84 ms, GPU 38→32 ms) while
//! regressing everything sustained: sequential decode 20.4→24.3 ms mean CPU
//! (−19%) and 5.75→6.03 ms GPU, seek-per-frame 1 545→1 638 ms, random scrub
//! 2 363→2 539 ms. The flag caps the decoder MFT's internal pipelining, and
//! seek cost here *is* prefix-decode throughput (long GOPs), so shrinking
//! the queue makes seeks slower, not faster.
//!
//! ## Time base
//!
//! Media Foundation timestamps are in **100-nanosecond units** (`hns`), so every
//! frame's PTS rate — and [`SourceInfo::time_base`] — is [`HNS_PER_SEC`].

use core::slice;
use std::path::Path;
use std::sync::{Arc, Once};

use windows::Win32::Graphics::Direct3D11::ID3D11Texture2D;
use windows::Win32::Media::MediaFoundation::{
    IMF2DBuffer, IMFAttributes, IMFByteStream, IMFDXGIBuffer, IMFMediaBuffer, IMFMediaType,
    IMFSample, IMFSourceReader, MF_ACCESSMODE_READ, MF_E_INVALIDMEDIATYPE,
    MF_E_TOPO_CODEC_NOT_FOUND, MF_E_UNSUPPORTED_BYTESTREAM_TYPE, MF_E_UNSUPPORTED_D3D_TYPE,
    MF_FILEFLAGS_NONE, MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE, MF_MT_MAJOR_TYPE,
    MF_MT_MINIMUM_DISPLAY_APERTURE, MF_MT_SUBTYPE, MF_MT_TRANSFER_FUNCTION,
    MF_MT_VIDEO_NOMINAL_RANGE, MF_MT_VIDEO_PRIMARIES, MF_MT_VIDEO_ROTATION, MF_MT_YUV_MATRIX,
    MF_OPENMODE_FAIL_IF_NOT_EXIST, MF_PD_DURATION, MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS,
    MF_SOURCE_READER_ALL_STREAMS, MF_SOURCE_READER_D3D_MANAGER,
    MF_SOURCE_READER_FIRST_AUDIO_STREAM, MF_SOURCE_READER_FIRST_VIDEO_STREAM,
    MF_SOURCE_READER_MEDIASOURCE, MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED,
    MF_SOURCE_READERF_ENDOFSTREAM, MF_SOURCE_READERF_ERROR, MF_VERSION, MFCreateAttributes,
    MFCreateFile, MFCreateMediaType, MFCreateSourceReaderFromByteStream,
    MFCreateSourceReaderFromURL, MFMediaType_Video, MFNominalRange_0_255, MFSTARTUP_FULL,
    MFStartup, MFVideoArea, MFVideoFormat_NV12, MFVideoFormat_P010, MFVideoPrimaries_BT470_2_SysBG,
    MFVideoPrimaries_BT709, MFVideoPrimaries_BT2020, MFVideoPrimaries_DCI_P3,
    MFVideoPrimaries_SMPTE170M, MFVideoTransFunc_709, MFVideoTransFunc_2084, MFVideoTransFunc_HLG,
    MFVideoTransFunc_sRGB, MFVideoTransferMatrix_BT601, MFVideoTransferMatrix_BT709,
    MFVideoTransferMatrix_BT2020_10, MFVideoTransferMatrix_BT2020_12,
};
use windows::Win32::System::Com::StructuredStorage::PROPVARIANT;
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx};
use windows::core::{GUID, HSTRING, Interface};

use cutlass_core::{
    ColorPrimaries, ColorRange, ColorSpace, CpuImage, DecodeError, FrameData, GpuSurface,
    GpuSurfaceKind, MatrixCoefficients, PixelFormat, Plane, Rational, RationalTime, Rect, Rotation,
    SourceInfo, TransferFunction, VideoDecoder, VideoFrame,
};

use crate::OutputMode;
use crate::wmf_d3d::D3dShared;
use crate::wmf_gpu::SharedNv12Pool;

/// Media Foundation's timestamp tick rate: 100-nanosecond units per second.
pub(crate) const HNS_PER_SEC: i64 = 10_000_000;

/// Upper bound on consecutive empty `ReadSample` returns (stream ticks / gaps)
/// before we give up, so a pathological source can't hang the decode worker.
pub(crate) const MAX_EMPTY_READS: usize = 1024;

/// A pull-based decoder over one video file, backed by Media Foundation.
pub struct WmfDecoder {
    reader: IMFSourceReader,
    /// The magic "first video stream" selector, reused for every reader call.
    stream: u32,
    info: SourceInfo,
    /// Visible region within the coded frame (the clean aperture); the
    /// decoder pads coded sizes to macroblock multiples (e.g. 1080 → 1088).
    visible: Rect,
    mode: OutputMode,
    /// The shared decode device when this reader runs the hardware (DXVA)
    /// path; `None` means software MFTs deliver system-memory samples.
    d3d: Option<&'static D3dShared>,
    /// Shared-texture pool for `Gpu` mode, built on first GPU frame (and
    /// rebuilt if the coded size renegotiates mid-stream).
    pool: Option<Arc<SharedNv12Pool>>,
    /// Set after the first failed zero-copy export: this source delivers CPU
    /// planes from then on instead of re-attempting per frame.
    gpu_unavailable: bool,
    eos: bool,
    /// PTS of the last emitted frame — the [`crate::seek`] roll-forward anchor.
    /// Cleared on [`VideoDecoder::seek`] (decode position moved).
    last_pts: Option<RationalTime>,
    /// Observed seek/decode costs driving the adaptive roll window.
    seek_stats: crate::seek::SeekStats,
}

// SAFETY: the Source Reader and its COM children are owned and touched by a
// single thread at a time — the decode-ahead worker the engine parks this
// decoder on. We open in the multithreaded apartment (see `open`) so the COM
// objects may legally travel between MTA worker threads, and we never share
// `&self` across threads, which satisfies `VideoDecoder: Send` without `Sync`.
unsafe impl Send for WmfDecoder {}

impl WmfDecoder {
    /// Open `path` and probe its first video track.
    ///
    /// Tries the **hardware decode** path first: the reader gets the shared
    /// D3D11 device manager plus `MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS`,
    /// so it loads the GPU vendor's decoder MFT (DXVA) instead of Microsoft's
    /// software one. Any failure along that path — no usable GPU, a codec the
    /// hardware can't handle, a negotiation quirk — falls back to the plain
    /// software reader, which is the previous behavior verbatim.
    pub fn open(path: &Path, mode: OutputMode) -> Result<Self, DecodeError> {
        ensure_mf_platform();
        // The source resolver instantiates COM objects on the calling thread, so
        // it must have COM initialized. Join the MTA so the decoder can live on
        // any worker thread; best-effort — if the host already put this thread in
        // an STA, MF still works and `CoInitializeEx` just reports the conflict.
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        }

        let stream = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;

        let d3d = crate::wmf_d3d::shared();
        let hardware = d3d.and_then(|d3d| match configure_reader(path, stream, Some(d3d)) {
            Ok(reader) => Some(reader),
            Err(error) => {
                tracing::debug!(%error, "hardware decode path failed; retrying in software");
                None
            }
        });
        let (reader, d3d) = match hardware {
            Some(reader) => (reader, d3d),
            None => (configure_reader(path, stream, None)?, None),
        };

        let (info, visible) = probe(&reader, stream, Some(path))?;

        // One line per opened source: which decode path this file actually
        // got (DXVA hardware MFT vs Microsoft's software one) and the output
        // mode it will try first — the first thing to check when preview
        // renders run slow.
        tracing::info!(
            path = %path.display(),
            hardware = d3d.is_some(),
            mode = ?mode,
            width = info.coded_size.0,
            height = info.coded_size.1,
            "opened video decoder"
        );

        Ok(Self {
            reader,
            stream,
            info,
            visible,
            mode,
            d3d,
            pool: None,
            gpu_unavailable: false,
            eos: false,
            last_pts: None,
            seek_stats: Default::default(),
        })
    }

    /// Whether this decoder runs on the hardware (DXVA) path. Diagnostic —
    /// the output contract is identical either way.
    pub fn is_hardware_accelerated(&self) -> bool {
        self.d3d.is_some()
    }

    /// Whether the source also carries an audio track, queried from this
    /// decoder's own reader — so the probe doesn't open the file a second
    /// time just to answer it. Stream selection doesn't matter here:
    /// `GetNativeMediaType` reads the presentation descriptor.
    pub(crate) fn has_audio_track(&self) -> bool {
        unsafe {
            self.reader
                .GetNativeMediaType(MF_SOURCE_READER_FIRST_AUDIO_STREAM.0 as u32, 0)
                .is_ok()
        }
    }

    /// Turn one decoded [`IMFSample`] into a [`VideoFrame`].
    ///
    /// `Gpu` mode is **best effort**: when the hardware path is live, the
    /// sample's D3D11 texture slice is exported zero-copy (see
    /// [`crate::wmf_gpu`]); when it isn't — software decode fallback, a
    /// non-DXGI buffer, or a machine without shared-fence support — frames
    /// degrade to CPU planes, which every consumer of [`VideoFrame`] already
    /// handles. Renderer-side import failures still trigger its own
    /// permanent CPU fallback.
    fn sample_to_frame(
        &mut self,
        sample: &IMFSample,
        pts: RationalTime,
    ) -> Result<VideoFrame, DecodeError> {
        if self.mode == OutputMode::Gpu && !self.gpu_unavailable {
            match self.gpu_frame(sample, pts) {
                Ok(frame) => return Ok(frame),
                Err(error) => {
                    // Structural, not per-frame: stop retrying for this source.
                    self.gpu_unavailable = true;
                    tracing::info!(
                        %error,
                        "zero-copy GPU output unavailable; delivering CPU planes"
                    );
                }
            }
        }

        // Flatten any multi-buffer sample into one contiguous buffer.
        let buffer = unsafe { sample.ConvertToContiguousBuffer() }.map_err(decode_err)?;
        let (width, height) = self.info.coded_size;
        let planes = read_planes(&buffer, self.info.pixel_format, width, height)?;
        Ok(VideoFrame::new(
            pts,
            self.info.pixel_format,
            self.info.color,
            (width, height),
            self.visible,
            self.info.rotation,
            FrameData::Cpu(CpuImage::new(planes)),
        ))
    }

    /// The zero-copy arm: copy the sample's decoder-owned texture slice into
    /// a shared pool texture and wrap the NT handles as a [`GpuSurface`].
    fn gpu_frame(
        &mut self,
        sample: &IMFSample,
        pts: RationalTime,
    ) -> Result<VideoFrame, DecodeError> {
        let d3d = self
            .d3d
            .ok_or_else(|| DecodeError::unsupported("decoder is on the software path"))?;

        // Hardware decoders put the frame in an `IMFDXGIBuffer` wrapping one
        // slice of their internal texture array.
        let buffer = unsafe { sample.GetBufferByIndex(0) }.map_err(decode_err)?;
        let dxgi: IMFDXGIBuffer = buffer
            .cast()
            .map_err(|_| DecodeError::unsupported("sample buffer is not DXGI-backed"))?;
        let mut resource: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: valid DXGI buffer; asks for the ID3D11Texture2D interface.
        unsafe { dxgi.GetResource(&ID3D11Texture2D::IID, &mut resource) }.map_err(decode_err)?;
        // SAFETY: on success `resource` is a +1 ID3D11Texture2D pointer.
        let texture = unsafe { ID3D11Texture2D::from_raw(resource) };
        let subresource = unsafe { dxgi.GetSubresourceIndex() }.map_err(decode_err)?;

        // (Re)build the pool on first use and when the coded size changes
        // mid-stream; in-flight frames keep the old pool alive via their Arcs.
        let coded = self.info.coded_size;
        if self.pool.as_ref().is_none_or(|pool| pool.size() != coded) {
            self.pool = Some(Arc::new(SharedNv12Pool::new(d3d, coded)?));
        }
        let pool = self.pool.as_ref().expect("pool ensured above");

        // Box first, then point: the payload's address must be its final one
        // before it goes out via `handle`.
        let exported = Box::new(pool.export_frame(&texture, subresource)?);
        let handle = &exported.payload as *const cutlass_core::D3d11SharedSurface as u64;
        let surface = GpuSurface::new(GpuSurfaceKind::D3D11Texture2D, handle, Some(exported));

        Ok(VideoFrame::new(
            pts,
            self.info.pixel_format,
            self.info.color,
            coded,
            self.visible,
            self.info.rotation,
            FrameData::Gpu(surface),
        ))
    }
}

impl VideoDecoder for WmfDecoder {
    fn info(&self) -> &SourceInfo {
        &self.info
    }

    fn seek(&mut self, target: RationalTime) -> Result<(), DecodeError> {
        // `GUID_NULL` selects the 100-ns time format; the position rides in a
        // `VT_I8` PROPVARIANT. The reader lands at the keyframe at/before the
        // target and flushes, matching the trait contract.
        let position = PROPVARIANT::from(to_hns(target));
        let time_format = GUID::from_u128(0);
        unsafe { self.reader.SetCurrentPosition(&time_format, &position) }.map_err(seek_err)?;
        self.eos = false;
        self.last_pts = None;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<VideoFrame>, DecodeError> {
        if self.eos {
            return Ok(None);
        }

        for _ in 0..MAX_EMPTY_READS {
            let mut stream_flags: u32 = 0;
            let mut timestamp: i64 = 0;
            let mut sample: Option<IMFSample> = None;
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

            // A fatal stream error (bitstream corruption, a decoder failure
            // mid-file). The reader is done producing frames; surface it as a
            // decode error rather than draining the read budget on it.
            if stream_flags & MF_SOURCE_READERF_ERROR.0 as u32 != 0 {
                self.eos = true;
                return Err(DecodeError::Decode(
                    "source reader reported an unrecoverable stream error".into(),
                ));
            }

            // The decoder renegotiated (resolution / format / color changed
            // mid-stream); re-probe so frames after the change are described
            // correctly. The duration is already known, so no path is needed.
            if stream_flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED.0 as u32 != 0 {
                let duration = self.info.duration;
                let (info, visible) = probe(&self.reader, self.stream, None)?;
                self.info = info;
                self.visible = visible;
                self.info.duration = self.info.duration.or(duration);
            }

            if stream_flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0 {
                self.eos = true;
            }

            match sample {
                Some(sample) => {
                    let pts = RationalTime::new(timestamp, self.info.time_base);
                    let frame = self.sample_to_frame(&sample, pts)?;
                    self.last_pts = Some(pts);
                    return Ok(Some(frame));
                }
                // A stream tick or gap can return `S_OK` with no sample; keep
                // draining unless that empty read was also the end of stream.
                None if self.eos => return Ok(None),
                None => continue,
            }
        }

        Err(DecodeError::Decode(
            "source reader produced no frame within the read budget".into(),
        ))
    }

    fn frame_at(&mut self, target: RationalTime) -> Result<Option<VideoFrame>, DecodeError> {
        let last_pts = self.last_pts;
        // Copied out and written back: `frame_at_rolling` borrows `self` for
        // the decode walk, so it can't also borrow the field.
        let mut stats = self.seek_stats;
        let result = crate::seek::frame_at_rolling(self, last_pts, &mut stats, target);
        self.seek_stats = stats;
        result
    }

    fn frame_at_nearest(
        &mut self,
        target: RationalTime,
    ) -> Result<Option<VideoFrame>, DecodeError> {
        let last_pts = self.last_pts;
        let mut stats = self.seek_stats;
        let result = crate::seek::frame_at_nearest_rolling(self, last_pts, &mut stats, target);
        self.seek_stats = stats;
        result
    }
}

/// Start the Media Foundation platform once per process. The matching
/// `MFShutdown` is intentionally skipped: the platform lives for the process
/// (like a global logger), and decoders open/close freely against it.
pub(crate) fn ensure_mf_platform() {
    static MF_PLATFORM: Once = Once::new();
    MF_PLATFORM.call_once(|| unsafe {
        let _ = MFStartup(MF_VERSION, MFSTARTUP_FULL);
    });
}

/// Duration of the first audio track of `path`, for the audio-only probe
/// fallback (music/voiceover files with no video stream). `None` when the
/// file has no audio track or reports no duration.
pub(crate) fn audio_track_duration(path: &Path) -> Option<RationalTime> {
    ensure_mf_platform();
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
    }
    let reader = open_source_reader(path, None).ok()?;
    unsafe {
        reader
            .GetNativeMediaType(MF_SOURCE_READER_FIRST_AUDIO_STREAM.0 as u32, 0)
            .ok()?;
    }
    probe_duration(&reader, Some(path))
}

/// Open a Media Foundation source reader for the local file at `path`,
/// optionally with reader-configuration `attributes`.
///
/// Prefers a **self-opened byte stream** over [`MFCreateSourceReaderFromURL`].
/// Once the process brings up a Direct3D device (wgpu, in the desktop app),
/// MF's URL *source resolver* starts failing local-file opens with
/// `ERROR_BAD_NETPATH` (`0x80070035`) — it misroutes a plain `C:\…` path as if
/// it were a UNC/network path. Opening the file ourselves with [`MFCreateFile`]
/// and resolving the source from that byte stream sidesteps the URL resolver
/// entirely (plain file I/O is unaffected by the D3D init). The URL path stays
/// as a fallback for the pre-D3D case and anything the byte-stream resolver
/// can't handle (e.g. non-`file:` URLs, should we ever pass one).
pub(crate) fn open_source_reader(
    path: &Path,
    attributes: Option<&IMFAttributes>,
) -> Result<IMFSourceReader, DecodeError> {
    let url = path
        .to_str()
        .ok_or_else(|| DecodeError::Open("path is not valid UTF-8".into()))?;
    let url = HSTRING::from(url);

    match reader_from_byte_stream(&url, attributes) {
        Ok(reader) => Ok(reader),
        // `&HSTRING: Param<PCWSTR>`.
        Err(stream_err) => {
            unsafe { MFCreateSourceReaderFromURL(&url, attributes) }.map_err(|url_err| {
                // An unrecognized container is a capability gap, not an I/O
                // failure — tell the caller (and the user) which it is.
                if stream_err.code() == MF_E_UNSUPPORTED_BYTESTREAM_TYPE
                    || url_err.code() == MF_E_UNSUPPORTED_BYTESTREAM_TYPE
                {
                    return DecodeError::Unsupported(
                        "Media Foundation does not recognize this container format".into(),
                    );
                }
                DecodeError::Open(format!(
                    "byte-stream open failed ({stream_err}); URL open failed ({url_err})"
                ))
            })
        }
    }
}

/// Open `url` as a file byte stream and build a source reader from it, so the
/// source is resolved via `CreateObjectFromByteStream` rather than the URL
/// resolver. See [`open_source_reader`] for why this is preferred.
fn reader_from_byte_stream(
    url: &HSTRING,
    attributes: Option<&IMFAttributes>,
) -> Result<IMFSourceReader, windows::core::Error> {
    let stream: IMFByteStream = unsafe {
        MFCreateFile(
            MF_ACCESSMODE_READ,
            MF_OPENMODE_FAIL_IF_NOT_EXIST,
            MF_FILEFLAGS_NONE,
            url,
        )
    }?;
    unsafe { MFCreateSourceReaderFromByteStream(&stream, attributes) }
}

/// Reader-creation attributes for the hardware decode path: the shared D3D11
/// device (so decoder MFTs output DXGI textures) and permission to load
/// hardware transforms at all (off by default in the source reader).
///
/// `MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING` is deliberately *not*
/// set: it would splice in a video processor that auto-rotates and converts,
/// while rotation/colorimetry are already reported to (and applied by) the
/// renderer — the processor would double-apply them.
fn video_reader_attributes(d3d: &D3dShared) -> Result<IMFAttributes, windows::core::Error> {
    let mut attributes: Option<IMFAttributes> = None;
    unsafe { MFCreateAttributes(&mut attributes, 2) }?;
    let attributes = attributes.expect("MFCreateAttributes succeeded");
    unsafe {
        attributes.SetUnknown(&MF_SOURCE_READER_D3D_MANAGER, &d3d.manager)?;
        attributes.SetUINT32(&MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, 1)?;
    }
    Ok(attributes)
}

/// Open a source reader for `path` and negotiate NV12 on the video stream —
/// in hardware when `d3d` is given, in software otherwise.
fn configure_reader(
    path: &Path,
    stream: u32,
    d3d: Option<&D3dShared>,
) -> Result<IMFSourceReader, DecodeError> {
    let attributes = match d3d {
        Some(d3d) => Some(video_reader_attributes(d3d).map_err(open_err)?),
        None => None,
    };
    let reader = open_source_reader(path, attributes.as_ref())?;

    // Decode only the video track (leave audio deselected so MF never spins
    // up an audio decoder for it).
    unsafe {
        reader
            .SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS.0 as u32, false)
            .map_err(open_err)?;
        reader.SetStreamSelection(stream, true).map_err(open_err)?;
    }

    // Request uncompressed NV12. Setting an uncompressed subtype is what
    // makes the reader insert the decoder (and a converter for 10-bit
    // sources); without it the reader would hand back compressed samples.
    let out_type = unsafe { MFCreateMediaType() }.map_err(open_err)?;
    unsafe {
        out_type
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
            .map_err(open_err)?;
        out_type
            .SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)
            .map_err(open_err)?;
        reader
            .SetCurrentMediaType(stream, None, &out_type)
            .map_err(negotiate_err)?;
    }
    Ok(reader)
}

/// Map an NV12-negotiation failure to a [`DecodeError::Unsupported`], turning
/// the common "this machine can't decode this file" HRESULTs into
/// plain-language messages instead of raw COM errors.
fn negotiate_err(error: windows::core::Error) -> DecodeError {
    let reason = match error.code() {
        MF_E_TOPO_CODEC_NOT_FOUND => "no decoder is installed for this codec".into(),
        MF_E_INVALIDMEDIATYPE => "the decoder cannot produce NV12 for this stream".into(),
        MF_E_UNSUPPORTED_D3D_TYPE => "the hardware decoder rejected this stream's format".into(),
        _ => format!("could not negotiate NV12 output: {error}"),
    };
    DecodeError::Unsupported(reason)
}

/// Source duration from the presentation descriptor (`MF_PD_DURATION`, a
/// `VT_UI8` count of 100-ns units), or `None` when the container doesn't
/// report one.
///
/// **Fragmented MP4s report no duration here.** MF's MPEG-4 media source only
/// surfaces `MF_PD_DURATION` from the `moov/mvhd` movie header; fragmented
/// files (common for downloaded/streamed footage) declare `0` there and carry
/// real timing in per-fragment `moof` boxes, so the attribute is simply
/// absent. For those, fall back to scanning the container's metadata boxes
/// ourselves ([`crate::mp4_duration`]) — without a duration the editor would
/// treat the source as zero-length and reject every clip placement.
pub(crate) fn probe_duration(
    reader: &IMFSourceReader,
    path: Option<&Path>,
) -> Option<RationalTime> {
    let from_source = unsafe {
        reader.GetPresentationAttribute(MF_SOURCE_READER_MEDIASOURCE.0 as u32, &MF_PD_DURATION)
    }
    .ok()
    .and_then(|value| u64::try_from(&value).ok())
    .filter(|hns| *hns > 0)
    .map(|hns| RationalTime::new(hns as i64, Rational::new(HNS_PER_SEC as i32, 1)));
    from_source.or_else(|| crate::mp4_duration::duration_from_file(path?))
}

/// Read static source properties from the reader's negotiated output type.
/// Returns the info plus the visible rect within the coded frame.
///
/// `path` (when available) feeds the fragmented-MP4 duration fallback; pass
/// `None` for mid-stream re-probes where the duration is already known.
fn probe(
    reader: &IMFSourceReader,
    stream: u32,
    path: Option<&Path>,
) -> Result<(SourceInfo, Rect), DecodeError> {
    let media_type = unsafe { reader.GetCurrentMediaType(stream) }.map_err(open_err)?;

    // FRAME_SIZE packs width in the high 32 bits, height in the low 32.
    let frame_size = unsafe { media_type.GetUINT64(&MF_MT_FRAME_SIZE) }.map_err(open_err)?;
    let width = (frame_size >> 32) as u32;
    let height = (frame_size & 0xffff_ffff) as u32;

    // The clean-aperture crop. Software decoders usually pre-crop FRAME_SIZE,
    // but hardware (DXVA) decoders report the macroblock-padded coded size
    // (e.g. 1920x1088) and describe the visible 1920x1080 here — without the
    // crop those padding rows would show as green/garbage edges.
    let visible = display_aperture(&media_type, width, height)
        .unwrap_or_else(|| Rect::from_size(width, height));

    // FRAME_RATE packs numerator in the high 32 bits, denominator in the low 32.
    let frame_rate = match unsafe { media_type.GetUINT64(&MF_MT_FRAME_RATE) } {
        Ok(packed) => {
            let num = (packed >> 32) as u32;
            let den = (packed & 0xffff_ffff) as u32;
            Rational::new(num.max(1) as i32, den.max(1) as i32)
        }
        Err(_) => Rational::FPS_30,
    };

    // VIDEO_ROTATION is stored as degrees clockwise (0/90/180/270).
    let rotation = unsafe { media_type.GetUINT32(&MF_MT_VIDEO_ROTATION) }
        .ok()
        .and_then(|deg| Rotation::from_degrees(deg as i32))
        .unwrap_or(Rotation::None);

    let pixel_format = subtype_to_format(unsafe { media_type.GetGUID(&MF_MT_SUBTYPE) }.ok());
    let color = color_from_type(&media_type);

    let info = SourceInfo {
        coded_size: (width, height),
        display_size: (visible.width, visible.height),
        rotation,
        pixel_format,
        color,
        frame_rate,
        time_base: Rational::new(HNS_PER_SEC as i32, 1),
        duration: probe_duration(reader, path),
    };
    Ok((info, visible))
}

/// Parse `MF_MT_MINIMUM_DISPLAY_APERTURE` into a [`Rect`], clamped to the
/// coded size. `None` when the attribute is absent, malformed, or degenerate
/// (callers then treat the full coded frame as visible).
fn display_aperture(media_type: &IMFMediaType, coded_w: u32, coded_h: u32) -> Option<Rect> {
    let mut area = MFVideoArea::default();
    unsafe {
        media_type.GetBlob(
            &MF_MT_MINIMUM_DISPLAY_APERTURE,
            core::slice::from_raw_parts_mut(
                (&raw mut area).cast::<u8>(),
                core::mem::size_of::<MFVideoArea>(),
            ),
            None,
        )
    }
    .ok()?;
    // MFOffset is a fixed-point value; only whole-pixel offsets make sense
    // for a crop, so the fractional part is ignored.
    let x = u32::try_from(area.OffsetX.value).ok()?.min(coded_w);
    let y = u32::try_from(area.OffsetY.value).ok()?.min(coded_h);
    let w = u32::try_from(area.Area.cx).ok()?.min(coded_w - x);
    let h = u32::try_from(area.Area.cy).ok()?.min(coded_h - y);
    let rect = Rect::new(x, y, w, h);
    (!rect.is_empty()).then_some(rect)
}

/// Pull colorimetry from the output type's color attributes, falling back to the
/// BT.709 SDR default for any attribute the source under-signals.
fn color_from_type(media_type: &IMFMediaType) -> ColorSpace {
    let mut color = ColorSpace::BT709;
    if let Ok(v) = unsafe { media_type.GetUINT32(&MF_MT_VIDEO_PRIMARIES) } {
        color.primaries = map_primaries(v);
    }
    if let Ok(v) = unsafe { media_type.GetUINT32(&MF_MT_TRANSFER_FUNCTION) } {
        color.transfer = map_transfer(v);
    }
    if let Ok(v) = unsafe { media_type.GetUINT32(&MF_MT_YUV_MATRIX) } {
        color.matrix = map_matrix(v);
    }
    if let Ok(v) = unsafe { media_type.GetUINT32(&MF_MT_VIDEO_NOMINAL_RANGE) } {
        color.range = map_range(v);
    }
    color
}

/// Copy a biplanar 4:2:0 sample (NV12 / P010) out of its media buffer.
///
/// Prefers [`IMF2DBuffer`] so we honor the decoder's real row pitch (which it
/// pads for alignment), and falls back to a flat `Lock` assuming an unpadded
/// `width * bytes-per-sample` stride.
fn read_planes(
    buffer: &IMFMediaBuffer,
    format: PixelFormat,
    width: u32,
    height: u32,
) -> Result<Vec<Plane>, DecodeError> {
    if let Ok(buffer_2d) = buffer.cast::<IMF2DBuffer>() {
        let mut scanline0: *mut u8 = core::ptr::null_mut();
        let mut pitch: i32 = 0;
        unsafe { buffer_2d.Lock2D(&mut scanline0, &mut pitch) }.map_err(decode_err)?;
        let planes = (|| -> Result<Vec<Plane>, DecodeError> {
            if scanline0.is_null() || pitch <= 0 {
                return Err(DecodeError::Decode(
                    "2D buffer returned a null base or non-positive pitch".into(),
                ));
            }
            // SAFETY: between Lock2D/Unlock2D the buffer guarantees `pitch` bytes
            // per row across the full biplanar extent.
            unsafe { copy_biplanar(scanline0, pitch as usize, height, None) }
        })();
        unsafe { buffer_2d.Unlock2D() }.map_err(decode_err)?;
        return planes;
    }

    let mut base: *mut u8 = core::ptr::null_mut();
    let mut max_len: u32 = 0;
    let mut cur_len: u32 = 0;
    unsafe { buffer.Lock(&mut base, Some(&mut max_len), Some(&mut cur_len)) }
        .map_err(decode_err)?;
    let planes = (|| -> Result<Vec<Plane>, DecodeError> {
        if base.is_null() {
            return Err(DecodeError::Decode("locked buffer is null".into()));
        }
        let stride = width as usize * format.bytes_per_sample();
        // SAFETY: the contiguous buffer holds `cur_len` valid bytes; copy_biplanar
        // verifies the biplanar extent fits before reading.
        unsafe { copy_biplanar(base, stride, height, Some(cur_len as usize)) }
    })();
    unsafe { buffer.Unlock() }.map_err(decode_err)?;
    planes
}

/// Copy `height` luma rows plus `ceil(height/2)` interleaved-chroma rows, each
/// `stride` bytes, out of a biplanar buffer starting at `base`.
///
/// # Safety
/// `base` must point at a readable region of at least `stride * (height +
/// ceil(height/2))` bytes (the caller holds the buffer lock). When `available`
/// is `Some`, that extent is checked against it first.
unsafe fn copy_biplanar(
    base: *mut u8,
    stride: usize,
    height: u32,
    available: Option<usize>,
) -> Result<Vec<Plane>, DecodeError> {
    let y_rows = height as usize;
    let chroma_rows = (height as usize).div_ceil(2);
    let needed = stride
        .checked_mul(y_rows + chroma_rows)
        .ok_or_else(|| DecodeError::Decode("plane size overflow".into()))?;
    if let Some(available) = available
        && available < needed
    {
        return Err(DecodeError::Decode(format!(
            "buffer too small for {height}-row biplanar frame: {available} < {needed} bytes"
        )));
    }

    let luma = unsafe { slice::from_raw_parts(base, stride * y_rows) }.to_vec();
    let chroma =
        unsafe { slice::from_raw_parts(base.add(stride * y_rows), stride * chroma_rows) }.to_vec();
    Ok(vec![
        Plane::new(luma, stride, y_rows),
        Plane::new(chroma, stride, chroma_rows),
    ])
}

/// Convert a [`RationalTime`] to Media Foundation's 100-ns ticks.
///
/// `seconds = value * den / num`, so `hns = value * den * 1e7 / num`. Computed in
/// `i128` to avoid overflow before truncating to the `i64` MF expects.
fn to_hns(target: RationalTime) -> i64 {
    let num = i128::from(target.rate.num.max(1));
    let den = i128::from(target.rate.den.max(1));
    ((i128::from(target.value) * den * i128::from(HNS_PER_SEC)) / num) as i64
}

/// Map an `MF_MT_SUBTYPE` GUID to our pixel format. We only ever request NV12,
/// but P010 is recognized so a future 10-bit request maps cleanly.
fn subtype_to_format(subtype: Option<GUID>) -> PixelFormat {
    match subtype {
        Some(guid) if guid == MFVideoFormat_P010 => PixelFormat::P010,
        _ => PixelFormat::Nv12,
    }
}

/// Map an `MFVideoPrimaries` code to [`ColorPrimaries`].
fn map_primaries(value: u32) -> ColorPrimaries {
    if value == MFVideoPrimaries_BT709.0 as u32 {
        ColorPrimaries::Bt709
    } else if value == MFVideoPrimaries_SMPTE170M.0 as u32 {
        ColorPrimaries::Bt601_525
    } else if value == MFVideoPrimaries_BT470_2_SysBG.0 as u32 {
        ColorPrimaries::Bt601_625
    } else if value == MFVideoPrimaries_BT2020.0 as u32 {
        ColorPrimaries::Bt2020
    } else if value == MFVideoPrimaries_DCI_P3.0 as u32 {
        ColorPrimaries::DisplayP3
    } else {
        ColorPrimaries::Unspecified
    }
}

/// Map an `MFVideoTransferFunction` code to [`TransferFunction`].
fn map_transfer(value: u32) -> TransferFunction {
    if value == MFVideoTransFunc_709.0 as u32 {
        TransferFunction::Bt709
    } else if value == MFVideoTransFunc_sRGB.0 as u32 {
        TransferFunction::Srgb
    } else if value == MFVideoTransFunc_2084.0 as u32 {
        TransferFunction::Pq
    } else if value == MFVideoTransFunc_HLG.0 as u32 {
        TransferFunction::Hlg
    } else {
        TransferFunction::Unspecified
    }
}

/// Map an `MFVideoTransferMatrix` code to [`MatrixCoefficients`].
fn map_matrix(value: u32) -> MatrixCoefficients {
    if value == MFVideoTransferMatrix_BT709.0 as u32 {
        MatrixCoefficients::Bt709
    } else if value == MFVideoTransferMatrix_BT601.0 as u32 {
        MatrixCoefficients::Bt601
    } else if value == MFVideoTransferMatrix_BT2020_10.0 as u32
        || value == MFVideoTransferMatrix_BT2020_12.0 as u32
    {
        MatrixCoefficients::Bt2020Ncl
    } else {
        MatrixCoefficients::Unspecified
    }
}

/// Map an `MFNominalRange` code to [`ColorRange`]. Only the explicit full-range
/// code (`0_255`) is full; limited and unknown both map to limited (the video
/// norm).
fn map_range(value: u32) -> ColorRange {
    if value == MFNominalRange_0_255.0 as u32 {
        ColorRange::Full
    } else {
        ColorRange::Limited
    }
}

pub(crate) fn open_err(error: windows::core::Error) -> DecodeError {
    DecodeError::Open(error.to_string())
}

pub(crate) fn decode_err(error: windows::core::Error) -> DecodeError {
    DecodeError::Decode(error.to_string())
}

pub(crate) fn seek_err(error: windows::core::Error) -> DecodeError {
    DecodeError::Seek(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_primaries_from_mf_codes() {
        assert_eq!(
            map_primaries(MFVideoPrimaries_BT709.0 as u32),
            ColorPrimaries::Bt709
        );
        assert_eq!(
            map_primaries(MFVideoPrimaries_SMPTE170M.0 as u32),
            ColorPrimaries::Bt601_525
        );
        assert_eq!(
            map_primaries(MFVideoPrimaries_BT470_2_SysBG.0 as u32),
            ColorPrimaries::Bt601_625
        );
        assert_eq!(
            map_primaries(MFVideoPrimaries_BT2020.0 as u32),
            ColorPrimaries::Bt2020
        );
        assert_eq!(
            map_primaries(MFVideoPrimaries_DCI_P3.0 as u32),
            ColorPrimaries::DisplayP3
        );
        assert_eq!(map_primaries(0), ColorPrimaries::Unspecified);
    }

    #[test]
    fn maps_transfer_from_mf_codes() {
        assert_eq!(
            map_transfer(MFVideoTransFunc_709.0 as u32),
            TransferFunction::Bt709
        );
        assert_eq!(
            map_transfer(MFVideoTransFunc_sRGB.0 as u32),
            TransferFunction::Srgb
        );
        assert_eq!(
            map_transfer(MFVideoTransFunc_2084.0 as u32),
            TransferFunction::Pq
        );
        assert_eq!(
            map_transfer(MFVideoTransFunc_HLG.0 as u32),
            TransferFunction::Hlg
        );
        assert_eq!(map_transfer(0), TransferFunction::Unspecified);
    }

    #[test]
    fn maps_matrix_from_mf_codes() {
        assert_eq!(
            map_matrix(MFVideoTransferMatrix_BT709.0 as u32),
            MatrixCoefficients::Bt709
        );
        assert_eq!(
            map_matrix(MFVideoTransferMatrix_BT601.0 as u32),
            MatrixCoefficients::Bt601
        );
        assert_eq!(
            map_matrix(MFVideoTransferMatrix_BT2020_10.0 as u32),
            MatrixCoefficients::Bt2020Ncl
        );
        assert_eq!(
            map_matrix(MFVideoTransferMatrix_BT2020_12.0 as u32),
            MatrixCoefficients::Bt2020Ncl
        );
        assert_eq!(map_matrix(0), MatrixCoefficients::Unspecified);
    }

    #[test]
    fn maps_range_from_mf_codes() {
        assert_eq!(map_range(MFNominalRange_0_255.0 as u32), ColorRange::Full);
        assert_eq!(map_range(2), ColorRange::Limited);
        assert_eq!(map_range(0), ColorRange::Limited);
    }

    #[test]
    fn subtype_maps_to_pixel_format() {
        assert_eq!(
            subtype_to_format(Some(MFVideoFormat_NV12)),
            PixelFormat::Nv12
        );
        assert_eq!(
            subtype_to_format(Some(MFVideoFormat_P010)),
            PixelFormat::P010
        );
        assert_eq!(subtype_to_format(None), PixelFormat::Nv12);
    }

    #[test]
    fn hns_conversion_matches_time_base() {
        // 3 ticks already on the 100-ns base is exactly 3 hns.
        let exact = RationalTime::new(3, Rational::new(HNS_PER_SEC as i32, 1));
        assert_eq!(to_hns(exact), 3);
        // Frame 30 at 30 fps is one second = 1e7 hns.
        let one_second = RationalTime::new(30, Rational::FPS_30);
        assert_eq!(to_hns(one_second), HNS_PER_SEC);
        // Frame 5 at 30 fps: 5/30 s -> 1_666_666 hns (truncated toward zero).
        let frame_5 = RationalTime::new(5, Rational::FPS_30);
        assert_eq!(to_hns(frame_5), 1_666_666);
    }
}
