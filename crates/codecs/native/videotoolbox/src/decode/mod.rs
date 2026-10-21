//! One `VTDecompressionSession` path for every codec. The codec modules only
//! build the format description.

mod av1;
mod h264;
mod hevc;
mod mjpeg;
mod mpeg2;
mod mpeg4;
mod prores;
mod reorder;
mod vp9;

use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::sync::Arc;

use cutlass_core::{ColorRange, Packet, PixelFormat, VideoCodec, VideoFrame, VideoStreamConfig};
use objc2_core_foundation::{CFBoolean, CFData, CFDictionary, CFRetained, CFString, CFType};
use objc2_core_media::{
    CMBlockBuffer, CMSampleBuffer, CMSampleTimingInfo, CMTime, CMVideoCodecType,
    CMVideoFormatDescription, CMVideoFormatDescriptionCreate, kCMBlockBufferAssureMemoryNowFlag,
    kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms,
};
use objc2_core_video::CVImageBuffer;
use objc2_video_toolbox::{
    VTDecodeFrameFlags, VTDecodeInfoFlags, VTDecompressionOutputCallbackRecord,
    VTDecompressionSession, kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder,
    kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder,
    kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder,
};

use crate::caps::register_supplemental_decoder;
use crate::cf::{copy_property, dictionary, owned};
use crate::error::{Error, check};
use crate::pixel_buffer::{self, PixelBuffer};
use crate::queue::Queue;
use crate::time::{from_cm_time, to_cm_time, to_cm_time_or_invalid};
use reorder::Reorder;

type Decoded = Result<VideoFrame<PixelBuffer>, Error>;

/// What a codec module tells the session about its stream.
struct Description<'a> {
    codec_type: CMVideoCodecType,
    /// Sample description atom, such as `avcC`, and its contents.
    atom: Option<(&'static str, &'a [u8])>,
    output: PixelFormat,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecoderOptions {
    /// Fail instead of decoding in software.
    pub require_hardware: bool,
}

/// A VideoToolbox decode session for one stream.
///
/// Frames come out in presentation order as IOSurface-backed
/// [`PixelBuffer`]s, in NV12, or P010 for streams that can carry more than 8
/// bits. Packets with decode times come out after the fewest frames of delay;
/// without them, after up to 16.
pub struct Decoder {
    session: CFRetained<VTDecompressionSession>,
    format: CFRetained<CMVideoFormatDescription>,
    /// The output callback's refcon points here, so it must outlive the
    /// session. `Drop` invalidates the session first.
    output: Arc<Queue<Decoded>>,
    reorder: Reorder<VideoFrame<PixelBuffer>>,
}

// SAFETY: VideoToolbox sessions and format descriptions may be used from any
// thread, one call at a time. Every method that calls into the session takes
// `&mut self`, or `&self` on a type that isn't `Sync`.
unsafe impl Send for Decoder {}

impl Decoder {
    pub fn new(config: &VideoStreamConfig, options: DecoderOptions) -> Result<Self, Error> {
        let description = describe(config)?;
        register_supplemental_decoder(description.codec_type);
        let format = format_description(&description, config.width, config.height)?;

        // SAFETY: these are immutable CFString constants exported by
        // VideoToolbox.
        let (enable_key, require_key) = unsafe {
            (
                kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder,
                kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder,
            )
        };
        let mut spec: Vec<(&CFString, &CFType)> = vec![(enable_key, CFBoolean::new(true).as_ref())];
        if options.require_hardware {
            spec.push((require_key, CFBoolean::new(true).as_ref()));
        }
        let spec = dictionary(&spec);
        let attributes = pixel_buffer::attributes(
            &[
                (description.output, ColorRange::Limited),
                (description.output, ColorRange::Full),
            ],
            None,
        );

        let output = Arc::new(Queue::new());
        let callback = VTDecompressionOutputCallbackRecord {
            decompressionOutputCallback: Some(decoded),
            decompressionOutputRefCon: Arc::as_ptr(&output).cast_mut().cast(),
        };
        let mut session = ptr::null_mut();
        // SAFETY: the dictionaries map CFString keys to the value types
        // VideoToolbox documents. It copies `callback` during the call. The
        // refcon points to `output`, which the returned `Decoder` keeps alive
        // until after it invalidates the session.
        let status = unsafe {
            VTDecompressionSession::create(
                None,
                &format,
                Some(spec.as_opaque()),
                Some(attributes.as_opaque()),
                &callback,
                NonNull::from(&mut session),
            )
        };
        check("VTDecompressionSessionCreate", status)?;
        // SAFETY: Create functions return an owned reference.
        let session = unsafe { owned(session, "VTDecompressionSessionCreate") }?;
        Ok(Self {
            session,
            format,
            output,
            reorder: Reorder::new(),
        })
    }

    /// Decodes one packet. Packets go in decode order, starting at a keyframe.
    pub fn send_packet(&mut self, packet: &Packet) -> Result<(), Error> {
        let sample = self.sample_buffer(packet)?;
        self.reorder.packet_sent(packet.dts);
        let mut info = VTDecodeInfoFlags::empty();
        // SAFETY: the session and sample are live. The frame refcon is unused,
        // and `info` is a valid place for the flags. Without the asynchronous
        // flag, the output callback runs before this returns.
        let status = unsafe {
            self.session.decode_frame(
                &sample,
                VTDecodeFrameFlags::empty(),
                ptr::null_mut(),
                &mut info,
            )
        };
        check("VTDecompressionSessionDecodeFrame", status)
    }

    /// The next frame in presentation order, or `None` until more packets go
    /// in. Errors come out as soon as VideoToolbox reports them.
    pub fn receive_frame(&mut self) -> Option<Decoded> {
        while let Some(decoded) = self.output.pop() {
            match decoded {
                Ok(frame) => self.reorder.push(frame.pts, frame),
                Err(error) => return Some(Err(error)),
            }
        }
        self.reorder.pop().map(Ok)
    }

    /// Releases every frame still held back. Call at the end of the stream;
    /// sending another packet resumes normal reordering.
    pub fn flush(&mut self) -> Result<(), Error> {
        // SAFETY: the session is live.
        let status = unsafe { self.session.wait_for_asynchronous_frames() };
        check("VTDecompressionSessionWaitForAsynchronousFrames", status)?;
        self.reorder.drain();
        Ok(())
    }

    /// Drops held-back and undelivered frames, before a seek. The next packet
    /// must be a keyframe.
    pub fn reset(&mut self) -> Result<(), Error> {
        // SAFETY: the session is live.
        let status = unsafe { self.session.wait_for_asynchronous_frames() };
        check("VTDecompressionSessionWaitForAsynchronousFrames", status)?;
        self.output.clear();
        self.reorder.clear();
        Ok(())
    }

    /// False when VideoToolbox picked a software decoder.
    pub fn is_hardware(&self) -> Result<bool, Error> {
        // SAFETY: an immutable CFString constant exported by VideoToolbox.
        let key = unsafe { kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder };
        let value = copy_property(&self.session, key)?;
        Ok(value
            .and_then(|value| value.downcast::<CFBoolean>().ok())
            .is_some_and(|value| value.as_bool()))
    }

    fn sample_buffer(&self, packet: &Packet) -> Result<CFRetained<CMSampleBuffer>, Error> {
        let len = packet.data.len();
        if len == 0 {
            return Err(Error::InvalidInput("empty packet"));
        }

        let mut block = ptr::null_mut();
        // SAFETY: a null memory block with the default allocator asks Core
        // Media to allocate `len` bytes now. `block` is a valid place for the
        // result.
        let status = unsafe {
            CMBlockBuffer::create_with_memory_block(
                None,
                ptr::null_mut(),
                len,
                None,
                ptr::null(),
                0,
                len,
                kCMBlockBufferAssureMemoryNowFlag,
                NonNull::from(&mut block),
            )
        };
        check("CMBlockBufferCreateWithMemoryBlock", status)?;
        // SAFETY: Create functions return an owned reference.
        let block = unsafe { owned(block, "CMBlockBufferCreateWithMemoryBlock") }?;
        // SAFETY: `block` holds exactly `len` bytes, and the source is `len`
        // bytes long. The copy means VideoToolbox can keep the sample after
        // this call returns.
        let status = unsafe {
            CMBlockBuffer::replace_data_bytes(
                NonNull::from(packet.data.as_slice()).cast(),
                &block,
                0,
                len,
            )
        };
        check("CMBlockBufferReplaceDataBytes", status)?;

        let timing = CMSampleTimingInfo {
            duration: to_cm_time_or_invalid(packet.duration)?,
            presentationTimeStamp: to_cm_time(packet.pts)?,
            decodeTimeStamp: to_cm_time_or_invalid(packet.dts)?,
        };
        let mut sample = ptr::null_mut();
        // SAFETY: one sample with one timing entry and one size entry, each
        // pointing at a live local. `sample` is a valid place for the result.
        let status = unsafe {
            CMSampleBuffer::create_ready(
                None,
                Some(&block),
                Some(&self.format),
                1,
                1,
                &timing,
                1,
                &len,
                NonNull::from(&mut sample),
            )
        };
        check("CMSampleBufferCreateReady", status)?;
        // SAFETY: Create functions return an owned reference.
        unsafe { owned(sample, "CMSampleBufferCreateReady") }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: the session is live. After invalidation VideoToolbox makes no
        // more callbacks, so `output` can be freed afterwards.
        unsafe {
            self.session.wait_for_asynchronous_frames();
            self.session.invalidate();
        }
    }
}

/// VideoToolbox calls this for each decoded frame, possibly on its own thread.
unsafe extern "C-unwind" fn decoded(
    refcon: *mut c_void,
    _frame_refcon: *mut c_void,
    status: i32,
    flags: VTDecodeInfoFlags,
    image: *mut CVImageBuffer,
    pts: CMTime,
    duration: CMTime,
) {
    // SAFETY: `refcon` is the queue `Decoder::new` registered, which outlives
    // the session.
    let output = unsafe { &*refcon.cast_const().cast::<Queue<Decoded>>() };
    if let Err(error) = check("VTDecompressionOutputCallback", status) {
        output.push(Err(error));
        return;
    }
    if flags.contains(VTDecodeInfoFlags::FrameDropped) {
        return;
    }
    let Some(image) = NonNull::new(image) else {
        output.push(Err(Error::NullOutput {
            call: "VTDecompressionOutputCallback",
        }));
        return;
    };
    let Some(pts) = from_cm_time(pts) else {
        output.push(Err(Error::MissingTimestamp));
        return;
    };
    // SAFETY: VideoToolbox lends `image` for the duration of the callback.
    // Retaining it keeps it alive afterwards.
    let image = unsafe { CFRetained::retain(image) };
    let writable = flags.contains(VTDecodeInfoFlags::ImageBufferModifiable);
    output.push(Ok(VideoFrame {
        pts,
        duration: from_cm_time(duration),
        image: PixelBuffer::from_retained(image, writable),
    }));
}

/// True when this crate can open a decoder for `codec`.
pub fn decodes(codec: VideoCodec) -> bool {
    matches!(codec, VideoCodec::H264 | VideoCodec::Hevc)
}

fn describe(config: &VideoStreamConfig) -> Result<Description<'_>, Error> {
    match config.codec {
        VideoCodec::H264 => h264::describe(&config.config_record),
        VideoCodec::Hevc => hevc::describe(&config.config_record),
        VideoCodec::Vp8
        | VideoCodec::Vp9
        | VideoCodec::Av1
        | VideoCodec::Mpeg2
        | VideoCodec::Mpeg4Part2
        | VideoCodec::Mjpeg
        | VideoCodec::ProRes => Err(Error::DecodeUnsupported(config.codec)),
    }
}

fn format_description(
    description: &Description<'_>,
    width: u32,
    height: u32,
) -> Result<CFRetained<CMVideoFormatDescription>, Error> {
    let width = i32::try_from(width).map_err(|_| Error::InvalidInput("width above i32::MAX"))?;
    let height = i32::try_from(height).map_err(|_| Error::InvalidInput("height above i32::MAX"))?;

    // SAFETY: an immutable CFString constant exported by Core Media.
    let atoms_key = unsafe { kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms };
    let atoms = description.atom.map(|(name, bytes)| {
        let name = CFString::from_static_str(name);
        let bytes = CFData::from_bytes(bytes);
        CFDictionary::<CFString, CFData>::from_slices(&[&name], &[&bytes])
    });
    let extensions = match &atoms {
        Some(atoms) => dictionary(&[(atoms_key, atoms.as_ref())]),
        None => dictionary(&[]),
    };

    let mut format = ptr::null();
    // SAFETY: `extensions` maps CFString keys to the value types Core Media
    // documents, and `format` is a valid place for the result.
    let status = unsafe {
        CMVideoFormatDescriptionCreate(
            None,
            description.codec_type,
            width,
            height,
            Some(extensions.as_opaque()),
            NonNull::from(&mut format),
        )
    };
    check("CMVideoFormatDescriptionCreate", status)?;
    // SAFETY: Create functions return an owned reference.
    unsafe { owned(format.cast_mut(), "CMVideoFormatDescriptionCreate") }
}

/// Skips `count` entries of a 16-bit length followed by that many bytes, as
/// parameter sets are stored in `avcC` and `hvcC`.
fn skip_length_prefixed<'a>(
    mut data: &'a [u8],
    count: usize,
    error: &'static str,
) -> Result<&'a [u8], Error> {
    for _ in 0..count {
        let &[len_hi, len_lo, ref rest @ ..] = data else {
            return Err(Error::InvalidConfig(error));
        };
        let len = usize::from(u16::from_be_bytes([len_hi, len_lo]));
        data = rest.get(len..).ok_or(Error::InvalidConfig(error))?;
    }
    Ok(data)
}
