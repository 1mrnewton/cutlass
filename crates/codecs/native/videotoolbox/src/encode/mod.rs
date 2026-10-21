//! One `VTCompressionSession` path for every codec. The codec modules only set
//! codec properties.

mod h264;
mod hevc;
mod prores;

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::sync::Arc;

use cutlass_core::{ColorRange, Packet, PixelFormat, Timestamp, VideoCodec};
use objc2_core_foundation::{
    CFArray, CFBoolean, CFData, CFDictionary, CFNumber, CFRetained, CFString, CFType,
};
use objc2_core_media::{
    CMFormatDescription, CMSampleBuffer, CMVideoCodecType,
    kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms, kCMSampleAttachmentKey_NotSync,
};
use objc2_core_video::CVPixelBufferPool;
use objc2_video_toolbox::{
    VTCompressionSession, VTEncodeInfoFlags, kVTCompressionPropertyKey_AllowFrameReordering,
    kVTCompressionPropertyKey_AverageBitRate, kVTCompressionPropertyKey_ExpectedFrameRate,
    kVTCompressionPropertyKey_MaxKeyFrameInterval, kVTCompressionPropertyKey_ProfileLevel,
    kVTCompressionPropertyKey_RealTime,
    kVTCompressionPropertyKey_UsingHardwareAcceleratedVideoEncoder,
    kVTEncodeFrameOptionKey_ForceKeyFrame,
    kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder,
    kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder,
};

use crate::cf::{copy_property, dictionary, owned, set_property};
use crate::error::{Error, check};
use crate::pixel_buffer::{self, PixelBuffer};
use crate::queue::Queue;
use crate::time::{INVALID, from_cm_time, to_cm_time, to_cm_time_or_invalid};

/// Packets held back while measuring how far a decode timestamp can fall
/// after its presentation timestamp. Sixteen is the most H.264 or HEVC can
/// reorder, so the offset is known by then.
const DTS_WINDOW: usize = 16;

/// What a codec module sets on the session.
struct Settings {
    codec_type: CMVideoCodecType,
    profile: &'static CFString,
    /// The sample description atom that holds the config record.
    atom: &'static str,
}

/// A compressed frame as VideoToolbox emitted it, converted on the caller's
/// thread.
struct Encoded(CFRetained<CMSampleBuffer>);

// SAFETY: VideoToolbox doesn't modify a sample buffer after emitting it, and
// Core Foundation reference counts are atomic.
unsafe impl Send for Encoded {}

#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    /// H.264 or HEVC.
    pub codec: VideoCodec,
    pub width: u32,
    pub height: u32,
    /// [`PixelFormat::P010`] makes 10-bit HEVC. H.264 takes NV12 only.
    pub pixel_format: PixelFormat,
    pub color_range: ColorRange,
    /// Average bits per second. `None` leaves it to the encoder.
    pub bitrate: Option<u32>,
    /// Most frames between keyframes. `None` leaves it to the encoder.
    pub keyframe_interval: Option<u32>,
    /// Allow B-frames. They compress better, but packets then come out in
    /// decode order, a few frames late.
    pub b_frames: bool,
    /// Expected frames per second, a hint for rate control.
    pub frame_rate: Option<f64>,
    /// Lower latency at some cost in quality, for live use.
    pub real_time: bool,
    /// Fail instead of encoding in software.
    pub require_hardware: bool,
}

impl EncoderConfig {
    /// 8-bit, limited range, B-frames on, everything else left to the encoder.
    pub fn new(codec: VideoCodec, width: u32, height: u32) -> Self {
        Self {
            codec,
            width,
            height,
            pixel_format: PixelFormat::Nv12,
            color_range: ColorRange::Limited,
            bitrate: None,
            keyframe_interval: None,
            b_frames: true,
            frame_rate: None,
            real_time: false,
            require_hardware: false,
        }
    }
}

/// A VideoToolbox encode session.
///
/// Packets come out in decode order with length-prefixed NAL units, ready for
/// an MP4 muxer, along with the `avcC` or `hvcC` config record. Decode
/// timestamps are at or before presentation timestamps.
pub struct Encoder {
    session: CFRetained<VTCompressionSession>,
    /// The output callback's refcon points here, so it must outlive the
    /// session. `Drop` invalidates the session first.
    output: Arc<Queue<Result<Encoded, Error>>>,
    atom: &'static str,
    format: Option<CFRetained<CMFormatDescription>>,
    config_record: Option<Vec<u8>>,
    /// Ticks subtracted from VideoToolbox's decode timestamps, in their
    /// timescale. VideoToolbox counts decode time in output order, so a
    /// B-frame's raw decode time falls after its presentation time. `None`
    /// until the first group of pictures shows the offset.
    dts_shift: Option<i64>,
    /// Packets waiting for `dts_shift`.
    pending: VecDeque<Packet>,
    ready: VecDeque<Packet>,
    saw_keyframe: bool,
    flushed: bool,
}

// SAFETY: VideoToolbox sessions may be used from any thread, one call at a
// time. Every method that calls into the session takes `&mut self`, or `&self`
// on a type that isn't `Sync`.
unsafe impl Send for Encoder {}

impl Encoder {
    pub fn new(config: &EncoderConfig) -> Result<Self, Error> {
        let settings = match config.codec {
            VideoCodec::H264 => h264::settings(config.pixel_format)?,
            VideoCodec::Hevc => hevc::settings(config.pixel_format)?,
            VideoCodec::Vp8
            | VideoCodec::Vp9
            | VideoCodec::Av1
            | VideoCodec::Mpeg2
            | VideoCodec::Mpeg4Part2
            | VideoCodec::Mjpeg
            | VideoCodec::ProRes => return Err(Error::EncodeUnsupported(config.codec)),
        };
        if config.width == 0 || config.height == 0 {
            return Err(Error::InvalidInput(
                "encoder width and height must be nonzero",
            ));
        }
        let width =
            i32::try_from(config.width).map_err(|_| Error::InvalidInput("width above i32::MAX"))?;
        let height = i32::try_from(config.height)
            .map_err(|_| Error::InvalidInput("height above i32::MAX"))?;

        // SAFETY: these are immutable CFString constants exported by
        // VideoToolbox.
        let (enable_key, require_key) = unsafe {
            (
                kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder,
                kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder,
            )
        };
        let mut spec: Vec<(&CFString, &CFType)> = vec![(enable_key, CFBoolean::new(true).as_ref())];
        if config.require_hardware {
            spec.push((require_key, CFBoolean::new(true).as_ref()));
        }
        let spec = dictionary(&spec);
        let attributes = pixel_buffer::attributes(
            &[(config.pixel_format, config.color_range)],
            Some((config.width, config.height)),
        );

        let output = Arc::new(Queue::new());
        let mut session = ptr::null_mut();
        // SAFETY: the dictionaries map CFString keys to the value types
        // VideoToolbox documents. The refcon points to `output`, which the
        // returned `Encoder` keeps alive until after it invalidates the
        // session.
        let status = unsafe {
            VTCompressionSession::create(
                None,
                width,
                height,
                settings.codec_type,
                Some(spec.as_opaque()),
                Some(attributes.as_opaque()),
                None,
                Some(encoded),
                Arc::as_ptr(&output).cast_mut().cast(),
                NonNull::from(&mut session),
            )
        };
        check("VTCompressionSessionCreate", status)?;
        // SAFETY: Create functions return an owned reference.
        let session = unsafe { owned(session, "VTCompressionSessionCreate") }?;
        let encoder = Self {
            session,
            output,
            atom: settings.atom,
            format: None,
            config_record: None,
            dts_shift: None,
            pending: VecDeque::new(),
            ready: VecDeque::new(),
            saw_keyframe: false,
            flushed: false,
        };
        encoder.configure(config, settings.profile)?;
        Ok(encoder)
    }

    /// A buffer from the session's pool, already in the layout the encoder
    /// wants. Fill it with [`PixelBuffer::write_planes`], then pass it to
    /// [`send_frame`](Self::send_frame).
    pub fn create_pixel_buffer(&self) -> Result<PixelBuffer, Error> {
        // SAFETY: the session is live.
        let pool = unsafe { self.session.pixel_buffer_pool() }.ok_or(Error::NullOutput {
            call: "VTCompressionSessionGetPixelBufferPool",
        })?;
        let mut buffer = ptr::null_mut();
        // SAFETY: `pool` is live, and `buffer` is a valid place for the result.
        let status = unsafe {
            CVPixelBufferPool::create_pixel_buffer(None, &pool, NonNull::from(&mut buffer))
        };
        check("CVPixelBufferPoolCreatePixelBuffer", status)?;
        // SAFETY: Create functions return an owned reference.
        let buffer = unsafe { owned(buffer, "CVPixelBufferPoolCreatePixelBuffer") }?;
        Ok(PixelBuffer::from_retained(buffer, true))
    }

    /// Encodes one frame. Timestamps must increase from frame to frame. The
    /// encoder keeps `image` until it is done with it.
    pub fn send_frame(
        &mut self,
        image: PixelBuffer,
        pts: Timestamp,
        duration: Option<Timestamp>,
        force_keyframe: bool,
    ) -> Result<(), Error> {
        let pts = to_cm_time(pts)?;
        let duration = to_cm_time_or_invalid(duration)?;
        // SAFETY: an immutable CFString constant exported by VideoToolbox.
        let force_key = unsafe { kVTEncodeFrameOptionKey_ForceKeyFrame };
        let properties =
            force_keyframe.then(|| dictionary(&[(force_key, CFBoolean::new(true).as_ref())]));
        // SAFETY: the session and image are live, and VideoToolbox retains the
        // image while it needs it. The frame refcon is unused, and the info
        // flags aren't requested.
        let status = unsafe {
            self.session.encode_frame(
                image.as_cv(),
                pts,
                duration,
                properties
                    .as_deref()
                    .map(|properties| properties.as_opaque()),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        check("VTCompressionSessionEncodeFrame", status)
    }

    /// The next packet, or `None` until more frames go in. The first packets
    /// wait until the decode-time offset is known: the next keyframe, 16
    /// packets, or [`flush`](Self::flush).
    pub fn receive_packet(&mut self) -> Option<Result<Packet, Error>> {
        if let Some(packet) = self.ready.pop_front() {
            return Some(Ok(packet));
        }
        loop {
            let Some(encoded) = self.output.pop() else {
                if self.flushed && self.dts_shift.is_none() && !self.pending.is_empty() {
                    if let Err(error) = self.lock_dts_shift() {
                        return Some(Err(error));
                    }
                    return self.ready.pop_front().map(Ok);
                }
                return None;
            };
            let packet = match encoded.and_then(|encoded| self.packet(&encoded.0)) {
                Ok(packet) => packet,
                Err(error) => return Some(Err(error)),
            };
            if let Err(error) = self.queue_packet(packet) {
                return Some(Err(error));
            }
            if let Some(packet) = self.ready.pop_front() {
                return Some(Ok(packet));
            }
        }
    }

    /// Outputs every frame the encoder still holds. Call at the end of the
    /// stream.
    pub fn flush(&mut self) -> Result<(), Error> {
        // SAFETY: the session is live. An invalid time asks for every frame.
        let status = unsafe { self.session.complete_frames(INVALID) };
        check("VTCompressionSessionCompleteFrames", status)?;
        self.flushed = true;
        Ok(())
    }

    /// The `avcC` or `hvcC` record for the container. Known once the first
    /// packet has been received.
    pub fn config_record(&self) -> Option<&[u8]> {
        self.config_record.as_deref()
    }

    /// False when VideoToolbox picked a software encoder.
    pub fn is_hardware(&self) -> Result<bool, Error> {
        // SAFETY: an immutable CFString constant exported by VideoToolbox.
        let key = unsafe { kVTCompressionPropertyKey_UsingHardwareAcceleratedVideoEncoder };
        let value = copy_property(&self.session, key)?;
        Ok(value
            .and_then(|value| value.downcast::<CFBoolean>().ok())
            .is_some_and(|value| value.as_bool()))
    }

    fn configure(&self, config: &EncoderConfig, profile: &CFString) -> Result<(), Error> {
        // SAFETY: these are immutable CFString constants exported by
        // VideoToolbox.
        let (profile_key, real_time_key, reordering_key, bitrate_key, keyframe_key, rate_key) = unsafe {
            (
                kVTCompressionPropertyKey_ProfileLevel,
                kVTCompressionPropertyKey_RealTime,
                kVTCompressionPropertyKey_AllowFrameReordering,
                kVTCompressionPropertyKey_AverageBitRate,
                kVTCompressionPropertyKey_MaxKeyFrameInterval,
                kVTCompressionPropertyKey_ExpectedFrameRate,
            )
        };
        let session: &CFType = &self.session;
        set_property(
            session,
            "VTSessionSetProperty(ProfileLevel)",
            profile_key,
            profile,
        )?;
        set_property(
            session,
            "VTSessionSetProperty(RealTime)",
            real_time_key,
            CFBoolean::new(config.real_time),
        )?;
        set_property(
            session,
            "VTSessionSetProperty(AllowFrameReordering)",
            reordering_key,
            CFBoolean::new(config.b_frames),
        )?;
        if let Some(bitrate) = config.bitrate {
            let bitrate = CFNumber::new_i64(bitrate.into());
            set_property(
                session,
                "VTSessionSetProperty(AverageBitRate)",
                bitrate_key,
                &bitrate,
            )?;
        }
        if let Some(interval) = config.keyframe_interval {
            let interval = CFNumber::new_i64(interval.into());
            set_property(
                session,
                "VTSessionSetProperty(MaxKeyFrameInterval)",
                keyframe_key,
                &interval,
            )?;
        }
        if let Some(rate) = config.frame_rate {
            let rate = CFNumber::new_f64(rate);
            set_property(
                session,
                "VTSessionSetProperty(ExpectedFrameRate)",
                rate_key,
                &rate,
            )?;
        }
        // SAFETY: the session is live.
        let status = unsafe { self.session.prepare_to_encode_frames() };
        check("VTCompressionSessionPrepareToEncodeFrames", status)
    }

    fn queue_packet(&mut self, mut packet: Packet) -> Result<(), Error> {
        if packet.dts.is_none() {
            packet.dts = Some(packet.pts);
        }
        if let Some(shift) = self.dts_shift {
            self.ready.push_back(apply_dts_shift(packet, shift)?);
            return Ok(());
        }
        let closes_group = packet.keyframe && self.saw_keyframe;
        if packet.keyframe {
            self.saw_keyframe = true;
        }
        self.pending.push_back(packet);
        if closes_group || self.pending.len() > DTS_WINDOW {
            self.lock_dts_shift()?;
        }
        Ok(())
    }

    fn lock_dts_shift(&mut self) -> Result<(), Error> {
        let shift = self
            .pending
            .iter()
            .map(decode_lag)
            .try_fold(0_i64, |largest, lag| lag.map(|lag| largest.max(lag)))?
            .max(0);
        self.dts_shift = Some(shift);
        while let Some(packet) = self.pending.pop_front() {
            self.ready.push_back(apply_dts_shift(packet, shift)?);
        }
        Ok(())
    }

    fn packet(&mut self, sample: &CMSampleBuffer) -> Result<Packet, Error> {
        // SAFETY: the sample is a live buffer from the encoder.
        let block = unsafe { sample.data_buffer() }.ok_or(Error::NullOutput {
            call: "CMSampleBufferGetDataBuffer",
        })?;
        // SAFETY: the block buffer is live.
        let len = unsafe { block.data_length() };
        let mut data = vec![0; len];
        // SAFETY: `data` has room for the `len` bytes the block holds.
        let status =
            unsafe { block.copy_data_bytes(0, len, NonNull::from(data.as_mut_slice()).cast()) };
        check("CMBlockBufferCopyDataBytes", status)?;

        // SAFETY: the sample is live.
        let (pts, dts, duration) = unsafe {
            (
                sample.presentation_time_stamp(),
                sample.decode_time_stamp(),
                sample.duration(),
            )
        };
        let pts = from_cm_time(pts).ok_or(Error::MissingTimestamp)?;
        self.update_config_record(sample)?;
        Ok(Packet {
            data,
            pts,
            dts: from_cm_time(dts),
            duration: from_cm_time(duration),
            keyframe: is_keyframe(sample),
        })
    }

    /// Reads the config record whenever the format description changes, which
    /// is on the first packet and rarely after.
    fn update_config_record(&mut self, sample: &CMSampleBuffer) -> Result<(), Error> {
        // SAFETY: the sample is live.
        let format = unsafe { sample.format_description() }.ok_or(Error::NullOutput {
            call: "CMSampleBufferGetFormatDescription",
        })?;
        if self
            .format
            .as_deref()
            .is_some_and(|current| ptr::eq(current, &*format))
        {
            return Ok(());
        }
        // SAFETY: an immutable CFString constant exported by Core Media.
        let atoms_key = unsafe { kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms };
        // SAFETY: the format description is live.
        let atoms = unsafe { format.extension(atoms_key) }
            .and_then(|atoms| atoms.downcast::<CFDictionary>().ok())
            .ok_or(Error::NullOutput {
                call: "CMFormatDescriptionGetExtension(SampleDescriptionExtensionAtoms)",
            })?;
        // SAFETY: Core Media documents this extension as a dictionary from
        // atom names to atom contents.
        let atoms = unsafe { atoms.cast_unchecked::<CFString, CFType>() };
        let record = atoms
            .get(&CFString::from_static_str(self.atom))
            .and_then(|record| record.downcast::<CFData>().ok())
            .ok_or(Error::NullOutput {
                call: "CMFormatDescriptionGetExtension(config record)",
            })?;
        self.config_record = Some(record.to_vec());
        self.format = Some(format);
        Ok(())
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY: the session is live. Completing first means no callback is
        // in flight when the session goes away, and after invalidation
        // VideoToolbox makes no more callbacks, so `output` can be freed.
        unsafe {
            self.session.complete_frames(INVALID);
            self.session.invalidate();
        }
    }
}

/// How far the decode time falls after the presentation time, in decode-time
/// ticks. Negative when the decode time is earlier, which is what a container
/// wants.
fn decode_lag(packet: &Packet) -> Result<i64, Error> {
    let dts = packet.dts.ok_or(Error::MissingTimestamp)?;
    let pts_ticks = i128::from(packet.pts.value) * i128::from(dts.timescale.get())
        / i128::from(packet.pts.timescale.get());
    i64::try_from(i128::from(dts.value) - pts_ticks)
        .map_err(|_| Error::InvalidInput("decode timestamp overflow"))
}

fn apply_dts_shift(mut packet: Packet, shift: i64) -> Result<Packet, Error> {
    let dts = packet.dts.ok_or(Error::MissingTimestamp)?;
    let value = dts
        .value
        .checked_sub(shift)
        .ok_or(Error::InvalidInput("decode timestamp overflow"))?;
    packet.dts = Some(Timestamp::new(value, dts.timescale));
    Ok(packet)
}

/// A sample is a keyframe unless its attachments mark it `NotSync`.
fn is_keyframe(sample: &CMSampleBuffer) -> bool {
    // SAFETY: the sample is live, and asking without creation doesn't modify it.
    let Some(attachments) = (unsafe { sample.sample_attachments_array(false) }) else {
        return true;
    };
    // SAFETY: Core Media documents the attachments as an array of dictionaries
    // keyed by CFString.
    let attachments = unsafe {
        CFRetained::cast_unchecked::<CFArray<CFDictionary<CFString, CFType>>>(attachments)
    };
    // SAFETY: an immutable CFString constant exported by Core Media.
    let not_sync_key = unsafe { kCMSampleAttachmentKey_NotSync };
    let not_sync = attachments
        .get(0)
        .and_then(|attachment| attachment.get(not_sync_key))
        .and_then(|value| value.downcast::<CFBoolean>().ok())
        .is_some_and(|value| value.as_bool());
    !not_sync
}

/// VideoToolbox calls this for each compressed frame, in decode order,
/// possibly on its own thread.
unsafe extern "C-unwind" fn encoded(
    refcon: *mut c_void,
    _frame_refcon: *mut c_void,
    status: i32,
    flags: VTEncodeInfoFlags,
    sample: *mut CMSampleBuffer,
) {
    // SAFETY: `refcon` is the queue `Encoder::new` registered, which outlives
    // the session.
    let output = unsafe { &*refcon.cast_const().cast::<Queue<Result<Encoded, Error>>>() };
    if let Err(error) = check("VTCompressionOutputCallback", status) {
        output.push(Err(error));
        return;
    }
    if flags.contains(VTEncodeInfoFlags::FrameDropped) {
        return;
    }
    match NonNull::new(sample) {
        // SAFETY: VideoToolbox lends `sample` for the duration of the
        // callback. Retaining it keeps it alive afterwards.
        Some(sample) => output.push(Ok(Encoded(unsafe { CFRetained::retain(sample) }))),
        None => output.push(Err(Error::NullOutput {
            call: "VTCompressionOutputCallback",
        })),
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cutlass_core::{Packet, Timestamp};

    use super::{apply_dts_shift, decode_lag};

    fn packet(pts: i64, dts: i64) -> Packet {
        let timescale = NonZeroU32::new(30).unwrap();
        Packet {
            data: Vec::new(),
            pts: Timestamp::new(pts, timescale),
            dts: Some(Timestamp::new(dts, timescale)),
            duration: None,
            keyframe: false,
        }
    }

    #[test]
    fn decode_lag_is_how_far_decode_time_falls_behind_presentation() {
        assert_eq!(decode_lag(&packet(1, 3)).unwrap(), 2);
        assert_eq!(decode_lag(&packet(4, 1)).unwrap(), -3);
    }

    #[test]
    fn the_shift_puts_a_late_decode_time_back_on_its_presentation_time() {
        let shifted = apply_dts_shift(packet(1, 3), 2).unwrap();
        assert_eq!(shifted.dts.unwrap().value, 1);
        assert!(shifted.dts.unwrap() <= shifted.pts);
    }
}
