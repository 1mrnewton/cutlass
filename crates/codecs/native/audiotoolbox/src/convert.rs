//! An owned `AudioConverter`, and the buffers it fills.

use std::ffi::c_void;
use std::ptr::{self, NonNull};

use cutlass_core::AudioSamples;
use objc2_audio_toolbox::{
    AudioConverterDispose, AudioConverterGetProperty, AudioConverterGetPropertyInfo,
    AudioConverterNew, AudioConverterRef, AudioConverterSetProperty, kAudioConverterEncodeBitRate,
    kAudioConverterErr_FormatNotSupported, kAudioConverterErr_InputSampleRateOutOfRange,
    kAudioConverterErr_OutputSampleRateOutOfRange, kAudioConverterErr_PropertyNotSupported,
};
use objc2_core_audio_types::{AudioBuffer, AudioBufferList, AudioStreamBasicDescription};

use crate::error::{Error, check};
use crate::format::BITRATE;

pub(crate) const DECODE_FRAMES: usize = 4096;
/// Returned by an input callback that has no more data for this call, but
/// has not reached the end of the stream. Zero would mean end of stream.
pub(crate) const NEED_MORE: i32 = 100;
pub(crate) const MAX_PACKET_BYTES: usize = 16 * 1024;
pub(crate) const MAX_COOKIE_BYTES: usize = 4096;
pub(crate) const MAX_FLUSH_PACKETS: usize = 8;

pub(crate) const NO_CODEC: [i32; 3] = [
    kAudioConverterErr_FormatNotSupported,
    kAudioConverterErr_InputSampleRateOutOfRange,
    kAudioConverterErr_OutputSampleRateOutOfRange,
];

pub(crate) struct Converter {
    raw: AudioConverterRef,
}

impl Converter {
    pub(crate) fn new(
        mut source: AudioStreamBasicDescription,
        mut destination: AudioStreamBasicDescription,
    ) -> Result<Self, Error> {
        let mut raw = ptr::null_mut();
        // SAFETY: both descriptions are live for the call, and `raw` is a
        // place for the created converter.
        let status = unsafe {
            AudioConverterNew(
                NonNull::from(&mut source),
                NonNull::from(&mut destination),
                NonNull::from(&mut raw),
            )
        };
        check("AudioConverterNew", status, &NO_CODEC)?;
        if raw.is_null() {
            return Err(Error::Os {
                call: "AudioConverterNew",
                status: 0,
            });
        }
        Ok(Self { raw })
    }

    pub(crate) fn raw(&self) -> AudioConverterRef {
        self.raw
    }
}

impl Drop for Converter {
    fn drop(&mut self) {
        // SAFETY: `raw` is the converter this value created. Nothing else
        // disposes it.
        unsafe { AudioConverterDispose(self.raw) };
    }
}

/// One-buffer list. Its prefix matches [`AudioBufferList`].
#[repr(C)]
pub(crate) struct OutList {
    count: u32,
    pub(crate) buffer: AudioBuffer,
}

pub(crate) fn one_buffer(channels: u16, data: *mut c_void, bytes: usize) -> Result<OutList, Error> {
    Ok(OutList {
        count: 1,
        buffer: AudioBuffer {
            mNumberChannels: u32::from(channels),
            mDataByteSize: u32::try_from(bytes).map_err(|_| Error::InvalidInput("audio buffer"))?,
            mData: data,
        },
    })
}

pub(crate) fn list_ptr(list: &mut OutList) -> NonNull<AudioBufferList> {
    NonNull::from(list).cast::<AudioBufferList>()
}

pub(crate) fn set_bitrate(converter: AudioConverterRef) -> Result<(), Error> {
    let mut bitrate = BITRATE;
    // SAFETY: `converter` is live, and `bitrate` is the documented u32 value.
    let status = unsafe {
        AudioConverterSetProperty(
            converter,
            kAudioConverterEncodeBitRate,
            4,
            NonNull::from(&mut bitrate).cast(),
        )
    };
    if status == kAudioConverterErr_PropertyNotSupported {
        return Ok(());
    }
    check("AudioConverterSetProperty", status, &NO_CODEC)
}

pub(crate) fn set_cookie(
    converter: AudioConverterRef,
    property: u32,
    cookie: &[u8],
) -> Result<(), Error> {
    let size = u32::try_from(cookie.len()).map_err(|_| Error::InvalidConfig("AAC magic cookie"))?;
    let ptr = NonNull::new(cookie.as_ptr().cast_mut().cast::<c_void>())
        .ok_or(Error::InvalidConfig("AAC magic cookie"))?;
    // SAFETY: `converter` is live, and `cookie` is the magic cookie bytes.
    // The call reads them.
    let status = unsafe { AudioConverterSetProperty(converter, property, size, ptr) };
    check("AudioConverterSetProperty", status, &NO_CODEC)
}

pub(crate) fn magic_cookie(converter: AudioConverterRef, property: u32) -> Result<Vec<u8>, Error> {
    let mut size = 0u32;
    // SAFETY: `converter` is live, and `size` is a place for the byte count.
    let status =
        unsafe { AudioConverterGetPropertyInfo(converter, property, &mut size, ptr::null_mut()) };
    check("AudioConverterGetPropertyInfo", status, &NO_CODEC)?;
    let max = u32::try_from(MAX_COOKIE_BYTES).unwrap_or(u32::MAX);
    if size == 0 || size > max {
        return Err(Error::InvalidConfig("AAC magic cookie"));
    }
    let mut bytes =
        vec![0u8; usize::try_from(size).map_err(|_| Error::InvalidConfig("AAC magic cookie"))?];
    // SAFETY: `bytes` has room for `size` bytes, and `converter` is live.
    let status = unsafe {
        AudioConverterGetProperty(
            converter,
            property,
            NonNull::from(&mut size),
            NonNull::new(bytes.as_mut_ptr().cast())
                .ok_or(Error::InvalidConfig("AAC magic cookie"))?,
        )
    };
    check("AudioConverterGetProperty", status, &NO_CODEC)?;
    let size = usize::try_from(size).map_err(|_| Error::InvalidConfig("AAC magic cookie"))?;
    if size > bytes.len() {
        return Err(Error::InvalidConfig("AAC magic cookie"));
    }
    bytes.truncate(size);
    if bytes.is_empty() {
        return Err(Error::InvalidConfig("AAC magic cookie"));
    }
    Ok(bytes)
}

pub(crate) fn interleave(samples: &AudioSamples) -> Result<Vec<f32>, Error> {
    let frames = samples.frames();
    let channels = usize::from(samples.channels());
    let mut out = vec![
        0f32;
        frames
            .checked_mul(channels)
            .ok_or(Error::InvalidInput("audio is too large"))?
    ];
    for channel in 0..samples.channels() {
        let plane = samples
            .plane(channel)
            .ok_or(Error::InvalidInput("missing audio plane"))?;
        for (frame, sample) in plane.iter().enumerate() {
            out[frame * channels + usize::from(channel)] = *sample;
        }
    }
    Ok(out)
}

pub(crate) fn deinterleave(
    interleaved: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<AudioSamples, Error> {
    let channel_count = usize::from(channels);
    if channel_count == 0 || !interleaved.len().is_multiple_of(channel_count) {
        return Err(Error::InvalidInput("decoded AAC size"));
    }
    let frames = interleaved.len() / channel_count;
    let mut planar = vec![0f32; interleaved.len()];
    for frame in 0..frames {
        for channel in 0..channel_count {
            planar[channel * frames + frame] = interleaved[frame * channel_count + channel];
        }
    }
    AudioSamples::new(sample_rate, channels, planar)
        .map_err(|_| Error::InvalidInput("decoded AAC size"))
}

pub(crate) fn packet_count(packets: NonNull<u32>) -> usize {
    // SAFETY: `packets` is the pointer AudioConverter passed for this fill.
    usize::try_from(unsafe { packets.read() }).unwrap_or(0)
}

pub(crate) fn set_packet_count(packets: NonNull<u32>, count: u32) {
    // SAFETY: `packets` is the pointer AudioConverter passed for this fill.
    unsafe { packets.write(count) };
}
