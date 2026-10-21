//! One `AudioConverter` encode path. AAC-LC is the only codec.

mod aac;

use std::ffi::c_void;
use std::ptr::NonNull;

use cutlass_core::AudioSamples;
use objc2_audio_toolbox::{
    AudioConverterFillComplexBuffer, AudioConverterRef, kAudioConverterCompressionMagicCookie,
    kAudioConverterErr_InvalidInputSize,
};
use objc2_core_audio_types::{AudioBufferList, AudioStreamPacketDescription};

use crate::convert::{self, Converter, MAX_FLUSH_PACKETS, MAX_PACKET_BYTES, NEED_MORE, NO_CODEC};
use crate::error::{Error, check};
use crate::format::{self, aac as aac_format, pcm};

/// One AAC packet. AAC-LC holds 1024 frames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packet {
    pub data: Vec<u8>,
}

/// Encodes planar PCM to AAC-LC.
pub struct Encoder {
    converter: Converter,
    sample_rate: u32,
    channels: u16,
    cookie: Vec<u8>,
}

// SAFETY: an AudioConverter is used from one thread at a time. Every call
// takes `&mut self`, and the type is not `Sync`.
unsafe impl Send for Encoder {}

impl Encoder {
    /// Opens an AAC-LC encoder at `sample_rate` for `channels`.
    ///
    /// Stereo is left, then right. `channels` is at most 8.
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self, Error> {
        format::check_layout(sample_rate, channels)?;
        let converter = Converter::new(
            pcm(sample_rate, channels),
            aac_format(sample_rate, channels),
        )?;
        convert::set_bitrate(converter.raw())?;
        let cookie = convert::magic_cookie(converter.raw(), kAudioConverterCompressionMagicCookie)?;
        let cookie = aac::audio_specific_config(&cookie)?;
        Ok(Self {
            converter,
            sample_rate,
            channels,
            cookie,
        })
    }

    /// The audio specific config an MP4 `esds` box stores.
    pub fn config_record(&self) -> &[u8] {
        &self.cookie
    }

    /// Encodes `samples`. Packets still inside the converter come out of
    /// [`Self::finish`].
    pub fn encode(&mut self, samples: &AudioSamples) -> Result<Vec<Packet>, Error> {
        self.same_stream(samples)?;
        let interleaved = convert::interleave(samples)?;
        let mut cursor = PcmCursor {
            data: &interleaved,
            frame: 0,
            frames: samples.frames(),
            channels: self.channels,
            end_of_stream: false,
        };
        let mut packets = Vec::new();
        while cursor.frame < cursor.frames {
            let before = cursor.frame;
            if let Some(packet) = pull_packet(&mut self.converter, &mut cursor)? {
                packets.push(packet);
            } else if cursor.frame == before {
                break;
            }
        }
        Ok(packets)
    }

    /// Ends the stream and returns the packets still in the converter.
    pub fn finish(&mut self) -> Result<Vec<Packet>, Error> {
        let mut cursor = PcmCursor {
            data: &[],
            frame: 0,
            frames: 0,
            channels: self.channels,
            end_of_stream: true,
        };
        let mut packets = Vec::new();
        for _ in 0..MAX_FLUSH_PACKETS {
            match pull_packet(&mut self.converter, &mut cursor)? {
                Some(packet) => packets.push(packet),
                None => return Ok(packets),
            }
        }
        Err(Error::InvalidInput("AAC encoder did not finish"))
    }

    fn same_stream(&self, samples: &AudioSamples) -> Result<(), Error> {
        if samples.sample_rate() != self.sample_rate || samples.channels() != self.channels {
            return Err(Error::InvalidInput(
                "audio samples do not match the AAC encoder",
            ));
        }
        Ok(())
    }
}

struct PcmCursor<'a> {
    data: &'a [f32],
    frame: usize,
    frames: usize,
    channels: u16,
    end_of_stream: bool,
}

fn pull_packet(
    converter: &mut Converter,
    cursor: &mut PcmCursor<'_>,
) -> Result<Option<Packet>, Error> {
    let mut bytes = vec![0u8; MAX_PACKET_BYTES];
    let mut list = convert::one_buffer(cursor.channels, bytes.as_mut_ptr().cast(), bytes.len())?;
    let mut description = AudioStreamPacketDescription {
        mStartOffset: 0,
        mVariableFramesInPacket: 0,
        mDataByteSize: 0,
    };
    let mut packets = 1u32;
    // SAFETY: `cursor` lives until this call returns, and the callback runs
    // only during it. `list` is one buffer, which is the layout of
    // `AudioBufferList`. The description slot holds one packet.
    let status = unsafe {
        AudioConverterFillComplexBuffer(
            converter.raw(),
            Some(supply_pcm),
            (cursor as *mut PcmCursor).cast(),
            NonNull::from(&mut packets),
            convert::list_ptr(&mut list),
            &mut description,
        )
    };
    if packets == 0 {
        if status == 0 || status == NEED_MORE {
            return Ok(None);
        }
        check("AudioConverterFillComplexBuffer", status, &NO_CODEC)?;
        return Ok(None);
    }
    if status != 0 && status != NEED_MORE {
        check("AudioConverterFillComplexBuffer", status, &NO_CODEC)?;
    }
    let start = usize::try_from(description.mStartOffset)
        .map_err(|_| Error::InvalidInput("AAC packet offset"))?;
    let size = usize::try_from(description.mDataByteSize)
        .map_err(|_| Error::InvalidInput("AAC packet size"))?;
    let end = start
        .checked_add(size)
        .ok_or(Error::InvalidInput("AAC packet size"))?;
    if size == 0 || end > bytes.len() {
        return Err(Error::InvalidInput("AAC packet size"));
    }
    Ok(Some(Packet {
        data: bytes[start..end].to_vec(),
    }))
}

/// # Safety
///
/// `user` points at a live [`PcmCursor`] for this fill. `packets` and `data`
/// are the pointers AudioConverter passes, and this runs only during that fill.
unsafe extern "C-unwind" fn supply_pcm(
    _converter: AudioConverterRef,
    packets: NonNull<u32>,
    mut data: NonNull<AudioBufferList>,
    _descriptions: *mut *mut AudioStreamPacketDescription,
    user: *mut c_void,
) -> i32 {
    // SAFETY: the caller of `AudioConverterFillComplexBuffer` passed `user`.
    let cursor = unsafe { &mut *user.cast::<PcmCursor>() };
    let want = convert::packet_count(packets);
    let left = cursor.frames.saturating_sub(cursor.frame);
    let give = want.min(left);
    if give == 0 {
        convert::set_packet_count(packets, 0);
        return if cursor.end_of_stream { 0 } else { NEED_MORE };
    }
    let channels = usize::from(cursor.channels);
    let Some(floats) = give.checked_mul(channels) else {
        convert::set_packet_count(packets, 0);
        return kAudioConverterErr_InvalidInputSize;
    };
    let Some(bytes) = floats
        .checked_mul(4)
        .and_then(|bytes| u32::try_from(bytes).ok())
    else {
        convert::set_packet_count(packets, 0);
        return kAudioConverterErr_InvalidInputSize;
    };
    let start = cursor.frame * channels;
    let end = start + floats;
    if end > cursor.data.len() {
        convert::set_packet_count(packets, 0);
        return kAudioConverterErr_InvalidInputSize;
    }
    // SAFETY: `data` is the list AudioConverter asked us to fill.
    let list = unsafe { data.as_mut() };
    list.mNumberBuffers = 1;
    let buffer = &mut list.mBuffers[0];
    buffer.mNumberChannels = u32::from(cursor.channels);
    buffer.mDataByteSize = bytes;
    // SAFETY: the floats live in `cursor.data` until this fill returns, and
    // the converter only reads them.
    buffer.mData = cursor.data[start..end].as_ptr().cast::<c_void>().cast_mut();
    cursor.frame += give;
    let written = u32::try_from(give).unwrap_or(0);
    convert::set_packet_count(packets, written);
    0
}
