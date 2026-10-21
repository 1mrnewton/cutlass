//! One `AudioConverter` decode path. AAC-LC is the only codec.

mod aac;

use std::ffi::c_void;
use std::ptr::{self, NonNull};

use cutlass_core::{AudioCodec, AudioSamples, AudioStreamConfig};
use objc2_audio_toolbox::{
    AudioConverterFillComplexBuffer, AudioConverterRef, kAudioConverterDecompressionMagicCookie,
    kAudioConverterErr_InvalidInputSize,
};
use objc2_core_audio_types::{AudioBufferList, AudioStreamPacketDescription};

use crate::convert::{
    self, Converter, DECODE_FRAMES, MAX_COOKIE_BYTES, MAX_FLUSH_PACKETS, MAX_PACKET_BYTES,
    NEED_MORE, NO_CODEC,
};
use crate::error::{Error, check};
use crate::format::{self, aac as aac_format, pcm};

/// True when this crate can decode `codec`.
pub fn decodes(codec: AudioCodec) -> bool {
    matches!(codec, AudioCodec::Aac)
}

/// Decodes AAC-LC packets to planar PCM.
pub struct Decoder {
    converter: Converter,
    sample_rate: u32,
    channels: u16,
}

// SAFETY: an AudioConverter is used from one thread at a time. Every call
// takes `&mut self`, and the type is not `Sync`.
unsafe impl Send for Decoder {}

impl Decoder {
    /// Opens an AAC-LC decoder. `config.config_record` is the audio specific
    /// config an MP4 `esds` box stores.
    pub fn new(config: &AudioStreamConfig) -> Result<Self, Error> {
        if config.codec != AudioCodec::Aac {
            return Err(Error::DecodeUnsupported(config.codec));
        }
        format::check_layout(config.sample_rate, config.channels)?;
        if config.config_record.is_empty() || config.config_record.len() > MAX_COOKIE_BYTES {
            return Err(Error::InvalidConfig("AAC audio specific config"));
        }
        let converter = Converter::new(
            aac_format(config.sample_rate, config.channels),
            pcm(config.sample_rate, config.channels),
        )?;
        let cookie = aac::cookie_for_specific_config(&config.config_record)?;
        convert::set_cookie(
            converter.raw(),
            kAudioConverterDecompressionMagicCookie,
            &cookie,
        )?;
        Ok(Self {
            converter,
            sample_rate: config.sample_rate,
            channels: config.channels,
        })
    }

    /// Decodes one packet. An empty buffer means the converter is still priming.
    pub fn decode(&mut self, packet: &[u8]) -> Result<AudioSamples, Error> {
        if packet.is_empty() || packet.len() > MAX_PACKET_BYTES {
            return Err(Error::InvalidInput("AAC packet"));
        }
        self.pull_packets(packet)
    }

    /// Ends the stream and returns the samples still in the converter.
    pub fn flush(&mut self) -> Result<AudioSamples, Error> {
        self.pull_packets(&[])
    }

    fn pull_packets(&mut self, packet: &[u8]) -> Result<AudioSamples, Error> {
        let mut cursor = PacketCursor::new(packet, self.channels, packet.is_empty());
        let mut planes = vec![Vec::new(); usize::from(self.channels)];
        let mut empty_pulls = 0;
        for _ in 0..MAX_FLUSH_PACKETS {
            let chunk = pull_pcm(
                &mut self.converter,
                &mut cursor,
                self.sample_rate,
                self.channels,
            )?;
            let frames = chunk.frames();
            if frames == 0 {
                empty_pulls += 1;
                // A fill can take the packet and write no PCM. The samples
                // come out of the next fill.
                if packet.is_empty() || empty_pulls == 2 {
                    break;
                }
                continue;
            }
            empty_pulls = 0;
            for channel in 0..self.channels {
                let plane = chunk
                    .plane(channel)
                    .ok_or(Error::InvalidInput("decoded AAC is missing a plane"))?;
                planes[usize::from(channel)].extend_from_slice(plane);
            }
            if frames < DECODE_FRAMES {
                break;
            }
        }
        let mut packed = Vec::new();
        for plane in &planes {
            packed.extend_from_slice(plane);
        }
        AudioSamples::new(self.sample_rate, self.channels, packed)
            .map_err(|_| Error::InvalidInput("decoded AAC has uneven planes"))
    }
}

struct PacketCursor<'a> {
    data: &'a [u8],
    channels: u16,
    sent: bool,
    end_of_stream: bool,
    description: AudioStreamPacketDescription,
}

impl<'a> PacketCursor<'a> {
    fn new(data: &'a [u8], channels: u16, end_of_stream: bool) -> Self {
        Self {
            data,
            channels,
            sent: data.is_empty(),
            end_of_stream,
            description: AudioStreamPacketDescription {
                mStartOffset: 0,
                mVariableFramesInPacket: 0,
                mDataByteSize: 0,
            },
        }
    }
}

fn pull_pcm(
    converter: &mut Converter,
    cursor: &mut PacketCursor<'_>,
    sample_rate: u32,
    channels: u16,
) -> Result<AudioSamples, Error> {
    let channel_count = usize::from(channels);
    let mut interleaved = vec![0f32; DECODE_FRAMES * channel_count];
    let byte_len = interleaved
        .len()
        .checked_mul(4)
        .ok_or(Error::InvalidInput("decoded AAC size"))?;
    let mut list = convert::one_buffer(channels, interleaved.as_mut_ptr().cast(), byte_len)?;
    let mut packets =
        u32::try_from(DECODE_FRAMES).map_err(|_| Error::InvalidInput("decoded AAC size"))?;
    // SAFETY: `cursor` lives until this call returns, and the callback runs
    // only during it. `list` is one buffer, which is the layout of
    // `AudioBufferList`. PCM output has no packet descriptions.
    let status = unsafe {
        AudioConverterFillComplexBuffer(
            converter.raw(),
            Some(supply_packet),
            (cursor as *mut PacketCursor).cast(),
            NonNull::from(&mut packets),
            convert::list_ptr(&mut list),
            ptr::null_mut(),
        )
    };
    if packets == 0 {
        if status != 0 && status != NEED_MORE {
            check("AudioConverterFillComplexBuffer", status, &NO_CODEC)?;
        }
        return AudioSamples::new(sample_rate, channels, Vec::new())
            .map_err(|_| Error::InvalidInput("empty audio"));
    }
    if status != 0 && status != NEED_MORE {
        check("AudioConverterFillComplexBuffer", status, &NO_CODEC)?;
    }
    let bytes = list.buffer.mDataByteSize;
    let frame_bytes = 4u32
        .checked_mul(u32::from(channels))
        .ok_or(Error::InvalidInput("AAC frame size"))?;
    if frame_bytes == 0 || !bytes.is_multiple_of(frame_bytes) {
        return Err(Error::InvalidInput("decoded AAC size"));
    }
    let frames = usize::try_from(bytes / frame_bytes)
        .map_err(|_| Error::InvalidInput("decoded AAC size"))?;
    let floats = frames * channel_count;
    if floats > interleaved.len() {
        return Err(Error::InvalidInput("decoded AAC size"));
    }
    convert::deinterleave(&interleaved[..floats], sample_rate, channels)
}

/// # Safety
///
/// `user` points at a live [`PacketCursor`] for this fill. `packets`, `data`,
/// and `descriptions` are the pointers AudioConverter passes.
unsafe extern "C-unwind" fn supply_packet(
    _converter: AudioConverterRef,
    packets: NonNull<u32>,
    mut data: NonNull<AudioBufferList>,
    descriptions: *mut *mut AudioStreamPacketDescription,
    user: *mut c_void,
) -> i32 {
    // SAFETY: the caller of `AudioConverterFillComplexBuffer` passed `user`.
    let cursor = unsafe { &mut *user.cast::<PacketCursor>() };
    if cursor.sent || cursor.data.is_empty() {
        convert::set_packet_count(packets, 0);
        return if cursor.end_of_stream { 0 } else { NEED_MORE };
    }
    let Some(size) = u32::try_from(cursor.data.len()).ok() else {
        convert::set_packet_count(packets, 0);
        return kAudioConverterErr_InvalidInputSize;
    };
    // SAFETY: `data` is the list AudioConverter asked us to fill.
    let list = unsafe { data.as_mut() };
    list.mNumberBuffers = 1;
    let buffer = &mut list.mBuffers[0];
    buffer.mNumberChannels = u32::from(cursor.channels);
    buffer.mDataByteSize = size;
    // SAFETY: the packet bytes live until this fill returns. The converter
    // reads them and does not write them.
    buffer.mData = cursor.data.as_ptr().cast::<c_void>().cast_mut();
    cursor.description.mStartOffset = 0;
    cursor.description.mVariableFramesInPacket = 0;
    cursor.description.mDataByteSize = size;
    if !descriptions.is_null() {
        // SAFETY: `descriptions` is the slot AudioConverter gave us. The
        // description stays in `cursor` until the next callback.
        unsafe { descriptions.write(&mut cursor.description) };
    }
    cursor.sent = true;
    convert::set_packet_count(packets, 1);
    0
}
