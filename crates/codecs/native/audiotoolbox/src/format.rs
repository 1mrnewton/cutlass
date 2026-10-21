//! PCM and AAC-LC stream descriptions.

use objc2_core_audio_types::{
    AudioStreamBasicDescription, kAudioFormatFlagIsFloat, kAudioFormatFlagIsPacked,
    kAudioFormatLinearPCM, kAudioFormatMPEG4AAC,
};

use crate::error::Error;

/// AAC-LC target bitrate. The decoder's magic cookie repeats it.
pub(crate) const BITRATE: u32 = 128_000;

pub(crate) fn check_layout(sample_rate: u32, channels: u16) -> Result<(), Error> {
    if sample_rate == 0 {
        return Err(Error::InvalidInput("audio sample rate is zero"));
    }
    if channels == 0 || channels > 8 {
        return Err(Error::InvalidInput("AAC channel count must be 1 to 8"));
    }
    Ok(())
}

pub(crate) fn pcm(sample_rate: u32, channels: u16) -> AudioStreamBasicDescription {
    let channels = u32::from(channels);
    AudioStreamBasicDescription {
        mSampleRate: f64::from(sample_rate),
        mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
        mBytesPerPacket: 4 * channels,
        mFramesPerPacket: 1,
        mBytesPerFrame: 4 * channels,
        mChannelsPerFrame: channels,
        mBitsPerChannel: 32,
        mReserved: 0,
    }
}

pub(crate) fn aac(sample_rate: u32, channels: u16) -> AudioStreamBasicDescription {
    AudioStreamBasicDescription {
        mSampleRate: f64::from(sample_rate),
        mFormatID: kAudioFormatMPEG4AAC,
        mFormatFlags: 0,
        mBytesPerPacket: 0,
        mFramesPerPacket: 1024,
        mBytesPerFrame: 0,
        mChannelsPerFrame: u32::from(channels),
        mBitsPerChannel: 0,
        mReserved: 0,
    }
}
