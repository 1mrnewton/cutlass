/// Interleaved PCM. Full scale for [`SampleFormat::F32`] is 1.0.
/// Full scale for [`SampleFormat::I16`] is 32767.
///
/// Compressed-audio bit allocation is not a PCM width. An AAC packet does
/// not have a PCM bit depth; the decoder always writes [`SampleFormat::F32`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    F32,
    I16,
}

/// A speaker, in the order stored in [`Pcm`].
///
/// AAC indexed layouts use the bitstream element order, not the order a
/// wave file usually stores. Each channel keeps its label so a later
/// remap can be explicit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channel {
    FrontLeft,
    FrontRight,
    FrontCenter,
    FrontLeftOfCenter,
    FrontRightOfCenter,
    Lfe,
    BackLeft,
    BackRight,
    BackCenter,
    SideLeft,
    SideRight,
}

/// Channel identity and buffer order. `channels[i]` is sample `i` inside
/// each interleaved frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelLayout {
    channels: Vec<Channel>,
}

impl ChannelLayout {
    pub fn new(channels: Vec<Channel>) -> Result<Self, LayoutError> {
        if channels.is_empty() {
            return Err(LayoutError::Empty);
        }
        if channels.len() > 8 {
            return Err(LayoutError::TooMany);
        }
        Ok(Self { channels })
    }

    #[must_use]
    pub fn mono() -> Self {
        Self {
            channels: vec![Channel::FrontCenter],
        }
    }

    #[must_use]
    pub fn stereo() -> Self {
        Self {
            channels: vec![Channel::FrontLeft, Channel::FrontRight],
        }
    }

    /// 5.1 in AAC channel-configuration 6 order.
    #[must_use]
    pub fn five_one() -> Self {
        Self {
            channels: vec![
                Channel::FrontCenter,
                Channel::FrontLeft,
                Channel::FrontRight,
                Channel::BackLeft,
                Channel::BackRight,
                Channel::Lfe,
            ],
        }
    }

    /// 7.1 wide in AAC channel-configuration 7 order.
    #[must_use]
    pub fn seven_one_wide() -> Self {
        Self {
            channels: vec![
                Channel::FrontCenter,
                Channel::FrontLeftOfCenter,
                Channel::FrontRightOfCenter,
                Channel::FrontLeft,
                Channel::FrontRight,
                Channel::BackLeft,
                Channel::BackRight,
                Channel::Lfe,
            ],
        }
    }

    /// 7.1 in AAC channel-configuration 12 order: front, side, back, LFE.
    #[must_use]
    pub fn seven_one() -> Self {
        Self {
            channels: vec![
                Channel::FrontCenter,
                Channel::FrontLeft,
                Channel::FrontRight,
                Channel::SideLeft,
                Channel::SideRight,
                Channel::BackLeft,
                Channel::BackRight,
                Channel::Lfe,
            ],
        }
    }

    #[must_use]
    pub fn channels(&self) -> &[Channel] {
        &self.channels
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutError {
    Empty,
    TooMany,
}

/// Owned interleaved PCM.
///
/// `samples` is the number of frames. `data` length is `samples * channels`
/// for the active format. [`SampleFormat::F32`] stores `f32` bits in
/// [`Pcm::f32`]; [`SampleFormat::I16`] stores `i16` in [`Pcm::i16`].
#[derive(Clone, Debug, PartialEq)]
pub struct Pcm {
    pub format: SampleFormat,
    pub rate: u32,
    pub layout: ChannelLayout,
    pub samples: u32,
    f32_data: Vec<f32>,
    i16_data: Vec<i16>,
}

impl Pcm {
    pub fn from_f32(rate: u32, layout: ChannelLayout, samples: Vec<f32>) -> Result<Self, PcmError> {
        check_len(layout.len(), samples.len())?;
        let count = samples.len() / layout.len();
        let count = u32::try_from(count).map_err(|_| PcmError::TooManySamples)?;
        Ok(Self {
            format: SampleFormat::F32,
            rate,
            layout,
            samples: count,
            f32_data: samples,
            i16_data: Vec::new(),
        })
    }

    pub fn from_i16(rate: u32, layout: ChannelLayout, samples: Vec<i16>) -> Result<Self, PcmError> {
        check_len(layout.len(), samples.len())?;
        let count = samples.len() / layout.len();
        let count = u32::try_from(count).map_err(|_| PcmError::TooManySamples)?;
        Ok(Self {
            format: SampleFormat::I16,
            rate,
            layout,
            samples: count,
            f32_data: Vec::new(),
            i16_data: samples,
        })
    }

    #[must_use]
    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.format == SampleFormat::F32 {
            Some(&self.f32_data)
        } else {
            None
        }
    }

    #[must_use]
    pub fn as_i16(&self) -> Option<&[i16]> {
        if self.format == SampleFormat::I16 {
            Some(&self.i16_data)
        } else {
            None
        }
    }
}

fn check_len(channels: usize, len: usize) -> Result<(), PcmError> {
    if channels == 0 || !len.is_multiple_of(channels) {
        return Err(PcmError::Uneven);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcmError {
    Uneven,
    TooManySamples,
}

#[cfg(test)]
mod tests {
    use super::{Channel, ChannelLayout, Pcm, SampleFormat};

    #[test]
    fn stereo_order_is_left_then_right() {
        let layout = ChannelLayout::stereo();
        assert_eq!(
            layout.channels(),
            &[Channel::FrontLeft, Channel::FrontRight]
        );
    }

    #[test]
    fn five_one_uses_aac_element_order() {
        let layout = ChannelLayout::five_one();
        assert_eq!(layout.len(), 6);
        assert_eq!(layout.channels()[0], Channel::FrontCenter);
        assert_eq!(layout.channels()[5], Channel::Lfe);
    }

    #[test]
    fn rejects_more_than_eight_channels() {
        let channels = vec![Channel::FrontLeft; 9];
        assert!(ChannelLayout::new(channels).is_err());
    }

    #[test]
    fn f32_length_must_match_channels() {
        let err = Pcm::from_f32(48_000, ChannelLayout::stereo(), vec![0.0, 0.1, 0.2]);
        assert!(err.is_err());
    }

    #[test]
    fn i16_and_f32_are_different_storage() {
        let pcm = Pcm::from_i16(48_000, ChannelLayout::mono(), vec![0, 32767]).expect("pcm");
        assert_eq!(pcm.format, SampleFormat::I16);
        assert_eq!(pcm.samples, 2);
        assert!(pcm.as_f32().is_none());
        assert_eq!(pcm.as_i16(), Some(&[0, 32767][..]));
    }
}
