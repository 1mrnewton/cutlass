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

/// Every [`Channel`], in WAVEFORMATEXTENSIBLE speaker-bit order.
const SPEAKERS: [Channel; 11] = [
    Channel::FrontLeft,
    Channel::FrontRight,
    Channel::FrontCenter,
    Channel::Lfe,
    Channel::BackLeft,
    Channel::BackRight,
    Channel::FrontLeftOfCenter,
    Channel::FrontRightOfCenter,
    Channel::BackCenter,
    Channel::SideLeft,
    Channel::SideRight,
];

impl Channel {
    /// The WAVEFORMATEXTENSIBLE `dwChannelMask` bit for this speaker.
    ///
    /// The bits run front left, front right, center, LFE, back left, back
    /// right, left and right of center, back center, side left, side right.
    /// A wave file and FLAC (RFC 9639 section 9.1.3) store channels in this
    /// order.
    #[must_use]
    pub const fn wave_bit(self) -> u32 {
        match self {
            Self::FrontLeft => 0x1,
            Self::FrontRight => 0x2,
            Self::FrontCenter => 0x4,
            Self::Lfe => 0x8,
            Self::BackLeft => 0x10,
            Self::BackRight => 0x20,
            Self::FrontLeftOfCenter => 0x40,
            Self::FrontRightOfCenter => 0x80,
            Self::BackCenter => 0x100,
            Self::SideLeft => 0x200,
            Self::SideRight => 0x400,
        }
    }
}

/// A standard speaker set. [`StandardLayout::wave_mask`] is its
/// WAVEFORMATEXTENSIBLE `dwChannelMask`; the channels are stored in that
/// bit order, which is also FLAC's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StandardLayout {
    /// Center.
    Mono,
    /// Left, right.
    Stereo,
    /// 3.0: left, right, center.
    Surround,
    /// Left, right, back left, back right.
    Quad,
    /// 5.0: left, right, center, back left, back right.
    FiveZero,
    /// 5.1: left, right, center, LFE, back left, back right.
    FiveOne,
    /// 6.1: left, right, center, LFE, back center, side left, side right.
    SixOne,
    /// 7.1: left, right, center, LFE, back left, back right, side left, side right.
    SevenOne,
}

impl StandardLayout {
    /// The layout a file with `channels` channels and no channel mask
    /// carries: FLAC's fixed assignment for one through eight channels,
    /// which a wave file without a mask follows too.
    #[must_use]
    pub const fn for_count(channels: usize) -> Option<Self> {
        Some(match channels {
            1 => Self::Mono,
            2 => Self::Stereo,
            3 => Self::Surround,
            4 => Self::Quad,
            5 => Self::FiveZero,
            6 => Self::FiveOne,
            7 => Self::SixOne,
            8 => Self::SevenOne,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn wave_mask(self) -> u32 {
        match self {
            Self::Mono => 0x4,
            Self::Stereo => 0x3,
            Self::Surround => 0x7,
            Self::Quad => 0x33,
            Self::FiveZero => 0x37,
            Self::FiveOne => 0x3F,
            Self::SixOne => 0x70F,
            Self::SevenOne => 0x63F,
        }
    }
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

    /// `layout` in WAVEFORMATEXTENSIBLE order.
    #[must_use]
    pub fn standard(layout: StandardLayout) -> Self {
        Self {
            channels: speakers_of(layout.wave_mask()),
        }
    }

    /// The speakers a WAVEFORMATEXTENSIBLE `dwChannelMask` names, in bit
    /// order. A bit with no [`Channel`] (the top speakers, and the reserved
    /// bits) is [`LayoutError::Speaker`].
    pub fn from_wave_mask(mask: u32) -> Result<Self, LayoutError> {
        let known = SPEAKERS
            .iter()
            .fold(0_u32, |bits, channel| bits | channel.wave_bit());
        if mask & !known != 0 {
            return Err(LayoutError::Speaker);
        }
        Self::new(speakers_of(mask))
    }

    /// The layout an AAC `channelConfiguration` names, in the bitstream
    /// element order the decoder writes. Configuration 0 (a program config
    /// element) and the layouts wider than eight channels are [`None`].
    #[must_use]
    pub fn from_aac_config(config: u8) -> Option<Self> {
        use Channel::{BackCenter, BackLeft, BackRight, FrontCenter, FrontLeft, FrontRight, Lfe};
        let channels = match config {
            1 => return Some(Self::mono()),
            2 => return Some(Self::stereo()),
            3 => vec![FrontCenter, FrontLeft, FrontRight],
            4 => vec![FrontCenter, FrontLeft, FrontRight, BackCenter],
            5 => vec![FrontCenter, FrontLeft, FrontRight, BackLeft, BackRight],
            6 => return Some(Self::five_one()),
            7 => return Some(Self::seven_one_wide()),
            11 => vec![
                FrontCenter,
                FrontLeft,
                FrontRight,
                BackLeft,
                BackRight,
                BackCenter,
                Lfe,
            ],
            12 => return Some(Self::seven_one()),
            _ => return None,
        };
        Some(Self { channels })
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
    /// A channel mask names a speaker [`Channel`] does not have.
    Speaker,
}

fn speakers_of(mask: u32) -> Vec<Channel> {
    SPEAKERS
        .iter()
        .copied()
        .filter(|channel| mask & channel.wave_bit() != 0)
        .collect()
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
    use super::{Channel, ChannelLayout, LayoutError, Pcm, SPEAKERS, SampleFormat, StandardLayout};

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
    fn standard_layouts_follow_the_wave_speaker_order() {
        use Channel::{
            BackCenter, BackLeft, BackRight, FrontCenter, FrontLeft, FrontRight, Lfe, SideLeft,
            SideRight,
        };
        let expected: [(StandardLayout, &[Channel]); 8] = [
            (StandardLayout::Mono, &[FrontCenter]),
            (StandardLayout::Stereo, &[FrontLeft, FrontRight]),
            (
                StandardLayout::Surround,
                &[FrontLeft, FrontRight, FrontCenter],
            ),
            (
                StandardLayout::Quad,
                &[FrontLeft, FrontRight, BackLeft, BackRight],
            ),
            (
                StandardLayout::FiveZero,
                &[FrontLeft, FrontRight, FrontCenter, BackLeft, BackRight],
            ),
            (
                StandardLayout::FiveOne,
                &[FrontLeft, FrontRight, FrontCenter, Lfe, BackLeft, BackRight],
            ),
            (
                StandardLayout::SixOne,
                &[
                    FrontLeft,
                    FrontRight,
                    FrontCenter,
                    Lfe,
                    BackCenter,
                    SideLeft,
                    SideRight,
                ],
            ),
            (
                StandardLayout::SevenOne,
                &[
                    FrontLeft,
                    FrontRight,
                    FrontCenter,
                    Lfe,
                    BackLeft,
                    BackRight,
                    SideLeft,
                    SideRight,
                ],
            ),
        ];
        for (count, (standard, channels)) in expected.into_iter().enumerate() {
            assert_eq!(StandardLayout::for_count(count + 1), Some(standard));
            assert_eq!(ChannelLayout::standard(standard).channels(), channels);
        }
        assert_eq!(
            ChannelLayout::standard(StandardLayout::Mono),
            ChannelLayout::mono()
        );
        assert_eq!(
            ChannelLayout::standard(StandardLayout::Stereo),
            ChannelLayout::stereo()
        );
        assert_eq!(StandardLayout::for_count(0), None);
        assert_eq!(StandardLayout::for_count(9), None);
    }

    #[test]
    fn a_wave_mask_lists_its_speakers_in_bit_order() {
        // 5.1 with side surrounds, as ffmpeg writes it.
        let side = ChannelLayout::from_wave_mask(0x60F).expect("5.1 side");
        assert_eq!(
            side.channels(),
            &[
                Channel::FrontLeft,
                Channel::FrontRight,
                Channel::FrontCenter,
                Channel::Lfe,
                Channel::SideLeft,
                Channel::SideRight,
            ]
        );
        for channel in SPEAKERS {
            let one = ChannelLayout::from_wave_mask(channel.wave_bit()).expect("one speaker");
            assert_eq!(one.channels(), &[channel]);
        }
        assert_eq!(ChannelLayout::from_wave_mask(0), Err(LayoutError::Empty));
        // Top center, and the reserved all-speakers bit, have no channel.
        assert_eq!(
            ChannelLayout::from_wave_mask(0x803),
            Err(LayoutError::Speaker)
        );
        assert_eq!(
            ChannelLayout::from_wave_mask(0x8000_0003),
            Err(LayoutError::Speaker)
        );
        // Nine speakers do not fit a layout.
        assert_eq!(
            ChannelLayout::from_wave_mask(0x1FF),
            Err(LayoutError::TooMany)
        );
    }

    #[test]
    fn aac_configurations_keep_the_element_order() {
        let three = ChannelLayout::from_aac_config(3).expect("3");
        assert_eq!(
            three.channels(),
            &[
                Channel::FrontCenter,
                Channel::FrontLeft,
                Channel::FrontRight
            ]
        );
        assert_eq!(
            ChannelLayout::from_aac_config(6),
            Some(ChannelLayout::five_one())
        );
        assert_eq!(ChannelLayout::from_aac_config(11).map(|l| l.len()), Some(7));
        assert_eq!(
            ChannelLayout::from_aac_config(12),
            Some(ChannelLayout::seven_one())
        );
        for config in [0, 8, 9, 10, 13, 14, 15] {
            assert_eq!(ChannelLayout::from_aac_config(config), None);
        }
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
