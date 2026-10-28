//! AudioSpecificConfig and the layouts this crate will open.
//!
//! Indexed sample rates are the 13 MPEG-4 rates from 96 kHz down to 7.35 kHz.
//! MPEG-2 AAC-LC uses that list through 8 kHz only, frame length 1024, and no
//! spectral band replication. An explicit 24-bit frequency is parsed and then
//! rejected until a band table for that exact rate has been validated; the
//! next-higher indexed table is recorded so the rejection says which table
//! would have been required.
//!
//! Channel order is the AAC element order. Configuration 7 is the wide 7.1
//! layout. Configuration 12 is the common side-surround 7.1 layout.

use cutlass_decode::{Channel, ChannelLayout, CodecError, CodecId, CodecParams};

use crate::bands::{
    LONG_960, LONG_1024, SAMPLE_RATES, SHORT_120, SHORT_128, TNS_MAX_LONG, TNS_MAX_SHORT,
};
use crate::bits::{BitReader, BitSink, BitWriter};

pub const MIN_BITRATE: u32 = 8_000;
pub const MAX_FRAME_BITS_PER_CHANNEL: u32 = 6_144;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    AacLc,
    HeAacV1,
    HeAacV2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateControl {
    /// Target payload rate in bits per second.
    ///
    /// Access units are not padded to a constant size. Over a long signal the
    /// reservoir keeps the mean near `bitrate`. Silence may fall below it.
    Average { bitrate: u32 },
    /// `quality` is 1..=10, with 10 the finest. `cap` is an optional hard
    /// average ceiling in bits per second. There is no promised bitrate.
    Variable { quality: u8, cap: Option<u32> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementKind {
    Sce,
    Cpe,
    Lfe,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioConfig {
    pub profile: Profile,
    pub mpeg2: bool,
    pub sample_rate: u32,
    pub rate_index: Option<u8>,
    pub table_index: u8,
    pub explicit_rate: bool,
    pub channel_config: u8,
    pub layout: ChannelLayout,
    pub frame_length: usize,
    pub sbr: bool,
    pub ps: bool,
    pub elements: Vec<ElementKind>,
    pub asc: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderConfig {
    pub profile: Profile,
    pub mpeg2: bool,
    pub sample_rate: u32,
    pub layout: ChannelLayout,
    pub frame_length: usize,
    pub rate: RateControl,
}

impl EncoderConfig {
    pub fn audio(&self) -> Result<AudioConfig, CodecError> {
        if self.profile != Profile::AacLc {
            return Err(CodecError::Unsupported { what: "sbr" });
        }
        if self.frame_length != 1024 && self.frame_length != 960 {
            return Err(CodecError::Unsupported {
                what: "frame length",
            });
        }
        if self.mpeg2 && self.frame_length != 1024 {
            return Err(CodecError::Unsupported {
                what: "mpeg-2 frame length",
            });
        }
        let Some(channel_config) = config_for_layout(&self.layout) else {
            return Err(CodecError::Unsupported {
                what: "channel configuration",
            });
        };
        let (rate_index, table_index) = match index_of_rate(self.sample_rate) {
            Some(index) => (Some(index), index),
            None => {
                return Err(CodecError::Unsupported {
                    what: "sample rate",
                });
            }
        };
        if self.mpeg2 && rate_index.is_some_and(|index| index > 11) {
            return Err(CodecError::Unsupported {
                what: "mpeg-2 sample rate",
            });
        }
        check_rate_control(
            self.sample_rate,
            self.frame_length,
            self.layout.len(),
            self.rate,
        )?;
        let Some(elements) = elements_for(channel_config) else {
            return Err(CodecError::Unsupported {
                what: "channel configuration",
            });
        };
        let mut config = AudioConfig {
            profile: Profile::AacLc,
            mpeg2: self.mpeg2,
            sample_rate: self.sample_rate,
            rate_index,
            table_index,
            explicit_rate: false,
            channel_config,
            layout: self.layout.clone(),
            frame_length: self.frame_length,
            sbr: false,
            ps: false,
            elements: elements.to_vec(),
            asc: Vec::new(),
        };
        config.asc = write_asc(&config)?;
        Ok(config)
    }
}

impl AudioConfig {
    pub fn from_params(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Aac {
            return Err(CodecError::Unsupported {
                what: "audio object type",
            });
        }
        let mut config = if params.extra.is_empty() {
            let layout = params
                .layout
                .clone()
                .ok_or(CodecError::InvalidApi { what: "layout" })?;
            EncoderConfig {
                profile: Profile::AacLc,
                mpeg2: false,
                sample_rate: params.sample_rate,
                layout,
                frame_length: 1024,
                rate: RateControl::Average {
                    bitrate: default_bitrate(
                        params.layout.as_ref().map(ChannelLayout::len).unwrap_or(1),
                    ),
                },
            }
            .audio()?
        } else {
            parse_asc(&params.extra, false)?
        };
        if config.explicit_rate {
            return Err(CodecError::Unsupported {
                what: "explicit sample rate",
            });
        }
        if config.profile != Profile::AacLc || config.sbr || config.ps {
            return Err(CodecError::Unsupported { what: "sbr" });
        }
        if params.sample_rate != 0 && params.sample_rate != config.sample_rate {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "sample rate",
            });
        }
        config.asc = if params.extra.is_empty() {
            config.asc
        } else {
            params.extra.clone()
        };
        Ok(config)
    }

    pub fn bands(&self, short: bool) -> Result<&'static [u16], CodecError> {
        let index = usize::from(self.table_index);
        let table: &[&[u16]; 13] = match (self.frame_length, short) {
            (1024, false) => &LONG_1024,
            (1024, true) => &SHORT_128,
            (960, false) => &LONG_960,
            (960, true) => &SHORT_120,
            _ => {
                return Err(CodecError::Unsupported {
                    what: "frame length",
                });
            }
        };
        table.get(index).copied().ok_or(CodecError::Unsupported {
            what: "sample rate",
        })
    }

    pub fn tns_max_bands(&self, short: bool) -> usize {
        let index = usize::from(self.table_index);
        let table = if short { TNS_MAX_SHORT } else { TNS_MAX_LONG };
        usize::from(table.get(index).copied().unwrap_or(0))
    }

    pub fn max_bitrate(&self) -> u32 {
        max_bitrate(self.sample_rate, self.frame_length, self.layout.len())
    }
}

pub fn default_bitrate(channels: usize) -> u32 {
    64_000 * u32::try_from(channels).unwrap_or(1)
}

pub fn max_bitrate(sample_rate: u32, frame: usize, channels: usize) -> u32 {
    let channels = u64::try_from(channels).unwrap_or(0);
    let frame = u64::try_from(frame).unwrap_or(1).max(1);
    let bits = u64::from(MAX_FRAME_BITS_PER_CHANNEL) * channels * u64::from(sample_rate) / frame;
    u32::try_from(bits).unwrap_or(u32::MAX)
}

pub fn check_rate_control(
    sample_rate: u32,
    frame: usize,
    channels: usize,
    rate: RateControl,
) -> Result<(), CodecError> {
    match rate {
        RateControl::Average { bitrate } => {
            let max = max_bitrate(sample_rate, frame, channels);
            if bitrate < MIN_BITRATE || bitrate > max {
                return Err(CodecError::InvalidApi { what: "bitrate" });
            }
        }
        RateControl::Variable { quality, cap } => {
            if !(1..=10).contains(&quality) {
                return Err(CodecError::InvalidApi { what: "quality" });
            }
            if let Some(bitrate) = cap {
                let max = max_bitrate(sample_rate, frame, channels);
                if bitrate < MIN_BITRATE || bitrate > max {
                    return Err(CodecError::InvalidApi { what: "bitrate" });
                }
            }
        }
    }
    Ok(())
}

pub fn index_of_rate(rate: u32) -> Option<u8> {
    SAMPLE_RATES
        .iter()
        .position(|&candidate| candidate == rate)
        .and_then(|index| u8::try_from(index).ok())
}

pub fn config_for_layout(layout: &ChannelLayout) -> Option<u8> {
    [1u8, 2, 3, 4, 5, 6, 7, 11, 12].into_iter().find(|&config| {
        layout_for(config).is_some_and(|candidate| candidate.channels() == layout.channels())
    })
}

pub fn elements_for(config: u8) -> Option<&'static [ElementKind]> {
    Some(match config {
        1 => &[ElementKind::Sce],
        2 => &[ElementKind::Cpe],
        3 => &[ElementKind::Sce, ElementKind::Cpe],
        4 => &[ElementKind::Sce, ElementKind::Cpe, ElementKind::Sce],
        5 => &[ElementKind::Sce, ElementKind::Cpe, ElementKind::Cpe],
        6 => &[
            ElementKind::Sce,
            ElementKind::Cpe,
            ElementKind::Cpe,
            ElementKind::Lfe,
        ],
        7 | 12 => &[
            ElementKind::Sce,
            ElementKind::Cpe,
            ElementKind::Cpe,
            ElementKind::Cpe,
            ElementKind::Lfe,
        ],
        11 => &[
            ElementKind::Sce,
            ElementKind::Cpe,
            ElementKind::Cpe,
            ElementKind::Sce,
            ElementKind::Lfe,
        ],
        _ => return None,
    })
}

pub fn layout_for(config: u8) -> Option<ChannelLayout> {
    let channels = match config {
        1 => ChannelLayout::mono().channels().to_vec(),
        2 => ChannelLayout::stereo().channels().to_vec(),
        3 => vec![
            Channel::FrontCenter,
            Channel::FrontLeft,
            Channel::FrontRight,
        ],
        4 => vec![
            Channel::FrontCenter,
            Channel::FrontLeft,
            Channel::FrontRight,
            Channel::BackCenter,
        ],
        5 => vec![
            Channel::FrontCenter,
            Channel::FrontLeft,
            Channel::FrontRight,
            Channel::BackLeft,
            Channel::BackRight,
        ],
        6 => ChannelLayout::five_one().channels().to_vec(),
        7 => ChannelLayout::seven_one_wide().channels().to_vec(),
        11 => vec![
            Channel::FrontCenter,
            Channel::FrontLeft,
            Channel::FrontRight,
            Channel::BackLeft,
            Channel::BackRight,
            Channel::BackCenter,
            Channel::Lfe,
        ],
        12 => ChannelLayout::seven_one().channels().to_vec(),
        _ => return None,
    };
    ChannelLayout::new(channels).ok()
}

pub fn parse_asc(data: &[u8], mpeg2: bool) -> Result<AudioConfig, CodecError> {
    if data.is_empty() || data.len() > 64 {
        return Err(CodecError::Limit {
            what: "audio specific config",
        });
    }
    let mut bits = BitReader::new(data);
    let aot = read_aot(&mut bits)?;
    // The first frequency is the core rate. For object types 5 and 29 the
    // following frequency is the spectral-band-replication output rate.
    let (sample_rate, rate_index, mut explicit_rate) = read_rate(&mut bits)?;
    let channel_config = u8::try_from(bits.read(4)?).unwrap_or(255);
    let mut sbr = false;
    let mut ps = false;
    let mut core = aot;
    if aot == 5 || aot == 29 {
        sbr = true;
        ps = aot == 29;
        let (_output_rate, _output_index, output_explicit) = read_rate(&mut bits)?;
        explicit_rate = explicit_rate || output_explicit;
        core = read_aot(&mut bits)?;
    }
    if core != 2 {
        return Err(named_aot(core));
    }
    let frame_flag = bits.read(1)?;
    if bits.read(1)? == 1 {
        let _delay = bits.read(14)?;
        return Err(CodecError::Unsupported { what: "core coder" });
    }
    let extension_flag = bits.read(1)?;
    let (layout, elements) = if channel_config == 0 {
        let start = bits.position();
        read_pce(&mut bits, start, rate_index)?
    } else {
        let Some(layout) = layout_for(channel_config) else {
            return Err(CodecError::Unsupported {
                what: "channel configuration",
            });
        };
        let Some(elements) = elements_for(channel_config) else {
            return Err(CodecError::Unsupported {
                what: "channel configuration",
            });
        };
        (layout, elements.to_vec())
    };
    if extension_flag == 1 && bits.read(1)? == 1 {
        return Err(CodecError::Unsupported { what: "extension" });
    }
    if bits.bits_left() >= 16 {
        let sync = bits.read(11)?;
        if sync == 0x2b7 {
            let ext = read_aot(&mut bits)?;
            if ext == 5 && bits.read(1)? == 1 {
                let (_output_rate, _, output_explicit) = read_rate(&mut bits)?;
                explicit_rate = explicit_rate || output_explicit;
                sbr = true;
                if bits.bits_left() >= 12 && bits.read(11)? == 0x548 {
                    ps = bits.read(1)? == 1;
                }
            }
        }
    }
    let frame_length = if frame_flag == 1 { 960 } else { 1024 };
    if mpeg2
        && (sbr
            || ps
            || explicit_rate
            || frame_length != 1024
            || rate_index.is_some_and(|i| i > 11))
    {
        return Err(CodecError::Unsupported { what: "mpeg-2 aac" });
    }
    let table_index = match rate_index {
        Some(index) => index,
        None => next_table(sample_rate)?,
    };
    let profile = if ps {
        Profile::HeAacV2
    } else if sbr {
        Profile::HeAacV1
    } else {
        Profile::AacLc
    };
    Ok(AudioConfig {
        profile,
        mpeg2,
        sample_rate,
        rate_index,
        table_index,
        explicit_rate,
        channel_config,
        layout,
        frame_length,
        sbr,
        ps,
        elements,
        asc: data.to_vec(),
    })
}

pub fn write_asc(config: &AudioConfig) -> Result<Vec<u8>, CodecError> {
    if config.profile != Profile::AacLc || config.sbr || config.ps || config.explicit_rate {
        return Err(CodecError::Unsupported { what: "sbr" });
    }
    let Some(index) = config.rate_index else {
        return Err(CodecError::Unsupported {
            what: "explicit sample rate",
        });
    };
    let mut writer = BitWriter::with_cap(32);
    writer.write_bits(2, 5)?;
    writer.write_bits(u32::from(index), 4)?;
    writer.write_bits(u32::from(config.channel_config), 4)?;
    writer.write_bits(u32::from(config.frame_length == 960), 1)?;
    writer.write_bits(0, 1)?;
    writer.write_bits(0, 1)?;
    Ok(writer.finish())
}

fn named_aot(aot: u8) -> CodecError {
    let what = match aot {
        1 => "main",
        3 => "ssr",
        4 => "ltp",
        17 | 19 | 20 | 23 => "er aac",
        39 | 42 => "usac",
        _ => "audio object type",
    };
    CodecError::Unsupported { what }
}

fn read_aot(bits: &mut BitReader<'_>) -> Result<u8, CodecError> {
    let mut aot = bits.read(5)?;
    if aot == 31 {
        aot = 32 + bits.read(6)?;
    }
    u8::try_from(aot).map_err(|_| CodecError::Malformed {
        at_bit: bits.position() as u64,
        what: "audio object type",
    })
}

fn read_rate(bits: &mut BitReader<'_>) -> Result<(u32, Option<u8>, bool), CodecError> {
    let index = bits.read(4)?;
    if index == 0x0f {
        let rate = bits.read(24)?;
        if rate == 0 {
            return Err(CodecError::Malformed {
                at_bit: bits.position() as u64,
                what: "sample rate",
            });
        }
        return Ok((rate, None, true));
    }
    if index > 12 {
        return Err(CodecError::Malformed {
            at_bit: bits.position() as u64,
            what: "sample rate",
        });
    }
    let index = u8::try_from(index).unwrap_or(0);
    Ok((SAMPLE_RATES[usize::from(index)], Some(index), false))
}

fn next_table(rate: u32) -> Result<u8, CodecError> {
    if rate == 0 || rate > 96_000 {
        return Err(CodecError::Unsupported {
            what: "sample rate",
        });
    }
    for (index, &standard) in SAMPLE_RATES.iter().enumerate() {
        if standard >= rate {
            return u8::try_from(index).map_err(|_| CodecError::Unsupported {
                what: "sample rate",
            });
        }
    }
    Err(CodecError::Unsupported {
        what: "sample rate",
    })
}

fn read_pce(
    bits: &mut BitReader<'_>,
    start_bit: usize,
    expected_rate: Option<u8>,
) -> Result<(ChannelLayout, Vec<ElementKind>), CodecError> {
    let _tag = bits.read(4)?;
    let _object = bits.read(2)?;
    let rate_index = bits.read(4)?;
    if let Some(expected) = expected_rate
        && rate_index != u32::from(expected)
    {
        return Err(CodecError::Malformed {
            at_bit: bits.position() as u64,
            what: "sample rate",
        });
    }
    let num_front = bits.read(4)?;
    let num_side = bits.read(4)?;
    let num_back = bits.read(4)?;
    let num_lfe = bits.read(2)?;
    let num_assoc = bits.read(3)?;
    let num_cc = bits.read(4)?;
    if num_cc > 0 {
        return Err(CodecError::Unsupported { what: "coupling" });
    }
    if num_front > 8 || num_side > 8 || num_back > 8 || num_lfe > 2 || num_assoc > 8 {
        return Err(CodecError::Limit {
            what: "program config",
        });
    }
    if bits.read(1)? == 1 {
        let _mix = bits.read(4)?;
    }
    if bits.read(1)? == 1 {
        let _mix = bits.read(4)?;
    }
    if bits.read(1)? == 1 {
        let _idx = bits.read(2)?;
        let _surround = bits.read(1)?;
    }
    let mut channels = Vec::new();
    let mut elements = Vec::new();
    read_speaker_elements(
        bits,
        num_front,
        SpeakerGroup::Front,
        &mut channels,
        &mut elements,
    )?;
    read_speaker_elements(
        bits,
        num_side,
        SpeakerGroup::Side,
        &mut channels,
        &mut elements,
    )?;
    read_speaker_elements(
        bits,
        num_back,
        SpeakerGroup::Back,
        &mut channels,
        &mut elements,
    )?;
    for _ in 0..num_lfe {
        let _tag = bits.read(4)?;
        channels.push(Channel::Lfe);
        elements.push(ElementKind::Lfe);
    }
    for _ in 0..num_assoc {
        let _tag = bits.read(4)?;
    }
    align_from(bits, start_bit)?;
    let comment = bits.read(8)? as usize;
    for _ in 0..comment {
        let _byte = bits.read(8)?;
    }
    if channels.is_empty() || channels.len() > 8 {
        return Err(CodecError::Unsupported {
            what: "channel configuration",
        });
    }
    let layout = ChannelLayout::new(channels).map_err(|_| CodecError::Unsupported {
        what: "channel configuration",
    })?;
    Ok((layout, elements))
}

enum SpeakerGroup {
    Front,
    Side,
    Back,
}

fn read_speaker_elements(
    bits: &mut BitReader<'_>,
    count: u32,
    group: SpeakerGroup,
    channels: &mut Vec<Channel>,
    elements: &mut Vec<ElementKind>,
) -> Result<(), CodecError> {
    let mut saw_center = false;
    let mut saw_pair = false;
    for _ in 0..count {
        let is_cpe = bits.read(1)? == 1;
        let _tag = bits.read(4)?;
        if is_cpe {
            if saw_pair {
                return Err(CodecError::Unsupported {
                    what: "program config",
                });
            }
            saw_pair = true;
            let pair = match group {
                SpeakerGroup::Front => [Channel::FrontLeft, Channel::FrontRight],
                SpeakerGroup::Side => [Channel::SideLeft, Channel::SideRight],
                SpeakerGroup::Back => [Channel::BackLeft, Channel::BackRight],
            };
            channels.extend(pair);
            elements.push(ElementKind::Cpe);
        } else {
            let channel = match group {
                SpeakerGroup::Front => Channel::FrontCenter,
                SpeakerGroup::Side => {
                    return Err(CodecError::Unsupported {
                        what: "program config",
                    });
                }
                SpeakerGroup::Back => Channel::BackCenter,
            };
            if saw_center {
                return Err(CodecError::Unsupported {
                    what: "program config",
                });
            }
            saw_center = true;
            channels.push(channel);
            elements.push(ElementKind::Sce);
        }
    }
    Ok(())
}

fn align_from(bits: &mut BitReader<'_>, start_bit: usize) -> Result<(), CodecError> {
    let rel = bits.position().saturating_sub(start_bit) % 8;
    if rel != 0 {
        let _pad = bits.read(u32::try_from(8 - rel).unwrap_or(0))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{AudioConfig, EncoderConfig, Profile, RateControl, parse_asc, write_asc};
    use cutlass_decode::{ChannelLayout, CodecError};

    #[test]
    fn lc_stereo_configuration_round_trips() {
        let config = EncoderConfig {
            profile: Profile::AacLc,
            mpeg2: false,
            sample_rate: 48_000,
            layout: ChannelLayout::stereo(),
            frame_length: 1024,
            rate: RateControl::Average { bitrate: 128_000 },
        }
        .audio()
        .unwrap();
        let parsed = parse_asc(&config.asc, false).unwrap();
        assert_eq!(parsed.profile, Profile::AacLc);
        assert_eq!(parsed.sample_rate, 48_000);
        assert_eq!(parsed.channel_config, 2);
        assert_eq!(parsed.frame_length, 1024);
        assert!(!parsed.sbr);
        let again = write_asc(&parsed).unwrap();
        assert_eq!(again, config.asc);
    }

    #[test]
    fn nine_sixty_and_seven_three_five_zero_are_signaled() {
        let config = EncoderConfig {
            profile: Profile::AacLc,
            mpeg2: false,
            sample_rate: 7_350,
            layout: ChannelLayout::mono(),
            frame_length: 960,
            rate: RateControl::Average { bitrate: 16_000 },
        }
        .audio()
        .unwrap();
        let parsed = parse_asc(&config.asc, false).unwrap();
        assert_eq!(parsed.frame_length, 960);
        assert_eq!(parsed.rate_index, Some(12));
    }

    #[test]
    fn mpeg2_rejects_the_lowest_mpeg4_rate_and_short_frames() {
        let err = EncoderConfig {
            profile: Profile::AacLc,
            mpeg2: true,
            sample_rate: 7_350,
            layout: ChannelLayout::mono(),
            frame_length: 1024,
            rate: RateControl::Average { bitrate: 16_000 },
        }
        .audio()
        .unwrap_err();
        assert!(matches!(err, CodecError::Unsupported { .. }));
        let err = EncoderConfig {
            profile: Profile::AacLc,
            mpeg2: true,
            sample_rate: 48_000,
            layout: ChannelLayout::mono(),
            frame_length: 960,
            rate: RateControl::Average { bitrate: 64_000 },
        }
        .audio()
        .unwrap_err();
        assert!(matches!(err, CodecError::Unsupported { .. }));
    }

    #[test]
    fn main_and_sbr_and_explicit_rates_are_rejected() {
        // AOT 1, 48 kHz, stereo, then a dummy GA bit so the parser reaches the object type.
        let err = parse_asc(&[0x08, 0x10], false).unwrap_err();
        assert_eq!(err, CodecError::Unsupported { what: "main" });
        let err = EncoderConfig {
            profile: Profile::HeAacV1,
            mpeg2: false,
            sample_rate: 48_000,
            layout: ChannelLayout::stereo(),
            frame_length: 1024,
            rate: RateControl::Average { bitrate: 48_000 },
        }
        .audio()
        .unwrap_err();
        assert_eq!(err, CodecError::Unsupported { what: "sbr" });
        // AOT 2, explicit rate marker 0xF, 24-bit rate 40000 = 0x009C40, channel 1, GA flags 0.
        let bytes = [0x17, 0x80, 0x09, 0xC4, 0x08];
        let parsed = parse_asc(&bytes, false).unwrap();
        assert!(parsed.explicit_rate);
        let opened = AudioConfig::from_params(&cutlass_decode::CodecParams {
            extra: bytes.to_vec(),
            ..cutlass_decode::CodecParams::audio(
                cutlass_decode::CodecId::Aac,
                40_000,
                ChannelLayout::mono(),
            )
        });
        assert!(matches!(
            opened,
            Err(CodecError::Unsupported {
                what: "explicit sample rate"
            })
        ));
    }

    #[test]
    fn impossible_bitrates_are_rejected() {
        let err = EncoderConfig {
            profile: Profile::AacLc,
            mpeg2: false,
            sample_rate: 48_000,
            layout: ChannelLayout::mono(),
            frame_length: 1024,
            rate: RateControl::Average { bitrate: 1_000 },
        }
        .audio()
        .unwrap_err();
        assert_eq!(err, CodecError::InvalidApi { what: "bitrate" });
        let err = EncoderConfig {
            profile: Profile::AacLc,
            mpeg2: false,
            sample_rate: 48_000,
            layout: ChannelLayout::mono(),
            frame_length: 1024,
            rate: RateControl::Average {
                bitrate: 10_000_000,
            },
        }
        .audio()
        .unwrap_err();
        assert_eq!(err, CodecError::InvalidApi { what: "bitrate" });
    }
}
