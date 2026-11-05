//! Layer III side information.
//!
//! ISO/IEC 11172-3 clause 2.4.1.7 and ISO/IEC 13818-3 clause 2.4.1.7 (LSF:
//! one granule, 8-bit `main_data_begin`, 9-bit `scalefac_compress`, no
//! `scfsi` and no `preflag` bit).

use cutlass_decode::CodecError;

use crate::bits::BitReader;
#[cfg(test)]
use crate::bits::BitWriter;
use crate::header::FrameHeader;

/// Block type 2, three short windows.
pub(crate) const SHORT: u8 = 2;

/// Side information of one granule of one channel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GranuleInfo {
    pub part2_3_length: u16,
    pub big_values: u16,
    pub global_gain: u8,
    pub scalefac_compress: u16,
    pub window_switching: bool,
    /// 0 normal, 1 start, 2 short, 3 stop.
    pub block_type: u8,
    pub mixed: bool,
    pub table_select: [u8; 3],
    pub subblock_gain: [u8; 3],
    pub region0_count: u8,
    pub region1_count: u8,
    pub preflag: bool,
    pub scalefac_scale: bool,
    pub count1_table_b: bool,
}

/// Side information of one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SideInfo {
    pub main_data_begin: u16,
    pub private_bits: u8,
    /// `scfsi[ch][group]`, MPEG-1 only.
    pub scfsi: [[bool; 4]; 2],
    /// `granules[gr][ch]`.
    pub granules: [[GranuleInfo; 2]; 2],
}

impl SideInfo {
    /// Parse the side information that follows the header and CRC.
    ///
    /// `big_values` above 288 and a window-switching granule with block
    /// type 0 are [`CodecError::Malformed`].
    pub(crate) fn parse(header: &FrameHeader, bytes: &[u8]) -> Result<Self, CodecError> {
        let mut reader = BitReader::new(bytes);
        let lsf = header.version.lsf();
        let channels = header.channels();
        let mut side = Self::default();
        if lsf {
            side.main_data_begin = reader.read(8, "main_data_begin")? as u16;
            side.private_bits = reader.read(channels as u32, "private_bits")? as u8;
        } else {
            side.main_data_begin = reader.read(9, "main_data_begin")? as u16;
            let private = if channels == 1 { 5 } else { 3 };
            side.private_bits = reader.read(private, "private_bits")? as u8;
            for ch in 0..channels {
                for group in 0..4 {
                    side.scfsi[ch][group] = reader.bit("scfsi")?;
                }
            }
        }
        for gr in 0..header.granules() {
            for ch in 0..channels {
                let at = reader.pos() as u64;
                let mut g = GranuleInfo {
                    part2_3_length: reader.read(12, "part2_3_length")? as u16,
                    big_values: reader.read(9, "big_values")? as u16,
                    global_gain: reader.read(8, "global_gain")? as u8,
                    scalefac_compress: reader.read(if lsf { 9 } else { 4 }, "scalefac_compress")?
                        as u16,
                    window_switching: reader.bit("window_switching_flag")?,
                    ..GranuleInfo::default()
                };
                if g.big_values > 288 {
                    return Err(CodecError::Malformed {
                        at_bit: at + 12,
                        what: "big_values",
                    });
                }
                if g.window_switching {
                    g.block_type = reader.read(2, "block_type")? as u8;
                    if g.block_type == 0 {
                        return Err(CodecError::Malformed {
                            at_bit: reader.pos() as u64 - 2,
                            what: "block_type",
                        });
                    }
                    g.mixed = reader.bit("mixed_block_flag")?;
                    for slot in 0..2 {
                        g.table_select[slot] = reader.read(5, "table_select")? as u8;
                    }
                    for slot in 0..3 {
                        g.subblock_gain[slot] = reader.read(3, "subblock_gain")? as u8;
                    }
                } else {
                    for slot in 0..3 {
                        g.table_select[slot] = reader.read(5, "table_select")? as u8;
                    }
                    g.region0_count = reader.read(4, "region0_count")? as u8;
                    g.region1_count = reader.read(3, "region1_count")? as u8;
                }
                if !lsf {
                    g.preflag = reader.bit("preflag")?;
                }
                g.scalefac_scale = reader.bit("scalefac_scale")?;
                g.count1_table_b = reader.bit("count1table_select")?;
                side.granules[gr][ch] = g;
            }
        }
        Ok(side)
    }

    /// Write side information in the layout [`Self::parse`] reads.
    #[cfg(test)]
    pub(crate) fn write(&self, header: &FrameHeader, writer: &mut BitWriter) {
        let lsf = header.version.lsf();
        let channels = header.channels();
        if lsf {
            writer.put(u32::from(self.main_data_begin), 8);
            writer.put(u32::from(self.private_bits), channels as u32);
        } else {
            writer.put(u32::from(self.main_data_begin), 9);
            writer.put(
                u32::from(self.private_bits),
                if channels == 1 { 5 } else { 3 },
            );
            for ch in 0..channels {
                for group in 0..4 {
                    writer.put(u32::from(self.scfsi[ch][group]), 1);
                }
            }
        }
        for gr in 0..header.granules() {
            for ch in 0..channels {
                let g = &self.granules[gr][ch];
                writer.put(u32::from(g.part2_3_length), 12);
                writer.put(u32::from(g.big_values), 9);
                writer.put(u32::from(g.global_gain), 8);
                writer.put(u32::from(g.scalefac_compress), if lsf { 9 } else { 4 });
                writer.put(u32::from(g.window_switching), 1);
                if g.window_switching {
                    writer.put(u32::from(g.block_type), 2);
                    writer.put(u32::from(g.mixed), 1);
                    for slot in 0..2 {
                        writer.put(u32::from(g.table_select[slot]), 5);
                    }
                    for slot in 0..3 {
                        writer.put(u32::from(g.subblock_gain[slot]), 3);
                    }
                } else {
                    for slot in 0..3 {
                        writer.put(u32::from(g.table_select[slot]), 5);
                    }
                    writer.put(u32::from(g.region0_count), 4);
                    writer.put(u32::from(g.region1_count), 3);
                }
                if !lsf {
                    writer.put(u32::from(g.preflag), 1);
                }
                writer.put(u32::from(g.scalefac_scale), 1);
                writer.put(u32::from(g.count1_table_b), 1);
            }
        }
    }
}

/// CRC-16 of ISO/IEC 11172-3 clause 2.4.3.1: generator
/// `x^16 + x^15 + x^2 + 1`, initial value `0xFFFF`, most significant bit
/// first, over the last 16 header bits and then the side information.
pub(crate) fn crc16(header: [u8; 4], side: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &byte in header[2..].iter().chain(side.iter()) {
        crc ^= u16::from(byte) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::{GranuleInfo, SideInfo};
    use crate::bits::BitWriter;
    use crate::header::{FrameHeader, Mode, Version};

    fn header(version: Version, mode: Mode) -> FrameHeader {
        FrameHeader {
            version,
            layer: 3,
            protected: false,
            bitrate_index: 9,
            bitrate_kbps: 0,
            rate_index: 0,
            sample_rate: 44_100,
            padding: false,
            private: false,
            mode,
            mode_extension: 0,
            copyright: false,
            original: false,
            emphasis: 0,
        }
    }

    fn sample() -> SideInfo {
        let mut side = SideInfo {
            main_data_begin: 200,
            private_bits: 0,
            scfsi: [[true, false, true, false], [false, true, false, true]],
            ..SideInfo::default()
        };
        side.granules[0][0] = GranuleInfo {
            part2_3_length: 1234,
            big_values: 288,
            global_gain: 170,
            scalefac_compress: 9,
            table_select: [24, 13, 1],
            region0_count: 7,
            region1_count: 5,
            preflag: true,
            count1_table_b: true,
            ..GranuleInfo::default()
        };
        side.granules[0][1] = GranuleInfo {
            part2_3_length: 99,
            big_values: 10,
            global_gain: 140,
            scalefac_compress: 3,
            window_switching: true,
            block_type: 2,
            mixed: true,
            table_select: [5, 6, 0],
            subblock_gain: [1, 2, 7],
            scalefac_scale: true,
            ..GranuleInfo::default()
        };
        side.granules[1] = side.granules[0];
        side
    }

    #[test]
    fn side_info_round_trips_for_mpeg1_stereo() {
        let h = header(Version::Mpeg1, Mode::JointStereo);
        let side = sample();
        let mut writer = BitWriter::new();
        side.write(&h, &mut writer);
        assert_eq!(writer.bits(), 256);
        let bytes = writer.finish();
        assert_eq!(SideInfo::parse(&h, &bytes).expect("parse"), side);
    }

    #[test]
    fn lsf_mono_side_info_is_nine_bytes() {
        let h = header(Version::Mpeg2, Mode::Mono);
        let mut side = SideInfo {
            main_data_begin: 255,
            ..SideInfo::default()
        };
        side.granules[0][0] = GranuleInfo {
            part2_3_length: 500,
            big_values: 100,
            global_gain: 150,
            scalefac_compress: 511,
            table_select: [1, 2, 3],
            ..GranuleInfo::default()
        };
        let mut writer = BitWriter::new();
        side.write(&h, &mut writer);
        assert_eq!(writer.bits(), 72);
        let bytes = writer.finish();
        assert_eq!(SideInfo::parse(&h, &bytes).expect("parse"), side);
    }

    #[test]
    fn oversized_big_values_and_block_type_zero_are_malformed() {
        let h = header(Version::Mpeg1, Mode::Mono);
        let mut side = SideInfo::default();
        side.granules[0][0].big_values = 289;
        let mut writer = BitWriter::new();
        side.write(&h, &mut writer);
        assert!(matches!(
            SideInfo::parse(&h, &writer.finish()),
            Err(cutlass_decode::CodecError::Malformed {
                what: "big_values",
                ..
            })
        ));
        let mut side = SideInfo::default();
        side.granules[0][0].window_switching = true;
        side.granules[0][0].block_type = 0;
        let mut writer = BitWriter::new();
        side.write(&h, &mut writer);
        assert!(matches!(
            SideInfo::parse(&h, &writer.finish()),
            Err(cutlass_decode::CodecError::Malformed {
                what: "block_type",
                ..
            })
        ));
    }

    #[test]
    fn crc_matches_the_check_value_of_the_generator() {
        // CRC-16/CMS ("x^16+x^15+x^2+1", init 0xFFFF, not reflected) of
        // "123456789" is 0xAEE7. The header bytes before index 2 are skipped.
        assert_eq!(super::crc16([0, 0, b'1', b'2'], b"3456789"), 0xAEE7);
    }
}
