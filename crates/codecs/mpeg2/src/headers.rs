//! Sequence, GOP, and picture headers and their extensions (H.262 6.2.2,
//! 6.2.3), read from and written to the bytes after their start code.

use cutlass_decode::CodecError;

use crate::bits::{BitReader, BitWriter};
use crate::tables::SCAN;

pub(crate) const PICTURE_START: u8 = 0x00;
pub(crate) const SLICE_FIRST: u8 = 0x01;
pub(crate) const SLICE_LAST: u8 = 0xaf;
pub(crate) const USER_DATA: u8 = 0xb2;
pub(crate) const SEQUENCE_HEADER: u8 = 0xb3;
pub(crate) const SEQUENCE_ERROR: u8 = 0xb4;
pub(crate) const EXTENSION: u8 = 0xb5;
pub(crate) const SEQUENCE_END: u8 = 0xb7;
pub(crate) const GROUP: u8 = 0xb8;

pub(crate) const EXT_SEQUENCE: u8 = 1;
pub(crate) const EXT_SEQUENCE_DISPLAY: u8 = 2;
pub(crate) const EXT_QUANT_MATRIX: u8 = 3;
pub(crate) const EXT_SEQUENCE_SCALABLE: u8 = 5;
pub(crate) const EXT_PICTURE_CODING: u8 = 8;
pub(crate) const EXT_PICTURE_SPATIAL: u8 = 9;
pub(crate) const EXT_PICTURE_TEMPORAL: u8 = 10;

pub(crate) const PICTURE_I: u8 = 1;
pub(crate) const PICTURE_P: u8 = 2;
pub(crate) const PICTURE_B: u8 = 3;
pub(crate) const PICTURE_D: u8 = 4;

pub(crate) const STRUCTURE_TOP: u8 = 1;
#[cfg(test)]
pub(crate) const STRUCTURE_BOTTOM: u8 = 2;
pub(crate) const STRUCTURE_FRAME: u8 = 3;

pub(crate) const CHROMA_420: u8 = 1;
pub(crate) const CHROMA_422: u8 = 2;
#[cfg(test)]
pub(crate) const CHROMA_444: u8 = 3;

/// A quantiser matrix in raster order.
pub(crate) type Matrix = [u8; 64];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SequenceHeader {
    pub horizontal_size_value: u16,
    pub vertical_size_value: u16,
    pub aspect_ratio_information: u8,
    pub frame_rate_code: u8,
    pub bit_rate_value: u32,
    pub vbv_buffer_size_value: u16,
    pub constrained_parameters: bool,
    pub intra_matrix: Option<Matrix>,
    pub non_intra_matrix: Option<Matrix>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SequenceExtension {
    pub profile_and_level: u8,
    pub progressive_sequence: bool,
    pub chroma_format: u8,
    pub horizontal_size_extension: u8,
    pub vertical_size_extension: u8,
    pub bit_rate_extension: u16,
    pub vbv_buffer_size_extension: u8,
    pub low_delay: bool,
    pub frame_rate_extension_n: u8,
    pub frame_rate_extension_d: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SequenceDisplayExtension {
    pub video_format: u8,
    /// `colour_primaries`, `transfer_characteristics`, `matrix_coefficients`.
    pub colour: Option<(u8, u8, u8)>,
    pub display_horizontal_size: u16,
    pub display_vertical_size: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GopHeader {
    pub time_code: u32,
    pub closed_gop: bool,
    pub broken_link: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PictureHeader {
    pub temporal_reference: u16,
    pub picture_coding_type: u8,
    pub vbv_delay: u16,
    pub full_pel_forward: bool,
    pub forward_f_code: u8,
    pub full_pel_backward: bool,
    pub backward_f_code: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PictureCodingExtension {
    /// `f_code[s][t]`: `s` forward 0 or backward 1, `t` horizontal 0 or vertical 1.
    pub f_code: [[u8; 2]; 2],
    pub intra_dc_precision: u8,
    pub picture_structure: u8,
    pub top_field_first: bool,
    pub frame_pred_frame_dct: bool,
    pub concealment_motion_vectors: bool,
    pub q_scale_type: bool,
    pub intra_vlc_format: bool,
    pub alternate_scan: bool,
    pub repeat_first_field: bool,
    pub chroma_420_type: bool,
    pub progressive_frame: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct QuantMatrixExtension {
    pub intra: Option<Matrix>,
    pub non_intra: Option<Matrix>,
    pub chroma_intra: Option<Matrix>,
    pub chroma_non_intra: Option<Matrix>,
}

/// Reads 64 bytes sent in zigzag order into raster order (7.3.1).
fn read_matrix(reader: &mut BitReader<'_>) -> Result<Matrix, CodecError> {
    let mut matrix = [0u8; 64];
    for n in 0..64 {
        let at = reader.at_bit();
        let value = reader.read_checked(8, "quantiser matrix")? as u8;
        if value == 0 {
            return Err(CodecError::Malformed {
                at_bit: at,
                what: "zero quantiser matrix entry",
            });
        }
        matrix[usize::from(SCAN[0][n])] = value;
    }
    Ok(matrix)
}

fn write_matrix(writer: &mut BitWriter, matrix: &Matrix) {
    for n in 0..64 {
        writer.put(8, u32::from(matrix[usize::from(SCAN[0][n])]));
    }
}

impl SequenceHeader {
    pub(crate) fn parse(payload: &[u8]) -> Result<Self, CodecError> {
        let mut r = BitReader::new(payload);
        let horizontal_size_value = r.read_checked(12, "horizontal_size_value")? as u16;
        let vertical_size_value = r.read_checked(12, "vertical_size_value")? as u16;
        let aspect_ratio_information = r.read_checked(4, "aspect_ratio_information")? as u8;
        let frame_rate_code = r.read_checked(4, "frame_rate_code")? as u8;
        let bit_rate_value = r.read_checked(18, "bit_rate_value")?;
        r.marker("sequence header marker")?;
        let vbv_buffer_size_value = r.read_checked(10, "vbv_buffer_size_value")? as u16;
        let constrained_parameters = r.read_checked(1, "constrained_parameters_flag")? == 1;
        let intra_matrix = if r.read_checked(1, "load_intra_quantiser_matrix")? == 1 {
            Some(read_matrix(&mut r)?)
        } else {
            None
        };
        let non_intra_matrix = if r.read_checked(1, "load_non_intra_quantiser_matrix")? == 1 {
            Some(read_matrix(&mut r)?)
        } else {
            None
        };
        Ok(Self {
            horizontal_size_value,
            vertical_size_value,
            aspect_ratio_information,
            frame_rate_code,
            bit_rate_value,
            vbv_buffer_size_value,
            constrained_parameters,
            intra_matrix,
            non_intra_matrix,
        })
    }

    pub(crate) fn write(&self, w: &mut BitWriter) {
        w.start_code(SEQUENCE_HEADER);
        w.put(12, u32::from(self.horizontal_size_value));
        w.put(12, u32::from(self.vertical_size_value));
        w.put(4, u32::from(self.aspect_ratio_information));
        w.put(4, u32::from(self.frame_rate_code));
        w.put(18, self.bit_rate_value);
        w.put_bit(true);
        w.put(10, u32::from(self.vbv_buffer_size_value));
        w.put_bit(self.constrained_parameters);
        w.put_bit(self.intra_matrix.is_some());
        if let Some(matrix) = &self.intra_matrix {
            write_matrix(w, matrix);
        }
        w.put_bit(self.non_intra_matrix.is_some());
        if let Some(matrix) = &self.non_intra_matrix {
            write_matrix(w, matrix);
        }
    }
}

/// The 4-bit identifier that starts every extension payload.
pub(crate) fn extension_id(payload: &[u8]) -> Option<u8> {
    payload.first().map(|byte| byte >> 4)
}

impl SequenceExtension {
    pub(crate) fn parse(payload: &[u8]) -> Result<Self, CodecError> {
        let mut r = BitReader::new(payload);
        r.skip(4);
        let profile_and_level = r.read_checked(8, "profile_and_level_indication")? as u8;
        let progressive_sequence = r.read_checked(1, "progressive_sequence")? == 1;
        let at = r.at_bit();
        let chroma_format = r.read_checked(2, "chroma_format")? as u8;
        if chroma_format == 0 {
            return Err(CodecError::Malformed {
                at_bit: at,
                what: "reserved chroma_format",
            });
        }
        let horizontal_size_extension = r.read_checked(2, "horizontal_size_extension")? as u8;
        let vertical_size_extension = r.read_checked(2, "vertical_size_extension")? as u8;
        let bit_rate_extension = r.read_checked(12, "bit_rate_extension")? as u16;
        r.marker("sequence extension marker")?;
        let vbv_buffer_size_extension = r.read_checked(8, "vbv_buffer_size_extension")? as u8;
        let low_delay = r.read_checked(1, "low_delay")? == 1;
        let frame_rate_extension_n = r.read_checked(2, "frame_rate_extension_n")? as u8;
        let frame_rate_extension_d = r.read_checked(5, "frame_rate_extension_d")? as u8;
        Ok(Self {
            profile_and_level,
            progressive_sequence,
            chroma_format,
            horizontal_size_extension,
            vertical_size_extension,
            bit_rate_extension,
            vbv_buffer_size_extension,
            low_delay,
            frame_rate_extension_n,
            frame_rate_extension_d,
        })
    }

    pub(crate) fn write(&self, w: &mut BitWriter) {
        w.start_code(EXTENSION);
        w.put(4, u32::from(EXT_SEQUENCE));
        w.put(8, u32::from(self.profile_and_level));
        w.put_bit(self.progressive_sequence);
        w.put(2, u32::from(self.chroma_format));
        w.put(2, u32::from(self.horizontal_size_extension));
        w.put(2, u32::from(self.vertical_size_extension));
        w.put(12, u32::from(self.bit_rate_extension));
        w.put_bit(true);
        w.put(8, u32::from(self.vbv_buffer_size_extension));
        w.put_bit(self.low_delay);
        w.put(2, u32::from(self.frame_rate_extension_n));
        w.put(5, u32::from(self.frame_rate_extension_d));
    }
}

impl SequenceDisplayExtension {
    pub(crate) fn parse(payload: &[u8]) -> Result<Self, CodecError> {
        let mut r = BitReader::new(payload);
        r.skip(4);
        let video_format = r.read_checked(3, "video_format")? as u8;
        let colour = if r.read_checked(1, "colour_description")? == 1 {
            Some((
                r.read_checked(8, "colour_primaries")? as u8,
                r.read_checked(8, "transfer_characteristics")? as u8,
                r.read_checked(8, "matrix_coefficients")? as u8,
            ))
        } else {
            None
        };
        let display_horizontal_size = r.read_checked(14, "display_horizontal_size")? as u16;
        r.marker("display extension marker")?;
        let display_vertical_size = r.read_checked(14, "display_vertical_size")? as u16;
        Ok(Self {
            video_format,
            colour,
            display_horizontal_size,
            display_vertical_size,
        })
    }

    pub(crate) fn write(&self, w: &mut BitWriter) {
        w.start_code(EXTENSION);
        w.put(4, u32::from(EXT_SEQUENCE_DISPLAY));
        w.put(3, u32::from(self.video_format));
        w.put_bit(self.colour.is_some());
        if let Some((primaries, transfer, matrix)) = self.colour {
            w.put(8, u32::from(primaries));
            w.put(8, u32::from(transfer));
            w.put(8, u32::from(matrix));
        }
        w.put(14, u32::from(self.display_horizontal_size));
        w.put_bit(true);
        w.put(14, u32::from(self.display_vertical_size));
    }
}

impl GopHeader {
    pub(crate) fn parse(payload: &[u8]) -> Result<Self, CodecError> {
        let mut r = BitReader::new(payload);
        let time_code = r.read_checked(25, "time_code")?;
        let closed_gop = r.read_checked(1, "closed_gop")? == 1;
        let broken_link = r.read_checked(1, "broken_link")? == 1;
        Ok(Self {
            time_code,
            closed_gop,
            broken_link,
        })
    }

    pub(crate) fn write(&self, w: &mut BitWriter) {
        w.start_code(GROUP);
        w.put(25, self.time_code);
        w.put_bit(self.closed_gop);
        w.put_bit(self.broken_link);
    }
}

impl PictureHeader {
    pub(crate) fn parse(payload: &[u8]) -> Result<Self, CodecError> {
        let mut r = BitReader::new(payload);
        let temporal_reference = r.read_checked(10, "temporal_reference")? as u16;
        let at = r.at_bit();
        let picture_coding_type = r.read_checked(3, "picture_coding_type")? as u8;
        if !(PICTURE_I..=PICTURE_D).contains(&picture_coding_type) {
            return Err(CodecError::Malformed {
                at_bit: at,
                what: "picture_coding_type",
            });
        }
        let vbv_delay = r.read_checked(16, "vbv_delay")? as u16;
        let mut header = Self {
            temporal_reference,
            picture_coding_type,
            vbv_delay,
            full_pel_forward: false,
            forward_f_code: 0,
            full_pel_backward: false,
            backward_f_code: 0,
        };
        if picture_coding_type == PICTURE_P || picture_coding_type == PICTURE_B {
            header.full_pel_forward = r.read_checked(1, "full_pel_forward_vector")? == 1;
            header.forward_f_code = r.read_checked(3, "forward_f_code")? as u8;
        }
        if picture_coding_type == PICTURE_B {
            header.full_pel_backward = r.read_checked(1, "full_pel_backward_vector")? == 1;
            header.backward_f_code = r.read_checked(3, "backward_f_code")? as u8;
        }
        Ok(header)
    }

    pub(crate) fn write(&self, w: &mut BitWriter) {
        w.start_code(PICTURE_START);
        w.put(10, u32::from(self.temporal_reference));
        w.put(3, u32::from(self.picture_coding_type));
        w.put(16, u32::from(self.vbv_delay));
        if self.picture_coding_type == PICTURE_P || self.picture_coding_type == PICTURE_B {
            w.put_bit(self.full_pel_forward);
            w.put(3, u32::from(self.forward_f_code));
        }
        if self.picture_coding_type == PICTURE_B {
            w.put_bit(self.full_pel_backward);
            w.put(3, u32::from(self.backward_f_code));
        }
        w.put_bit(false);
    }
}

impl PictureCodingExtension {
    pub(crate) fn parse(payload: &[u8]) -> Result<Self, CodecError> {
        let mut r = BitReader::new(payload);
        r.skip(4);
        let mut f_code = [[0u8; 2]; 2];
        for s in &mut f_code {
            for t in s.iter_mut() {
                *t = r.read_checked(4, "f_code")? as u8;
            }
        }
        let intra_dc_precision = r.read_checked(2, "intra_dc_precision")? as u8;
        let at = r.at_bit();
        let picture_structure = r.read_checked(2, "picture_structure")? as u8;
        if picture_structure == 0 {
            return Err(CodecError::Malformed {
                at_bit: at,
                what: "reserved picture_structure",
            });
        }
        let mut flag = |what| r.read_checked(1, what).map(|bit| bit == 1);
        Ok(Self {
            f_code,
            intra_dc_precision,
            picture_structure,
            top_field_first: flag("top_field_first")?,
            frame_pred_frame_dct: flag("frame_pred_frame_dct")?,
            concealment_motion_vectors: flag("concealment_motion_vectors")?,
            q_scale_type: flag("q_scale_type")?,
            intra_vlc_format: flag("intra_vlc_format")?,
            alternate_scan: flag("alternate_scan")?,
            repeat_first_field: flag("repeat_first_field")?,
            chroma_420_type: flag("chroma_420_type")?,
            progressive_frame: flag("progressive_frame")?,
        })
    }

    pub(crate) fn write(&self, w: &mut BitWriter) {
        w.start_code(EXTENSION);
        w.put(4, u32::from(EXT_PICTURE_CODING));
        for s in &self.f_code {
            for &t in s {
                w.put(4, u32::from(t));
            }
        }
        w.put(2, u32::from(self.intra_dc_precision));
        w.put(2, u32::from(self.picture_structure));
        for flag in [
            self.top_field_first,
            self.frame_pred_frame_dct,
            self.concealment_motion_vectors,
            self.q_scale_type,
            self.intra_vlc_format,
            self.alternate_scan,
            self.repeat_first_field,
            self.chroma_420_type,
            self.progressive_frame,
        ] {
            w.put_bit(flag);
        }
        // composite_display_flag
        w.put_bit(false);
    }
}

impl QuantMatrixExtension {
    pub(crate) fn parse(payload: &[u8]) -> Result<Self, CodecError> {
        let mut r = BitReader::new(payload);
        r.skip(4);
        let mut out = Self::default();
        for slot in [
            &mut out.intra,
            &mut out.non_intra,
            &mut out.chroma_intra,
            &mut out.chroma_non_intra,
        ] {
            if r.read_checked(1, "load quantiser matrix")? == 1 {
                *slot = Some(read_matrix(&mut r)?);
            }
        }
        Ok(out)
    }

    #[cfg(test)]
    pub(crate) fn write(&self, w: &mut BitWriter) {
        w.start_code(EXTENSION);
        w.put(4, u32::from(EXT_QUANT_MATRIX));
        for matrix in [
            &self.intra,
            &self.non_intra,
            &self.chroma_intra,
            &self.chroma_non_intra,
        ] {
            w.put_bit(matrix.is_some());
            if let Some(matrix) = matrix {
                write_matrix(w, matrix);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::start_codes;
    use crate::tables::DEFAULT_INTRA;

    /// Payloads after each start code: `(code, bytes)`.
    fn units(bytes: &[u8]) -> Vec<(u8, Vec<u8>)> {
        let starts = start_codes(bytes);
        starts
            .iter()
            .enumerate()
            .map(|(i, &at)| {
                let end = starts.get(i + 1).copied().unwrap_or(bytes.len());
                (bytes[at + 3], bytes[at + 4..end].to_vec())
            })
            .collect()
    }

    #[test]
    fn every_header_round_trips_through_its_writer() {
        let mut matrix = DEFAULT_INTRA;
        matrix[63] = 99;
        let sequence = SequenceHeader {
            horizontal_size_value: 1920,
            vertical_size_value: 1080,
            aspect_ratio_information: 3,
            frame_rate_code: 4,
            bit_rate_value: 0x3ffff,
            vbv_buffer_size_value: 977,
            constrained_parameters: false,
            intra_matrix: Some(matrix),
            non_intra_matrix: None,
        };
        let extension = SequenceExtension {
            profile_and_level: 0x44,
            progressive_sequence: true,
            chroma_format: CHROMA_422,
            horizontal_size_extension: 1,
            vertical_size_extension: 2,
            bit_rate_extension: 0xabc,
            vbv_buffer_size_extension: 3,
            low_delay: false,
            frame_rate_extension_n: 1,
            frame_rate_extension_d: 30,
        };
        let display = SequenceDisplayExtension {
            video_format: 5,
            colour: Some((1, 1, 1)),
            display_horizontal_size: 1919,
            display_vertical_size: 1081,
        };
        let gop = GopHeader {
            time_code: 0x1ff_ffff,
            closed_gop: true,
            broken_link: false,
        };
        let picture = PictureHeader {
            temporal_reference: 1023,
            picture_coding_type: PICTURE_B,
            vbv_delay: 0xffff,
            full_pel_forward: false,
            forward_f_code: 7,
            full_pel_backward: true,
            backward_f_code: 3,
        };
        let coding = PictureCodingExtension {
            f_code: [[1, 2], [9, 15]],
            intra_dc_precision: 3,
            picture_structure: STRUCTURE_BOTTOM,
            top_field_first: true,
            frame_pred_frame_dct: false,
            concealment_motion_vectors: true,
            q_scale_type: true,
            intra_vlc_format: false,
            alternate_scan: true,
            repeat_first_field: false,
            chroma_420_type: true,
            progressive_frame: false,
        };
        let quant = QuantMatrixExtension {
            intra: None,
            non_intra: Some([17; 64]),
            chroma_intra: Some(matrix),
            chroma_non_intra: None,
        };
        let mut w = BitWriter::new();
        sequence.write(&mut w);
        extension.write(&mut w);
        display.write(&mut w);
        gop.write(&mut w);
        picture.write(&mut w);
        coding.write(&mut w);
        quant.write(&mut w);
        let units = units(&w.finish());
        assert_eq!(units.len(), 7);
        assert_eq!(units[0].0, SEQUENCE_HEADER);
        assert_eq!(SequenceHeader::parse(&units[0].1), Ok(sequence));
        assert_eq!(extension_id(&units[1].1), Some(EXT_SEQUENCE));
        assert_eq!(SequenceExtension::parse(&units[1].1), Ok(extension));
        assert_eq!(extension_id(&units[2].1), Some(EXT_SEQUENCE_DISPLAY));
        assert_eq!(SequenceDisplayExtension::parse(&units[2].1), Ok(display));
        assert_eq!(units[3].0, GROUP);
        assert_eq!(GopHeader::parse(&units[3].1), Ok(gop));
        assert_eq!(units[4].0, PICTURE_START);
        assert_eq!(PictureHeader::parse(&units[4].1), Ok(picture));
        assert_eq!(PictureCodingExtension::parse(&units[5].1), Ok(coding));
        assert_eq!(QuantMatrixExtension::parse(&units[6].1), Ok(quant));
    }

    #[test]
    fn short_and_reserved_headers_are_refused() {
        assert!(matches!(
            SequenceHeader::parse(&[0x78, 0x04]),
            Err(CodecError::Truncated { .. })
        ));
        // A zero marker bit.
        assert!(matches!(
            SequenceHeader::parse(&[0x16, 0x00, 0xf0, 0x13, 0xff, 0xff, 0xc0, 0x00]),
            Err(CodecError::Malformed { .. })
        ));
        // picture_coding_type 0 and 5.
        assert!(PictureHeader::parse(&[0x00, 0x00, 0xff, 0xf8]).is_err());
        assert!(PictureHeader::parse(&[0x00, 0x28, 0xff, 0xf8]).is_err());
        // chroma_format 0.
        assert!(matches!(
            SequenceExtension::parse(&[0x14, 0x88, 0x00, 0x01, 0x00, 0x00]),
            Err(CodecError::Malformed { .. })
        ));
        // A downloaded matrix with a zero entry.
        let mut w = BitWriter::new();
        SequenceHeader {
            horizontal_size_value: 16,
            vertical_size_value: 16,
            aspect_ratio_information: 1,
            frame_rate_code: 3,
            bit_rate_value: 1,
            vbv_buffer_size_value: 1,
            constrained_parameters: false,
            intra_matrix: None,
            non_intra_matrix: Some([0; 64]),
        }
        .write(&mut w);
        let bytes = w.finish();
        assert!(SequenceHeader::parse(&bytes[4..]).is_err());
    }
}
