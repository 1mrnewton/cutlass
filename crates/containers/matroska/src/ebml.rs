//! EBML variable-size integers, element headers, and the primitive types
//! (RFC 8794 sections 4 to 7), read from and written to byte slices.
//!
//! An element ID is one to four octets (EBMLMaxIDLength 4, the Matroska
//! value). A data size is one to eight octets; all data bits set is an
//! unknown size (6.2). A value of 2^(7n) - 1 must not use n octets for a
//! known size, which this writer honours by choosing the length from the
//! value plus one.

use std::fmt;

/// The largest known size an eight-octet VINT holds (RFC 8794, 6.3).
pub const VINT_MAX: u64 = (1 << 56) - 2;

/// A data size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Size {
    Known(u64),
    /// All VINT_DATA bits were one (RFC 8794, 6.2).
    Unknown,
}

/// What went wrong while reading EBML bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EbmlError {
    /// The first octet of an ID has no marker in its first four bits.
    BadId { first: u8 },
    /// The first octet of a size is zero: the VINT would be longer than eight octets.
    BadSize,
    /// The bytes end inside a header or a value.
    Short,
    /// A child's declared size runs past its parent.
    PastParent { id: u32, size: u64, room: u64 },
    /// An unknown size where the schema does not allow one.
    UnknownSize { id: u32 },
    /// A value is longer than its type allows.
    BadLength { id: u32, len: usize },
}

impl fmt::Display for EbmlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadId { first } => write!(formatter, "invalid element id byte {first:#04x}"),
            Self::BadSize => formatter.write_str("invalid element size"),
            Self::Short => formatter.write_str("element ends early"),
            Self::PastParent { id, size, room } => write!(
                formatter,
                "element {id:#x} of {size} bytes runs past its parent ({room} left)"
            ),
            Self::UnknownSize { id } => {
                write!(formatter, "element {id:#x} has an unknown size")
            }
            Self::BadLength { id, len } => {
                write!(formatter, "element {id:#x} has a {len}-byte value")
            }
        }
    }
}

impl std::error::Error for EbmlError {}

/// Octets in an ID that starts with `first`, or `None` when the marker is
/// not in the first four bits.
#[must_use]
pub const fn id_len(first: u8) -> Option<usize> {
    match first.leading_zeros() {
        0 => Some(1),
        1 => Some(2),
        2 => Some(3),
        3 => Some(4),
        _ => None,
    }
}

/// Octets in a size VINT that starts with `first`, or `None` for a zero octet.
#[must_use]
pub const fn size_len(first: u8) -> Option<usize> {
    if first == 0 {
        None
    } else {
        Some(first.leading_zeros() as usize + 1)
    }
}

/// Reads an element ID. The marker stays in the value.
pub fn read_id(bytes: &[u8]) -> Result<(u32, usize), EbmlError> {
    let first = *bytes.first().ok_or(EbmlError::Short)?;
    let len = id_len(first).ok_or(EbmlError::BadId { first })?;
    let raw = bytes.get(..len).ok_or(EbmlError::Short)?;
    let mut id = 0_u32;
    for byte in raw {
        id = (id << 8) | u32::from(*byte);
    }
    Ok((id, len))
}

/// Reads a data size.
pub fn read_size(bytes: &[u8]) -> Result<(Size, usize), EbmlError> {
    let first = *bytes.first().ok_or(EbmlError::Short)?;
    let len = size_len(first).ok_or(EbmlError::BadSize)?;
    let raw = bytes.get(..len).ok_or(EbmlError::Short)?;
    Ok((decode_size(raw, len), len))
}

/// The size held by `raw`, which is exactly `len` octets of a VINT.
#[must_use]
pub fn decode_size(raw: &[u8], len: usize) -> Size {
    let mut value = 0_u64;
    for byte in raw {
        value = (value << 8) | u64::from(*byte);
    }
    // Clear the marker. `len` is 1..=8, so the shift is below 64.
    let bits = 7 * u32::try_from(len).unwrap_or(8);
    let mask = (1_u64 << bits) - 1;
    let data = value & mask;
    if data == mask {
        Size::Unknown
    } else {
        Size::Known(data)
    }
}

/// Reads an unsigned VINT that cannot be unknown, such as a track number.
pub fn read_vint(bytes: &[u8]) -> Result<(u64, usize), EbmlError> {
    let first = *bytes.first().ok_or(EbmlError::Short)?;
    let len = size_len(first).ok_or(EbmlError::BadSize)?;
    let raw = bytes.get(..len).ok_or(EbmlError::Short)?;
    let mut value = 0_u64;
    for byte in raw {
        value = (value << 8) | u64::from(*byte);
    }
    let bits = 7 * u32::try_from(len).unwrap_or(8);
    Ok((value & ((1_u64 << bits) - 1), len))
}

/// Reads a signed VINT as EBML lacing codes it: the unsigned value less
/// 2^(7n-1) - 1 (RFC 9559, 10.3.3).
pub fn read_signed_vint(bytes: &[u8]) -> Result<(i64, usize), EbmlError> {
    let (value, len) = read_vint(bytes)?;
    let bits = 7 * u32::try_from(len).unwrap_or(8) - 1;
    let bias = (1_i64 << bits) - 1;
    let value = i64::try_from(value).map_err(|_| EbmlError::BadSize)?;
    Ok((value - bias, len))
}

/// Big-endian unsigned integer of 0 to 8 octets (RFC 8794, 7.2).
pub fn uint(id: u32, data: &[u8]) -> Result<u64, EbmlError> {
    if data.len() > 8 {
        return Err(EbmlError::BadLength {
            id,
            len: data.len(),
        });
    }
    let mut value = 0_u64;
    for byte in data {
        value = (value << 8) | u64::from(*byte);
    }
    Ok(value)
}

/// Big-endian two's-complement integer of 0 to 8 octets (7.1).
pub fn int(id: u32, data: &[u8]) -> Result<i64, EbmlError> {
    let value = uint(id, data)?;
    if data.is_empty() || data.len() == 8 {
        return Ok(value.cast_signed());
    }
    let bits = 8 * u32::try_from(data.len()).unwrap_or(8);
    let shift = 64 - bits;
    Ok((value << shift).cast_signed() >> shift)
}

/// IEEE 754 binary32 or binary64, or zero for an empty value (7.3).
pub fn float(id: u32, data: &[u8]) -> Result<f64, EbmlError> {
    match data.len() {
        0 => Ok(0.0),
        4 => {
            let mut raw = [0_u8; 4];
            raw.copy_from_slice(data);
            Ok(f64::from(f32::from_be_bytes(raw)))
        }
        8 => {
            let mut raw = [0_u8; 8];
            raw.copy_from_slice(data);
            Ok(f64::from_be_bytes(raw))
        }
        len => Err(EbmlError::BadLength { id, len }),
    }
}

/// A string value up to its first NUL: writers may pad with zero octets (7.4).
#[must_use]
pub fn text(data: &[u8]) -> &[u8] {
    match data.iter().position(|byte| *byte == 0) {
        Some(end) => data.get(..end).unwrap_or(data),
        None => data,
    }
}

/// One child of a master element held in memory.
#[derive(Clone, Copy, Debug)]
pub struct Child<'a> {
    pub id: u32,
    pub data: &'a [u8],
    /// Offset of the child's ID within the parent's data.
    pub offset: usize,
}

/// The children of a master element whose data is in memory.
///
/// Every child must have a known size that fits in the parent. Children are
/// yielded in file order; an error ends the walk.
pub struct Children<'a> {
    data: &'a [u8],
    at: usize,
    failed: bool,
}

impl<'a> Children<'a> {
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            at: 0,
            failed: false,
        }
    }
}

impl<'a> Iterator for Children<'a> {
    type Item = Result<Child<'a>, EbmlError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.at >= self.data.len() {
            return None;
        }
        let result = child_at(self.data, self.at);
        match result {
            Ok((child, end)) => {
                self.at = end;
                Some(Ok(child))
            }
            Err(error) => {
                self.failed = true;
                Some(Err(error))
            }
        }
    }
}

fn child_at(data: &[u8], at: usize) -> Result<(Child<'_>, usize), EbmlError> {
    let rest = data.get(at..).ok_or(EbmlError::Short)?;
    let (id, id_len) = read_id(rest)?;
    let (size, size_len) = read_size(rest.get(id_len..).ok_or(EbmlError::Short)?)?;
    let header = id_len + size_len;
    let room = rest.len().saturating_sub(header);
    let Size::Known(size) = size else {
        return Err(EbmlError::UnknownSize { id });
    };
    let room_u64 = u64::try_from(room).unwrap_or(u64::MAX);
    if size > room_u64 {
        return Err(EbmlError::PastParent {
            id,
            size,
            room: room_u64,
        });
    }
    let len = usize::try_from(size).map_err(|_| EbmlError::Short)?;
    let start = at + header;
    let child = Child {
        id,
        data: data.get(start..start + len).ok_or(EbmlError::Short)?,
        offset: at,
    };
    Ok((child, start + len))
}

// Writing.

/// Octets an ID takes on disk.
#[must_use]
pub const fn id_width(id: u32) -> usize {
    if id > 0x00FF_FFFF {
        4
    } else if id > 0xFFFF {
        3
    } else if id > 0xFF {
        2
    } else {
        1
    }
}

pub fn put_id(out: &mut Vec<u8>, id: u32) {
    let bytes = id.to_be_bytes();
    let width = id_width(id);
    out.extend_from_slice(&bytes[4 - width..]);
}

/// The shortest size VINT that holds `value` as a known size.
#[must_use]
pub const fn size_width(value: u64) -> usize {
    let mut width = 1;
    // `value` must stay below the all-ones pattern of its width.
    while width < 8 && value >= (1_u64 << (7 * width)) - 1 {
        width += 1;
    }
    width
}

/// Writes `value` as a size VINT of exactly `width` octets. The caller
/// chooses a width that holds it ([`size_width`] or more, up to 8).
pub fn put_size_width(out: &mut Vec<u8>, value: u64, width: usize) {
    let width = width.clamp(1, 8);
    let marker = 1_u64 << (7 * width);
    let encoded = value | marker;
    let bytes = encoded.to_be_bytes();
    out.extend_from_slice(&bytes[8 - width..]);
}

pub fn put_size(out: &mut Vec<u8>, value: u64) {
    put_size_width(out, value, size_width(value));
}

/// The eight-octet unknown size (RFC 8794, 6.2).
pub const UNKNOWN_SIZE: [u8; 8] = [0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];

/// Octets needed for an unsigned value, at least one.
#[must_use]
pub const fn uint_width(value: u64) -> usize {
    let bits = 64 - value.leading_zeros() as usize;
    if bits == 0 { 1 } else { bits.div_ceil(8) }
}

pub fn put_uint(out: &mut Vec<u8>, id: u32, value: u64) {
    let width = uint_width(value);
    put_id(out, id);
    put_size(out, width as u64);
    let bytes = value.to_be_bytes();
    out.extend_from_slice(&bytes[8 - width..]);
}

/// An unsigned value padded to exactly eight octets, for a field patched later.
pub fn put_uint8(out: &mut Vec<u8>, id: u32, value: u64) {
    put_id(out, id);
    put_size(out, 8);
    out.extend_from_slice(&value.to_be_bytes());
}

pub fn put_float(out: &mut Vec<u8>, id: u32, value: f64) {
    put_id(out, id);
    put_size(out, 8);
    out.extend_from_slice(&value.to_be_bytes());
}

pub fn put_bytes(out: &mut Vec<u8>, id: u32, value: &[u8]) {
    put_id(out, id);
    put_size(out, value.len() as u64);
    out.extend_from_slice(value);
}

/// A master element around `body`.
pub fn put_master(out: &mut Vec<u8>, id: u32, body: &[u8]) {
    put_bytes(out, id, body);
}

/// A Void element of exactly `total` octets, header included. `total`
/// must be at least two.
pub fn put_void(out: &mut Vec<u8>, total: usize) {
    if total < 2 {
        return;
    }
    // One ID octet; the size takes one octet up to 126 bytes of data and
    // eight octets beyond, so the data length is exact either way.
    let data = total - 2;
    if data < 127 {
        out.push(0xEC);
        put_size_width(out, data as u64, 1);
        out.resize(out.len() + data, 0);
    } else {
        let data = total - 9;
        out.push(0xEC);
        put_size_width(out, data as u64, 8);
        out.resize(out.len() + data, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Children, EbmlError, Size, VINT_MAX, decode_size, float, int, put_size, put_uint, put_void,
        read_id, read_signed_vint, read_size, read_vint, size_width, text, uint,
    };

    #[test]
    fn ids_keep_their_marker_and_refuse_a_long_first_byte() {
        assert_eq!(read_id(&[0x1A, 0x45, 0xDF, 0xA3]), Ok((0x1A45_DFA3, 4)));
        assert_eq!(read_id(&[0xA3]), Ok((0xA3, 1)));
        assert_eq!(read_id(&[0x42, 0x86]), Ok((0x4286, 2)));
        assert_eq!(
            read_id(&[0x08, 0, 0, 0, 0]),
            Err(EbmlError::BadId { first: 0x08 })
        );
        assert_eq!(read_id(&[0x1A, 0x45]), Err(EbmlError::Short));
        assert_eq!(read_id(&[]), Err(EbmlError::Short));
    }

    #[test]
    fn sizes_decode_every_width_and_the_unknown_pattern() {
        assert_eq!(read_size(&[0x81]), Ok((Size::Known(1), 1)));
        assert_eq!(read_size(&[0x40, 0x02]), Ok((Size::Known(2), 2)));
        assert_eq!(read_size(&[0xFF]), Ok((Size::Unknown, 1)));
        assert_eq!(
            read_size(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]),
            Ok((Size::Unknown, 8))
        );
        assert_eq!(
            read_size(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE]),
            Ok((Size::Known(VINT_MAX), 8))
        );
        // A zero first octet would be a nine-octet VINT.
        assert_eq!(read_size(&[0x00, 0x81]), Err(EbmlError::BadSize));
        assert_eq!(read_size(&[0x10, 0x00]), Err(EbmlError::Short));
        assert_eq!(decode_size(&[0x7F, 0xFF], 2), Size::Unknown);
    }

    #[test]
    fn a_size_never_uses_the_all_ones_pattern_of_its_width() {
        assert_eq!(size_width(0), 1);
        assert_eq!(size_width(126), 1);
        assert_eq!(size_width(127), 2);
        assert_eq!(size_width(16_382), 2);
        assert_eq!(size_width(16_383), 3);
        assert_eq!(size_width(VINT_MAX), 8);
        for value in [0_u64, 1, 126, 127, 128, 16_383, 1 << 30, VINT_MAX] {
            let mut out = Vec::new();
            put_size(&mut out, value);
            assert_eq!(read_size(&out), Ok((Size::Known(value), out.len())));
        }
    }

    #[test]
    fn signed_lacing_vints_use_the_bias_of_their_width() {
        // One octet: 0xBF is 63, which is zero after the bias of 63.
        assert_eq!(read_signed_vint(&[0xBF]), Ok((0, 1)));
        assert_eq!(read_signed_vint(&[0x80]), Ok((-63, 1)));
        assert_eq!(read_signed_vint(&[0xFE]), Ok((63, 1)));
        // Two octets: bias 8191.
        assert_eq!(read_signed_vint(&[0x5F, 0xFF]), Ok((0, 2)));
        assert_eq!(read_signed_vint(&[0x40, 0x00]), Ok((-8191, 2)));
        assert_eq!(read_vint(&[0x81]), Ok((1, 1)));
    }

    #[test]
    fn primitives_follow_their_lengths() {
        assert_eq!(uint(1, &[]), Ok(0));
        assert_eq!(uint(1, &[0x0F, 0x42, 0x40]), Ok(1_000_000));
        assert_eq!(
            uint(1, &[0; 9]),
            Err(EbmlError::BadLength { id: 1, len: 9 })
        );
        assert_eq!(int(1, &[0xFF]), Ok(-1));
        assert_eq!(int(1, &[0xFF, 0x00]), Ok(-256));
        assert_eq!(int(1, &[0x7F]), Ok(127));
        assert_eq!(float(1, &[0x3F, 0x80, 0, 0]), Ok(1.0));
        assert_eq!(float(1, &2.5_f64.to_be_bytes()), Ok(2.5));
        assert_eq!(
            float(1, &[0; 3]),
            Err(EbmlError::BadLength { id: 1, len: 3 })
        );
        assert_eq!(text(b"webm\0\0"), b"webm");
    }

    #[test]
    fn children_refuse_a_size_past_the_parent_and_an_unknown_size() {
        let mut body = Vec::new();
        put_uint(&mut body, 0x4286, 1);
        put_void(&mut body, 5);
        let ids: Vec<u32> = Children::new(&body)
            .map(|child| child.expect("child").id)
            .collect();
        assert_eq!(ids, vec![0x4286, 0xEC]);

        let past = [0x42, 0x86, 0x84, 1];
        let error = Children::new(&past).next().expect("one").expect_err("past");
        assert!(matches!(error, EbmlError::PastParent { .. }));
        let unknown = [0x42, 0x86, 0xFF];
        let error = Children::new(&unknown)
            .next()
            .expect("one")
            .expect_err("unknown");
        assert_eq!(error, EbmlError::UnknownSize { id: 0x4286 });
        assert_eq!(Children::new(&past).count(), 1);
    }

    #[test]
    fn a_void_is_exactly_the_requested_length() {
        for total in [2_usize, 3, 128, 129, 130, 4096] {
            let mut out = Vec::new();
            put_void(&mut out, total);
            assert_eq!(out.len(), total, "{total}");
            let child = Children::new(&out).next().expect("void").expect("ok");
            assert_eq!(child.id, 0xEC);
        }
    }
}
