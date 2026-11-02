//! Packed `WAVEFORMATEXTENSIBLE` for one stereo `f32` stream.
//!
//! `mmreg.h` packs this struct to 1 byte. The IEEE float subtype is the
//! waveformatex GUID with format tag 3.

use std::mem::size_of;

/// `GUID` as Windows lays it out.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

/// `WAVEFORMATEXTENSIBLE`. The prefix is `WAVEFORMATEX`.
#[repr(C, packed)]
pub(crate) struct WaveFormatExtensible {
    format_tag: u16,
    channels: u16,
    samples_per_sec: u32,
    avg_bytes_per_sec: u32,
    block_align: u16,
    bits_per_sample: u16,
    extra: u16,
    valid_bits: u16,
    channel_mask: u32,
    sub_format: Guid,
}

const _: () = assert!(size_of::<WaveFormatExtensible>() == 40);

/// `KSDATAFORMAT_SUBTYPE_IEEE_FLOAT` from `mmreg.h`.
pub(crate) fn ieee_float() -> Guid {
    Guid {
        data1: 0x0000_0003,
        data2: 0x0000,
        data3: 0x0010,
        data4: [0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71],
    }
}

/// Stereo `f32` at `rate`, little-endian, front left and front right.
pub(crate) fn float_stereo(rate: u32) -> WaveFormatExtensible {
    let block = 8_u16;
    WaveFormatExtensible {
        format_tag: 0xFFFE,
        channels: 2,
        samples_per_sec: rate,
        avg_bytes_per_sec: rate.saturating_mul(u32::from(block)),
        block_align: block,
        bits_per_sample: 32,
        extra: 22,
        valid_bits: 32,
        channel_mask: 0x3,
        sub_format: ieee_float(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_float_subtype_is_the_waveformatex_guid() {
        let guid = ieee_float();
        assert_eq!(guid.data1, 3);
        assert_eq!(guid.data2, 0);
        assert_eq!(guid.data3, 0x0010);
        assert_eq!(guid.data4, [0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71]);
        assert_eq!(size_of::<WaveFormatExtensible>(), 40);
        assert_eq!(size_of::<Guid>(), 16);
    }
}
