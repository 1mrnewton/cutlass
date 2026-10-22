//! Header fields and the scanline sizes they imply.

use crate::error::Error;

/// Each edge, and the inflated bytes, stop here. `docs/codecs.md`.
pub(crate) const MAX_EDGE: u32 = 32_768;
pub(crate) const MAX_BYTES: usize = 256 * 1024 * 1024;

/// One reduced image in the PNG pass order.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Pass {
    pub x0: u32,
    pub y0: u32,
    pub x_step: u32,
    pub y_step: u32,
}

/// PNG 8.1. The null method is one pass. Adam7 is these seven.
const WHOLE: [Pass; 1] = [Pass {
    x0: 0,
    y0: 0,
    x_step: 1,
    y_step: 1,
}];

const ADAM7: [Pass; 7] = [
    Pass {
        x0: 0,
        y0: 0,
        x_step: 8,
        y_step: 8,
    },
    Pass {
        x0: 4,
        y0: 0,
        x_step: 8,
        y_step: 8,
    },
    Pass {
        x0: 0,
        y0: 4,
        x_step: 4,
        y_step: 8,
    },
    Pass {
        x0: 2,
        y0: 0,
        x_step: 4,
        y_step: 4,
    },
    Pass {
        x0: 0,
        y0: 2,
        x_step: 2,
        y_step: 4,
    },
    Pass {
        x0: 1,
        y0: 0,
        x_step: 2,
        y_step: 2,
    },
    Pass {
        x0: 0,
        y0: 1,
        x_step: 1,
        y_step: 2,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColorKind {
    Grey,
    Truecolor,
    Indexed,
    GreyAlpha,
    TruecolorAlpha,
}

impl ColorKind {
    pub(crate) fn channels(self) -> u8 {
        match self {
            Self::Grey | Self::Indexed => 1,
            Self::Truecolor => 3,
            Self::GreyAlpha => 2,
            Self::TruecolorAlpha => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Info {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub color: ColorKind,
    pub interlace: bool,
}

/// A `tRNS` key. Indexed alpha is one byte per leading palette entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Key {
    None,
    Grey(u16),
    Rgb([u16; 3]),
    Index(Vec<u8>),
}

pub(crate) fn passes(interlace: bool) -> &'static [Pass] {
    if interlace { &ADAM7 } else { &WHOLE }
}

pub(crate) fn limits(info: &Info) -> Result<(), Error> {
    if info.width == 0 || info.height == 0 {
        return Err(Error::Inconsistent("width or height is zero"));
    }
    if info.width > MAX_EDGE || info.height > MAX_EDGE {
        return Err(Error::TooLarge);
    }
    let _ = rgba_len(info.width, info.height)?;
    let _ = inflated_len(info)?;
    Ok(())
}

pub(crate) fn rgba_len(width: u32, height: u32) -> Result<usize, Error> {
    let width = usize::try_from(width).map_err(|_| Error::TooLarge)?;
    let height = usize::try_from(height).map_err(|_| Error::TooLarge)?;
    let bytes = width
        .checked_mul(height)
        .and_then(|count| count.checked_mul(4))
        .ok_or(Error::TooLarge)?;
    if bytes > MAX_BYTES {
        Err(Error::TooLarge)
    } else {
        Ok(bytes)
    }
}

pub(crate) fn inflated_len(info: &Info) -> Result<usize, Error> {
    let mut total = 0usize;
    for pass in passes(info.interlace) {
        let width = span(info.width, pass.x0, pass.x_step);
        let height = span(info.height, pass.y0, pass.y_step);
        if width == 0 || height == 0 {
            continue;
        }
        let row = row_bytes(width, info.color.channels(), info.bit_depth)?;
        let scan = row.checked_add(1).ok_or(Error::TooLarge)?;
        let rows = usize::try_from(height).map_err(|_| Error::TooLarge)?;
        let block = scan.checked_mul(rows).ok_or(Error::TooLarge)?;
        total = total.checked_add(block).ok_or(Error::TooLarge)?;
        if total > MAX_BYTES {
            return Err(Error::TooLarge);
        }
    }
    if total == 0 {
        return Err(Error::Inconsistent("the image has no scanlines"));
    }
    Ok(total)
}

pub(crate) fn row_bytes(width: u32, channels: u8, bit_depth: u8) -> Result<usize, Error> {
    let bits = u64::from(width)
        .checked_mul(u64::from(channels))
        .and_then(|count| count.checked_mul(u64::from(bit_depth)))
        .ok_or(Error::TooLarge)?;
    usize::try_from(bits.div_ceil(8)).map_err(|_| Error::TooLarge)
}

pub(crate) fn span(size: u32, start: u32, step: u32) -> u32 {
    if start >= size {
        0
    } else {
        (size - start - 1) / step + 1
    }
}

pub(crate) fn filter_bytes(channels: u8, bit_depth: u8) -> usize {
    let bits = usize::from(channels) * usize::from(bit_depth);
    (bits / 8).max(1)
}
