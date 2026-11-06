//! Native FLAC files (RFC 9639): demux and mux.
//!
//! A file is the `fLaC` marker, metadata blocks with STREAMINFO first, and
//! frames. One packet is one frame, header to CRC-16, as the codec crate
//! decodes it. The stream timescale is the sample rate. The stream's
//! extradata is the 34-byte STREAMINFO body.
//!
//! | Capability | Demux | Mux |
//! | --- | --- | --- |
//! | Metadata | STREAMINFO, SEEKTABLE read; other blocks skipped by length | STREAMINFO and SEEKTABLE written |
//! | Frames | Split at the next valid header whose CRC-16 closes the frame | Checked: header, CRC-16, numbering, duration |
//! | Timestamps | Frame number times the block size, or the sample number | From the sample count |
//! | Seek | SEEKTABLE, else a bounded bisection, then a frame walk | Points about every 10 s |
//! | Finish | — | STREAMINFO sample count, sizes, and MD5 written back |
//! | ID3v2 before `fLaC` | Refused as unsupported | Not written |
//! | Ogg FLAC, MP4 `dfLa` | Not in this crate | Not in this crate |

#![forbid(unsafe_code)]

mod demux;
mod format;
mod mux;
mod probe;

pub use demux::{FlacDemux, MAX_METADATA_BLOCKS};
pub use format::{
    FrameHeader, Header, MAX_HEADER_LEN, SEEK_POINT_LEN, STREAMINFO_LEN, SeekPoint, StreamInfo,
    crc8, crc16, parse_header,
};
pub use mux::{DEFAULT_SEEK_POINTS, FlacMux, SEEK_SPACING_SECONDS};
pub use probe::{FlacFamily, probe};

#[cfg(test)]
mod tests;
