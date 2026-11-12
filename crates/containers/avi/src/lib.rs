//! AVI and OpenDML AVI 2.0 framing.
//!
//! The crate finds streams, codec ids, extradata, chunk tables, and
//! timestamps in each stream's own `dwRate / dwScale` clock. It does not
//! decode payloads.
//!
//! Normative layout: Microsoft "AVI RIFF File Reference" (`AVIMAINHEADER`,
//! `AVISTREAMHEADER`, `AVIOLDINDEX`, `AVISUPERINDEX`, `AVISTDINDEX`) and
//! the OpenDML AVI File Format Extensions 1.02. Where those leave a choice
//! open, this crate reads files the way FFmpeg's demuxer does, so times
//! agree with `ffprobe`; each such choice is documented where it is made.

#![forbid(unsafe_code)]

mod bytes;
mod demux;
mod format;
mod frames;
mod header;
mod index;
mod layout;
mod limits;
mod mux;
mod probe;
mod sniff;
mod source;
mod track;

pub use demux::{AviDemux, Repair};
pub use format::{StreamKind, VideoFormat};
pub use header::{
    AVIF_HASINDEX, AVIF_ISINTERLEAVED, AVIF_MUSTUSEINDEX, AVIF_TRUSTCKTYPE, MainHeader,
    StreamHeader,
};
pub use index::{Fallback, IndexSource};
pub use limits::{
    MAX_CHUNKS, MAX_DRIFT_SECONDS, MAX_RESYNC, MAX_RIFFS, MAX_SKIPPED_FRAMES, MAX_STREAMS,
    MAX_SUPER_ENTRIES, PROBE_BYTES, SNIFF_BYTES,
};
pub use mux::{AviMux, AviMuxOptions};
pub use probe::{AviFamily, probe};

/// One row of the support matrix. A `false` flag is unsupported in this build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Feature {
    pub name: &'static str,
    pub demux: bool,
    pub mux: bool,
}

/// What this build opens and writes. Unsupported rows stay in the list.
/// The crate README says how each row is checked.
pub const FEATURES: &[Feature] = &[
    Feature {
        name: "avi-1.0",
        demux: true,
        mux: true,
    },
    Feature {
        name: "opendml-2.0",
        demux: true,
        mux: true,
    },
    Feature {
        name: "idx1-relative",
        demux: true,
        mux: true,
    },
    Feature {
        name: "idx1-absolute",
        demux: true,
        mux: false,
    },
    Feature {
        name: "index-rebuild-scan",
        demux: true,
        mux: false,
    },
    Feature {
        name: "rec-lists",
        demux: true,
        mux: false,
    },
    Feature {
        name: "files-over-1gib",
        demux: true,
        mux: true,
    },
    Feature {
        name: "h264",
        demux: true,
        mux: true,
    },
    Feature {
        name: "hevc",
        demux: true,
        mux: true,
    },
    Feature {
        name: "mpeg4-part2",
        demux: true,
        mux: true,
    },
    Feature {
        name: "mpeg2",
        demux: true,
        mux: true,
    },
    Feature {
        name: "mjpeg",
        demux: true,
        mux: true,
    },
    Feature {
        name: "vp8",
        demux: true,
        mux: true,
    },
    Feature {
        name: "vp9",
        demux: true,
        mux: true,
    },
    Feature {
        name: "av1",
        demux: true,
        mux: true,
    },
    Feature {
        name: "raw-video",
        demux: true,
        mux: false,
    },
    Feature {
        name: "pcm-integer",
        demux: true,
        mux: true,
    },
    Feature {
        name: "pcm-float",
        demux: true,
        mux: true,
    },
    Feature {
        name: "waveformatextensible",
        demux: true,
        mux: true,
    },
    Feature {
        name: "mp3",
        demux: true,
        mux: true,
    },
    Feature {
        name: "aac",
        demux: true,
        mux: true,
    },
    Feature {
        name: "ac3",
        demux: true,
        mux: true,
    },
    Feature {
        name: "other-audio",
        demux: true,
        mux: false,
    },
    Feature {
        name: "text-and-other-streams",
        demux: true,
        mux: false,
    },
    Feature {
        name: "forward-only-input",
        demux: false,
        mux: false,
    },
];

#[cfg(test)]
mod behavior;

#[cfg(test)]
mod features {
    use super::FEATURES;

    #[test]
    fn every_feature_is_named_once() {
        let mut names: Vec<_> = FEATURES.iter().map(|feature| feature.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), FEATURES.len());
    }
}
