//! Matroska and WebM framing.

#![forbid(unsafe_code)]

pub mod block;
pub mod codec;
mod demux;
pub mod ebml;
mod error;
pub mod ids;
pub mod limits;
pub mod meta;
mod mux;
mod probe;
mod seek;
mod source;
mod tail;

pub use demux::{Clock, MatroskaDemux};
pub use error::Error;
pub use mux::MatroskaMux;
pub use probe::{MatroskaFamily, probe};

/// One row of the support matrix. A `false` flag is unsupported in this build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Feature {
    pub name: &'static str,
    pub demux: bool,
    pub mux: bool,
}

/// What this build reads and writes, as the README's table says. Rows that
/// are not supported stay in the list. A subtitle track, and a track whose
/// frames are compressed or encrypted (`zlib-bzlib-lzo`, `encryption`),
/// still opens as a stream with an unknown codec tag and the stored bytes;
/// `false` there means nothing more is done with them. BlockAdditions are
/// skipped.
pub const FEATURES: &[Feature] = &[
    Feature {
        name: "matroska",
        demux: true,
        mux: true,
    },
    Feature {
        name: "webm",
        demux: true,
        mux: true,
    },
    Feature {
        name: "ebml-header",
        demux: true,
        mux: true,
    },
    Feature {
        name: "seek-head",
        demux: true,
        mux: true,
    },
    Feature {
        name: "info-duration",
        demux: true,
        mux: true,
    },
    Feature {
        name: "tracks",
        demux: true,
        mux: true,
    },
    Feature {
        name: "codec-private",
        demux: true,
        mux: true,
    },
    Feature {
        name: "cluster-known-size",
        demux: true,
        mux: true,
    },
    Feature {
        name: "cluster-unknown-size",
        demux: true,
        mux: false,
    },
    Feature {
        name: "simple-block",
        demux: true,
        mux: true,
    },
    Feature {
        name: "block-group",
        demux: true,
        mux: true,
    },
    Feature {
        name: "lacing-xiph",
        demux: true,
        mux: false,
    },
    Feature {
        name: "lacing-ebml",
        demux: true,
        mux: false,
    },
    Feature {
        name: "lacing-fixed",
        demux: true,
        mux: false,
    },
    Feature {
        name: "cues",
        demux: true,
        mux: true,
    },
    Feature {
        name: "seek-without-cues",
        demux: true,
        mux: false,
    },
    Feature {
        name: "forward-only-input",
        demux: true,
        mux: false,
    },
    Feature {
        name: "header-stripping",
        demux: true,
        mux: false,
    },
    Feature {
        name: "codec-delay",
        demux: true,
        mux: true,
    },
    Feature {
        name: "discard-padding",
        demux: true,
        mux: true,
    },
    Feature {
        name: "default-duration",
        demux: true,
        mux: true,
    },
    Feature {
        name: "timestamp-scale",
        demux: true,
        mux: true,
    },
    Feature {
        name: "prores-icpf",
        demux: true,
        mux: true,
    },
    Feature {
        name: "vfw-acm-codec-private",
        demux: true,
        mux: false,
    },
    Feature {
        name: "unknown-codec",
        demux: true,
        mux: false,
    },
    Feature {
        name: "track-timestamp-scale",
        demux: false,
        mux: false,
    },
    Feature {
        name: "zlib-bzlib-lzo",
        demux: false,
        mux: false,
    },
    Feature {
        name: "encryption",
        demux: false,
        mux: false,
    },
    Feature {
        name: "chained-segments",
        demux: false,
        mux: false,
    },
    Feature {
        name: "chapters",
        demux: false,
        mux: false,
    },
    Feature {
        name: "tags",
        demux: false,
        mux: false,
    },
    Feature {
        name: "attachments",
        demux: false,
        mux: true,
    },
    Feature {
        name: "block-additions",
        demux: false,
        mux: false,
    },
    Feature {
        name: "subtitles",
        demux: false,
        mux: false,
    },
];

#[cfg(test)]
mod tests {
    use super::{FEATURES, Feature};

    fn feature(name: &str) -> Feature {
        *FEATURES
            .iter()
            .find(|feature| feature.name == name)
            .expect(name)
    }

    #[test]
    fn the_support_matrix_names_implemented_and_unsupported_rows() {
        for name in [
            "matroska",
            "webm",
            "cues",
            "simple-block",
            "block-group",
            "codec-delay",
            "discard-padding",
        ] {
            let row = feature(name);
            assert!(row.demux && row.mux, "{name}");
        }
        for name in ["lacing-xiph", "cluster-unknown-size", "header-stripping"] {
            let row = feature(name);
            assert!(row.demux && !row.mux, "{name}");
        }
        let row = feature("attachments");
        assert!(!row.demux && row.mux);
        for name in [
            "encryption",
            "zlib-bzlib-lzo",
            "chained-segments",
            "track-timestamp-scale",
        ] {
            let row = feature(name);
            assert!(!row.demux && !row.mux, "{name}");
        }
        let mut names: Vec<&str> = FEATURES.iter().map(|row| row.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), FEATURES.len());
    }
}
