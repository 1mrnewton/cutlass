//! The C headers this crate's struct layouts and constants were taken
//! from, for the app's Licenses window.

/// One set of headers: what it is, whose, its license and home, and the
/// license text.
#[derive(Clone, Copy, Debug)]
pub struct HeaderCredit {
    pub name: &'static str,
    pub author: &'static str,
    pub license: &'static str,
    pub link: &'static str,
    pub text: &'static str,
}

/// libva's headers, which the VA-API offsets in `src/video/imp/va/abi.rs`
/// were printed from, and FFmpeg's copy of NVIDIA's codec headers, which
/// the NVENC offsets in `src/video/imp/nvenc_abi.rs` were printed from.
pub const HEADER_CREDITS: [HeaderCredit; 2] = [
    HeaderCredit {
        name: "libva headers",
        author: "Intel Corporation",
        license: "MIT License",
        link: "https://github.com/intel/libva",
        text: include_str!("../licenses/libva.txt"),
    },
    HeaderCredit {
        name: "NVIDIA Video Codec SDK headers (nv-codec-headers)",
        author: "NVIDIA Corporation",
        license: "MIT License",
        link: "https://github.com/FFmpeg/nv-codec-headers",
        text: include_str!("../licenses/nv-codec-headers.txt"),
    },
];
