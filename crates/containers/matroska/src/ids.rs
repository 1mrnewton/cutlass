//! Element IDs, written as the spec writes them: the marker bits are part
//! of the value, so `0x1A45DFA3` is four octets on disk.
//!
//! Sources: RFC 8794 section 11.2 (EBML header) and 11.3 (global elements),
//! RFC 9559 section 5.1 (the Matroska schema). The values were checked
//! against FFmpeg's `libavformat/matroska.h`.

// EBML header (RFC 8794, 11.2).
pub const EBML: u32 = 0x1A45_DFA3;
pub const EBML_VERSION: u32 = 0x4286;
pub const EBML_READ_VERSION: u32 = 0x42F7;
pub const EBML_MAX_ID_LENGTH: u32 = 0x42F2;
pub const EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;
pub const DOC_TYPE: u32 = 0x4282;
pub const DOC_TYPE_VERSION: u32 = 0x4287;
pub const DOC_TYPE_READ_VERSION: u32 = 0x4285;

// Global elements (RFC 8794, 11.3).
pub const VOID: u32 = 0xEC;
pub const CRC32: u32 = 0xBF;

// Segment and its top-level children (RFC 9559, 5.1).
pub const SEGMENT: u32 = 0x1853_8067;
pub const SEEK_HEAD: u32 = 0x114D_9B74;
pub const INFO: u32 = 0x1549_A966;
pub const TRACKS: u32 = 0x1654_AE6B;
pub const CLUSTER: u32 = 0x1F43_B675;
pub const CUES: u32 = 0x1C53_BB6B;
pub const CHAPTERS: u32 = 0x1043_A770;
pub const TAGS: u32 = 0x1254_C367;
pub const ATTACHMENTS: u32 = 0x1941_A469;

// SeekHead (5.1.1).
pub const SEEK: u32 = 0x4DBB;
pub const SEEK_ID: u32 = 0x53AB;
pub const SEEK_POSITION: u32 = 0x53AC;

// Info (5.1.2).
pub const TIMESTAMP_SCALE: u32 = 0x2A_D7B1;
pub const DURATION: u32 = 0x4489;
pub const DATE_UTC: u32 = 0x4461;
pub const TITLE: u32 = 0x7BA9;
pub const MUXING_APP: u32 = 0x4D80;
pub const WRITING_APP: u32 = 0x5741;
pub const SEGMENT_UUID: u32 = 0x73A4;

// Cluster (5.1.3).
pub const TIMESTAMP: u32 = 0xE7;
pub const POSITION: u32 = 0xA7;
pub const PREV_SIZE: u32 = 0xAB;
pub const SIMPLE_BLOCK: u32 = 0xA3;
pub const BLOCK_GROUP: u32 = 0xA0;
pub const BLOCK: u32 = 0xA1;
pub const BLOCK_DURATION: u32 = 0x9B;
pub const REFERENCE_BLOCK: u32 = 0xFB;
pub const DISCARD_PADDING: u32 = 0x75A2;

// Tracks (5.1.4).
pub const TRACK_ENTRY: u32 = 0xAE;
pub const TRACK_NUMBER: u32 = 0xD7;
pub const TRACK_UID: u32 = 0x73C5;
pub const TRACK_TYPE: u32 = 0x83;
pub const FLAG_LACING: u32 = 0x9C;
pub const DEFAULT_DURATION: u32 = 0x23_E383;
pub const TRACK_TIMESTAMP_SCALE: u32 = 0x23_314F;
pub const LANGUAGE: u32 = 0x22_B59C;
pub const LANGUAGE_BCP47: u32 = 0x22_B59D;
pub const CODEC_ID: u32 = 0x86;
pub const CODEC_PRIVATE: u32 = 0x63A2;
pub const CODEC_DELAY: u32 = 0x56AA;
pub const SEEK_PRE_ROLL: u32 = 0x56BB;
pub const VIDEO: u32 = 0xE0;
pub const PIXEL_WIDTH: u32 = 0xB0;
pub const PIXEL_HEIGHT: u32 = 0xBA;
pub const DISPLAY_WIDTH: u32 = 0x54B0;
pub const DISPLAY_HEIGHT: u32 = 0x54BA;
pub const AUDIO: u32 = 0xE1;
pub const SAMPLING_FREQUENCY: u32 = 0xB5;
pub const OUTPUT_SAMPLING_FREQUENCY: u32 = 0x78B5;
pub const CHANNELS: u32 = 0x9F;
pub const BIT_DEPTH: u32 = 0x6264;
pub const CONTENT_ENCODINGS: u32 = 0x6D80;
pub const CONTENT_ENCODING: u32 = 0x6240;
pub const CONTENT_ENCODING_ORDER: u32 = 0x5031;
pub const CONTENT_ENCODING_SCOPE: u32 = 0x5032;
pub const CONTENT_ENCODING_TYPE: u32 = 0x5033;
pub const CONTENT_COMPRESSION: u32 = 0x5034;
pub const CONTENT_COMP_ALGO: u32 = 0x4254;
pub const CONTENT_COMP_SETTINGS: u32 = 0x4255;
pub const CONTENT_ENCRYPTION: u32 = 0x5035;

// Cues (5.1.5).
pub const CUE_POINT: u32 = 0xBB;
pub const CUE_TIME: u32 = 0xB3;
pub const CUE_TRACK_POSITIONS: u32 = 0xB7;
pub const CUE_TRACK: u32 = 0xF7;
pub const CUE_CLUSTER_POSITION: u32 = 0xF1;
pub const CUE_RELATIVE_POSITION: u32 = 0xF0;

// Attachments (5.1.6).
pub const ATTACHED_FILE: u32 = 0x61A7;
pub const FILE_NAME: u32 = 0x466E;
pub const FILE_MEDIA_TYPE: u32 = 0x4660;
pub const FILE_DATA: u32 = 0x465C;
pub const FILE_UID: u32 = 0x46AE;

/// Elements whose appearance ends an unknown-size Cluster: the Segment's
/// other children and the root (RFC 8794, 6.2: a valid parent, sibling, or
/// root element ends an unknown-sized element).
pub const fn ends_cluster(id: u32) -> bool {
    matches!(
        id,
        CLUSTER | CUES | SEEK_HEAD | INFO | TRACKS | CHAPTERS | TAGS | ATTACHMENTS | SEGMENT | EBML
    )
}
