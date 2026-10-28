use cutlass_codec::StreamParams;

/// One edit from an edit list.
///
/// `segment_duration` is in the movie header timescale. `media_time` is in the
/// track media timescale. `None` is an empty edit (a dwell). The rate is the
/// 16.16 value stored in the file. Packet timestamps are not rewritten from this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edit {
    pub segment_duration: u64,
    pub media_time: Option<i64>,
    pub media_rate: u32,
}

/// A track the demuxer can read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stream {
    pub params: StreamParams,
    pub edits: Vec<Edit>,
}
