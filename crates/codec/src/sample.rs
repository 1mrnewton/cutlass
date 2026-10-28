/// PCM layout of an uncompressed audio sample entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SampleFormat {
    #[default]
    Unknown,
    U8,
    S16Le,
    S16Be,
    S24Le,
    S24Be,
    S32Le,
    S32Be,
    F32Le,
    F32Be,
    F64Le,
    F64Be,
}
