/// Colour tags from a container header, as stored (ISO colour primaries and transfer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorTags {
    pub primaries: u16,
    pub transfer: u16,
    pub matrix: u16,
    pub full_range: bool,
}
