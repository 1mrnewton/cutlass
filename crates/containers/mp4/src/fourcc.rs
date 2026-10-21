//! A four-character box or codec code.

use std::fmt;

/// Four bytes, shown as characters when they are printable ASCII.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fourcc(pub(crate) [u8; 4]);

impl Fourcc {
    /// The four bytes, in order.
    pub const fn new(bytes: [u8; 4]) -> Self {
        Self(bytes)
    }

    /// The four bytes.
    pub const fn bytes(self) -> [u8; 4] {
        self.0
    }
}

impl fmt::Display for Fourcc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            let character = if byte.is_ascii() && !byte.is_ascii_control() {
                char::from(byte)
            } else {
                '.'
            };
            write!(f, "{character}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Fourcc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fourcc({self})")
    }
}
