//! A plane's bytes: owned, or a shared view into a buffer another owner
//! keeps (a decoder's reference picture), so that a decoded picture can
//! reach the caller without a copy.

use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// Bytes a [`Samples`] can share: `bytes()` must return the same bytes for
/// as long as the value lives.
pub trait SharedBytes: Send + Sync {
    fn bytes(&self) -> &[u8];
}

/// A plane's sample bytes. Reads see a slice either way; the first write
/// to a shared view copies it out (copy on write).
#[derive(Clone)]
pub struct Samples(Repr);

#[derive(Clone)]
enum Repr {
    Owned(Vec<u8>),
    Shared {
        buf: Arc<dyn SharedBytes>,
        start: usize,
        len: usize,
    },
}

impl Samples {
    /// A view of `len` bytes from `start` of `buf`; `None` if they do not
    /// lie in it.
    pub fn shared(buf: Arc<dyn SharedBytes>, start: usize, len: usize) -> Option<Self> {
        let end = start.checked_add(len)?;
        if end > buf.bytes().len() {
            return None;
        }
        Some(Self(Repr::Shared { buf, start, len }))
    }

    /// The bytes as an owned vector (copied when shared).
    #[must_use]
    pub fn into_vec(self) -> Vec<u8> {
        match self.0 {
            Repr::Owned(v) => v,
            Repr::Shared { buf, start, len } => buf.bytes()[start..start + len].to_vec(),
        }
    }

    /// The owned vector, copying a shared view out first.
    pub fn make_mut(&mut self) -> &mut Vec<u8> {
        if let Repr::Shared { buf, start, len } = &self.0 {
            let v = buf.bytes()[*start..*start + *len].to_vec();
            self.0 = Repr::Owned(v);
        }
        match &mut self.0 {
            Repr::Owned(v) => v,
            Repr::Shared { .. } => unreachable!("made owned above"),
        }
    }

    /// The bytes.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        self
    }

    /// Whether the bytes are a view into another owner's buffer.
    #[must_use]
    pub fn is_shared(&self) -> bool {
        matches!(self.0, Repr::Shared { .. })
    }
}

impl Default for Samples {
    fn default() -> Self {
        Self(Repr::Owned(Vec::new()))
    }
}

impl Deref for Samples {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match &self.0 {
            Repr::Owned(v) => v,
            Repr::Shared { buf, start, len } => &buf.bytes()[*start..*start + *len],
        }
    }
}

impl DerefMut for Samples {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.make_mut()
    }
}

impl From<Vec<u8>> for Samples {
    fn from(v: Vec<u8>) -> Self {
        Self(Repr::Owned(v))
    }
}

impl From<Samples> for Vec<u8> {
    fn from(s: Samples) -> Self {
        s.into_vec()
    }
}

impl AsRef<[u8]> for Samples {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl PartialEq for Samples {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for Samples {}

impl PartialEq<Vec<u8>> for Samples {
    fn eq(&self, other: &Vec<u8>) -> bool {
        **self == other[..]
    }
}

impl PartialEq<[u8]> for Samples {
    fn eq(&self, other: &[u8]) -> bool {
        **self == *other
    }
}

impl PartialEq<&[u8]> for Samples {
    fn eq(&self, other: &&[u8]) -> bool {
        **self == **other
    }
}

impl PartialEq<Samples> for Vec<u8> {
    fn eq(&self, other: &Samples) -> bool {
        self[..] == **other
    }
}

impl fmt::Debug for Samples {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}
