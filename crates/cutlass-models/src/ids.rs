//! Strongly-typed entity identifiers.
//!
//! Each ID is a distinct newtype around `u64`, so the type system prevents
//! passing (say) a [`ClipId`] where a [`TrackId`] is expected. IDs are cheap to
//! copy, hash, and compare. Use [`from_raw`](ProjectId::from_raw) for
//! deterministic IDs in tests, or `next()` for process-unique allocation.

use std::sync::atomic::{AtomicU64, Ordering};

macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(u64);

        impl $name {
            /// Construct from a raw value (useful for deterministic tests).
            pub const fn from_raw(value: u64) -> Self {
                Self(value)
            }

            /// The underlying numeric value.
            pub const fn raw(self) -> u64 {
                self.0
            }

            /// Allocate the next process-unique ID for this type.
            pub fn next() -> Self {
                static COUNTER: AtomicU64 = AtomicU64::new(1);
                Self(COUNTER.fetch_add(1, Ordering::Relaxed))
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}#{}", stringify!($name), self.0)
            }
        }
    };
}

define_id!(
    /// Identifies a [`Project`](crate::Project).
    ProjectId
);
define_id!(
    /// Identifies a [`MediaSource`](crate::MediaSource) in the media pool.
    MediaId
);
define_id!(
    /// Identifies a [`Track`](crate::Track) within the timeline.
    TrackId
);
define_id!(
    /// Identifies a [`Clip`](crate::Clip) placed on a track.
    ClipId
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_distinct_types() {
        let a = ClipId::next();
        let b = ClipId::next();
        assert_ne!(a, b);
        assert!(b.raw() > a.raw());
    }

    #[test]
    fn from_raw_roundtrips() {
        assert_eq!(TrackId::from_raw(42).raw(), 42);
    }
}
