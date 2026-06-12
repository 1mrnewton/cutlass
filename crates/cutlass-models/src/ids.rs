//! Strongly-typed entity identifiers.
//!
//! Each ID is a distinct newtype around `u64`, so the type system prevents
//! passing (say) a [`ClipId`] where a [`TrackId`] is expected. IDs are cheap to
//! copy, hash, and compare. Use [`from_raw`](ProjectId::from_raw) for
//! deterministic IDs in tests, or `next()` for process-unique allocation.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
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
define_id!(
    /// Identifies a link group: clips sharing a `LinkId` move/trim together
    /// (CapCut linkage — e.g. the video+audio pair from one media drop).
    LinkId
);
define_id!(
    /// Identifies a [`Marker`](crate::Marker) on the timeline ruler.
    MarkerId
);

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    macro_rules! assert_from_raw_roundtrips {
        ($from_raw:path) => {
            assert_eq!($from_raw(0).raw(), 0);
            assert_eq!($from_raw(42).raw(), 42);
            assert_eq!($from_raw(u64::MAX).raw(), u64::MAX);
        };
    }

    // --- from_raw / raw ---------------------------------------------------

    #[test]
    fn project_id_from_raw_roundtrips() {
        assert_from_raw_roundtrips!(ProjectId::from_raw);
    }

    #[test]
    fn media_id_from_raw_roundtrips() {
        assert_from_raw_roundtrips!(MediaId::from_raw);
    }

    #[test]
    fn track_id_from_raw_roundtrips() {
        assert_from_raw_roundtrips!(TrackId::from_raw);
    }

    #[test]
    fn clip_id_from_raw_roundtrips() {
        assert_from_raw_roundtrips!(ClipId::from_raw);
    }

    // --- next() allocation ------------------------------------------------

    #[test]
    fn next_allocates_monotonically_within_a_type() {
        let a = ClipId::next();
        let b = ClipId::next();
        let c = ClipId::next();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert!(a.raw() < b.raw());
        assert!(b.raw() < c.raw());
    }

    #[test]
    fn each_id_type_has_its_own_counter() {
        // Counters are per-type statics — two different types can share the
        // same raw value without being comparable to each other.
        let project = ProjectId::next();
        let media = MediaId::next();
        let track = TrackId::next();
        let clip = ClipId::next();

        // All are freshly allocated; raw values are independent per type.
        assert!(project.raw() >= 1);
        assert!(media.raw() >= 1);
        assert!(track.raw() >= 1);
        assert!(clip.raw() >= 1);
    }

    #[test]
    fn from_raw_and_next_are_independent() {
        let fixed = TrackId::from_raw(999);
        let allocated = TrackId::next();
        assert_eq!(fixed.raw(), 999);
        // `next()` does not consult previously `from_raw` values.
        assert_ne!(allocated, fixed);
    }

    // --- Display ----------------------------------------------------------

    #[test]
    fn display_formats_type_name_and_raw_value() {
        assert_eq!(ProjectId::from_raw(1).to_string(), "ProjectId#1");
        assert_eq!(MediaId::from_raw(2).to_string(), "MediaId#2");
        assert_eq!(TrackId::from_raw(3).to_string(), "TrackId#3");
        assert_eq!(ClipId::from_raw(4).to_string(), "ClipId#4");
    }

    // --- Copy / Clone / Eq ------------------------------------------------

    #[test]
    fn copy_and_clone_preserve_raw_value() {
        let original = MediaId::from_raw(77);
        let copied = original;
        let cloned = original.clone();
        assert_eq!(original, copied);
        assert_eq!(original, cloned);
        assert_eq!(copied.raw(), 77);
        assert_eq!(cloned.raw(), 77);
    }

    #[test]
    fn equal_ids_have_same_raw_value() {
        let a = ClipId::from_raw(10);
        let b = ClipId::from_raw(10);
        assert_eq!(a, b);
        assert_eq!(a.raw(), b.raw());
    }

    #[test]
    fn different_raw_values_are_not_equal() {
        assert_ne!(ClipId::from_raw(1), ClipId::from_raw(2));
    }

    // --- Ord ----------------------------------------------------------------

    #[test]
    fn ordering_follows_raw_numeric_order() {
        let low = TrackId::from_raw(1);
        let mid = TrackId::from_raw(50);
        let high = TrackId::from_raw(100);
        assert!(low < mid);
        assert!(mid < high);
        assert!(high > low);
    }

    #[test]
    fn ids_sort_by_raw_value() {
        let mut ids = [
            ClipId::from_raw(30),
            ClipId::from_raw(10),
            ClipId::from_raw(20),
        ];
        ids.sort();
        assert_eq!(
            ids.map(|id| id.raw()),
            [10, 20, 30]
        );
    }

    // --- Hash -------------------------------------------------------------

    #[test]
    fn hash_is_stable_for_equal_ids() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        fn hash_of<T: Hash>(value: &T) -> u64 {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        }

        let a = MediaId::from_raw(42);
        let b = MediaId::from_raw(42);
        assert_eq!(hash_of(&a), hash_of(&b));
        assert_ne!(hash_of(&a), hash_of(&MediaId::from_raw(43)));
    }

    #[test]
    fn ids_work_as_hashset_keys() {
        let mut set = HashSet::new();
        let a = ProjectId::from_raw(1);
        let b = ProjectId::from_raw(2);
        assert!(set.insert(a));
        assert!(set.insert(b));
        assert!(!set.insert(a)); // duplicate
        assert_eq!(set.len(), 2);
        assert!(set.contains(&ProjectId::from_raw(1)));
    }

    // --- Debug ------------------------------------------------------------

    #[test]
    fn debug_includes_type_and_value() {
        let s = format!("{:?}", ClipId::from_raw(5));
        assert!(s.contains("ClipId"));
        assert!(s.contains('5'));
    }
}
