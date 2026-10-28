//! Which cached previews to drop so a new file fits.
//!
//! Victims are cached previews, oldest first. A pinned file, a bundled
//! package, and a cached package are not victims. A plan that cannot make
//! room returns no victims, so the caller deletes nothing.

use crate::ResourceId;

/// The cache budget, in bytes. Bundled files are not counted against it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quota {
    limit: u64,
}

impl Quota {
    #[must_use]
    pub const fn bytes(limit: u64) -> Self {
        Self { limit }
    }

    #[must_use]
    pub const fn limit(self) -> u64 {
        self.limit
    }
}

/// One indexed file, as the planner sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub id: ResourceId,
    pub size: u64,
    pub age: u64,
    /// A cached preview that nothing is showing.
    pub evictable: bool,
}

/// The incoming file does not fit, even after every eligible preview is dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shortfall {
    pub limit: u64,
    pub kept: u64,
    pub incoming: u64,
}

/// Oldest evictable previews that make `used + incoming` fit in `limit`.
///
/// `used` already includes every entry. An entry with size 0 is not a victim.
/// The result is empty when nothing has to be dropped.
pub fn select(
    entries: &[Entry],
    used: u64,
    incoming: u64,
    limit: u64,
) -> Result<Vec<ResourceId>, Shortfall> {
    if fits(used, incoming, limit) {
        return Ok(Vec::new());
    }
    let mut ranked: Vec<&Entry> = entries
        .iter()
        .filter(|entry| entry.evictable && entry.size > 0)
        .collect();
    ranked.sort_by(|left, right| {
        left.age
            .cmp(&right.age)
            .then_with(|| left.id.as_str().cmp(right.id.as_str()))
    });
    let mut chosen = Vec::new();
    let mut freed = 0u64;
    for entry in ranked {
        let Some(next) = freed.checked_add(entry.size) else {
            break;
        };
        if used.checked_sub(next).is_none() {
            break;
        }
        chosen.push(entry.id.clone());
        freed = next;
        if fits(used - freed, incoming, limit) {
            return Ok(chosen);
        }
    }
    Err(Shortfall {
        limit,
        kept: used.saturating_sub(freed),
        incoming,
    })
}

fn fits(used: u64, incoming: u64, limit: u64) -> bool {
    match used.checked_add(incoming) {
        Some(total) => total <= limit,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{Entry, Quota, select};
    use crate::ResourceId;

    fn entry(id: &str, size: u64, age: u64, evictable: bool) -> Entry {
        Entry {
            id: ResourceId::new(id).unwrap(),
            size,
            age,
            evictable,
        }
    }

    #[test]
    fn an_exact_fit_drops_nothing() {
        let entries = [entry("a", 10, 1, true)];
        assert!(select(&entries, 10, 0, 10).unwrap().is_empty());
        assert!(select(&[], 0, 0, 0).unwrap().is_empty());
        assert_eq!(Quota::bytes(10).limit(), 10);
    }

    #[test]
    fn the_oldest_preview_is_chosen_and_a_tie_uses_the_id() {
        let entries = [
            entry("newer", 3, 2, true),
            entry("oldest", 3, 1, true),
            entry("also", 3, 3, true),
        ];
        let victims = select(&entries, 9, 0, 6).unwrap();
        assert_eq!(victims, vec![ResourceId::new("oldest").unwrap()]);
        let tied = [entry("b", 3, 1, true), entry("a", 3, 1, true)];
        let victims = select(&tied, 6, 0, 3).unwrap();
        assert_eq!(victims, vec![ResourceId::new("a").unwrap()]);
    }

    #[test]
    fn a_pinned_package_or_empty_preview_is_not_a_victim() {
        let entries = [
            entry("pinned", 5, 1, false),
            entry("package", 5, 0, false),
            entry("empty", 0, 2, true),
            entry("old", 4, 3, true),
        ];
        let victims = select(&entries, 14, 0, 10).unwrap();
        assert_eq!(victims, vec![ResourceId::new("old").unwrap()]);
    }

    #[test]
    fn a_file_that_cannot_fit_drops_nothing() {
        let entries = [entry("pinned", 5, 1, false)];
        let error = select(&entries, 5, 6, 10).unwrap_err();
        assert_eq!(error.limit, 10);
        assert_eq!(error.kept, 5);
        assert_eq!(error.incoming, 6);
        let error = select(&[], 0, 1, 0).unwrap_err();
        assert_eq!(error.incoming, 1);
        let overflow = select(&[], 1, u64::MAX, u64::MAX).unwrap_err();
        assert_eq!(overflow.kept, 1);
        let error = select(&[entry("old", 5, 1, true)], 5, 11, 10).unwrap_err();
        assert_eq!(error.kept, 0);
        assert_eq!(error.incoming, 11);
    }
}
