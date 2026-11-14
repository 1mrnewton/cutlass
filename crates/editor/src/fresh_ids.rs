//! New ids for requests, past every id the draft already holds.
//!
//! A shell counts from zero each launch, so a reopened project would
//! otherwise be handed ids it already uses and the edit would be refused as
//! a duplicate. The desktop and the phones plan their ids here.

use cutlass_model::Draft;

/// New ids `prefix-N` for several requests planned at once, each past
/// every `prefix-N` the draft already holds.
pub struct FreshIds<'a> {
    draft: &'a Draft,
    /// The last number handed out.
    pub next: u64,
    /// The prefixes the draft has been searched for, once each.
    looked: Vec<String>,
}

impl<'a> FreshIds<'a> {
    /// Ids for `draft`, starting past `next`.
    #[must_use]
    pub fn new(draft: &'a Draft, next: u64) -> Self {
        Self {
            draft,
            next,
            looked: Vec::new(),
        }
    }

    /// The next `prefix-N` no string in the draft and no id made before holds.
    pub fn make(&mut self, prefix: &str) -> String {
        if !self.looked.iter().any(|looked| looked == prefix) {
            self.next = self.next.max(highest_suffix(self.draft, prefix));
            self.looked.push(prefix.to_owned());
        }
        self.next = self.next.saturating_add(1);
        format!("{prefix}-{}", self.next)
    }
}

/// The largest `N` of any string `prefix-N` in `draft`, or zero.
#[must_use]
pub fn highest_suffix(draft: &Draft, prefix: &str) -> u64 {
    fn walk(value: &serde_json::Value, prefix: &str, best: &mut u64) {
        match value {
            serde_json::Value::String(text) => {
                let number = text
                    .strip_prefix(prefix)
                    .and_then(|rest| rest.strip_prefix('-'))
                    .and_then(|digits| digits.parse::<u64>().ok());
                if let Some(number) = number {
                    *best = (*best).max(number);
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    walk(item, prefix, best);
                }
            }
            serde_json::Value::Object(map) => {
                for item in map.values() {
                    walk(item, prefix, best);
                }
            }
            _ => {}
        }
    }
    let mut best = 0;
    if let Ok(value) = serde_json::to_value(draft) {
        walk(&value, prefix, &mut best);
    }
    best
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialVideo};

    use super::*;

    #[test]
    fn ids_start_past_every_id_the_draft_holds() {
        let mut draft = Draft::empty("d");
        draft.materials.videos.push(MaterialVideo::new("mat-7"));
        draft.materials.videos.push(MaterialVideo::new("seg-2"));
        let mut ids = FreshIds::new(&draft, 0);
        assert_eq!(ids.make("mat"), "mat-8");
        assert_eq!(ids.make("mat"), "mat-9");
        // One counter for every prefix, so two prefixes never meet.
        assert_eq!(ids.make("seg"), "seg-10");
    }

    #[test]
    fn an_empty_draft_starts_at_one_and_a_counter_carries_on() {
        let draft = Draft::empty("d");
        assert_eq!(FreshIds::new(&draft, 0).make("track"), "track-1");
        assert_eq!(FreshIds::new(&draft, 41).make("track"), "track-42");
    }

    #[test]
    fn a_suffix_that_is_not_a_number_is_not_counted() {
        let mut draft = Draft::empty("d");
        draft.materials.videos.push(MaterialVideo::new("mat-x"));
        draft.materials.videos.push(MaterialVideo::new("mat-3b"));
        draft
            .materials
            .videos
            .push(MaterialVideo::new("material-9"));
        assert_eq!(highest_suffix(&draft, "mat"), 0);
    }
}
