//! Byte-budgeted LRU caches for CPU (and similarly metered) raster memos.
//!
//! Entry-count caps are unsafe once supersampled bitmaps approach the texture
//! edge (a single 4096² RGBA raster is 64 MiB). These caches evict oldest
//! entries by total payload bytes instead.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

/// Default soft budget for a CPU raster memo (text bitmaps, shaped clusters,
/// pen-path rasters). 256 MiB holds a few max-edge frames while bounding a
/// scale-ramp across quantization steps to well under a gigabyte.
pub const RASTER_MEMO_BUDGET_BYTES: usize = 256 * 1024 * 1024;

/// LRU map that tracks a caller-supplied byte cost per entry and evicts the
/// oldest entries until a insert fits under [`Self::budget`].
///
/// **Oversized entries** (`cost > budget`) bypass the cache entirely — they
/// are returned to the caller but never retained, so one absurd raster cannot
/// pin the whole budget alone.
#[derive(Debug, Clone)]
pub struct ByteBudgetLru<K, V> {
    map: HashMap<K, (V, usize)>,
    order: VecDeque<K>,
    bytes: usize,
    budget: usize,
}

impl<K, V> ByteBudgetLru<K, V>
where
    K: Eq + Hash + Clone,
{
    pub fn new(budget: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            budget: budget.max(1),
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
        self.bytes = 0;
    }

    /// Touch-on-read lookup. Returns a clone so the caller can drop the borrow.
    pub fn get_cloned(&mut self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        if !self.map.contains_key(key) {
            return None;
        }
        self.touch(key);
        self.map.get(key).map(|(v, _)| v.clone())
    }

    /// Insert `value` costing `cost` bytes. No-ops (bypass) when `cost` alone
    /// exceeds the budget. Otherwise LRU-evicts until it fits, then inserts
    /// (replacing any prior entry for `key`).
    pub fn insert(&mut self, key: K, value: V, cost: usize) {
        if cost > self.budget {
            return;
        }
        self.remove(&key);
        while self.bytes + cost > self.budget {
            if !self.evict_oldest() {
                break;
            }
        }
        if self.bytes + cost > self.budget {
            // Budget smaller than cost should have returned above; defensive.
            return;
        }
        self.bytes += cost;
        self.order.push_back(key.clone());
        self.map.insert(key, (value, cost));
    }

    fn touch(&mut self, key: &K) {
        if let Some(i) = self.order.iter().position(|k| k == key) {
            if let Some(k) = self.order.remove(i) {
                self.order.push_back(k);
            }
        }
    }

    fn remove(&mut self, key: &K) {
        if let Some((_, cost)) = self.map.remove(key) {
            self.bytes = self.bytes.saturating_sub(cost);
            self.order.retain(|k| k != key);
        }
    }

    fn evict_oldest(&mut self) -> bool {
        let Some(old) = self.order.pop_front() else {
            return false;
        };
        if let Some((_, cost)) = self.map.remove(&old) {
            self.bytes = self.bytes.saturating_sub(cost);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_beyond_budget_evicts_oldest() {
        let mut cache = ByteBudgetLru::new(100);
        cache.insert("a", vec![0u8; 40], 40);
        cache.insert("b", vec![0u8; 40], 40);
        cache.insert("c", vec![0u8; 40], 40); // needs room → evict a
        assert!(cache.get_cloned(&"a").is_none());
        assert!(cache.get_cloned(&"b").is_some());
        assert!(cache.get_cloned(&"c").is_some());
        assert!(cache.bytes() <= cache.budget());
        assert_eq!(cache.bytes(), 80);
    }

    #[test]
    fn get_touches_lru_order() {
        let mut cache = ByteBudgetLru::new(100);
        cache.insert("a", 1u8, 40);
        cache.insert("b", 2u8, 40);
        // Touch a so b is oldest.
        assert_eq!(cache.get_cloned(&"a"), Some(1));
        cache.insert("c", 3u8, 40); // evicts b
        assert!(cache.get_cloned(&"b").is_none());
        assert_eq!(cache.get_cloned(&"a"), Some(1));
        assert_eq!(cache.get_cloned(&"c"), Some(3));
    }

    #[test]
    fn oversized_entry_bypasses_cache() {
        let mut cache = ByteBudgetLru::new(50);
        cache.insert("tiny", 1u8, 10);
        cache.insert("huge", 2u8, 51); // > budget → bypass
        assert_eq!(cache.get_cloned(&"tiny"), Some(1));
        assert!(cache.get_cloned(&"huge").is_none());
        assert_eq!(cache.bytes(), 10);
    }

    #[test]
    fn replace_same_key_updates_bytes() {
        let mut cache = ByteBudgetLru::new(100);
        cache.insert("a", vec![0u8; 30], 30);
        cache.insert("a", vec![0u8; 70], 70);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes(), 70);
    }
}
