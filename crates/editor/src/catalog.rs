//! The bundled catalog the agent's `catalog.search` reads: effects,
//! filters, transitions and animations, each as one line with the id a
//! request takes.

use cutlass_model::{ANIMATION_PRESETS, AnimType};

use crate::effect_shelf::{effect_cards, filter_cards};
use crate::transition_shelf::transition_offers;

/// One line per bundled item: its kind, the id a request takes, then its
/// name. `effect recipe/shake Shake`, `transition cutlass-dissolve Dissolve`.
#[must_use]
pub fn catalog_entries() -> Vec<String> {
    let mut entries = Vec::new();
    for card in effect_cards() {
        entries.push(format!("effect {} {}", card.effect_id, card.name));
    }
    for (card, name) in filter_cards() {
        let id = card
            .strip_prefix(crate::effect_shelf::FILTER_CARD_PREFIX)
            .unwrap_or(&card);
        entries.push(format!("filter {id} {name}"));
    }
    for offer in transition_offers() {
        entries.push(format!(
            "transition {} {}",
            offer.kind.effect_id(),
            offer.name
        ));
    }
    for preset in ANIMATION_PRESETS {
        let kind = match preset.kind {
            AnimType::In => "in",
            AnimType::Out => "out",
            AnimType::Loop => "loop",
            AnimType::Caption => "caption",
            AnimType::Group => "group",
            AnimType::None => "none",
        };
        entries.push(format!(
            "animation-{kind} {} {}",
            preset.resource_id, preset.name
        ));
    }
    entries
}

/// Whether `id` names an item Cutlass ships. Every package is bundled in
/// the first release, so this is the whole answer to `package.ensure`.
#[must_use]
pub fn is_bundled(id: &str) -> bool {
    catalog_entries()
        .iter()
        .any(|entry| entry.split(' ').nth(1) == Some(id))
}

/// The entries every word of `query` is in, ignoring case. A blank query
/// lists them all.
#[must_use]
pub fn catalog_search(query: &str) -> Vec<String> {
    let words: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
    catalog_entries()
        .into_iter()
        .filter(|entry| {
            let entry = entry.to_lowercase();
            words.iter().all(|word| entry.contains(word.as_str()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_is_listed_and_words_narrow_it() {
        let all = catalog_entries();
        for kind in ["effect ", "filter ", "transition ", "animation-in "] {
            assert!(all.iter().any(|entry| entry.starts_with(kind)), "no {kind}");
        }
        assert_eq!(catalog_search("").len(), all.len());
        let dissolve = catalog_search("Transition DISSOLVE");
        assert_eq!(dissolve.len(), 1, "{dissolve:?}");
        assert!(
            dissolve[0].starts_with("transition cutlass-"),
            "{dissolve:?}"
        );
        let fades = catalog_search("animation fade");
        assert!(fades.iter().any(|entry| entry.contains("Fade In")));
        assert!(catalog_search("no such thing at all").is_empty());
    }

    #[test]
    fn a_listed_id_is_bundled_and_a_name_or_kind_is_not() {
        for entry in catalog_entries() {
            let id = entry.split(' ').nth(1).unwrap_or_default();
            assert!(is_bundled(id), "{entry}");
        }
        assert!(is_bundled("cutlass-dissolve"));
        assert!(!is_bundled("Dissolve"));
        assert!(!is_bundled("transition"));
        assert!(!is_bundled("recipe/not-a-package"));
    }
}
