//! The shapes the sticker shelf offers, as library cards.

use cutlass_model::ShapeKind;

/// A shape card's id starts with this, then the shape's name.
pub const SHAPE_CARD_PREFIX: &str = "shape:";

/// The shapes in shelf order, as `(card id, kind)`.
#[must_use]
pub fn shape_cards() -> Vec<(String, ShapeKind)> {
    ShapeKind::ALL
        .into_iter()
        .map(|kind| (format!("{SHAPE_CARD_PREFIX}{}", kind.name()), kind))
        .collect()
}

/// The shape a card id names.
#[must_use]
pub fn shape_of_card(card_id: &str) -> Option<ShapeKind> {
    ShapeKind::from_name(card_id.strip_prefix(SHAPE_CARD_PREFIX)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_card_names_its_shape() {
        let cards = shape_cards();
        assert_eq!(cards.len(), ShapeKind::ALL.len());
        for (id, kind) in cards {
            assert_eq!(shape_of_card(&id), Some(kind));
        }
        assert_eq!(shape_of_card("shape:Hexagon"), None);
        assert_eq!(shape_of_card("effect:Rectangle"), None);
    }
}
