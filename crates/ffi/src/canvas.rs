//! The canvas ratios the phone offers.

use cutlass_model::Draft;
use cutlass_session::{CANVAS_SHAPES, canvas_for};

/// One ratio in the picker and the frame it gives this project.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneRatio {
    /// What the draft stores, such as `original` or `9:16`.
    pub ratio: String,
    pub label: String,
    pub width: i32,
    pub height: i32,
    /// The canvas has this ratio now.
    pub chosen: bool,
}

/// Original first, then every named shape, each sized for `draft`.
pub fn phone_ratios(draft: &Draft) -> Vec<PhoneRatio> {
    let chosen = draft.canvas_config.ratio.as_str();
    std::iter::once("original")
        .chain(CANVAS_SHAPES.iter().map(|shape| shape.ratio))
        .filter_map(|ratio| canvas_for(draft, ratio).ok())
        .map(|shape| PhoneRatio {
            ratio: shape.ratio.to_owned(),
            label: if shape.ratio == "original" {
                "Original".to_owned()
            } else {
                shape.ratio.to_owned()
            },
            width: shape.width,
            height: shape.height,
            chosen: shape.ratio == chosen,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use cutlass_model::Draft;

    use super::phone_ratios;

    #[test]
    fn original_leads_and_the_stored_ratio_is_chosen() {
        let mut draft = Draft::empty("D");
        let ratios = phone_ratios(&draft);
        assert_eq!(ratios.len(), 14);
        assert_eq!(ratios[0].label, "Original");
        assert!(ratios[0].chosen);
        assert_eq!(ratios.iter().filter(|ratio| ratio.chosen).count(), 1);

        draft.canvas_config.ratio = "9:16".to_owned();
        let chosen: Vec<_> = phone_ratios(&draft)
            .into_iter()
            .filter(|ratio| ratio.chosen)
            .map(|ratio| (ratio.ratio, ratio.width, ratio.height))
            .collect();
        assert_eq!(chosen, [("9:16".to_owned(), 1_080, 1_920)]);
    }
}
