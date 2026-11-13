//! The canvas ratios and backgrounds the phone offers.

use std::collections::HashSet;

use cutlass_model::{Draft, MaterialId, MaterialRef, SegmentId};
use cutlass_session::{CANVAS_SHAPES, Request, canvas_for};

use crate::media::fresh;

/// The swatches the background picker shows, as `#RRGGBB`.
pub const BACKGROUND_COLORS: [&str; 12] = [
    "#000000", "#FFFFFF", "#808080", "#E5484D", "#F76B15", "#FFC53D", "#46A758", "#12A594",
    "#0090FF", "#6E56CF", "#D6409F", "#8D6E63",
];

/// The blur strengths the background picker shows, weakest first, as a
/// fraction of the canvas's short side.
pub const BACKGROUND_BLURS: [f64; 4] = [0.02, 0.05, 0.1, 0.2];

/// What fills the canvas around the pictures on the main lane.
#[derive(Clone, Debug, PartialEq)]
pub enum PhoneBackground {
    /// Nothing: the canvas is clear and shows black.
    None,
    /// The project color, `#RRGGBB`.
    Color(String),
    /// Every main picture over a blurred copy of itself.
    Blur(f64),
}

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

/// Original first, then every named shape, each sized for `draft`. The
/// chosen one is the canvas as it is: an Original canvas keeps the size it
/// took from the first picture even after that picture is replaced or
/// moved, and the phone draws its boxes on that canvas.
pub fn phone_ratios(draft: &Draft) -> Vec<PhoneRatio> {
    let config = &draft.canvas_config;
    let chosen = config.ratio.as_str();
    std::iter::once("original")
        .chain(CANVAS_SHAPES.iter().map(|shape| shape.ratio))
        .filter_map(|ratio| canvas_for(draft, ratio).ok())
        .map(|shape| {
            let chosen = shape.ratio == chosen;
            let (width, height) = if chosen && config.width > 0 && config.height > 0 {
                (config.width, config.height)
            } else {
                (shape.width, shape.height)
            };
            PhoneRatio {
                ratio: shape.ratio.to_owned(),
                label: if shape.ratio == "original" {
                    "Original".to_owned()
                } else {
                    shape.ratio.to_owned()
                },
                width,
                height,
                chosen,
            }
        })
        .collect()
}

/// The background the panel shows as chosen: a blur when every main
/// picture has the same one, else the project color.
pub fn phone_background(draft: &Draft) -> PhoneBackground {
    let blurs: Vec<Option<f64>> = main_pictures(draft)
        .iter()
        .map(|segment_id| blur_of(draft, segment_id))
        .collect();
    if let Some(Some(first)) = blurs.first()
        && blurs.iter().all(|blur| *blur == Some(*first))
    {
        return PhoneBackground::Blur(*first);
    }
    match draft.canvas_config.background.as_deref() {
        Some(color) if !color.is_empty() => PhoneBackground::Color(color.to_owned()),
        _ => PhoneBackground::None,
    }
}

/// The requests that give the canvas `background`, as one step.
///
/// A color is the project color, and the main pictures lose their own
/// backdrops so it shows. A blur goes on every main picture; with none
/// there, it is refused.
pub(crate) fn background_requests(
    draft: &Draft,
    background: &PhoneBackground,
    taken: &HashSet<String>,
) -> Option<Vec<Request>> {
    let pictures = main_pictures(draft);
    match background {
        PhoneBackground::Blur(blur) => {
            if pictures.is_empty() {
                return None;
            }
            let mut taken = taken.clone();
            Some(
                pictures
                    .into_iter()
                    .map(|segment_id| {
                        let canvas_id = fresh(&taken, "canvas");
                        taken.insert(canvas_id.clone());
                        Request::SetLetterboxBlur {
                            segment_id,
                            blur: Some(*blur),
                            canvas_id: MaterialId::new(canvas_id),
                        }
                    })
                    .collect(),
            )
        }
        PhoneBackground::None | PhoneBackground::Color(_) => {
            let color = match background {
                PhoneBackground::Color(color) => color.clone(),
                _ => String::new(),
            };
            let mut requests = vec![Request::SetCanvasBackground { color }];
            requests.extend(
                pictures
                    .into_iter()
                    .map(|segment_id| Request::SetLetterboxBlur {
                        segment_id,
                        blur: None,
                        canvas_id: MaterialId::new(""),
                    }),
            );
            Some(requests)
        }
    }
}

/// The segments on the main lane that show a picture.
fn main_pictures(draft: &Draft) -> Vec<SegmentId> {
    let Some(main) = draft.main_track() else {
        return Vec::new();
    };
    main.segments
        .iter()
        .filter(|segment| {
            matches!(
                draft.materials.get(segment.material_id.as_str()),
                Some(MaterialRef::Video(_) | MaterialRef::Draft(_))
            )
        })
        .map(|segment| segment.id.clone())
        .collect()
}

/// The blur behind a segment's picture, if its canvas is a blur.
fn blur_of(draft: &Draft, segment_id: &SegmentId) -> Option<f64> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| segment.id == *segment_id)?;
    segment
        .extra_material_refs
        .iter()
        .find_map(|id| match draft.materials.get(id.as_str()) {
            Some(MaterialRef::Canvas(canvas)) if canvas.kind == "canvas_blur" => Some(canvas.blur),
            _ => None,
        })
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

        // As choosing 9:16 stores it.
        draft.canvas_config.ratio = "9:16".to_owned();
        draft.canvas_config.width = 1_080;
        draft.canvas_config.height = 1_920;
        let chosen: Vec<_> = phone_ratios(&draft)
            .into_iter()
            .filter(|ratio| ratio.chosen)
            .map(|ratio| (ratio.ratio, ratio.width, ratio.height))
            .collect();
        assert_eq!(chosen, [("9:16".to_owned(), 1_080, 1_920)]);
    }

    #[test]
    fn the_chosen_ratio_is_the_canvas_the_project_has() {
        // An Original canvas set by a wide clip stays wide when a square
        // picture leads later; only choosing Original again re-fits it.
        let mut draft = Draft::empty("D");
        draft.canvas_config.ratio = "original".to_owned();
        draft.canvas_config.width = 1_440;
        draft.canvas_config.height = 1_080;
        let original = &phone_ratios(&draft)[0];
        assert!(original.chosen);
        assert_eq!((original.width, original.height), (1_440, 1_080));

        draft.canvas_config.ratio = "9:16".to_owned();
        let original = &phone_ratios(&draft)[0];
        assert!(!original.chosen);
        assert_eq!((original.width, original.height), (1_920, 1_080));
    }
}
