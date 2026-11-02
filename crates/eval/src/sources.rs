use cutlass_generated::SolidColor;
use cutlass_model::{Draft, MaterialRef, TrackAttribute};

pub(crate) struct SolidLayer {
    pub(crate) render_index: i32,
    pub(crate) track_index: usize,
    pub(crate) segment_index: usize,
    pub(crate) solid: SolidColor,
}

/// Solid color clips that cover part of the timeline.
///
/// A hidden track, an invisible segment, a gradient, and a color that is not
/// a packed `0xAARRGGBB` value contribute nothing. The caller sorts by
/// `render_index`, then track, then segment. The solid fills the canvas.
pub(crate) fn solid_layers(draft: &Draft, width: u32, height: u32) -> Vec<SolidLayer> {
    let mut layers = Vec::new();
    for (track_index, track) in draft.tracks.iter().enumerate() {
        if track.attribute.contains(TrackAttribute::HIDE) {
            continue;
        }
        for (segment_index, segment) in track.segments.iter().enumerate() {
            if !segment.visible {
                continue;
            }
            let Some(MaterialRef::Color(color)) = draft.materials.get(segment.material_id.as_str())
            else {
                continue;
            };
            if !color.is_color_clip || color.is_gradient {
                continue;
            }
            let Some(packed) = cutlass_generated::parse_packed(&color.solid_color) else {
                continue;
            };
            let Ok(solid) = SolidColor::new(
                width,
                height,
                packed,
                segment.target.start,
                segment.target.duration,
            ) else {
                continue;
            };
            layers.push(SolidLayer {
                render_index: segment.render_index,
                track_index,
                segment_index,
                solid,
            });
        }
    }
    layers
}
