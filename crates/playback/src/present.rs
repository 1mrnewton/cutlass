//! The preview picture at one time.
//!
//! This is `eval::frame`. The late-frame drop is not applied here; playback
//! drops a late picture on its clock, and export keeps one on its clock.

use cutlass_eval::{Device, EvalError, Frame, Micros, Snapshot};

/// The software frame playback and export both start from.
pub fn present<D: Device>(
    snapshot: &Snapshot<'_>,
    time: Micros,
    device: &mut D,
) -> Result<Frame, EvalError<D::Error>> {
    cutlass_eval::frame(snapshot, time, device)
}

#[cfg(test)]
mod tests {
    use cutlass_eval::{Snapshot, frame};
    use cutlass_model::{Draft, MaterialColorConfig, Micros, Segment, TimeRange};
    use cutlass_renderer_software::SoftwareDevice;

    use super::present;

    #[test]
    fn present_is_the_eval_frame_including_the_top_layer() {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 1;
        draft.canvas_config.height = 1;
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "blue".into(),
            is_color_clip: true,
            solid_color: "0xFF0000FF".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "red".into(),
            is_color_clip: true,
            solid_color: "0xFFFF0000".to_owned(),
            ..MaterialColorConfig::default()
        });
        let mut under = Segment::new("under", "blue", TimeRange::new(0, 1_000));
        under.render_index = 1;
        let mut over = Segment::new("over", "red", TimeRange::new(0, 1_000));
        over.render_index = 2;
        draft.tracks[0].segments.push(over);
        draft.tracks[0].segments.push(under);
        let before = draft.clone();
        let mut device = SoftwareDevice::new();
        let snapshot = Snapshot::new(&draft);
        let played = present(&snapshot, Micros(0), &mut device).expect("present");
        let evaluated = frame(&snapshot, Micros(0), &mut device).expect("eval");
        assert_eq!(played.rgba, [255, 0, 0, 255]);
        assert_eq!(played.rgba, evaluated.rgba);
        assert_eq!(draft, before);
    }
}
