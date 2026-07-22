//! End-to-end smoke: resolve + realize a typewriter text clip through the
//! GPU glyph path and confirm the frame has ink.

use cutlass_core::{Rational, RationalTime};
use cutlass_models::{
    AnimationRef, AnimationSlot, Generator, Project, TextStyle as ModelTextStyle, TimeRange,
    TrackKind,
};
use cutlass_render::Renderer;

const FPS_24: Rational = Rational::FPS_24;

#[test]
fn typewriter_text_renders_glyph_coverage() {
    let mut project = Project::new("glyph-smoke", FPS_24);
    let track = project.add_track(TrackKind::Text, "T1");
    let clip = project
        .add_generated(
            track,
            Generator::Text {
                content: "Cutlass".into(),
                style: ModelTextStyle {
                    size: 72.0,
                    fill: [255, 255, 255, 255],
                    ..ModelTextStyle::default()
                },
            },
            TimeRange::at_rate(0, 48, FPS_24),
        )
        .unwrap();
    project
        .set_clip_animation(
            clip,
            AnimationSlot::Combo,
            Some(AnimationRef::new("typewriter")),
        )
        .unwrap();

    let mut renderer = match Renderer::new_headless() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("skipping glyph smoke: no GPU ({e})");
            return;
        }
    };

    // Mid-period: some characters revealed.
    let img = renderer
        .render_frame(&project, RationalTime::new(12, FPS_24))
        .expect("render typewriter frame");
    let lit = img.pixels.chunks_exact(4).filter(|p| p[3] > 0).count();
    assert!(
        lit > 0,
        "typewriter mid-reveal should produce glyph coverage"
    );

    // Near end of period: more (or all) characters visible.
    let later = renderer
        .render_frame(&project, RationalTime::new(36, FPS_24))
        .expect("render later frame");
    let lit_later = later.pixels.chunks_exact(4).filter(|p| p[3] > 0).count();
    assert!(
        lit_later >= lit,
        "later typewriter phase should not lose coverage ({lit_later} < {lit})"
    );
}
