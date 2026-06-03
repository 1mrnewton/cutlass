//! Timeline resolution: turn a timeline frame into the ordered layers to draw.
//!
//! Given a timeline frame, [`resolve_frame`] walks the track stack bottom-to-top
//! and collects every enabled video clip covering that frame, mapping each
//! media clip to the exact *source* frame to decode. The result is the input the
//! compositor consumes: a back-to-front list of layers it blends into one image.
//!
//! # Frame-rate conformance
//!
//! A clip's `timeline` range is in timeline frames; its `source` range is in the
//! media's native frames. [`Project::add_clip`] conforms the *duration* between
//! rates, so the two ranges span the same wall-clock time but differ in count
//! when the rates differ (a 30fps source on a 24fps timeline). Mapping a
//! position therefore needs a rate conversion, not a raw offset add:
//!
//! ```text
//! timeline_offset = n - clip.timeline.start          (timeline frames)
//! source_offset   = convert(timeline_offset, tl_fps, media_fps)
//! source_frame    = clip.source.start + source_offset (source frames)
//! ```
//!
//! Audio tracks are skipped here; audio mixing is resolved separately.

use cutlass_models::{
    ClipId, ClipSource, Generator, MediaId, Project, TrackId, TrackKind, convert_frames,
};

/// What a single resolved layer draws at the requested frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerContent<'a> {
    /// A media frame: decode `source_frame` of `media` and draw it.
    Media { media: MediaId, source_frame: i64 },
    /// Engine-generated content (text, solid, shape, adjustment).
    Generated(&'a Generator),
}

/// One layer of a resolved frame, tagged with its originating clip and track.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedLayer<'a> {
    pub track: TrackId,
    pub clip: ClipId,
    pub content: LayerContent<'a>,
}

/// Resolve the layers covering `timeline_frame`, ordered back-to-front.
///
/// Only **enabled video** tracks contribute. Index 0 of the result is the
/// bottommost layer; the last element is the topmost. Media clips referencing
/// media missing from the pool are skipped (referential integrity normally
/// prevents this, but resolution stays total rather than panicking).
pub fn resolve_frame(project: &Project, timeline_frame: i64) -> Vec<ResolvedLayer<'_>> {
    let timeline = project.timeline();
    let timeline_fps = timeline.frame_rate;
    let mut layers = Vec::new();

    for &track_id in timeline.order() {
        let Some(track) = timeline.track(track_id) else {
            continue;
        };
        if track.kind != TrackKind::Video || !track.enabled {
            continue;
        }
        let Some(clip) = track.clip_at(timeline_frame) else {
            continue;
        };

        let content = match &clip.content {
            ClipSource::Generated(generator) => LayerContent::Generated(generator),
            ClipSource::Media { media, source } => {
                let Some(media_source) = project.media(*media) else {
                    continue;
                };
                let timeline_offset = timeline_frame - clip.timeline.start;
                let source_offset =
                    convert_frames(timeline_offset, timeline_fps, media_source.frame_rate);
                // Clamp into the clip's source range: rate rounding can land one
                // past the end near the clip boundary.
                let last_offset = (source.duration - 1).max(0);
                let source_frame = source.start + source_offset.clamp(0, last_offset);
                LayerContent::Media {
                    media: *media,
                    source_frame,
                }
            }
        };

        layers.push(ResolvedLayer {
            track: track_id,
            clip: clip.id,
            content,
        });
    }

    layers
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::{Generator, MediaSource, Rational, TimeRange, TrackKind};

    fn project_30fps_media_on_24fps_timeline() -> (Project, MediaId, TrackId) {
        let mut project = Project::new("test", Rational::FPS_24);
        let media = MediaSource::new("/tmp/a.mp4", 1920, 1080, Rational::FPS_30, 3000, false);
        let media_id = project.add_media(media);
        let track = project.add_track(TrackKind::Video, "V1");
        (project, media_id, track)
    }

    #[test]
    fn empty_frame_resolves_to_no_layers() {
        let (project, _, _) = project_30fps_media_on_24fps_timeline();
        assert!(resolve_frame(&project, 0).is_empty());
    }

    #[test]
    fn media_clip_maps_timeline_frame_to_source_frame() {
        let (mut project, media_id, track) = project_30fps_media_on_24fps_timeline();
        // Source frames [300, 600) at 30fps; placed at timeline frame 0.
        project
            .add_clip(track, media_id, TimeRange::new(300, 300), 0)
            .unwrap();

        // At the clip start, the source frame is exactly the in-point.
        let layers = resolve_frame(&project, 0);
        assert_eq!(layers.len(), 1);
        assert_eq!(
            layers[0].content,
            LayerContent::Media {
                media: media_id,
                source_frame: 300,
            }
        );

        // 24 timeline frames in (1s @ 24fps) == 30 source frames in (1s @ 30fps).
        let layers = resolve_frame(&project, 24);
        assert_eq!(
            layers[0].content,
            LayerContent::Media {
                media: media_id,
                source_frame: 330,
            }
        );
    }

    #[test]
    fn generated_clip_resolves_to_generator() {
        let (mut project, _, track) = project_30fps_media_on_24fps_timeline();
        project
            .add_generated(
                track,
                Generator::Text {
                    content: "hi".into(),
                },
                TimeRange::new(0, 48),
            )
            .unwrap();

        let layers = resolve_frame(&project, 10);
        assert_eq!(layers.len(), 1);
        match layers[0].content {
            LayerContent::Generated(Generator::Text { content }) => assert_eq!(content, "hi"),
            other => panic!("expected text generator, got {other:?}"),
        }
    }

    #[test]
    fn layers_are_ordered_bottom_to_top() {
        let mut project = Project::new("test", Rational::FPS_24);
        let bottom = project.add_track(TrackKind::Video, "V1");
        let top = project.add_track(TrackKind::Video, "V2");
        project
            .add_generated(
                bottom,
                Generator::SolidColor { rgba: [0, 0, 0, 255] },
                TimeRange::new(0, 48),
            )
            .unwrap();
        project
            .add_generated(
                top,
                Generator::Text { content: "x".into() },
                TimeRange::new(0, 48),
            )
            .unwrap();

        let layers = resolve_frame(&project, 5);
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].track, bottom, "index 0 is bottommost");
        assert_eq!(layers[1].track, top, "last is topmost");
    }

    #[test]
    fn disabled_video_track_is_skipped() {
        let (mut project, media_id, track) = project_30fps_media_on_24fps_timeline();
        project
            .add_clip(track, media_id, TimeRange::new(0, 300), 0)
            .unwrap();
        project.timeline_mut().track_mut(track).unwrap().enabled = false;

        assert!(resolve_frame(&project, 5).is_empty());
    }

    #[test]
    fn audio_track_is_skipped() {
        let mut project = Project::new("test", Rational::FPS_24);
        let audio = project.add_track(TrackKind::Audio, "A1");
        project
            .add_generated(
                audio,
                Generator::SolidColor { rgba: [1, 2, 3, 4] },
                TimeRange::new(0, 48),
            )
            .unwrap();
        assert!(resolve_frame(&project, 5).is_empty());
    }

    #[test]
    fn source_frame_is_clamped_within_clip_range() {
        let (mut project, media_id, track) = project_30fps_media_on_24fps_timeline();
        // Short clip: source [0,10) at 30fps -> 8 timeline frames @ 24fps.
        project
            .add_clip(track, media_id, TimeRange::new(0, 10), 0)
            .unwrap();

        // Last covered timeline frame must still map inside [0, 10).
        let end = project.timeline().duration();
        let layers = resolve_frame(&project, end - 1);
        if let LayerContent::Media { source_frame, .. } = layers[0].content {
            assert!((0..10).contains(&source_frame), "got {source_frame}");
        } else {
            panic!("expected media layer");
        }
    }
}
