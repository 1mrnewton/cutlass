//! Edits the interface and the agent both send.
//!
//! A command does no I/O. The project assigns every new id and returns it.

use cutlass_core::Timestamp;

use crate::Error;
use crate::asset::Asset;
use crate::attachment::Attachment;
use crate::id::{AssetId, SegmentId, TrackId};
use crate::rate::SequenceRate;
use crate::segment::{Blend, Crop, Easing, Keyframe, KeyframeProperty, Segment, Transform};
use crate::settings::{AspectRatio, Color, ColorSpace};
use crate::time::{self, Speed, TimeRange};

/// One edit. Several of these can be one undo step.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    AddAsset(Asset),
    /// Drops one asset. A clip that uses it, or a reverse that names it, is refused.
    RemoveAsset(AssetId),
    /// Replaces the source string. Kind, duration, and size stay. The project does not open the path.
    SetSource {
        asset: AssetId,
        source: String,
    },
    Rename(String),
    SetCanvas {
        width: u32,
        height: u32,
        ratio: AspectRatio,
    },
    SetRate(SequenceRate),
    SetColorSpace(ColorSpace),
    SetBackground(Color),
    SetMagnet(bool),
    SetLinkage(bool),
    /// `track: None` opens a track of the segment's kind.
    Place {
        track: Option<TrackId>,
        segment: Segment,
    },
    RemoveSegment(SegmentId),
    Trim {
        segment: SegmentId,
        source: TimeRange,
    },
    SetSpeed {
        segment: SegmentId,
        speed: Speed,
    },
    /// `track: None` keeps the segment on its track. `start` is microseconds.
    Move {
        segment: SegmentId,
        track: Option<TrackId>,
        start: i64,
    },
    /// `at` is a sequence time strictly inside the segment. The left piece keeps
    /// the id. The outcome is the new right piece.
    Split {
        segment: SegmentId,
        at: i64,
    },
    /// Placement on the canvas. It does not move the segment in time.
    SetTransform {
        segment: SegmentId,
        transform: Transform,
    },
    /// Linear gain. `1` is unity. A negative or non-finite gain is refused.
    SetVolume {
        segment: SegmentId,
        volume: f64,
    },
    /// Fade in and fade out, in microseconds. `0` is no fade.
    SetFades {
        segment: SegmentId,
        fade_in: i64,
        fade_out: i64,
    },
    /// Turns noise reduction on or off. A segment with no sound cannot turn it on.
    SetDenoise {
        segment: SegmentId,
        on: bool,
    },
    /// Whether a speed change keeps the pitch. A segment with no sound cannot turn it off.
    SetKeepPitch {
        segment: SegmentId,
        keep: bool,
    },
    /// Four insets of the picture. The whole picture is the start.
    SetCrop {
        segment: SegmentId,
        crop: Crop,
    },
    /// How the picture combines with the canvas. Normal is the start.
    SetBlend {
        segment: SegmentId,
        blend: Blend,
    },
    /// Pulls the sound off a picture-and-sound segment into a new sound segment.
    DetachAudio {
        segment: SegmentId,
    },
    /// Splits the segment at `at` and inserts a still of one source tick.
    ///
    /// `at` is a sequence time strictly inside the segment. `hold` is how long
    /// the still sits, in microseconds.
    Freeze {
        segment: SegmentId,
        at: i64,
        hold: i64,
    },
    /// Points the segment at the reverse of its asset. The timeline range stays.
    Reverse {
        segment: SegmentId,
    },
    /// Replaces the words on a text segment.
    SetText {
        segment: SegmentId,
        words: String,
    },
    /// Replaces the name on a sticker segment. An empty name is refused.
    SetSticker {
        segment: SegmentId,
        name: String,
    },
    /// Sets or clears the plugin attachment. `None` clears it.
    SetAttachment {
        segment: SegmentId,
        attachment: Option<Attachment>,
    },
    /// Locks the track when on. A locked track refuses an edit.
    SetLocked {
        track: TrackId,
        on: bool,
    },
    /// Hides the track when on. The preview skips it.
    SetHidden {
        track: TrackId,
        on: bool,
    },
    /// Mutes the track when on. The mix does not play it yet.
    SetMuted {
        track: TrackId,
        on: bool,
    },
    /// Shown when on. It does not move the segment.
    SetVisible {
        segment: SegmentId,
        visible: bool,
    },
    /// One sample. The offset is microseconds from the segment's timeline start.
    /// Its easing blends it into the next sample of the same property.
    AddKeyframe {
        segment: SegmentId,
        keyframe: Keyframe,
    },
    /// Removes the sample of `property` at `offset_us`.
    RemoveKeyframe {
        segment: SegmentId,
        offset_us: i64,
        property: KeyframeProperty,
    },
    /// Sets how the sample at `offset_us` blends into the next one of `property`.
    SetEasing {
        segment: SegmentId,
        offset_us: i64,
        property: KeyframeProperty,
        easing: Easing,
    },
}

/// What one applied command produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Unchanged,
    Asset(AssetId),
    Segment(SegmentId),
    /// The segment landed on a track this command opened.
    Opened {
        segment: SegmentId,
        track: TrackId,
    },
}

/// Which edge a trim drag moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimEdge {
    /// The start of the clip. The other edge stays put.
    Start,
    /// The end of the clip. The start stays put.
    End,
}

/// Commands for dragging one edge by `delta_us` on the sequence clock.
///
/// A start-edge drag that grows the clip moves first, then trims, so the
/// clip never covers more than its final range on the way. An empty vec
/// changes nothing. The source stays inside the asset.
pub fn commands_for_trim(
    segment: &Segment,
    asset: &Asset,
    edge: TrimEdge,
    delta_us: i64,
) -> Result<Vec<Command>, Error> {
    segment.refuse_no_file()?;
    let Some(id) = segment.asset_id() else {
        return Err(Error::Text);
    };
    if asset.id() != id {
        return Err(Error::UnknownAsset(id));
    }
    let Some(source) = segment.source() else {
        return Err(Error::MediaSource);
    };
    let Some(media) = asset.duration() else {
        return Err(Error::ImageSource);
    };
    let old_start = segment.timeline().start().value;
    let old_duration = segment.timeline().duration().value;
    let old_end = old_start
        .checked_add(old_duration)
        .ok_or(Error::DurationOverflow)?;
    let want_duration = match edge {
        TrimEdge::End => old_duration.saturating_add(delta_us).max(1),
        TrimEdge::Start => {
            let start = old_start
                .saturating_add(delta_us)
                .max(0)
                .min(old_end.saturating_sub(1));
            old_end.saturating_sub(start).max(1)
        }
    };
    let scale = source.start().timescale;
    let mut ticks = time::source_ticks(want_duration, scale, segment.speed())?;
    let source_start = source.start().value;
    let source_end = source.end().value;
    let (new_source_start, new_source_duration) = match edge {
        TrimEdge::End => {
            let room = media.value.saturating_sub(source_start).max(0);
            if ticks > room {
                ticks = room;
            }
            if ticks <= 0 {
                return Ok(Vec::new());
            }
            (source_start, ticks)
        }
        TrimEdge::Start => {
            let mut start = source_end.saturating_sub(ticks);
            if start < 0 {
                start = 0;
            }
            let duration = source_end.saturating_sub(start);
            if duration <= 0 {
                return Ok(Vec::new());
            }
            (start, duration)
        }
    };
    let new_source = TimeRange::new(
        Timestamp::new(new_source_start, scale),
        Timestamp::new(new_source_duration, scale),
    )?;
    if new_source == source && edge == TrimEdge::End {
        return Ok(Vec::new());
    }
    let actual = time::timeline_duration(new_source.duration(), segment.speed())?.value;
    let new_start = match edge {
        TrimEdge::End => old_start,
        TrimEdge::Start => old_end.saturating_sub(actual).max(0),
    };
    if new_source == source && new_start == old_start {
        return Ok(Vec::new());
    }
    let trim = Command::Trim {
        segment: segment.id(),
        source: new_source,
    };
    let moved = new_start != old_start;
    let lengthens = actual > old_duration;
    let mut commands = Vec::with_capacity(2);
    if edge == TrimEdge::Start && moved && lengthens {
        commands.push(Command::Move {
            segment: segment.id(),
            track: None,
            start: new_start,
        });
        commands.push(trim);
    } else {
        commands.push(trim);
        if moved {
            commands.push(Command::Move {
                segment: segment.id(),
                track: None,
                start: new_start,
            });
        }
    }
    Ok(commands)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cutlass_core::Timestamp;

    use crate::{Asset, Channels, Command, Segment, Speed, TimeRange, TrimEdge, commands_for_trim};

    fn scale() -> NonZeroU32 {
        NonZeroU32::new(30_000).expect("scale")
    }

    fn clip(source_start: i64, source_duration: i64, timeline_start: i64) -> (Asset, Segment) {
        let asset = Asset::video(
            "file:///cuts/v.mp4",
            Timestamp::new(30_000, scale()),
            320,
            180,
            true,
        )
        .expect("video");
        let source = TimeRange::new(
            Timestamp::new(source_start, scale()),
            Timestamp::new(source_duration, scale()),
        )
        .expect("source");
        let segment = Segment::media(&asset, source, timeline_start, Channels::Both, Speed::ONE)
            .expect("segment");
        (asset, segment)
    }

    #[test]
    fn the_right_edge_shortens_the_source_and_keeps_the_start() {
        let (asset, segment) = clip(0, 30_000, 0);
        let commands = commands_for_trim(&segment, &asset, TrimEdge::End, -500_000).expect("trim");
        let source = TimeRange::new(Timestamp::new(0, scale()), Timestamp::new(15_000, scale()))
            .expect("source");
        assert_eq!(
            commands,
            vec![Command::Trim {
                segment: segment.id(),
                source,
            }]
        );
        assert!(
            commands_for_trim(&segment, &asset, TrimEdge::End, 1_000_000)
                .expect("past the file")
                .is_empty()
        );
        assert!(
            commands_for_trim(&segment, &asset, TrimEdge::End, 0)
                .expect("still")
                .is_empty()
        );
    }

    #[test]
    fn the_left_edge_keeps_the_other_edge_and_grows_by_moving_first() {
        let (asset, segment) = clip(0, 30_000, 0);
        let commands = commands_for_trim(&segment, &asset, TrimEdge::Start, 500_000).expect("trim");
        let source = TimeRange::new(
            Timestamp::new(15_000, scale()),
            Timestamp::new(15_000, scale()),
        )
        .expect("source");
        assert_eq!(
            commands,
            vec![
                Command::Trim {
                    segment: segment.id(),
                    source,
                },
                Command::Move {
                    segment: segment.id(),
                    track: None,
                    start: 500_000,
                },
            ]
        );

        let (asset, segment) = clip(15_000, 15_000, 500_000);
        let commands =
            commands_for_trim(&segment, &asset, TrimEdge::Start, -500_000).expect("grow");
        let source = TimeRange::new(Timestamp::new(0, scale()), Timestamp::new(30_000, scale()))
            .expect("source");
        assert_eq!(
            commands,
            vec![
                Command::Move {
                    segment: segment.id(),
                    track: None,
                    start: 0,
                },
                Command::Trim {
                    segment: segment.id(),
                    source,
                },
            ]
        );
    }
}
