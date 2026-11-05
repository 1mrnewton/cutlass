//! A transition package on the outgoing segment.
//!
//! The cut is that segment's target end. The two target ranges still only
//! touch. `is_overlap` false keeps the window on the outgoing side. `is_overlap`
//! true splits it, with the outgoing half rounded toward zero. Playback
//! shortens a side that would run past either segment. It does not move the
//! ranges. A duration of 0, a negative duration, a gap, or a missing incoming
//! segment is a hard cut.
//!
//! Before the cut, the incoming picture is sampled from the incoming head.
//! After the cut, the outgoing picture holds its last frame. A built-in
//! transition (`cutlass_effects::TransitionKind`), or the fade package as a
//! dissolve, mixes those two pictures. A missing package, an unreadable
//! package, or any other package stays the hard cut: the segment that covers the time, alone.
//! A text segment on the side that does not cover the time also stays a hard
//! cut. The draft is not written.

use std::collections::HashMap;

use cutlass_compositor::{Device, crossfade};
use cutlass_effects::{
    PackageKind, PackageStore, Transition, TransitionKind, entry, load_transition, mix_transition,
};
use cutlass_model::{Draft, MaterialRef, MaterialTransition, Micros, Segment, TrackAttribute};
use cutlass_picture::CpuBytes;

use crate::EvalError;
use crate::media::video_picture;
use crate::nest::{NestPass, picture_at};

pub(crate) struct Cut {
    outgoing_track: usize,
    outgoing_segment: usize,
    incoming_track: usize,
    incoming_segment: usize,
    cover_track: usize,
    cover_segment: usize,
    outgoing_time: Micros,
    incoming_time: Micros,
    numer: u64,
    denom: u64,
}

impl Cut {
    pub(crate) fn cover(&self) -> (usize, usize) {
        (self.cover_track, self.cover_segment)
    }

    pub(crate) fn partner(&self) -> (usize, usize, Micros) {
        if self.cover_segment == self.outgoing_segment {
            (
                self.incoming_track,
                self.incoming_segment,
                self.incoming_time,
            )
        } else {
            (
                self.outgoing_track,
                self.outgoing_segment,
                self.outgoing_time,
            )
        }
    }

    pub(crate) fn ordered<'a>(
        &self,
        covering: &'a [u8],
        partner: &'a [u8],
    ) -> (&'a [u8], &'a [u8]) {
        if self.cover_segment == self.outgoing_segment {
            (covering, partner)
        } else {
            (partner, covering)
        }
    }
}

pub(crate) struct Prepared {
    cuts: Vec<Cut>,
    pictures: HashMap<(usize, usize), CpuBytes>,
}

impl Prepared {
    pub(crate) fn cuts(&self) -> &[Cut] {
        &self.cuts
    }

    pub(crate) fn picture(&self, track: usize, segment: usize) -> Option<&CpuBytes> {
        self.pictures.get(&(track, segment))
    }
}

/// Pictures for the side of each cut that does not cover `time`.
pub(crate) fn prepare<D: Device>(
    draft: &Draft,
    store: &dyn PackageStore,
    time: Micros,
    width: u32,
    height: u32,
    pass: &mut NestPass<'_, D>,
) -> Result<Prepared, EvalError<D::Error>> {
    let mut cuts = cuts_at(draft, time);
    cuts.retain(|cut| is_fade(store, draft, cut));
    let mut pictures = HashMap::new();
    for cut in &cuts {
        let (track, segment, when) = cut.partner();
        let Some(partner) = draft
            .tracks
            .get(track)
            .and_then(|item| item.segments.get(segment))
        else {
            continue;
        };
        match draft.materials.get(partner.material_id.as_str()) {
            Some(MaterialRef::Video(_)) => {
                if let Some(bytes) =
                    video_picture(draft, partner, pass.media, width, height, when, pass.clips)
                {
                    pictures.insert((track, segment), bytes);
                }
            }
            Some(MaterialRef::Draft(_)) => {
                if let Some(bytes) = picture_at(draft, partner, when, pass)? {
                    pictures.insert((track, segment), bytes);
                }
            }
            _ => {}
        }
    }
    Ok(Prepared { cuts, pictures })
}

/// The mixed picture for a cut's transition, or nothing when it names no
/// transition Cutlass draws.
pub(crate) fn fade(
    store: &dyn PackageStore,
    draft: &Draft,
    cut: &Cut,
    outgoing: &[u8],
    incoming: &[u8],
    width: u32,
    height: u32,
) -> Option<CpuBytes> {
    let rgba = match kind_of(store, draft, cut)? {
        TransitionKind::Dissolve => crossfade(outgoing, incoming, cut.numer, cut.denom)?,
        kind => {
            #[allow(clippy::cast_precision_loss)]
            let t = (cut.numer as f64 / cut.denom as f64) as f32;
            mix_transition(kind, outgoing, incoming, width, height, t)?
        }
    };
    CpuBytes::new(width, height, rgba).ok()
}

/// The transition on a cut: a built-in one by its material, or the fade
/// package as a dissolve.
fn kind_of(store: &dyn PackageStore, draft: &Draft, cut: &Cut) -> Option<TransitionKind> {
    let transition = transition_on(draft, cut.outgoing_track, cut.outgoing_segment)?;
    if let Some(kind) = TransitionKind::of(transition) {
        return Some(kind);
    }
    let package = load_transition(store, transition).ok()?;
    (entry(package.id()).map(|row| row.kind) == Some(PackageKind::Transition(Transition::Fade)))
        .then_some(TransitionKind::Dissolve)
}

fn is_fade(store: &dyn PackageStore, draft: &Draft, cut: &Cut) -> bool {
    kind_of(store, draft, cut).is_some()
}

fn cuts_at(draft: &Draft, time: Micros) -> Vec<Cut> {
    if draft.materials.transitions.is_empty() {
        return Vec::new();
    }
    let mut cuts = Vec::new();
    for (track_index, track) in draft.tracks.iter().enumerate() {
        if track.attribute.contains(TrackAttribute::HIDE) {
            continue;
        }
        let Some(cover) = track.segments.iter().position(|segment| {
            segment.visible && covers(segment.target.start.0, segment.target.duration.0, time.0)
        }) else {
            continue;
        };
        if let Some(cut) = cut_for(draft, track_index, cover, time.0) {
            cuts.push(cut);
        }
    }
    cuts
}

fn cut_for(draft: &Draft, track_index: usize, cover: usize, time: i64) -> Option<Cut> {
    let track = draft.tracks.get(track_index)?;
    let segment = track.segments.get(cover)?;
    if let Some(window) = window_from(draft, track_index, cover)
        && time >= window.start
        && time < window.end
        && time < window.cut
    {
        return window.at(track_index, time);
    }
    let previous = track.segments.iter().position(|other| {
        other.id != segment.id
            && other.visible
            && other.target.start.0.checked_add(other.target.duration.0)
                == Some(segment.target.start.0)
    })?;
    let window = window_from(draft, track_index, previous)?;
    if time >= window.start && time < window.end && time >= window.cut {
        return window.at(track_index, time);
    }
    None
}

struct Window {
    outgoing: usize,
    incoming: usize,
    cut: i64,
    start: i64,
    end: i64,
}

impl Window {
    fn at(self, track: usize, time: i64) -> Option<Cut> {
        let numer = u64::try_from(time.checked_sub(self.start)?).ok()?;
        let denom = u64::try_from(self.end.checked_sub(self.start)?).ok()?;
        if denom == 0 || numer >= denom {
            return None;
        }
        let outgoing_time = if time < self.cut {
            time
        } else {
            self.cut.checked_sub(1)?
        };
        let incoming_time = if time >= self.cut {
            time
        } else {
            self.cut.checked_add(time.checked_sub(self.start)?)?
        };
        let cover = if time < self.cut {
            self.outgoing
        } else {
            self.incoming
        };
        Some(Cut {
            outgoing_track: track,
            outgoing_segment: self.outgoing,
            incoming_track: track,
            incoming_segment: self.incoming,
            cover_track: track,
            cover_segment: cover,
            outgoing_time: Micros(outgoing_time),
            incoming_time: Micros(incoming_time),
            numer,
            denom,
        })
    }
}

fn window_from(draft: &Draft, track_index: usize, outgoing_index: usize) -> Option<Window> {
    let track = draft.tracks.get(track_index)?;
    let outgoing = track.segments.get(outgoing_index)?;
    if !outgoing.visible {
        return None;
    }
    let transition = transition_on(draft, track_index, outgoing_index)?;
    let (left, right) = cutlass_model::blend_extents(transition.duration, transition.is_overlap)?;
    let out_len = outgoing.target.duration.0;
    if out_len <= 0 {
        return None;
    }
    let cut = outgoing.target.start.0.checked_add(out_len)?;
    let incoming = track.segments.iter().position(|other| {
        other.id != outgoing.id && other.visible && other.target.start.0 == cut
    })?;
    let in_len = track.segments.get(incoming)?.target.duration.0;
    if in_len <= 0 {
        return None;
    }
    let left = left.0.min(out_len).min(in_len);
    let right = right.0.min(in_len);
    if left == 0 && right == 0 {
        return None;
    }
    let start = cut.checked_sub(left)?;
    let end = cut.checked_add(right)?;
    if start >= end {
        return None;
    }
    Some(Window {
        outgoing: outgoing_index,
        incoming,
        cut,
        start,
        end,
    })
}

fn transition_on(
    draft: &Draft,
    track_index: usize,
    segment_index: usize,
) -> Option<&MaterialTransition> {
    let segment = draft.tracks.get(track_index)?.segments.get(segment_index)?;
    first_transition(draft, segment)
}

fn first_transition<'a>(draft: &'a Draft, segment: &Segment) -> Option<&'a MaterialTransition> {
    segment
        .extra_material_refs
        .iter()
        .find_map(|id| match draft.materials.get(id.as_str()) {
            Some(MaterialRef::Transition(transition)) => Some(transition),
            _ => None,
        })
}

fn covers(start: i64, duration: i64, time: i64) -> bool {
    if duration <= 0 {
        return false;
    }
    let Some(end) = start.checked_add(duration) else {
        return false;
    };
    time >= start && time < end
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs::{self, File};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use cutlass_codec::CodecId;
    use cutlass_container::{Demux, Input, OpenError, Probe, ProbeResult};
    use cutlass_decode::{CodecError, CodecParams, Decoder};
    use cutlass_effects::{PackageError, PackageStore, package::Located};
    use cutlass_engine::{DisplayQuarter, ReaderFactory};
    use cutlass_model::{
        Draft, MaterialColorConfig, MaterialDraft, MaterialTransition, MaterialVideo, Micros,
        Segment, TimeRange,
    };
    use cutlass_png::PngDecoder;
    use cutlass_renderer_software::SoftwareDevice;
    use cutlass_still::{StillDemux, StillProbe, still_limits};
    use cutlass_storage::ResourceId;

    use super::cuts_at;
    use crate::frame::{Snapshot, frame};
    use crate::media::OpenClips;

    fn two_colors(incoming_start: i64, incoming_len: i64, duration: i64, overlap: bool) -> Draft {
        let mut draft = Draft::empty("draft");
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "out".into(),
            is_color_clip: true,
            solid_color: "0xFFFF0000".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "in".into(),
            is_color_clip: true,
            solid_color: "0xFF0000FF".to_owned(),
            ..MaterialColorConfig::default()
        });
        let mut outgoing = Segment::new("out", "out", TimeRange::new(0, 100));
        outgoing.extra_material_refs.push("blend".into());
        draft.tracks[0].segments.push(outgoing);
        draft.tracks[0].segments.push(Segment::new(
            "in",
            "in",
            TimeRange::new(incoming_start, incoming_len),
        ));
        draft.materials.transitions.push(MaterialTransition {
            id: "blend".into(),
            effect_id: "transition-fade".to_owned(),
            duration: Micros(duration),
            is_overlap: overlap,
            ..MaterialTransition::default()
        });
        draft
    }

    #[test]
    fn a_non_overlap_window_ends_at_the_cut() {
        let draft = two_colors(100, 100, 10, false);
        let cut = &cuts_at(&draft, Micros(95))[0];
        assert_eq!(cut.numer, 5);
        assert_eq!(cut.denom, 10);
        assert_eq!(cut.outgoing_time, Micros(95));
        assert_eq!(cut.incoming_time, Micros(105));
        assert_eq!(cut.cover_segment, 0);
        assert!(cuts_at(&draft, Micros(89)).is_empty());
        assert!(cuts_at(&draft, Micros(100)).is_empty());
        assert!(cuts_at(&two_colors(100, 100, 0, false), Micros(95)).is_empty());
        assert!(cuts_at(&two_colors(100, 100, -1, true), Micros(95)).is_empty());
        assert!(cuts_at(&two_colors(110, 100, 10, false), Micros(95)).is_empty());
    }

    #[test]
    fn overlap_holds_the_outgoing_frame_after_the_cut() {
        let draft = two_colors(100, 100, 10, true);
        let before = &cuts_at(&draft, Micros(99))[0];
        assert_eq!(before.outgoing_time, Micros(99));
        assert_eq!(before.incoming_time, Micros(104));
        assert_eq!((before.numer, before.denom), (4, 10));
        let after = &cuts_at(&draft, Micros(100))[0];
        assert_eq!(after.outgoing_time, Micros(99));
        assert_eq!(after.incoming_time, Micros(100));
        assert_eq!(after.cover_segment, 1);
        assert_eq!((after.numer, after.denom), (5, 10));
        assert!(cuts_at(&draft, Micros(105)).is_empty());
    }

    #[test]
    fn a_short_incoming_clip_shortens_the_window() {
        let draft = two_colors(100, 30, 50, false);
        let early = &cuts_at(&draft, Micros(70))[0];
        assert_eq!(early.incoming_time, Micros(100));
        assert_eq!(early.denom, 30);
        let late = &cuts_at(&draft, Micros(99))[0];
        assert_eq!(late.incoming_time, Micros(129));
        assert!(cuts_at(&draft, Micros(69)).is_empty());
    }

    #[test]
    fn a_built_in_transition_mixes_with_no_package_store() {
        let mut draft = palette(100, 100, 10, false);
        draft.materials.transitions[0].effect_id = "cutlass-dip-white".to_owned();
        assert_eq!(pixels(&draft, None, 95), [255, 255, 255, 255]);
        assert_eq!(pixels(&draft, None, 89), [255, 0, 0, 255]);
        // A dip stored before the catalog had ids is known by its name.
        draft.materials.transitions[0].effect_id = String::new();
        draft.materials.transitions[0].name = "Dip".to_owned();
        assert_eq!(pixels(&draft, None, 95), [0, 0, 0, 255]);
        draft.materials.transitions[0].name = "Dissolve".to_owned();
        let middle = pixels(&draft, None, 95);
        assert!(middle[0] > 100 && middle[2] > 100, "{middle:?}");
    }

    #[test]
    fn a_missing_package_stays_a_hard_cut_and_a_fade_mixes() {
        let draft = palette(100, 100, 10, false);
        let before = draft.clone();
        assert_eq!(pixels(&draft, None, 95), [255, 0, 0, 255]);
        assert_eq!(pixels(&draft, None, 100), [0, 0, 255, 255]);
        assert_eq!(pixels(&draft, None, 89), [255, 0, 0, 255]);
        assert_eq!(draft, before);

        let mut hostile = palette(100, 100, 10, false);
        hostile.materials.transitions[0].effect_id = "../secret".to_owned();
        assert_eq!(pixels(&hostile, None, 95), [255, 0, 0, 255]);

        let store = publish("transition-fade");
        assert_eq!(pixels(&draft, Some(&store), 95), [127, 0, 127, 255]);
        assert_eq!(pixels(&draft, Some(&store), 100), [0, 0, 255, 255]);
        assert_eq!(draft, before);

        let overlap = palette(100, 100, 10, true);
        assert_eq!(pixels(&overlap, None, 100), [0, 0, 255, 255]);
        assert_eq!(pixels(&overlap, Some(&store), 100), [127, 0, 127, 255]);
        assert_eq!(pixels(&overlap, Some(&store), 104), [25, 0, 229, 255]);
        assert_eq!(pixels(&overlap, Some(&store), 105), [0, 0, 255, 255]);

        let gap = palette(110, 100, 10, false);
        assert_eq!(pixels(&gap, Some(&store), 95), [255, 0, 0, 255]);
        assert_eq!(
            pixels(&palette(100, 100, 0, false), Some(&store), 95),
            [255, 0, 0, 255]
        );
        assert_eq!(
            pixels(&palette(100, 100, -1, true), Some(&store), 95),
            [255, 0, 0, 255]
        );

        let mut other = palette(100, 100, 10, false);
        other.materials.transitions[0].effect_id = "other-pack".to_owned();
        let other_store = publish("other-pack");
        assert_eq!(pixels(&other, Some(&other_store), 95), [255, 0, 0, 255]);
    }

    #[test]
    fn a_fade_mixes_the_nested_picture() {
        let mut child = Draft::empty("child");
        child.canvas_config.width = 1;
        child.canvas_config.height = 1;
        child.materials.material_colors.push(MaterialColorConfig {
            id: "blue".into(),
            is_color_clip: true,
            solid_color: "0xFF0000FF".to_owned(),
            ..MaterialColorConfig::default()
        });
        child.tracks[0]
            .segments
            .push(Segment::new("blue", "blue", TimeRange::new(0, 100)));
        let mut draft = palette(100, 100, 10, false);
        draft.materials.drafts.push(MaterialDraft {
            id: "nest".into(),
            draft: Some(Box::new(child)),
            ..MaterialDraft::default()
        });
        let mut incoming = Segment::new("nest", "nest", TimeRange::new(100, 100));
        incoming.source_range = Some(TimeRange::new(0, 100));
        draft.tracks[0].segments[1] = incoming;
        let store = publish("transition-fade");
        assert_eq!(pixels(&draft, None, 95), [255, 0, 0, 255]);
        assert_eq!(pixels(&draft, Some(&store), 95), [127, 0, 127, 255]);
    }

    #[test]
    fn a_fade_mixes_a_still_with_the_incoming_color() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine/fixtures/pattern.png");
        assert_mixed_still(&path);
        let local = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files/04-motion-graphics.png");
        if local.is_file() {
            assert_mixed_still(&local);
        }
    }

    fn assert_mixed_still(path: &Path) {
        let media = stills();
        let (width, height) = oriented(&media, path);
        let alone = Draft::empty("draft");
        let mut alone = alone;
        alone.canvas_config.width = i32::try_from(width).expect("width");
        alone.canvas_config.height = i32::try_from(height).expect("height");
        let mut video = MaterialVideo::new("still");
        video.path = path.display().to_string();
        alone.materials.videos.push(video);
        let mut segment = Segment::new("still", "still", TimeRange::new(0, 100));
        segment.source_range = Some(TimeRange::new(0, 100));
        alone.tracks[0].segments.push(segment);
        let still = draw(&alone, &media, None, 0);

        let mut draft = alone.clone();
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "blue".into(),
            is_color_clip: true,
            solid_color: "0xFF0000FF".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.tracks[0].segments[0]
            .extra_material_refs
            .push("blend".into());
        draft.tracks[0]
            .segments
            .push(Segment::new("in", "blue", TimeRange::new(100, 100)));
        draft.materials.transitions.push(MaterialTransition {
            id: "blend".into(),
            effect_id: "transition-fade".to_owned(),
            duration: Micros(10),
            ..MaterialTransition::default()
        });
        let before = draft.clone();
        assert_eq!(draw(&draft, &media, None, 95), still);
        let store = publish("transition-fade");
        assert_eq!(draw(&draft, &media, Some(&store), 95), mixed(&still));
        assert_eq!(draft, before);
    }

    fn mixed(still: &[u8]) -> Vec<u8> {
        let blue = [0_u8, 0, 255, 255];
        still
            .chunks(4)
            .flat_map(|pixel| {
                (0..4).map(|channel| {
                    let value = u16::from(pixel[channel]) * 5 + u16::from(blue[channel]) * 5;
                    u8::try_from(value / 10).expect("byte")
                })
            })
            .collect()
    }

    fn draw(draft: &Draft, media: &ReaderFactory, store: Option<&Store>, time: i64) -> Vec<u8> {
        let mut device = SoftwareDevice::new();
        let mut clips = OpenClips::new();
        match store {
            Some(store) => frame(
                &Snapshot::with_packages(draft, store),
                Micros(time),
                &mut device,
                media,
                &mut clips,
                0,
            ),
            None => frame(
                &Snapshot::new(draft),
                Micros(time),
                &mut device,
                media,
                &mut clips,
                0,
            ),
        }
        .expect("frame")
        .rgba
    }

    fn oriented(media: &ReaderFactory, path: &Path) -> (u32, u32) {
        let info = media.probe(path).expect("probe");
        let width = info.width.expect("width");
        let height = info.height.expect("height");
        match info.display {
            DisplayQuarter::Clockwise90 | DisplayQuarter::Clockwise270 => (height, width),
            DisplayQuarter::Zero | DisplayQuarter::Half => (width, height),
        }
    }

    fn stills() -> ReaderFactory {
        let mut factory = ReaderFactory::new();
        factory.register_container(probe_still, open_still);
        factory.decoders_mut().register(CodecId::Png, open_png);
        factory
    }

    fn probe_still(header: &[u8]) -> ProbeResult {
        StillProbe.probe(header)
    }

    fn open_still(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
        let file = File::open(path).map_err(|error| OpenError::Source {
            message: error.to_string(),
        })?;
        let len = file
            .metadata()
            .map_err(|error| OpenError::Source {
                message: error.to_string(),
            })?
            .len();
        Ok(Box::new(StillDemux::open(Input::random(
            file,
            len,
            still_limits(),
        ))?))
    }

    fn open_png(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
        Ok(Box::new(PngDecoder::open(params)?))
    }

    fn palette(incoming_start: i64, incoming_len: i64, duration: i64, overlap: bool) -> Draft {
        let mut draft = two_colors(incoming_start, incoming_len, duration, overlap);
        draft.canvas_config.width = 1;
        draft.canvas_config.height = 1;
        draft
    }

    fn pixels(draft: &Draft, store: Option<&Store>, time: i64) -> Vec<u8> {
        let mut device = SoftwareDevice::new();
        let media = ReaderFactory::new();
        let mut clips = OpenClips::new();
        let rgba = match store {
            Some(store) => frame(
                &Snapshot::with_packages(draft, store),
                Micros(time),
                &mut device,
                &media,
                &mut clips,
                0,
            ),
            None => frame(
                &Snapshot::new(draft),
                Micros(time),
                &mut device,
                &media,
                &mut clips,
                0,
            ),
        }
        .expect("frame")
        .rgba;
        assert_eq!(rgba.len(), 4);
        rgba
    }

    struct Store {
        files: BTreeMap<String, Located>,
        _temp: Temp,
    }

    struct Temp {
        path: PathBuf,
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn publish(id: &str) -> Store {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "cutlass-eval-transition-{id}-{}-{n}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("temp");
        let file = path.join(id);
        let bytes = b"fade";
        fs::write(&file, bytes).expect("package");
        let mut files = BTreeMap::new();
        files.insert(
            id.to_owned(),
            Located {
                path: file,
                size: u64::try_from(bytes.len()).expect("size"),
                checksum_hex: None,
            },
        );
        Store {
            files,
            _temp: Temp { path },
        }
    }

    impl PackageStore for Store {
        fn locate(&self, id: &ResourceId) -> Result<Located, PackageError> {
            self.files
                .get(id.as_str())
                .cloned()
                .ok_or_else(|| PackageError::Missing {
                    id: id.as_str().to_owned(),
                })
        }
    }
}
