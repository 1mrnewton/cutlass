mod adjust;
mod animation;
mod beat;
mod blend;
mod canvas;
mod color;
mod crop;
mod effect;
mod extract;
mod fade;
mod freeze;
mod grade;
mod import;
mod keyframe;
mod lane;
mod linkage;
mod magnet;
mod mark;
mod mirror;
mod move_segment;
mod name;
mod own;
mod place;
mod placement;
mod rate;
mod reverse;
mod rotate;
mod speed;
mod split;
mod text;
mod track;
mod transition;
mod trim;
mod volume;
mod words;

use cutlass_model::{Draft, LinkageSettings, Micros, RangeError, TimeRange, TrackAttribute};

use crate::request::Request;

pub use canvas::{CANVAS_SHAPES, CanvasShape, canvas_for, follow_first_picture, ratio_request};
pub use move_segment::move_segment;
pub use place::place;
pub use text::NEW_TEXT_SIZE;
pub use trim::trim;
pub use words::set_words;

/// Why one request did not change the draft.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditError {
    /// The caller read a different revision than the open draft.
    RevisionMismatch {
        expected: u64,
        actual: u64,
    },
    MissingTrack,
    MissingSegment,
    /// The segment id is empty or already on the draft.
    DuplicateSegment,
    /// The track attribute has the lock bit set.
    LockedTrack,
    /// Two positive spans on one track overlap. Touching does not.
    Overlap,
    /// The constant-speed mapping refused the duration or the speed.
    Speed(cutlass_model::ConstantSpeedError),
    /// Word columns disagree, or a word ends before it starts.
    WordShape,
    Range(RangeError),
    EmptyUndo,
    EmptyRedo,
    /// `begin` or `record` was called while a transaction was already open.
    HistoryOpen,
    /// `commit`, `amend`, or `rollback` needs the transaction `begin` or `record` opened.
    HistoryClosed,
    /// `amend` folds into a recorded step, and none exists yet.
    NothingToAmend,
    /// The bit is not mute, hide, lock, solo, or muted-by-solo.
    UnknownAttribute,
    MissingMaterial,
    /// A trim reaches before the media's first frame or past its last.
    SourceBounds,
    /// A clip would start before the timeline does.
    BeforeZero,
    /// The lane holds another kind of clip: sound on a picture lane, a
    /// title on a sound lane, and so on.
    LaneKind,
    /// The material id is already in a pool.
    DuplicateMaterial,
    /// The split time falls inside a blend window.
    TransitionWindow,
    /// The dependent already has an owner, or it is an owner itself.
    AlreadyLinked,
    /// A marker is already stored at that time.
    DuplicateMark,
    /// A beat is already stored at that source time.
    DuplicateBeat,
    /// A beat can be stored only on an audio segment.
    NotAudio,
    /// Extract audio applies to a video clip.
    NotVideo,
    /// The picture has no sound to place.
    NoSound,
    /// The picture's sound is already its own clip.
    AlreadyExtracted,
    /// A track with this id is already on the draft, or this type cannot be a lane.
    TrackKind,
    /// The main track cannot be removed.
    MainTrack,
    /// The rotation is not a finite number of degrees.
    Angle,
    /// Position is not a finite number.
    Position,
    /// Scale is not finite and greater than zero.
    Scale,
    /// A fade duration is negative.
    Fade,
    /// Volume is negative or not a finite number.
    Volume,
    /// Opacity is not finite or is outside zero to one.
    Opacity,
    /// A key's time is outside its clip, or the clip's keys do not read.
    KeyTime,
    /// No key is at that time.
    NoKey,
    /// The curve is not one the Graphs panel offers.
    Graph,
    /// The animation is not one the Animations panel offers for that side.
    Animation,
    /// The segment is not a text clip.
    NotText,
    /// Text size is not finite, or it is outside zero to 512.
    TextSize,
    /// Text color is not empty and not a hex color.
    TextColor,
    /// Text alignment is not left, center, or right.
    TextAlign,
    /// Letter spacing is not a fraction from -1 through 4.
    LetterSpacing,
    /// Line spacing is not greater than -1 and at most 4.
    LineSpacing,
    /// The wrap width is negative or past the picture edge.
    LineWidth,
    /// The face is empty or is not a file name.
    TextFont,
    /// An outline, shadow, or background amount is out of its range.
    TextStyle,
    /// A new text clip needs something to say.
    TextEmpty,
    /// The effect is not one the app ships.
    UnknownEffect,
    /// The clip is not an effect, a filter, or an adjustment.
    NotEffect,
    /// An effect binds to a picture clip, and only an effect clip binds.
    EffectTarget,
    /// An effect strength is outside 0 through 1.
    EffectStrength,
    /// The canvas color is not empty and not a color the preview paints.
    CanvasColor,
    /// A canvas side is outside 1 through 7680.
    CanvasSize,
    /// The canvas ratio is not one of the project shapes.
    CanvasRatio,
    /// A letterbox color belongs on a video or a composition.
    Letterbox,
    /// A letterbox blur is not a fraction from 0 through 1.
    LetterboxBlur,
    /// The frame rate is not a clock the draft can step.
    FrameRate,
    /// The color space is not one the player keeps.
    ColorSpace,
}

pub fn apply_request(draft: &mut Draft, request: &Request) -> Result<(), EditError> {
    match request {
        Request::Place { track_id, segment } => place(draft, track_id, (**segment).clone()),
        Request::Move {
            segment_id,
            track_id,
            start,
        } => move_segment(draft, segment_id, track_id, *start),
        Request::Trim {
            segment_id,
            edge,
            time,
        } => trim(draft, segment_id, *edge, *time),
        Request::SetWords { material_id, words } => set_words(draft, material_id, words.clone()),
        Request::AddCaption {
            source_id,
            track_id,
            segment_id,
            material_id,
            words,
        } => words::add_caption(
            draft,
            source_id,
            track_id,
            segment_id,
            material_id,
            words.clone(),
        ),
        Request::ExtractAudio {
            source_id,
            track_id,
            segment_id,
            material_id,
        } => extract::extract_audio(draft, source_id, track_id, segment_id, material_id),
        Request::SetTrackAttribute { track_id, bit, on } => {
            track::set_track_attribute(draft, track_id, *bit, *on)
        }
        Request::SetConstantSpeed { segment_id, speed } => {
            speed::set_constant_speed(draft, segment_id, *speed)
        }
        Request::SetCurveSpeed {
            segment_id,
            speed_id,
            points,
        } => speed::set_curve_speed(draft, segment_id, speed_id, points),
        Request::SetTransition {
            segment_id,
            transition,
        } => transition::set_transition(draft, segment_id, (**transition).clone()),
        Request::Split {
            segment_id,
            at,
            right_id,
        } => split::split(draft, segment_id, *at, right_id),
        Request::Link { owner, dependent } => linkage::link(draft, owner, dependent),
        Request::Remove { segment_id } => linkage::remove(draft, segment_id),
        Request::SetReverse {
            segment_id,
            reverse,
            path,
        } => reverse::set_reverse(draft, segment_id, *reverse, path),
        Request::SetFreeze { segment_id, freeze } => {
            freeze::set_freeze(draft, segment_id, freeze.clone())
        }
        Request::SetActive { segment_id, active } => mirror::set_active(draft, segment_id, *active),
        Request::SetMirror {
            segment_id,
            horizontal,
        } => mirror::set_mirror(draft, segment_id, *horizontal),
        Request::SetVerticalFlip {
            segment_id,
            vertical,
        } => mirror::set_vertical(draft, segment_id, *vertical),
        Request::SetRotation {
            segment_id,
            degrees,
        } => rotate::set_rotation(draft, segment_id, *degrees),
        Request::SetPosition { segment_id, x, y } => {
            placement::set_position(draft, segment_id, *x, *y)
        }
        Request::SetScale { segment_id, scale } => placement::set_scale(draft, segment_id, *scale),
        Request::SetOpacity {
            segment_id,
            opacity,
        } => placement::set_opacity(draft, segment_id, *opacity),
        Request::SetVolume { segment_id, volume } => volume::set_volume(draft, segment_id, *volume),
        Request::SetFade {
            segment_id,
            fade_id,
            fade_in,
            fade_out,
        } => fade::set_fade(draft, segment_id, fade_id, *fade_in, *fade_out),
        Request::SetCrop { segment_id, crop } => crop::set_crop(draft, segment_id, *crop),
        Request::SetTextContent {
            segment_id,
            content,
        } => text::set_text(draft, segment_id, content),
        Request::SetTextSize {
            segment_id,
            size,
            face,
        } => text::set_text_size(draft, segment_id, *size, face),
        Request::SetTextColor {
            segment_id,
            color,
            face,
        } => text::set_text_color(draft, segment_id, color, face),
        Request::SetTextAlign {
            segment_id,
            alignment,
            face,
            line_width,
        } => text::set_text_align(draft, segment_id, *alignment, face, *line_width),
        Request::SetLetterSpacing {
            segment_id,
            spacing,
            face,
        } => text::set_letter_spacing(draft, segment_id, *spacing, face),
        Request::SetLineSpacing {
            segment_id,
            spacing,
            face,
        } => text::set_line_spacing(draft, segment_id, *spacing, face),
        Request::SetLineWidth {
            segment_id,
            width,
            face,
        } => text::set_line_width(draft, segment_id, *width, face),
        Request::SetTextFont { segment_id, face } => text::set_text_font(draft, segment_id, face),
        Request::SetTextOutline {
            segment_id,
            color,
            width,
        } => text::set_text_outline(draft, segment_id, color, *width),
        Request::SetTextShadow {
            segment_id,
            color,
            distance,
            angle,
            blur,
        } => text::set_text_shadow(draft, segment_id, color, [*distance, *angle, *blur]),
        Request::SetTextBackground {
            segment_id,
            color,
            alpha,
            round,
        } => text::set_text_background(draft, segment_id, color, *alpha, *round),
        Request::SetTextGlow {
            segment_id,
            glow_id,
            color,
            strength,
            range,
        } => text::set_text_glow(draft, segment_id, glow_id, color, [*strength, *range]),
        Request::AddText {
            track_id,
            segment_id,
            material_id,
            start,
            duration,
            content,
            face,
        } => text::add_text(
            draft,
            track_id,
            segment_id,
            material_id,
            TimeRange::new(start.0, duration.0),
            content,
            face,
        ),
        Request::AddEffect {
            track_id,
            segment_id,
            material_id,
            start,
            duration,
            effect_id,
        } => effect::add_effect(
            draft,
            track_id,
            segment_id,
            material_id,
            TimeRange::new(start.0, duration.0),
            effect_id,
        ),
        Request::BindEffect { segment_id, clip } => {
            effect::bind_effect(draft, segment_id, clip.as_ref())
        }
        Request::SetBlend {
            segment_id,
            material_id,
            mode,
        } => blend::set_blend(draft, segment_id, material_id, mode),
        Request::AddAdjustment {
            track_id,
            segment_id,
            material_id,
            start,
            duration,
        } => adjust::add_adjustment(
            draft,
            track_id,
            segment_id,
            material_id,
            TimeRange::new(start.0, duration.0),
        ),
        Request::AddFilter {
            track_id,
            segment_id,
            material_id,
            start,
            duration,
            filter_id,
        } => adjust::add_filter(
            draft,
            track_id,
            segment_id,
            material_id,
            TimeRange::new(start.0, duration.0),
            filter_id,
        ),
        Request::SetHsl {
            segment_id,
            material_id,
            color,
            hue,
            saturation,
            lightness,
        } => grade::set_hsl(
            draft,
            segment_id,
            material_id,
            *color,
            [*hue, *saturation, *lightness],
        ),
        Request::SetCurve {
            segment_id,
            material_id,
            channel,
            points,
        } => grade::set_curve(draft, segment_id, material_id, channel, points),
        Request::SetWheel {
            segment_id,
            material_id,
            wheel,
            x,
            y,
            luma,
        } => grade::set_wheel(draft, segment_id, material_id, wheel, [*x, *y, *luma]),
        Request::SetAdjust {
            segment_id,
            material_id,
            key,
            value,
        } => adjust::set_adjust(draft, segment_id, material_id, key, *value),
        Request::SetEffectStrength {
            segment_id,
            material_id,
            value,
        } => effect::set_effect_strength(draft, segment_id, material_id, *value),
        Request::SetKeyframes {
            segment_id,
            keyframes,
        } => keyframe::set_keyframes(draft, segment_id, keyframes),
        Request::AddKeyframe { segment_id, offset } => {
            keyframe::add_keyframe(draft, segment_id, *offset)
        }
        Request::RemoveKeyframe { segment_id, offset } => {
            keyframe::remove_keyframe(draft, segment_id, *offset)
        }
        Request::SetKeyframeGraph {
            segment_id,
            offset,
            graph,
        } => keyframe::set_keyframe_graph(draft, segment_id, *offset, graph.as_deref()),
        Request::SetAnimation {
            segment_id,
            kind,
            animation,
            duration,
        } => animation::set_animation(draft, segment_id, *kind, animation.as_deref(), *duration),
        Request::KeyPose {
            segment_id,
            offset,
            x,
            y,
            scale,
            rotation,
        } => keyframe::key_pose(
            draft,
            segment_id,
            *offset,
            keyframe::Pose {
                x: *x,
                y: *y,
                scale: *scale,
                rotation: *rotation,
            },
        ),
        Request::Import { medium } => import::import(draft, medium.clone()),
        Request::InsertTrack { track_id, kind } => track::insert_track(draft, track_id, kind),
        Request::RemoveTrack { track_id } => track::remove_track(draft, track_id),
        Request::RenameTrack { track_id, name } => track::rename_track(draft, track_id, name),
        Request::AddTimeMark {
            segment_id,
            at,
            color,
            material_id,
        } => mark::add_time_mark(draft, segment_id.as_ref(), *at, color, material_id),
        Request::RemoveTimeMark { segment_id, start } => {
            mark::remove_time_mark(draft, segment_id.as_ref(), *start)
        }
        Request::AddBeat {
            segment_id,
            source,
            material_id,
        } => beat::add_beat(draft, segment_id, *source, material_id),
        Request::RemoveBeat { segment_id, source } => beat::remove_beat(draft, segment_id, *source),
        Request::SetMagnet { on } => magnet::set_magnet(draft, *on),
        Request::SetCanvasBackground { color } => canvas::set_background(draft, color),
        Request::SetCanvasSize {
            width,
            height,
            ratio,
        } => canvas::set_size(draft, *width, *height, ratio),
        Request::SetLetterbox {
            segment_id,
            color,
            canvas_id,
        } => canvas::set_letterbox(draft, segment_id, color, canvas_id),
        Request::SetLetterboxBlur {
            segment_id,
            blur,
            canvas_id,
        } => canvas::set_letterbox_blur(draft, segment_id, *blur, canvas_id),
        Request::SetFrameRate { fps, drop_frame } => rate::set_frame_rate(draft, *fps, *drop_frame),
        Request::SetDropFrame { on } => rate::set_drop_frame(draft, *on),
        Request::SetColorSpace { space } => color::set_color_space(draft, *space),
        Request::SetProjectName { name } => {
            name::set_project_name(draft, name);
            Ok(())
        }
    }
}

pub fn apply_all(
    draft: &mut Draft,
    requests: &[Request],
    linkage: &LinkageSettings,
) -> Result<(), EditError> {
    for request in requests {
        let before = draft.clone();
        apply_request(draft, request)?;
        linkage::after_request(draft, &before, linkage)?;
        magnet::pack(draft, linkage)?;
    }
    sync_duration(draft);
    Ok(())
}

/// Stored duration becomes the latest target end. Playback reads that field.
pub fn sync_duration(draft: &mut Draft) {
    let mut end = 0_i64;
    for track in &draft.tracks {
        for segment in &track.segments {
            end = end.max(segment.target.end().0);
        }
    }
    draft.duration = Micros(end);
}

pub fn ensure_unlocked(attribute: TrackAttribute) -> Result<(), EditError> {
    if attribute.contains(TrackAttribute::LOCK) {
        Err(EditError::LockedTrack)
    } else {
        Ok(())
    }
}
