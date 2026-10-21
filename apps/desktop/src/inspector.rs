//! The inspector. Nothing selected shows the project. A video clip edits its
//! placement. An audio clip lists its sound. `docs/inspector.md`.

use eframe::egui::{self, Key, RichText, ScrollArea, Sense, TextEdit, UiBuilder, WidgetInfo, vec2};

use cutlass_project::{
    AspectRatio, Channels, ColorSpace, Command, FrameRate, Project, SegmentId, TrackKind, Transform,
};

use crate::library::card_time;
use crate::theme;
use crate::timeline::{file_name, kind_name};
use crate::words::{self, Language, Phrase};

pub fn show(ui: &mut egui::Ui, project: &Project, selected: Option<SegmentId>) -> Vec<Command> {
    crate::tip::region(ui, words::of(ui, Phrase::RegionInspector));
    let language = words::active(ui);
    if let Some(id) = selected
        && let Some(kind) = track_kind(project, id)
    {
        return match kind {
            TrackKind::Video => video(ui, project, id, language),
            TrackKind::Audio => {
                audio(ui, project, id, language);
                Vec::new()
            }
            _ => {
                readout(ui, project, id, language);
                Vec::new()
            }
        };
    }
    let rows = facts(project, language);
    paint(ui, words::say(language, Phrase::Details), "project", |ui| {
        rows_of(ui, &rows);
    });
    Vec::new()
}

struct Panel {
    title: &'static str,
    rows: Vec<(&'static str, String)>,
}

struct Draft {
    x: Option<f64>,
    y: Option<f64>,
    width: Option<f64>,
    height: Option<f64>,
    rotation: Option<f64>,
    opacity: Option<f64>,
}

struct Field {
    salt: &'static str,
    value: f64,
    name: String,
}

#[derive(Clone, Copy)]
enum Slot {
    X,
    Y,
    Width,
    Height,
    Rotation,
    Opacity,
}

fn video(ui: &mut egui::Ui, project: &Project, id: SegmentId, language: Language) -> Vec<Command> {
    let Some(panel) = describe(project, id, language, false) else {
        return Vec::new();
    };
    let Some(transform) = transform_of(project, id) else {
        return Vec::new();
    };
    let mut draft = Draft {
        x: None,
        y: None,
        width: None,
        height: None,
        rotation: None,
        opacity: None,
    };
    paint(ui, panel.title, "video", |ui| {
        rows_of(ui, &panel.rows);
        section(ui, words::say(language, Phrase::Basic));
        draft = basic(ui, id, transform, language);
    });
    let Some(next) = committed(transform, &draft) else {
        return Vec::new();
    };
    if crate::timeline::segment_locked(ui, project, id) {
        crate::timeline::show_note(ui, words::of(ui, Phrase::TrackIsLocked));
        return Vec::new();
    }
    vec![Command::SetTransform {
        segment: id,
        transform: next,
    }]
}

fn audio(ui: &mut egui::Ui, project: &Project, id: SegmentId, language: Language) {
    let Some(panel) = describe(project, id, language, false) else {
        return;
    };
    paint(ui, panel.title, "audio", |ui| {
        rows_of(ui, &panel.rows);
    });
}

fn readout(ui: &mut egui::Ui, project: &Project, id: SegmentId, language: Language) {
    let Some(panel) = describe(project, id, language, true) else {
        return;
    };
    paint(ui, panel.title, "clip", |ui| {
        rows_of(ui, &panel.rows);
    });
}

fn paint(ui: &mut egui::Ui, title: &str, salt: &str, body: impl FnOnce(&mut egui::Ui)) {
    let full = ui.max_rect();
    ui.allocate_rect(full, Sense::hover());
    let padded = full.shrink2(vec2(16.0, 0.0));
    let mut details = ui.new_child(UiBuilder::new().max_rect(padded));
    details.set_clip_rect(padded);
    ScrollArea::vertical()
        .id_salt(("inspector", salt))
        .show(&mut details, |ui| {
            ui.set_min_width(ui.available_width());
            ui.add_space(14.0);
            ui.label(
                RichText::new(title)
                    .size(15.0)
                    .strong()
                    .color(theme::EXPORT),
            );
            ui.add_space(14.0);
            body(ui);
        });
}

fn rows_of(ui: &mut egui::Ui, rows: &[(&str, String)]) {
    for (label, value) in rows {
        fact(ui, label, value);
    }
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(10.0);
    ui.label(RichText::new(title).size(12.0).color(theme::MUTED));
    ui.add_space(8.0);
}

fn fact(ui: &mut egui::Ui, label: &str, value: &str) {
    let width = ui.available_width();
    ui.horizontal(|ui| {
        ui.set_max_width(width);
        ui.spacing_mut().item_spacing.x = 12.0;
        ui.add_sized(
            [108.0, 18.0],
            egui::Label::new(RichText::new(label).size(13.0).color(theme::MUTED)),
        );
        ui.vertical(|ui| {
            ui.set_width((width - 120.0).max(40.0));
            ui.add(egui::Label::new(RichText::new(value).size(13.0).color(theme::INK)).wrap());
        });
    });
    ui.add_space(4.0);
}

fn basic(ui: &mut egui::Ui, id: SegmentId, transform: Transform, language: Language) -> Draft {
    let (x, y) = transform.position();
    let (width, height) = transform.scale();
    let position = words::say(language, Phrase::Position);
    let size = words::say(language, Phrase::Size);
    let (x, y) = pair(
        ui,
        id,
        position,
        Field {
            salt: "px",
            value: x,
            name: format!("{position} x"),
        },
        Field {
            salt: "py",
            value: y,
            name: format!("{position} y"),
        },
    );
    let (width, height) = pair(
        ui,
        id,
        size,
        Field {
            salt: "sx",
            value: width,
            name: format!("{size} x"),
        },
        Field {
            salt: "sy",
            value: height,
            name: format!("{size} y"),
        },
    );
    Draft {
        x,
        y,
        width,
        height,
        rotation: line(
            ui,
            id,
            "rot",
            transform.rotation(),
            Phrase::Rotation,
            language,
        ),
        opacity: line(
            ui,
            id,
            "opacity",
            transform.opacity(),
            Phrase::Opacity,
            language,
        ),
    }
}

fn pair(
    ui: &mut egui::Ui,
    id: SegmentId,
    label: &str,
    left: Field,
    right: Field,
) -> (Option<f64>, Option<f64>) {
    let width = ui.available_width();
    let field = ((width - 128.0) / 2.0).max(36.0);
    let mut left_edit = None;
    let mut right_edit = None;
    ui.horizontal(|ui| {
        ui.set_max_width(width);
        ui.spacing_mut().item_spacing.x = 8.0;
        ui.add_sized(
            [108.0, 18.0],
            egui::Label::new(RichText::new(label).size(13.0).color(theme::MUTED)),
        );
        left_edit = edit(
            ui,
            ui.scope_id().with((id, left.salt)),
            left.value,
            &left.name,
            field,
        );
        right_edit = edit(
            ui,
            ui.scope_id().with((id, right.salt)),
            right.value,
            &right.name,
            field,
        );
    });
    ui.add_space(4.0);
    (left_edit, right_edit)
}

fn line(
    ui: &mut egui::Ui,
    id: SegmentId,
    salt: &str,
    value: f64,
    phrase: Phrase,
    language: Language,
) -> Option<f64> {
    let label = words::say(language, phrase);
    let width = ui.available_width();
    let field = (width - 120.0).max(48.0);
    let mut edited = None;
    ui.horizontal(|ui| {
        ui.set_max_width(width);
        ui.spacing_mut().item_spacing.x = 12.0;
        ui.add_sized(
            [108.0, 18.0],
            egui::Label::new(RichText::new(label).size(13.0).color(theme::MUTED)),
        );
        edited = edit(ui, ui.scope_id().with((id, salt)), value, label, field);
    });
    ui.add_space(4.0);
    edited
}

fn edit(ui: &mut egui::Ui, id: egui::Id, value: f64, name: &str, width: f32) -> Option<f64> {
    let focused = ui.memory(|memory| memory.has_focus(id));
    let mut text = if focused {
        ui.data(|data| data.get_temp::<String>(id))
            .unwrap_or_else(|| number(value))
    } else {
        number(value)
    };
    let enter = ui.input(|input| input.key_pressed(Key::Enter));
    let output = TextEdit::singleline(&mut text)
        .id(id)
        .frame(field_frame())
        .desired_width(width)
        .show(ui);
    output
        .response
        .widget_info(|| WidgetInfo::labeled(egui::accesskit::Role::TextInput, true, name));
    let submit = output.response.lost_focus() || (output.response.has_focus() && enter);
    if submit {
        ui.data_mut(|data| data.remove::<String>(id));
        parse_number(&text)
    } else if output.response.has_focus() {
        ui.data_mut(|data| data.insert_temp(id, text));
        None
    } else {
        None
    }
}

fn field_frame() -> egui::Frame {
    egui::Frame::NONE
        .fill(theme::BAR)
        .corner_radius(6)
        .inner_margin(egui::Margin::symmetric(8, 4))
}

fn committed(transform: Transform, draft: &Draft) -> Option<Transform> {
    let mut current = transform;
    let mut changed = false;
    for (slot, value) in [
        (Slot::X, draft.x),
        (Slot::Y, draft.y),
        (Slot::Width, draft.width),
        (Slot::Height, draft.height),
        (Slot::Rotation, draft.rotation),
        (Slot::Opacity, draft.opacity),
    ] {
        let Some(value) = value else {
            continue;
        };
        let Some(next) = edited_value(current, slot, value) else {
            continue;
        };
        current = next;
        changed = true;
    }
    changed.then_some(current)
}

fn edited_value(transform: Transform, slot: Slot, value: f64) -> Option<Transform> {
    if !value.is_finite() {
        return None;
    }
    let (mut x, mut y) = transform.position();
    let (mut width, mut height) = transform.scale();
    let mut rotation = transform.rotation();
    let mut opacity = transform.opacity();
    match slot {
        Slot::X => x = value,
        Slot::Y => y = value,
        Slot::Width => {
            if value <= 0.0 {
                return None;
            }
            width = value;
        }
        Slot::Height => {
            if value <= 0.0 {
                return None;
            }
            height = value;
        }
        Slot::Rotation => rotation = value,
        Slot::Opacity => {
            if !(0.0..=1.0).contains(&value) {
                return None;
            }
            opacity = value;
        }
    }
    let (flip_horizontal, flip_vertical) = transform.flips();
    let next = Transform::new(
        (x, y),
        (width, height),
        rotation,
        opacity,
        flip_horizontal,
        flip_vertical,
    )
    .ok()?;
    (next != transform).then_some(next)
}

fn parse_number(text: &str) -> Option<f64> {
    let value: f64 = text.trim().parse().ok()?;
    value.is_finite().then_some(value)
}

fn track_kind(project: &Project, id: SegmentId) -> Option<TrackKind> {
    project
        .timeline()
        .tracks()
        .iter()
        .find(|track| track.segments().iter().any(|segment| segment.id() == id))
        .map(|track| track.kind())
}

fn transform_of(project: &Project, id: SegmentId) -> Option<Transform> {
    project.timeline().tracks().iter().find_map(|track| {
        track
            .segments()
            .iter()
            .find(|segment| segment.id() == id)
            .map(|segment| segment.transform())
    })
}

fn describe(
    project: &Project,
    id: SegmentId,
    language: Language,
    placement: bool,
) -> Option<Panel> {
    let track = project
        .timeline()
        .tracks()
        .iter()
        .find(|track| track.segments().iter().any(|segment| segment.id() == id))?;
    let segment = track.segments().iter().find(|segment| segment.id() == id)?;
    let asset = project.asset(segment.asset_id())?;
    let range = segment.timeline();
    let speed = segment.speed();
    let say = |phrase| words::say(language, phrase);
    let mut rows = vec![
        (say(Phrase::Name), file_name(asset.source()).to_string()),
        (
            say(Phrase::Start),
            card_time(range.start().value, range.start().timescale.get()),
        ),
        (
            say(Phrase::Duration),
            card_time(range.duration().value, range.duration().timescale.get()),
        ),
        (
            say(Phrase::Speed),
            format!("{}/{}", speed.numerator(), speed.denominator().get()),
        ),
    ];
    if segment.channels() != Channels::Picture {
        rows.push((say(Phrase::Volume), number(segment.volume())));
    }
    if placement && segment.channels() != Channels::Sound {
        let (x, y) = segment.transform().position();
        let (width, height) = segment.transform().scale();
        rows.push((
            say(Phrase::Position),
            format!("{}, {}", number(x), number(y)),
        ));
        rows.push((
            say(Phrase::Size),
            format!("{} × {}", number(width), number(height)),
        ));
    }
    Some(Panel {
        title: kind_name(track.kind(), language),
        rows,
    })
}

fn number(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 1e6 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

fn facts(project: &Project, language: Language) -> Vec<(&'static str, String)> {
    let settings = project.settings();
    let say = |phrase| words::say(language, phrase);
    let mut rows = vec![
        (say(Phrase::Name), project.name().to_string()),
        (say(Phrase::Saved), say(Phrase::NotSaved).to_string()),
        (
            say(Phrase::ColorSpace),
            color_space(settings.color_space()).to_string(),
        ),
        (
            say(Phrase::Ratio),
            ratio(settings.ratio(), language).to_string(),
        ),
        (
            say(Phrase::Resolution),
            format!("{}×{}", settings.width(), settings.height()),
        ),
        (
            say(Phrase::FrameRate),
            frame_rate(settings.rate().rate()).to_string(),
        ),
    ];
    if settings.rate().rate().allows_drop_frame() {
        let value = if settings.rate().drop_frame() {
            say(Phrase::On)
        } else {
            say(Phrase::Off)
        };
        rows.push((say(Phrase::DropFrame), value.to_string()));
    }
    rows
}

fn color_space(space: ColorSpace) -> &'static str {
    match space {
        ColorSpace::Rec709 => "Rec. 709",
        ColorSpace::Rec2020Hlg => "Rec. 2020 HLG",
        ColorSpace::Rec2100Pq => "Rec. 2100 PQ",
    }
}

fn ratio(ratio: AspectRatio, language: Language) -> &'static str {
    match ratio {
        AspectRatio::Original => words::say(language, Phrase::Original),
        AspectRatio::R16x9 => "16:9",
        AspectRatio::R9x16 => "9:16",
        AspectRatio::Square => "1:1",
        AspectRatio::R4x5 => "4:5",
        AspectRatio::R4x3 => "4:3",
    }
}

fn frame_rate(rate: FrameRate) -> &'static str {
    match rate {
        FrameRate::Fps23976 => "23.976 fps",
        FrameRate::Fps24 => "24 fps",
        FrameRate::Fps25 => "25 fps",
        FrameRate::Fps2997 => "29.97 fps",
        FrameRate::Fps30 => "30 fps",
        FrameRate::Fps50 => "50 fps",
        FrameRate::Fps5994 => "59.94 fps",
        FrameRate::Fps60 => "60 fps",
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cutlass_core::Timestamp;
    use cutlass_project::{
        Asset, Channels, Command, FrameRate, History, Outcome, Project, ProjectId, Segment,
        SequenceRate, Speed, TimeRange, Transform,
    };

    use super::{Slot, describe, edited_value, facts, parse_number};
    use crate::words::Language;

    fn project() -> Project {
        Project::new(ProjectId::from_bytes([1; 16]), "Untitled")
    }

    fn labels(project: &Project) -> Vec<&str> {
        facts(project, Language::English)
            .into_iter()
            .map(|(label, _)| label)
            .collect()
    }

    fn video_cut() -> (Project, cutlass_project::SegmentId) {
        let mut history = History::new();
        let mut cut = project();
        let video = Asset::video(
            "file:///cuts/v.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
            320,
            180,
            true,
        )
        .expect("video");
        let asset = match history
            .apply(&mut cut, &[Command::AddAsset(video)])
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let segment = {
            let stored = cut.asset(asset).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            Segment::media(stored, source, 0, Channels::Both, Speed::ONE).expect("segment")
        };
        let main = cut.timeline().main_track().id();
        let id = match history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: Some(main),
                    segment,
                }],
            )
            .expect("place")
            .as_slice()
        {
            [Outcome::Segment(id)] => *id,
            other => panic!("expected a segment, got {other:?}"),
        };
        (cut, id)
    }

    #[test]
    fn an_empty_cut_shows_the_starting_project() {
        let rows = facts(&project(), Language::English);
        assert_eq!(
            rows,
            vec![
                ("Name", "Untitled".to_string()),
                ("Saved", "Not saved".to_string()),
                ("Color space", "Rec. 709".to_string()),
                ("Ratio", "Original".to_string()),
                ("Resolution", "1920×1080".to_string()),
                ("Frame rate", "30 fps".to_string()),
            ]
        );
    }

    #[test]
    fn drop_frame_appears_only_for_the_two_rates_that_allow_it() {
        let mut history = History::new();
        let mut cut = project();
        history
            .apply(&mut cut, &[Command::SetRate(FrameRate::Fps2997.sequence())])
            .expect("rate");
        let rows = facts(&cut, Language::English);
        assert_eq!(rows[5].1, "29.97 fps");
        assert_eq!(rows[6], ("Drop frame", "On".to_string()));

        history
            .apply(
                &mut cut,
                &[Command::SetRate(
                    SequenceRate::new(FrameRate::Fps2997, false).expect("drop can be off"),
                )],
            )
            .expect("drop off");
        assert_eq!(
            facts(&cut, Language::English)[6],
            ("Drop frame", "Off".to_string())
        );

        history
            .apply(&mut cut, &[Command::SetRate(FrameRate::Fps24.sequence())])
            .expect("24");
        assert!(!labels(&cut).contains(&"Drop frame"));
    }

    #[test]
    fn a_selected_video_lists_its_file_time_speed_and_volume() {
        let (cut, id) = video_cut();
        let panel = describe(&cut, id, Language::English, false).expect("panel");
        assert_eq!(panel.title, "Video");
        assert_eq!(
            panel.rows,
            vec![
                ("Name", "v.mp4".to_string()),
                ("Start", "0:00.0".to_string()),
                ("Duration", "0:01.0".to_string()),
                ("Speed", "1/1".to_string()),
                ("Volume", "1".to_string()),
            ]
        );
    }

    #[test]
    fn an_audio_clip_lists_the_sound_and_not_its_placement() {
        let mut history = History::new();
        let mut cut = project();
        let audio = Asset::audio(
            "file:///cuts/a.wav",
            Timestamp::new(30_000, NonZeroU32::new(30_000).expect("scale")),
        )
        .expect("audio");
        let asset = match history
            .apply(&mut cut, &[Command::AddAsset(audio)])
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let segment = {
            let stored = cut.asset(asset).expect("asset");
            let duration = stored.duration().expect("duration");
            let source = TimeRange::new(
                Timestamp::new(0, duration.timescale),
                Timestamp::new(duration.value, duration.timescale),
            )
            .expect("source");
            Segment::media(stored, source, 0, Channels::Sound, Speed::ONE).expect("segment")
        };
        let id = match history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: None,
                    segment,
                }],
            )
            .expect("place")
            .as_slice()
        {
            [Outcome::Opened { segment, .. }] => *segment,
            other => panic!("expected an opened segment, got {other:?}"),
        };
        let panel = describe(&cut, id, Language::English, false).expect("panel");
        assert_eq!(panel.title, "Audio");
        assert_eq!(
            panel
                .rows
                .iter()
                .map(|(label, _)| *label)
                .collect::<Vec<_>>(),
            vec!["Name", "Start", "Duration", "Speed", "Volume"]
        );
    }

    #[test]
    fn a_position_edit_changes_only_that_axis() {
        let next = edited_value(Transform::IDENTITY, Slot::X, 0.25).expect("edit");
        assert_eq!(next.position(), (0.25, 0.0));
        assert_eq!(next.scale(), (1.0, 1.0));
        assert_eq!(next.rotation(), 0.0);
        assert_eq!(next.opacity(), 1.0);
        assert_eq!(next.flips(), (false, false));
    }

    #[test]
    fn a_bad_scale_or_opacity_is_refused() {
        assert!(edited_value(Transform::IDENTITY, Slot::Width, 0.0).is_none());
        assert!(edited_value(Transform::IDENTITY, Slot::Height, -1.0).is_none());
        assert!(edited_value(Transform::IDENTITY, Slot::Opacity, 1.1).is_none());
        assert!(edited_value(Transform::IDENTITY, Slot::Opacity, -0.01).is_none());
        assert!(parse_number("no").is_none());
        assert!(parse_number("inf").is_none());
        assert_eq!(parse_number(" 0.5 "), Some(0.5));
    }

    #[test]
    fn an_unchanged_number_writes_nothing_and_flips_stay() {
        assert!(edited_value(Transform::IDENTITY, Slot::Opacity, 1.0).is_none());
        let start = Transform::new((0.0, 0.0), (1.0, 1.0), 0.0, 1.0, true, false).expect("start");
        let next = edited_value(start, Slot::Rotation, 15.0).expect("turn");
        assert_eq!(next.rotation(), 15.0);
        assert_eq!(next.flips(), (true, false));
    }

    #[test]
    fn a_position_field_writes_a_transform() {
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::words::install(&ctx, Language::English);
        let (cut, id) = video_cut();
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(320.0, 800.0),
        );
        let mut time = 0.0;
        let mut pass = |events: Vec<eframe::egui::Event>| {
            time += 0.05;
            let mut commands = Vec::new();
            let mut output = ctx.run_ui(
                eframe::egui::RawInput {
                    screen_rect: Some(screen),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ui| {
                    commands.clone_from(&super::show(ui, &cut, Some(id)));
                },
            );
            let tree = output.platform_output.accesskit_update.take();
            output.textures_delta.clear();
            (tree.expect("the tree is built"), commands)
        };
        let (tree, _) = pass(vec![]);
        let field = center(&tree, "Position x");
        pass(vec![
            eframe::egui::Event::PointerMoved(field),
            eframe::egui::Event::PointerButton {
                pos: field,
                button: eframe::egui::PointerButton::Primary,
                pressed: true,
                modifiers: eframe::egui::Modifiers::default(),
            },
        ]);
        pass(vec![eframe::egui::Event::PointerButton {
            pos: field,
            button: eframe::egui::PointerButton::Primary,
            pressed: false,
            modifiers: eframe::egui::Modifiers::default(),
        }]);
        let (_, commands) = pass(vec![
            eframe::egui::Event::Key {
                key: eframe::egui::Key::Backspace,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: eframe::egui::Modifiers::default(),
            },
            eframe::egui::Event::Text("0.25".to_string()),
            eframe::egui::Event::Key {
                key: eframe::egui::Key::Enter,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: eframe::egui::Modifiers::default(),
            },
        ]);
        let Command::SetTransform { segment, transform } =
            commands.first().expect("the field writes a transform")
        else {
            panic!("expected a transform, got {commands:?}");
        };
        assert_eq!(*segment, id);
        assert_eq!(transform.position(), (0.25, 0.0));
        assert_eq!(transform.scale(), (1.0, 1.0));
    }

    fn center(tree: &eframe::egui::accesskit::TreeUpdate, label: &str) -> eframe::egui::Pos2 {
        let (_, node) = tree
            .nodes
            .iter()
            .find(|(_, node)| node.label() == Some(label) || node.value() == Some(label))
            .unwrap_or_else(|| panic!("missing {label}"));
        let rect = node.bounds().expect("bounds");
        eframe::egui::pos2(
            ((rect.x0 + rect.x1) / 2.0) as f32,
            ((rect.y0 + rect.y1) / 2.0) as f32,
        )
    }
}
