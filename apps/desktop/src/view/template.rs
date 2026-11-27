//! Project templates in the window: the Template sections in the
//! inspector, Save as template, the Templates tab at home, the strip of
//! slots that sits over a template clip, and the template clip's panel.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, RichText, Sense, Ui, Vec2};

use cutlass_model::{SlotTakes, TemplateMark};
use cutlass_store::ListedTemplate;

use crate::shell::{
    Inspection, SILENT_DB, Shell, SlotView, TemplateAsk, TemplateClip, gain_of_db, volume_db,
};
use crate::widgets::{self, SheetEnd};
use crate::{icons, tokens, ui};

use super::inspector::{InspectorState, body, label_row, note, section, send, slider};

/// Which kind of clip the inspector shows a Template section for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MarkKind {
    /// A photo or video: it can be a slot.
    Picture,
    /// A title: its words can be editable.
    Text,
    /// A sound: it can be the music.
    Sound,
}

/// The Template section: whether the clip is part of the template, and
/// for a slot, what it takes.
pub(super) fn template_section(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    kind: MarkKind,
) {
    // A compound clip plays as one piece: it is not a slot.
    if kind == MarkKind::Picture && shell.compound_selected() {
        return;
    }
    let mark = shell.template_mark();
    let (label, on_mark) = match kind {
        MarkKind::Picture => ("Template slot", TemplateMark::slot(SlotTakes::Either)),
        MarkKind::Text => ("Editable in template", TemplateMark::Text),
        MarkKind::Sound => ("Template music", TemplateMark::Music),
    };
    let on = mark.is_some();
    let header = section(ui, state, "template", label, Some((on, true)), false);
    if header.toggled {
        let next = (!on).then_some(on_mark);
        send(shell, |shell| shell.set_template_mark(next));
    }
    if !header.open {
        return;
    }
    body(ui, |ui| match (kind, mark) {
        (MarkKind::Picture, Some(TemplateMark::Slot { takes, .. })) => {
            let mut picked = None;
            label_row(ui, "Takes", |ui| {
                egui::ComboBox::from_id_salt("template-takes")
                    .selected_text(takes_label(takes))
                    .show_ui(ui, |ui| {
                        for choice in [SlotTakes::Either, SlotTakes::Video, SlotTakes::Photo] {
                            if ui
                                .add(ui::Choice::new(choice == takes, takes_label(choice)).wide())
                                .clicked()
                            {
                                picked = Some(choice);
                            }
                        }
                    });
            });
            if let Some(choice) = picked.filter(|choice| *choice != takes) {
                send(shell, move |shell| {
                    shell.set_template_mark(Some(TemplateMark::slot(choice)))
                });
            }
            note(ui, "Someone using the template puts their own media here.");
        }
        (MarkKind::Picture, _) => {
            note(
                ui,
                "Mark this clip so a template can take other media here.",
            );
        }
        (MarkKind::Text, _) => {
            note(ui, "Someone using the template can change these words.");
        }
        (MarkKind::Sound, _) => {
            note(
                ui,
                "Someone using the template can swap this sound for theirs.",
            );
        }
    });
}

const fn takes_label(takes: SlotTakes) -> &'static str {
    match takes {
        SlotTakes::Either => "Photo or video",
        SlotTakes::Video => "Video only",
        SlotTakes::Photo => "Photo only",
    }
}

/// The home screen's tabs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum HomeTab {
    #[default]
    Projects,
    Templates,
}

/// What the window keeps for templates between frames.
#[derive(Default)]
pub(crate) struct TemplateView {
    pub(super) tab: HomeTab,
    /// The Save as template sheet, while it is up.
    pub(super) save: Option<SaveSheet>,
    /// The file picker that is up: which template clip, and which slot of
    /// it, or its music, the picked file is for.
    pub(super) pending: Option<TemplateAsk>,
    listed: Vec<ListedTemplate>,
    listed_at: Option<Instant>,
}

pub(super) struct SaveSheet {
    name: String,
    category: String,
}

impl TemplateView {
    /// The user's templates under `projects`' sibling folder, read again
    /// at most once a second.
    pub(super) fn listed(&mut self, projects: &Path) -> &[ListedTemplate] {
        if self
            .listed_at
            .is_none_or(|at| at.elapsed() >= TEMPLATES_CHECK)
        {
            self.listed = Shell::list_templates(&templates_root(projects));
            self.listed_at = Some(Instant::now());
        }
        &self.listed
    }

    pub(super) fn open_save(&mut self, shell: &Shell) {
        self.save = Some(SaveSheet {
            name: shell
                .draft()
                .map(|draft| draft.name.clone())
                .unwrap_or_default(),
            category: String::new(),
        });
    }
}

/// Where the user's templates live: beside the projects folder.
pub(super) fn templates_root(projects: &Path) -> PathBuf {
    projects.with_file_name("Templates")
}

/// How often the Templates tab looks at the folder again.
const TEMPLATES_CHECK: Duration = Duration::from_secs(1);

/// The tabs over the home screen's body.
pub(super) fn home_tabs(ui: &mut Ui, view: &mut TemplateView) {
    ui::Segmented::show(ui, |ui| {
        for (tab, name) in [
            (HomeTab::Projects, "Projects"),
            (HomeTab::Templates, "Templates"),
        ] {
            if ui.add(ui::Choice::new(view.tab == tab, name)).clicked() {
                view.tab = tab;
                view.listed_at = None;
            }
        }
    });
    ui.add_space(20.0);
}

/// The Templates tab: the user's templates as cards. A click starts a new
/// project that holds the template as one clip, and opens it.
pub(super) fn templates_page(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut TemplateView,
    projects: &Path,
) -> bool {
    view.listed(projects);
    if view.listed.is_empty() {
        ui.label(
            RichText::new(
                "No templates yet. Open a project, mark clips as template slots in the \
                 inspector, then choose File › Save as template.",
            )
            .size(tokens::TYPE_SIZE)
            .color(tokens::MUTED),
        );
        return false;
    }
    let mut chosen = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(16.0, 16.0);
        for template in &view.listed {
            let meta = template_meta(template);
            if super::card(ui, shell, &template.path, &template.name, &meta) {
                chosen = Some(template.path.clone());
            }
        }
    });
    let Some(template) = chosen else {
        return false;
    };
    match shell.use_template(&template, projects) {
        Ok(()) => true,
        Err(error) => {
            shell.notice_error(&error);
            false
        }
    }
}

/// "9:16 · 15.0 s · 3 clips".
pub(super) fn template_meta(template: &ListedTemplate) -> String {
    let shape = shape_label(template.width, template.height);
    let clips = if template.slots == 1 {
        "1 clip".to_owned()
    } else {
        format!("{} clips", template.slots)
    };
    #[allow(clippy::cast_precision_loss)]
    let seconds = template.duration.0 as f64 / 1_000_000.0;
    format!("{shape} · {seconds:.1} s · {clips}")
}

fn shape_label(width: i32, height: i32) -> String {
    if width <= 0 || height <= 0 {
        return "?".to_owned();
    }
    let gcd = |mut a: i32, mut b: i32| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let common = gcd(width, height);
    format!("{}:{}", width / common, height / common)
}

/// The Save as template sheet.
pub(super) fn save_sheet(shell: &mut Shell, ui: &mut Ui, view: &mut TemplateView, projects: &Path) {
    let Some(sheet) = view.save.as_mut() else {
        return;
    };
    if shell.draft().is_none() {
        view.save = None;
        return;
    }
    let slots = shell.template_slots().len();
    let ((), end) = widgets::sheet(ui, "Save as template", 480.0, "Save", |ui| {
        widgets::form_section(ui, "Template");
        widgets::form_row(ui, "Name", |ui| {
            widgets::text_box(ui, &mut sheet.name);
        });
        widgets::form_row(ui, "Category", |ui| {
            widgets::text_box(ui, &mut sheet.category);
        });
        ui.add_space(8.0);
        let line = match slots {
            0 => "Mark at least one clip as a template slot in the inspector first.".to_owned(),
            1 => "1 clip is a slot. The rest stays as you made it.".to_owned(),
            n => format!("{n} clips are slots. The rest stays as you made it."),
        };
        ui.label(
            RichText::new(line)
                .size(tokens::TYPE_SIZE)
                .color(tokens::MUTED),
        );
    });
    match end {
        SheetEnd::Open => {}
        SheetEnd::Cancel => view.save = None,
        SheetEnd::Confirm => {
            let (name, category) = (sheet.name.clone(), sheet.category.clone());
            match shell.save_as_template(&templates_root(projects), &name, &category) {
                Ok(_) => {
                    shell.set_notice(format!("Saved the template {}", name.trim()));
                    view.save = None;
                    view.listed_at = None;
                }
                Err(error) => shell.notice_error(&error),
            }
        }
    }
}

/// The picked file for a template clip: fill the slot, or swap the music.
pub(super) fn picked(shell: &mut Shell, view: &mut TemplateView, path: &Path) {
    let Some(ask) = view.pending.take() else {
        return;
    };
    let done = match &ask.slot {
        Some(slot) => shell.fill_template_slot(&ask.clip, slot, path),
        None => shell.set_template_music(&ask.clip, path),
    };
    if let Err(error) = done {
        shell.notice_error(&error);
    }
}

/// A slot's tile in the strip.
const TILE: f32 = 52.0;
const STRIP_PAD: f32 = 6.0;
const STRIP_GAP: f32 = 4.0;

/// "3.0s".
fn length_label(micros: i64) -> String {
    #[allow(clippy::cast_precision_loss)]
    let seconds = micros as f64 / 1_000_000.0;
    format!("{seconds:.1}s")
}

/// The strip of a template clip's slots, sitting over the clip like a
/// popup: one tile per slot with its picture and length, and a replace
/// mark on the ones still holding their sample. A click on a tile asks
/// for a file to put there; Hide puts the strip away.
pub(super) fn slot_strip(shell: &mut Shell, ui: &Ui, template: &TemplateClip, clip: Rect) {
    if template.slots.is_empty() {
        return;
    }
    let screen = ui.ctx().content_rect();
    // Over the clip, growing upward; under it when there is no room above.
    let above = clip.top() - (TILE + STRIP_PAD * 2.0) - 6.0 >= screen.top();
    let (pivot, y) = if above {
        (Align2::LEFT_BOTTOM, clip.top() - 6.0)
    } else {
        (Align2::LEFT_TOP, clip.bottom() + 6.0)
    };
    let left = clip.left().max(screen.left() + 8.0);
    let room = (screen.right() - left - 16.0).max(TILE * 3.0);
    let mut hide = false;
    let mut ask = None;
    egui::Area::new(egui::Id::unique((
        "template strip",
        template.segment_id.as_str(),
    )))
    .order(egui::Order::Foreground)
    .pivot(pivot)
    .fixed_pos(Pos2::new(left, y))
    .show(ui.ctx(), |ui| {
        egui::Frame::new()
            .fill(tokens::MENU)
            .corner_radius(CornerRadius::same(8))
            .inner_margin(egui::Margin::same(STRIP_PAD as i8))
            .shadow(egui::Shadow {
                offset: [0, 4],
                blur: 16,
                spread: 0,
                color: Color32::from_black_alpha(110),
            })
            .show(ui, |ui| {
                ui.set_max_width(room);
                // Many slots wrap onto more rows.
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = Vec2::splat(STRIP_GAP);
                    if ui
                        .add(
                            ui::Button::ghost("Hide")
                                .text_size(11.0)
                                .min_size(Vec2::new(0.0, TILE)),
                        )
                        .clicked()
                    {
                        hide = true;
                    }
                    for (index, slot) in template.slots.iter().enumerate() {
                        if slot_tile(shell, ui, index, slot) {
                            ask = Some(slot.slot.segment_id.clone());
                        }
                    }
                });
            });
    });
    if hide {
        shell.hide_template_strip(&template.segment_id);
    }
    if let Some(slot) = ask {
        shell.ask_template_file(TemplateAsk {
            clip: template.segment_id.clone(),
            slot: Some(slot),
        });
    }
}

/// One slot's tile; true when it was clicked.
fn slot_tile(shell: &mut Shell, ui: &mut Ui, index: usize, slot: &SlotView) -> bool {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(TILE), Sense::click());
    let painter = ui.painter().with_clip_rect(rect);
    let round = CornerRadius::same(5);
    painter.rect_filled(rect, round, tokens::FIELD);
    let poster = slot
        .file
        .as_deref()
        .and_then(|file| shell.file_poster(file))
        .and_then(|(key, thumb)| Some((key, thumb?)));
    if let Some((key, thumb)) = poster {
        let texture = super::with_film_textures(ui.ctx(), |textures| {
            super::film_texture(ui.ctx(), textures, key, &thumb)
        });
        // The middle of the picture, covering the square.
        #[allow(clippy::cast_precision_loss)]
        let (width, height) = (thumb.width.max(1) as f32, thumb.height.max(1) as f32);
        let (u, v) = if width > height {
            ((1.0 - height / width) / 2.0, 0.0)
        } else {
            (0.0, (1.0 - width / height) / 2.0)
        };
        let uv = Rect::from_min_max(Pos2::new(u, v), Pos2::new(1.0 - u, 1.0 - v));
        painter.add(
            egui::epaint::RectShape::filled(rect, round, Color32::WHITE).with_texture(texture, uv),
        );
    }
    if !slot.slot.filled {
        // Still the sample: dimmed, with the replace mark over it.
        painter.rect_filled(rect, round, Color32::from_black_alpha(120));
        let mark = Rect::from_center_size(rect.center(), Vec2::splat(18.0));
        let _ = icons::paint(&painter, mark, "replace", Color32::WHITE);
    }
    let plate = Rect::from_min_max(Pos2::new(rect.left(), rect.bottom() - 15.0), rect.max);
    painter.rect_filled(plate, CornerRadius::ZERO, Color32::from_black_alpha(140));
    painter.text(
        plate.left_center() + Vec2::new(4.0, 0.0),
        Align2::LEFT_CENTER,
        length_label(slot.slot.length.0),
        FontId::proportional(10.0),
        Color32::WHITE,
    );
    if response.hovered() {
        painter.rect_stroke(
            rect,
            round,
            egui::Stroke::new(1.5, tokens::SELECTED),
            egui::StrokeKind::Inside,
        );
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let needs = match slot.slot.takes {
        SlotTakes::Video => format!("a video of {} or more", length_label(slot.slot.needs.0)),
        SlotTakes::Photo => "a photo".to_owned(),
        SlotTakes::Either => format!(
            "a photo, or a video of {} or more",
            length_label(slot.slot.needs.0)
        ),
    };
    response
        .on_hover_text(format!("Clip {}: replace with {needs}", index + 1))
        .clicked()
}

/// The template clip's panel: its clips to replace, how loud it and its
/// music play, and a box for each text the author left editable.
pub(super) fn template_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
    template: &TemplateClip,
) {
    if section(ui, state, "template-clips", "Template", None, false).open {
        body(ui, |ui| {
            label_row(ui, "Name", |ui| {
                ui.label(
                    RichText::new(&template.name)
                        .size(tokens::TYPE_SIZE)
                        .color(tokens::TEXT),
                );
            });
            let left = match (template.slots.len(), template.unfilled) {
                (0, _) => "This template has no clips to replace.".to_owned(),
                (_, 0) => "Every clip is replaced.".to_owned(),
                (_, 1) => "1 clip to be replaced.".to_owned(),
                (_, left) => format!("{left} clips to be replaced."),
            };
            note(ui, &left);
            if !template.slots.is_empty()
                && !shell.template_strip_shown(&template.segment_id)
                && ui
                    .add(ui::Button::new("Show clips").text_size(12.0))
                    .clicked()
            {
                shell.show_template_strip();
            }
        });
    }
    if section(ui, state, "template-volume", "Volume", None, false).open {
        body(ui, |ui| {
            let shown = |value: f64| format!("{value:.1}dB");
            let db = (volume_db(inspection.volume) * 10.0).round() / 10.0;
            if let Some(value) =
                slider(ui, state, "volume", "Template", db, SILENT_DB..=12.0, shown)
            {
                send(shell, |shell| shell.set_volume_db(value));
            }
            if let Some((_, volume)) = template.music {
                let db = (volume_db(volume) * 10.0).round() / 10.0;
                let moved = slider(
                    ui,
                    state,
                    "template-music",
                    "Music",
                    db,
                    SILENT_DB..=12.0,
                    shown,
                );
                // The music is inside the template, so each change stores
                // the inside again: sent once, when the pointer lets go.
                if let Some(value) = moved {
                    state
                        .fields
                        .insert(MUSIC_HELD.to_owned(), value.to_string());
                }
                let held = ui.input(|input| input.pointer.any_down());
                if !held
                    && let Some(value) = state
                        .fields
                        .remove(MUSIC_HELD)
                        .and_then(|value| value.parse::<f64>().ok())
                {
                    let clip = template.segment_id.clone();
                    send(shell, move |shell| {
                        shell.set_template_music_volume(&clip, gain_of_db(value))
                    });
                }
                if ui
                    .add(ui::Button::new("Replace music…").text_size(12.0))
                    .clicked()
                {
                    shell.ask_template_file(TemplateAsk {
                        clip: template.segment_id.clone(),
                        slot: None,
                    });
                }
            }
        });
    }
    if template.texts.is_empty() {
        return;
    }
    if section(ui, state, "template-text", "Text", None, false).open {
        body(ui, |ui| {
            for (index, text) in template.texts.iter().enumerate() {
                ui.label(
                    RichText::new(format!("Text {}", index + 1))
                        .size(tokens::TYPE_SIZE - 1.0)
                        .color(tokens::MUTED),
                );
                let key = format!("template-text {}", text.segment_id.as_str());
                let typed = state.fields.entry(key).or_default();
                let response = ui::Field::multiline(typed, 2)
                    .id_salt(("template-text", text.segment_id.as_str()))
                    .name(format!("Text {}", index + 1))
                    .show(ui)
                    .response;
                if super::commits(ui, &response) && *typed != text.words {
                    let (clip, id, words) = (
                        template.segment_id.clone(),
                        text.segment_id.clone(),
                        typed.clone(),
                    );
                    send(shell, move |shell| {
                        shell.set_template_text(&clip, &id, &words)
                    });
                } else if !response.has_focus() {
                    text.words.clone_into(typed);
                }
                ui.add_space(6.0);
            }
        });
    }
}

/// Where the music slider's value waits while the pointer holds it.
const MUSIC_HELD: &str = "template-music held";

#[cfg(test)]
mod tests {
    use std::path::Path;

    use cutlass_model::{
        Draft, MaterialVideo, MetaType, Micros, Segment, SlotTakes, TemplateMark, TimeRange,
    };
    use cutlass_settings::Settings;
    use egui::{Pos2, Rect, Vec2};

    use super::{HomeTab, shape_label, templates_root};
    use crate::dialogs::Picker;
    use crate::shell::Shell;
    use crate::shell::TemplateAsk;
    use crate::view::{ViewState, run_menu, show};

    /// A project with one photo slot under `root/Projects`, open in a shell.
    fn slotted(root: &Path) -> Shell {
        let project = root.join("Projects").join("Trip");
        let mut draft = Draft::empty("trip");
        draft.name = "Trip".to_owned();
        draft.duration = Micros(1_000_000);
        let mut photo = MaterialVideo::new("p");
        photo.kind = MetaType::Photo.as_str().to_owned();
        draft.materials.videos.push(photo);
        let mut segment = Segment::new("a", "p", TimeRange::new(0, 1_000_000));
        segment.template_mark = Some(TemplateMark::slot(SlotTakes::Either));
        draft.tracks[0].segments.push(segment);
        cutlass_store::Folder::new(draft).save(&project).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&project).unwrap();
        shell
    }

    struct Temp {
        path: std::path::PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "cutlass-desktop-template-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn frame(ctx: &egui::Context, shell: &mut Shell, view: &mut ViewState) {
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                ..Default::default()
            },
            |ui| show(shell, ui, view, None),
        );
        output.textures_delta.clear();
    }

    #[test]
    fn save_as_template_opens_its_sheet_named_after_the_project() {
        let temp = Temp::new();
        let mut shell = slotted(&temp.path);
        let mut view = ViewState::new();
        view.projects_root = Some(temp.path.join("Projects"));
        run_menu(&mut shell, &mut view, None, crate::menubar::SAVE_TEMPLATE);
        assert_eq!(
            view.template.save.as_ref().map(|sheet| sheet.name.as_str()),
            Some("Trip")
        );
        let ctx = egui::Context::default();
        frame(&ctx, &mut shell, &mut view);
        assert!(
            view.template.save.is_some(),
            "the sheet stays up until answered"
        );
    }

    #[test]
    fn a_saved_template_shows_on_the_tab_and_a_picked_file_fills_its_clip() {
        let temp = Temp::new();
        let mut shell = slotted(&temp.path);
        let projects = temp.path.join("Projects");
        shell
            .save_as_template(&templates_root(&projects), "Beach", "")
            .unwrap();
        shell.close_project().unwrap();
        let mut view = ViewState::new();
        view.projects_root = Some(projects);
        view.template.tab = HomeTab::Templates;
        let ctx = egui::Context::default();
        frame(&ctx, &mut shell, &mut view);
        assert_eq!(view.template.listed.len(), 1);
        assert_eq!(view.template.listed[0].name, "Beach");

        // Using it opens a project that holds the template as one clip,
        // picked, with its strip and its panel drawn.
        let template = view.template.listed[0].path.clone();
        shell
            .use_template(&template, view.projects_root.as_ref().unwrap())
            .unwrap();
        let clip = shell.selected().cloned().unwrap();
        frame(&ctx, &mut shell, &mut view);
        frame(&ctx, &mut shell, &mut view);
        assert!(shell.template_strip_shown(&clip));
        let shown = shell.template_clip(&clip).unwrap();
        assert_eq!(shown.unfilled, 1);

        // A tile asks for a file; the picked one goes in that slot.
        view.template.pending = Some(TemplateAsk {
            clip: clip.clone(),
            slot: Some(shown.slots[0].slot.segment_id.clone()),
        });
        let picture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        view.dialogs
            .finish_for_test(Picker::TemplateMedia(Some(0)), Some(picture));
        frame(&ctx, &mut shell, &mut view);
        frame(&ctx, &mut shell, &mut view);
        assert!(view.template.pending.is_none());
        let shown = shell.template_clip(&clip).unwrap();
        assert_eq!(shown.unfilled, 0);
        assert_eq!(
            shell.template_badge(&clip).as_deref(),
            Some("All clips replaced")
        );
    }

    #[test]
    fn the_templates_shelf_lists_saved_templates_as_cards() {
        let temp = Temp::new();
        let mut shell = slotted(&temp.path);
        let projects = temp.path.join("Projects");
        let template = shell
            .save_as_template(&templates_root(&projects), "Beach", "")
            .unwrap();
        let mut view = ViewState::new();
        view.projects_root = Some(projects.clone());
        let listed = view.template.listed(&projects).to_vec();
        assert_eq!(listed.len(), 1);
        assert_eq!(super::template_meta(&listed[0]), "16:9 · 1.0 s · 1 clip");
        let card = crate::shell::template_card_id(&template);
        assert!(card.starts_with(cutlass_editor::TEMPLATE_CARD_PREFIX));
        shell.add_template(&card).unwrap();
        let clip = shell.selected().cloned().unwrap();
        assert!(shell.template_clip(&clip).is_some());
    }

    #[test]
    fn a_canvas_shape_reads_as_its_ratio() {
        assert_eq!(shape_label(1080, 1920), "9:16");
        assert_eq!(shape_label(1920, 1080), "16:9");
        assert_eq!(shape_label(1000, 1000), "1:1");
        assert_eq!(shape_label(0, 1080), "?");
    }

    #[test]
    fn templates_live_beside_the_projects() {
        assert_eq!(
            templates_root(Path::new("/data/Cutlass/Projects")),
            Path::new("/data/Cutlass/Templates")
        );
    }

    /// The template clip as the real renderer draws it: the badge on the
    /// clip, the strip of slots over it and the template's panel. Run by
    /// hand with `--ignored`; with `CUTLASS_RENDER_DIR` set the window
    /// pictures are kept there.
    #[test]
    #[ignore = "draws the window with the real renderer"]
    fn a_template_clip_draws_its_badge_strip_and_panel() {
        use egui_kittest::kittest::Queryable;
        let temp = Temp::new();
        let project = temp.path.join("Projects").join("Trip");
        let pattern = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        std::fs::create_dir_all(project.join("Resources")).unwrap();
        std::fs::copy(&pattern, project.join("Resources/pattern.png")).unwrap();
        let mut draft = Draft::empty("trip");
        draft.name = "Trip".to_owned();
        let mut photo = MaterialVideo::new("p");
        photo.kind = MetaType::Photo.as_str().to_owned();
        photo.path = "Resources/pattern.png".to_owned();
        photo.width = 64;
        photo.height = 64;
        draft.materials.videos.push(photo);
        draft
            .materials
            .texts
            .push(cutlass_model::MaterialText::new("words"));
        for (index, length) in [3_000_000, 800_000, 1_500_000, 2_000_000]
            .into_iter()
            .enumerate()
        {
            let start = [0, 3_000_000, 3_800_000, 5_300_000][index];
            let mut segment = Segment::new(format!("s{index}"), "p", TimeRange::new(start, length));
            segment.source_range = Some(TimeRange::new(0, length));
            segment.template_mark = Some(TemplateMark::slot(SlotTakes::Either));
            draft.tracks[0].segments.push(segment);
        }
        let mut title = Segment::new("t", "words", TimeRange::new(0, 2_000_000));
        title.template_mark = Some(TemplateMark::Text);
        let mut row = cutlass_model::Track::main("text");
        row.kind = cutlass_model::TrackType::Text;
        row.flag = cutlass_model::TrackFlag::None;
        row.segments.push(title);
        draft.tracks.push(row);
        cutlass_store::Folder::new(draft).save(&project).unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell = Shell::new(decoders, encoders, Settings::default());
        shell.open_folder(&project).unwrap();
        let projects = temp.path.join("Projects");
        let template = shell
            .save_as_template(&templates_root(&projects), "Beach day", "")
            .unwrap();
        shell.use_template(&template, &projects).unwrap();
        let clip = shell.selected().cloned().unwrap();
        let slot = shell.template_clip(&clip).unwrap().slots[1]
            .slot
            .segment_id
            .clone();
        shell.fill_template_slot(&clip, &slot, &pattern).unwrap();
        let text = shell.template_clip(&clip).unwrap().texts[0]
            .segment_id
            .clone();
        shell.set_template_text(&clip, &text, "Our trip").unwrap();
        shell.start_thumbnails(|| {});
        let mut view = ViewState::new();
        view.projects_root = Some(projects);
        let mut harness = egui_kittest::Harness::builder()
            .with_size(Vec2::new(1600.0, 1000.0))
            .wgpu()
            .build_ui_state(
                |ui, (shell, view): &mut (Shell, ViewState)| show(shell, ui, view, None),
                (shell, view),
            );
        // The tiles' pictures are read off the window's thread.
        for _ in 0..20 {
            harness.run_steps(4);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let image = loop {
            match harness.render() {
                Ok(image) => break image,
                Err(error)
                    if error.contains("timed out") && std::time::Instant::now() < deadline => {}
                Err(error) => panic!("the renderer draws the window: {error:?}"),
            }
        };
        if let Some(dir) = std::env::var_os("CUTLASS_RENDER_DIR").map(std::path::PathBuf::from) {
            std::fs::create_dir_all(&dir).unwrap();
            image.save(dir.join("template-clip.png")).unwrap();
        }
        assert!(
            harness.query_all_by_label("Hide").next().is_some(),
            "the strip is up"
        );
        assert!(
            harness.query_all_by_label("Text 1").next().is_some(),
            "the panel is up"
        );
    }
}
