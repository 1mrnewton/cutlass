//! Project templates in the window: the Template sections in the
//! inspector, Save as template, the Templates tab at home, and the sheet
//! that fills a template's slots with the person's own files.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use egui::{RichText, Ui, Vec2};

use cutlass_model::{SegmentId, SlotTakes, TemplateMark};
use cutlass_store::ListedTemplate;

use crate::shell::Shell;
use crate::widgets::{self, SheetEnd};
use crate::{tokens, ui};

use super::inspector::{InspectorState, body, label_row, note, section, send};

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
    /// The fill sheet, while it is up.
    pub(super) fill: Option<FillSheet>,
    listed: Vec<ListedTemplate>,
    listed_at: Option<Instant>,
}

pub(super) struct SaveSheet {
    name: String,
    category: String,
}

/// What the person has picked so far to fill the open template.
pub(super) struct FillSheet {
    /// One per slot, in fill order; `None` keeps the sample.
    picks: Vec<Option<PathBuf>>,
    /// One per editable text: its id and the words in the box.
    texts: Vec<(SegmentId, String)>,
    music: Option<PathBuf>,
}

impl FillSheet {
    pub(super) fn open(shell: &Shell) -> Self {
        Self {
            picks: vec![None; shell.template_slots().len()],
            texts: shell
                .template_texts()
                .into_iter()
                .map(|text| (text.segment_id, text.words))
                .collect(),
            music: None,
        }
    }

    /// The file picked for slot `index`, or for the music with `None`.
    pub(super) fn picked(&mut self, index: Option<usize>, path: PathBuf) {
        match index {
            Some(index) => {
                if let Some(slot) = self.picks.get_mut(index) {
                    *slot = Some(path);
                }
            }
            None => self.music = Some(path),
        }
    }
}

impl TemplateView {
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
/// project from one and opens the sheet to fill it.
pub(super) fn templates_page(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut TemplateView,
    projects: &Path,
) -> bool {
    let root = templates_root(projects);
    if view
        .listed_at
        .is_none_or(|at| at.elapsed() >= TEMPLATES_CHECK)
    {
        view.listed = Shell::list_templates(&root);
        view.listed_at = Some(Instant::now());
    }
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
        Ok(()) => {
            view.fill = Some(FillSheet::open(shell));
            true
        }
        Err(error) => {
            shell.notice_error(&error);
            false
        }
    }
}

/// "9:16 · 15.0 s · 3 clips".
fn template_meta(template: &ListedTemplate) -> String {
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

/// What the fill sheet asks the window to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FillAsk {
    Nothing,
    /// Pick a file for slot `index`.
    Slot(usize),
    /// Pick a sound for the music.
    Music,
}

/// The sheet that fills the open template.
pub(super) fn fill_sheet(shell: &mut Shell, ui: &mut Ui, view: &mut TemplateView) -> FillAsk {
    let Some(sheet) = view.fill.as_mut() else {
        return FillAsk::Nothing;
    };
    if shell.draft().is_none() {
        view.fill = None;
        return FillAsk::Nothing;
    }
    let slots = shell.template_slots();
    let music = shell.template_has_music();
    let mut ask = FillAsk::Nothing;
    let ((), end) = widgets::sheet(ui, "Fill the template", 560.0, "Fill", |ui| {
        widgets::form_section(ui, "Your media");
        for (index, slot) in slots.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let seconds = slot.length.0 as f64 / 1_000_000.0;
            let label = format!(
                "{}. {} · {seconds:.1} s",
                index + 1,
                takes_label(slot.takes)
            );
            let chosen = sheet.picks.get(index).cloned().flatten();
            if file_row(ui, &label, chosen.as_deref()) {
                ask = FillAsk::Slot(index);
            }
        }
        if !sheet.texts.is_empty() {
            widgets::form_section(ui, "Text");
            for (index, (_, words)) in sheet.texts.iter_mut().enumerate() {
                widgets::form_row(ui, &format!("Text {}", index + 1), |ui| {
                    widgets::text_box(ui, words);
                });
            }
        }
        if music {
            widgets::form_section(ui, "Music");
            if file_row(ui, "Sound", sheet.music.as_deref()) {
                ask = FillAsk::Music;
            }
        }
        ui.add_space(8.0);
        ui.label(
            RichText::new("A slot you leave keeps its sample. One undo takes the fill back.")
                .size(tokens::TYPE_SIZE)
                .color(tokens::MUTED),
        );
    });
    match end {
        SheetEnd::Open => {}
        SheetEnd::Cancel => view.fill = None,
        SheetEnd::Confirm => {
            let picks: Vec<(SegmentId, PathBuf)> = slots
                .iter()
                .zip(&sheet.picks)
                .filter_map(|(slot, pick)| Some((slot.segment_id.clone(), pick.clone()?)))
                .collect();
            let words = shell.template_texts();
            let texts: Vec<(SegmentId, String)> = sheet
                .texts
                .iter()
                .filter(|(id, text)| {
                    words
                        .iter()
                        .any(|now| &now.segment_id == id && &now.words != text)
                })
                .cloned()
                .collect();
            let music = sheet.music.clone();
            match shell.fill_template(&picks, &texts, music.as_deref()) {
                Ok(()) => {
                    shell.set_notice("Filled the template".to_owned());
                    view.fill = None;
                }
                Err(error) => shell.notice_error(&error),
            }
        }
    }
    ask
}

/// A row naming the picked file, or "Sample", with a Choose button.
fn file_row(ui: &mut Ui, label: &str, chosen: Option<&Path>) -> bool {
    widgets::form_row(ui, label, |ui| {
        let name = chosen.and_then(Path::file_name).map_or_else(
            || "Sample".to_owned(),
            |name| name.to_string_lossy().into_owned(),
        );
        let color = if chosen.is_some() {
            tokens::TEXT
        } else {
            tokens::MUTED
        };
        ui.add(
            egui::Label::new(
                RichText::new(name)
                    .size(tokens::TYPE_SIZE + 1.0)
                    .color(color),
            )
            .truncate(),
        );
        ui.add(ui::Button::new("Choose…").text_size(12.0)).clicked()
    })
}

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
    fn a_saved_template_shows_on_the_templates_tab_and_a_pick_fills_the_sheet() {
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

        // Starting from it opens the fill sheet; a picked file lands in its row.
        let template = view.template.listed[0].path.clone();
        shell
            .use_template(&template, view.projects_root.as_ref().unwrap())
            .unwrap();
        view.template.fill = Some(super::FillSheet::open(&shell));
        let picture = temp.path.join("mine.png");
        view.dialogs
            .finish_for_test(Picker::TemplateMedia(Some(0)), Some(picture.clone()));
        frame(&ctx, &mut shell, &mut view);
        frame(&ctx, &mut shell, &mut view);
        let sheet = view.template.fill.as_ref().unwrap();
        assert_eq!(sheet.picks, vec![Some(picture)]);
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
}
