//! The effect studio screen: make an effect or a transition from the
//! blocks the app ships, watch it play, and save it as a recipe.
//!
//! The list of what can be opened is on the left. The preview is in the
//! middle, drawn by the same GPU blocks as the timeline, with the curve of
//! the picked value under it. The passes, their values, the effect's
//! controls and its card are on the right. Every change goes through
//! [`crate::studio::Studio`]; this file draws and asks.

use std::collections::HashMap;

use egui::{
    Align2, Color32, CornerRadius, FontId, Pos2, Rect, RichText, Sense, Stroke, Ui, Vec2, pos2,
    vec2,
};

use cutlass_effects::{
    Block, CROSS, ControlUnit, CrossKind, Curve, EFFECT_CATEGORIES, Ease, MAX_CONTROLS, MAX_PASSES,
    TRANSITION_CATEGORIES, gap_ease,
};

use crate::shell::Shell;
use crate::shell::studio::Backdrop;
use crate::studio::{CurveKind, Doc, Entry, LOOPS, Link, Slot, Stack, Studio};
use crate::{tokens, ui};

const LIST_WIDTH: f32 = 248.0;
const SIDE_WIDTH: f32 = 372.0;
const LANE_HEIGHT: f32 = 236.0;
const ROW: f32 = 26.0;
const LABEL_WIDTH: f32 = 86.0;
const NUMBER_WIDTH: f32 = 58.0;
/// How long a transition's preview rests on each picture, in seconds.
const HOLD: f64 = 0.6;
/// A problem's color: the close button's red, which the app already has.
const PROBLEM: Color32 = tokens::CLOSE;

/// Which list the left side shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    Effects,
    Transitions,
}

/// Something that would throw unsaved changes away, waiting for a yes.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Ask {
    Open(Tab, String),
    New(Tab),
    Delete,
}

/// What the studio screen remembers between frames.
pub struct StudioView {
    /// Whether the studio is the screen the window shows.
    pub open: bool,
    tab: Tab,
    query: String,
    backdrop: Backdrop,
    /// The scene a transition arrives at.
    to: String,
    playing: bool,
    /// Seconds into the preview's loop.
    clock: f64,
    last: Option<f64>,
    /// Numbers being typed, by field.
    fields: HashMap<String, String>,
    asked: Option<Ask>,
    notice: Option<String>,
    /// Set to ask the window for a file to use as the designer's clip.
    pub(super) wants_clip: bool,
    /// The values the graph shows from bottom to top, for the value it
    /// shows. Kept while the pointer is down, so a key being dragged does
    /// not move the graph under it.
    fit: Option<(Slot, f32, f32)>,
}

impl Default for StudioView {
    fn default() -> Self {
        Self {
            open: false,
            tab: Tab::Effects,
            query: String::new(),
            backdrop: Backdrop::Scene("dancer".to_owned()),
            to: "beach".to_owned(),
            playing: true,
            clock: 0.0,
            last: None,
            fields: HashMap::new(),
            asked: None,
            notice: None,
            wants_clip: false,
            fit: None,
        }
    }
}

impl StudioView {
    /// Run the preview over the designer's own clip from now on.
    pub(super) fn use_clip(&mut self) {
        self.backdrop = Backdrop::Clip;
    }
}

/// `value` as a short number: three figures or so.
fn short(value: f32) -> String {
    let size = value.abs();
    let text = if size >= 100.0 {
        format!("{value:.0}")
    } else if size >= 10.0 {
        format!("{value:.1}")
    } else if size >= 1.0 {
        format!("{value:.2}")
    } else {
        format!("{value:.3}")
    };
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        text
    }
}

/// A block's name as words: `RgbSplit` is "RGB split".
fn block_label(block: Block) -> String {
    let name = format!("{block:?}");
    let mut words = String::new();
    for (index, character) in name.chars().enumerate() {
        if index > 0 && character.is_ascii_uppercase() {
            words.push(' ');
            words.push(character.to_ascii_lowercase());
        } else {
            words.push(character);
        }
    }
    words
        .strip_prefix("Rgb ")
        .map_or(words.clone(), |rest| format!("RGB {rest}"))
}

/// A parameter's name as a label: `soft_edge` is "Soft edge".
fn param_label(name: &str) -> String {
    let mut label = name.replace('_', " ");
    if let Some(first) = label.get_mut(..1) {
        first.make_ascii_uppercase();
    }
    label
}

fn ease_label(ease: Ease) -> &'static str {
    match ease {
        Ease::Linear => "Straight",
        Ease::In => "Ease in",
        Ease::Out => "Ease out",
        Ease::InOut => "Ease in and out",
        Ease::Hold => "Hold",
        Ease::Overshoot => "Overshoot",
        Ease::Bounce => "Bounce",
        Ease::Spring => "Spring",
    }
}

fn stack_label(stack: Stack) -> &'static str {
    match stack {
        Stack::Effect => "Passes",
        Stack::Outgoing => "Outgoing picture",
        Stack::Incoming => "Incoming picture",
    }
}

/// How long the preview's loop is, in seconds.
fn loop_length(studio: &Studio) -> f64 {
    match studio.doc() {
        Some(Doc::Transition(recipe)) => HOLD * 2.0 + recipe.duration().0 as f64 / 1_000_000.0,
        _ => f64::from(studio.loop_seconds()),
    }
}

/// Where the lane's playhead is: seconds of an effect, or the progress of
/// a transition from 0 to 1.
fn lane_time(studio: &Studio, clock: f64) -> f32 {
    match studio.doc() {
        Some(Doc::Transition(recipe)) => {
            let run = (recipe.duration().0 as f64 / 1_000_000.0).max(0.001);
            ((clock - HOLD) / run).clamp(0.0, 1.0) as f32
        }
        _ => clock as f32,
    }
}

/// The clock that puts the lane's playhead at `time`.
fn clock_at(studio: &Studio, time: f32) -> f64 {
    match studio.doc() {
        Some(Doc::Transition(recipe)) => {
            HOLD + f64::from(time.clamp(0.0, 1.0)) * recipe.duration().0 as f64 / 1_000_000.0
        }
        _ => f64::from(time.max(0.0)),
    }
}

/// The values a graph shows from bottom to top so that a curve running
/// from `low` to `high` fills about half of it, inside the parameter's own
/// range `min` to `max`. A curve that is one value still gets room each
/// side to drag into.
fn fitted(low: f32, high: f32, min: f32, max: f32) -> (f32, f32) {
    let whole = (max - min).max(f32::EPSILON);
    let room = ((high - low) * 0.5).max(whole * 0.04);
    let (bottom, top) = ((low - room).max(min), (high + room).min(max));
    if top - bottom < whole * 0.02 {
        return (min, max);
    }
    (bottom, top)
}

fn faint(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).size(12.0).color(tokens::FAINT));
}

fn heading(ui: &mut Ui, text: &str) {
    ui.add_space(10.0);
    ui.label(
        RichText::new(text)
            .size(tokens::TYPE_SIZE)
            .color(tokens::MUTED)
            .strong(),
    );
    ui.add_space(4.0);
}

/// A line of text in a box `width` wide that takes a click.
fn tag(ui: &mut Ui, text: &str, width: f32, color: Color32) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(vec2(width, ROW), Sense::click());
    ui.painter().text(
        rect.left_center(),
        Align2::LEFT_CENTER,
        text,
        FontId::proportional(12.5),
        color,
    );
    response
}

/// A number box. It shows `value` until it is typed in, and gives the
/// number typed once the box is left.
fn number(
    ui: &mut Ui,
    fields: &mut HashMap<String, String>,
    key: &str,
    value: f32,
    width: f32,
) -> Option<f32> {
    let mut text = fields.get(key).cloned().unwrap_or_else(|| short(value));
    let field = ui::Field::new(&mut text)
        .id_salt(("studio-number", key))
        .name(key)
        .width(width)
        .text_size(12.0)
        .show(ui);
    if field.response.has_focus() {
        fields.insert(key.to_owned(), text.clone());
    } else {
        fields.remove(key);
    }
    if field.committed {
        return text
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|typed| typed.is_finite());
    }
    None
}

/// A text box over `current`; gives the text when it changed.
fn text(ui: &mut Ui, key: &str, current: &str, width: f32) -> Option<String> {
    let mut typed = current.to_owned();
    let field = ui::Field::new(&mut typed)
        .id_salt(("studio-text", key))
        .name(key)
        .width(width)
        .text_size(12.5)
        .show(ui);
    (field.response.changed() && typed != current).then_some(typed)
}

/// A small picker showing `current`; gives the index picked.
fn pick(ui: &mut Ui, salt: &str, current: &str, options: &[&str], width: f32) -> Option<usize> {
    let mut picked = None;
    egui::ComboBox::from_id_salt(("studio-pick", salt))
        .width(width)
        .selected_text(RichText::new(current).size(12.0))
        .show_ui(ui, |ui| {
            for (index, option) in options.iter().enumerate() {
                if ui
                    .add(ui::Choice::new(*option == current, *option).wide())
                    .clicked()
                {
                    picked = Some(index);
                }
            }
        });
    picked
}

fn icon_button(ui: &mut Ui, name: &str, tip: &str, enabled: bool) -> bool {
    ui.add_enabled(enabled, ui::Button::icon(name).tip(tip))
        .clicked()
}

/// A small button that is an arrowhead, up or down. The app's font has no
/// arrows, so the head is drawn.
fn arrow_button(ui: &mut Ui, up: bool, tip: &str, enabled: bool) -> bool {
    let sense = if enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let (rect, response) = ui.allocate_exact_size(vec2(22.0, ROW), sense);
    let color = if !enabled {
        tokens::MENU_DISABLED
    } else if response.hovered() {
        tokens::TEXT
    } else {
        tokens::MUTED
    };
    if enabled && response.hovered() {
        ui.painter().rect_filled(
            rect,
            CornerRadius::same(tokens::ICON_RADIUS),
            tokens::CHOICE_HOVER,
        );
    }
    let center = rect.center();
    let tip_y = if up { -3.5 } else { 3.5 };
    ui.painter().add(egui::Shape::convex_polygon(
        vec![
            pos2(center.x, center.y + tip_y),
            pos2(center.x - 4.5, center.y - tip_y),
            pos2(center.x + 4.5, center.y - tip_y),
        ],
        color,
        Stroke::NONE,
    ));
    enabled && response.on_hover_text(tip).clicked()
}

/// The studio screen over the whole window.
pub(super) fn show(shell: &mut Shell, ui: &mut Ui, view: &mut StudioView, full: Rect) {
    tick(shell, ui, view);
    let bar = Rect::from_min_size(full.min, vec2(full.width(), tokens::HEAD_HEIGHT));
    ui::Surface::bar().paint(ui.painter(), bar);
    top_bar(shell, ui, view, bar);

    let body = Rect::from_min_max(pos2(full.min.x, bar.max.y + tokens::SPLIT), full.max);
    let list = Rect::from_min_size(body.min, vec2(LIST_WIDTH, body.height()));
    ui::Surface::panel().paint(ui.painter(), list);
    list_panel(shell, ui, view, list.shrink(10.0));

    let side = Rect::from_min_max(pos2(body.max.x - SIDE_WIDTH, body.min.y), body.max);
    let middle = Rect::from_min_max(
        pos2(list.max.x + tokens::SPLIT, body.min.y),
        pos2(side.min.x - tokens::SPLIT, body.max.y),
    );
    if shell.studio().doc().is_none() {
        let rest = Rect::from_min_max(middle.min, body.max);
        ui::Surface::panel().paint(ui.painter(), rest);
        ui.painter().text(
            rest.center(),
            Align2::CENTER_CENTER,
            "Open an effect or a transition from the list, or start a new one.",
            FontId::proportional(tokens::TYPE_SIZE),
            tokens::MUTED,
        );
    } else {
        let lane = Rect::from_min_max(pos2(middle.min.x, middle.max.y - LANE_HEIGHT), middle.max);
        let stage = Rect::from_min_max(middle.min, pos2(middle.max.x, lane.min.y - tokens::SPLIT));
        ui::Surface::panel().paint(ui.painter(), stage);
        stage_panel(shell, ui, view, stage.shrink(10.0));
        ui::Surface::panel().paint(ui.painter(), lane);
        lane_panel(shell, ui, view, lane.shrink(10.0));
        ui::Surface::panel().paint(ui.painter(), side);
        side_panel(shell, ui, view, side);
    }
    keys(shell, ui, view);
    // The pointer is up and nothing is being typed: the edit is over.
    let busy = ui.input(|input| input.pointer.any_down())
        || ui.memory(|memory| memory.focused().is_some());
    if !busy {
        shell.studio_mut().settle();
    }
}

/// Move the preview's clock on.
fn tick(shell: &mut Shell, ui: &Ui, view: &mut StudioView) {
    let now = ui.input(|input| input.time);
    if view.playing {
        if let Some(last) = view.last {
            view.clock += (now - last).clamp(0.0, 0.25);
        }
        ui.ctx().request_repaint();
    }
    view.last = Some(now);
    let length = loop_length(shell.studio()).max(0.05);
    if view.clock >= length || view.clock < 0.0 {
        view.clock = view.clock.rem_euclid(length);
    }
    if shell.studio_clip_loading().is_some() {
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(100));
    }
}

fn keys(shell: &mut Shell, ui: &Ui, view: &mut StudioView) {
    use egui::{Key, Modifiers};
    let typing = ui.memory(|memory| memory.focused().is_some());
    let redo =
        ui.input_mut(|input| input.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z));
    let undo = !redo && ui.input_mut(|input| input.consume_key(Modifiers::COMMAND, Key::Z));
    let save = ui.input_mut(|input| input.consume_key(Modifiers::COMMAND, Key::S));
    if redo && !typing {
        shell.studio_mut().redo();
    }
    if undo && !typing {
        shell.studio_mut().undo();
    }
    if save {
        save_doc(shell, view);
    }
    if typing {
        return;
    }
    if ui.input(|input| input.key_pressed(Key::Space)) {
        view.playing = !view.playing;
    }
    if ui.input(|input| input.key_pressed(Key::Delete) || input.key_pressed(Key::Backspace))
        && let (Some(slot), Some(key)) = (
            shell.studio().picked_slot().cloned(),
            shell.studio().picked_key(),
        )
    {
        shell.studio_mut().remove_key(&slot, key);
    }
}

fn save_doc(shell: &mut Shell, view: &mut StudioView) {
    view.notice = match shell.studio_mut().save() {
        Ok(()) => None,
        Err(problem) => Some(format!("Not saved: {problem}")),
    };
}

/// Do what was asked for, now that unsaved changes are not in the way.
fn act(shell: &mut Shell, view: &mut StudioView, ask: Ask) {
    view.notice = None;
    view.clock = 0.0;
    match ask {
        Ask::Open(Tab::Effects, id) => {
            shell.studio_mut().open_effect(&id);
        }
        Ask::Open(Tab::Transitions, id) => {
            shell.studio_mut().open_transition(&id);
        }
        Ask::New(Tab::Effects) => shell.studio_mut().new_effect(),
        Ask::New(Tab::Transitions) => shell.studio_mut().new_transition(),
        Ask::Delete => {
            if let Err(problem) = shell.studio_mut().delete() {
                view.notice = Some(format!("Not deleted: {problem}"));
            }
        }
    }
    // A card that names scenes opens over them.
    match shell.studio().doc() {
        Some(Doc::Effect(recipe)) if !matches!(view.backdrop, Backdrop::Clip) => {
            view.backdrop = Backdrop::Scene(recipe.preview_scene().to_owned());
        }
        Some(Doc::Transition(recipe)) => {
            if let [from, to] = recipe.preview.as_slice() {
                view.backdrop = Backdrop::Scene(from.clone());
                view.to.clone_from(to);
            }
        }
        _ => {}
    }
}

/// Ask first when the open document has changes that would be lost.
fn ask(shell: &mut Shell, view: &mut StudioView, wanted: Ask) {
    let loses = shell.studio().is_dirty() || wanted == Ask::Delete && shell.studio().is_saved();
    if loses {
        view.asked = Some(wanted);
    } else {
        act(shell, view, wanted);
    }
}

fn top_bar(shell: &mut Shell, ui: &mut Ui, view: &mut StudioView, bar: Rect) {
    let inner = Rect::from_min_max(
        pos2(bar.min.x + 84.0, bar.min.y),
        pos2(bar.max.x - 10.0, bar.max.y),
    );
    let mut row = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(inner)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    let ui = &mut row;
    if ui
        .add(
            ui::Button::ghost("Back")
                .with_icon("chevronLeft")
                .tip("Back to the editor"),
        )
        .clicked()
    {
        view.open = false;
    }
    ui.label(
        RichText::new("Effect studio")
            .size(14.0)
            .color(tokens::TEXT),
    );
    ui.add_space(12.0);
    let Some(doc) = shell.studio().doc().cloned() else {
        return;
    };
    if let Some(name) = text(ui, "name", doc.name(), 170.0) {
        shell.studio_mut().set_name(&name);
    }
    faint(ui, "id");
    if let Some(id) = text(ui, "id", doc.id(), 150.0) {
        shell.studio_mut().set_id(&id);
    }
    let categories = if doc.is_effect() {
        EFFECT_CATEGORIES
    } else {
        TRANSITION_CATEGORIES
    };
    if let Some(index) = pick(ui, "category", doc.category(), categories, 84.0) {
        shell.studio_mut().set_category(categories[index]);
    }
    let problem = shell.studio().problem();
    let (status, color) = match (&view.notice, &problem) {
        (Some(notice), _) => (notice.clone(), PROBLEM),
        (None, Some(problem)) => (problem.clone(), PROBLEM),
        (None, None) if shell.studio().is_dirty() => ("Unsaved changes".to_owned(), tokens::MUTED),
        (None, None) => ("Saved".to_owned(), tokens::FAINT),
    };
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        let can_save =
            problem.is_none() && shell.studio().is_dirty() && shell.studio().folder().is_some();
        if ui
            .add_enabled(can_save, ui::Button::primary("Save"))
            .clicked()
        {
            save_doc(shell, view);
        }
        if icon_button(ui, "delete", "Delete this recipe", true) {
            ask(shell, view, Ask::Delete);
        }
        if icon_button(ui, "duplicate", "Carry on as a copy", true) {
            shell.studio_mut().duplicate();
        }
        if icon_button(ui, "redo", "Redo", shell.studio().can_redo()) {
            shell.studio_mut().redo();
        }
        if icon_button(ui, "undo", "Undo", shell.studio().can_undo()) {
            shell.studio_mut().undo();
        }
        ui.add_space(8.0);
        ui.add(
            egui::Label::new(RichText::new(status).size(12.0).color(color))
                .truncate()
                .selectable(false),
        );
    });
}

fn list_panel(shell: &mut Shell, ui: &mut Ui, view: &mut StudioView, rect: Rect) {
    let mut column = ui.new_child(egui::UiBuilder::new().max_rect(rect));
    let ui = &mut column;
    ui::Segmented::show(ui, |ui| {
        for (tab, label) in [(Tab::Effects, "Effects"), (Tab::Transitions, "Transitions")] {
            if ui.add(ui::Choice::new(view.tab == tab, label)).clicked() {
                view.tab = tab;
            }
        }
    });
    ui.add_space(8.0);
    let label = match view.tab {
        Tab::Effects => "New effect",
        Tab::Transitions => "New transition",
    };
    if ui
        .add(ui::Button::new(label).with_icon("plus").full_width())
        .clicked()
    {
        ask(shell, view, Ask::New(view.tab));
    }
    ui.add_space(8.0);
    ui::Field::new(&mut view.query)
        .id_salt("studio-search")
        .name("Search")
        .hint("Search")
        .icon("search")
        .width(ui.available_width())
        .show(ui);
    ui.add_space(6.0);
    if let Some(asked) = view.asked.clone() {
        let question = if asked == Ask::Delete {
            "Delete this recipe's file?"
        } else {
            "Throw away the unsaved changes?"
        };
        ui.label(RichText::new(question).size(12.5).color(tokens::TEXT));
        ui.horizontal(|ui| {
            let yes = if asked == Ask::Delete {
                "Delete"
            } else {
                "Throw away"
            };
            if ui.add(ui::Button::new(yes)).clicked() {
                view.asked = None;
                act(shell, view, asked.clone());
            }
            if ui.add(ui::Button::ghost("Keep")).clicked() {
                view.asked = None;
            }
        });
        ui.add_space(6.0);
    }
    let entries = match view.tab {
        Tab::Effects => Studio::effects(),
        Tab::Transitions => Studio::transitions(),
    };
    let query = view.query.trim().to_lowercase();
    let open_id = shell
        .studio()
        .doc()
        .filter(|doc| doc.is_effect() == (view.tab == Tab::Effects))
        .map(|doc| doc.id().to_owned());
    let refused: Vec<(String, String)> = shell
        .studio()
        .refused()
        .iter()
        .map(|refused| (refused.file.clone(), refused.error.to_string()))
        .collect();
    let mut opened = None;
    ui::Scroll::vertical("studio-list")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for (file, why) in &refused {
                ui.label(RichText::new(file).size(12.0).color(PROBLEM));
                ui.label(RichText::new(why).size(11.5).color(tokens::MUTED));
                ui.add_space(4.0);
            }
            if query.is_empty() && !entries.iter().any(|entry| entry.studio) {
                faint(
                    ui,
                    "Nothing of your own yet. A bundled one opens as a copy.",
                );
                ui.add_space(6.0);
            }
            for entry in entries
                .iter()
                .filter(|entry| query.is_empty() || entry.name.to_lowercase().contains(&query))
            {
                if entry_row(ui, entry, open_id.as_deref() == Some(entry.id.as_str())) {
                    opened = Some(entry.id.clone());
                }
            }
        });
    if let Some(id) = opened {
        ask(shell, view, Ask::Open(view.tab, id));
    }
}

/// One row of the list: the name, and "Yours" on one made here.
fn entry_row(ui: &mut Ui, entry: &Entry, open: bool) -> bool {
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), ROW), Sense::click());
    if open || response.hovered() {
        let fill = if open {
            tokens::ACTIVE
        } else {
            tokens::CHOICE_HOVER
        };
        ui.painter()
            .rect_filled(rect, CornerRadius::same(tokens::RADIUS_SM), fill);
    }
    ui.painter().text(
        rect.left_center() + vec2(8.0, 0.0),
        Align2::LEFT_CENTER,
        &entry.name,
        FontId::proportional(12.5),
        if entry.studio {
            tokens::TEXT
        } else {
            tokens::MUTED
        },
    );
    let note = if entry.studio {
        "Yours"
    } else {
        entry.category.as_str()
    };
    ui.painter().text(
        rect.right_center() - vec2(8.0, 0.0),
        Align2::RIGHT_CENTER,
        note,
        FontId::proportional(11.0),
        if entry.studio {
            tokens::EXPORT
        } else {
            tokens::FAINT
        },
    );
    response.clicked()
}

fn stage_panel(shell: &mut Shell, ui: &mut Ui, view: &mut StudioView, rect: Rect) {
    let strip = 2.0 * ROW + 14.0;
    let picture = Rect::from_min_max(rect.min, pos2(rect.max.x, rect.max.y - strip));
    let controls = Rect::from_min_max(pos2(rect.min.x, picture.max.y + 8.0), rect.max);
    let transition = matches!(shell.studio().doc(), Some(Doc::Transition(_)));
    let time = if transition {
        f64::from(lane_time(shell.studio(), view.clock))
    } else {
        view.clock
    };

    let mut strip_ui = ui.new_child(egui::UiBuilder::new().max_rect(controls));
    let mut plain = false;
    strip_ui.horizontal(|ui| {
        let icon = if view.playing { "pause" } else { "play" };
        if icon_button(ui, icon, "Play or pause (Space)", true) {
            view.playing = !view.playing;
        }
        let length = loop_length(shell.studio());
        ui.label(
            RichText::new(format!("{:.2} s of {:.1} s", view.clock, length))
                .size(12.0)
                .color(tokens::MUTED)
                .monospace(),
        );
        if !transition {
            let current = format!("{} s loop", short(shell.studio().loop_seconds()));
            let options: Vec<String> = LOOPS
                .iter()
                .map(|s| format!("{} s loop", short(*s)))
                .collect();
            let options: Vec<&str> = options.iter().map(String::as_str).collect();
            if let Some(index) = pick(ui, "loop", &current, &options, 84.0) {
                shell.studio_mut().set_loop_seconds(LOOPS[index]);
            }
            let before =
                ui.add(ui::Button::new("Before").tip("Hold to see the picture without the effect"));
            plain = before.is_pointer_button_down_on();
        }
    });
    strip_ui.add_space(4.0);
    backdrop_row(shell, &mut strip_ui, view, transition);

    ui.painter()
        .rect_filled(picture, CornerRadius::same(tokens::RADIUS_MD), tokens::GAP);
    let frame = shell.studio_frame(&view.backdrop, &view.to, time, plain);
    match frame {
        Some((texture, [width, height])) => {
            let size = vec2(width as f32, height as f32);
            let scale = (picture.width() / size.x).min(picture.height() / size.y);
            let placed = Rect::from_center_size(picture.center(), size * scale);
            ui.painter().image(
                texture,
                placed,
                Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        }
        None => {
            let why = if shell.studio_clip_loading().is_some() {
                "Reading the clip…"
            } else if matches!(view.backdrop, Backdrop::Clip) {
                "The clip is not loaded."
            } else {
                "The preview needs the GPU."
            };
            ui.painter().text(
                picture.center(),
                Align2::CENTER_CENTER,
                why,
                FontId::proportional(tokens::TYPE_SIZE),
                tokens::MUTED,
            );
        }
    }
}

/// What the preview runs over: the bundled scenes, or the designer's clip.
fn backdrop_row(shell: &mut Shell, ui: &mut Ui, view: &mut StudioView, transition: bool) {
    ui.horizontal(|ui| {
        let scenes = cutlass_effects::preview_scenes();
        let current = match &view.backdrop {
            Backdrop::Scene(name) => param_label(name),
            Backdrop::Clip => "My clip".to_owned(),
        };
        faint(ui, if transition { "From" } else { "Over" });
        let mut options: Vec<String> = scenes.iter().map(|scene| param_label(scene)).collect();
        if shell.studio_clip().is_some() {
            options.push("My clip".to_owned());
        }
        let names: Vec<&str> = options.iter().map(String::as_str).collect();
        if let Some(index) = pick(ui, "backdrop", &current, &names, 96.0) {
            view.backdrop = scenes
                .get(index)
                .map_or(Backdrop::Clip, |scene| Backdrop::Scene((*scene).to_owned()));
        }
        if transition {
            faint(ui, "To");
            let scene_names: Vec<String> = scenes.iter().map(|scene| param_label(scene)).collect();
            let scene_names: Vec<&str> = scene_names.iter().map(String::as_str).collect();
            if let Some(index) = pick(ui, "to", &param_label(&view.to), &scene_names, 96.0) {
                view.to = scenes[index].to_owned();
            }
        }
        if ui
            .add(
                ui::Button::ghost("Use my clip…")
                    .tip("Preview over a video or picture of your own"),
            )
            .clicked()
        {
            view.wants_clip = true;
        }
        if let Some(path) = shell.studio_clip_loading() {
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned());
            faint(ui, &format!("Reading {}…", name.unwrap_or_default()));
        } else if let Some(error) = shell.studio_clip_error() {
            ui.label(RichText::new(error).size(12.0).color(PROBLEM));
        } else if let Some(path) = shell.studio_clip() {
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned());
            faint(ui, &name.unwrap_or_default());
        }
    });
}

fn lane_panel(shell: &mut Shell, ui: &mut Ui, view: &mut StudioView, rect: Rect) {
    let head = Rect::from_min_size(rect.min, vec2(rect.width(), ROW + 4.0));
    let graph = Rect::from_min_max(pos2(rect.min.x, head.max.y + 6.0), rect.max);
    let slot = shell.studio().picked_slot().cloned();
    let mut head_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(head)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    match &slot {
        Some(slot) => lane_head(shell, &mut head_ui, view, slot),
        None => faint(
            &mut head_ui,
            "Pick a value on the right to see how it moves. Drag here to move the playhead.",
        ),
    }
    lane_graph(shell, ui, view, graph, slot.as_ref());
}

/// The picked value's own settings: how it moves, and the numbers of that.
fn lane_head(shell: &mut Shell, ui: &mut Ui, view: &mut StudioView, slot: &Slot) {
    let Some(doc) = shell.studio().doc().cloned() else {
        return;
    };
    let Some(param) = doc.param(slot) else {
        return;
    };
    ui.label(
        RichText::new(param_label(slot.name()))
            .size(tokens::TYPE_SIZE)
            .color(tokens::TEXT),
    );
    let kind = CurveKind::of(doc.curve(slot));
    ui::Segmented::show(ui, |ui| {
        for option in CurveKind::ALL {
            if ui
                .add(ui::Choice::new(kind == option, option.label()))
                .clicked()
            {
                shell.studio_mut().set_kind(slot, option);
            }
        }
    });
    let time = lane_time(shell.studio(), view.clock);
    match doc.curve(slot) {
        Some(Curve::Keys { keys, repeat, ease }) => {
            let picked = shell.studio().picked_key().filter(|key| *key < keys.len());
            if icon_button(ui, "keyframe", "Add a key at the playhead", true) {
                let value = doc.value_at(slot, time).unwrap_or(param.default);
                shell.studio_mut().add_key(slot, time, value);
            }
            if let Some(key) = picked {
                let [at, value] = keys[key];
                faint(ui, "at");
                if let Some(typed) = number(ui, &mut view.fields, "key time", at, NUMBER_WIDTH) {
                    shell.studio_mut().move_key(slot, key, typed, value);
                }
                faint(ui, "is");
                if let Some(typed) = number(ui, &mut view.fields, "key value", value, NUMBER_WIDTH)
                {
                    shell.studio_mut().move_key(slot, key, at, typed);
                }
                if key + 1 < keys.len() {
                    let current = gap_ease(ease, key);
                    let labels: Vec<&str> =
                        Ease::ALL.iter().map(|ease| ease_label(*ease)).collect();
                    faint(ui, "then");
                    if let Some(index) = pick(ui, "ease", ease_label(current), &labels, 112.0) {
                        shell.studio_mut().set_ease(slot, key, Ease::ALL[index]);
                    }
                }
                if icon_button(ui, "delete", "Remove this key (Delete)", keys.len() > 1) {
                    shell.studio_mut().remove_key(slot, key);
                }
            } else {
                faint(ui, "Double-click the graph to add a key.");
            }
            let mut repeats = repeat.is_some();
            if ui.add(ui::Check::new(&mut repeats, "Repeat")).changed() {
                let every = keys.last().map_or(1.0, |last| last[0].max(0.1));
                shell
                    .studio_mut()
                    .set_repeat(slot, repeats.then_some(every));
            }
            if let Some(every) = repeat
                && let Some(typed) = number(ui, &mut view.fields, "repeat", *every, NUMBER_WIDTH)
            {
                shell.studio_mut().set_repeat(slot, Some(typed));
            }
        }
        Some(Curve::Wave { wave }) => {
            let mut next = wave.clone();
            for (label, value) in [
                ("from", &mut next.from),
                ("to", &mut next.to),
                ("every", &mut next.period),
            ] {
                faint(ui, label);
                if let Some(typed) = number(
                    ui,
                    &mut view.fields,
                    &format!("wave {label}"),
                    *value,
                    NUMBER_WIDTH,
                ) {
                    *value = typed;
                }
            }
            faint(ui, "s");
            if next != *wave {
                shell.studio_mut().set_wave(slot, &next);
            }
        }
        Some(Curve::Jitter { jitter }) => {
            let mut next = jitter.clone();
            for (label, value) in [
                ("from", &mut next.from),
                ("to", &mut next.to),
                ("steps a second", &mut next.rate),
            ] {
                faint(ui, label);
                if let Some(typed) = number(
                    ui,
                    &mut view.fields,
                    &format!("random {label}"),
                    *value,
                    NUMBER_WIDTH,
                ) {
                    *value = typed;
                }
            }
            faint(ui, "seed");
            if let Some(typed) = number(
                ui,
                &mut view.fields,
                "random seed",
                jitter.seed as f32,
                NUMBER_WIDTH,
            ) {
                next.seed = typed.clamp(0.0, 1_000_000.0) as u32;
            }
            if next != *jitter {
                shell.studio_mut().set_jitter(slot, &next);
            }
        }
        Some(Curve::Constant(_)) | None => {
            faint(
                ui,
                "One value the whole way. Keys, Wave or Random make it move.",
            );
        }
    }
}

/// The graph: time across, the value up. Keys drag; the rest of it moves
/// the playhead.
fn lane_graph(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut StudioView,
    rect: Rect,
    slot: Option<&Slot>,
) {
    let span = shell.studio().span().max(0.001);
    ui.painter()
        .rect_filled(rect, CornerRadius::same(tokens::RADIUS_SM), tokens::GAP);
    let plot = rect.shrink2(vec2(10.0, 12.0));
    let x_of = |time: f32| plot.min.x + (time / span).clamp(0.0, 1.0) * plot.width();
    let time_of = |x: f32| ((x - plot.min.x) / plot.width().max(1.0)).clamp(0.0, 1.0) * span;
    // A line each quarter of the way across.
    for quarter in 0..=4 {
        let x = plot.min.x + plot.width() * quarter as f32 / 4.0;
        ui.painter().line_segment(
            [pos2(x, rect.min.y), pos2(x, rect.max.y)],
            Stroke::new(1.0, tokens::LANE_LINE),
        );
        ui.painter().text(
            pos2(x + 3.0, rect.max.y - 2.0),
            Align2::LEFT_BOTTOM,
            short(span * quarter as f32 / 4.0),
            FontId::proportional(10.0),
            tokens::FAINT,
        );
    }
    let background = ui.interact(
        rect,
        ui.scope_id().with("studio-lane"),
        Sense::click_and_drag(),
    );
    let doc = shell.studio().doc().cloned();
    let shown = slot.and_then(|slot| {
        let doc = doc.as_ref()?;
        Some((slot, doc, doc.param(slot)?))
    });
    let mut on_key = false;
    if let Some((slot, doc, param)) = shown {
        let steps = (plot.width() / 2.0).max(8.0) as usize;
        let samples: Vec<(f32, f32)> = (0..=steps)
            .filter_map(|step| {
                let time = span * step as f32 / steps as f32;
                Some((time, doc.value_at(slot, time)?))
            })
            .collect();
        // The graph shows the part of the range the curve uses. It is
        // fitted again only with the pointer up.
        let held = ui.input(|input| input.pointer.any_down());
        let kept = view
            .fit
            .as_ref()
            .filter(|(fitted_for, _, _)| held && fitted_for == slot)
            .map(|(_, bottom, top)| (*bottom, *top));
        let (bottom, top) = kept.unwrap_or_else(|| {
            let low = samples
                .iter()
                .map(|sample| sample.1)
                .fold(f32::MAX, f32::min);
            let high = samples
                .iter()
                .map(|sample| sample.1)
                .fold(f32::MIN, f32::max);
            if low > high {
                (param.min, param.max)
            } else {
                fitted(low, high, param.min, param.max)
            }
        });
        view.fit = Some((slot.clone(), bottom, top));
        let range = (top - bottom).max(f32::EPSILON);
        let y_of =
            |value: f32| plot.max.y - ((value - bottom) / range).clamp(0.0, 1.0) * plot.height();
        let value_of =
            |y: f32| bottom + ((plot.max.y - y) / plot.height().max(1.0)).clamp(0.0, 1.0) * range;
        ui.painter().text(
            pos2(rect.min.x + 4.0, rect.min.y + 2.0),
            Align2::LEFT_TOP,
            short(top),
            FontId::proportional(10.0),
            tokens::FAINT,
        );
        ui.painter().text(
            pos2(rect.min.x + 4.0, plot.max.y),
            Align2::LEFT_BOTTOM,
            short(bottom),
            FontId::proportional(10.0),
            tokens::FAINT,
        );
        let line: Vec<Pos2> = samples
            .iter()
            .map(|(time, value)| pos2(x_of(*time), y_of(*value)))
            .collect();
        ui.painter()
            .add(egui::Shape::line(line, Stroke::new(1.5, tokens::EXPORT)));
        if let Some(Curve::Keys { keys, .. }) = doc.curve(slot) {
            let picked = shell.studio().picked_key();
            for (index, [time, value]) in keys.iter().copied().enumerate() {
                let at = pos2(x_of(time), y_of(value));
                let handle = ui.interact(
                    Rect::from_center_size(at, Vec2::splat(16.0)),
                    ui.scope_id().with(("studio-key", index)),
                    Sense::click_and_drag(),
                );
                on_key |= handle.hovered() || handle.dragged();
                if handle.clicked() || handle.drag_started() {
                    shell.studio_mut().pick_key(Some(index));
                }
                if handle.dragged()
                    && let Some(pointer) = handle.interact_pointer_pos()
                {
                    shell.studio_mut().move_key(
                        slot,
                        index,
                        time_of(pointer.x),
                        value_of(pointer.y),
                    );
                }
                let fill = if picked == Some(index) {
                    tokens::TEXT
                } else {
                    tokens::EXPORT
                };
                ui.painter().circle_filled(at, 5.0, fill);
                ui.painter()
                    .circle_stroke(at, 5.0, Stroke::new(1.0, tokens::GAP));
            }
            if background.double_clicked()
                && let Some(pointer) = background.interact_pointer_pos()
            {
                shell
                    .studio_mut()
                    .add_key(slot, time_of(pointer.x), value_of(pointer.y));
            }
        }
    }
    if !on_key
        && (background.dragged() || background.clicked())
        && !background.double_clicked()
        && let Some(pointer) = background.interact_pointer_pos()
    {
        view.clock = clock_at(shell.studio(), time_of(pointer.x));
        view.playing = false;
    }
    let playhead = x_of(lane_time(shell.studio(), view.clock));
    ui.painter().line_segment(
        [pos2(playhead, rect.min.y), pos2(playhead, rect.max.y)],
        Stroke::new(1.0, tokens::PLAYHEAD),
    );
}

fn side_panel(shell: &mut Shell, ui: &mut Ui, view: &mut StudioView, rect: Rect) {
    let mut column = ui.new_child(egui::UiBuilder::new().max_rect(rect));
    ui::Scroll::vertical("studio-side")
        .auto_shrink([false, false])
        .content_margin(egui::Margin::symmetric(12, 6))
        .show(&mut column, |ui| {
            let Some(doc) = shell.studio().doc().cloned() else {
                return;
            };
            for stack in doc.stacks() {
                stack_section(shell, ui, &doc, *stack);
            }
            if let Some((stack, pass)) = shell.studio().picked_pass()
                && let Some(block) = doc.passes(stack).get(pass).map(|pass| pass.block)
            {
                heading(ui, &block_label(block));
                for param in block.params() {
                    value_row(shell, ui, view, &doc, &Slot::pass(stack, pass, param.name));
                }
                if block.params().is_empty() {
                    faint(ui, "This block has nothing to set.");
                }
            }
            if !doc.is_effect() {
                heading(ui, "The swap");
                for param in CROSS {
                    value_row(shell, ui, view, &doc, &Slot::cross(param.name));
                }
            }
            controls_section(shell, ui, view, &doc);
            card_section(shell, ui, view, &doc);
            ui.add_space(16.0);
        });
}

/// One stack: its passes in order, and a way to add one.
fn stack_section(shell: &mut Shell, ui: &mut Ui, doc: &Doc, stack: Stack) {
    let passes = doc.passes(stack);
    heading(
        ui,
        &format!("{} · {} of {MAX_PASSES}", stack_label(stack), passes.len()),
    );
    let picked = shell.studio().picked_pass();
    let (mut removed, mut moved, mut chosen) = (None, None, None);
    for (index, pass) in passes.iter().enumerate() {
        let on = picked == Some((stack, index));
        ui.horizontal(|ui| {
            let muted = shell.studio().is_muted(stack, index);
            let eye = if muted { "viewOff" } else { "view" };
            if icon_button(ui, eye, "Leave this pass out of the preview", true) {
                shell.studio_mut().set_muted(stack, index, !muted);
            }
            let room = (ui.available_width() - 3.0 * (tokens::ICON_BUTTON_SIZE + 4.0)).max(60.0);
            let (rect, response) = ui.allocate_exact_size(vec2(room, ROW), Sense::click());
            if on || response.hovered() {
                let fill = if on {
                    tokens::ACTIVE
                } else {
                    tokens::CHOICE_HOVER
                };
                ui.painter()
                    .rect_filled(rect, CornerRadius::same(tokens::RADIUS_SM), fill);
            }
            ui.painter().text(
                rect.left_center() + vec2(8.0, 0.0),
                Align2::LEFT_CENTER,
                format!("{}  {}", index + 1, block_label(pass.block)),
                FontId::proportional(12.5),
                if muted { tokens::FAINT } else { tokens::TEXT },
            );
            if response.clicked() {
                chosen = Some(index);
            }
            if arrow_button(ui, true, "Run earlier", index > 0) {
                moved = Some((index, index - 1));
            }
            if arrow_button(ui, false, "Run later", index + 1 < passes.len()) {
                moved = Some((index, index + 1));
            }
            if icon_button(ui, "delete", "Remove this pass", true) {
                removed = Some(index);
            }
        });
    }
    if let Some(index) = chosen {
        shell.studio_mut().pick_pass(stack, index);
    }
    if let Some((from, to)) = moved {
        shell.studio_mut().move_pass(stack, from, to);
    }
    if let Some(index) = removed {
        shell.studio_mut().remove_pass(stack, index);
    }
    if passes.len() >= MAX_PASSES {
        faint(ui, "The stack is full.");
        return;
    }
    let mut added = None;
    ui.menu_button(RichText::new("+ Add a block").size(12.5), |ui| {
        ui::Scroll::vertical(("studio-blocks", stack_label(stack)))
            .max_height(340.0)
            .show(ui, |ui| {
                for block in Block::ALL {
                    if ui.button(block_label(block)).clicked() {
                        added = Some(block);
                        ui.close();
                    }
                }
            });
    });
    if let Some(block) = added {
        shell.studio_mut().add_pass(stack, block);
    }
}

/// What a value that moves is, in a few words.
fn curve_note(curve: &Curve) -> String {
    match curve {
        Curve::Constant(value) => short(*value),
        Curve::Keys { keys, repeat, .. } => {
            let keys = if keys.len() == 1 {
                "1 key".to_owned()
            } else {
                format!("{} keys", keys.len())
            };
            match repeat {
                Some(every) => format!("{keys}, every {} s", short(*every)),
                None => keys,
            }
        }
        Curve::Wave { wave } => format!(
            "Wave {} to {}, every {} s",
            short(wave.from),
            short(wave.to),
            short(wave.period)
        ),
        Curve::Jitter { jitter } => format!(
            "Random {} to {}, {} a second",
            short(jitter.from),
            short(jitter.to),
            short(jitter.rate)
        ),
    }
}

/// One value: its name, a slider while it is fixed or a note of how it
/// moves, the number it is now, and a way back to the block's default.
fn value_row(shell: &mut Shell, ui: &mut Ui, view: &mut StudioView, doc: &Doc, slot: &Slot) {
    let Some(param) = doc.param(slot) else {
        return;
    };
    let curve = doc.curve(slot);
    let picked = shell.studio().picked_slot() == Some(slot);
    let now = doc
        .value_at(slot, lane_time(shell.studio(), view.clock))
        .unwrap_or(param.default);
    let key = format!("{slot:?}");
    ui.horizontal(|ui| {
        let color = if picked {
            tokens::EXPORT
        } else if curve.is_some() {
            tokens::TEXT
        } else {
            tokens::MUTED
        };
        if tag(ui, &param_label(slot.name()), LABEL_WIDTH, color)
            .on_hover_text("Show how this value moves")
            .clicked()
        {
            shell.studio_mut().pick_slot(Some(slot.clone()));
        }
        let rest =
            (ui.available_width() - NUMBER_WIDTH - tokens::ICON_BUTTON_SIZE - 20.0).max(60.0);
        let kinds = matches!(slot, Slot::Cross { name } if name == "kind");
        if kinds {
            // The swap's kind is one of a list, not an amount.
            let names: Vec<String> = CrossKind::ALL
                .iter()
                .map(|kind| format!("{kind:?}"))
                .collect();
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            let index = (now.round().max(0.0) as usize).min(names.len() - 1);
            if let Some(index) = pick(ui, &key, names[index], &names, rest - 8.0) {
                shell.studio_mut().set_fixed(slot, index as f32);
            }
        } else if curve.is_none() && matches!(slot, Slot::Cross { name } if name == "progress") {
            // Left out, the swap keeps pace with the transition itself.
            tag(ui, "Follows the transition", rest, tokens::MUTED);
        } else if matches!(curve, None | Some(Curve::Constant(_))) {
            let mut value = f64::from(now);
            let slider = ui::Slider::new(&mut value, f64::from(param.min)..=f64::from(param.max))
                .name(slot.name())
                .width(rest);
            if ui.add(slider).changed() {
                shell.studio_mut().set_fixed(slot, value as f32);
                shell.studio_mut().pick_slot(Some(slot.clone()));
            }
        } else if let Some(curve) = curve
            && tag(ui, &curve_note(curve), rest, tokens::MUTED).clicked()
        {
            shell.studio_mut().pick_slot(Some(slot.clone()));
        }
        let fixed = matches!(curve, None | Some(Curve::Constant(_)));
        if fixed && !kinds {
            if let Some(typed) = number(ui, &mut view.fields, &key, now, NUMBER_WIDTH) {
                shell.studio_mut().set_fixed(slot, typed);
            }
        } else {
            tag(ui, &short(now), NUMBER_WIDTH, tokens::MUTED);
        }
        if icon_button(
            ui,
            "reset",
            "Back to the block's own value",
            curve.is_some(),
        ) {
            shell.studio_mut().reset(slot);
        }
    });
}

/// The effect's controls: what its clip's inspector shows.
fn controls_section(shell: &mut Shell, ui: &mut Ui, view: &mut StudioView, doc: &Doc) {
    let Doc::Effect(recipe) = doc else {
        return;
    };
    heading(
        ui,
        &format!("Controls · {} of {MAX_CONTROLS}", recipe.controls.len()),
    );
    faint(ui, "The settings an editor sees on the effect's clip.");
    let picked = shell
        .studio()
        .picked_pass()
        .filter(|(stack, _)| *stack == Stack::Effect)
        .map(|(_, pass)| pass);
    let mut removed = None;
    for (index, control) in recipe.controls.iter().enumerate() {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if let Some(name) = text(ui, &format!("control name {index}"), &control.name, 132.0) {
                shell
                    .studio_mut()
                    .set_control(index, |control| control.name = name);
            }
            faint(ui, "id");
            if let Some(id) = text(ui, &format!("control id {index}"), &control.id, 96.0) {
                shell
                    .studio_mut()
                    .set_control(index, |control| control.id = id);
            }
            if icon_button(ui, "delete", "Remove this control", true) {
                removed = Some(index);
            }
        });
        ui.horizontal(|ui| {
            for (label, value, which) in [
                ("from", control.min, 0),
                ("starts at", control.default, 1),
                ("to", control.max, 2),
            ] {
                faint(ui, label);
                let key = format!("control {index} {label}");
                if let Some(typed) = number(ui, &mut view.fields, &key, value, 46.0) {
                    shell
                        .studio_mut()
                        .set_control(index, |control| match which {
                            0 => control.min = typed,
                            1 => control.default = typed,
                            _ => control.max = typed,
                        });
                }
            }
        });
        ui.horizontal(|ui| {
            const UNITS: [(ControlUnit, &str); 4] = [
                (ControlUnit::Plain, "Number"),
                (ControlUnit::Percent, "Percent"),
                (ControlUnit::Times, "Times"),
                (ControlUnit::Degrees, "Degrees"),
            ];
            let labels: Vec<&str> = UNITS.iter().map(|(_, label)| *label).collect();
            let current = UNITS
                .iter()
                .find(|(unit, _)| *unit == control.unit)
                .map_or("Number", |(_, label)| *label);
            faint(ui, "shown as");
            if let Some(picked) = pick(ui, &format!("unit {index}"), current, &labels, 76.0) {
                let unit = UNITS[picked].0;
                shell
                    .studio_mut()
                    .set_control(index, |control| control.unit = unit);
            }
            let mut speed = control.speed;
            if ui
                .add(ui::Check::new(&mut speed, "Plays the effect faster"))
                .changed()
            {
                shell
                    .studio_mut()
                    .set_control(index, |control| control.speed = speed);
            }
        });
        let Some(pass) = picked else {
            continue;
        };
        let Some(block) = recipe.passes.get(pass).map(|pass| pass.block) else {
            continue;
        };
        for param in block.params() {
            let link = shell.studio().link(index, pass, param.name);
            ui.horizontal(|ui| {
                tag(
                    ui,
                    &format!("{} of pass {}", param_label(param.name), pass + 1),
                    150.0,
                    tokens::MUTED,
                );
                const LINKS: [(Link, &str); 3] = [
                    (Link::None, "Left alone"),
                    (Link::Scale, "Multiplied by it"),
                    (Link::Set, "Set to it"),
                ];
                let labels: Vec<&str> = LINKS.iter().map(|(_, label)| *label).collect();
                let current = LINKS
                    .iter()
                    .find(|(kind, _)| *kind == link)
                    .map_or("Left alone", |(_, label)| *label);
                let salt = format!("link {index} {}", param.name);
                if let Some(picked) = pick(ui, &salt, current, &labels, 118.0) {
                    shell
                        .studio_mut()
                        .set_link(index, pass, param.name, LINKS[picked].0);
                }
            });
        }
    }
    if let Some(index) = removed {
        shell.studio_mut().remove_control(index);
    }
    ui.add_space(6.0);
    if recipe.controls.len() < MAX_CONTROLS
        && ui
            .add(ui::Button::new("Add a control").with_icon("plus"))
            .clicked()
    {
        shell.studio_mut().add_control();
    }
}

/// What the shelf's card shows, and for a transition how long it lasts.
fn card_section(shell: &mut Shell, ui: &mut Ui, view: &mut StudioView, doc: &Doc) {
    heading(ui, "On the shelf");
    match doc {
        Doc::Effect(recipe) => {
            ui.horizontal(|ui| {
                tag(ui, "Card scene", LABEL_WIDTH, tokens::MUTED);
                let scenes = cutlass_effects::preview_scenes();
                let labels: Vec<String> = scenes.iter().map(|scene| param_label(scene)).collect();
                let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
                let current = param_label(recipe.preview_scene());
                if let Some(index) = pick(ui, "card scene", &current, &labels, 110.0) {
                    shell.studio_mut().set_preview_scene(scenes[index]);
                }
            });
            ui.horizontal(|ui| {
                tag(ui, "Card moment", LABEL_WIDTH, tokens::MUTED);
                let poster = recipe.poster.unwrap_or(cutlass_effects::DEFAULT_POSTER);
                if let Some(typed) = number(ui, &mut view.fields, "poster", poster, NUMBER_WIDTH) {
                    shell.studio_mut().set_poster(Some(typed.clamp(0.0, 60.0)));
                }
                faint(ui, "s");
                if ui
                    .add(
                        ui::Button::ghost("Use the playhead")
                            .tip("The card's still shows this moment"),
                    )
                    .clicked()
                {
                    shell.studio_mut().set_poster(Some(view.clock as f32));
                }
            });
        }
        Doc::Transition(recipe) => {
            ui.horizontal(|ui| {
                tag(ui, "Lasts", LABEL_WIDTH, tokens::MUTED);
                let seconds = recipe.duration().0 as f32 / 1_000_000.0;
                if let Some(typed) = number(ui, &mut view.fields, "duration", seconds, NUMBER_WIDTH)
                {
                    shell.studio_mut().set_duration(Some(typed));
                }
                faint(ui, "s when dropped on a cut");
            });
            ui.horizontal(|ui| {
                tag(ui, "Card scenes", LABEL_WIDTH, tokens::MUTED);
                let shown = match recipe.preview.as_slice() {
                    [from, to] => format!("{} to {}", param_label(from), param_label(to)),
                    _ => "The app's own pair".to_owned(),
                };
                faint(ui, &shown);
                if let Backdrop::Scene(from) = &view.backdrop
                    && ui
                        .add(
                            ui::Button::ghost("Use these")
                                .tip("The card plays between the scenes shown now"),
                        )
                        .clicked()
                {
                    shell.studio_mut().set_preview_pair(from, &view.to);
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{block_label, param_label, short};
    use cutlass_effects::Block;

    #[test]
    fn a_graph_fits_the_curve_inside_the_values_own_range() {
        // Scale runs from 0.1 to 4; a punch from 1 to 1.35 fills the graph.
        let (bottom, top) = super::fitted(1.0, 1.35, 0.1, 4.0);
        assert!(bottom < 1.0 && bottom > 0.7, "{bottom}");
        assert!(top > 1.35 && top < 1.7, "{top}");
        // One fixed value still has room each side.
        let (bottom, top) = super::fitted(1.0, 1.0, 0.1, 4.0);
        assert!(bottom < 1.0 && top > 1.0 && top - bottom < 1.0);
        // The fit never leaves the range, and a curve over all of it shows
        // all of it.
        assert_eq!(super::fitted(0.0, 0.1, 0.0, 0.1), (0.0, 0.1));
        let (bottom, top) = super::fitted(0.0, 0.0, 0.0, 0.1);
        assert!(bottom == 0.0 && top > 0.0 && top <= 0.1);
    }

    #[test]
    fn names_and_numbers_read_as_words() {
        assert_eq!(block_label(Block::RgbSplit), "RGB split");
        assert_eq!(block_label(Block::ZoomBlur), "Zoom blur");
        assert_eq!(block_label(Block::Blur), "Blur");
        assert_eq!(param_label("radius"), "Radius");
        assert_eq!(param_label("soft_edge"), "Soft edge");
        assert_eq!(param_label(""), "");
        assert_eq!(short(0.02), "0.02");
        assert_eq!(short(1.0), "1");
        assert_eq!(short(12.34), "12.3");
        assert_eq!(short(180.0), "180");
        assert_eq!(short(-0.5), "-0.5");
    }
}
