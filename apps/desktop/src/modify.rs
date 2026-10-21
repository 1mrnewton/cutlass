//! Project settings. Save keeps them on the project and does not write a file.
//! `docs/inspector.md`.

use eframe::egui::text::{CCursor, CCursorRange};
use eframe::egui::{
    self, Align2, Color32, FontFamily, FontId, Frame, Id, Key, Margin, Modifiers, Rect, RichText,
    ScrollArea, Sense, TextEdit, UiBuilder, pos2, vec2,
};

use cutlass_project::{AspectRatio, Color, ColorSpace, Command, FrameRate, Project, SequenceRate};

use crate::inspector;
use crate::theme;
use crate::words::{self, Phrase};

const DIALOG: Color32 = Color32::from_rgb(0x32, 0x32, 0x32);
const CLOSE: Color32 = Color32::from_rgb(0xff, 0x5f, 0x57);

const RATIOS: [AspectRatio; 6] = [
    AspectRatio::Original,
    AspectRatio::R16x9,
    AspectRatio::R9x16,
    AspectRatio::Square,
    AspectRatio::R4x5,
    AspectRatio::R4x3,
];

const RATES: [FrameRate; 8] = [
    FrameRate::Fps23976,
    FrameRate::Fps24,
    FrameRate::Fps25,
    FrameRate::Fps2997,
    FrameRate::Fps30,
    FrameRate::Fps50,
    FrameRate::Fps5994,
    FrameRate::Fps60,
];

const SPACES: [ColorSpace; 3] = [
    ColorSpace::Rec709,
    ColorSpace::Rec2020Hlg,
    ColorSpace::Rec2100Pq,
];

/// What the dialog did this frame.
#[derive(Debug)]
pub(crate) enum Outcome {
    Open,
    Closed,
    Saved(Vec<Command>),
}

/// A draft of the project settings. Dropping it discards the edits.
pub(crate) struct Draft {
    name: String,
    width: String,
    height: String,
    ratio: AspectRatio,
    rate: FrameRate,
    drop: bool,
    color: ColorSpace,
    red: String,
    green: String,
    blue: String,
    armed: bool,
    refused: bool,
}

impl Draft {
    pub(crate) fn open(project: &Project) -> Self {
        let settings = project.settings();
        let (red, green, blue) = settings.background().channels();
        Self {
            name: project.name().to_string(),
            width: settings.width().to_string(),
            height: settings.height().to_string(),
            ratio: settings.ratio(),
            rate: settings.rate().rate(),
            drop: settings.rate().drop_frame(),
            color: settings.color_space(),
            red: red.to_string(),
            green: green.to_string(),
            blue: blue.to_string(),
            armed: false,
            refused: false,
        }
    }
}

/// The project panel's Modify button. A click asks the shell to open the dialog.
pub(crate) fn button(ui: &mut egui::Ui) {
    let label = words::of(ui, Phrase::Modify);
    let font = FontId::new(13.0, FontFamily::Proportional);
    let width = words::control_width(ui, label, font, 24.0, 88.0);
    let response = ui.add_sized(
        vec2(width, 28.0),
        egui::Button::new(RichText::new(label).size(13.0).color(theme::EXPORT_INK))
            .fill(theme::EXPORT)
            .corner_radius(8),
    );
    crate::tip::name(&response, label);
    if response.clicked() {
        ui.ctx().data_mut(|data| data.insert_temp(ask_id(), true));
    }
}

/// Takes the Modify click, if there was one this frame.
pub(crate) fn take_ask(ctx: &egui::Context) -> bool {
    let asked = ctx.data(|data| data.get_temp(ask_id()).unwrap_or(false));
    if asked {
        ctx.data_mut(|data| data.remove::<bool>(ask_id()));
    }
    asked
}

fn ask_id() -> Id {
    Id::unique("project-settings-ask")
}

/// Draws the dialog. Nothing is written to a file.
pub(crate) fn show(ui: &egui::Ui, project: &Project, draft: &mut Draft) -> Outcome {
    let mut outcome = Outcome::Open;
    let screen = ui.ctx().content_rect();
    egui::Area::new(Id::unique("project-settings"))
        .order(egui::Order::Foreground)
        .fixed_pos(screen.min)
        .movable(false)
        .constrain(false)
        .fade_in(false)
        .accessible_name(words::of(ui, Phrase::Details))
        .role(egui::accesskit::Role::Dialog)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(screen);
            ui.set_min_size(screen.size());
            let size = vec2(480.0, 560.0).min(screen.size() - vec2(48.0, 48.0));
            let size = vec2(size.x.max(360.0), size.y.max(280.0));
            let dialog = Rect::from_center_size(screen.center(), size);
            ui.painter()
                .rect_filled(screen, 0.0, Color32::from_black_alpha(140));
            let scrim = ui.interact(screen, ui.scope_id().with("scrim"), Sense::click());
            let _plate = ui.interact(dialog, ui.scope_id().with("plate"), Sense::click());
            if scrim.clicked() {
                outcome = Outcome::Closed;
            }
            ui.painter().rect_filled(dialog, 12.0, DIALOG);
            let head = Rect::from_min_size(dialog.min, vec2(dialog.width(), 40.0));
            let foot = Rect::from_min_max(
                pos2(dialog.left(), dialog.bottom() - 52.0),
                dialog.right_bottom(),
            );
            let body = Rect::from_min_max(head.left_bottom(), foot.right_top());
            ui.painter().rect_filled(body, 0.0, theme::BAR);
            if paint_head(ui, head) {
                outcome = Outcome::Closed;
            }
            paint_body(ui, body, draft);
            if let Some(next) = paint_foot(ui, foot, project, draft) {
                outcome = next;
            }
        });
    if matches!(outcome, Outcome::Open)
        && ui.input_mut(|input| input.consume_key(Modifiers::NONE, Key::Escape))
    {
        outcome = Outcome::Closed;
    }
    outcome
}

fn paint_head(ui: &mut egui::Ui, head: Rect) -> bool {
    ui.painter().text(
        head.center(),
        Align2::CENTER_CENTER,
        words::of(ui, Phrase::Details),
        FontId::new(13.0, FontFamily::Proportional),
        theme::INK,
    );
    let hit = Rect::from_center_size(pos2(head.left() + 20.0, head.center().y), vec2(24.0, 24.0));
    let response = ui.interact(hit, ui.scope_id().with("close"), Sense::click());
    ui.painter().circle_filled(hit.center(), 6.0, CLOSE);
    crate::tip::show(&response, words::of(ui, Phrase::Close));
    response.clicked()
}

fn paint_body(ui: &mut egui::Ui, body: Rect, draft: &mut Draft) {
    let inner = Rect::from_min_max(body.min + vec2(24.0, 16.0), body.max - vec2(24.0, 8.0));
    if inner.width() < 80.0 || inner.height() < 40.0 {
        return;
    }
    let clip = ui.clip_rect();
    ui.set_clip_rect(body);
    let mut column = ui.new_child(UiBuilder::new().max_rect(inner));
    column.set_clip_rect(inner.intersect(body));
    ScrollArea::vertical()
        .id_salt("project-settings")
        .show(&mut column, |ui| {
            ui.set_min_width(ui.available_width());
            name_row(ui, draft);
            number_row(ui, Phrase::Width, "project-width", &mut draft.width);
            number_row(ui, Phrase::Height, "project-height", &mut draft.height);
            let language = words::active(ui);
            choice(
                ui,
                "ratio",
                words::of(ui, Phrase::Ratio),
                &mut draft.ratio,
                &RATIOS,
                |ratio| inspector::ratio(ratio, language),
            );
            let previous = draft.rate;
            choice(
                ui,
                "rate",
                words::of(ui, Phrase::FrameRate),
                &mut draft.rate,
                &RATES,
                inspector::frame_rate,
            );
            settle_rate(draft, previous);
            if draft.rate.allows_drop_frame() {
                drop_row(ui, draft);
            }
            choice(
                ui,
                "color",
                words::of(ui, Phrase::ColorSpace),
                &mut draft.color,
                &SPACES,
                inspector::color_space,
            );
            channels(ui, draft);
            if draft.refused {
                ui.add_space(8.0);
                let text = words::of(ui, Phrase::InvalidSettings);
                let response = ui.label(RichText::new(text).size(12.0).color(theme::INK));
                crate::tip::alert(&response, text);
            }
        });
    ui.set_clip_rect(clip);
}

fn paint_foot(
    ui: &mut egui::Ui,
    foot: Rect,
    project: &Project,
    draft: &mut Draft,
) -> Option<Outcome> {
    let save_label = words::of(ui, Phrase::Save);
    let cancel_label = words::of(ui, Phrase::Cancel);
    let font = FontId::new(13.0, FontFamily::Proportional);
    let save_w = words::control_width(ui, save_label, font.clone(), 24.0, 72.0);
    let cancel_w = words::control_width(ui, cancel_label, font, 20.0, 64.0);
    let save = Rect::from_center_size(
        pos2(foot.right() - 16.0 - save_w / 2.0, foot.center().y),
        vec2(save_w, 26.0),
    );
    let cancel = Rect::from_center_size(
        pos2(save.left() - 8.0 - cancel_w / 2.0, foot.center().y),
        vec2(cancel_w, 26.0),
    );
    let cancel_response = ui.interact(cancel, ui.scope_id().with("cancel"), Sense::click());
    crate::tip::name(&cancel_response, cancel_label);
    ui.painter().text(
        cancel.center(),
        Align2::CENTER_CENTER,
        cancel_label,
        FontId::new(13.0, FontFamily::Proportional),
        if cancel_response.hovered() {
            theme::INK
        } else {
            theme::MUTED
        },
    );
    let save_response = ui.interact(save, ui.scope_id().with("save"), Sense::click());
    crate::tip::name(&save_response, save_label);
    ui.painter().rect_filled(save, 8.0, theme::EXPORT);
    ui.painter().text(
        save.center(),
        Align2::CENTER_CENTER,
        save_label,
        FontId::new(13.0, FontFamily::Proportional),
        theme::EXPORT_INK,
    );
    if save_response.clicked() {
        return Some(match changes(project, draft) {
            Ok(commands) => Outcome::Saved(commands),
            Err(Refused) => {
                draft.refused = true;
                Outcome::Open
            }
        });
    }
    cancel_response.clicked().then_some(Outcome::Closed)
}

fn name_row(ui: &mut egui::Ui, draft: &mut Draft) {
    labeled(ui, words::of(ui, Phrase::Name), |ui, width| {
        let output = TextEdit::singleline(&mut draft.name)
            .id(Id::unique("project-settings-name"))
            .frame(field_frame())
            .desired_width(width)
            .show(ui);
        if !draft.armed {
            output.response.request_focus();
            if let Some(mut state) = TextEdit::load_state(ui.ctx(), output.response.id) {
                let end = CCursor::new(draft.name.chars().count());
                state
                    .cursor
                    .set_char_range(Some(CCursorRange::two(CCursor::new(0), end)));
                TextEdit::store_state(ui.ctx(), output.response.id, state);
            }
            draft.armed = true;
        }
    });
}

fn number_row(ui: &mut egui::Ui, phrase: Phrase, salt: &str, text: &mut String) {
    labeled(ui, words::of(ui, phrase), |ui, width| {
        TextEdit::singleline(text)
            .id(Id::unique(salt))
            .frame(field_frame())
            .desired_width(width)
            .show(ui);
    });
}

fn drop_row(ui: &mut egui::Ui, draft: &mut Draft) {
    labeled(ui, words::of(ui, Phrase::DropFrame), |ui, width| {
        let word = words::of(ui, if draft.drop { Phrase::On } else { Phrase::Off });
        let response = ui.add_sized(
            vec2(width, 32.0),
            egui::Button::new(RichText::new(word).size(13.0).color(theme::INK))
                .fill(theme::PANEL)
                .corner_radius(10),
        );
        crate::tip::show_switch(
            &response,
            words::of(ui, Phrase::DropFrame),
            draft.drop,
            true,
        );
        if response.clicked() {
            draft.drop = !draft.drop;
        }
    });
}

fn channels(ui: &mut egui::Ui, draft: &mut Draft) {
    labeled(ui, words::of(ui, Phrase::Background), |ui, width| {
        let each = ((width - 16.0) / 3.0).max(36.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            channel_field(ui, "project-red", &mut draft.red, each);
            channel_field(ui, "project-green", &mut draft.green, each);
            channel_field(ui, "project-blue", &mut draft.blue, each);
        });
    });
}

fn channel_field(ui: &mut egui::Ui, salt: &str, text: &mut String, width: f32) {
    TextEdit::singleline(text)
        .id(Id::unique(salt))
        .frame(field_frame())
        .desired_width(width)
        .show(ui);
}

fn choice<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    salt: &str,
    label: &str,
    current: &mut T,
    options: &[T],
    text: impl Fn(T) -> &'static str,
) {
    labeled(ui, label, |ui, width| {
        egui::ComboBox::from_id_salt((salt, "project"))
            .selected_text(text(*current))
            .width(width)
            .show_ui(ui, |ui| {
                for option in options {
                    ui.selectable_value(current, *option, text(*option));
                }
            });
    });
}

fn labeled(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui, f32)) {
    let width = ui.available_width();
    let field = (width - 124.0).max(80.0);
    ui.horizontal(|ui| {
        ui.set_min_width(width);
        ui.spacing_mut().item_spacing.x = 16.0;
        let name = ui.add_sized(
            vec2(108.0, 32.0),
            egui::Label::new(RichText::new(label).size(12.0).color(theme::MUTED)),
        );
        ui.label_inputs_by(name.id, |ui| add(ui, field));
    });
    ui.add_space(8.0);
}

fn field_frame() -> Frame {
    Frame::NONE
        .fill(theme::PANEL)
        .corner_radius(10)
        .inner_margin(Margin::symmetric(12, 6))
}

/// A newly chosen rate starts with that rate's drop-frame default.
fn settle_rate(draft: &mut Draft, previous: FrameRate) {
    if draft.rate != previous {
        draft.drop = draft.rate.allows_drop_frame();
    }
}

#[derive(Debug)]
struct Refused;

fn changes(project: &Project, draft: &Draft) -> Result<Vec<Command>, Refused> {
    let width = whole(&draft.width)?;
    let height = whole(&draft.height)?;
    if width == 0 || height == 0 {
        return Err(Refused);
    }
    let red = channel(&draft.red)?;
    let green = channel(&draft.green)?;
    let blue = channel(&draft.blue)?;
    let drop = draft.drop && draft.rate.allows_drop_frame();
    let rate = SequenceRate::new(draft.rate, drop).expect("drop frame follows the rate");
    let settings = project.settings();
    let mut commands = Vec::new();
    if draft.name != project.name() {
        commands.push(Command::Rename(draft.name.clone()));
    }
    if width != settings.width() || height != settings.height() || draft.ratio != settings.ratio() {
        commands.push(Command::SetCanvas {
            width,
            height,
            ratio: draft.ratio,
        });
    }
    if rate != settings.rate() {
        commands.push(Command::SetRate(rate));
    }
    if draft.color != settings.color_space() {
        commands.push(Command::SetColorSpace(draft.color));
    }
    if (red, green, blue) != settings.background().channels() {
        commands.push(Command::SetBackground(Color::new(red, green, blue)));
    }
    Ok(commands)
}

fn whole(text: &str) -> Result<u32, Refused> {
    text.trim().parse().map_err(|_| Refused)
}

fn channel(text: &str) -> Result<u8, Refused> {
    text.trim().parse().map_err(|_| Refused)
}

#[cfg(test)]
mod tests {
    use cutlass_project::{
        AspectRatio, Color, ColorSpace, Command, FrameRate, History, Project, ProjectId,
        SequenceRate,
    };

    use super::{Draft, Outcome, changes, settle_rate};

    fn empty() -> Project {
        Project::new(ProjectId::from_bytes([3; 16]), "Untitled")
    }

    #[test]
    fn save_writes_only_what_changed_and_a_bad_value_is_refused() {
        let mut project = empty();
        let mut draft = Draft::open(&project);
        assert!(
            changes(&project, &draft)
                .expect("the open draft")
                .is_empty()
        );
        draft.drop = true;
        assert!(
            changes(&project, &draft)
                .expect("drop on 30 is ignored")
                .is_empty()
        );

        draft.name = "Cut".to_string();
        draft.width = "1280".to_string();
        draft.height = "720".to_string();
        draft.ratio = AspectRatio::R16x9;
        draft.rate = FrameRate::Fps24;
        draft.color = ColorSpace::Rec2100Pq;
        draft.red = "8".to_string();
        draft.green = "16".to_string();
        draft.blue = "24".to_string();
        let commands = changes(&project, &draft).expect("valid");
        assert_eq!(
            commands,
            vec![
                Command::Rename("Cut".to_string()),
                Command::SetCanvas {
                    width: 1280,
                    height: 720,
                    ratio: AspectRatio::R16x9,
                },
                Command::SetRate(SequenceRate::new(FrameRate::Fps24, false).expect("24")),
                Command::SetColorSpace(ColorSpace::Rec2100Pq),
                Command::SetBackground(Color::new(8, 16, 24)),
            ]
        );
        History::new()
            .apply(&mut project, &commands)
            .expect("the settings land");
        assert_eq!(project.name(), "Cut");
        assert_eq!(project.settings().width(), 1280);
        assert_eq!(project.settings().background().channels(), (8, 16, 24));

        draft.rate = FrameRate::Fps2997;
        draft.drop = false;
        assert_eq!(
            changes(&project, &draft).expect("29.97"),
            vec![Command::SetRate(
                SequenceRate::new(FrameRate::Fps2997, false).expect("29.97")
            )]
        );

        draft.width = "0".to_string();
        assert!(changes(&project, &draft).is_err());
        draft.width = "1280".to_string();
        draft.blue = "256".to_string();
        assert!(changes(&project, &draft).is_err());
        draft.blue = "no".to_string();
        assert!(changes(&project, &draft).is_err());
    }

    #[test]
    fn a_new_rate_sets_drop_frame_from_that_rate() {
        let mut draft = Draft::open(&empty());
        settle_rate(&mut draft, FrameRate::Fps30);
        assert!(!draft.drop);
        draft.rate = FrameRate::Fps2997;
        settle_rate(&mut draft, FrameRate::Fps30);
        assert!(draft.drop);
        draft.drop = false;
        settle_rate(&mut draft, FrameRate::Fps2997);
        assert!(!draft.drop);
        draft.rate = FrameRate::Fps24;
        settle_rate(&mut draft, FrameRate::Fps2997);
        assert!(!draft.drop);
    }

    #[test]
    fn save_keeps_a_new_name_and_cancel_or_a_zero_size_does_not() {
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::words::install(&ctx, crate::words::Language::English);
        let project = empty();
        let mut draft = Draft::open(&project);
        draft.name = "Cut".to_string();
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(1280.0, 800.0),
        );
        let mut time = 0.0;
        let mut pass = |events: Vec<eframe::egui::Event>, draft: &mut Draft| {
            time += 0.05;
            let mut outcome = Outcome::Open;
            let mut output = ctx.run_ui(
                eframe::egui::RawInput {
                    screen_rect: Some(screen),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ui| {
                    outcome = super::show(ui, &project, draft);
                },
            );
            let tree = output.platform_output.accesskit_update.take();
            output.textures_delta.clear();
            (tree.expect("the tree is built"), outcome)
        };
        let press = |point| eframe::egui::Event::PointerButton {
            pos: point,
            button: eframe::egui::PointerButton::Primary,
            pressed: true,
            modifiers: eframe::egui::Modifiers::NONE,
        };
        let release = |point| eframe::egui::Event::PointerButton {
            pos: point,
            button: eframe::egui::PointerButton::Primary,
            pressed: false,
            modifiers: eframe::egui::Modifiers::NONE,
        };

        // The dialog's first frame sizes itself and is not in the tree yet.
        pass(vec![], &mut draft);
        let (tree, _) = pass(vec![], &mut draft);
        let save = center(&tree, "Save");
        pass(
            vec![eframe::egui::Event::PointerMoved(save), press(save)],
            &mut draft,
        );
        let (_, outcome) = pass(vec![release(save)], &mut draft);
        match outcome {
            Outcome::Saved(commands) => {
                assert_eq!(commands, vec![Command::Rename("Cut".to_string())]);
            }
            Outcome::Open | Outcome::Closed => panic!("save should keep the name"),
        }

        let (tree, _) = pass(vec![], &mut draft);
        let cancel = center(&tree, "Cancel");
        pass(
            vec![eframe::egui::Event::PointerMoved(cancel), press(cancel)],
            &mut draft,
        );
        let (_, outcome) = pass(vec![release(cancel)], &mut draft);
        assert!(matches!(outcome, Outcome::Closed));

        draft.width = "0".to_string();
        let (tree, _) = pass(vec![], &mut draft);
        let save = center(&tree, "Save");
        pass(
            vec![eframe::egui::Event::PointerMoved(save), press(save)],
            &mut draft,
        );
        let (_, outcome) = pass(vec![release(save)], &mut draft);
        assert!(matches!(outcome, Outcome::Open));
        assert!(draft.refused);
    }

    fn center(tree: &eframe::egui::accesskit::TreeUpdate, label: &str) -> eframe::egui::Pos2 {
        let (_, node) = tree
            .nodes
            .iter()
            .find(|(_, node)| node.label() == Some(label) || node.value() == Some(label))
            .unwrap_or_else(|| {
                let labels: Vec<_> = tree
                    .nodes
                    .iter()
                    .filter_map(|(_, node)| node.label().or(node.value()))
                    .collect();
                panic!("missing {label} in {labels:?}");
            });
        let rect = node.bounds().expect("bounds");
        eframe::egui::pos2(
            ((rect.x0 + rect.x1) / 2.0) as f32,
            ((rect.y0 + rect.y1) / 2.0) as f32,
        )
    }
}
