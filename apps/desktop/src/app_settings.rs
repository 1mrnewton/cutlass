//! App settings. The provider and a cloud key stay on the app for the session.
//! They are not project fields, and the key is not written to disk.
//! `docs/app-settings.md`.

use eframe::egui::{
    self, Align2, Color32, FontFamily, FontId, Frame, Id, Key, Margin, Modifiers, Rect, RichText,
    ScrollArea, Sense, TextEdit, UiBuilder, pos2, vec2,
};

use crate::theme;
use crate::words::{self, Language, Phrase};

const DIALOG: Color32 = Color32::from_rgb(0x32, 0x32, 0x32);
const CLOSE: Color32 = Color32::from_rgb(0xff, 0x5f, 0x57);

/// Where a model would run. None of these runs yet. `docs/roadmap.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Runtime {
    /// OS frameworks, the same order as the codecs.
    Device,
    /// A Rust runtime inside the app.
    InApp,
    Cloud,
}

const RUNTIMES: [Runtime; 3] = [Runtime::Device, Runtime::InApp, Runtime::Cloud];

/// The provider, and a cloud key while that provider is selected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AppSettings {
    runtime: Runtime,
    key: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            runtime: Runtime::Device,
            key: String::new(),
        }
    }
}

/// What the dialog did this frame.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Open,
    Closed,
    Saved(AppSettings),
}

/// A draft. Dropping it discards the edits.
pub(crate) struct Draft {
    runtime: Runtime,
    key: String,
}

impl Draft {
    pub(crate) fn open(settings: &AppSettings) -> Self {
        Self {
            runtime: settings.runtime,
            key: settings.key.clone(),
        }
    }

    fn saved(&self) -> AppSettings {
        AppSettings {
            runtime: self.runtime,
            key: if self.runtime == Runtime::Cloud {
                self.key.clone()
            } else {
                String::new()
            },
        }
    }
}

/// Draws the dialog. Nothing is written to a file.
pub(crate) fn show(ui: &egui::Ui, draft: &mut Draft) -> Outcome {
    let mut outcome = Outcome::Open;
    let screen = ui.ctx().content_rect();
    egui::Area::new(Id::unique("app-settings"))
        .order(egui::Order::Foreground)
        .fixed_pos(screen.min)
        .movable(false)
        .constrain(false)
        .fade_in(false)
        .accessible_name(words::of(ui, Phrase::Settings))
        .role(egui::accesskit::Role::Dialog)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(screen);
            ui.set_min_size(screen.size());
            let size = vec2(440.0, 320.0).min(screen.size() - vec2(48.0, 48.0));
            let size = vec2(size.x.max(360.0), size.y.max(240.0));
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
            if let Some(next) = paint_foot(ui, foot, draft) {
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
        words::of(ui, Phrase::Settings),
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
        .id_salt("app-settings")
        .show(&mut column, |ui| {
            ui.set_min_width(ui.available_width());
            let language = words::active(ui);
            choice(ui, &mut draft.runtime, language);
            if draft.runtime == Runtime::Cloud {
                key_row(ui, &mut draft.key);
            }
        });
    ui.set_clip_rect(clip);
}

fn paint_foot(ui: &mut egui::Ui, foot: Rect, draft: &Draft) -> Option<Outcome> {
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
        return Some(Outcome::Saved(draft.saved()));
    }
    cancel_response.clicked().then_some(Outcome::Closed)
}

fn choice(ui: &mut egui::Ui, runtime: &mut Runtime, language: Language) {
    labeled(ui, words::of(ui, Phrase::Provider), |ui, width| {
        egui::ComboBox::from_id_salt(("provider", "app"))
            .selected_text(runtime_label(*runtime, language))
            .width(width)
            .show_ui(ui, |ui| {
                for option in RUNTIMES {
                    ui.selectable_value(runtime, option, runtime_label(option, language));
                }
            });
    });
}

fn key_row(ui: &mut egui::Ui, key: &mut String) {
    labeled(ui, words::of(ui, Phrase::ApiKey), |ui, width| {
        TextEdit::singleline(key)
            .id(Id::unique("app-settings-key"))
            .password(true)
            .frame(field_frame())
            .desired_width(width)
            .show(ui);
    });
}

fn runtime_label(runtime: Runtime, language: Language) -> &'static str {
    let phrase = match runtime {
        Runtime::Device => Phrase::OnThisDevice,
        Runtime::InApp => Phrase::InTheApp,
        Runtime::Cloud => Phrase::InTheCloud,
    };
    words::say(language, phrase)
}

fn labeled(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui, f32)) {
    let width = ui.available_width();
    let field = (width - 140.0).max(80.0);
    ui.horizontal(|ui| {
        ui.set_min_width(width);
        ui.spacing_mut().item_spacing.x = 16.0;
        let name = ui.add_sized(
            vec2(124.0, 32.0),
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

#[cfg(test)]
mod tests {
    use super::{AppSettings, Draft, Outcome, Runtime};

    #[test]
    fn a_device_choice_drops_the_key_and_a_cloud_choice_keeps_it() {
        let draft = Draft::open(&AppSettings::default());
        let saved = draft.saved();
        assert_eq!(saved.runtime, Runtime::Device);
        assert_eq!(saved.key, "");

        let cloud = Draft {
            runtime: Runtime::Cloud,
            key: "test-key".to_string(),
        };
        let saved = cloud.saved();
        assert_eq!(saved.runtime, Runtime::Cloud);
        assert_eq!(saved.key, "test-key");

        let elsewhere = Draft {
            runtime: Runtime::InApp,
            key: "test-key".to_string(),
        };
        assert_eq!(elsewhere.saved().key, "");
        assert_eq!(elsewhere.saved().runtime, Runtime::InApp);
    }

    #[test]
    fn save_keeps_a_cloud_key_and_cancel_drops_the_draft() {
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::words::install(&ctx, crate::words::Language::English);
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
                    outcome = super::show(ui, draft);
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

        let mut draft = Draft::open(&AppSettings::default());
        // The dialog's first frame sizes itself and is not in the tree yet.
        pass(vec![], &mut draft);
        let (tree, _) = pass(vec![], &mut draft);
        assert!(node(&tree, "API key").is_none());

        draft.runtime = Runtime::Cloud;
        draft.key = "test-key".to_string();
        let (tree, _) = pass(vec![], &mut draft);
        assert!(node(&tree, "API key").is_some());
        let save = center(&tree, "Save");
        pass(
            vec![eframe::egui::Event::PointerMoved(save), press(save)],
            &mut draft,
        );
        let (_, outcome) = pass(vec![release(save)], &mut draft);
        match outcome {
            Outcome::Saved(settings) => {
                assert_eq!(settings.runtime, Runtime::Cloud);
                assert_eq!(settings.key, "test-key");
            }
            Outcome::Open | Outcome::Closed => panic!("save should keep the key"),
        }

        let (tree, _) = pass(vec![], &mut draft);
        let cancel = center(&tree, "Cancel");
        pass(
            vec![eframe::egui::Event::PointerMoved(cancel), press(cancel)],
            &mut draft,
        );
        let (_, outcome) = pass(vec![release(cancel)], &mut draft);
        assert_eq!(outcome, Outcome::Closed);
    }

    fn node<'a>(
        tree: &'a eframe::egui::accesskit::TreeUpdate,
        label: &str,
    ) -> Option<&'a eframe::egui::accesskit::Node> {
        tree.nodes
            .iter()
            .find(|(_, node)| node.label() == Some(label) || node.value() == Some(label))
            .map(|(_, node)| node)
    }

    fn center(tree: &eframe::egui::accesskit::TreeUpdate, label: &str) -> eframe::egui::Pos2 {
        let node = node(tree, label).unwrap_or_else(|| panic!("missing {label}"));
        let rect = node.bounds().expect("bounds");
        eframe::egui::pos2(
            ((rect.x0 + rect.x1) / 2.0) as f32,
            ((rect.y0 + rect.y1) / 2.0) as f32,
        )
    }
}
