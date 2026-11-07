//! The assistant panel: the conversation, each tool call as a step, the
//! command box, and Stop. It takes the inspector's place while open.

use egui::{
    Align, Color32, FontId, Frame, Key, Layout, Margin, Rect, RichText, TextEdit, Ui, Vec2,
};

use crate::assistant::{Assistant, Entry};
use crate::shell::Shell;
use crate::{tokens, ui};

const FAILED: Color32 = Color32::from_rgb(0xff, 0x7a, 0x70);
const DONE: Color32 = Color32::from_rgb(0x2a, 0xd4, 0xe0);

/// What the box says before anything is typed.
pub const HINT: &str = "Ask Cutlass to edit…";

/// The panel in `rect`.
pub fn panel(shell: &mut Shell, ui: &mut Ui, assistant: &mut Assistant, rect: Rect) {
    let inner = rect.shrink(12.0);
    let mut column = ui.new_child(egui::UiBuilder::new().max_rect(inner));
    let ui = &mut column;
    ui.horizontal(|ui| {
        ui.label(
            RichText::new("Assistant")
                .size(14.0)
                .strong()
                .color(tokens::TEXT),
        );
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui
                .add(ui::Button::icon("close"))
                .on_hover_text("Close the assistant")
                .clicked()
            {
                assistant.open = false;
            }
            if ui
                .add_enabled(
                    !assistant.entries.is_empty(),
                    ui::Button::new("New chat").text_size(12.0),
                )
                .clicked()
            {
                assistant.clear();
            }
        });
    });
    let model = &shell.settings().model;
    let source = if model.local {
        format!("{} on this machine", model.local_name)
    } else {
        "The local model is off".to_owned()
    };
    ui.label(RichText::new(source).size(11.0).color(tokens::FAINT));
    ui.add_space(8.0);

    let box_height = 92.0;
    let list_height = (ui.available_height() - box_height).max(60.0);
    ui::Scroll::vertical("assistant-entries")
        .max_height(list_height)
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            if assistant.entries.is_empty() {
                ui.label(
                    RichText::new(
                        "Cutlass reads the project, edits it with its tools, and each edit is one undo step.",
                    )
                    .size(12.0)
                    .color(tokens::MUTED),
                );
            }
            let mut decided = None;
            for (index, entry) in assistant.entries.iter().enumerate() {
                if let Some(allow) = entry_row(ui, index, entry) {
                    decided = Some(allow);
                }
                ui.add_space(6.0);
            }
            if let Some(allow) = decided {
                assistant.decide(shell, allow);
            }
            if assistant.busy() && !assistant.asking() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new("Working…").size(12.0).color(tokens::MUTED));
                });
            }
        });
    ui.add_space(6.0);
    command_box(shell, ui, assistant);
}

/// One line of the conversation; an open question's answer, when clicked.
fn entry_row(ui: &mut Ui, index: usize, entry: &Entry) -> Option<bool> {
    let mut decided = None;
    match entry {
        Entry::Ask { question, allowed } => {
            ui::Surface::outline(DONE).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new(question).size(13.0).color(tokens::TEXT));
                ui.add_space(4.0);
                match allowed {
                    None => {
                        ui.horizontal(|ui| {
                            if ui.add(ui::Button::new("Allow").text_size(12.0)).clicked() {
                                decided = Some(true);
                            }
                            if ui
                                .add(ui::Button::new("Don't allow").text_size(12.0))
                                .clicked()
                            {
                                decided = Some(false);
                            }
                        });
                    }
                    Some(true) => {
                        ui.label(RichText::new("Allowed").size(11.0).color(DONE));
                    }
                    Some(false) => {
                        ui.label(RichText::new("Not allowed").size(11.0).color(FAILED));
                    }
                }
            });
        }
        Entry::Asked(text) => {
            ui.with_layout(Layout::top_down(Align::Max), |ui| {
                ui::Surface::raised()
                    .padding(Margin::symmetric(10, 6))
                    .show(ui, |ui| {
                        ui.set_max_width(ui.available_width() * 0.85);
                        ui.label(RichText::new(text).size(13.0).color(tokens::TEXT));
                    });
            });
        }
        Entry::Said(text) => {
            ui.label(RichText::new(text).size(13.0).color(tokens::TEXT));
        }
        Entry::Notice(text) => {
            ui.label(
                RichText::new(text)
                    .size(12.0)
                    .italics()
                    .color(tokens::MUTED),
            );
        }
        Entry::Step {
            tool,
            input,
            result,
            failed,
        } => {
            let (mark, colour) = match (result, failed) {
                (None, _) => ("…", tokens::MUTED),
                (Some(_), true) => ("✕", FAILED),
                (Some(_), false) => ("✓", DONE),
            };
            let title = format!("{mark} {tool} {}", short(input, 40));
            egui::CollapsingHeader::new(RichText::new(title).size(12.0).color(colour))
                .id_salt(("assistant-step", index))
                .default_open(false)
                .show(ui, |ui| {
                    if !input.is_empty() {
                        ui.label(
                            RichText::new(short(input, 600))
                                .font(FontId::monospace(11.0))
                                .color(tokens::MUTED),
                        );
                    }
                    if let Some(result) = result {
                        ui.label(
                            RichText::new(short(result, 1200))
                                .font(FontId::monospace(11.0))
                                .color(if *failed { FAILED } else { tokens::TEXT }),
                        );
                    }
                });
        }
    }
    decided
}

fn command_box(shell: &Shell, ui: &mut Ui, assistant: &mut Assistant) {
    ui::Surface::raised()
        .stroke(tokens::LANE_LINE)
        .padding(8)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            let response = ui.add(
                TextEdit::multiline(&mut assistant.draft)
                    .id_salt("assistant-box")
                    .hint_text(HINT)
                    .desired_rows(2)
                    .desired_width(f32::INFINITY)
                    .frame(Frame::NONE),
            );
            // Enter sends; Shift+Enter starts a new line.
            let enter = response.has_focus()
                && ui.input(|input| input.key_pressed(Key::Enter) && !input.modifiers.shift);
            if enter {
                let trimmed = assistant.draft.trim_end_matches('\n').to_owned();
                assistant.draft = trimmed;
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if assistant.busy() {
                    if ui
                        .add(
                            ui::Button::new("Stop")
                                .text_size(12.0)
                                .min_size(Vec2::new(56.0, 24.0)),
                        )
                        .clicked()
                    {
                        assistant.stop();
                    }
                } else {
                    let ready = !assistant.draft.trim().is_empty() && shell.draft().is_some();
                    let send = ui
                        .add_enabled(
                            ready,
                            ui::Button::primary("Send")
                                .text_size(12.0)
                                .min_size(Vec2::new(56.0, 24.0)),
                        )
                        .clicked();
                    if (send || enter) && ready {
                        assistant.send(shell.settings());
                        response.request_focus();
                    }
                }
            });
        });
}

/// `text` on one line, cut to `most` characters.
fn short(text: &str, most: usize) -> String {
    let line = text.replace('\n', " ");
    if line.chars().count() <= most {
        return line;
    }
    let cut: String = line.chars().take(most).collect();
    format!("{cut}…")
}
