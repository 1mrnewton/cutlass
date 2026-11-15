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

/// How a step stands, which its mark shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepState {
    Running,
    Failed,
    Done,
}

/// The mark before a step, drawn in `rect`: three dots while it runs, a
/// cross when it failed, a check when it is done. It is drawn and not
/// written, as the window's fonts have no check or cross and a glyph they
/// lack shows as an empty box.
fn step_mark(state: StepState, rect: Rect, colour: Color32) -> Vec<egui::Shape> {
    let side = rect.width().min(rect.height());
    let square = Rect::from_center_size(rect.center(), Vec2::splat(side));
    let at = |x: f32, y: f32| square.min + Vec2::new(x * side, y * side);
    let stroke = egui::Stroke::new((side * 0.16).max(1.2), colour);
    match state {
        StepState::Running => [0.2, 0.5, 0.8]
            .into_iter()
            .map(|x| egui::Shape::circle_filled(at(x, 0.5), side * 0.09, colour))
            .collect(),
        StepState::Failed => vec![
            egui::Shape::line_segment([at(0.22, 0.22), at(0.78, 0.78)], stroke),
            egui::Shape::line_segment([at(0.78, 0.22), at(0.22, 0.78)], stroke),
        ],
        StepState::Done => vec![egui::Shape::line(
            vec![at(0.14, 0.54), at(0.4, 0.78), at(0.86, 0.24)],
            stroke,
        )],
    }
}

/// The button that takes the assistant's last turn back.
pub const UNDO_TURN: &str = "Undo turn";

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

    let can_undo_turn = assistant.can_undo_turn(shell);
    let box_height = if can_undo_turn { 92.0 + 30.0 } else { 92.0 };
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
                        "Cutlass reads the project and edits it with its tools. Each edit is one undo step, and Undo turn takes a whole turn back.",
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
    if can_undo_turn {
        ui.horizontal(|ui| {
            if ui
                .add(ui::Button::new(UNDO_TURN).text_size(12.0))
                .on_hover_text(
                    "Put the project back to how the assistant's last turn found it. Undo brings its edits back.",
                )
                .clicked()
            {
                assistant.undo_turn(shell);
            }
        });
        ui.add_space(6.0);
    }
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
            let (state, colour) = match (result, failed) {
                (None, _) => (StepState::Running, tokens::MUTED),
                (Some(_), true) => (StepState::Failed, FAILED),
                (Some(_), false) => (StepState::Done, DONE),
            };
            let title = format!("{tool} {}", short(input, 40));
            // The step's mark takes the place of the header's arrow; a
            // click on the row still opens what was sent and what came
            // back.
            egui::CollapsingHeader::new(RichText::new(title).size(12.0).color(colour))
                .id_salt(("assistant-step", index))
                .default_open(false)
                .icon(move |ui, _openness, response| {
                    ui.painter().extend(step_mark(state, response.rect, colour));
                })
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
                        let notes = shell.assistant_notes();
                        let seen = shell.assistant_view();
                        assistant.send(shell.settings(), &notes, &seen);
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

#[cfg(test)]
mod tests {
    use egui::{Color32, Rect, Shape, pos2, vec2};

    use super::{StepState, step_mark};

    #[test]
    fn a_step_mark_is_drawn_inside_its_box_and_each_state_has_its_own() {
        // A box wider than tall: the mark keeps to a square in its middle.
        let rect = Rect::from_min_size(pos2(10.0, 20.0), vec2(18.0, 12.0));
        let colour = Color32::from_rgb(1, 2, 3);
        let running = step_mark(StepState::Running, rect, colour);
        let failed = step_mark(StepState::Failed, rect, colour);
        let done = step_mark(StepState::Done, rect, colour);
        // Three dots, a cross of two strokes, and a check of one.
        assert_eq!(running.len(), 3);
        assert!(
            running
                .iter()
                .all(|shape| matches!(shape, Shape::Circle(_)))
        );
        assert_eq!(failed.len(), 2);
        assert!(
            failed
                .iter()
                .all(|shape| matches!(shape, Shape::LineSegment { .. }))
        );
        assert_eq!(done.len(), 1);
        let Shape::Path(check) = &done[0] else {
            panic!("the check is one line: {:?}", done[0]);
        };
        assert_eq!(check.points.len(), 3);
        assert!(!check.closed);
        // The check's corner is its lowest point, right of its start.
        assert!(check.points[1].y > check.points[0].y && check.points[1].y > check.points[2].y);
        assert!(check.points[0].x < check.points[1].x && check.points[1].x < check.points[2].x);
        for shape in running.iter().chain(&failed).chain(&done) {
            let bounds = shape.visual_bounding_rect();
            assert!(rect.contains_rect(bounds), "{bounds:?} is outside {rect:?}");
            assert!(bounds.width() > 0.0 && bounds.height() > 0.0);
        }
    }

    #[test]
    fn the_windows_fonts_have_no_check_or_cross_to_write() {
        // Why the marks are drawn: written, each of these is an empty box.
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(egui::RawInput::default(), |_| {});
        output.textures_delta.clear();
        let font = egui::FontId::proportional(12.0);
        for glyph in ['✓', '✕', '✔', '✖'] {
            assert!(
                !ctx.fonts_mut(|fonts| fonts.has_glyph(&font, glyph)),
                "{glyph} can be written now; the mark could be a glyph again"
            );
        }
    }
}
