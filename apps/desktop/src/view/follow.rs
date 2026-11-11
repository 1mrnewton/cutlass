//! Tracking on the item that follows: a text, sticker or shape's Tracking
//! tab, and the Mask page's Track mask. The item picks how to find the
//! subject, a box is drawn around it on the preview, and Start tracking
//! follows it through the video below as a job. Once it follows, the tab
//! says what it follows, a frame can be fixed by drawing the box again,
//! and Stop following takes the motion off in one undo step.
//!
//! Finding a face, a body or a hand by itself comes later: Custom, a box
//! drawn by hand, is the way in for now.

use egui::{RichText, Ui};

use cutlass_eval::TrackSpan;

use crate::shell::Shell;
use crate::{tokens, ui};

use super::inspector::{InspectorState, LATER, ROW_GAP, label_row, later_badge, note, send};

/// The ways to track, by the reference's names.
const DIRECTIONS: [(TrackSpan, &str); 3] = [
    (TrackSpan::Both, "Both"),
    (TrackSpan::After, "Forward"),
    (TrackSpan::Before, "Backward"),
];

/// How the tiles at the top of the tab read.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Tiles {
    /// Face, Body, Hand and Custom, as a text or sticker has.
    Subjects,
    /// None and Motion tracking, as a shape has.
    Motion,
}

/// Where the selected item stands.
#[derive(Clone, Debug, PartialEq)]
enum Stage {
    /// It follows nothing.
    Free,
    /// A track for it is being set up; the box is drawn or not.
    Drawing { boxed: bool, fixing: bool },
    /// It follows the video clip named.
    Following(String),
}

fn stage(shell: &Shell) -> Stage {
    let Some(id) = shell.selected().cloned() else {
        return Stage::Free;
    };
    if let Some(tracking) = shell.tracking()
        && (tracking.followers.contains(&id) || tracking.masks.contains(&id))
    {
        return Stage::Drawing {
            boxed: tracking.canvas_box.is_some(),
            fixing: tracking.fixing.is_some(),
        };
    }
    match shell.following() {
        Some(following) => Stage::Following(following.target_name),
        None => Stage::Free,
    }
}

/// A text, sticker or shape's Tracking tab.
pub(super) fn follow_tab(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, tiles: Tiles) {
    let stage = stage(shell);
    let live = !matches!(stage, Stage::Free);
    ui.add_space(8.0);
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::Vec2::splat(6.0);
        match tiles {
            Tiles::Subjects => {
                ui.add_enabled_ui(false, |ui| {
                    for name in ["Face", "Body", "Hand"] {
                        ui.add_sized([64.0, 48.0], ui::Button::new(name).text_size(11.0))
                            .on_disabled_hover_text(LATER);
                    }
                });
                if ui
                    .add_sized(
                        [64.0, 48.0],
                        ui::Button::new("Custom").text_size(11.0).on(live),
                    )
                    .on_hover_text("Draw a box around what to follow")
                    .clicked()
                    && !live
                {
                    send(shell, Shell::start_following);
                }
            }
            Tiles::Motion => {
                if super::inspector::none_tile(ui, !live).clicked() && live {
                    leave(shell, &stage);
                }
                if super::audio::name_tile(ui, "Motion tracking", live)
                    .on_hover_text("Draw a box around what to follow")
                    .clicked()
                    && !live
                {
                    send(shell, Shell::start_following);
                }
            }
        }
    });
    ui.add_space(ROW_GAP);
    if tiles == Tiles::Subjects {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Face, Body and Hand")
                    .size(11.0)
                    .color(tokens::FAINT),
            );
            later_badge(ui);
        });
    }
    match stage {
        Stage::Free => {
            let reason = shell.follow_target().err();
            note(
                ui,
                &reason.map_or_else(
                    || "Pick Custom, then click the subject on the preview or draw a box around it.".to_owned(),
                    |error| format!("To track, {error}."),
                ),
            );
        }
        Stage::Drawing { boxed, fixing } => drawing(shell, ui, state, boxed, fixing, "item"),
        Stage::Following(name) => following(shell, ui, &name, "Stop following"),
    }
}

/// The Mask page's Track mask: the clip's own picture is tracked and its
/// mask follows. `colour` is the Adjust tab's mask, which cannot follow.
pub(super) fn track_mask(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, colour: bool) {
    ui.horizontal(|ui| {
        ui.set_min_height(36.0);
        ui.label(RichText::new("Track mask").size(13.0));
        if colour {
            later_badge(ui);
        }
    });
    if colour {
        note(ui, "A colour mask cannot follow a subject yet.");
        return;
    }
    match stage(shell) {
        Stage::Free => {
            let current = state
                .fields
                .get("track direction")
                .and_then(|name| DIRECTIONS.iter().find(|(_, n)| n == name))
                .map_or(TrackSpan::Both, |(span, _)| *span);
            if let Some(span) = direction_row(ui, current) {
                let name = DIRECTIONS
                    .iter()
                    .find(|(s, _)| *s == span)
                    .map_or("Both", |(_, n)| *n);
                state
                    .fields
                    .insert("track direction".into(), name.to_owned());
            }
            if ui
                .add(ui::Button::new("Track").text_size(12.0))
                .on_hover_text("Draw a box around what the mask follows")
                .clicked()
            {
                send(shell, |shell| {
                    shell.start_mask_following()?;
                    shell.set_track_direction(current);
                    Ok(())
                });
            }
        }
        Stage::Drawing { boxed, fixing } => drawing(shell, ui, state, boxed, fixing, "mask"),
        Stage::Following(name) => following(shell, ui, &name, "Stop tracking"),
    }
}

/// A track being set up: what to do on the preview, which way to track,
/// and Start tracking once a box is drawn.
fn drawing(
    shell: &mut Shell,
    ui: &mut Ui,
    _state: &mut InspectorState,
    boxed: bool,
    fixing: bool,
    what: &str,
) {
    let target = shell.tracking_target_name().unwrap_or_default();
    let way = if shell.track_subject_enabled() {
        "Click the subject on the preview, or drag a box around it"
    } else {
        "Drag a box around the subject on the preview"
    };
    let text = match (boxed, fixing) {
        (false, false) => format!("{way}. The {what} keeps its place against it."),
        (false, true) => format!("{way} where it is at the playhead."),
        (true, _) => format!("Tracking {target}. Click or drag again to change the box."),
    };
    note(ui, &text);
    if !fixing {
        let current = shell.tracking().map_or(TrackSpan::Both, |t| t.direction);
        if let Some(span) = direction_row(ui, current) {
            shell.set_track_direction(span);
        }
    }
    ui.horizontal(|ui| {
        let start = ui
            .add_enabled(
                boxed && shell.track_ready(),
                ui::Button::new("Start tracking").text_size(12.0),
            )
            .on_disabled_hover_text("Click the subject or draw a box on the preview first")
            .clicked();
        if ui.add(ui::Button::new("Cancel").text_size(12.0)).clicked() {
            shell.stop_tracking();
        }
        if start {
            send(shell, Shell::run_tracking);
        }
    });
}

/// Which way to track from the drawn frame, as a choice row.
fn direction_row(ui: &mut Ui, current: TrackSpan) -> Option<TrackSpan> {
    let name = DIRECTIONS
        .iter()
        .find(|(span, _)| *span == current)
        .map_or("Both", |(_, name)| *name);
    let mut picked = None;
    label_row(ui, "Direction", |ui| {
        egui::ComboBox::from_id_salt("track direction")
            .width(ui.available_width())
            .selected_text(name)
            .show_ui(ui, |ui| {
                for (span, name) in DIRECTIONS {
                    if ui
                        .add(ui::Choice::new(span == current, name).wide())
                        .clicked()
                    {
                        picked = Some(span);
                    }
                }
            });
    });
    picked
}

/// An item that follows a track: what it follows, Fix frame and stop.
fn following(shell: &mut Shell, ui: &mut Ui, target: &str, stop: &str) {
    note(ui, &format!("Follows {target}."));
    ui.horizontal(|ui| {
        if ui
            .add(ui::Button::new("Fix frame").text_size(12.0))
            .on_hover_text("Draw the box again at the playhead and track from there")
            .clicked()
        {
            send(shell, Shell::start_fixing_following);
        }
        if ui.add(ui::Button::new(stop).text_size(12.0)).clicked() {
            send(shell, Shell::stop_following);
        }
    });
}

/// None on a shape's tab: stop setting up, or stop following.
fn leave(shell: &mut Shell, stage: &Stage) {
    match stage {
        Stage::Drawing { .. } => shell.stop_tracking(),
        Stage::Following(_) => send(shell, Shell::stop_following),
        Stage::Free => {}
    }
}
