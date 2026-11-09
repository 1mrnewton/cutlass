//! A video clip's Motion blur section: Blur and Blend sliders, Direction
//! and Speed, and a reset to the first settings.
//!
//! Each change makes a new blurred copy as a job, so a slider sends its
//! value once it is let go, not as it moves; the job's progress shows
//! here.

use egui::Ui;

use cutlass_model::MotionBlur;

use crate::shell::{KeyedSetting, Shell};
use crate::ui;

use super::inspector::{InspectorState, body, label_row, note, section, send, setting_row, slider};

/// What each choice is called.
const DIRECTIONS: [(i64, &str); 3] = [
    (MotionBlur::BOTH, "Both"),
    (MotionBlur::FORWARD, "Forward"),
    (MotionBlur::BACKWARD, "Backward"),
];
const SPEEDS: [(i64, &str); 4] = [(1, "Once"), (2, "Twice"), (4, "4 times"), (6, "6 times")];

/// The settings a clip's Motion blur starts at and resets to.
const FIRST: MotionBlur = MotionBlur {
    blur: 1.0,
    blend: 0.0,
    direction: MotionBlur::BOTH,
    times: 1,
};

/// The Motion blur section of a video clip's Basic page.
pub(super) fn motion_blur(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState) {
    let worn = shell.motion_blur();
    let header = section(
        ui,
        state,
        "motion-blur",
        "Motion blur",
        Some((worn.is_some(), true)),
        true,
    );
    if header.toggled {
        let next = if worn.is_some() { None } else { Some(FIRST) };
        send(shell, move |shell| shell.set_motion_blur(next));
        return;
    }
    if header.reset && worn.as_ref().is_some_and(|blur| *blur != FIRST) {
        send(shell, |shell| shell.set_motion_blur(Some(FIRST)));
        return;
    }
    let Some(blur) = worn.filter(|_| header.open) else {
        return;
    };
    let progress = shell.motion_blur_progress();
    let mut wanted = None;
    body(ui, |ui| {
        if let Some(done) = progress {
            note(ui, &format!("Processing {:.1}%", done * 100.0));
        }
        let held = ui.input(|input| input.pointer.any_down());
        for (key, label, stored, setting) in [
            (
                "motion blur length",
                "Blur",
                blur.blur,
                KeyedSetting::MotionBlurLength,
            ),
            (
                "motion blur blend",
                "Blend",
                blur.blend,
                KeyedSetting::MotionBlurBlend,
            ),
        ] {
            let pending = format!("{key} pending");
            let changed = setting_row(shell, ui, setting, true, |ui| {
                slider(
                    ui,
                    state,
                    key,
                    label,
                    (stored * 100.0).round(),
                    0.0..=100.0,
                    |v| format!("{v:.0}"),
                )
            });
            if let Some(value) = changed {
                state.fields.insert(pending.clone(), value.to_string());
            }
            // Sent once the slider is let go.
            if !held
                && let Some(value) = state
                    .fields
                    .remove(&pending)
                    .and_then(|value| value.parse::<f64>().ok())
            {
                let mut next = wanted.clone().unwrap_or_else(|| blur.clone());
                if key == "motion blur length" {
                    next.blur = value / 100.0;
                } else {
                    next.blend = value / 100.0;
                }
                wanted = Some(next);
            }
        }
        for (salt, label, choices, current) in [
            (
                "motion-blur-direction",
                "Direction",
                &DIRECTIONS[..],
                blur.direction,
            ),
            ("motion-blur-speed", "Speed", &SPEEDS[..], blur.times),
        ] {
            let name = choices
                .iter()
                .find(|(value, _)| *value == current)
                .map_or("", |(_, name)| *name);
            let mut picked = None;
            label_row(ui, label, |ui| {
                egui::ComboBox::from_id_salt(salt)
                    .width(ui.available_width())
                    .selected_text(name)
                    .show_ui(ui, |ui| {
                        for (value, name) in choices {
                            if ui
                                .add(ui::Choice::new(*value == current, *name).wide())
                                .clicked()
                            {
                                picked = Some(*value);
                            }
                        }
                    });
            });
            if let Some(value) = picked.filter(|value| *value != current) {
                let mut next = wanted.clone().unwrap_or_else(|| blur.clone());
                if salt == "motion-blur-direction" {
                    next.direction = value;
                } else {
                    next.times = value;
                }
                wanted = Some(next);
            }
        }
    });
    if let Some(next) = wanted.filter(|next| *next != blur) {
        send(shell, move |shell| shell.set_motion_blur(Some(next)));
    }
}
