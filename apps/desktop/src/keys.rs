//! Keys for the tools the shell can already run.
//!
//! The letters follow CapCut. Undo and redo follow `docs/timeline.md`.

use eframe::egui::{self, Key, Modifiers};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shortcut {
    Undo,
    Redo,
    Select,
    Split,
    Snap,
    Magnet,
    Linkage,
    Marker,
    ZoomIn,
    ZoomOut,
    ZoomFit,
    Copy,
    Cut,
    Paste,
    Duplicate,
    Play,
}

impl Shortcut {
    fn history(self) -> bool {
        matches!(self, Self::Undo | Self::Redo)
    }
}

pub(crate) fn history(ui: &egui::Ui) -> Option<Shortcut> {
    take(ui, Shortcut::history)
}

pub(crate) fn tool(ui: &egui::Ui) -> Option<Shortcut> {
    take(ui, |shortcut| {
        !shortcut.history() && !matches!(shortcut, Shortcut::Play)
    })
}

pub(crate) fn play(ui: &egui::Ui) -> bool {
    take(ui, |shortcut| matches!(shortcut, Shortcut::Play)).is_some()
}

pub(crate) fn set_blocked(ui: &egui::Ui, blocked: bool) {
    let id = block_id();
    if blocked {
        ui.data_mut(|data| data.insert_temp(id, true));
    } else {
        ui.data_mut(|data| data.remove::<bool>(id));
    }
}

fn block_id() -> egui::Id {
    egui::Id::unique("shortcuts-blocked")
}

fn take(ui: &egui::Ui, want: impl Fn(Shortcut) -> bool) -> Option<Shortcut> {
    if ui.memory(|memory| memory.focused().is_some()) {
        return None;
    }
    if ui
        .data(|data| data.get_temp::<bool>(block_id()))
        .unwrap_or(false)
    {
        return None;
    }
    ui.input_mut(|input| {
        let mut found = None;
        input.events.retain(|event| {
            if found.is_some() {
                return true;
            }
            let egui::Event::Key {
                key,
                pressed,
                repeat,
                modifiers,
                ..
            } = event
            else {
                return true;
            };
            if !pressed || *repeat {
                return true;
            }
            let Some(shortcut) = shortcut(modifiers, *key) else {
                return true;
            };
            if !want(shortcut) {
                return true;
            }
            found = Some(shortcut);
            false
        });
        found
    })
}

pub(crate) fn shortcut(modifiers: &Modifiers, key: Key) -> Option<Shortcut> {
    if modifiers.alt {
        return None;
    }
    let command = modifiers.command;
    let ctrl = modifiers.ctrl;
    let shift = modifiers.shift;
    if command && shift && key == Key::Z {
        return Some(Shortcut::Redo);
    }
    if ctrl && !shift && key == Key::Y {
        return Some(Shortcut::Redo);
    }
    if command && !shift && key == Key::Z {
        return Some(Shortcut::Undo);
    }
    if command && !shift && matches!(key, Key::Plus | Key::Equals) {
        return Some(Shortcut::ZoomIn);
    }
    if command && !shift && key == Key::Minus {
        return Some(Shortcut::ZoomOut);
    }
    if command && !shift && key == Key::B {
        return Some(Shortcut::Split);
    }
    if command && !shift && key == Key::C {
        return Some(Shortcut::Copy);
    }
    if command && !shift && key == Key::X {
        return Some(Shortcut::Cut);
    }
    if command && !shift && key == Key::V {
        return Some(Shortcut::Paste);
    }
    if command && !shift && key == Key::D {
        return Some(Shortcut::Duplicate);
    }
    if !command && !ctrl && shift && key == Key::Z {
        return Some(Shortcut::ZoomFit);
    }
    if !command && !ctrl && key == Key::Backtick {
        return Some(Shortcut::Linkage);
    }
    if !command && !ctrl && !shift && key == Key::Space {
        return Some(Shortcut::Play);
    }
    if !command && !ctrl && !shift {
        return match key {
            Key::A => Some(Shortcut::Select),
            Key::B => Some(Shortcut::Split),
            Key::N => Some(Shortcut::Snap),
            Key::P => Some(Shortcut::Magnet),
            Key::M => Some(Shortcut::Marker),
            _ => None,
        };
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{Key, Modifiers, Shortcut, shortcut};

    fn mods(command: bool, ctrl: bool, shift: bool) -> Modifiers {
        Modifiers {
            command,
            ctrl,
            shift,
            ..Modifiers::NONE
        }
    }

    #[test]
    fn command_z_undoes_and_shift_or_control_y_redoes() {
        assert_eq!(
            shortcut(&mods(true, false, false), Key::Z),
            Some(Shortcut::Undo)
        );
        assert_eq!(
            shortcut(&mods(true, true, false), Key::Z),
            Some(Shortcut::Undo)
        );
        assert_eq!(
            shortcut(&mods(true, false, true), Key::Z),
            Some(Shortcut::Redo)
        );
        assert_eq!(
            shortcut(&mods(true, true, true), Key::Z),
            Some(Shortcut::Redo)
        );
        assert_eq!(
            shortcut(&mods(false, true, false), Key::Y),
            Some(Shortcut::Redo)
        );
        assert_eq!(
            shortcut(&mods(true, true, false), Key::Y),
            Some(Shortcut::Redo)
        );
        assert_eq!(shortcut(&mods(true, false, false), Key::Y), None);
        assert_eq!(shortcut(&mods(false, true, true), Key::Y), None);
        assert_eq!(shortcut(&mods(false, false, false), Key::Z), None);
    }

    #[test]
    fn the_blade_snap_magnet_linkage_and_marker_use_their_capcut_keys() {
        assert_eq!(
            shortcut(&mods(false, false, false), Key::A),
            Some(Shortcut::Select)
        );
        assert_eq!(
            shortcut(&mods(false, false, false), Key::B),
            Some(Shortcut::Split)
        );
        assert_eq!(
            shortcut(&mods(true, false, false), Key::B),
            Some(Shortcut::Split)
        );
        assert_eq!(
            shortcut(&mods(true, true, false), Key::B),
            Some(Shortcut::Split)
        );
        assert_eq!(shortcut(&mods(true, false, true), Key::B), None);
        assert_eq!(
            shortcut(&mods(false, false, false), Key::N),
            Some(Shortcut::Snap)
        );
        assert_eq!(
            shortcut(&mods(false, false, false), Key::P),
            Some(Shortcut::Magnet)
        );
        assert_eq!(
            shortcut(&mods(false, false, false), Key::Backtick),
            Some(Shortcut::Linkage)
        );
        assert_eq!(
            shortcut(&mods(false, false, true), Key::Backtick),
            Some(Shortcut::Linkage)
        );
        assert_eq!(
            shortcut(&mods(false, false, false), Key::M),
            Some(Shortcut::Marker)
        );
        assert_eq!(
            shortcut(&mods(false, false, false), Key::Space),
            Some(Shortcut::Play)
        );
        assert_eq!(shortcut(&mods(true, false, false), Key::Space), None);
        assert_eq!(shortcut(&mods(false, false, true), Key::Space), None);
        let mut alt = mods(false, false, false);
        alt.alt = true;
        assert_eq!(shortcut(&alt, Key::M), None);
        assert_eq!(shortcut(&mods(false, false, true), Key::B), None);
    }

    #[test]
    fn command_c_copies_command_x_cuts_command_v_pastes_and_command_d_duplicates() {
        assert_eq!(
            shortcut(&mods(true, false, false), Key::C),
            Some(Shortcut::Copy)
        );
        assert_eq!(
            shortcut(&mods(true, true, false), Key::X),
            Some(Shortcut::Cut)
        );
        assert_eq!(
            shortcut(&mods(true, false, false), Key::V),
            Some(Shortcut::Paste)
        );
        assert_eq!(
            shortcut(&mods(true, false, false), Key::D),
            Some(Shortcut::Duplicate)
        );
        assert_eq!(shortcut(&mods(true, false, true), Key::C), None);
        assert_eq!(shortcut(&mods(false, false, false), Key::C), None);
        assert_eq!(shortcut(&mods(false, true, false), Key::V), None);
        let mut alt = mods(true, false, false);
        alt.alt = true;
        assert_eq!(shortcut(&alt, Key::D), None);
    }

    #[test]
    fn zoom_fits_on_shift_z_and_steps_on_the_command_key() {
        assert_eq!(
            shortcut(&mods(false, false, true), Key::Z),
            Some(Shortcut::ZoomFit)
        );
        assert_eq!(
            shortcut(&mods(true, false, false), Key::Equals),
            Some(Shortcut::ZoomIn)
        );
        assert_eq!(
            shortcut(&mods(true, true, false), Key::Plus),
            Some(Shortcut::ZoomIn)
        );
        assert_eq!(
            shortcut(&mods(true, false, false), Key::Minus),
            Some(Shortcut::ZoomOut)
        );
        assert_eq!(shortcut(&mods(false, false, false), Key::Equals), None);
        assert_eq!(shortcut(&mods(true, false, true), Key::Minus), None);
    }
}
