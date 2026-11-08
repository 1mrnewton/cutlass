//! The inspector's clip panels, laid out as the desktop prototype lays them
//! out: tabs, then a page switch, then sections that fold open, each with an
//! on switch. The selected clip's kind picks the panel.
//!
//! Every control that has an edit sends one request through the shell, so
//! each change is one undo step; a slider sends when the pointer lets go. A
//! control the prototype shows with no edit in Rust yet is drawn dimmed and
//! marked "Later". A clip on a locked row shows its values without edits.

use std::collections::{HashMap, HashSet};

use egui::{
    Align, Align2, Color32, CornerRadius, FontId, Layout, Rect, RichText, Sense, Stroke, Ui, Vec2,
};

use crate::shell::{
    Inspection, KeyedSetting, LetterCase, Panel, SILENT_DB, Shell, ShellError, TextLook, volume_db,
};
use crate::{icons, tokens, ui};
use cutlass_model::{ANIMATION_PRESETS, AnimType, Micros, Motion};

use super::ViewState;

/// The active tab color, and an on switch.
pub(super) const ACCENT: Color32 = Color32::from_rgb(0x2a, 0xd4, 0xe0);
const PAGES: Color32 = tokens::INSET;
const PAGE_ON: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x3a);
pub(super) const BOX: Color32 = tokens::RAISED;
const LABEL_WIDTH: f32 = 92.0;
pub(super) const VALUE_WIDTH: f32 = 60.0;
const PAIR_BOX: f32 = 84.0;
/// The height of a control row, and the gap under it.
pub(super) const ROW_HEIGHT: f32 = 28.0;
pub(super) const ROW_GAP: f32 = 6.0;
/// The widest and narrowest gap between the inspector's tabs.
const TAB_GAP: f32 = 18.0;
const TAB_GAP_MIN: f32 = 8.0;
pub(super) const LATER: &str = "Later: Cutlass has no edit for this yet";

/// Which tab, page, and sections are open. They reset when another clip
/// is picked, as the prototype's panels do.
#[derive(Default)]
pub(super) struct InspectorState {
    clip: String,
    pub(super) tab: usize,
    pub(super) page: usize,
    pub(super) open: HashSet<String>,
    /// Typed text in a number or word field while it has focus.
    fields: HashMap<String, String>,
    /// A slider or color being dragged; it is sent when the pointer lets go.
    live: HashMap<String, f64>,
    colors: HashMap<String, Color32>,
    /// A tab, page, and section a menu row asked for, opened on the next
    /// frame the panel draws.
    pending: Option<Focus>,
    /// The HSL colour shown, 0 red to 7 magenta.
    pub(super) hsl_color: usize,
    /// The curve shown: RGB, red, green, blue.
    pub(super) curve_channel: usize,
    /// A curve point being dragged, and the points as they are now.
    pub(super) curve_drag: Option<(usize, Vec<(f32, f32)>)>,
    /// A speed curve point being dragged, and the points as they are now.
    pub(super) speed_drag: Option<(usize, Vec<(f64, f64)>)>,
    /// The chip picked in each row of chips on a page, by row.
    pub(super) chips: HashMap<&'static str, usize>,
    /// The LUT file picker was asked for; the view opens it.
    pub(super) wants_lut: bool,
    /// The pictures drawn for the Templates, Bubble and Effects tiles, by
    /// item id. They outlast a pick of another clip.
    tiles: HashMap<String, egui::TextureHandle>,
}

/// Where a clip menu row sends the inspector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Focus {
    pub(super) tab: usize,
    pub(super) page: usize,
    pub(super) section: Option<&'static str>,
}

impl InspectorState {
    /// Open `focus` the next time the panel draws, after any reset for a
    /// newly picked clip.
    pub(super) fn focus(&mut self, focus: Focus) {
        self.pending = Some(focus);
    }
}

pub(super) fn clip_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    view: &mut ViewState,
    inspection: &Inspection,
    _width: f32,
) {
    let panel = shell.panel().unwrap_or(Panel::Other);
    let state = &mut view.inspector;
    let pending = state.pending.take();
    if state.clip != inspection.segment_id.as_str() {
        *state = InspectorState {
            clip: inspection.segment_id.as_str().to_owned(),
            open: [
                "position",
                "sound",
                "custom-removal",
                "chroma",
                "auto-adjust",
                "color-match",
                "lut",
                "adjust",
            ]
            .into_iter()
            .chain(matches!(panel, Panel::Text | Panel::Caption).then_some("blend"))
            .map(str::to_owned)
            .collect(),
            tiles: std::mem::take(&mut state.tiles),
            ..InspectorState::default()
        };
    }
    if let Some(focus) = pending {
        // A caption's Captions tab comes before the text tabs a menu names.
        state.tab = focus.tab + usize::from(panel == Panel::Caption);
        state.page = focus.page;
        if let Some(section) = focus.section {
            state.open.insert(section.to_owned());
        }
    }
    if inspection.locked {
        note(ui, "This clip's row is locked.");
    }
    ui.add_enabled_ui(!inspection.locked, |ui| match panel {
        Panel::Main { still } | Panel::Overlay { still } => {
            video_panel(shell, ui, state, inspection, still);
        }
        Panel::Text | Panel::Caption => {
            text_panel(shell, ui, state, inspection, panel == Panel::Caption);
        }
        Panel::Audio => audio_panel(shell, ui, state, inspection),
        Panel::Sticker => sticker_panel(shell, ui, state, inspection),
        Panel::Effect => effect_panel(shell, ui, state, inspection, "Effect", false),
        Panel::Filter => effect_panel(shell, ui, state, inspection, "Filter", true),
        Panel::Adjust => adjust_panel(shell, ui, state, inspection),
        Panel::Other => {
            tabs(ui, &["Clip"], &mut 0);
            note(ui, &inspection.material);
        }
    });
}

// Panels.

fn video_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
    still: bool,
) {
    let tab = state.tab;
    tabs(
        ui,
        &[
            "Video",
            "Audio",
            "Speed",
            "Animation",
            "Adjust",
            "AI stylize",
        ],
        &mut state.tab,
    );
    if tab != state.tab {
        state.page = 0;
    }
    match state.tab {
        0 => {
            pages(
                ui,
                &["Basic", "Remove BG", "Mask", "Retouch"],
                &mut state.page,
            );
            if state.page == 1 {
                super::remove_bg::remove_bg_page(shell, ui, state, inspection);
                return;
            }
            if state.page == 2 {
                mask_page(shell, ui, state, inspection.mask, false);
                return;
            }
            if state.page == 3 {
                super::retouch::retouch_page(ui, state);
                return;
            }
            position_size(shell, ui, state, inspection, "Rotate");
            blend(shell, ui, state, inspection, true);
            for label in [
                "Stabilize",
                "Enhance quality",
                "Noise reduction",
                "Optical flow",
                "AI remove",
                "AI expand",
                "AI remix",
                "Lip sync",
                "Relight",
                "Auto reframe",
                "Removing video flickers",
                "AI movement",
                "Camera tracking",
                "Motion blur",
            ] {
                later_row(ui, label);
            }
            canvas(shell, ui, state, inspection);
        }
        1 => {
            pages(ui, &["Basic", "Voice changer"], &mut state.page);
            if still {
                note(ui, "A photo has no sound.");
                return;
            }
            if state.page == 1 {
                later_note(ui, "Voice changer");
                return;
            }
            sound_section(shell, ui, state, inspection);
            for label in [
                "Loudness normalization",
                "Enhance voice",
                "Video translator",
                "Reduce noise",
                "Isolate voice",
                "Fill channel",
            ] {
                later_row(ui, label);
            }
        }
        2 if still => note(ui, "A photo has no speed."),
        2 => super::speed::speed_tab(shell, ui, state, inspection),
        3 => animation_tab(shell, ui, state, &["In", "Out", "Combo"], VIDEO_MOTIONS),
        4 => adjust_tab(shell, ui, state, inspection),
        _ => later_note(ui, "AI stylize"),
    }
}

fn text_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
    caption: bool,
) {
    // A caption's tabs lead with its list. AI avatars is paid, and shown
    // off for now.
    let labels: &[&str] = if caption {
        &[
            "Captions",
            "Text",
            "Animation",
            "Tracking",
            "Text to speech",
            "AI avatars",
        ]
    } else {
        &[
            "Text",
            "Animation",
            "Tracking",
            "Text to speech",
            "AI avatars",
        ]
    };
    let tab = state.tab;
    paid_tabs(ui, labels, &["AI avatars"], &mut state.tab);
    if tab != state.tab {
        state.page = 0;
    }
    match labels.get(state.tab).copied().unwrap_or_default() {
        "Captions" => caption_list(shell, ui, state),
        "Text" => {
            // A caption has its templates before its bubble.
            let names: &[&str] = if caption {
                &["Basic", "Templates", "Bubble", "Effects"]
            } else {
                &["Basic", "Bubble", "Effects"]
            };
            pages(ui, names, &mut state.page);
            // "Apply to all main captions" has its own row under the pages,
            // the same on each: a change on any page can style every
            // caption.
            if caption {
                all_captions_box(shell, ui);
            }
            match names.get(state.page).copied().unwrap_or_default() {
                "Basic" => {
                    text_basic(shell, ui, state, inspection, caption);
                    ui.add_space(8.0);
                    if ui
                        .add(ui::Button::new("Save as preset").text_size(12.0))
                        .on_hover_text("Keep this look in Preset style for every project")
                        .clicked()
                    {
                        send(shell, Shell::save_text_preset);
                    }
                }
                "Templates" => {
                    caption_templates(shell, ui, state);
                    if !inspection.words.text.is_empty() {
                        ui.add_space(12.0);
                        ui.label(RichText::new("Word styles").size(12.0).color(tokens::MUTED));
                        caption_styles(shell, ui, inspection);
                    }
                }
                "Bubble" => bubble_page(shell, ui, state),
                _ => effect_page(shell, ui, state),
            }
        }
        "Animation" => animation_tab(shell, ui, state, &["In", "Out", "Loop"], TEXT_MOTIONS),
        "Tracking" => {
            later_tiles(ui, &["Face", "Body", "Hand", "Custom"]);
            later_button(ui, "Start tracking");
        }
        _ => {
            later_note(ui, "Text to speech");
            later_tiles(
                ui,
                &[
                    "Warm", "Bright", "Even", "Clear", "Light", "Low", "Soft", "Steady", "Quick",
                    "Dry",
                ],
            );
        }
    }
}

/// The Captions tab: every caption on the lane, numbered in timeline order,
/// the one under the playhead in the accent color. A line is typed over in
/// place, and its words keep their times; + adds a caption after it, and
/// the bin takes it off.
fn caption_list(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState) {
    let rows = shell.caption_rows();
    let current = cutlass_editor::row_at(&rows, shell.playhead());
    ui.add_space(8.0);
    let mut rerun = false;
    ui.horizontal(|ui| {
        // The tools sit at the right; the search takes what is left.
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.add(ui::Button::new("?").text_size(12.0))
                .on_hover_text(CAPTION_LIST_HELP);
            let open = state.open.contains(REPLACE_ROW);
            let toggle = ui::Button::new("Replace").text_size(12.0).on(open);
            if ui
                .add(toggle)
                .on_hover_text("Find and replace across every caption")
                .clicked()
                && !state.open.remove(REPLACE_ROW)
            {
                state.open.insert(REPLACE_ROW.to_owned());
            }
            rerun = ui
                .add_enabled(
                    shell.rerun_caption_enabled(),
                    ui::Button::new("Re-run").text_size(12.0),
                )
                .on_hover_text("Hear this caption's clip again, keeping its look")
                .on_disabled_hover_text("This caption was not heard from a clip on the timeline")
                .clicked();
            later_button(ui, "Translate");
            let query = state.fields.entry("caption search".to_owned()).or_default();
            ui::Field::new(query).hint("Search").icon("search").show(ui);
        });
    });
    if rerun {
        send(shell, Shell::rerun_caption);
    }
    if state.open.contains(REPLACE_ROW) {
        replace_row(shell, ui, state);
    }
    let query = state
        .fields
        .get("caption search")
        .cloned()
        .unwrap_or_default();
    let shown = cutlass_editor::rows_matching(&rows, &query);
    if rows.is_empty() {
        note(ui, "This clip is not on a caption row.");
        return;
    }
    if shown.is_empty() {
        note(ui, "No caption says that.");
        return;
    }
    let mut edit = None;
    let mut add = None;
    let mut remove = None;
    for index in shown {
        let row = &rows[index];
        let on = current == Some(index);
        let color = if on { ACCENT } else { tokens::TEXT };
        ui.separator();
        ui.horizontal(|ui| {
            ui.add_sized(
                [28.0, 24.0],
                egui::Label::new(
                    RichText::new(format!("{}", index + 1))
                        .size(12.0)
                        .color(if on { ACCENT } else { tokens::MUTED }),
                ),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui
                    .add(ui::Button::icon("delete"))
                    .on_hover_text("Delete this caption")
                    .clicked()
                {
                    remove = Some(row.segment_id.clone());
                }
                if ui
                    .add(ui::Button::new("+").text_size(12.0))
                    .on_hover_text("Add a caption after this one")
                    .clicked()
                {
                    add = Some(row.segment_id.clone());
                }
                let key = format!("caption line {}", row.segment_id.as_str());
                let text = state.fields.entry(key).or_default();
                let response = ui.add(
                    egui::TextEdit::singleline(text)
                        .frame(egui::Frame::NONE)
                        .text_color(color)
                        .desired_width(f32::INFINITY),
                );
                if super::commits(ui, &response) && *text != row.text {
                    edit = Some((row.segment_id.clone(), text.clone()));
                } else if !response.has_focus() {
                    row.text.clone_into(text);
                }
            });
        });
    }
    ui.separator();
    if let Some((segment_id, text)) = edit {
        send(shell, |shell| shell.set_caption_line(&segment_id, &text));
    } else if let Some(segment_id) = add {
        send(shell, |shell| shell.add_caption_after(&segment_id));
    } else if let Some(segment_id) = remove {
        send(shell, |shell| shell.remove_caption(&segment_id));
    }
}

/// What the Captions list's "?" says.
const CAPTION_LIST_HELP: &str = "Search shows the captions that say it. Replace swaps what the \
search finds in every caption, as one undo step. Re-run hears the clip a caption was heard from \
again and keeps its look. Click a line to type over it; + adds a caption after it.";

/// The Captions list's replace row opens under the search.
const REPLACE_ROW: &str = "caption replace";

/// Replace what the search finds, in every caption on the lane, as one
/// undo step. The list below shows the captions it will reach.
fn replace_row(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState) {
    let mut replace = false;
    ui.horizontal(|ui| {
        let with = state
            .fields
            .entry("caption replace".to_owned())
            .or_default();
        let response = ui::Field::new(with)
            .hint("Replace with")
            .width(ui.available_width() - 90.0)
            .show(ui)
            .response;
        replace = response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
        replace |= ui
            .add(ui::Button::new("Replace all").text_size(12.0))
            .clicked();
    });
    if !replace {
        return;
    }
    let find = state
        .fields
        .get("caption search")
        .cloned()
        .unwrap_or_default();
    let with = state
        .fields
        .get("caption replace")
        .cloned()
        .unwrap_or_default();
    match shell.replace_in_captions(&find, &with) {
        Ok(count) => {
            let noun = if count == 1 { "match" } else { "matches" };
            shell.set_notice(format!("Replaced {count} {noun}"));
            // The list shows every caption again, the new words in it.
            state.fields.remove("caption search");
        }
        Err(error) => shell.notice_error(&error),
    }
}

fn audio_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    super::audio::audio_panel(shell, ui, state, inspection);
}

fn sticker_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let tab = state.tab;
    tabs(ui, &["Sticker", "Animation"], &mut state.tab);
    if tab != state.tab {
        state.page = 0;
    }
    if state.tab == 1 {
        animation_tab(
            shell,
            ui,
            state,
            &["In", "Out", "Loop"],
            &[
                &["Pop", "Spin", "Fade"],
                &["Pop", "Spin", "Fade"],
                &["Pop", "Spin", "Fade"],
            ],
        );
        return;
    }
    position_size(shell, ui, state, inspection, "Rotate");
    body(ui, |ui| opacity_slider(shell, ui, state, inspection));
}

fn effect_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
    title: &str,
    filter: bool,
) {
    tabs(ui, &[title], &mut 0);
    note(ui, &shell.effect_name().unwrap_or_default());
    body(ui, |ui| {
        if let Some(stored) = shell.effect_strength() {
            let amount = (stored * 100.0).round();
            if let Some(value) = slider(ui, state, "amount", "Amount", amount, 0.0..=100.0, |v| {
                format!("{v:.0}")
            }) {
                send(shell, |shell| shell.set_effect_strength(value / 100.0));
            }
        } else {
            later_slider(ui, "Amount", 0.0, "0");
        }
        if filter {
            let key = cutlass_effects::AdjustKey::Temperature;
            let stored = (f64::from(inspection.adjust.get(key)) * 100.0).round();
            if let Some(value) = slider(ui, state, "temp", "Temp", stored, -100.0..=100.0, |v| {
                format!("{v:.0}")
            }) {
                send(shell, |shell| shell.set_adjust(key, value / 100.0));
            }
        }
    });
}

fn adjust_panel(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    tabs(ui, &["Adjust"], &mut 0);
    pages(
        ui,
        &["Basic", "HSL", "Curves", "Color wheel", "Mask"],
        &mut state.page,
    );
    if colour_page(shell, ui, state, inspection) {
        return;
    }
    state.open.insert("adjust".to_owned());
    adjust_sliders(shell, ui, state, inspection);
}

// Sections shared by panels.

fn position_size(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
    rotate_label: &str,
) {
    placement(
        shell,
        ui,
        state,
        inspection,
        "Position & Size",
        Some(rotate_label),
    );
}

/// A caption's Transform: scale, uniform scale and position, each with its
/// keyframe controls, as the video panel keys them.
fn transform(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, inspection: &Inspection) {
    placement(shell, ui, state, inspection, "Transform", None);
}

fn placement(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
    title: &str,
    rotate_label: Option<&str>,
) {
    let header = section(ui, state, "position", title, None, true);
    if header.reset {
        send(shell, Shell::reset_placement);
    }
    if !header.open {
        return;
    }
    let canvas = shell.draft().map_or((0, 0), |draft| {
        (draft.canvas_config.width, draft.canvas_config.height)
    });
    let keys = shell.keys();
    let playhead = shell.playhead();
    let mut step = None;
    body(ui, |ui| {
        let scale = (inspection.scale_x * 100.0).round();
        let scaled = keyed_row(ui, keys.as_ref(), playhead, &mut step, |ui| {
            slider(ui, state, "scale", "Scale", scale, 1.0..=500.0, |v| {
                format!("{v:.0}%")
            })
        });
        if let Some(value) = scaled {
            send(shell, |shell| shell.set_scale(value / 100.0));
        }
        switch_row(ui, "Uniform scale", true, false).on_disabled_hover_text(LATER);
        // The prototype counts canvas pixels from the middle, with Y down.
        let (x, y) =
            cutlass_editor::canvas_pixels((inspection.transform_x, inspection.transform_y), canvas);
        let (new_x, new_y) = keyed_row(ui, keys.as_ref(), playhead, &mut step, |ui| {
            pair_row(ui, state, "Position", ("X", x), ("Y", y))
        });
        if new_x.is_some() || new_y.is_some() {
            let (stored_x, stored_y) =
                cutlass_editor::stored_position((new_x.unwrap_or(x), new_y.unwrap_or(y)), canvas);
            let stored_x = if new_x.is_some() {
                stored_x
            } else {
                inspection.transform_x
            };
            let stored_y = if new_y.is_some() {
                stored_y
            } else {
                inspection.transform_y
            };
            send(shell, |shell| shell.set_position(stored_x, stored_y));
        }
        let rotation = inspection.rotation;
        let turned = rotate_label.map_or((None, None), |label| {
            keyed_row(ui, keys.as_ref(), playhead, &mut step, |ui| {
                pair_row(ui, state, label, ("°", rotation), ("", f64::NAN))
            })
        });
        match step {
            Some(KeyStep::Back) => shell.step_keyframe(false),
            Some(KeyStep::Toggle) => send(shell, Shell::toggle_keyframe),
            Some(KeyStep::Forward) => shell.step_keyframe(true),
            None => {}
        }
        if let (Some(value), _) = turned {
            send(shell, |shell| shell.set_rotation(value));
        }
        if let Some(keys) = keys.as_ref().filter(|keys| keys.span.is_some())
            && let Some(graph) = graph_row(ui, keys)
        {
            send(shell, |shell| shell.ease_keyframes(graph));
        }
        if rotate_label == Some("Rotate") {
            use cutlass_editor::StageAlign;
            let aims = [
                ("alignLeft", "Align left", Some(StageAlign::Left)),
                ("alignCenter", "Align center", Some(StageAlign::Center)),
                ("alignRight", "Align right", Some(StageAlign::Right)),
                ("alignTop", "Align top", Some(StageAlign::Top)),
                ("alignMiddle", "Align middle", Some(StageAlign::Middle)),
                ("alignBottom", "Align bottom", Some(StageAlign::Bottom)),
            ];
            if let Some(align) = icon_bar(ui, &aims) {
                send(shell, |shell| shell.align_selected(align));
            }
        }
        // Mirror, side to side, is on the timeline toolbar. A video or
        // photo clip has no upside-down flip here, as in the reference.
        if rotate_label != Some("Rotate") && shell.mirror_enabled() {
            let flipped = shell.flipped();
            if switch_row(ui, "Flip upside down", flipped, true).clicked() {
                send(shell, Shell::toggle_vertical_flip);
            }
        }
    });
}

fn blend(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
    mode: bool,
) {
    let header = section(ui, state, "blend", "Blend", Some((true, false)), false);
    if !header.open {
        return;
    }
    body(ui, |ui| {
        if mode {
            let mut picked = None;
            label_row(ui, "Mode", |ui| {
                egui::ComboBox::from_id_salt("blend-mode")
                    .selected_text(inspection.blend.label())
                    .show_ui(ui, |ui| {
                        for choice in cutlass_effects::BlendMode::ALL {
                            if ui
                                .add(
                                    ui::Choice::new(inspection.blend == choice, choice.label())
                                        .wide(),
                                )
                                .clicked()
                            {
                                picked = Some(choice);
                            }
                        }
                    });
            });
            if let Some(choice) = picked.filter(|choice| *choice != inspection.blend) {
                send(shell, |shell| shell.set_blend(choice));
            }
        }
        opacity_slider(shell, ui, state, inspection);
    });
}

fn opacity_slider(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let opacity = (inspection.opacity * 100.0).round();
    if let Some(value) = slider(ui, state, "opacity", "Opacity", opacity, 0.0..=100.0, |v| {
        format!("{v:.0}%")
    }) {
        send(shell, |shell| shell.set_opacity(value / 100.0));
    }
}

fn canvas(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, inspection: &Inspection) {
    if !shell.crop_enabled() {
        later_row(ui, "Canvas");
        return;
    }
    let on = !inspection.letterbox.is_empty();
    let header = section(ui, state, "canvas", "Canvas", Some((on, true)), false);
    if header.toggled {
        let color = if on { "" } else { "#000000" };
        send(shell, |shell| shell.set_letterbox(color));
    }
    if header.open {
        body(ui, |ui| {
            if let Some(color) =
                color_row(ui, state, "canvas-color", "Color", &inspection.letterbox)
            {
                send(shell, |shell| shell.set_letterbox(&color));
            }
            later_button(ui, "Apply to all");
        });
    }
}

pub(super) fn sound_section(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let header = section(ui, state, "sound", "Basic", Some((true, false)), true);
    if header.reset {
        send(shell, Shell::reset_sound);
    }
    if header.open {
        body(ui, |ui| sound_sliders(shell, ui, state, inspection));
    }
}

fn sound_sliders(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let db = (volume_db(inspection.volume) * 10.0).round() / 10.0;
    let keys = shell.setting_keys(KeyedSetting::Volume);
    let playhead = shell.playhead();
    let mut step = None;
    let changed = keyed_row(ui, keys.as_ref(), playhead, &mut step, |ui| {
        slider(ui, state, "volume", "Volume", db, SILENT_DB..=12.0, |v| {
            format!("{v:.1}dB")
        })
    });
    if let Some(value) = changed {
        send(shell, |shell| shell.set_volume_db(value));
    }
    setting_step(shell, KeyedSetting::Volume, step);
    if !shell.fade_enabled() {
        return;
    }
    let seconds = |micros: i64| (micros as f64 / 100_000.0).round() / 10.0;
    let fade_in = seconds(inspection.fade_in);
    if let Some(value) = slider(ui, state, "fade-in", "Fade in", fade_in, 0.0..=10.0, |v| {
        format!("{v:.1}s")
    }) {
        send(shell, |shell| shell.set_fade_in(value));
    }
    let fade_out = seconds(inspection.fade_out);
    if let Some(value) = slider(
        ui,
        state,
        "fade-out",
        "Fade out",
        fade_out,
        0.0..=10.0,
        |v| format!("{v:.1}s"),
    ) {
        send(shell, |shell| shell.set_fade_out(value));
    }
}

const VIDEO_MOTIONS: &[&[&str]] = &[
    &[
        "Fade in",
        "Slide up",
        "Slide down",
        "Slide left",
        "Slide right",
        "Zoom in",
        "Pop",
        "Spin",
        "Wipe",
        "Blur in",
        "Rise",
        "Flip",
    ],
    &[
        "Fade out",
        "Slide up",
        "Slide down",
        "Slide left",
        "Slide right",
        "Zoom out",
        "Shrink",
        "Spin",
        "Wipe",
        "Blur out",
        "Fall",
        "Flip",
    ],
    &[
        "Shake", "Pulse", "Float", "Bounce", "Drift", "Flicker", "Swing", "Breathe",
    ],
];

const TEXT_MOTIONS: &[&[&str]] = &[
    &[
        "Fade", "Rise", "Drop", "Slide", "Pop", "Type on", "Blur", "Spin", "Bounce", "Unfold",
    ],
    &[
        "Fade", "Rise", "Drop", "Slide", "Shrink", "Type off", "Blur", "Spin",
    ],
    &["Pulse", "Flicker", "Float", "Shake", "Swing", "Breathe"],
];

/// The In, Out, and Combo or Loop slots, each a grid of animations that
/// starts with None. In, Out and Loop store the shared presets on a
/// picture or title, with a length (a loop's is one round); Combo is still
/// to come.
fn animation_tab(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    slots: &[&str],
    motions: &[&[&str]],
) {
    pages(ui, slots, &mut state.page);
    let kind = match (state.page, slots.get(2)) {
        (0, _) => Some(AnimType::In),
        (1, _) => Some(AnimType::Out),
        (2, Some(&"Loop")) => Some(AnimType::Loop),
        _ => None,
    };
    if let Some(kind) = kind
        && let Some(animations) = shell.animations()
    {
        let letters = std::ptr::eq(motions, TEXT_MOTIONS);
        animation_side(shell, ui, state, &animations, kind, letters);
        return;
    }
    later_note(ui, "Animations");
    ui.add_enabled_ui(false, |ui| {
        let mut query = String::new();
        ui::Field::new(&mut query)
            .hint("Search animations")
            .icon("search")
            .width(ui.available_width() - 28.0)
            .show(ui);
    });
    let mut names = vec!["None"];
    names.extend_from_slice(motions.get(state.page).copied().unwrap_or_default());
    later_tiles(ui, &names);
    later_slider(ui, "Duration", 2.0, "2.0s");
}

/// Act on a press of `setting`'s keyframe buttons.
fn setting_step(shell: &mut Shell, setting: KeyedSetting, step: Option<KeyStep>) {
    match step {
        Some(KeyStep::Back) => shell.step_setting_key(setting, false),
        Some(KeyStep::Toggle) => send(shell, |shell| shell.toggle_setting_key(setting)),
        Some(KeyStep::Forward) => shell.step_setting_key(setting, true),
        None => {}
    }
}

/// A press on a row's keyframe buttons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyStep {
    Back,
    Toggle,
    Forward,
}

/// One ◀ ◆ ▶ button, and the gap between two.
const KEY_BUTTON: f32 = 20.0;
const KEY_GAP: f32 = 2.0;
/// The room the ◀ ◆ ▶ buttons take at a row's end.
const KEYS_WIDTH: f32 = 3.0 * KEY_BUTTON + 2.0 * KEY_GAP;

/// `row` with the clip's keyframe buttons at its end, when the clip takes
/// keys. The model keys a clip's whole pose at once, so each row's buttons
/// step through and toggle the same keys.
fn keyed_row<R>(
    ui: &mut Ui,
    keys: Option<&cutlass_editor::ClipKeys>,
    playhead: Micros,
    step: &mut Option<KeyStep>,
    row: impl FnOnce(&mut Ui) -> R,
) -> R {
    let Some(keys) = keys else {
        return row(ui);
    };
    ui.horizontal(|ui| {
        // The buttons follow the row after one item gap.
        let width = (ui.available_width() - KEYS_WIDTH - ui.spacing().item_spacing.x).max(80.0);
        let inner = ui
            .allocate_ui(Vec2::new(width, 30.0), |ui| {
                // A short row still takes its width, so the buttons line up
                // at the end of every row.
                ui.set_min_width(width);
                ui.vertical(row).inner
            })
            .inner;
        if let Some(pressed) = key_buttons(ui, keys, playhead) {
            *step = Some(pressed);
        }
        inner
    })
    .inner
}

/// ◀ to the key before the playhead, ◆ to key here or unkey, ▶ to the key
/// after. The diamond is filled on a key; an arrow is faint with no key
/// that way.
fn key_buttons(ui: &mut Ui, keys: &cutlass_editor::ClipKeys, playhead: Micros) -> Option<KeyStep> {
    let mut pressed = None;
    let on = keys.here.is_some();
    let buttons = [
        (
            KeyStep::Back,
            keys.before(playhead).is_some(),
            "Previous keyframe",
        ),
        (
            KeyStep::Toggle,
            true,
            if on {
                "Remove keyframe"
            } else {
                "Add keyframe"
            },
        ),
        (
            KeyStep::Forward,
            keys.after(playhead).is_some(),
            "Next keyframe",
        ),
    ];
    ui.spacing_mut().item_spacing.x = KEY_GAP;
    for (which, live, tip) in buttons {
        let sense = if live { Sense::click() } else { Sense::hover() };
        let (rect, response) = ui.allocate_exact_size(Vec2::new(KEY_BUTTON, 24.0), sense);
        let color = if !live {
            tokens::FAINT
        } else if response.hovered() {
            tokens::TEXT
        } else {
            tokens::MUTED
        };
        let c = rect.center();
        let painter = ui.painter();
        match which {
            KeyStep::Back | KeyStep::Forward => {
                let toward = if which == KeyStep::Back { -1.0 } else { 1.0 };
                let points = vec![
                    c + Vec2::new(4.0 * toward, 0.0),
                    c + Vec2::new(-3.0 * toward, -4.5),
                    c + Vec2::new(-3.0 * toward, 4.5),
                ];
                painter.add(egui::Shape::convex_polygon(points, color, Stroke::NONE));
            }
            KeyStep::Toggle => {
                let points = vec![
                    c + Vec2::new(0.0, -6.0),
                    c + Vec2::new(6.0, 0.0),
                    c + Vec2::new(0.0, 6.0),
                    c + Vec2::new(-6.0, 0.0),
                ];
                let (fill, edge) = if on {
                    (tokens::EXPORT, tokens::EXPORT)
                } else {
                    (Color32::TRANSPARENT, color)
                };
                painter.add(egui::Shape::convex_polygon(
                    points,
                    fill,
                    Stroke::new(1.2, edge),
                ));
            }
        }
        if response.on_hover_text(tip).clicked() {
            pressed = Some(which);
        }
    }
    pressed
}

/// The Graphs row for the span of keys the playhead is in: None, then
/// each curve drawn in its tile, the one easing the span outlined. A
/// press answers the curve to ease along, or `None` inside for none.
fn graph_row(ui: &mut Ui, keys: &cutlass_editor::ClipKeys) -> Option<Option<String>> {
    let mut picked = None;
    ui.add_space(ROW_GAP);
    row_label(ui, "Graphs");
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(6.0);
        let straight = cutlass_editor::KeyGraph {
            name: String::new(),
            points: vec![(0.0, 0.0), (1.0, 1.0)],
        };
        let curves = std::iter::once((None, straight)).chain(
            cutlass_editor::key_graphs()
                .into_iter()
                .map(|graph| (Some(graph.name.clone()), graph)),
        );
        for (name, graph) in curves {
            let on = name == keys.graph;
            let (rect, response) = ui.allocate_exact_size(Vec2::new(48.0, 40.0), Sense::click());
            let painter = ui.painter();
            painter.rect_filled(rect, CornerRadius::same(6), BOX);
            if on {
                painter.rect_stroke(
                    rect,
                    CornerRadius::same(6),
                    Stroke::new(tokens::SELECTED_EDGE * 2.0, tokens::EXPORT),
                    egui::StrokeKind::Inside,
                );
            }
            let inner = rect.shrink(9.0);
            let line: Vec<egui::Pos2> = graph
                .points
                .iter()
                .map(|&(x, y)| {
                    egui::pos2(
                        inner.left() + inner.width() * x as f32,
                        inner.bottom() - inner.height() * y as f32,
                    )
                })
                .collect();
            painter.add(egui::Shape::line(line, Stroke::new(1.5, tokens::TEXT)));
            let tip = name.as_deref().unwrap_or("None");
            if response.on_hover_text(tip).clicked() && !on {
                picked = Some(name);
            }
        }
    });
    ui.add_space(ROW_GAP);
    picked
}

/// One side's grid: None, then each preset of that side (a title's letter
/// animations only with `letters`), the chosen one outlined; then the chosen one's length, which previews as it drags.
fn animation_side(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    animations: &cutlass_editor::ClipAnimations,
    kind: AnimType,
    letters: bool,
) {
    let chosen = match kind {
        AnimType::In => &animations.enter,
        AnimType::Out => &animations.leave,
        _ => &animations.looped,
    };
    let chosen_id = chosen.as_ref().map(|side| side.resource_id.as_str());
    let mut picked = None;
    ui.add_space(8.0);
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(6.0);
        let presets = ANIMATION_PRESETS
            .iter()
            .filter(|preset| preset.kind == kind)
            .filter(|preset| letters || !matches!(preset.motion, Motion::Letters(_)));
        let tiles = std::iter::once(("None", None)).chain(
            presets.map(|preset| (preset.name, Some((preset.resource_id, preset.duration)))),
        );
        for (name, preset) in tiles {
            let on = preset.map(|(id, _)| id) == chosen_id;
            let stroke = if on {
                Stroke::new(tokens::SELECTED_EDGE * 2.0, tokens::EXPORT)
            } else {
                Stroke::NONE
            };
            let tile = egui::Button::new(RichText::new(name).size(11.0))
                .fill(BOX)
                .stroke(stroke);
            if ui.add_sized([64.0, 48.0], tile).clicked() && !on {
                picked = Some(preset);
            }
        }
    });
    if let Some(preset) = picked {
        let (id, duration) = preset.map_or((None, Micros(0)), |(id, duration)| {
            (Some(id.to_owned()), duration)
        });
        send(shell, |shell| shell.set_animation(kind, id, duration));
        return;
    }
    let Some(side) = chosen else {
        return;
    };
    ui.add_space(8.0);
    let seconds = side.duration.0 as f64 / 1e6;
    let longest = (animations.length.0 as f64 / 1e6).max(0.1);
    let (key, label) = match kind {
        AnimType::In => ("anim-in", "Duration"),
        AnimType::Out => ("anim-out", "Duration"),
        _ => ("anim-loop", "Speed"),
    };
    if let Some(value) = slider(ui, state, key, label, seconds, 0.1..=longest, |v| {
        format!("{v:.1}s")
    }) {
        let id = side.resource_id.clone();
        let duration = Micros((value * 1e6).round() as i64);
        send(shell, |shell| shell.set_animation(kind, Some(id), duration));
    }
}

/// The Adjust tab of a picture: the color rows, each to come.
fn adjust_tab(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, inspection: &Inspection) {
    pages(
        ui,
        &["Basic", "HSL", "Curves", "Color wheel", "Mask"],
        &mut state.page,
    );
    if colour_page(shell, ui, state, inspection) {
        return;
    }
    auto_adjust(shell, ui, state);
    color_match(shell, ui, state);
    super::audio::later_section(
        ui,
        state,
        "color-correction",
        "Color correction",
        Some("Correct the colour cast and exposure for you."),
        Some(&mut |ui: &mut Ui| super::audio::later_slider(ui, "Intensity", 80.0)),
    );
    lut_section(shell, ui, state, inspection);
    adjust_sliders(shell, ui, state, inspection);
    footer(ui, &["Apply to all", "Save as preset"]);
}

/// Auto adjust: Apply evens out the clip's exposure, contrast and colour
/// cast in one step. How strongly comes later.
fn auto_adjust(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState) {
    let header = section(ui, state, "auto-adjust", "Auto adjust", None, false);
    if !header.open {
        return;
    }
    body(ui, |ui| {
        ui.horizontal(|ui| {
            row_label(ui, "");
            if ui
                .add(ui::Button::new("Apply").text_size(12.0))
                .on_hover_text("Even out exposure, contrast and colour cast")
                .clicked()
            {
                send(shell, Shell::auto_adjust);
            }
        });
        ui.add_space(ROW_GAP);
        super::audio::later_value(ui, "Intensity", 80.0);
    });
}

/// Color match: pick another clip and this one takes its colours. A still
/// picked from the clip or an imported image, and how strongly, come later.
fn color_match(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState) {
    let header = section(ui, state, "color-match", "Color match", None, false);
    if !header.open {
        return;
    }
    let choices = shell.match_choices();
    let mut matched = None;
    body(ui, |ui| {
        ui.label(
            RichText::new("Pick a clip to match, or import an image.")
                .size(12.0)
                .color(tokens::MUTED),
        );
        ui.add_space(ROW_GAP);
        label_row(ui, "Clip", |ui| {
            ui.add_enabled_ui(!choices.is_empty(), |ui| {
                egui::ComboBox::from_id_salt("color-match")
                    .selected_text(RichText::new("Match to…").size(12.0))
                    .show_ui(ui, |ui| {
                        for (id, name) in &choices {
                            if ui.add(ui::Choice::new(false, name).wide()).clicked() {
                                matched = Some(id.clone());
                            }
                        }
                    })
                    .response
                    .on_disabled_hover_text("Add another picture clip to match");
            });
        });
        ui.add_space(ROW_GAP);
        ui.add_enabled_ui(false, |ui| {
            ui.horizontal(|ui| {
                let (rect, response) =
                    ui.allocate_exact_size(Vec2::splat(CATALOG_TILE), Sense::click());
                response.widget_info(|| {
                    egui::WidgetInfo::labeled(egui::Role::Button, false, "Import an image")
                });
                tile_frame(ui, rect, &response, false);
                let (centre, arm) = (rect.center(), 12.0);
                let stroke = Stroke::new(2.0, tokens::MUTED);
                ui.painter().line_segment(
                    [centre - Vec2::new(arm, 0.0), centre + Vec2::new(arm, 0.0)],
                    stroke,
                );
                ui.painter().line_segment(
                    [centre - Vec2::new(0.0, arm), centre + Vec2::new(0.0, arm)],
                    stroke,
                );
                later_badge(ui);
            });
        })
        .response
        .on_disabled_hover_text(LATER);
        ui.add_space(ROW_GAP);
        super::audio::later_value(ui, "Intensity", 80.0);
    });
    if let Some(reference) = matched {
        send(shell, |shell| shell.match_colour(&reference));
    }
}

fn colour_page(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) -> bool {
    match state.page {
        0 => return false,
        1 => super::grade::hsl_page(shell, ui, state, inspection),
        2 => super::grade::curves_page(shell, ui, state, inspection),
        3 => super::grade::wheels_page(shell, ui, state, inspection),
        _ => mask_page(shell, ui, state, inspection.color_mask, true),
    }
    true
}

/// The Cutout page: chroma key, with the cutouts that need a model marked
/// for later.
/// The Mask page: a shape cuts the clip in its own frame, or with `colour`
/// limits where its colour work shows. None takes the mask off; the
/// sliders place, size, turn and soften it.
fn mask_page(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    mask: Option<cutlass_effects::Mask>,
    colour: bool,
) {
    use cutlass_effects::{Mask, MaskShape};
    let set: fn(&mut Shell, Option<Mask>) -> Result<(), ShellError> = if colour {
        Shell::set_color_mask
    } else {
        Shell::set_mask
    };
    let prefix = if colour { "color-" } else { "" };
    body(ui, |ui| {
        let mut picked: Option<Option<MaskShape>> = None;
        ui.horizontal_wrapped(|ui| {
            if ui
                .add(ui::Choice::new(
                    mask.is_none(),
                    RichText::new("None").size(12.0),
                ))
                .clicked()
            {
                picked = Some(None);
            }
            for shape in MaskShape::ALL {
                let on = mask.is_some_and(|mask| mask.shape == shape);
                if ui
                    .add(ui::Choice::new(on, RichText::new(shape.label()).size(12.0)))
                    .clicked()
                {
                    picked = Some(Some(shape));
                }
            }
        });
        match picked {
            Some(None) if mask.is_some() => send(shell, |shell| set(shell, None)),
            Some(Some(shape)) if mask.is_none_or(|mask| mask.shape != shape) => {
                let next = mask.map_or_else(|| Mask::new(shape), |mask| Mask { shape, ..mask });
                send(shell, |shell| set(shell, Some(next)));
            }
            _ => {}
        }
        if !colour && mask.is_some() {
            let keys = shell.setting_keys(KeyedSetting::Mask);
            let playhead = shell.playhead();
            let mut step = None;
            keyed_row(ui, keys.as_ref(), playhead, &mut step, |ui| {
                ui.label(
                    RichText::new("Keyframe the mask")
                        .size(12.0)
                        .color(tokens::MUTED),
                );
            });
            setting_step(shell, KeyedSetting::Mask, step);
        }
        let Some(mask) = mask else {
            ui.add_space(6.0);
            ui.label(
                RichText::new(if colour {
                    "Pick a shape to keep the colour work inside it."
                } else {
                    "Pick a shape to mask the clip."
                })
                .size(11.0)
                .color(tokens::MUTED),
            );
            return;
        };
        ui.add_space(6.0);
        let percent = |v: f64| format!("{v:.0}%");
        let mut changed = None;
        let mut field = |ui: &mut Ui,
                         state: &mut InspectorState,
                         key: &str,
                         label: &str,
                         stored: f32,
                         scale: f64,
                         range: std::ops::RangeInclusive<f64>,
                         show: &dyn Fn(f64) -> String,
                         put: fn(&mut Mask, f32)| {
            if let Some(value) = slider(
                ui,
                state,
                &format!("{prefix}{key}"),
                label,
                (f64::from(stored) * scale).round(),
                range,
                show,
            ) {
                let mut next = mask;
                #[allow(clippy::cast_possible_truncation)]
                put(&mut next, (value / scale) as f32);
                changed = Some(next);
            }
        };
        field(
            ui,
            state,
            "mask-x",
            "X",
            mask.x,
            100.0,
            -200.0..=200.0,
            &percent,
            |m, v| m.x = v,
        );
        field(
            ui,
            state,
            "mask-y",
            "Y",
            mask.y,
            100.0,
            -200.0..=200.0,
            &percent,
            |m, v| m.y = v,
        );
        match mask.shape {
            MaskShape::Linear => {}
            MaskShape::Mirror => {
                field(
                    ui,
                    state,
                    "mask-height",
                    "Band",
                    mask.height,
                    100.0,
                    0.0..=400.0,
                    &percent,
                    |m, v| m.height = v,
                );
            }
            _ => {
                field(
                    ui,
                    state,
                    "mask-width",
                    "Width",
                    mask.width,
                    100.0,
                    0.0..=400.0,
                    &percent,
                    |m, v| m.width = v,
                );
                field(
                    ui,
                    state,
                    "mask-height",
                    "Height",
                    mask.height,
                    100.0,
                    0.0..=400.0,
                    &percent,
                    |m, v| m.height = v,
                );
            }
        }
        field(
            ui,
            state,
            "mask-rotation",
            "Rotate",
            mask.rotation,
            1.0,
            -180.0..=180.0,
            &|v| format!("{v:.0}°"),
            |m, v| m.rotation = v,
        );
        field(
            ui,
            state,
            "mask-feather",
            "Feather",
            mask.feather,
            100.0,
            0.0..=100.0,
            &|v| format!("{v:.0}"),
            |m, v| m.feather = v,
        );
        if mask.shape == MaskShape::Rectangle {
            field(
                ui,
                state,
                "mask-round",
                "Round corners",
                mask.round,
                100.0,
                0.0..=100.0,
                &|v| format!("{v:.0}"),
                |m, v| m.round = v,
            );
        }
        if switch_row(ui, "Invert", mask.invert, true).clicked() {
            changed = Some(Mask {
                invert: !mask.invert,
                ..mask
            });
        }
        if let Some(next) = changed {
            send(shell, |shell| set(shell, Some(next)));
        }
    });
}

/// The LUT section: the switch takes the LUT off or asks for a file; the
/// body names the table, picks another, and sets its strength.
fn lut_section(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let on = inspection.lut.is_some();
    let header = section(ui, state, "lut", "LUT", Some((on, true)), false);
    if header.toggled {
        if on {
            send(shell, Shell::remove_lut);
        } else {
            state.wants_lut = true;
            state.open.insert("lut".to_owned());
        }
    }
    if !header.open && !state.open.contains("lut") {
        return;
    }
    body(ui, |ui| {
        ui.horizontal(|ui| {
            row_label(ui, "Name");
            let name = inspection
                .lut
                .as_ref()
                .map_or("None", |(name, _)| name.as_str());
            ui.label(RichText::new(name).size(12.0).color(Color32::WHITE));
            if ui
                .add(ui::Button::new("Choose .cube…").text_size(12.0))
                .clicked()
            {
                state.wants_lut = true;
            }
        });
        ui.add_space(ROW_GAP);
        if let Some((_, strength)) = &inspection.lut {
            let stored = (f64::from(*strength) * 100.0).round();
            if let Some(value) = slider(
                ui,
                state,
                "lut-strength",
                "Intensity",
                stored,
                0.0..=100.0,
                |v| format!("{v:.0}"),
            ) {
                send(shell, |shell| shell.set_lut_strength(value / 100.0));
            }
        }
        ui.add_enabled_ui(false, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Protect skin tone").size(12.0));
                        later_badge(ui);
                    });
                    ui.label(
                        RichText::new("Keep skin tones natural under the LUT.")
                            .size(11.0)
                            .color(tokens::MUTED),
                    );
                });
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    check_box(ui, false, "Protect skin tone");
                });
            });
        })
        .response
        .on_disabled_hover_text(LATER);
    });
}

/// The Adjust section: a switch that resets every slider, and the sliders.
fn adjust_sliders(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let on = !inspection.adjust.is_identity();
    let header = section(ui, state, "adjust", "Adjust", Some((on, true)), false);
    if header.toggled {
        if on {
            send(shell, Shell::reset_adjust);
        } else {
            state.open.insert("adjust".to_owned());
        }
    }
    if !header.open && !state.open.contains("adjust") {
        return;
    }
    use cutlass_effects::AdjustKey as Key;
    // Grouped as the reference groups them; Brightness and Hue, which it
    // does not show, sit with their kind.
    let groups: [(&str, &[Key]); 3] = [
        (
            "Color",
            &[Key::Temperature, Key::Tint, Key::Saturation, Key::Hue],
        ),
        (
            "Lightness",
            &[
                Key::Exposure,
                Key::Brightness,
                Key::Contrast,
                Key::Highlights,
                Key::Shadows,
                Key::Whites,
                Key::Blacks,
                Key::Brilliance,
            ],
        ),
        (
            "Effects",
            &[
                Key::Sharpen,
                Key::Clarity,
                Key::Grain,
                Key::Fade,
                Key::Vignette,
            ],
        ),
    ];
    body(ui, |ui| {
        for (title, keys) in groups {
            ui.add_space(4.0);
            ui.label(RichText::new(title).size(12.0).color(tokens::MUTED));
            divider(ui);
            for key in keys {
                let key = *key;
                let stored = (f64::from(inspection.adjust.get(key)) * 100.0).round();
                let low = f64::from(key.min()) * 100.0;
                let id = format!("adjust-{}", key.kind());
                if let Some(value) = slider(ui, state, &id, key.label(), stored, low..=100.0, |v| {
                    format!("{v:.0}")
                }) {
                    send(shell, |shell| shell.set_adjust(key, value / 100.0));
                }
            }
        }
    });
}

// The text panel's Basic page.

fn text_basic(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
    caption: bool,
) {
    body(ui, |ui| {
        if caption && !inspection.words.text.is_empty() {
            caption_paragraphs(shell, ui, state);
            egui::CollapsingHeader::new(RichText::new("Words").size(12.0))
                .id_salt("caption words")
                .show(ui, |ui| caption_words(shell, ui, state, inspection));
            #[allow(clippy::cast_precision_loss)]
            let stored = inspection.caption_stagger_ms as f64;
            #[allow(clippy::cast_precision_loss)]
            let most = cutlass_session::MAX_CAPTION_STAGGER_MS as f64;
            if let Some(value) = slider(
                ui,
                state,
                "caption stagger",
                "Stagger",
                stored,
                0.0..=most,
                |value| format!("{value:.0} ms"),
            ) {
                #[allow(clippy::cast_possible_truncation)]
                let millis = value.round() as i64;
                send(shell, |shell| shell.set_caption_stagger(millis));
            }
        } else {
            words_box(shell, ui, state);
        }
        font_row(shell, ui, inspection);
        let size = inspection.text_size.round();
        if let Some(value) = slider(
            ui,
            state,
            "font-size",
            "Font size",
            size,
            1.0..=512.0,
            |v| format!("{v:.0}"),
        ) {
            send(shell, |shell| shell.set_text_size(value));
        }
        pattern_row(shell, ui, inspection);
        label_row(ui, "Case", |ui| {
            for (label, tip, case) in [
                ("TT", "Upper case", LetterCase::Upper),
                ("tt", "Lower case", LetterCase::Lower),
                ("Tt", "Title case", LetterCase::Title),
            ] {
                if ui
                    .add(ui::Button::new(label).text_size(12.0))
                    .on_hover_text(tip)
                    .clicked()
                {
                    send(shell, |shell| shell.set_text_case(case));
                }
            }
        });
        let keys = shell.setting_keys(KeyedSetting::TextColor);
        let playhead = shell.playhead();
        let mut step = None;
        let picked = keyed_row(ui, keys.as_ref(), playhead, &mut step, |ui| {
            color_row(ui, state, "text-color", "Color", &inspection.text_color)
        });
        if let Some(color) = picked {
            send(shell, |shell| shell.set_text_color(&color));
        }
        setting_step(shell, KeyedSetting::TextColor, step);
        spacing_row(shell, ui, state, inspection);
        let (width, _) = pair_row(
            ui,
            state,
            "Line width",
            ("px", inspection.line_width.round()),
            ("", f64::NAN),
        );
        if let Some(value) = width {
            send(shell, |shell| shell.set_line_width(value.max(0.0)));
        }
        alignment_row(shell, ui, inspection);
        preset_styles(shell, ui);
    });
    if caption {
        transform(shell, ui, state, inspection);
    } else {
        position_size(shell, ui, state, inspection, "Plane rotation");
    }
    blend(shell, ui, state, inspection, false);
    if let Some(look) = shell.text_look() {
        text_parts(shell, ui, state, &look);
    }
}

fn words_box(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState) {
    let stored = shell.text_words().unwrap_or_default();
    let text = state.fields.entry("words".to_owned()).or_default();
    let response = ui::Field::multiline(text, 3).name("Text").show(ui).response;
    if super::commits(ui, &response) && *text != stored {
        let content = text.clone();
        send(shell, |shell| shell.set_text(content));
    } else if !response.has_focus() {
        stored.clone_into(text);
    }
    ui.add_space(6.0);
}

/// One field per paragraph, "The N paragraph", each a page or line of the
/// caption as it shows. Typing over one rematches only its words.
fn caption_paragraphs(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState) {
    let paragraphs = shell.caption_paragraphs();
    let mut edited = None;
    for (index, paragraph) in paragraphs.iter().enumerate() {
        ui.add_space(6.0);
        ui.label(
            RichText::new(format!("The {} paragraph", index + 1))
                .size(12.0)
                .strong()
                .color(tokens::TEXT),
        );
        let text = state
            .fields
            .entry(format!("paragraph {index}"))
            .or_default();
        let response = ui::Field::new(text).show(ui).response;
        if super::commits(ui, &response) && *text != paragraph.text {
            edited = Some((index, text.clone()));
        } else if !response.has_focus() {
            paragraph.text.clone_into(text);
        }
    }
    ui.add_space(6.0);
    if let Some((index, text)) = edited {
        send(shell, |shell| shell.set_caption_paragraph(index, &text));
    }
}

/// One row per recognized word: its time and its letters. A word keeps
/// its time when its letters change.
fn caption_words(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let columns = &inspection.words;
    let count = columns.text.len();
    for (index, word) in columns.text.iter().enumerate() {
        // A line break shows as a rule between the lines' words.
        if word == cutlass_editor::LINE_BREAK {
            ui.separator();
            continue;
        }
        let start = columns.start_time.get(index).copied().unwrap_or_default();
        let text = state.fields.entry(format!("word {index}")).or_default();
        let mut edited = None;
        let mut join = false;
        let mut line = false;
        let mut mark = None;
        label_row(ui, &format!("{:.2} s", start as f64 / 1000.0), |ui| {
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let marked = inspection.caption_keywords.contains(&index);
                if ui
                    .add(ui::Button::new("★").text_size(11.0).on(marked))
                    .on_hover_text(if marked {
                        "Take the keyword mark off"
                    } else {
                        "Mark as a keyword, drawn in the spoken look"
                    })
                    .clicked()
                {
                    mark = Some(!marked);
                }
                let next_is_word = columns
                    .text
                    .get(index + 1)
                    .is_some_and(|next| next != cutlass_editor::LINE_BREAK);
                join = ui
                    .add_enabled(
                        index + 1 < count && next_is_word,
                        ui::Button::new("+").text_size(11.0),
                    )
                    .on_hover_text("Join with the next word")
                    .clicked();
                let broken = cutlass_editor::breaks_before(columns, index);
                line = ui
                    .add_enabled(index > 0, ui::Button::new("↵").text_size(11.0).on(broken))
                    .on_hover_text(if broken {
                        "Join this line to the one before"
                    } else {
                        "Start a new line here"
                    })
                    .clicked();
                let response = ui::Field::new(text).show(ui).response;
                if super::commits(ui, &response) && text != word {
                    edited = Some(text.clone());
                }
                if !response.has_focus() {
                    word.clone_into(text);
                }
            });
        });
        if let Some(on) = mark {
            send(shell, |shell| shell.set_caption_keyword(index, on));
        } else if let Some(value) = edited {
            send(shell, |shell| shell.set_caption_word(index, &value));
        } else if join {
            send(shell, |shell| shell.join_caption_words(index));
        } else if line {
            send(shell, |shell| shell.toggle_caption_break(index));
        }
    }
    ui.add_space(6.0);
}

/// Bold, underline and italic, each a switch over the whole text.
fn pattern_row(shell: &mut Shell, ui: &mut Ui, inspection: &Inspection) {
    let stored = inspection.text_pattern;
    let mut pattern = stored;
    label_row(ui, "Pattern", |ui| {
        if icon_toggle(ui, "bold", "Bold", stored.bold).clicked() {
            pattern.bold = !stored.bold;
        }
        if icon_toggle(ui, "underline", "Underline", stored.underline).clicked() {
            pattern.underline = !stored.underline;
        }
        if icon_toggle(ui, "italic", "Italic", stored.italic).clicked() {
            pattern.italic = !stored.italic;
        }
    });
    if pattern != stored {
        send(shell, |shell| shell.set_text_pattern(pattern));
    }
}

fn font_row(shell: &mut Shell, ui: &mut Ui, inspection: &Inspection) {
    let faces = shell.text_faces();
    let current = if inspection.font.is_empty() {
        "System".to_owned()
    } else {
        face_label(&inspection.font)
    };
    let mut picked = None;
    label_row(ui, "Font", |ui| {
        let shown = current.clone();
        egui::ComboBox::from_id_salt("text-font")
            .selected_text(current)
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for face in &faces {
                    if ui
                        .add(ui::Choice::new(*face == inspection.font, face_label(face)).wide())
                        .clicked()
                    {
                        picked = Some(face.clone());
                    }
                }
            })
            .response
            .widget_info(|| {
                let mut info = egui::WidgetInfo::labeled(egui::Role::ComboBox, true, "Font");
                info.current_text_value = Some(shown.clone());
                info
            });
    });
    if let Some(face) = picked {
        send(shell, |shell| shell.set_text_face(&face));
    }
}

fn face_label(face: &str) -> String {
    let name = face.rsplit(['/', '\\']).next().unwrap_or(face);
    name.rsplit_once('.')
        .map_or(name, |(stem, _)| stem)
        .to_owned()
}

fn spacing_row(
    shell: &mut Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    inspection: &Inspection,
) {
    let (letters, lines) = pair_row(
        ui,
        state,
        "Character",
        ("", inspection.letter_spacing),
        ("Line", inspection.line_spacing),
    );
    if let Some(value) = letters {
        send(shell, |shell| shell.set_letter_spacing(value));
    }
    if let Some(value) = lines {
        send(shell, |shell| shell.set_line_spacing(value));
    }
}

fn alignment_row(shell: &mut Shell, ui: &mut Ui, inspection: &Inspection) {
    let mut picked = None;
    label_row(ui, "Alignment", |ui| {
        for (index, (icon, tip)) in [
            ("textLeft", "Align left"),
            ("textCenter", "Align center"),
            ("textRight", "Align right"),
            ("alignTop", "Vertical text, top"),
            ("alignMiddle", "Vertical text, middle"),
            ("alignBottom", "Vertical text, bottom"),
        ]
        .into_iter()
        .enumerate()
        {
            let alignment = i32::try_from(index).unwrap_or_default();
            if icon_toggle(ui, icon, tip, inspection.text_align == alignment).clicked() {
                picked = Some(alignment);
            }
        }
    });
    if let Some(alignment) = picked.filter(|alignment| *alignment != inspection.text_align) {
        send(shell, |shell| shell.set_text_align(alignment));
    }
}

/// The preset tiles, each drawn in its whole look: letters, outline, box
/// and shadow. Picking one gives the text that look in one step.
use cutlass_editor::TEXT_PRESETS as PRESETS;

/// A press on a preset tile: a built-in look, a saved one, or deleting a
/// saved one.
#[derive(Clone, Copy)]
enum Pick {
    Built(usize),
    Saved(usize),
    Delete(usize),
}

fn preset_styles(shell: &mut Shell, ui: &mut Ui) {
    ui.add_space(6.0);
    ui.label(
        RichText::new("Preset style")
            .size(12.0)
            .color(tokens::MUTED),
    );
    let mut picked = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(6.0);
        for (index, preset) in PRESETS.iter().enumerate() {
            let (rect, response) = ui.allocate_exact_size(Vec2::splat(36.0), Sense::click());
            ui.painter().rect_filled(rect, CornerRadius::same(6), BOX);
            if index == 0 {
                paint_icon(ui, rect, "unavailable", tokens::MUTED);
            } else {
                preset_tile(
                    ui,
                    rect,
                    [preset.color, preset.outline, preset.plate, preset.shadow],
                );
            }
            if response.on_hover_text(preset.name).clicked() {
                picked = Some(Pick::Built(index));
            }
        }
        for (index, look) in shell.saved_looks().iter().enumerate() {
            let (rect, response) = ui.allocate_exact_size(Vec2::splat(36.0), Sense::click());
            ui.painter().rect_filled(rect, CornerRadius::same(6), BOX);
            preset_tile(
                ui,
                rect,
                [
                    &look.color,
                    &look.outline_color,
                    &look.background_color,
                    &look.shadow_color,
                ],
            );
            let response = response.on_hover_text(&look.name);
            if response.clicked() {
                picked = Some(Pick::Saved(index));
            }
            response.context_menu(|ui| {
                if ui.button("Delete preset").clicked() {
                    picked = Some(Pick::Delete(index));
                    ui.close();
                }
            });
        }
    });
    match picked {
        Some(Pick::Built(index)) => send(shell, |shell| shell.set_text_preset(index)),
        Some(Pick::Saved(index)) => send(shell, |shell| shell.set_saved_preset(index)),
        Some(Pick::Delete(index)) => send(shell, |shell| shell.delete_text_preset(index)),
        None => {}
    }
    ui.add_space(6.0);
}

/// "Aa" in one preset's look, in a tile: its letters', outline, box and
/// shadow colors, each left off when empty.
fn preset_tile(ui: &Ui, rect: Rect, [color, outline, plate, shadow]: [&str; 4]) {
    let painter = ui.painter();
    let font = FontId::proportional(14.0);
    let center = rect.center();
    if let Some(plate) = parse_hex(plate) {
        let back = Rect::from_center_size(center, Vec2::new(28.0, 20.0));
        painter.rect_filled(back, CornerRadius::same(4), plate);
    }
    let letters = |at: egui::Pos2, color: Color32| {
        painter.text(at, Align2::CENTER_CENTER, "Aa", font.clone(), color);
    };
    if let Some(shadow) = parse_hex(shadow) {
        letters(center + Vec2::new(1.5, 1.5), shadow);
    }
    if let Some(outline) = parse_hex(outline) {
        for (x, y) in [
            (-1.0, 0.0),
            (1.0, 0.0),
            (0.0, -1.0),
            (0.0, 1.0),
            (-1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
        ] {
            letters(center + Vec2::new(x, y), outline);
        }
    }
    letters(center, parse_hex(color).unwrap_or(Color32::WHITE));
}

/// Stroke, Background, Glow, and Shadow. A part's switch turns it on in
/// the prototype's first color, or off.
fn text_parts(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState, look: &TextLook) {
    let percent = |v: f64| format!("{v:.0}");

    let on = !look.outline_color.is_empty();
    let header = section(ui, state, "stroke", "Stroke", Some((on, true)), false);
    if header.toggled {
        let color = if on { "" } else { "#FFFFFF" };
        send(shell, |shell| {
            shell.set_text_outline(color, look.outline_width)
        });
    }
    if header.open {
        body(ui, |ui| {
            if let Some(color) = color_row(ui, state, "stroke-color", "Color", &look.outline_color)
            {
                send(shell, |shell| {
                    shell.set_text_outline(&color, look.outline_width)
                });
            }
            // Thickness 100 is the widest outline a title takes.
            let thickness = (look.outline_width / 0.25 * 100.0).round();
            if let Some(value) = slider(
                ui,
                state,
                "thickness",
                "Thickness",
                thickness,
                0.0..=100.0,
                percent,
            ) {
                send(shell, |shell| {
                    shell.set_text_outline(&look.outline_color, value / 100.0 * 0.25)
                });
            }
        });
    }

    let on = !look.background_color.is_empty();
    let header = section(
        ui,
        state,
        "background",
        "Background",
        Some((on, true)),
        false,
    );
    if header.toggled {
        let color = if on { "" } else { "#000000" };
        send(shell, |shell| {
            shell.set_text_background(color, look.background_alpha, look.background_round)
        });
    }
    if header.open {
        body(ui, |ui| {
            if let Some(color) = color_row(
                ui,
                state,
                "background-color",
                "Color",
                &look.background_color,
            ) {
                send(shell, |shell| {
                    shell.set_text_background(&color, look.background_alpha, look.background_round)
                });
            }
            let alpha = (look.background_alpha * 100.0).round();
            if let Some(value) = slider(
                ui,
                state,
                "box-opacity",
                "Opacity",
                alpha,
                0.0..=100.0,
                |v| format!("{v:.0}%"),
            ) {
                send(shell, |shell| {
                    shell.set_text_background(
                        &look.background_color,
                        value / 100.0,
                        look.background_round,
                    )
                });
            }
        });
    }

    let on = !look.glow_color.is_empty();
    let header = section(ui, state, "glow", "Glow", Some((on, true)), false);
    if header.toggled {
        let color = if on { "" } else { "#FFFFFF" };
        send(shell, |shell| {
            shell.set_text_glow(color, look.glow_strength, look.glow_range)
        });
    }
    if header.open {
        body(ui, |ui| {
            later_tiles(ui, &["Soft glow", "Halo"]);
            if let Some(color) = color_row(ui, state, "glow-color", "Color", &look.glow_color) {
                send(shell, |shell| {
                    shell.set_text_glow(&color, look.glow_strength, look.glow_range)
                });
            }
            let strength = (look.glow_strength * 100.0).round();
            if let Some(value) = slider(
                ui,
                state,
                "intensity",
                "Intensity",
                strength,
                0.0..=100.0,
                percent,
            ) {
                send(shell, |shell| {
                    shell.set_text_glow(&look.glow_color, value / 100.0, look.glow_range)
                });
            }
            let range = (look.glow_range * 100.0).round();
            if let Some(value) = slider(ui, state, "range", "Range", range, 0.0..=100.0, percent) {
                send(shell, |shell| {
                    shell.set_text_glow(&look.glow_color, look.glow_strength, value / 100.0)
                });
            }
            later_slider(ui, "Y-offset", 0.0, "0°");
            later_slider(ui, "X-offset", 0.0, "0°");
        });
    }

    let on = !look.shadow_color.is_empty();
    let header = section(ui, state, "shadow", "Shadow", Some((on, true)), false);
    if header.toggled {
        let color = if on { "" } else { "#000000" };
        send(shell, |shell| {
            shell.set_text_shadow(
                color,
                look.shadow_distance,
                look.shadow_angle,
                look.shadow_blur,
            )
        });
    }
    if header.open {
        body(ui, |ui| {
            if let Some(color) = color_row(ui, state, "shadow-color", "Color", &look.shadow_color) {
                send(shell, |shell| {
                    shell.set_text_shadow(
                        &color,
                        look.shadow_distance,
                        look.shadow_angle,
                        look.shadow_blur,
                    )
                });
            }
            later_slider(ui, "Opacity", 90.0, "90%");
            let blur = (look.shadow_blur * 100.0).round();
            if let Some(value) = slider(ui, state, "blur", "Bluriness", blur, 0.0..=100.0, |v| {
                format!("{v:.0}%")
            }) {
                send(shell, |shell| {
                    shell.set_text_shadow(
                        &look.shadow_color,
                        look.shadow_distance,
                        look.shadow_angle,
                        value / 100.0,
                    )
                });
            }
            let distance = look.shadow_distance.round();
            if let Some(value) = slider(
                ui,
                state,
                "distance",
                "Distance",
                distance,
                0.0..=100.0,
                percent,
            ) {
                send(shell, |shell| {
                    shell.set_text_shadow(
                        &look.shadow_color,
                        value,
                        look.shadow_angle,
                        look.shadow_blur,
                    )
                });
            }
            let angle = look.shadow_angle.round();
            if let Some(value) = slider(ui, state, "angle", "Angle", angle, -180.0..=180.0, |v| {
                format!("{v:.0}°")
            }) {
                send(shell, |shell| {
                    shell.set_text_shadow(
                        &look.shadow_color,
                        look.shadow_distance,
                        value,
                        look.shadow_blur,
                    )
                });
            }
        });
    }
}

// Controls.

/// Send one edit and show its refusal.
pub(super) fn send(shell: &mut Shell, edit: impl FnOnce(&mut Shell) -> Result<(), ShellError>) {
    if let Err(error) = edit(shell) {
        shell.notice_error(&error);
    }
}

/// The text tabs across the top. The active one is cyan. Tabs that do not
/// fit scroll sideways, as the prototype's do.
pub(super) fn tabs(ui: &mut Ui, labels: &[&str], current: &mut usize) {
    paid_tabs(ui, labels, &[], current);
}

/// What a paid tab says on hover. Cutlass will offer these later.
const PAID: &str = "Paid: Cutlass will offer this later";
/// The width of the diamond after a paid tab's name.
const GEM: f32 = 8.0;

/// The tab row, with the `paid` tabs faint, marked with a diamond, and not
/// picked.
fn paid_tabs(ui: &mut Ui, labels: &[&str], paid: &[&str], current: &mut usize) {
    ui.add_space(ROW_GAP);
    // The gap shrinks first, then the names down to 11 points, before the
    // row has to scroll, so a narrow inspector still shows every tab.
    let measure = |size: f32| -> f32 {
        let font = FontId::proportional(size);
        let names: f32 = labels
            .iter()
            .map(|label| {
                ui.painter()
                    .layout_no_wrap((*label).to_owned(), font.clone(), Color32::WHITE)
                    .size()
                    .x
            })
            .sum();
        names + paid.len() as f32 * (GEM + 3.0)
    };
    let gaps = labels.len().saturating_sub(1).max(1) as f32;
    let room = ui.available_width() - 1.0;
    let full = measure(13.0);
    let size = if full + gaps * TAB_GAP_MIN <= room {
        13.0
    } else {
        (13.0 * (room - gaps * TAB_GAP_MIN) / full).clamp(11.0, 13.0)
    };
    let font = FontId::proportional(size);
    let text = measure(size);
    let gap = ((room - text) / gaps).clamp(TAB_GAP_MIN, TAB_GAP);
    ui::Scroll::horizontal(("inspector-tabs", labels.len()))
        .without_bar()
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.set_min_height(ROW_HEIGHT);
                ui.spacing_mut().item_spacing.x = gap;
                for (index, label) in labels.iter().enumerate() {
                    if paid.contains(label) {
                        // The name and a diamond after it, faint, in one
                        // rect sized to them.
                        let galley = ui.painter().layout_no_wrap(
                            (*label).to_owned(),
                            font.clone(),
                            tokens::FAINT,
                        );
                        let size = Vec2::new(galley.size().x + 3.0 + GEM, galley.size().y);
                        let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
                        let top = rect.left_center() - Vec2::new(0.0, galley.size().y / 2.0);
                        ui.painter().galley(top, galley, tokens::FAINT);
                        let c = egui::pos2(rect.right() - GEM / 2.0, rect.center().y);
                        let points = vec![
                            c + Vec2::new(-4.0, -1.0),
                            c + Vec2::new(-2.0, -4.0),
                            c + Vec2::new(2.0, -4.0),
                            c + Vec2::new(4.0, -1.0),
                            c + Vec2::new(0.0, 4.0),
                        ];
                        ui.painter().add(egui::Shape::convex_polygon(
                            points,
                            tokens::FAINT,
                            Stroke::NONE,
                        ));
                        response.on_hover_text(PAID).widget_info(|| {
                            egui::WidgetInfo::labeled(egui::Role::Label, false, *label)
                        });
                        continue;
                    }
                    let color = if index == *current {
                        ACCENT
                    } else {
                        tokens::MUTED
                    };
                    let response = ui.add(
                        egui::Label::new(RichText::new(*label).size(size).color(color))
                            .selectable(false)
                            .sense(Sense::click()),
                    );
                    if response.clicked() {
                        *current = index;
                    }
                }
            });
        });
    ui.add_space(6.0);
    let line = ui.cursor().min.y;
    let span = ui.max_rect().x_range();
    ui.painter()
        .hline(span, line, Stroke::new(1.0, Color32::from_white_alpha(20)));
}

/// The rounded page switch under the tabs.
pub(super) fn pages(ui: &mut Ui, labels: &[&str], current: &mut usize) {
    ui.add_space(12.0);
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 30.0), Sense::hover());
    ui.painter()
        .rect_filled(rect, CornerRadius::same(10), PAGES);
    let count = labels.len().max(1) as f32;
    let font = FontId::proportional(12.0);
    let widths: Vec<f32> = labels
        .iter()
        .map(|label| {
            ui.painter()
                .layout_no_wrap((*label).to_owned(), font.clone(), Color32::WHITE)
                .size()
                .x
        })
        .collect();
    let spare = ((rect.width() - 6.0 - widths.iter().sum::<f32>()) / count).max(0.0);
    let mut left = rect.min.x + 3.0;
    for (index, label) in labels.iter().enumerate() {
        let cell = Rect::from_min_size(
            egui::pos2(left, rect.min.y + 3.0),
            Vec2::new(widths[index] + spare, 24.0),
        );
        left += cell.width();
        let response = ui.interact(
            cell,
            ui.make_persistent_id(("page", label, index)),
            Sense::click(),
        );
        if response.clicked() {
            *current = index;
        }
        let on = index == *current;
        if on {
            ui.painter()
                .rect_filled(cell, CornerRadius::same(8), PAGE_ON);
        }
        ui.painter().text(
            cell.center(),
            Align2::CENTER_CENTER,
            *label,
            font.clone(),
            if on { Color32::WHITE } else { tokens::MUTED },
        );
    }
    ui.add_space(ROW_GAP);
}

pub(super) struct Header {
    pub(super) open: bool,
    pub(super) toggled: bool,
    pub(super) reset: bool,
}

/// A section header: an optional on switch (`(on, editable)`), the name
/// with a chevron that folds the section, and, with `keys`, the reset and
/// keyframe buttons.
pub(super) fn section(
    ui: &mut Ui,
    state: &mut InspectorState,
    key: &str,
    label: &str,
    check: Option<(bool, bool)>,
    keys: bool,
) -> Header {
    let mut header = Header {
        open: state.open.contains(key),
        toggled: false,
        reset: false,
    };
    ui.horizontal(|ui| {
        ui.set_min_height(40.0);
        if let Some((on, editable)) = check {
            let response = ui
                .add_enabled_ui(editable, |ui| check_box(ui, on, label))
                .inner;
            header.toggled = response.clicked();
        }
        ui.spacing_mut().item_spacing.x = 6.0;
        let name = if check.is_some() {
            let name = section_name(ui, label);
            name.union(chevron(ui, header.open))
        } else {
            let fold = chevron(ui, header.open);
            section_name(ui, label).union(fold)
        };
        if name.clicked() {
            header.open = !header.open;
            if header.open {
                state.open.insert(key.to_owned());
            } else {
                state.open.remove(key);
            }
        }
        if keys {
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.add_enabled_ui(false, |ui| {
                    icon_toggle(ui, "chevron", "Next keyframe", false);
                    icon_toggle(ui, "keyframe", "Add keyframe", false);
                    icon_toggle(ui, "chevronLeft", "Previous keyframe", false);
                })
                .response
                .on_disabled_hover_text(LATER);
                header.reset = icon_toggle(ui, "reset", &format!("Reset {label}"), false).clicked();
            });
        }
    });
    if header.toggled && !header.open {
        header.open = true;
        state.open.insert(key.to_owned());
    }
    divider(ui);
    header
}

fn section_name(ui: &mut Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Label::new(RichText::new(label).size(13.0).color(Color32::WHITE))
            .selectable(false)
            .sense(Sense::click()),
    )
}

/// A painted fold mark: down when open, right when shut.
pub(super) fn chevron(ui: &mut Ui, open: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(12.0), Sense::click());
    let c = rect.center();
    let points = if open {
        vec![
            c + Vec2::new(-3.5, -1.5),
            c + Vec2::new(0.0, 2.0),
            c + Vec2::new(3.5, -1.5),
        ]
    } else {
        vec![
            c + Vec2::new(-1.5, -3.5),
            c + Vec2::new(2.0, 0.0),
            c + Vec2::new(-1.5, 3.5),
        ]
    };
    ui.painter()
        .add(egui::Shape::line(points, Stroke::new(1.5, tokens::MUTED)));
    response
}

/// The faint line under a section header or a feature row.
pub(super) fn divider(ui: &mut Ui) {
    let line = ui.cursor().min.y;
    let span = ui.max_rect().x_range();
    ui.painter()
        .hline(span, line, Stroke::new(1.0, Color32::from_white_alpha(14)));
}

/// A feature the prototype lists that Cutlass cannot do yet: a dimmed row
/// with a "Later" mark.
fn later_row(ui: &mut Ui, label: &str) {
    ui.horizontal(|ui| {
        ui.set_min_height(40.0);
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.add_enabled_ui(false, |ui| check_box(ui, false, label));
        ui.label(RichText::new(label).size(13.0).color(tokens::FAINT));
        chevron(ui, false);
        later_badge(ui);
    })
    .response
    .on_hover_text(LATER);
    divider(ui);
}

/// A small "Later" mark, painted at text height so it never stretches a row.
pub(super) fn later_badge(ui: &mut Ui) {
    let galley =
        ui.painter()
            .layout_no_wrap("Later".to_owned(), FontId::proportional(10.0), ACCENT);
    let size = galley.size() + Vec2::new(10.0, 4.0);
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(6), BOX);
    ui.painter()
        .galley(rect.center() - galley.size() / 2.0, galley, ACCENT);
}

pub(super) fn later_note(ui: &mut Ui, what: &str) {
    note(
        ui,
        &format!("{what} comes later. Cutlass has no edit for it yet."),
    );
}

pub(super) fn note(ui: &mut Ui, text: &str) {
    ui.add_space(8.0);
    ui.label(RichText::new(text).size(12.0).color(tokens::MUTED));
    ui.add_space(ROW_GAP);
}

fn later_button(ui: &mut Ui, label: &str) {
    ui.add_space(6.0);
    ui.add_enabled(false, ui::Button::new(label).text_size(12.0).tip(LATER));
}

fn footer(ui: &mut Ui, labels: &[&str]) {
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        for label in labels {
            ui.add_enabled(false, ui::Button::new(*label).text_size(12.0))
                .on_disabled_hover_text(LATER);
        }
    });
}

/// "Apply to all main captions", on a caption of the main row.
fn all_captions_box(shell: &mut Shell, ui: &mut Ui) {
    if !shell.on_main_captions() {
        return;
    }
    ui.add_space(8.0);
    let mut all = shell.all_captions();
    if ui
        .add(ui::Check::new(&mut all, "Apply to all main captions"))
        .on_hover_text("A change to the look or place styles every caption on this row")
        .changed()
    {
        shell.set_all_captions(all);
    }
    ui.separator();
}

/// The caption styles: Plain, then each style, the caption's own
/// outlined. A pick is one undo step.
fn caption_styles(shell: &mut Shell, ui: &mut Ui, inspection: &Inspection) {
    let mut picked = None;
    ui.add_space(8.0);
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(6.0);
        let tiles = std::iter::once(("Plain", "")).chain(
            cutlass_captions::CAPTION_STYLES
                .iter()
                .map(|style| (style.name, style.key)),
        );
        for (name, key) in tiles {
            let on = inspection.caption_style == key;
            let stroke = if on {
                Stroke::new(tokens::SELECTED_EDGE * 2.0, tokens::EXPORT)
            } else {
                Stroke::NONE
            };
            let tile = egui::Button::new(RichText::new(name).size(11.0))
                .fill(BOX)
                .stroke(stroke);
            if ui.add_sized([64.0, 40.0], tile).clicked() && !on {
                picked = Some(key);
            }
        }
    });
    if let Some(key) = picked {
        send(shell, |shell| shell.set_caption_style(key));
    }
}

/// One item a catalog page offers.
struct CatalogItem {
    id: &'static str,
    name: &'static str,
    categories: &'static [&'static str],
    tile: cutlass_editor::CatalogTile<'static>,
}

/// A catalog page's name, the search box's hint when it has one, and its
/// categories.
struct CatalogPage {
    name: &'static str,
    search: Option<&'static str>,
    categories: &'static [&'static str],
}

/// The side of a catalog tile, and the gap between two.
pub(super) const CATALOG_TILE: f32 = 60.0;
const CATALOG_GAP: f32 = 6.0;
/// Tile pictures drawn in one frame; the rest follow in the next.
const TILES_PER_FRAME: usize = 4;

/// A catalog page: a search box when `search` names what it finds, the
/// category chips, then None and each item the search and chip let
/// through, as pictures. `worn` is the item the text wears, if any.
/// Returns the id picked, `""` for None.
fn catalog_page(
    shell: &Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    page: &CatalogPage,
    items: &[CatalogItem],
    worn: Option<&str>,
) -> Option<&'static str> {
    let query_key = format!("catalog {} search", page.name);
    let category_key = format!("catalog {} category", page.name);
    ui.add_space(8.0);
    if let Some(hint) = page.search {
        let query = state.fields.entry(query_key.clone()).or_default();
        ui::Field::new(query)
            .name(hint)
            .hint(hint)
            .icon("search")
            .width(ui.available_width())
            .show(ui);
        ui.add_space(8.0);
    }
    let query = state.fields.get(&query_key).cloned().unwrap_or_default();
    let mut category = state.fields.get(&category_key).cloned().unwrap_or_default();
    ui::Scroll::horizontal(("catalog chips", page.name))
        .without_bar()
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                for name in std::iter::once("All").chain(page.categories.iter().copied()) {
                    let value = if name == "All" { "" } else { name };
                    let chip = ui::Button::new(name)
                        .on(category == value)
                        .text_size(12.0)
                        .radius(12);
                    if ui.add(chip).clicked() {
                        value.clone_into(&mut category);
                    }
                }
            });
        });
    state.fields.insert(category_key, category.clone());
    ui.add_space(8.0);
    let heading = if category.is_empty() {
        "All"
    } else {
        &category
    };
    ui.label(RichText::new(heading).size(12.0).color(tokens::MUTED));
    ui.add_space(4.0);
    let shown: Vec<&CatalogItem> = items
        .iter()
        .filter(|item| cutlass_editor::catalog_shows(item.name, item.categories, &query, &category))
        .collect();
    let mut picked = None;
    let mut budget = TILES_PER_FRAME;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(CATALOG_GAP);
        if none_tile(ui, worn.is_none_or(str::is_empty)).clicked()
            && worn.is_some_and(|id| !id.is_empty())
        {
            picked = Some("");
        }
        for item in &shown {
            let on = worn == Some(item.id);
            if catalog_tile(shell, ui, state, item, on, &mut budget).clicked() && !on {
                picked = Some(item.id);
            }
        }
    });
    if shown.is_empty() {
        ui.add_space(8.0);
        ui.label(
            RichText::new("Nothing matches")
                .size(12.0)
                .color(tokens::MUTED),
        );
    }
    if budget == 0 {
        ui.ctx().request_repaint();
    }
    picked
}

/// The None tile: a ring with a slash.
pub(super) fn none_tile(ui: &mut Ui, on: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(CATALOG_TILE), Sense::click());
    let response = response.on_hover_text("None");
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, "None"));
    tile_frame(ui, rect, &response, on);
    let painter = ui.painter();
    let stroke = Stroke::new(1.5, tokens::MUTED);
    painter.circle_stroke(rect.center(), 11.0, stroke);
    let reach = Vec2::splat(11.0 * std::f32::consts::FRAC_1_SQRT_2);
    painter.line_segment([rect.center() - reach, rect.center() + reach], stroke);
    response
}

/// A tile's box: raised, lighter on hover, outlined when worn.
pub(super) fn tile_frame(ui: &Ui, rect: Rect, response: &egui::Response, on: bool) {
    let painter = ui.painter();
    let fill = if response.hovered() {
        tokens::RAISED.gamma_multiply(1.3)
    } else {
        BOX
    };
    painter.rect_filled(rect, CornerRadius::same(6), fill);
    if on {
        painter.rect_stroke(
            rect,
            CornerRadius::same(6),
            Stroke::new(tokens::SELECTED_EDGE * 2.0, tokens::EXPORT),
            egui::StrokeKind::Inside,
        );
    }
}

/// An item's tile: its picture once drawn, its name until then. At most
/// `budget` pictures are drawn this frame.
fn catalog_tile(
    shell: &Shell,
    ui: &mut Ui,
    state: &mut InspectorState,
    item: &CatalogItem,
    on: bool,
    budget: &mut usize,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(CATALOG_TILE), Sense::click());
    let response = response.on_hover_text(item.name);
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, item.name));
    if !state.tiles.contains_key(item.id) && *budget > 0 && ui.is_rect_visible(rect) {
        *budget -= 1;
        if let Some(picture) = shell.tile_picture(item.tile) {
            let size = [picture.width as usize, picture.height as usize];
            let image = egui::ColorImage::from_rgba_unmultiplied(size, &picture.rgba);
            let texture = ui.ctx().load_texture(
                format!("catalog tile {}", item.id),
                image,
                egui::TextureOptions::LINEAR,
            );
            state.tiles.insert(item.id.to_owned(), texture);
        }
    }
    tile_frame(ui, rect, &response, on);
    let inner = rect.shrink(2.0);
    if let Some(texture) = state.tiles.get(item.id) {
        ui.painter().with_clip_rect(inner).image(
            texture.id(),
            inner,
            Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    } else {
        if !state.tiles.contains_key(item.id) {
            *budget = 0;
        }
        ui.painter().with_clip_rect(inner).text(
            rect.center(),
            Align2::CENTER_CENTER,
            item.name,
            FontId::proportional(10.0),
            tokens::MUTED,
        );
    }
    response
}

/// The Templates page: each caption template, the caption's own outlined.
/// A pick styles every main caption while "Apply to all main captions" is
/// on, as one undo step.
fn caption_templates(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState) {
    let items: Vec<CatalogItem> = cutlass_editor::CAPTION_TEMPLATES
        .iter()
        .map(|template| CatalogItem {
            id: template.id,
            name: template.name,
            categories: template.categories,
            tile: cutlass_editor::CatalogTile::Template(template),
        })
        .collect();
    let worn = shell.caption_template();
    let picked = catalog_page(
        shell,
        ui,
        state,
        &CatalogPage {
            name: "templates",
            search: Some("Search text templates"),
            categories: cutlass_editor::TEMPLATE_CATEGORIES,
        },
        &items,
        Some(worn.as_deref().unwrap_or_default()),
    );
    match picked {
        Some("") => send(shell, Shell::clear_caption_template),
        Some(id) => send(shell, |shell| shell.set_caption_template(id)),
        None => {}
    }
}

/// The Bubble page: None, then each bubble, the title's own outlined. A
/// pick shapes its background and colors it and the letters; None takes
/// the bubble and its background off.
fn bubble_page(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState) {
    let Some(look) = shell.text_look() else {
        return;
    };
    let items: Vec<CatalogItem> = cutlass_editor::BUBBLE_LOOKS
        .iter()
        .map(|bubble| CatalogItem {
            id: bubble.id,
            name: bubble.name,
            categories: bubble.categories,
            tile: cutlass_editor::CatalogTile::Bubble(bubble),
        })
        .collect();
    // A bubble of another color still is not None.
    let worn = cutlass_editor::bubble_look_of(&look)
        .map(|bubble| bubble.id)
        .or(look.bubble.is_empty().then_some(""));
    let picked = catalog_page(
        shell,
        ui,
        state,
        &CatalogPage {
            name: "bubbles",
            search: None,
            categories: cutlass_editor::BUBBLE_CATEGORIES,
        },
        &items,
        worn,
    );
    match picked {
        Some("") => send(shell, |shell| shell.set_text_bubble("")),
        Some(id) => send(shell, |shell| shell.set_bubble_look(id)),
        None => {}
    }
}

/// The Effects page: None, then each look, the title's own outlined. A
/// pick sets its outline, shadow and glow, and for some its letters' color.
fn effect_page(shell: &mut Shell, ui: &mut Ui, state: &mut InspectorState) {
    let Some(look) = shell.text_look() else {
        return;
    };
    let items: Vec<CatalogItem> = cutlass_editor::TEXT_EFFECTS
        .iter()
        .map(|effect| CatalogItem {
            id: effect.id,
            name: effect.name,
            categories: effect.categories,
            tile: cutlass_editor::CatalogTile::Effect(effect),
        })
        .collect();
    let worn = cutlass_editor::text_effect_of(&look).map(|effect| {
        if effect.id == cutlass_editor::NO_TEXT_EFFECT.id {
            ""
        } else {
            effect.id
        }
    });
    let picked = catalog_page(
        shell,
        ui,
        state,
        &CatalogPage {
            name: "effects",
            search: Some("Search text effects"),
            categories: cutlass_editor::EFFECT_CATEGORIES,
        },
        &items,
        worn,
    );
    match picked {
        Some("") => send(shell, |shell| {
            shell.set_text_effect(cutlass_editor::NO_TEXT_EFFECT.id)
        }),
        Some(id) => send(shell, |shell| shell.set_text_effect(id)),
        None => {}
    }
}

fn later_tiles(ui: &mut Ui, names: &[&str]) {
    ui.add_space(8.0);
    ui.add_enabled_ui(false, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::splat(6.0);
            for name in names {
                ui.add_sized([64.0, 48.0], ui::Button::new(*name).text_size(11.0));
            }
        });
    })
    .response
    .on_disabled_hover_text(LATER);
}

/// The indented body under a section.
pub(super) fn body(ui: &mut Ui, add: impl FnOnce(&mut Ui)) {
    ui.add_space(ROW_GAP);
    ui.vertical(add);
    ui.add_space(ROW_GAP);
}

pub(super) fn row_label(ui: &mut Ui, label: &str) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(LABEL_WIDTH, ROW_HEIGHT), Sense::hover());
    let color = if ui.is_enabled() {
        tokens::TEXT
    } else {
        tokens::FAINT
    };
    ui.painter().with_clip_rect(rect).text(
        rect.left_center(),
        Align2::LEFT_CENTER,
        label,
        FontId::proportional(12.0),
        color,
    );
}

fn label_row(ui: &mut Ui, label: &str, add: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        row_label(ui, label);
        add(ui);
    });
    ui.add_space(ROW_GAP);
}

/// A slider row: name, track, value box. Each new value is sent as the
/// track moves, so the preview follows; the shell joins a held drag into
/// one undo step.
pub(super) fn slider(
    ui: &mut Ui,
    state: &mut InspectorState,
    key: &str,
    label: &str,
    stored: f64,
    range: std::ops::RangeInclusive<f64>,
    show: impl Fn(f64) -> String,
) -> Option<f64> {
    let mut value = state.live.get(key).copied().unwrap_or(stored);
    let mut moved = None;
    // The box takes typing only where it shows the value itself, give or
    // take its suffix; a box that shows a converted value stays read-only.
    let (low, high) = (*range.start(), *range.end());
    let near = |at: f64| {
        leading_number(&show(at))
            .is_some_and(|shown| (shown - at).abs() <= (high - low).abs() * 0.01)
    };
    let typable = near(low) && near(high);
    ui.horizontal(|ui| {
        row_label(ui, label);
        let track = (ui.available_width() - VALUE_WIDTH - 8.0).max(40.0);
        let response = ui.add(ui::Slider::new(&mut value, range).name(label).width(track));
        if response.changed() {
            state.live.insert(key.to_owned(), value);
            moved = Some(value).filter(|value| *value != stored);
        }
        if typable {
            let text = state.fields.entry(format!("{key} value")).or_default();
            let salt = ("slider-value", key);
            // Show the value before the box draws, unless it is being typed in.
            let id = ui.make_persistent_id(egui::IdSalt::new(salt));
            if !ui.memory(|memory| memory.has_focus(id)) {
                *text = show(value);
            }
            let field = ui::Field::new(text)
                .id_salt(salt)
                .name(label)
                .width(VALUE_WIDTH)
                .text_size(12.0)
                .show(ui);
            if super::commits(ui, &field.response)
                && let Some(typed) = leading_number(text)
            {
                let typed = typed.clamp(low, high);
                moved = Some(typed).filter(|typed| *typed != stored);
            }
        } else {
            value_box(ui, &show(value));
        }
    });
    ui.add_space(ROW_GAP);
    // The value shown follows the track while held, then the stored one.
    if !ui.input(|input| input.pointer.any_down()) {
        state.live.remove(key);
    }
    moved
}

/// The number a value box's text starts with, such as 90 in `90%`.
fn leading_number(text: &str) -> Option<f64> {
    let text = text.trim();
    let end = text
        .char_indices()
        .find(|&(index, c)| {
            !(c.is_ascii_digit() || c == '.' || (index == 0 && matches!(c, '-' | '+' | '−')))
        })
        .map_or(text.len(), |(index, _)| index);
    text[..end]
        .replace('−', "-")
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

fn later_slider(ui: &mut Ui, label: &str, value: f64, shown: &str) {
    ui.add_enabled_ui(false, |ui| {
        ui.horizontal(|ui| {
            row_label(ui, label);
            let track = (ui.available_width() - VALUE_WIDTH - 8.0).max(40.0);
            let mut value = value;
            ui.add(ui::Slider::new(&mut value, -1000.0..=1000.0).width(track));
            value_box(ui, shown);
        });
    })
    .response
    .on_disabled_hover_text(LATER);
    ui.add_space(ROW_GAP);
}

pub(super) fn value_box(ui: &mut Ui, text: &str) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(VALUE_WIDTH, ROW_HEIGHT), Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(8), BOX);
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        text,
        FontId::proportional(12.0),
        tokens::TEXT,
    );
}

/// A name and two number boxes at the right. A box whose stored value is
/// NaN is left out. `Some` is a new finite number typed into that box.
fn pair_row(
    ui: &mut Ui,
    state: &mut InspectorState,
    label: &str,
    first: (&str, f64),
    second: (&str, f64),
) -> (Option<f64>, Option<f64>) {
    let mut out = (None, None);
    ui.horizontal(|ui| {
        row_label(ui, label);
        // A lone box is as wide as the first of a pair, so the boxes line
        // up down the column.
        let width =
            ((ui.available_width() - ui.spacing().item_spacing.x) / 2.0).max(PAIR_BOX - 24.0);
        out.0 = number_box(ui, state, &format!("{label} {}", first.0), first, width);
        if !second.1.is_nan() {
            out.1 = number_box(ui, state, &format!("{label} {}", second.0), second, width);
        }
    });
    ui.add_space(ROW_GAP);
    out
}

fn number_box(
    ui: &mut Ui,
    state: &mut InspectorState,
    key: &str,
    (unit, stored): (&str, f64),
    width: f32,
) -> Option<f64> {
    // Never show a minus sign on zero.
    let stored = stored + 0.0;
    let text = state.fields.entry(key.to_owned()).or_default();
    let mut edited = None;
    let field = ui::Field::new(text)
        .name(key)
        .width(width)
        .height(ROW_HEIGHT)
        .ink(Color32::WHITE);
    let field = if unit == "°" {
        field.icon("rotate")
    } else {
        field.unit(unit)
    };
    let shown = field.scrub().show(ui);
    let response = shown.response;
    // A drag on the unit counts from the value at the press, one step a
    // point. The shell joins a held drag into one undo step.
    let start_key = format!("{key} scrub start");
    let step_key = format!("{key} scrub step");
    if let Some(moved) = shown.scrub {
        let start = *state.live.entry(start_key).or_insert(stored);
        let step = *state
            .live
            .entry(step_key)
            .or_insert(if stored.fract() == 0.0 { 1.0 } else { 0.1 });
        // Shift already makes `moved` a tenth; land on a tenth of a step
        // then too.
        let grain = if ui.input(|input| input.modifiers.shift) {
            step * 0.1
        } else {
            step
        };
        let value = start + f64::from(moved) * step;
        edited = Some((value / grain).round() * grain);
    } else {
        state.live.remove(&start_key);
        state.live.remove(&step_key);
    }
    if super::commits(ui, &response) {
        edited = text
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite());
    }
    if !response.has_focus() {
        let shown = if stored.fract() == 0.0 {
            format!("{stored:.0}")
        } else {
            format!("{stored:.2}")
        };
        *text = shown;
    }
    edited.filter(|value| *value != stored)
}

/// A name and a swatch that opens a color picker. The color is sent once,
/// when the pointer lets go. An empty stored color shows white.
pub(super) fn color_row(
    ui: &mut Ui,
    state: &mut InspectorState,
    key: &str,
    label: &str,
    stored: &str,
) -> Option<String> {
    let stored_color = parse_hex(stored).unwrap_or(Color32::WHITE);
    let mut color = state.colors.get(key).copied().unwrap_or(stored_color);
    label_row(ui, label, |ui| {
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let response = egui::color_picker::color_edit_button_srgba(
                ui,
                &mut color,
                egui::color_picker::Alpha::Opaque,
            );
            if response.changed() {
                state.colors.insert(key.to_owned(), color);
            }
        });
    });
    if ui.input(|input| input.pointer.any_down()) {
        return None;
    }
    let picked = state.colors.remove(key)?;
    let hex = format!("#{:02X}{:02X}{:02X}", picked.r(), picked.g(), picked.b());
    (picked != stored_color || stored.is_empty()).then_some(hex)
}

/// `#RRGGBB`, `RRGGBB`, or `0xAARRGGBB`.
fn parse_hex(text: &str) -> Option<Color32> {
    let text = text.trim();
    let digits = text
        .strip_prefix('#')
        .or_else(|| {
            text.strip_prefix("0x")
                .map(|rest| rest.get(2..).unwrap_or(rest))
        })
        .unwrap_or(text);
    if digits.len() != 6 {
        return None;
    }
    let value = u32::from_str_radix(digits, 16).ok()?;
    let [_, r, g, b] = value.to_be_bytes();
    Some(Color32::from_rgb(r, g, b))
}

pub(super) fn switch_row(ui: &mut Ui, label: &str, on: bool, editable: bool) -> egui::Response {
    // The caller flips the stored value on a click; this copy only draws.
    let mut shown = on;
    let response = ui
        .horizontal(|ui| {
            ui.set_min_height(ROW_HEIGHT);
            ui.add_enabled_ui(editable, |ui| ui::switch_row(ui, label, &mut shown))
                .inner
        })
        .inner;
    ui.add_space(ROW_GAP);
    response
}

pub(super) fn check_box(ui: &mut Ui, on: bool, name: &str) -> egui::Response {
    // The caller flips the stored value on a click; this copy only draws.
    let mut shown = on;
    ui.add(ui::Check::new(&mut shown, "").name(name))
}

fn icon_toggle(ui: &mut Ui, name: &str, tip: &str, on: bool) -> egui::Response {
    ui.add(
        ui::Button::icon(name)
            .on(on)
            .min_size(Vec2::splat(24.0))
            .tip(tip),
    )
}

/// A row of icon buttons on a dark bar. Returns the value of the one clicked.
fn icon_bar<T: Copy>(ui: &mut Ui, items: &[(&str, &str, Option<T>)]) -> Option<T> {
    let mut picked = None;
    ui::Surface::inset().show(ui, |ui| {
        ui.horizontal(|ui| {
            let gap = ((ui.available_width() - 24.0 * items.len() as f32)
                / (items.len().max(2) - 1) as f32)
                .max(2.0);
            ui.spacing_mut().item_spacing.x = gap;
            for (icon, tip, value) in items {
                if icon_toggle(ui, icon, tip, false).clicked() {
                    picked = *value;
                }
            }
        });
    });
    ui.add_space(ROW_GAP);
    picked
}

fn paint_icon(ui: &Ui, rect: Rect, name: &str, color: Color32) {
    let mark = Rect::from_center_size(rect.center(), Vec2::splat(16.0));
    if icons::paint(ui.painter(), mark, name, color).is_err() {
        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            "?",
            FontId::proportional(11.0),
            color,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::leading_number;

    #[test]
    fn a_value_box_reads_the_number_before_its_suffix() {
        assert_eq!(leading_number("90%"), Some(90.0));
        assert_eq!(leading_number(" -12.5 dB"), Some(-12.5));
        assert_eq!(leading_number("−3"), Some(-3.0));
        assert_eq!(leading_number("1.0x"), Some(1.0));
        assert_eq!(leading_number("-inf dB"), None);
        assert_eq!(leading_number("%"), None);
    }
}
