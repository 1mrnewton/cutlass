//! Right-click menus: the clip menu and the library menus.
//!
//! The entries follow the prototype's menus, all of them. A row this build
//! cannot do yet carries a TODO badge, so nothing on the list is forgotten.
//! A menu draws above every panel. A press outside it, Escape, or a scroll
//! closes it.

use cutlass_model::TrackType;
use egui::{Align2, Context, CornerRadius, FontId, Id, LayerId, Order, Pos2, Rect, Sense, Vec2};

use crate::tokens;

/// One menu row, or the line between two groups.
#[derive(Clone, Copy, Debug)]
pub enum Entry {
    Separator,
    Item(Item),
}

#[derive(Clone, Copy, Debug)]
pub struct Item {
    pub id: &'static str,
    pub label: &'static str,
    /// Empty for none.
    pub shortcut: &'static str,
    /// False marks a row this build cannot do yet.
    pub built: bool,
    pub children: &'static [Entry],
}

const fn done(id: &'static str, label: &'static str, shortcut: &'static str) -> Entry {
    Entry::Item(Item {
        id,
        label,
        shortcut,
        built: true,
        children: &[],
    })
}

const fn todo(id: &'static str, label: &'static str, shortcut: &'static str) -> Entry {
    Entry::Item(Item {
        id,
        label,
        shortcut,
        built: false,
        children: &[],
    })
}

/// A row that opens a submenu. It is built when every child is.
const fn sub(id: &'static str, label: &'static str, children: &'static [Entry]) -> Entry {
    let mut built = true;
    let mut index = 0;
    while index < children.len() {
        if let Entry::Item(item) = children[index]
            && !item.built
        {
            built = false;
        }
        index += 1;
    }
    Entry::Item(Item {
        id,
        label,
        shortcut: "",
        built,
        children,
    })
}

const SEP: Entry = Entry::Separator;

const COPY: Entry = done("copy", "Copy", "⌘C");
const CUT: Entry = done("cut", "Cut", "⌘X");
const DELETE: Entry = done("delete", "Delete", "⌫");
const DEACTIVATE: Entry = todo("deactivate", "Deactivate", "");
const REPLACE: Entry = todo("replace", "Replace", "");

const VIDEO: &[Entry] = &[
    COPY,
    CUT,
    DELETE,
    SEP,
    sub(
        "edit",
        "Edit",
        &[
            todo("edit-speed", "Speed", ""),
            todo("edit-animation", "Animation", ""),
            done("edit-crop", "Crop", ""),
            todo("edit-opacity", "Opacity", ""),
        ],
    ),
    done("split-scenes", "Split scenes", ""),
    done("transcript", "Transcript", ""),
    sub(
        "isolate-voice",
        "Isolate voice",
        &[
            todo("isolate-keep-voice", "Keep voice", ""),
            todo("isolate-keep-background", "Keep background", ""),
        ],
    ),
    done("extract-audio", "Extract audio", "⇧⌘S"),
    todo("sync", "Sync video and audio", ""),
    SEP,
    todo("compound", "Create compound clip", "⌥G"),
    todo("multicam", "Create multi-camera clip", ""),
    todo("group", "Group", ""),
    todo("ungroup", "Ungroup", ""),
    SEP,
    todo("export", "Export selected clips", ""),
    todo("deactivate", "Deactivate clip", "V"),
    todo("trim", "Trim clip", ""),
    todo("replace", "Replace clip", ""),
    todo("link-media", "Link to media", ""),
    todo("open-file", "Open file location", ""),
    SEP,
    todo("edit-effects", "Edit effects", ""),
    todo("variable-speed", "Show variable speed animation", "⌥K"),
    sub(
        "range",
        "Range",
        &[
            todo("range-in", "Set in", ""),
            todo("range-out", "Set out", ""),
        ],
    ),
    sub(
        "render",
        "Render",
        &[
            todo("render-preview", "Preview", ""),
            todo("render-full", "Full", ""),
        ],
    ),
];

const AUDIO: &[Entry] = &[
    COPY,
    CUT,
    DELETE,
    SEP,
    todo("volume", "Volume", ""),
    todo("speed", "Speed", ""),
    todo("fade", "Fade", ""),
    todo("beat", "Beat", ""),
    SEP,
    DEACTIVATE,
    REPLACE,
];

const TEXT: &[Entry] = &[
    COPY,
    CUT,
    DELETE,
    SEP,
    todo("edit-text", "Edit text", ""),
    todo("style", "Style", ""),
    SEP,
    DEACTIVATE,
    REPLACE,
];

const STICKER: &[Entry] = &[
    COPY,
    CUT,
    DELETE,
    SEP,
    done("flip", "Flip", ""),
    SEP,
    DEACTIVATE,
    REPLACE,
];

const EFFECT: &[Entry] = &[
    COPY,
    CUT,
    DELETE,
    SEP,
    todo("edit-effect", "Edit effect", ""),
    SEP,
    DEACTIVATE,
    REPLACE,
];

const FILTER: &[Entry] = &[
    COPY,
    CUT,
    DELETE,
    SEP,
    todo("edit-filter", "Edit filter", ""),
    SEP,
    DEACTIVATE,
    REPLACE,
];

const ADJUST: &[Entry] = &[
    COPY,
    CUT,
    DELETE,
    SEP,
    todo("edit-adjust", "Edit adjustment", ""),
    SEP,
    DEACTIVATE,
    REPLACE,
];

const COMPOSITION: &[Entry] = &[
    COPY,
    CUT,
    DELETE,
    SEP,
    todo("render-preview", "Render", ""),
    SEP,
    DEACTIVATE,
    REPLACE,
];

const OTHER: &[Entry] = &[COPY, CUT, DELETE, SEP, DEACTIVATE, REPLACE];

/// The menu for a clip on a track of `kind`.
#[must_use]
pub fn clip_entries(kind: &TrackType) -> &'static [Entry] {
    match kind {
        TrackType::Video => VIDEO,
        TrackType::Audio | TrackType::TextToVideoTextAudio => AUDIO,
        TrackType::Text | TrackType::TextToVideoText => TEXT,
        TrackType::Sticker => STICKER,
        TrackType::Effect | TrackType::PluginEffect => EFFECT,
        TrackType::Filter => FILTER,
        TrackType::Adjust => ADJUST,
        TrackType::Composition => COMPOSITION,
        _ => OTHER,
    }
}

/// The menu for a gap between clips on a row.
pub const GAP_ROW: &[Entry] = &[done("close-gap", "Close gap", "")];

/// The menu for a track's head. The prototype has none; these rows add,
/// name and delete tracks.
pub const TRACK_HEAD: &[Entry] = &[
    done("add-video", "Add video track", ""),
    done("add-audio", "Add audio track", ""),
    done("add-text", "Add text track", ""),
    SEP,
    done("rename-track", "Rename track", ""),
    done("delete-track", "Delete track", ""),
];

/// The menu for the picked file cards in the library. The rows in
/// [`LIBRARY_CARD_OFF`] are there but greyed, as in the prototype.
pub const LIBRARY_CARD: &[Entry] = &[
    done("open-file", "Open file location", ""),
    done("delete", "Delete", ""),
    done("new-folder-with", "New folder with selected item", ""),
    done("link-media", "Link to media", ""),
    todo("split-scenes", "Split scenes", ""),
    todo("export", "Export selected clips", ""),
];

/// The menu for a folder in the library. The rows in
/// [`LIBRARY_FOLDER_OFF`] are there but greyed, as in the prototype.
pub const LIBRARY_FOLDER: &[Entry] = &[
    done("delete", "Delete", ""),
    done("open-file", "Open file location", ""),
    done("new-folder-with", "New folder with selected item", ""),
    done("rename", "Rename", ""),
    done("link-media", "Link to media", ""),
    todo("export", "Export selected clips", ""),
];

/// The card menu's rows the prototype greys.
pub const LIBRARY_CARD_OFF: &[&str] = &["link-media"];

/// The folder menu's rows that cannot run on a folder.
pub const LIBRARY_FOLDER_OFF: &[&str] = &["open-file", "new-folder-with", "link-media"];

/// The menu for the empty space between library cards.
pub const LIBRARY_EMPTY: &[Entry] = &[
    done("new-folder", "New folder", ""),
    sub(
        "sort",
        "Sort",
        &[
            done("sort-name", "Name", ""),
            done("sort-duration", "Duration", ""),
        ],
    ),
];

const PAD: f32 = 6.0;
const ROW: f32 = 32.0;
const ROW_PAD: f32 = 10.0;
const GAP: f32 = 16.0;
const SEPARATOR: f32 = 13.0;
const RADIUS: u8 = 12;
const ROW_RADIUS: u8 = 8;
const LABEL_SIZE: f32 = 13.0;
const KEY_SIZE: f32 = 12.0;
const BADGE: &str = "TODO";
const BADGE_SIZE: f32 = 10.0;
const BADGE_HEIGHT: f32 = 16.0;
const CHEVRON: f32 = 14.0;
/// Space kept between a menu and the window edge.
const EDGE: f32 = 8.0;

/// What a menu did this frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct Shown {
    /// The row clicked. The menu should close.
    pub picked: Option<Item>,
    /// A press outside, Escape, or a scroll outside. The menu should close.
    pub dismissed: bool,
}

/// Draw the menu `id` with its corner at `at`, kept inside the window. A
/// menu taller than the window scrolls under the pointer. `enabled` says whether a built row can run now; a TODO row always can,
/// so a click on it can say it is not built yet.
pub fn show(
    ctx: &Context,
    id: Id,
    at: Pos2,
    entries: &'static [Entry],
    enabled: &dyn Fn(&str) -> bool,
) -> Shown {
    let open_key = id.with("open");
    let scroll_key = id.with("scroll");
    let mut open: Option<&'static str> = ctx.data(|data| data.get_temp(open_key)).flatten();
    let screen = ctx.content_rect().shrink(EDGE);
    let full = size(ctx, entries);
    let rect = place(screen, at, Vec2::new(full.x, full.y.min(screen.height())));
    let (pressed, pointer, escape, wheel) = ctx.input(|input| {
        (
            input.pointer.primary_pressed(),
            input.pointer.interact_pos(),
            input.key_pressed(egui::Key::Escape),
            input.smooth_scroll_delta.y,
        )
    });
    let over = pointer.is_some_and(|pos| rect.contains(pos));
    let mut scroll: f32 = ctx.data(|data| data.get_temp(scroll_key)).unwrap_or(0.0);
    if over && wheel != 0.0 {
        scroll = (scroll - wheel).clamp(0.0, full.y - rect.height());
        ctx.data_mut(|data| data.insert_temp(scroll_key, scroll));
        open = None;
    }
    let mut shown = Shown::default();
    let mut hovered = None;
    let mut submenu = None;
    let rows = paint_menu(
        ctx,
        id,
        rect,
        scroll,
        entries,
        open,
        enabled,
        &mut |item, row, response| {
            if response.hovered() {
                hovered = Some(item.id);
            }
            if !item.children.is_empty() && open == Some(item.id) {
                submenu = Some((item.children, row));
            }
            if response.clicked() && item.children.is_empty() {
                shown.picked = Some(*item);
            }
        },
    );
    let mut rects = vec![rect];
    if let Some((children, row)) = submenu {
        let size = size(ctx, children);
        let right = Pos2::new(row.right() - PAD, row.top() - PAD);
        let corner = if right.x + size.x > screen.right() {
            Pos2::new(row.left() - size.x + PAD, right.y)
        } else {
            right
        };
        let sub_rect = place(screen, corner, size);
        // A sublayer always paints just above its parent.
        ctx.set_sublayer(
            LayerId::new(Order::Foreground, id),
            LayerId::new(Order::Foreground, id.with("sub")),
        );
        paint_menu(
            ctx,
            id.with("sub"),
            sub_rect,
            0.0,
            children,
            None,
            enabled,
            &mut |item, _, response| {
                if response.clicked() {
                    shown.picked = Some(*item);
                }
            },
        );
        rects.push(sub_rect);
    }
    // Hovering a row opens its submenu and closes any other. Leaving the
    // menu keeps the submenu open, so the pointer can cross the gap.
    if let Some(hovered) = hovered {
        let next = rows
            .iter()
            .find(|item| item.id == hovered && !item.children.is_empty())
            .map(|item| item.id);
        ctx.data_mut(|data| data.insert_temp(open_key, next));
    }
    let outside = pointer.is_none_or(|pos| !rects.iter().any(|rect| rect.contains(pos)));
    shown.dismissed = escape || (outside && (pressed || wheel != 0.0));
    if shown.picked.is_some() || shown.dismissed {
        ctx.data_mut(|data| {
            data.remove::<Option<&'static str>>(open_key);
            data.remove::<f32>(scroll_key);
        });
    }
    shown
}

/// The menu's size: its widest row and every row's height.
fn size(ctx: &Context, entries: &[Entry]) -> Vec2 {
    let mut width = tokens::MENU_WIDTH - 2.0 * PAD;
    let mut height = 2.0 * PAD;
    for entry in entries {
        match entry {
            Entry::Separator => height += SEPARATOR,
            Entry::Item(item) => {
                height += ROW;
                let mut row = 2.0 * ROW_PAD + text_width(ctx, item.label, LABEL_SIZE);
                if !item.built {
                    row += 8.0 + badge_width(ctx);
                }
                if !item.children.is_empty() {
                    row += GAP + CHEVRON;
                } else if !item.shortcut.is_empty() {
                    row += GAP + text_width(ctx, item.shortcut, KEY_SIZE);
                }
                width = width.max(row);
            }
        }
    }
    Vec2::new(width.ceil() + 2.0 * PAD, height)
}

/// `at` moved just enough that a menu of `size` fits inside `screen`.
fn place(screen: Rect, at: Pos2, size: Vec2) -> Rect {
    let x = at.x.min(screen.right() - size.x).max(screen.left());
    let y = at.y.min(screen.bottom() - size.y).max(screen.top());
    Rect::from_min_size(Pos2::new(x, y), size)
}

fn text_width(ctx: &Context, text: &str, size: f32) -> f32 {
    ctx.fonts_mut(|fonts| {
        fonts
            .layout_no_wrap(text.to_owned(), FontId::proportional(size), tokens::TEXT)
            .size()
            .x
    })
}

fn badge_width(ctx: &Context) -> f32 {
    text_width(ctx, BADGE, BADGE_SIZE) + 12.0
}

/// Paint one menu panel on its own layer above the window, its rows moved
/// up by `scroll`, and report each row in view to `row`. Returns the rows'
/// items.
#[allow(clippy::too_many_arguments)]
fn paint_menu(
    ctx: &Context,
    id: Id,
    rect: Rect,
    scroll: f32,
    entries: &'static [Entry],
    open: Option<&str>,
    enabled: &dyn Fn(&str) -> bool,
    row: &mut dyn FnMut(&Item, Rect, &egui::Response),
) -> Vec<Item> {
    let layer = LayerId::new(Order::Foreground, id);
    // A layer painted by a bare Ui has no place in egui's stacking order
    // until it is raised, and then its paint order among equals is not fixed.
    ctx.move_to_top(layer);
    let ui = egui::Ui::new(
        ctx.clone(),
        id,
        egui::UiBuilder::new().layer_id(layer).max_rect(rect),
    );
    let painter = ui.painter().with_clip_rect(rect);
    let shadow = egui::Shadow {
        offset: [0, 16],
        blur: 40,
        spread: 0,
        color: egui::Color32::from_black_alpha(115),
    };
    // The shadow spreads past the panel, so it is clipped to the window, not
    // to the panel.
    ctx.layer_painter(layer)
        .add(shadow.as_shape(rect, CornerRadius::same(RADIUS)));
    painter.rect_filled(rect, CornerRadius::same(RADIUS), tokens::MENU);
    // The panel takes the clicks between rows, so they do not reach the
    // timeline under it.
    let _ = ui.interact(rect, id.with("panel"), Sense::click());
    let mut items = Vec::new();
    let mut top = rect.top() + PAD - scroll;
    let view = rect.shrink2(Vec2::new(0.0, PAD));
    let (left, right) = (rect.left() + PAD, rect.right() - PAD);
    for entry in entries {
        let item = match entry {
            Entry::Separator => {
                let y = top + SEPARATOR / 2.0;
                painter.line_segment(
                    [Pos2::new(left + 8.0, y), Pos2::new(right - 8.0, y)],
                    egui::Stroke::new(1.0, tokens::MENU_SEPARATOR),
                );
                top += SEPARATOR;
                continue;
            }
            Entry::Item(item) => item,
        };
        let bounds = Rect::from_min_max(Pos2::new(left, top), Pos2::new(right, top + ROW));
        top += ROW;
        items.push(*item);
        let seen = bounds.intersect(view);
        if seen.height() <= 0.0 {
            continue;
        }
        let live = !item.built || enabled(item.id);
        let response = ui.interact(
            seen,
            id.with(item.id),
            if live { Sense::click() } else { Sense::hover() },
        );
        if live && (response.hovered() || open == Some(item.id)) {
            painter.rect_filled(bounds, CornerRadius::same(ROW_RADIUS), tokens::MENU_HOVER);
        }
        let (ink, key) = if live {
            (tokens::TEXT, tokens::MENU_KEY)
        } else {
            (tokens::MENU_DISABLED, tokens::MENU_KEY_DISABLED)
        };
        let label = painter.text(
            Pos2::new(bounds.left() + ROW_PAD, bounds.center().y),
            Align2::LEFT_CENTER,
            item.label,
            FontId::proportional(LABEL_SIZE),
            ink,
        );
        if !item.built {
            let badge = Rect::from_min_size(
                Pos2::new(label.right() + 8.0, bounds.center().y - BADGE_HEIGHT / 2.0),
                Vec2::new(badge_width(ctx), BADGE_HEIGHT),
            );
            painter.rect_filled(badge, CornerRadius::same(8), tokens::MENU_BADGE);
            painter.text(
                badge.center(),
                Align2::CENTER_CENTER,
                BADGE,
                FontId::proportional(BADGE_SIZE),
                egui::Color32::WHITE,
            );
        }
        let end = bounds.right() - ROW_PAD;
        if !item.children.is_empty() {
            let center = Pos2::new(end - CHEVRON / 2.0, bounds.center().y);
            let stroke = egui::Stroke::new(1.5, tokens::MENU_CHEVRON);
            painter.line_segment(
                [center + Vec2::new(-2.0, -4.0), center + Vec2::new(2.0, 0.0)],
                stroke,
            );
            painter.line_segment(
                [center + Vec2::new(2.0, 0.0), center + Vec2::new(-2.0, 4.0)],
                stroke,
            );
        } else if !item.shortcut.is_empty() {
            painter.text(
                Pos2::new(end, bounds.center().y),
                Align2::RIGHT_CENTER,
                item.shortcut,
                FontId::proportional(KEY_SIZE),
                key,
            );
        }
        if live {
            row(item, bounds, &response);
        }
    }
    items
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(entries: &[Entry]) -> Vec<&'static str> {
        entries
            .iter()
            .filter_map(|entry| match entry {
                Entry::Item(item) => Some(item.id),
                Entry::Separator => None,
            })
            .collect()
    }

    #[test]
    fn every_clip_menu_starts_with_copy_cut_and_delete() {
        for kind in [
            TrackType::Video,
            TrackType::Audio,
            TrackType::Text,
            TrackType::Sticker,
            TrackType::Effect,
            TrackType::Filter,
            TrackType::Adjust,
            TrackType::Composition,
            TrackType::Adcube,
        ] {
            assert_eq!(ids(clip_entries(&kind))[..3], ["copy", "cut", "delete"]);
        }
    }

    #[test]
    fn the_video_menu_lists_every_prototype_row() {
        assert_eq!(
            ids(VIDEO),
            [
                "copy",
                "cut",
                "delete",
                "edit",
                "split-scenes",
                "transcript",
                "isolate-voice",
                "extract-audio",
                "sync",
                "compound",
                "multicam",
                "group",
                "ungroup",
                "export",
                "deactivate",
                "trim",
                "replace",
                "link-media",
                "open-file",
                "edit-effects",
                "variable-speed",
                "range",
                "render",
            ]
        );
    }

    #[test]
    fn a_submenu_is_built_only_when_every_child_is() {
        let built = |id: &str| {
            VIDEO.iter().find_map(|entry| match entry {
                Entry::Item(item) if item.id == id => Some(item.built),
                _ => None,
            })
        };
        assert_eq!(built("edit"), Some(false));
        assert_eq!(built("copy"), Some(true));
        assert_eq!(built("range"), Some(false));
    }

    fn frame(ctx: &Context, height: f32, events: Vec<egui::Event>) -> (Shown, egui::FullOutput) {
        let mut shown = Shown::default();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, height))),
                events,
                ..Default::default()
            },
            |ui| {
                shown = show(
                    ui.ctx(),
                    ui.scope_id().with("menu"),
                    Pos2::new(100.0, 100.0),
                    VIDEO,
                    &|_| true,
                );
            },
        );
        output.textures_delta.clear();
        (shown, output)
    }

    fn label_at(output: &egui::FullOutput, label: &str) -> Option<Pos2> {
        output
            .shapes
            .iter()
            .find_map(|clipped| match &clipped.shape {
                egui::Shape::Text(text) if text.galley.text() == label => Some(text.pos),
                _ => None,
            })
    }

    fn click(pos: Pos2) -> Vec<egui::Event> {
        let button = |pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        vec![egui::Event::PointerMoved(pos), button(true), button(false)]
    }

    #[test]
    fn the_shadow_is_not_cut_off_at_the_panel_edge() {
        let ctx = Context::default();
        let (_, output) = frame(&ctx, 1000.0, Vec::new());
        // The panel's corner is at (100, 100); the shadow reaches above it.
        assert!(
            output
                .shapes
                .iter()
                .any(|clipped| clipped.clip_rect.contains(Pos2::new(110.0, 90.0)))
        );
    }

    #[test]
    fn a_submenu_lies_above_its_menu() {
        let ctx = Context::default();
        let mut id = None;
        let mut frame = |events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 1000.0))),
                    events,
                    ..Default::default()
                },
                |ui| {
                    let menu = ui.scope_id().with("menu");
                    id = Some(menu);
                    show(ui.ctx(), menu, Pos2::new(100.0, 100.0), VIDEO, &|_| true);
                },
            );
            output.textures_delta.clear();
            output
        };
        let output = frame(Vec::new());
        let edit = label_at(&output, "Edit").expect("the Edit row") + Vec2::new(4.0, 6.0);
        frame(vec![egui::Event::PointerMoved(edit)]);
        let output = frame(vec![egui::Event::PointerMoved(edit)]);
        assert!(label_at(&output, "Crop").is_some(), "the submenu is open");
        let id = id.expect("shown");
        let layers: Vec<LayerId> = ctx.memory(|memory| memory.layer_ids().collect());
        let at = |id: Id| {
            layers
                .iter()
                .position(|layer| *layer == LayerId::new(Order::Foreground, id))
                .expect("stacked")
        };
        assert!(at(id.with("sub")) > at(id));
    }

    #[test]
    fn a_click_on_a_todo_row_picks_it_as_not_built() {
        let ctx = Context::default();
        let (_, output) = frame(&ctx, 1000.0, Vec::new());
        let row = label_at(&output, "Group").expect("the Group row") + Vec2::new(4.0, 6.0);
        let (shown, _) = frame(&ctx, 1000.0, click(row));
        let picked = shown.picked.expect("a pick");
        assert_eq!((picked.id, picked.built), ("group", false));
    }

    #[test]
    fn a_menu_taller_than_the_window_scrolls_under_the_wheel() {
        let ctx = Context::default();
        let (_, output) = frame(&ctx, 400.0, Vec::new());
        let before = label_at(&output, "Transcript").expect("the Transcript row");
        let wheel = egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: Vec2::new(0.0, -120.0),
            phase: egui::TouchPhase::Move,
            source: egui::MouseWheelSource::Wheel,
            modifiers: egui::Modifiers::default(),
        };
        let (shown, output) = frame(&ctx, 400.0, vec![egui::Event::PointerMoved(before), wheel]);
        assert!(!shown.dismissed);
        let after = label_at(&output, "Transcript").expect("the Transcript row");
        assert!(after.y < before.y, "Transcript stayed at {}", after.y);
    }

    #[test]
    fn escape_and_a_press_outside_close_the_menu() {
        let ctx = Context::default();
        frame(&ctx, 1000.0, Vec::new());
        let escape = egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        };
        assert!(frame(&ctx, 1000.0, vec![escape]).0.dismissed);
        assert!(
            frame(&ctx, 1000.0, click(Pos2::new(20.0, 20.0)))
                .0
                .dismissed
        );
        assert!(!frame(&ctx, 1000.0, Vec::new()).0.dismissed);
    }

    #[test]
    fn a_menu_near_the_corner_moves_inside_the_window() {
        let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let rect = place(screen, Pos2::new(780.0, 590.0), Vec2::new(248.0, 120.0));
        assert_eq!(rect.min, Pos2::new(552.0, 480.0));
        let rect = place(screen, Pos2::new(20.0, 30.0), Vec2::new(248.0, 120.0));
        assert_eq!(rect.min, Pos2::new(20.0, 30.0));
    }
}
