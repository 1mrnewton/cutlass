//! Right-click menus: the clip menu and the library menus.
//!
//! The entries follow the prototype's menus, all of them. A row this build
//! cannot do yet carries a Later badge, so nothing on the list is forgotten.
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
const COPY_ATTRIBUTES: Entry = todo("copy-attributes", "Copy attributes", "⇧⌘C");
const PASTE_ATTRIBUTES: Entry = todo("paste-attributes", "Paste attributes", "⇧⌘V");
const DELETE: Entry = done("delete", "Delete", "⌫");
const DEACTIVATE: Entry = done("deactivate", "Deactivate clip", "V");
const REPLACE_SOUND: Entry = done("replace", "Replace", "");
const SYNC: Entry = todo("sync", "Sync video and audio", "");
const COMPOUND: Entry = done("compound", "Create compound clip (subproject)", "⌥G");
const MULTICAM: Entry = todo("multicam", "Create multi-camera clip", "");
const SAVE_PRESET: Entry = todo("save-preset", "Save preset", "");
const GROUP: Entry = todo("group", "Group", "");
const LINK_MEDIA: Entry = todo("link-media", "Link to media", "");
const UNGROUP: Entry = done("ungroup", "Ungroup", "");
const VARIABLE_SPEED: Entry = done("variable-speed", "Show variable speed animation", "⌥K");

const RANGE: Entry = sub(
    "range",
    "Range",
    &[
        done("range-in", "Set in", ""),
        done("range-out", "Set out", ""),
    ],
);

/// The reference editor's stems, none of which this build separates yet.
const SEPARATE_AUDIO: Entry = sub(
    "separate-audio",
    "Separate audio",
    &[
        todo("separate-vocals", "Vocals", ""),
        todo("separate-instruments", "Instruments", ""),
        todo("separate-drums", "Drums", ""),
        todo("separate-guitar", "Guitar", ""),
        todo("separate-bass", "Bass", ""),
        todo("separate-strings", "Strings", ""),
        todo("separate-others", "Others", ""),
    ],
);

/// Stack a picture clip above or below the clips it overlaps, with Free
/// layer on in the project settings.
const ARRANGE: Entry = sub(
    "arrange",
    "Arrange",
    &[
        done("arrange-forward", "Bring forward", ""),
        done("arrange-backward", "Send backward", ""),
        done("arrange-front", "Bring to front", ""),
        done("arrange-back", "Send to back", ""),
    ],
);

/// The video clip menu: every row of the reference editor's, in its order
/// and by its names, with a Later badge on the ones this build does not do
/// yet, and this build's own Arrange row. `$sound` stands for the rows
/// that depend on whether the clip still has its sound, and `$compound`
/// for the rows only a compound clip has.
macro_rules! video_menu {
    (sound: [$($sound:expr),* $(,)?], compound: [$($compound:expr),* $(,)?]) => {
        &[
            COPY,
            CUT,
            COPY_ATTRIBUTES,
            PASTE_ATTRIBUTES,
            DELETE,
            SEP,
            sub(
                "edit",
                "Edit",
                &[
                    done("edit-freeze", "Freeze", ""),
                    done("edit-reverse", "Reverse", ""),
                    done("edit-mirror", "Mirror", ""),
                    done("edit-rotate", "Rotate", ""),
                    done("edit-crop", "Crop", ""),
                ],
            ),
            todo("generate-video", "Generate video", ""),
            todo("ai-edit", "AI edit", ""),
            todo("ai-extend", "AI extend", ""),
            done("split-scenes", "Split scenes", ""),
            done("transcript", "Transcript", ""),
            $($sound,)*
            SYNC,
            SEP,
            COMPOUND,
            $($compound,)*
            MULTICAM,
            SAVE_PRESET,
            GROUP,
            UNGROUP,
            SEP,
            done("export", "Export selected clips", ""),
            DEACTIVATE,
            todo("trim", "Trim clip", ""),
            done("replace", "Replace clip", ""),
            todo("restore-lut", "Restore LUT", ""),
            LINK_MEDIA,
            done("open-file", "Open file location", ""),
            ARRANGE,
            SEP,
            todo("edit-effects", "Edit effects", ""),
            VARIABLE_SPEED,
            RANGE,
            sub(
                "render",
                "Render",
                &[
                    done("render-preview", "Preview", ""),
                    done("render-full", "Full", ""),
                ],
            ),
        ]
    };
}

/// The menu of a video clip that has its own sound.
const VIDEO: &[Entry] = video_menu!(
    sound: [SEPARATE_AUDIO, done("extract-audio", "Extract audio", "⇧⌘S")],
    compound: []
);

/// The menu of a video clip whose sound was extracted: there is none to
/// separate, and the extract row gives the sound back.
const VIDEO_SILENT: &[Entry] = video_menu!(
    sound: [done("recover-audio", "Recover audio", "⇧⌘S")],
    compound: []
);

/// The menu of a compound clip, which the reference editor also shows for
/// a template: the video rows, and after Create compound clip its own Edit
/// more and Pre-process rows and this build's Open compound clip.
const VIDEO_COMPOUND: &[Entry] = video_menu!(
    sound: [SEPARATE_AUDIO, done("extract-audio", "Extract audio", "⇧⌘S")],
    compound: [
        done("open-compound", "Open compound clip", ""),
        todo("edit-more", "Edit more", ""),
        todo("preprocess-compound", "Pre-process compound clip (subproject)", ""),
    ]
);

/// A compound clip whose sound was extracted.
const VIDEO_COMPOUND_SILENT: &[Entry] = video_menu!(
    sound: [done("recover-audio", "Recover audio", "⇧⌘S")],
    compound: [
        done("open-compound", "Open compound clip", ""),
        todo("edit-more", "Edit more", ""),
        todo("preprocess-compound", "Pre-process compound clip (subproject)", ""),
    ]
);

/// The audio clip menu: the reference editor's rows in its order, with
/// this build's Volume, Speed, Fade and Beat rows after Delete and its
/// Replace row with the clip rows.
const AUDIO: &[Entry] = &[
    todo("generate-video", "Generate video", ""),
    COPY,
    CUT,
    COPY_ATTRIBUTES,
    PASTE_ATTRIBUTES,
    DELETE,
    SEP,
    done("volume", "Volume", ""),
    done("speed", "Speed", ""),
    done("fade", "Fade", ""),
    done("beat", "Beat", ""),
    SEP,
    done("transcript", "Transcript", ""),
    SEPARATE_AUDIO,
    SYNC,
    SEP,
    COMPOUND,
    MULTICAM,
    SAVE_PRESET,
    GROUP,
    UNGROUP,
    SEP,
    DEACTIVATE,
    REPLACE_SOUND,
    LINK_MEDIA,
    done("open-file", "Open file location", ""),
    SEP,
    RANGE,
];

/// The menu the reference editor gives a clip that is not media: a title,
/// a sticker, an effect, a filter or an adjustment. `$own` are this
/// build's rows for the kind, which open its settings, and `$turn` the
/// rows the reference puts just above Deactivate clip.
macro_rules! lane_menu {
    (own: [$($own:expr),* $(,)?], turn: [$($turn:expr),* $(,)?]) => {
        &[
            COPY,
            CUT,
            COPY_ATTRIBUTES,
            PASTE_ATTRIBUTES,
            DELETE,
            SEP,
            $($own,)*
            SYNC,
            SEP,
            COMPOUND,
            MULTICAM,
            SAVE_PRESET,
            GROUP,
            UNGROUP,
            SEP,
            $($turn,)*
            DEACTIVATE,
            SEP,
            VARIABLE_SPEED,
            RANGE,
        ]
    };
}

const TEXT: &[Entry] = lane_menu!(
    own: [
        done("edit-text", "Edit text", ""),
        done("style", "Style", ""),
        ARRANGE,
    ],
    turn: []
);

const STICKER: &[Entry] = lane_menu!(
    own: [ARRANGE],
    turn: [done("flip", "Mirror", "")]
);

const EFFECT: &[Entry] = lane_menu!(
    own: [done("edit-effect", "Edit effect", "")],
    turn: []
);

const FILTER: &[Entry] = lane_menu!(
    own: [done("edit-filter", "Edit filter", "")],
    turn: []
);

const ADJUST: &[Entry] = lane_menu!(
    own: [done("edit-adjust", "Edit adjustment", "")],
    turn: []
);

const COMPOSITION: &[Entry] = lane_menu!(
    own: [done("render-preview", "Render", "")],
    turn: []
);

const OTHER: &[Entry] = lane_menu!(own: [], turn: []);

/// The menu for two or more selected clips, as the reference editor lists
/// it, with this build's Export selected clips row. `$top` is Generate
/// video, there when every picked clip is a movie or a sound; `$sound` is
/// Separate audio, there when one of them has sound; `$compound` is Edit
/// more, there when one of them is a compound clip.
macro_rules! multi_menu {
    (top: [$($top:expr),*], sound: [$($sound:expr),*], compound: [$($compound:expr),*]) => {
        &[
            $($top,)*
            COPY,
            CUT,
            COPY_ATTRIBUTES,
            PASTE_ATTRIBUTES,
            DELETE,
            SEP,
            $($sound,)*
            SYNC,
            SEP,
            COMPOUND,
            $($compound,)*
            MULTICAM,
            SAVE_PRESET,
            GROUP,
            UNGROUP,
            SEP,
            DEACTIVATE,
            done("export", "Export selected clips", ""),
            LINK_MEDIA,
            SEP,
            VARIABLE_SPEED,
            RANGE,
        ]
    };
}

const GENERATE_VIDEO: Entry = todo("generate-video", "Generate video", "");
const EDIT_MORE: Entry = todo("edit-more", "Edit more", "");

/// The plainest of the multi-select menus: clips with no sound among them.
pub const MULTI: &[Entry] = multi_menu!(top: [], sound: [], compound: []);
const MULTI_SOUND: &[Entry] = multi_menu!(top: [], sound: [SEPARATE_AUDIO], compound: []);
const MULTI_COMPOUND: &[Entry] = multi_menu!(top: [], sound: [], compound: [EDIT_MORE]);
const MULTI_SOUND_COMPOUND: &[Entry] =
    multi_menu!(top: [], sound: [SEPARATE_AUDIO], compound: [EDIT_MORE]);
const MULTI_MEDIA: &[Entry] = multi_menu!(top: [GENERATE_VIDEO], sound: [], compound: []);
const MULTI_MEDIA_SOUND: &[Entry] =
    multi_menu!(top: [GENERATE_VIDEO], sound: [SEPARATE_AUDIO], compound: []);
const MULTI_MEDIA_COMPOUND: &[Entry] =
    multi_menu!(top: [GENERATE_VIDEO], sound: [], compound: [EDIT_MORE]);
const MULTI_MEDIA_SOUND_COMPOUND: &[Entry] =
    multi_menu!(top: [GENERATE_VIDEO], sound: [SEPARATE_AUDIO], compound: [EDIT_MORE]);

/// The menu for several selected clips: `media` when every one is a movie
/// or a sound, `sound` when one of them has sound, `compound` when one is
/// a compound clip.
#[must_use]
pub fn multi_entries(media: bool, sound: bool, compound: bool) -> &'static [Entry] {
    match (media, sound, compound) {
        (false, false, false) => MULTI,
        (false, true, false) => MULTI_SOUND,
        (false, false, true) => MULTI_COMPOUND,
        (false, true, true) => MULTI_SOUND_COMPOUND,
        (true, false, false) => MULTI_MEDIA,
        (true, true, false) => MULTI_MEDIA_SOUND,
        (true, false, true) => MULTI_MEDIA_COMPOUND,
        (true, true, true) => MULTI_MEDIA_SOUND_COMPOUND,
    }
}

/// [`clip_entries`] for the clip as it is: a video clip whose sound was
/// extracted (`silent`), and a compound clip, each have rows of their own.
#[must_use]
pub fn clip_entries_for(kind: &TrackType, silent: bool, compound: bool) -> &'static [Entry] {
    match (kind, silent, compound) {
        (TrackType::Video, false, true) => VIDEO_COMPOUND,
        (TrackType::Video, true, true) => VIDEO_COMPOUND_SILENT,
        (TrackType::Video, true, false) => VIDEO_SILENT,
        (kind, _, _) => clip_entries(kind),
    }
}

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
    done("solo-track", "Solo track", ""),
    done("delete-track", "Delete track", ""),
];

/// The menu for the picked file cards in the library. The rows in
/// [`LIBRARY_CARD_OFF`] are there but greyed, as in the prototype.
pub const LIBRARY_CARD: &[Entry] = &[
    done("open-file", "Open file location", ""),
    done("delete", "Delete", ""),
    done("new-folder-with", "New folder with selected item", ""),
    done("link-media", "Link to media", ""),
    done("make-proxy", "Make proxy", ""),
    todo("split-scenes", "Split scenes", ""),
    todo("export", "Export selected clips", ""),
];

/// [`LIBRARY_CARD`] when every picked movie has a proxy or is getting
/// one: the proxy row cancels it.
pub const LIBRARY_CARD_PROXIED: &[Entry] = &[
    done("open-file", "Open file location", ""),
    done("delete", "Delete", ""),
    done("new-folder-with", "New folder with selected item", ""),
    done("link-media", "Link to media", ""),
    done("cancel-proxy", "Cancel proxy", ""),
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

const PAD: f32 = tokens::MENU_PAD as f32;
/// A row's height. With the separator's, measured against the reference
/// editor's menus, so a menu of as many rows stands as tall as its own.
const ROW: f32 = 28.0;
const ROW_PAD: f32 = 10.0;
const GAP: f32 = 16.0;
/// The height a separator takes between two rows.
const SEPARATOR: f32 = 7.0;
const RADIUS: u8 = tokens::MENU_RADIUS;
const ROW_RADIUS: u8 = 8;
const LABEL_SIZE: f32 = 13.0;
const KEY_SIZE: f32 = 12.0;
const BADGE: &str = "Later";
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

/// Room around a menu's own window for its shadow to fall in.
const SHADOW_ROOM: f32 = 48.0;

/// Draw the menu `id` with its corner at `at`, in the window's coordinates.
///
/// On the desktop a menu is its own borderless window above the app, so it
/// can reach past the app window's edge and keeps its full size however small
/// the window is; it stays inside the monitor. Where egui draws every
/// viewport inside one window, as in tests, it is drawn inside the window.
pub fn show(
    ctx: &Context,
    id: Id,
    at: Pos2,
    entries: &'static [Entry],
    enabled: &dyn Fn(&str) -> bool,
) -> Shown {
    let window = ctx.input(|input| input.viewport().inner_rect);
    let (Some(window), false) = (window, ctx.embed_viewports()) else {
        return show_in(
            ctx,
            id,
            ctx.content_rect().shrink(EDGE),
            at,
            entries,
            enabled,
        )
        .0;
    };
    let monitor = ctx.input(|input| input.viewport().monitor_size);
    let area = menu_area(window, monitor);
    let bounds = area.shrink(EDGE);
    let full = size(ctx, entries);
    let at = window.min + at.to_vec2();
    let menu = place(bounds, at, Vec2::new(full.x, full.y.min(bounds.height())));
    // The window is as large as the menu and every place a submenu of it
    // can open, from the first frame to the last. Sized to the submenu
    // that happens to be open, it was resized and moved each time one
    // opened or closed, and a pointer swept down the rows made it flicker.
    let canvas = sub_reach(ctx, bounds, menu, full.y, entries)
        .map_or(menu, |reach| menu.union(reach))
        .expand(SHADOW_ROOM);
    // The shadow's room stops where the menu's does. A window asked for
    // above the top of the screen is pushed down by the system, and what
    // it draws lands that much lower, with its foot off the screen.
    let canvas = Rect::from_x_y_ranges(
        canvas.x_range(),
        canvas.top().max(area.top())..=canvas.bottom().min(area.bottom()),
    );

    // The menu is another window, so a press, a scroll, or Escape in the app
    // window is outside it.
    let (pressed, wheel, escape, app_focused) = ctx.input(|input| {
        (
            input.pointer.any_pressed(),
            input.smooth_scroll_delta != Vec2::ZERO,
            input.key_pressed(egui::Key::Escape),
            input.viewport().focused,
        )
    });
    let builder = egui::ViewportBuilder::default()
        .with_title("Cutlass menu")
        .with_decorations(false)
        .with_transparent(true)
        .with_has_shadow(false)
        .with_resizable(false)
        .with_taskbar(false)
        .with_active(false)
        .with_window_level(egui::WindowLevel::AlwaysOnTop)
        .with_position(canvas.min)
        .with_inner_size(canvas.size());
    let offset = canvas.min.to_vec2();
    let (mut shown, menu_focused) = ctx.show_viewport_immediate(
        egui::ViewportId::from_hash_of(id.with("menu-window")),
        builder,
        |child, _| {
            let (shown, _) = show_in(
                child,
                id,
                bounds.translate(-offset),
                at - offset,
                entries,
                enabled,
            );
            let focused = child.input(|input| input.viewport().focused);
            (shown, focused)
        },
    );
    // Another app took the focus from both windows.
    let away = app_focused == Some(false) && menu_focused == Some(false);
    shown.dismissed |= pressed || wheel || escape || away;
    shown
}

/// One key in a shortcut: a modifier drawn as an icon, or text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Key<'a> {
    Icon(&'static str),
    Text(&'a str),
}

/// The keys of `shortcut`, such as `⇧⌘C` or `Space`, in the order they are
/// drawn: the modifiers, then the key. On a Mac the modifiers are the
/// Command, Shift and Option icons and Control's `⌃`. Elsewhere they read
/// Ctrl, Alt and the Shift icon, in that order. Backspace is its icon.
fn keys(shortcut: &str, mac: bool) -> Vec<Key<'_>> {
    let key = shortcut.trim_start_matches(['⌃', '⌥', '⇧', '⌘']);
    let held = &shortcut[..shortcut.len() - key.len()];
    let has = |glyph| held.contains(glyph);
    let mut out = Vec::new();
    if mac {
        for glyph in held.chars() {
            out.push(match glyph {
                '⌘' => Key::Icon("keyCommand"),
                '⌥' => Key::Icon("keyOption"),
                '⇧' => Key::Icon("keyShift"),
                _ => Key::Text("⌃"),
            });
        }
    } else {
        if has('⌘') || has('⌃') {
            out.push(Key::Text("Ctrl"));
        }
        if has('⌥') {
            out.push(Key::Text("Alt"));
        }
        if has('⇧') {
            out.push(Key::Icon("keyShift"));
        }
    }
    match key {
        "" => {}
        "⌫" => out.push(Key::Icon("keyBackspace")),
        key => out.push(Key::Text(key)),
    }
    out
}

/// A key binding from the settings, such as `primary+shift+s`, written as a
/// menu row's shortcut: `⇧⌘S`. [`keys`] draws it for this platform.
#[must_use]
pub fn shortcut_for(chord: &str) -> String {
    cutlass_settings::keys_label(chord, true)
}

pub const MAC: bool = cfg!(target_os = "macos");
/// The edge of a key icon.
const KEY_ICON: f32 = 13.0;
/// The space between two keys.
const KEY_GAP: f32 = 8.0;

fn key_width(ctx: &Context, key: Key<'_>) -> f32 {
    match key {
        Key::Icon(_) => KEY_ICON,
        Key::Text(text) => text_width(ctx, text, KEY_SIZE),
    }
}

/// The width the keys of `shortcut` take.
pub fn keys_width(ctx: &Context, shortcut: &str) -> f32 {
    let keys = keys(shortcut, MAC);
    let gaps = keys.len().saturating_sub(1) as f32 * KEY_GAP;
    keys.iter().map(|key| key_width(ctx, *key)).sum::<f32>() + gaps
}

/// Paint the keys of `shortcut` with the last one's right edge at `end`.
pub fn paint_keys(
    ctx: &Context,
    painter: &egui::Painter,
    end: Pos2,
    shortcut: &str,
    color: egui::Color32,
) {
    let mut x = end.x - keys_width(ctx, shortcut);
    for key in keys(shortcut, MAC) {
        match key {
            Key::Icon(name) => {
                let rect = Rect::from_min_size(
                    Pos2::new(x, end.y - KEY_ICON / 2.0),
                    Vec2::splat(KEY_ICON),
                );
                if let Err(error) = crate::icons::paint(painter, rect, name, color) {
                    cutlass_logger::warn!("{error}");
                }
            }
            Key::Text(text) => {
                painter.text(
                    Pos2::new(x, end.y),
                    Align2::LEFT_CENTER,
                    text,
                    FontId::proportional(KEY_SIZE),
                    color,
                );
            }
        }
        x += key_width(ctx, key) + KEY_GAP;
    }
}

/// The monitor the window is on, in screen points. egui gives its size, not
/// its corner, so this takes the main monitor at the origin when the window
/// is on it, and no limit when it is on another.
fn monitor_bounds(window: Rect, monitor: Option<Vec2>) -> Rect {
    monitor
        .map(|size| Rect::from_min_size(Pos2::ZERO, size))
        .filter(|monitor| monitor.contains(window.min))
        .unwrap_or(Rect::EVERYTHING)
}

/// Where a menu's own window may be, in screen points: from the app
/// window's top down to the foot of its monitor, and as wide as the
/// monitor. A menu can run past the app window's sides and foot, but not
/// above its top, where the system's own menu bar may be; one taller than
/// that scrolls. With the monitor unknown it stops at the app window's
/// foot, so it still has an end to scroll within.
fn menu_area(window: Rect, monitor: Option<Vec2>) -> Rect {
    let monitor = monitor_bounds(window, monitor);
    let bottom = if monitor.bottom().is_finite() {
        monitor.bottom().max(window.bottom())
    } else {
        window.bottom()
    };
    Rect::from_x_y_ranges(monitor.x_range(), window.top()..=bottom)
}

/// Draw the menu `id` with its corner at `at`, kept inside `screen`, in
/// `ctx`'s own coordinates. A menu taller than `screen` scrolls under the
/// pointer. `enabled` says whether a built row can run now; a Later row
/// always can, so a click on it can say it is not built yet. Returns what the
/// menu did and the panels it painted: the menu, then any open submenu.
fn show_in(
    ctx: &Context,
    id: Id,
    screen: Rect,
    at: Pos2,
    entries: &'static [Entry],
    enabled: &dyn Fn(&str) -> bool,
) -> (Shown, Vec<Rect>) {
    let open_key = id.with("open");
    let scroll_key = id.with("scroll");
    let mut open: Option<&'static str> = ctx.data(|data| data.get_temp(open_key)).flatten();
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
    paint_scroll_hints(ctx, id, rect, scroll, full.y - rect.height());
    let mut rects = vec![rect];
    if let Some((children, row)) = submenu {
        let sub_rect = sub_place(ctx, screen, row, children);
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
    (shown, rects)
}

/// The height of the strip at an end of a menu that says more rows lie
/// past it.
const SCROLL_HINT: f32 = 18.0;

/// A menu taller than its room shows a small arrow at each end that has
/// rows past it: at the foot until it is scrolled to its last row, and at
/// the top once it has been scrolled at all. `furthest` is how far it can
/// scroll.
fn paint_scroll_hints(ctx: &Context, id: Id, rect: Rect, scroll: f32, furthest: f32) {
    if furthest <= 0.5 {
        return;
    }
    let painter = ctx
        .layer_painter(LayerId::new(Order::Foreground, id))
        .with_clip_rect(rect);
    let stroke = egui::Stroke::new(1.5, tokens::MENU_CHEVRON);
    for (more, up) in [(scroll > 0.5, true), (scroll < furthest - 0.5, false)] {
        if !more {
            continue;
        }
        let strip = if up {
            Rect::from_min_max(rect.min, Pos2::new(rect.right(), rect.top() + SCROLL_HINT))
        } else {
            Rect::from_min_max(
                Pos2::new(rect.left(), rect.bottom() - SCROLL_HINT),
                rect.max,
            )
        };
        // The strip covers the row sliding under it, so the arrow reads.
        let corner = RADIUS;
        let radius = if up {
            CornerRadius {
                nw: corner,
                ne: corner,
                sw: 0,
                se: 0,
            }
        } else {
            CornerRadius {
                nw: 0,
                ne: 0,
                sw: corner,
                se: corner,
            }
        };
        painter.rect_filled(strip, radius, tokens::MENU);
        let center = strip.center();
        let tip = if up { -3.0 } else { 3.0 };
        painter.line_segment(
            [center + Vec2::new(-5.0, -tip), center + Vec2::new(0.0, tip)],
            stroke,
        );
        painter.line_segment(
            [center + Vec2::new(5.0, -tip), center + Vec2::new(0.0, tip)],
            stroke,
        );
    }
}

/// Where the submenu `children` of the row at `row` opens: beside the row
/// on its right, on its left when the right has no room, and inside
/// `screen`.
fn sub_place(ctx: &Context, screen: Rect, row: Rect, children: &[Entry]) -> Rect {
    let size = size(ctx, children);
    let right = Pos2::new(row.right() - PAD, row.top() - PAD);
    let corner = if right.x + size.x > screen.right() {
        Pos2::new(row.left() - size.x + PAD, right.y)
    } else {
        right
    };
    place(screen, corner, size)
}

/// Everywhere a submenu of `entries` can open, for the menu at `rect`
/// whose rows are `full_height` tall in all: each submenu's place with the
/// menu scrolled to its top and to its foot, and so anywhere between.
/// `None` when no row has a submenu.
fn sub_reach(
    ctx: &Context,
    screen: Rect,
    rect: Rect,
    full_height: f32,
    entries: &[Entry],
) -> Option<Rect> {
    let furthest = (full_height - rect.height()).max(0.0);
    let (left, right) = (rect.left() + PAD, rect.right() - PAD);
    let mut top = rect.top() + PAD;
    let mut reach: Option<Rect> = None;
    for entry in entries {
        let item = match entry {
            Entry::Separator => {
                top += SEPARATOR;
                continue;
            }
            Entry::Item(item) => item,
        };
        if !item.children.is_empty() {
            for scroll in [0.0, furthest] {
                let row = Rect::from_min_max(
                    Pos2::new(left, top - scroll),
                    Pos2::new(right, top - scroll + ROW),
                );
                let sub = sub_place(ctx, screen, row, item.children);
                reach = Some(reach.map_or(sub, |reach| reach.union(sub)));
            }
        }
        top += ROW;
    }
    reach
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
                    row += GAP + keys_width(ctx, item.shortcut);
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

/// The Later badge centred on `center`, as a row this build cannot do yet
/// wears it. Returns where it was drawn.
pub fn paint_later_badge(ctx: &Context, painter: &egui::Painter, center: Pos2) -> Rect {
    let badge = Rect::from_center_size(center, Vec2::new(badge_width(ctx), BADGE_HEIGHT));
    painter.rect_filled(badge, CornerRadius::same(8), tokens::MENU_BADGE);
    painter.text(
        badge.center(),
        Align2::CENTER_CENTER,
        BADGE,
        FontId::proportional(BADGE_SIZE),
        egui::Color32::WHITE,
    );
    badge
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
    let shadow = tokens::MENU_SHADOW;
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
            let width = badge_width(ctx);
            paint_later_badge(
                ctx,
                &painter,
                Pos2::new(label.right() + 8.0 + width / 2.0, bounds.center().y),
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
            paint_keys(
                ctx,
                &painter,
                Pos2::new(end, bounds.center().y),
                item.shortcut,
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
    fn every_clip_menu_has_copy_cut_and_delete_in_order() {
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
            let ids = ids(clip_entries(&kind));
            let at = |id: &str| ids.iter().position(|row| *row == id);
            let (copy, cut, delete) = (at("copy"), at("cut"), at("delete"));
            assert!(copy.is_some() && copy < cut && cut < delete, "{kind:?}");
        }
    }

    #[test]
    fn the_video_menu_lists_every_prototype_row() {
        assert_eq!(
            ids(VIDEO),
            [
                "copy",
                "cut",
                "copy-attributes",
                "paste-attributes",
                "delete",
                "edit",
                "generate-video",
                "ai-edit",
                "ai-extend",
                "split-scenes",
                "transcript",
                "separate-audio",
                "extract-audio",
                "sync",
                "compound",
                "multicam",
                "save-preset",
                "group",
                "ungroup",
                "export",
                "deactivate",
                "trim",
                "replace",
                "restore-lut",
                "link-media",
                "open-file",
                "arrange",
                "edit-effects",
                "variable-speed",
                "range",
                "render",
            ]
        );
        // The reference editor's names for the rows this build renamed.
        let label = |id: &str| {
            VIDEO.iter().find_map(|entry| match entry {
                Entry::Item(item) if item.id == id => Some(item.label),
                _ => None,
            })
        };
        assert_eq!(label("compound"), Some("Create compound clip (subproject)"));
        assert_eq!(label("separate-audio"), Some("Separate audio"));
        let separate: Vec<&str> = VIDEO
            .iter()
            .find_map(|entry| match entry {
                Entry::Item(item) if item.id == "separate-audio" => Some(item.children),
                _ => None,
            })
            .unwrap()
            .iter()
            .filter_map(|entry| match entry {
                Entry::Item(item) => Some(item.label),
                Entry::Separator => None,
            })
            .collect();
        assert_eq!(
            separate,
            [
                "Vocals",
                "Instruments",
                "Drums",
                "Guitar",
                "Bass",
                "Strings",
                "Others"
            ]
        );
    }

    #[test]
    fn modifiers_draw_as_icons_on_a_mac_and_as_words_elsewhere() {
        assert_eq!(
            keys("⇧⌘C", true),
            [
                Key::Icon("keyShift"),
                Key::Icon("keyCommand"),
                Key::Text("C")
            ]
        );
        assert_eq!(keys("⌥G", false), [Key::Text("Alt"), Key::Text("G")]);
        assert_eq!(keys("⌘X", false), [Key::Text("Ctrl"), Key::Text("X")]);
        assert_eq!(keys("⌫", false), [Key::Icon("keyBackspace")]);
        assert_eq!(
            keys("⇧⌘Space", false),
            [Key::Text("Ctrl"), Key::Icon("keyShift"), Key::Text("Space")]
        );
        assert_eq!(keys("⌃Y", true), [Key::Text("⌃"), Key::Text("Y")]);
        assert_eq!(shortcut_for("primary+shift+s"), "⇧⌘S");
        assert_eq!(shortcut_for("shift+delete"), "⇧⌫");
        for name in ["keyCommand", "keyShift", "keyOption", "keyBackspace"] {
            assert!(crate::icons::require(name).is_ok(), "{name}");
        }
    }

    #[test]
    fn a_video_clip_edits_by_freeze_reverse_mirror_rotate_and_crop() {
        for menu in [VIDEO, VIDEO_SILENT] {
            let edit = menu
                .iter()
                .find_map(|entry| match entry {
                    Entry::Item(item) if item.id == "edit" => Some(item),
                    _ => None,
                })
                .expect("an Edit row");
            let rows: Vec<(&str, &str)> = edit
                .children
                .iter()
                .filter_map(|entry| match entry {
                    Entry::Item(item) => Some((item.id, item.label)),
                    Entry::Separator => None,
                })
                .collect();
            assert_eq!(
                rows,
                [
                    ("edit-freeze", "Freeze"),
                    ("edit-reverse", "Reverse"),
                    ("edit-mirror", "Mirror"),
                    ("edit-rotate", "Rotate"),
                    ("edit-crop", "Crop"),
                ]
            );
            assert!(edit.built, "every row runs");
        }
    }

    /// The rows of `menu` by label, a submenu's children left out.
    fn labels(menu: &[Entry]) -> Vec<&'static str> {
        menu.iter()
            .filter_map(|entry| match entry {
                Entry::Item(item) => Some(item.label),
                Entry::Separator => None,
            })
            .collect()
    }

    /// `rows` appear in `menu` in that order, whatever else sits between.
    fn in_order(menu: &[Entry], rows: &[&str]) -> bool {
        let mut shown = labels(menu).into_iter();
        rows.iter().all(|row| shown.any(|label| label == *row))
    }

    #[test]
    fn every_clip_menu_has_the_reference_editors_rows_in_its_order() {
        // A title, an effect, a filter and an adjustment share one list.
        let plain = [
            "Copy",
            "Cut",
            "Copy attributes",
            "Paste attributes",
            "Delete",
            "Sync video and audio",
            "Create compound clip (subproject)",
            "Create multi-camera clip",
            "Save preset",
            "Group",
            "Ungroup",
            "Deactivate clip",
            "Show variable speed animation",
            "Range",
        ];
        for kind in [
            TrackType::Text,
            TrackType::Effect,
            TrackType::Filter,
            TrackType::Adjust,
        ] {
            assert!(in_order(clip_entries(&kind), &plain), "{kind:?}");
        }
        // A sticker has Mirror just above Deactivate clip.
        let mut sticker = plain.to_vec();
        sticker.insert(11, "Mirror");
        assert!(in_order(STICKER, &sticker), "{:?}", labels(STICKER));
        assert!(!labels(TEXT).contains(&"Mirror"));

        assert!(in_order(
            AUDIO,
            &[
                "Generate video",
                "Copy",
                "Cut",
                "Copy attributes",
                "Paste attributes",
                "Delete",
                "Transcript",
                "Separate audio",
                "Sync video and audio",
                "Create compound clip (subproject)",
                "Create multi-camera clip",
                "Save preset",
                "Group",
                "Ungroup",
                "Deactivate clip",
                "Link to media",
                "Open file location",
                "Range",
            ]
        ));

        // A compound clip, as a template is, has two rows more after
        // Create compound clip, and this build's row that opens it.
        let compound = clip_entries_for(&TrackType::Video, false, true);
        assert!(in_order(
            compound,
            &[
                "Create compound clip (subproject)",
                "Edit more",
                "Pre-process compound clip (subproject)",
                "Create multi-camera clip",
            ]
        ));
        assert!(labels(compound).contains(&"Open compound clip"));
        assert!(!labels(VIDEO).contains(&"Open compound clip"));
        assert!(!labels(VIDEO).contains(&"Edit more"));
        let quiet = clip_entries_for(&TrackType::Video, true, true);
        assert!(labels(quiet).contains(&"Recover audio") && labels(quiet).contains(&"Edit more"));

        // What this build does not do yet wears the Later badge, in every
        // menu it is in; what it does is live.
        let menus = [
            VIDEO,
            VIDEO_SILENT,
            VIDEO_COMPOUND,
            VIDEO_COMPOUND_SILENT,
            AUDIO,
            TEXT,
            STICKER,
            EFFECT,
            FILTER,
            ADJUST,
            COMPOSITION,
            OTHER,
            MULTI,
            MULTI_SOUND,
            MULTI_COMPOUND,
            MULTI_SOUND_COMPOUND,
            MULTI_MEDIA,
            MULTI_MEDIA_SOUND,
            MULTI_MEDIA_COMPOUND,
            MULTI_MEDIA_SOUND_COMPOUND,
        ];
        let later = [
            "Copy attributes",
            "Paste attributes",
            "Generate video",
            "AI edit",
            "AI extend",
            "Separate audio",
            "Save preset",
            "Restore LUT",
            "Edit more",
            "Pre-process compound clip (subproject)",
            // Greyed until now; nothing in this build does them yet.
            "Sync video and audio",
            "Create multi-camera clip",
            "Group",
            "Trim clip",
            "Link to media",
            "Edit effects",
        ];
        for menu in menus {
            for entry in menu {
                let Entry::Item(item) = entry else {
                    continue;
                };
                assert_eq!(
                    !item.built,
                    later.contains(&item.label),
                    "{} in {:?}",
                    item.label,
                    labels(menu)
                );
            }
        }
    }

    #[test]
    fn a_video_clip_without_its_sound_offers_to_recover_it() {
        let with = ids(clip_entries_for(&TrackType::Video, false, false));
        let without = ids(clip_entries_for(&TrackType::Video, true, false));
        assert_eq!(with, ids(VIDEO));
        assert!(with.contains(&"extract-audio") && with.contains(&"separate-audio"));
        assert!(!with.contains(&"recover-audio"));
        // The same rows in the same order, but for the sound ones.
        let expected: Vec<&str> = with
            .iter()
            .copied()
            .filter(|id| *id != "separate-audio")
            .map(|id| {
                if id == "extract-audio" {
                    "recover-audio"
                } else {
                    id
                }
            })
            .collect();
        assert_eq!(without, expected);
        // Only a video clip has the other menu.
        assert_eq!(
            ids(clip_entries_for(&TrackType::Audio, true, true)),
            ids(clip_entries(&TrackType::Audio))
        );
    }

    #[test]
    fn the_multi_select_menu_holds_only_rows_for_every_clip() {
        // Two clips with nothing in common, as a title and a silent movie.
        assert_eq!(
            labels(multi_entries(false, false, false)),
            [
                "Copy",
                "Cut",
                "Copy attributes",
                "Paste attributes",
                "Delete",
                "Sync video and audio",
                "Create compound clip (subproject)",
                "Create multi-camera clip",
                "Save preset",
                "Group",
                "Ungroup",
                "Deactivate clip",
                "Export selected clips",
                "Link to media",
                "Show variable speed animation",
                "Range",
            ]
        );
        // A movie and a sound: Generate video leads, and Separate audio
        // comes before Sync.
        let media = labels(multi_entries(true, true, false));
        assert_eq!(media[0], "Generate video");
        assert!(in_order(
            multi_entries(true, true, false),
            &["Delete", "Separate audio", "Sync video and audio"]
        ));
        // Every kind at once, a compound clip among them: no Generate
        // video, and Edit more after Create compound clip.
        let all = multi_entries(false, true, true);
        assert!(!labels(all).contains(&"Generate video"));
        assert!(in_order(
            all,
            &[
                "Separate audio",
                "Create compound clip (subproject)",
                "Edit more",
                "Create multi-camera clip",
            ]
        ));
        // Each of the three is there or not on its own.
        for media in [false, true] {
            for sound in [false, true] {
                for compound in [false, true] {
                    let rows = labels(multi_entries(media, sound, compound));
                    assert_eq!(rows.contains(&"Generate video"), media);
                    assert_eq!(rows.contains(&"Separate audio"), sound);
                    assert_eq!(rows.contains(&"Edit more"), compound);
                }
            }
        }
    }

    #[test]
    fn the_audio_menu_lists_every_prototype_row() {
        assert_eq!(
            ids(AUDIO),
            [
                "generate-video",
                "copy",
                "cut",
                "copy-attributes",
                "paste-attributes",
                "delete",
                "volume",
                "speed",
                "fade",
                "beat",
                "transcript",
                "separate-audio",
                "sync",
                "compound",
                "multicam",
                "save-preset",
                "group",
                "ungroup",
                "deactivate",
                "replace",
                "link-media",
                "open-file",
                "range",
            ]
        );
        let later: Vec<&str> = AUDIO
            .iter()
            .filter_map(|entry| match entry {
                Entry::Item(item) if !item.built => Some(item.id),
                _ => None,
            })
            .collect();
        assert_eq!(
            later,
            [
                "generate-video",
                "copy-attributes",
                "paste-attributes",
                "separate-audio",
                "sync",
                "multicam",
                "save-preset",
                "group",
                "link-media",
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
        assert_eq!(built("edit"), Some(true));
        assert_eq!(built("copy"), Some(true));
        assert_eq!(built("range"), Some(true));
        assert_eq!(built("render"), Some(true));
        assert_eq!(built("separate-audio"), Some(false));
        // Rows the reference editor has and this build does not do yet.
        for later in [
            "copy-attributes",
            "paste-attributes",
            "generate-video",
            "ai-edit",
            "ai-extend",
            "save-preset",
            "restore-lut",
        ] {
            assert_eq!(built(later), Some(false), "{later}");
        }
        for ready in ["split-scenes", "transcript", "extract-audio", "compound"] {
            assert_eq!(built(ready), Some(true), "{ready}");
        }
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
    fn a_shortcut_is_drawn_as_keys_not_as_one_string() {
        let ctx = Context::default();
        let (_, output) = frame(&ctx, 1000.0, Vec::new());
        assert!(
            label_at(&output, "⌘C").is_none(),
            "Copy's keys are not one string"
        );
        assert!(
            label_at(&output, "C").is_some(),
            "Copy's letter is its own key"
        );
        assert!(label_at(&output, "⌫").is_none(), "Delete's key is an icon");
        if MAC {
            assert!(
                label_at(&output, "⌘").is_none(),
                "Command is an icon on a Mac"
            );
        } else {
            assert!(
                label_at(&output, "Ctrl").is_some(),
                "Command reads Ctrl elsewhere"
            );
        }
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

    /// The menu's own window is sized once to [`sub_reach`]. Were a
    /// submenu to open outside it, the window would cut it off.
    #[test]
    fn every_submenu_opens_inside_the_room_the_window_keeps_for_it() {
        // A wide screen, one too narrow for a submenu on the right, and one
        // too short for the menu, which then scrolls.
        for (screen, at) in [
            (Vec2::new(1600.0, 1400.0), Pos2::new(100.0, 100.0)),
            (Vec2::new(520.0, 1400.0), Pos2::new(260.0, 100.0)),
            (Vec2::new(1600.0, 420.0), Pos2::new(100.0, 40.0)),
        ] {
            let ctx = Context::default();
            let screen = Rect::from_min_size(Pos2::ZERO, screen);
            let bounds = screen.shrink(EDGE);
            let frame = |events: Vec<egui::Event>| {
                let mut reach = None;
                let mut rects = Vec::new();
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(screen),
                        events,
                        ..Default::default()
                    },
                    |ui| {
                        let id = ui.scope_id().with("menu");
                        let full = size(ui.ctx(), VIDEO);
                        let menu =
                            place(bounds, at, Vec2::new(full.x, full.y.min(bounds.height())));
                        reach = sub_reach(ui.ctx(), bounds, menu, full.y, VIDEO);
                        rects = show_in(ui.ctx(), id, bounds, at, VIDEO, &|_| true).1;
                    },
                );
                output.textures_delta.clear();
                (rects, reach)
            };
            let mut opened = 0;
            // Down the menu a few points at a time, with the wheel turned
            // to its foot on the way, as a tall menu is read.
            for step in 0..400 {
                let y = bounds.top() + 4.0 + (step % 200) as f32 * (bounds.height() - 8.0) / 200.0;
                let pointer = Pos2::new(at.x + 60.0, y);
                let mut events = vec![egui::Event::PointerMoved(pointer)];
                if step == 200 {
                    events.push(egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Point,
                        delta: Vec2::new(0.0, -4000.0),
                        phase: egui::TouchPhase::Move,
                        source: egui::MouseWheelSource::Wheel,
                        modifiers: egui::Modifiers::default(),
                    });
                }
                frame(events);
                let (rects, reach) = frame(vec![egui::Event::PointerMoved(pointer)]);
                if let Some(sub) = rects.get(1) {
                    opened += 1;
                    let reach = reach.expect("a menu with submenus has their reach");
                    assert!(
                        reach.expand(0.5).contains_rect(*sub),
                        "screen {screen:?}: {sub:?} is outside {reach:?}"
                    );
                }
            }
            assert!(
                opened > 4,
                "screen {screen:?}: only {opened} submenus opened"
            );
        }
        // A menu with no submenu keeps no room for one.
        let ctx = Context::default();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
            let full = size(ui.ctx(), GAP_ROW);
            let menu = place(screen, Pos2::new(10.0, 10.0), full);
            assert_eq!(sub_reach(ui.ctx(), screen, menu, full.y, GAP_ROW), None);
        });
        output.textures_delta.clear();
    }

    #[test]
    fn a_click_on_a_todo_row_picks_it_as_not_built() {
        let ctx = Context::default();
        let (_, output) = frame(&ctx, 1000.0, Vec::new());
        let isolate = label_at(&output, "Separate audio").expect("the Separate audio row")
            + Vec2::new(4.0, 6.0);
        frame(&ctx, 1000.0, vec![egui::Event::PointerMoved(isolate)]);
        let (_, output) = frame(&ctx, 1000.0, vec![egui::Event::PointerMoved(isolate)]);
        let row = label_at(&output, "Vocals").expect("the Vocals row") + Vec2::new(4.0, 6.0);
        frame(&ctx, 1000.0, vec![egui::Event::PointerMoved(row)]);
        let (shown, _) = frame(&ctx, 1000.0, click(row));
        let picked = shown.picked.expect("a pick");
        assert_eq!((picked.id, picked.built), ("separate-vocals", false));
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

    /// The strokes `output` draws in the middle of a menu's width, wholly
    /// above `top` or wholly below `bottom` and inside the menu: the
    /// arrows that say it has rows past an end.
    fn end_arrows(output: &egui::FullOutput, top: f32, bottom: f32) -> (usize, usize) {
        let (mut above, mut below) = (0, 0);
        for clipped in &output.shapes {
            let egui::Shape::LineSegment { points, .. } = &clipped.shape else {
                continue;
            };
            let menu = clipped.clip_rect;
            let middle = points
                .iter()
                .all(|point| menu.contains(*point) && (point.x - menu.center().x).abs() < 8.0);
            if !middle {
                continue;
            }
            if points.iter().all(|point| point.y < top) {
                above += 1;
            } else if points.iter().all(|point| point.y > bottom) {
                below += 1;
            }
        }
        (above, below)
    }

    #[test]
    fn a_menu_with_rows_past_an_end_points_that_way() {
        let wheel = |by: f32| egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: Vec2::new(0.0, by),
            phase: egui::TouchPhase::Move,
            source: egui::MouseWheelSource::Wheel,
            modifiers: egui::Modifiers::default(),
        };
        let height = 400.0;
        let ends = |output: &egui::FullOutput| {
            end_arrows(output, EDGE + SCROLL_HINT, height - EDGE - SCROLL_HINT)
        };
        let over = Pos2::new(160.0, 200.0);
        let ctx = Context::default();
        // At the top there is more below only: one arrow of two strokes.
        let (_, output) = frame(&ctx, height, Vec::new());
        assert_eq!(ends(&output), (0, 2));
        // Scrolled a little, there is more both ways.
        frame(
            &ctx,
            height,
            vec![egui::Event::PointerMoved(over), wheel(-120.0)],
        );
        let (_, output) = frame(&ctx, height, vec![egui::Event::PointerMoved(over)]);
        assert_eq!(ends(&output), (2, 2));
        // At the last row, only above.
        frame(
            &ctx,
            height,
            vec![egui::Event::PointerMoved(over), wheel(-9000.0)],
        );
        let (_, output) = frame(&ctx, height, vec![egui::Event::PointerMoved(over)]);
        assert_eq!(ends(&output), (2, 0));
        // A menu that fits has no arrows.
        let ctx = Context::default();
        let (_, output) = frame(&ctx, 2000.0, Vec::new());
        assert_eq!(
            end_arrows(&output, EDGE + SCROLL_HINT, 2000.0 - EDGE - SCROLL_HINT),
            (0, 0)
        );
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
    fn a_menu_window_is_bound_by_the_monitor_not_the_app_window() {
        let window = Rect::from_min_size(Pos2::new(300.0, 200.0), Vec2::new(800.0, 400.0));
        let monitor = monitor_bounds(window, Some(Vec2::new(1728.0, 1117.0)));
        assert_eq!(
            monitor,
            Rect::from_min_size(Pos2::ZERO, Vec2::new(1728.0, 1117.0))
        );
        // Near the window's foot, the menu runs down past it, onto the monitor.
        let at = window.min + Vec2::new(400.0, 380.0);
        let menu = place(monitor.shrink(EDGE), at, Vec2::new(248.0, 300.0));
        assert_eq!(menu.min, at);
        assert!(menu.bottom() > window.bottom());
        // A window on another monitor sets no limit rather than a wrong one.
        let elsewhere = window.translate(Vec2::new(-2000.0, 0.0));
        assert_eq!(
            monitor_bounds(elsewhere, Some(Vec2::new(1728.0, 1117.0))),
            Rect::EVERYTHING
        );
    }

    #[test]
    fn a_menu_stays_under_the_app_windows_top_and_scrolls_when_too_tall() {
        // A small laptop's screen, which the video clip menu outgrows.
        let screen = Vec2::new(1280.0, 800.0);
        let window = Rect::from_min_size(Pos2::new(40.0, 34.0), Vec2::new(1200.0, 700.0));
        let area = menu_area(window, Some(screen));
        // Not above the app window, where the system's menu bar is, and
        // down to the monitor's foot.
        assert_eq!(area.top(), window.top());
        assert_eq!(area.bottom(), screen.y);
        assert_eq!((area.left(), area.right()), (0.0, screen.x));

        // The video clip menu is taller than that: it takes the room there
        // is, starts at the top of it, and has the rest to scroll.
        let ctx = Context::default();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            let bounds = area.shrink(EDGE);
            let full = size(ui.ctx(), VIDEO);
            assert!(full.y > bounds.height(), "the menu outgrew the screen");
            let click = Pos2::new(700.0, 600.0);
            let menu = place(
                bounds,
                click,
                Vec2::new(full.x, full.y.min(bounds.height())),
            );
            assert_eq!(menu.top(), bounds.top());
            assert_eq!(menu.bottom(), bounds.bottom());
            assert!(full.y - menu.height() > 0.0, "there is something to scroll");
            // Its window, shadow and all, asks for nothing outside the area.
            let canvas = sub_reach(ui.ctx(), bounds, menu, full.y, VIDEO)
                .map_or(menu, |reach| menu.union(reach))
                .expand(SHADOW_ROOM);
            let canvas = Rect::from_x_y_ranges(
                canvas.x_range(),
                canvas.top().max(area.top())..=canvas.bottom().min(area.bottom()),
            );
            assert!(canvas.top() >= window.top());
            assert!(canvas.bottom() <= screen.y);
            assert!(canvas.contains_rect(menu));
        });
        output.textures_delta.clear();

        // On a 14-inch laptop the whole of it shows, as the reference
        // editor's does: its rows and separators are no taller than theirs.
        let roomy = menu_area(window, Some(Vec2::new(1512.0, 982.0))).shrink(EDGE);
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            assert!(size(ui.ctx(), VIDEO).y <= roomy.height());
        });
        output.textures_delta.clear();

        // On a monitor it cannot place, the menu ends with the app window.
        let elsewhere = window.translate(Vec2::new(-3000.0, 0.0));
        let area = menu_area(elsewhere, Some(screen));
        assert_eq!(
            (area.top(), area.bottom()),
            (elsewhere.top(), elsewhere.bottom())
        );
        assert!(!area.left().is_finite(), "its sides are not held");
        // A window hanging below the monitor still shows its menus.
        let low = window.translate(Vec2::new(0.0, 400.0));
        assert_eq!(menu_area(low, Some(screen)).bottom(), low.bottom());
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
