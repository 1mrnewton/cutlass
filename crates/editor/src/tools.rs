//! The phone's tool bars: which tools each bar holds, in order, and which of
//! them work yet.
//!
//! Every shell draws the same bars from this list and maps a tool to its own
//! icon. A tool that does not work yet still shows, so the layout is final,
//! and says so when tapped. A tool that opens another bar names it.

use cutlass_model::TrackType;

/// A tool bar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bar {
    /// Nothing selected.
    Root,
    /// A clip on the main row.
    MainClip,
    /// A picture clip on any other row.
    OverlayClip,
    /// A sound clip.
    AudioClip,
    /// A text clip.
    TextClip,
    /// Any other clip: an effect, filter, adjustment or sticker.
    OtherClip,
    Audio,
    Text,
    Effects,
    Overlay,
}

/// One tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToolId {
    Edit,
    Audio,
    Text,
    Effects,
    Overlay,
    Captions,
    Filters,
    Adjust,
    Stickers,
    AspectRatio,
    Background,
    Split,
    Volume,
    Speed,
    Animations,
    Delete,
    Duplicate,
    Replace,
    Crop,
    Mirror,
    Rotate,
    Reverse,
    Freeze,
    ExtractAudio,
    Fade,
    Beats,
    Unlink,
    Splice,
    AddOverlay,
    ImportSound,
    Record,
    AddText,
    EditText,
    Draw,
    VideoEffects,
}

/// A tool as a bar shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tool {
    pub id: ToolId,
    pub label: &'static str,
    /// It works on the phone today.
    pub ready: bool,
    /// The bar it opens instead of acting.
    pub opens: Option<Bar>,
}

const fn ready(id: ToolId, label: &'static str) -> Tool {
    Tool {
        id,
        label,
        ready: true,
        opens: None,
    }
}

const fn later(id: ToolId, label: &'static str) -> Tool {
    Tool {
        id,
        label,
        ready: false,
        opens: None,
    }
}

const fn opens(id: ToolId, label: &'static str, bar: Bar) -> Tool {
    Tool {
        id,
        label,
        ready: true,
        opens: Some(bar),
    }
}

use ToolId as T;

const ROOT: &[Tool] = &[
    ready(T::Edit, "Edit"),
    opens(T::Audio, "Audio", Bar::Audio),
    opens(T::Text, "Text", Bar::Text),
    opens(T::Effects, "Effects", Bar::Effects),
    opens(T::Overlay, "Overlay", Bar::Overlay),
    later(T::Captions, "Captions"),
    later(T::Filters, "Filters"),
    later(T::Adjust, "Adjust"),
    later(T::Stickers, "Stickers"),
    ready(T::AspectRatio, "Ratio"),
    ready(T::Background, "Background"),
];

const MAIN_CLIP: &[Tool] = &[
    ready(T::Split, "Split"),
    ready(T::Volume, "Volume"),
    ready(T::Speed, "Speed"),
    later(T::Animations, "Animations"),
    later(T::Effects, "Effects"),
    ready(T::Delete, "Delete"),
    later(T::Duplicate, "Duplicate"),
    later(T::Crop, "Crop"),
    later(T::Mirror, "Mirror"),
    later(T::Rotate, "Rotate"),
    later(T::Reverse, "Reverse"),
    later(T::Freeze, "Freeze"),
    later(T::ExtractAudio, "Extract audio"),
    later(T::Filters, "Filters"),
    later(T::Adjust, "Adjust"),
    later(T::Replace, "Replace"),
    later(T::Beats, "Beats"),
    later(T::Unlink, "Unlink"),
];

const OVERLAY_CLIP: &[Tool] = &[
    ready(T::Split, "Split"),
    ready(T::Volume, "Volume"),
    ready(T::Speed, "Speed"),
    later(T::Splice, "To main"),
    later(T::Animations, "Animations"),
    ready(T::Delete, "Delete"),
    later(T::Duplicate, "Duplicate"),
    later(T::Crop, "Crop"),
    later(T::Mirror, "Mirror"),
    later(T::Rotate, "Rotate"),
    later(T::Reverse, "Reverse"),
    later(T::Filters, "Filters"),
    later(T::Adjust, "Adjust"),
];

const AUDIO_CLIP: &[Tool] = &[
    ready(T::Split, "Split"),
    ready(T::Volume, "Volume"),
    ready(T::Speed, "Speed"),
    later(T::Fade, "Fade"),
    ready(T::Delete, "Delete"),
    later(T::Duplicate, "Duplicate"),
    later(T::Beats, "Beats"),
];

const TEXT_CLIP: &[Tool] = &[
    ready(T::EditText, "Edit"),
    ready(T::Split, "Split"),
    later(T::Animations, "Animations"),
    ready(T::Delete, "Delete"),
    later(T::Duplicate, "Duplicate"),
];

const OTHER_CLIP: &[Tool] = &[
    ready(T::Split, "Split"),
    ready(T::Delete, "Delete"),
    later(T::Duplicate, "Duplicate"),
];

const AUDIO: &[Tool] = &[
    later(T::ImportSound, "Sounds"),
    later(T::ExtractAudio, "Extract"),
    later(T::Record, "Record"),
];

const TEXT: &[Tool] = &[
    ready(T::AddText, "Add text"),
    later(T::Captions, "Captions"),
    later(T::Stickers, "Stickers"),
    later(T::Draw, "Draw"),
];

const EFFECTS: &[Tool] = &[ready(T::VideoEffects, "Video effects")];

const OVERLAY: &[Tool] = &[later(T::AddOverlay, "Add overlay")];

/// The tools on `bar`, in order.
#[must_use]
pub fn tools(bar: Bar) -> &'static [Tool] {
    match bar {
        Bar::Root => ROOT,
        Bar::MainClip => MAIN_CLIP,
        Bar::OverlayClip => OVERLAY_CLIP,
        Bar::AudioClip => AUDIO_CLIP,
        Bar::TextClip => TEXT_CLIP,
        Bar::OtherClip => OTHER_CLIP,
        Bar::Audio => AUDIO,
        Bar::Text => TEXT,
        Bar::Effects => EFFECTS,
        Bar::Overlay => OVERLAY,
    }
}

/// The bar for a clip selected on a row of `kind`.
#[must_use]
pub fn clip_bar(kind: TrackType, main: bool) -> Bar {
    match kind {
        TrackType::Video if main => Bar::MainClip,
        TrackType::Video => Bar::OverlayClip,
        TrackType::Audio => Bar::AudioClip,
        TrackType::Text => Bar::TextClip,
        _ => Bar::OtherClip,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    const BARS: [Bar; 10] = [
        Bar::Root,
        Bar::MainClip,
        Bar::OverlayClip,
        Bar::AudioClip,
        Bar::TextClip,
        Bar::OtherClip,
        Bar::Audio,
        Bar::Text,
        Bar::Effects,
        Bar::Overlay,
    ];

    #[test]
    fn every_bar_has_tools_once_each_with_a_label() {
        for bar in BARS {
            let list = tools(bar);
            assert!(!list.is_empty(), "{bar:?}");
            let ids: HashSet<_> = list.iter().map(|tool| tool.id).collect();
            assert_eq!(ids.len(), list.len(), "{bar:?} repeats a tool");
            assert!(list.iter().all(|tool| !tool.label.trim().is_empty()));
        }
    }

    #[test]
    fn a_tool_that_opens_a_bar_opens_a_bar_with_no_way_back_to_itself() {
        for bar in BARS {
            for tool in tools(bar) {
                if let Some(next) = tool.opens {
                    assert!(tool.ready, "{:?} opens a bar but is not ready", tool.id);
                    assert_ne!(next, bar);
                    assert!(tools(next).iter().all(|inner| inner.opens.is_none()));
                }
            }
        }
    }

    #[test]
    fn every_clip_bar_can_delete_and_split() {
        for bar in [
            Bar::MainClip,
            Bar::OverlayClip,
            Bar::AudioClip,
            Bar::TextClip,
            Bar::OtherClip,
        ] {
            for id in [ToolId::Split, ToolId::Delete] {
                assert!(
                    tools(bar).iter().any(|tool| tool.id == id && tool.ready),
                    "{bar:?} lacks {id:?}"
                );
            }
        }
    }

    #[test]
    fn a_row_picks_its_clip_bar() {
        assert_eq!(clip_bar(TrackType::Video, true), Bar::MainClip);
        assert_eq!(clip_bar(TrackType::Video, false), Bar::OverlayClip);
        assert_eq!(clip_bar(TrackType::Audio, false), Bar::AudioClip);
        assert_eq!(clip_bar(TrackType::Text, false), Bar::TextClip);
        assert_eq!(clip_bar(TrackType::Effect, false), Bar::OtherClip);
    }
}
