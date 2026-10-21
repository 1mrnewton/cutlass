//! The shell's words.
//!
//! English and French are complete. Any other language tag stays English.
//! The choice is view state on the window. It is not a project field.
//!
//! Codec names, containers, sample rates, frame rates, ratios other than
//! Original, and units stay as written. `Fit`, `−`, and `+` stay the zoom
//! marks. A lane name stays short enough for the track head.

use eframe::egui::{self, FontId};

/// A language the shell can speak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Language {
    English,
    French,
}

/// One phrase the shell can show.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phrase {
    RegionProject,
    RegionLibrary,
    RegionPreview,
    RegionInspector,
    RegionTimeline,
    Export,
    Close,
    Cancel,
    Play,
    Still,
    Playhead,
    ZoomOut,
    ZoomToFit,
    ZoomIn,
    Fit,
    Split,
    Snap,
    Marker,
    Beat,
    Hide,
    Hidden,
    Mute,
    Muted,
    Lock,
    Locked,
    TrackLocked,
    TrackUnlocked,
    MagnetLocked,
    MagnetOn,
    MagnetOff,
    LinkageOn,
    LinkageOff,
    MagnetLockedOn,
    MagnetLockedOff,
    MagnetFollower,
    DragClip,
    SpotTaken,
    WrongLane,
    CutTooClose,
    TrimLeaves,
    EditMissed,
    NothingToUndo,
    NothingToRedo,
    TrackIsLocked,
    Main,
    Video,
    Audio,
    Text,
    Sticker,
    Effect,
    Filter,
    Adjust,
    Media,
    Stickers,
    Effects,
    Transitions,
    Filters,
    Adjustments,
    Templates,
    OnTimeline,
    Empty,
    Details,
    Name,
    Saved,
    NotSaved,
    ColorSpace,
    Ratio,
    Resolution,
    FrameRate,
    DropFrame,
    On,
    Off,
    Start,
    Duration,
    Speed,
    Volume,
    Position,
    Size,
    Original,
    Format,
    Codec,
    Quality,
    RateControl,
    SampleRate,
    Channels,
    Bitrate,
    Caption,
    File,
    Loop,
    Gif,
    Standard,
    High,
    Best,
    Stereo,
    Mono,
    Once,
    BurnIn,
    Copy,
    Cut,
    Delete,
    Freeze,
    Reverse,
    Mirror,
    Rotate,
    Crop,
    Transcribe,
    SplitScenes,
    ResizeLibrary,
    ResizeInspector,
    ResizeTimeline,
    All,
    NewFolder,
    Sort,
    UntitledFolder,
    OpenLocation,
    NewFolderWith,
    Rename,
    LinkMedia,
    ExportClips,
    ExportWritesNothing,
    DoesNotOpenFile,
    Free,
    SortedByName,
    SortedByDuration,
    ProjectMedia,
    FreeStock,
    Basic,
    Rotation,
    Opacity,
    Keyframe,
}

#[cfg(test)]
impl Phrase {
    const ALL: &'static [Self] = &[
        Self::RegionProject,
        Self::RegionLibrary,
        Self::RegionPreview,
        Self::RegionInspector,
        Self::RegionTimeline,
        Self::Export,
        Self::Close,
        Self::Cancel,
        Self::Play,
        Self::Still,
        Self::Playhead,
        Self::ZoomOut,
        Self::ZoomToFit,
        Self::ZoomIn,
        Self::Fit,
        Self::Split,
        Self::Snap,
        Self::Marker,
        Self::Beat,
        Self::Hide,
        Self::Hidden,
        Self::Mute,
        Self::Muted,
        Self::Lock,
        Self::Locked,
        Self::TrackLocked,
        Self::TrackUnlocked,
        Self::MagnetLocked,
        Self::MagnetOn,
        Self::MagnetOff,
        Self::LinkageOn,
        Self::LinkageOff,
        Self::MagnetLockedOn,
        Self::MagnetLockedOff,
        Self::MagnetFollower,
        Self::DragClip,
        Self::SpotTaken,
        Self::WrongLane,
        Self::CutTooClose,
        Self::TrimLeaves,
        Self::EditMissed,
        Self::NothingToUndo,
        Self::NothingToRedo,
        Self::TrackIsLocked,
        Self::Main,
        Self::Video,
        Self::Audio,
        Self::Text,
        Self::Sticker,
        Self::Effect,
        Self::Filter,
        Self::Adjust,
        Self::Media,
        Self::Stickers,
        Self::Effects,
        Self::Transitions,
        Self::Filters,
        Self::Adjustments,
        Self::Templates,
        Self::OnTimeline,
        Self::Empty,
        Self::Details,
        Self::Name,
        Self::Saved,
        Self::NotSaved,
        Self::ColorSpace,
        Self::Ratio,
        Self::Resolution,
        Self::FrameRate,
        Self::DropFrame,
        Self::On,
        Self::Off,
        Self::Start,
        Self::Duration,
        Self::Speed,
        Self::Volume,
        Self::Position,
        Self::Size,
        Self::Original,
        Self::Format,
        Self::Codec,
        Self::Quality,
        Self::RateControl,
        Self::SampleRate,
        Self::Channels,
        Self::Bitrate,
        Self::Caption,
        Self::File,
        Self::Loop,
        Self::Gif,
        Self::Standard,
        Self::High,
        Self::Best,
        Self::Stereo,
        Self::Mono,
        Self::Once,
        Self::BurnIn,
        Self::Copy,
        Self::Cut,
        Self::Delete,
        Self::Freeze,
        Self::Reverse,
        Self::Mirror,
        Self::Rotate,
        Self::Crop,
        Self::Transcribe,
        Self::SplitScenes,
        Self::ResizeLibrary,
        Self::ResizeInspector,
        Self::ResizeTimeline,
        Self::All,
        Self::NewFolder,
        Self::Sort,
        Self::UntitledFolder,
        Self::OpenLocation,
        Self::NewFolderWith,
        Self::Rename,
        Self::LinkMedia,
        Self::ExportClips,
        Self::ExportWritesNothing,
        Self::DoesNotOpenFile,
        Self::Free,
        Self::SortedByName,
        Self::SortedByDuration,
        Self::ProjectMedia,
        Self::FreeStock,
        Self::Basic,
        Self::Rotation,
        Self::Opacity,
        Self::Keyframe,
    ];
}

const ENGLISH: &[&str] = &[
    "Project",
    "Library",
    "Preview",
    "Inspector",
    "Timeline",
    "Export",
    "Close",
    "Cancel",
    "Play",
    "Still",
    "Playhead",
    "Zoom out",
    "Zoom to fit",
    "Zoom in",
    "Fit",
    "Split",
    "Snap",
    "Marker",
    "Beat",
    "Hide",
    "Hidden",
    "Mute",
    "Muted",
    "Lock",
    "Locked",
    "Track locked",
    "Track unlocked",
    "Main track is locked",
    "Main track magnet on",
    "Main track magnet off",
    "Linkage on",
    "Linkage off",
    "Main track is locked. Couldn't turn on main track magnet.",
    "Main track is locked. Couldn't turn off main track magnet.",
    "A locked lane would move. Couldn't turn on main track magnet.",
    "Drag a clip here",
    "That spot is taken.",
    "That lane can't hold this clip.",
    "That cut is too close to the edge.",
    "That trim leaves the file.",
    "That edit did not land.",
    "Nothing to undo.",
    "Nothing to redo.",
    "This track is locked.",
    "Main",
    "Video",
    "Audio",
    "Text",
    "Sticker",
    "Effect",
    "Filter",
    "Adjust",
    "Media",
    "Stickers",
    "Effects",
    "Transitions",
    "Filters",
    "Adjustments",
    "Templates",
    "On timeline",
    "Empty",
    "Details",
    "Name",
    "Saved",
    "Not saved",
    "Color space",
    "Ratio",
    "Resolution",
    "Frame rate",
    "Drop frame",
    "On",
    "Off",
    "Start",
    "Duration",
    "Speed",
    "Volume",
    "Position",
    "Size",
    "Original",
    "Format",
    "Codec",
    "Quality",
    "Rate",
    "Sample rate",
    "Channels",
    "Bitrate",
    "Caption",
    "File",
    "Loop",
    "GIF",
    "Standard",
    "High",
    "Best",
    "Stereo",
    "Mono",
    "Once",
    "Burn in",
    "Copy",
    "Cut",
    "Delete",
    "Freeze",
    "Reverse",
    "Mirror",
    "Rotate",
    "Crop",
    "Transcribe",
    "Split scenes",
    "Resize library",
    "Resize inspector",
    "Resize timeline",
    "All",
    "New folder",
    "Sort",
    "Untitled folder",
    "Open file location",
    "New folder with selected item",
    "Rename",
    "Link to media",
    "Export selected clips",
    "Export does not write a file.",
    "Does not open the file.",
    "Free",
    "Sorted by name",
    "Sorted by duration",
    "Project",
    "Free stock",
    "Basic",
    "Rotation",
    "Opacity",
    "Keyframe",
];

const FRENCH: &[&str] = &[
    "Projet",
    "Bibliothèque",
    "Aperçu",
    "Inspecteur",
    "Montage",
    "Exporter",
    "Fermer",
    "Annuler",
    "Lecture",
    "Image fixe",
    "Tête de lecture",
    "Dézoomer",
    "Ajuster",
    "Zoomer",
    "Fit",
    "Scinder",
    "Caler",
    "Marqueur",
    "Pulsation",
    "Masquer",
    "Masqué",
    "Muet",
    "En sourdine",
    "Verrou",
    "Verrouillé",
    "Piste verrouillée",
    "Piste déverrouillée",
    "La piste principale est verrouillée",
    "Aimant de la piste principale activé",
    "Aimant de la piste principale désactivé",
    "Liaison activée",
    "Liaison désactivée",
    "La piste principale est verrouillée. L'aimant ne s'active pas.",
    "La piste principale est verrouillée. L'aimant ne se désactive pas.",
    "Une piste verrouillée bougerait. L'aimant ne s'active pas.",
    "Déposez un clip ici",
    "Cette place est prise.",
    "Cette piste ne peut pas recevoir ce clip.",
    "Cette coupe est trop près du bord.",
    "Ce rognage sort du fichier.",
    "Cette modification n'a pas abouti.",
    "Rien à annuler.",
    "Rien à rétablir.",
    "Cette piste est verrouillée.",
    "Piste",
    "Vidéo",
    "Audio",
    "Texte",
    "Sticker",
    "Effet",
    "Filtre",
    "Ajust",
    "Médias",
    "Autocollants",
    "Effets",
    "Transitions",
    "Filtres",
    "Réglages",
    "Modèles",
    "Au montage",
    "Vide",
    "Détails",
    "Nom",
    "Enregistré",
    "Pas enregistré",
    "Espace couleur",
    "Ratio",
    "Résolution",
    "Cadence",
    "Perte d'image",
    "Oui",
    "Non",
    "Début",
    "Durée",
    "Vitesse",
    "Volume",
    "Position",
    "Taille",
    "Original",
    "Format",
    "Codec",
    "Qualité",
    "Mode",
    "Fréquence",
    "Canaux",
    "Débit",
    "Sous-titres",
    "Fichier",
    "Boucle",
    "GIF",
    "Standard",
    "Élevée",
    "Maximale",
    "Stéréo",
    "Mono",
    "Une fois",
    "Incrusté",
    "Copier",
    "Couper",
    "Supprimer",
    "Figer",
    "Inverser",
    "Miroir",
    "Pivoter",
    "Rogner",
    "Transcrire",
    "Couper les scènes",
    "Redimensionner la bibliothèque",
    "Redimensionner l'inspecteur",
    "Redimensionner le montage",
    "Tout",
    "Nouveau dossier",
    "Trier",
    "Dossier sans titre",
    "Ouvrir l'emplacement du fichier",
    "Nouveau dossier avec la sélection",
    "Renommer",
    "Lier au média",
    "Exporter les éléments sélectionnés",
    "L'export n'écrit pas de fichier.",
    "N'ouvre pas le fichier.",
    "Gratuit",
    "Trié par nom",
    "Trié par durée",
    "Projet",
    "Stock libre",
    "Basique",
    "Rotation",
    "Opacité",
    "Image clé",
];

/// The word for `phrase` in `language`.
///
/// The tables follow [`Phrase`] in declaration order.
pub(crate) fn say(language: Language, phrase: Phrase) -> &'static str {
    let words = match language {
        Language::English => ENGLISH,
        Language::French => FRENCH,
    };
    words
        .get(phrase as usize)
        .copied()
        .expect("every phrase has a word")
}

/// `name on` or `name off`, in the active language.
pub(crate) fn named_switch(language: Language, name: &str, on: bool) -> String {
    match (language, on) {
        (Language::English, true) => format!("{name} on"),
        (Language::English, false) => format!("{name} off"),
        (Language::French, true) => format!("{name} : oui"),
        (Language::French, false) => format!("{name} : non"),
    }
}

/// Width of a control that paints `text`, never below `min`.
pub(crate) fn control_width(ui: &egui::Ui, text: &str, font: FontId, pad: f32, min: f32) -> f32 {
    let width = ui
        .painter()
        .layout_no_wrap(text.to_owned(), font, crate::theme::INK)
        .size()
        .x;
    (width + pad).max(min)
}

fn language_id() -> egui::Id {
    egui::Id::unique("shell-language")
}

/// Remember `language` for this frame. An unset window speaks English.
pub(crate) fn install(ctx: &egui::Context, language: Language) {
    ctx.data_mut(|data| data.insert_temp(language_id(), language));
}

/// The language installed for this frame, or English.
pub(crate) fn active(ui: &egui::Ui) -> Language {
    ui.data(|data| data.get_temp(language_id()))
        .unwrap_or(Language::English)
}

/// [`say`] for the language installed on `ui`.
pub(crate) fn of(ui: &egui::Ui, phrase: Phrase) -> &'static str {
    say(active(ui), phrase)
}

/// `LC_ALL`, then `LC_MESSAGES`, then `LANG`. An empty tag stays English.
pub(crate) fn system_tag() -> String {
    for key in ["LC_ALL", "LC_MESSAGES", "LANG"] {
        if let Ok(value) = std::env::var(key)
            && !value.is_empty()
            && value != "C"
            && value != "POSIX"
        {
            return value;
        }
    }
    String::new()
}

/// The language of a tag such as `fr_FR.UTF-8`. Anything other than French stays English.
pub(crate) fn language_from_tag(tag: &str) -> Language {
    match primary_subtag(tag).as_str() {
        "fr" => Language::French,
        _ => Language::English,
    }
}

fn primary_subtag(tag: &str) -> String {
    let token = tag.split(['.', '@']).next().unwrap_or_default();
    let primary = token.split(['_', '-']).next().unwrap_or_default();
    primary.to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::{Language, Phrase, language_from_tag, named_switch, say};

    const SHARED: &[Phrase] = &[
        Phrase::Fit,
        Phrase::Audio,
        Phrase::Sticker,
        Phrase::Transitions,
        Phrase::Ratio,
        Phrase::Volume,
        Phrase::Position,
        Phrase::Rotation,
        Phrase::Original,
        Phrase::Format,
        Phrase::Codec,
        Phrase::Gif,
        Phrase::Standard,
        Phrase::Mono,
    ];

    const LANES: &[Phrase] = &[
        Phrase::Main,
        Phrase::Video,
        Phrase::Audio,
        Phrase::Text,
        Phrase::Sticker,
        Phrase::Effect,
        Phrase::Filter,
        Phrase::Adjust,
    ];

    #[test]
    fn the_catalog_covers_every_phrase_in_both_languages() {
        assert_eq!(Phrase::ALL.len(), super::ENGLISH.len());
        assert_eq!(Phrase::ALL.len(), super::FRENCH.len());
        assert_eq!(say(Language::English, Phrase::RegionProject), "Project");
        assert_eq!(say(Language::English, Phrase::BurnIn), "Burn in");
        assert_eq!(say(Language::French, Phrase::BurnIn), "Incrusté");
        assert_eq!(say(Language::English, Phrase::SplitScenes), "Split scenes");
        assert_eq!(
            say(Language::English, Phrase::ResizeTimeline),
            "Resize timeline"
        );
        assert_eq!(
            say(Language::English, Phrase::SortedByDuration),
            "Sorted by duration"
        );
        assert_eq!(say(Language::English, Phrase::FreeStock), "Free stock");
        assert_eq!(say(Language::French, Phrase::FreeStock), "Stock libre");
        assert_eq!(say(Language::English, Phrase::Basic), "Basic");
        assert_eq!(say(Language::French, Phrase::Basic), "Basique");
        assert_eq!(say(Language::English, Phrase::Opacity), "Opacity");
        assert_eq!(say(Language::French, Phrase::Opacity), "Opacité");
        assert_eq!(say(Language::English, Phrase::Keyframe), "Keyframe");
        assert_eq!(say(Language::French, Phrase::Keyframe), "Image clé");
        assert_eq!(Phrase::Keyframe as usize, super::ENGLISH.len() - 1);
        for phrase in Phrase::ALL {
            let english = say(Language::English, *phrase);
            let french = say(Language::French, *phrase);
            assert!(!english.is_empty() && !french.is_empty());
            if SHARED.contains(phrase) {
                assert_eq!(english, french, "{phrase:?}");
            } else {
                assert_ne!(english, french, "{phrase:?}");
            }
        }
    }

    #[test]
    fn a_lane_name_stays_short_enough_for_the_track_head() {
        for phrase in LANES {
            let limit = if *phrase == Phrase::Sticker { 7 } else { 6 };
            let french = say(Language::French, *phrase);
            assert!(french.chars().count() <= limit, "{phrase:?} is {french}");
        }
    }

    #[test]
    fn a_language_tag_picks_french_or_stays_english() {
        assert_eq!(language_from_tag("fr_FR.UTF-8"), Language::French);
        assert_eq!(language_from_tag("fr-CA"), Language::French);
        assert_eq!(language_from_tag("FR"), Language::French);
        assert_eq!(language_from_tag("en_US.UTF-8"), Language::English);
        assert_eq!(language_from_tag("zh-Hans"), Language::English);
        assert_eq!(language_from_tag(""), Language::English);
        assert_eq!(language_from_tag("C"), Language::English);
    }

    #[test]
    fn a_section_switch_names_whether_that_part_is_on() {
        assert_eq!(named_switch(Language::English, "Video", true), "Video on");
        assert_eq!(named_switch(Language::English, "Video", false), "Video off");
        assert_eq!(named_switch(Language::French, "Vidéo", true), "Vidéo : oui");
        assert_eq!(
            named_switch(Language::French, "Vidéo", false),
            "Vidéo : non"
        );
    }

    #[test]
    fn split_and_snap_stay_short_enough_for_their_buttons() {
        assert!(say(Language::French, Phrase::Split).chars().count() <= 7);
        assert!(say(Language::French, Phrase::Snap).chars().count() <= 5);
    }
}
