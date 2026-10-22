//! The export dialog. It does not write a file. `docs/export.md`.

use eframe::egui::text::{CCursor, CCursorRange};
use eframe::egui::{
    self, Align2, Color32, FontFamily, FontId, Frame, Id, Key, Margin, Modifiers, Rect, RichText,
    ScrollArea, Sense, Stroke, TextEdit, UiBuilder, pos2, vec2,
};

use cutlass_project::FrameRate;

use crate::library::card_time;
use crate::theme;
use crate::words::{self, Language, Phrase};

const DIALOG: Color32 = Color32::from_rgb(0x32, 0x32, 0x32);
const CLOSE: Color32 = Color32::from_rgb(0xff, 0x5f, 0x57);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Resolution {
    P144,
    P240,
    P360,
    P480,
    P720,
    P1080,
    P1440,
    K4,
    K8,
}

impl Resolution {
    const ALL: [Self; 9] = [
        Self::P144,
        Self::P240,
        Self::P360,
        Self::P480,
        Self::P720,
        Self::P1080,
        Self::P1440,
        Self::K4,
        Self::K8,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::P144 => "144p",
            Self::P240 => "240p",
            Self::P360 => "360p",
            Self::P480 => "480p",
            Self::P720 => "720p",
            Self::P1080 => "1080p",
            Self::P1440 => "1440p",
            Self::K4 => "4K",
            Self::K8 => "8K",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rate {
    F24,
    F30,
    F60,
}

impl Rate {
    const ALL: [Self; 3] = [Self::F24, Self::F30, Self::F60];

    fn label(self) -> &'static str {
        match self {
            Self::F24 => "24",
            Self::F30 => "30",
            Self::F60 => "60",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Quality {
    Standard,
    High,
    Best,
}

impl Quality {
    const ALL: [Self; 3] = [Self::Standard, Self::High, Self::Best];

    fn label(self, language: Language) -> &'static str {
        words::say(
            language,
            match self {
                Self::Standard => Phrase::Standard,
                Self::High => Phrase::High,
                Self::Best => Phrase::Best,
            },
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Container {
    Mp4,
    Mov,
    Webm,
}

impl Container {
    const ALL: [Self; 3] = [Self::Mp4, Self::Mov, Self::Webm];

    fn label(self) -> &'static str {
        match self {
            Self::Mp4 => "MP4",
            Self::Mov => "MOV",
            Self::Webm => "WebM",
        }
    }

    fn video(self) -> &'static [VideoCodec] {
        match self {
            Self::Mp4 => &[VideoCodec::H264, VideoCodec::H265],
            Self::Mov => &[VideoCodec::H264, VideoCodec::H265, VideoCodec::ProRes],
            Self::Webm => &[VideoCodec::Vp9, VideoCodec::Av1],
        }
    }

    fn audio(self) -> &'static [AudioCodec] {
        match self {
            Self::Mp4 => &[AudioCodec::Aac],
            Self::Mov => &[AudioCodec::Aac, AudioCodec::Pcm],
            Self::Webm => &[AudioCodec::Opus],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VideoCodec {
    H264,
    H265,
    ProRes,
    Vp9,
    Av1,
}

impl VideoCodec {
    fn label(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::H265 => "H.265",
            Self::ProRes => "ProRes",
            Self::Vp9 => "VP9",
            Self::Av1 => "AV1",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AudioCodec {
    Aac,
    Pcm,
    Opus,
}

impl AudioCodec {
    fn label(self) -> &'static str {
        match self {
            Self::Aac => "AAC",
            Self::Pcm => "PCM",
            Self::Opus => "Opus",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RateControl {
    Vbr,
    Cbr,
}

impl RateControl {
    const ALL: [Self; 2] = [Self::Vbr, Self::Cbr];

    fn label(self) -> &'static str {
        match self {
            Self::Vbr => "VBR",
            Self::Cbr => "CBR",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Profile {
    Proxy,
    Lt,
    P422,
    Hq,
}

impl Profile {
    const ALL: [Self; 4] = [Self::Proxy, Self::Lt, Self::P422, Self::Hq];

    fn label(self) -> &'static str {
        match self {
            Self::Proxy => "Proxy",
            Self::Lt => "LT",
            Self::P422 => "422",
            Self::Hq => "HQ",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleRate {
    K441,
    K48,
}

impl SampleRate {
    const ALL: [Self; 2] = [Self::K441, Self::K48];

    fn label(self) -> &'static str {
        match self {
            Self::K441 => "44.1 kHz",
            Self::K48 => "48 kHz",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Channels {
    Stereo,
    Mono,
}

impl Channels {
    const ALL: [Self; 2] = [Self::Stereo, Self::Mono];

    fn label(self, language: Language) -> &'static str {
        words::say(
            language,
            match self {
                Self::Stereo => Phrase::Stereo,
                Self::Mono => Phrase::Mono,
            },
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AudioRate {
    K128,
    K192,
    K320,
}

impl AudioRate {
    const ALL: [Self; 3] = [Self::K128, Self::K192, Self::K320];

    fn label(self) -> &'static str {
        match self {
            Self::K128 => "128",
            Self::K192 => "192",
            Self::K320 => "320",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GifRate {
    F10,
    F15,
    F24,
}

impl GifRate {
    const ALL: [Self; 3] = [Self::F10, Self::F15, Self::F24];

    fn label(self) -> &'static str {
        match self {
            Self::F10 => "10",
            Self::F15 => "15",
            Self::F24 => "24",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GifLoop {
    Once,
    Loop,
}

impl GifLoop {
    const ALL: [Self; 2] = [Self::Once, Self::Loop];

    fn label(self, language: Language) -> &'static str {
        words::say(
            language,
            match self {
                Self::Once => Phrase::Once,
                Self::Loop => Phrase::Loop,
            },
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Caption {
    BurnIn,
    Srt,
}

impl Caption {
    const ALL: [Self; 2] = [Self::BurnIn, Self::Srt];

    fn label(self, language: Language) -> &'static str {
        match self {
            Self::BurnIn => words::say(language, Phrase::BurnIn),
            Self::Srt => "SRT",
        }
    }
}

pub(crate) struct Sheet {
    name: String,
    armed: bool,
    resolution: Resolution,
    rate: Rate,
    quality: Quality,
    format: Container,
    codec: VideoCodec,
    rate_control: RateControl,
    profile: Profile,
    bitrate: String,
    bitrate_custom: bool,
    video_on: bool,
    video_open: bool,
    audio_on: bool,
    audio_open: bool,
    audio: AudioCodec,
    sample_rate: SampleRate,
    channels: Channels,
    audio_bitrate: AudioRate,
    gif_on: bool,
    gif_open: bool,
    gif_rate: GifRate,
    gif_loop: GifLoop,
    caption_on: bool,
    caption_open: bool,
    caption: Caption,
}

impl Sheet {
    pub(crate) fn open(name: &str, rate: FrameRate) -> Self {
        let resolution = Resolution::P1080;
        let quality = Quality::High;
        Self {
            name: name.to_owned(),
            armed: false,
            resolution,
            rate: starting_rate(rate),
            quality,
            format: Container::Mp4,
            codec: VideoCodec::H264,
            rate_control: RateControl::Vbr,
            profile: Profile::P422,
            bitrate: suggested(resolution, quality).to_owned(),
            bitrate_custom: false,
            video_on: true,
            video_open: true,
            audio_on: true,
            audio_open: true,
            audio: AudioCodec::Aac,
            sample_rate: SampleRate::K48,
            channels: Channels::Stereo,
            audio_bitrate: AudioRate::K192,
            gif_on: false,
            gif_open: false,
            gif_rate: GifRate::F15,
            gif_loop: GifLoop::Loop,
            caption_on: false,
            caption_open: false,
            caption: Caption::Srt,
        }
    }

    fn set_resolution(&mut self, resolution: Resolution) {
        self.resolution = resolution;
        if !self.bitrate_custom {
            self.bitrate = suggested(resolution, self.quality).to_owned();
        }
    }

    fn set_quality(&mut self, quality: Quality) {
        self.quality = quality;
        self.bitrate = suggested(self.resolution, quality).to_owned();
        self.bitrate_custom = false;
    }

    fn set_format(&mut self, format: Container) {
        self.format = format;
        let videos = format.video();
        if !videos.contains(&self.codec)
            && let Some(codec) = videos.first().copied()
        {
            self.codec = codec;
        }
        let audios = format.audio();
        if !audios.contains(&self.audio)
            && let Some(audio) = audios.first().copied()
        {
            self.audio = audio;
        }
    }

    fn edit_bitrate(&mut self, text: &str) {
        self.bitrate = digits(text);
        self.bitrate_custom = true;
    }

    fn shows_bitrate(&self) -> bool {
        self.codec != VideoCodec::ProRes
    }

    fn shows_audio_bitrate(&self) -> bool {
        self.audio != AudioCodec::Pcm
    }
}

fn starting_rate(rate: FrameRate) -> Rate {
    match rate.nominal_fps() {
        24 => Rate::F24,
        60 => Rate::F60,
        _ => Rate::F30,
    }
}

fn suggested(resolution: Resolution, quality: Quality) -> &'static str {
    match (resolution, quality) {
        (Resolution::P144, Quality::Standard) => "0.2",
        (Resolution::P144, Quality::High) => "0.4",
        (Resolution::P144, Quality::Best) => "0.8",
        (Resolution::P240, Quality::Standard) => "0.4",
        (Resolution::P240, Quality::High) => "0.8",
        (Resolution::P240, Quality::Best) => "1.5",
        (Resolution::P360, Quality::Standard) => "0.8",
        (Resolution::P360, Quality::High) => "1.5",
        (Resolution::P360, Quality::Best) => "2.5",
        (Resolution::P480, Quality::Standard) => "1.5",
        (Resolution::P480, Quality::High) => "2.5",
        (Resolution::P480, Quality::Best) => "4",
        (Resolution::P720, Quality::Standard) => "5",
        (Resolution::P720, Quality::High) => "8",
        (Resolution::P720, Quality::Best) => "16",
        (Resolution::P1080, Quality::Standard) => "8",
        (Resolution::P1080, Quality::High) => "16",
        (Resolution::P1080, Quality::Best) => "40",
        (Resolution::P1440, Quality::Standard) => "12",
        (Resolution::P1440, Quality::High) => "24",
        (Resolution::P1440, Quality::Best) => "50",
        (Resolution::K4, Quality::Standard) => "20",
        (Resolution::K4, Quality::High) => "45",
        (Resolution::K4, Quality::Best) => "80",
        (Resolution::K8, Quality::Standard) => "50",
        (Resolution::K8, Quality::High) => "100",
        (Resolution::K8, Quality::Best) => "160",
    }
}

fn digits(text: &str) -> String {
    text.chars()
        .filter(|glyph| glyph.is_ascii_digit() || *glyph == '.')
        .collect()
}

/// Draws the dialog. `true` means it should close. Nothing is written.
pub(crate) fn show(ui: &egui::Ui, sheet: &mut Sheet, duration: &str) -> bool {
    let mut close = false;
    let screen = ui.ctx().content_rect();
    egui::Area::new(Id::unique("export-dialog"))
        .order(egui::Order::Foreground)
        .fixed_pos(screen.min)
        .movable(false)
        .constrain(false)
        .fade_in(false)
        .accessible_name(words::of(ui, Phrase::Export))
        .role(egui::accesskit::Role::Dialog)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(screen);
            ui.set_min_size(screen.size());
            let size = vec2(760.0, 820.0).min(screen.size() - vec2(48.0, 48.0));
            let size = vec2(size.x.max(360.0), size.y.max(280.0));
            let dialog = Rect::from_center_size(screen.center(), size);
            ui.painter()
                .rect_filled(screen, 0.0, Color32::from_black_alpha(140));
            let scrim = ui.interact(screen, ui.scope_id().with("scrim"), Sense::click());
            let _plate = ui.interact(dialog, ui.scope_id().with("plate"), Sense::click());
            if scrim.clicked() {
                close = true;
            }
            ui.painter().rect_filled(dialog, 12.0, DIALOG);
            let head = Rect::from_min_size(dialog.min, vec2(dialog.width(), 40.0));
            let foot = Rect::from_min_max(
                pos2(dialog.left(), dialog.bottom() - 52.0),
                dialog.right_bottom(),
            );
            let body = Rect::from_min_max(head.left_bottom(), foot.right_top());
            ui.painter().rect_filled(body, 0.0, theme::BAR);
            if paint_head(ui, head) {
                close = true;
            }
            paint_body(ui, body, sheet);
            if paint_foot(ui, foot, duration) {
                close = true;
            }
        });
    if ui.input_mut(|input| input.consume_key(Modifiers::NONE, Key::Escape)) {
        close = true;
    }
    close
}

/// `M:SS.t` for the footer. The transport keeps timecode.
pub(crate) fn duration_label(value: i64, timescale: u32) -> String {
    card_time(value, timescale)
}

fn paint_head(ui: &mut egui::Ui, head: Rect) -> bool {
    ui.painter().text(
        head.center(),
        Align2::CENTER_CENTER,
        words::of(ui, Phrase::Export),
        FontId::new(13.0, FontFamily::Proportional),
        theme::INK,
    );
    let hit = Rect::from_center_size(pos2(head.left() + 20.0, head.center().y), vec2(24.0, 24.0));
    let response = ui.interact(hit, ui.scope_id().with("close"), Sense::click());
    let (fill, _) = theme::face(CLOSE, CLOSE, theme::hot(&response));
    ui.painter().circle_filled(hit.center(), 6.0, fill);
    crate::tip::show(&response, words::of(ui, Phrase::Close));
    response.clicked()
}

fn paint_foot(ui: &mut egui::Ui, foot: Rect, duration: &str) -> bool {
    ui.painter().text(
        pos2(foot.left() + 16.0, foot.center().y),
        Align2::LEFT_CENTER,
        duration,
        FontId::new(12.0, FontFamily::Monospace),
        theme::MUTED,
    );
    let export_label = words::of(ui, Phrase::Export);
    let cancel_label = words::of(ui, Phrase::Cancel);
    let font = FontId::new(13.0, FontFamily::Proportional);
    let export_w = words::control_width(ui, export_label, font.clone(), 24.0, 72.0);
    let cancel_w = words::control_width(ui, cancel_label, font, 20.0, 64.0);
    let export = Rect::from_center_size(
        pos2(foot.right() - 16.0 - export_w / 2.0, foot.center().y),
        vec2(export_w, 26.0),
    );
    let cancel = Rect::from_center_size(
        pos2(export.left() - 8.0 - cancel_w / 2.0, foot.center().y),
        vec2(cancel_w, 26.0),
    );
    let duration_rect =
        Rect::from_min_max(foot.left_top(), pos2(cancel.left() - 8.0, foot.bottom()));
    if duration_rect.is_positive() {
        crate::tip::read(
            ui,
            duration_rect,
            ui.scope_id().with("export-duration"),
            duration,
        );
    }
    let cancel_response = ui.interact(cancel, ui.scope_id().with("cancel"), Sense::click());
    crate::tip::name(&cancel_response, cancel_label);
    let (cancel_fill, cancel_ink) = theme::face(
        Color32::TRANSPARENT,
        theme::MUTED,
        theme::hot(&cancel_response),
    );
    if cancel_fill.a() > 0 {
        ui.painter().rect_filled(cancel, 6.0, cancel_fill);
    }
    ui.painter().text(
        cancel.center(),
        Align2::CENTER_CENTER,
        cancel_label,
        FontId::new(13.0, FontFamily::Proportional),
        cancel_ink,
    );
    let export_response = ui.interact(export, ui.scope_id().with("export"), Sense::click());
    crate::tip::name(&export_response, export_label);
    let (export_fill, export_ink) = theme::face(
        theme::EXPORT,
        theme::EXPORT_INK,
        theme::hot(&export_response),
    );
    ui.painter().rect_filled(export, 8.0, export_fill);
    ui.painter().text(
        export.center(),
        Align2::CENTER_CENTER,
        export_label,
        FontId::new(13.0, FontFamily::Proportional),
        export_ink,
    );
    cancel_response.clicked() || export_response.clicked()
}

fn paint_body(ui: &mut egui::Ui, body: Rect, sheet: &mut Sheet) {
    let inner = Rect::from_min_max(body.min + vec2(32.0, 28.0), body.max - vec2(32.0, 20.0));
    if inner.width() < 80.0 || inner.height() < 40.0 {
        return;
    }
    let cover_w = (inner.width() - 36.0 - 220.0).clamp(120.0, 280.0);
    let cover = Rect::from_min_size(inner.min, vec2(cover_w, cover_w * 16.0 / 9.0));
    let settings = Rect::from_min_max(pos2(cover.right() + 36.0, inner.top()), inner.max);
    let clip = ui.clip_rect();
    ui.set_clip_rect(body);
    ui.painter().rect_filled(cover, 12.0, theme::VIDEO);
    let cover_name = if sheet.name.trim().is_empty() {
        "Untitled"
    } else {
        sheet.name.trim()
    };
    ui.painter().text(
        cover.left_bottom() + vec2(18.0, -18.0),
        Align2::LEFT_BOTTOM,
        cover_name,
        FontId::new(22.0, FontFamily::Proportional),
        Color32::WHITE,
    );
    let mut column = ui.new_child(UiBuilder::new().max_rect(settings));
    column.set_clip_rect(settings.intersect(body));
    ScrollArea::vertical()
        .id_salt("export-settings")
        .show(&mut column, |ui| {
            ui.set_min_width(ui.available_width());
            name_row(ui, sheet);
            ui.add_space(16.0);
            video_section(ui, sheet);
            ui.add_space(28.0);
            audio_section(ui, sheet);
            ui.add_space(28.0);
            gif_section(ui, sheet);
            ui.add_space(28.0);
            caption_section(ui, sheet);
        });
    ui.set_clip_rect(clip);
}

fn name_row(ui: &mut egui::Ui, sheet: &mut Sheet) {
    labeled(ui, words::of(ui, Phrase::Name), |ui, width| {
        let output = TextEdit::singleline(&mut sheet.name)
            .id(Id::unique("export-name"))
            .frame(field_frame())
            .desired_width(width)
            .show(ui);
        if !sheet.armed {
            output.response.request_focus();
            if let Some(mut state) = TextEdit::load_state(ui.ctx(), output.response.id) {
                let end = CCursor::new(sheet.name.chars().count());
                state
                    .cursor
                    .set_char_range(Some(CCursorRange::two(CCursor::new(0), end)));
                TextEdit::store_state(ui.ctx(), output.response.id, state);
            }
            sheet.armed = true;
        }
    });
}

fn video_section(ui: &mut egui::Ui, sheet: &mut Sheet) {
    let language = words::active(ui);
    section(
        ui,
        words::of(ui, Phrase::Video),
        &mut sheet.video_open,
        &mut sheet.video_on,
    );
    if !sheet.video_open {
        return;
    }
    let mut resolution = sheet.resolution;
    choice(
        ui,
        "resolution",
        words::of(ui, Phrase::Resolution),
        &mut resolution,
        &Resolution::ALL,
        Resolution::label,
    );
    if resolution != sheet.resolution {
        sheet.set_resolution(resolution);
    }
    choice(
        ui,
        "rate",
        words::of(ui, Phrase::FrameRate),
        &mut sheet.rate,
        &Rate::ALL,
        Rate::label,
    );
    let mut format = sheet.format;
    choice(
        ui,
        "format",
        words::of(ui, Phrase::Format),
        &mut format,
        &Container::ALL,
        Container::label,
    );
    if format != sheet.format {
        sheet.set_format(format);
    }
    let videos = sheet.format.video();
    choice(
        ui,
        "codec",
        words::of(ui, Phrase::Codec),
        &mut sheet.codec,
        videos,
        VideoCodec::label,
    );
    if sheet.shows_bitrate() {
        let mut quality = sheet.quality;
        choice(
            ui,
            "quality",
            words::of(ui, Phrase::Quality),
            &mut quality,
            &Quality::ALL,
            |item| Quality::label(item, language),
        );
        if quality != sheet.quality {
            sheet.set_quality(quality);
        }
        choice(
            ui,
            "rate-control",
            words::of(ui, Phrase::RateControl),
            &mut sheet.rate_control,
            &RateControl::ALL,
            RateControl::label,
        );
        bitrate_row(ui, sheet);
    } else {
        choice(
            ui,
            "profile",
            "ProRes",
            &mut sheet.profile,
            &Profile::ALL,
            Profile::label,
        );
    }
}

fn audio_section(ui: &mut egui::Ui, sheet: &mut Sheet) {
    let language = words::active(ui);
    section(
        ui,
        words::of(ui, Phrase::Audio),
        &mut sheet.audio_open,
        &mut sheet.audio_on,
    );
    if !sheet.audio_open {
        return;
    }
    let audios = sheet.format.audio();
    choice(
        ui,
        "audio-codec",
        words::of(ui, Phrase::Codec),
        &mut sheet.audio,
        audios,
        AudioCodec::label,
    );
    choice(
        ui,
        "sample-rate",
        words::of(ui, Phrase::SampleRate),
        &mut sheet.sample_rate,
        &SampleRate::ALL,
        SampleRate::label,
    );
    choice(
        ui,
        "channels",
        words::of(ui, Phrase::Channels),
        &mut sheet.channels,
        &Channels::ALL,
        |item| Channels::label(item, language),
    );
    if sheet.shows_audio_bitrate() {
        choice(
            ui,
            "audio-rate",
            "kbps",
            &mut sheet.audio_bitrate,
            &AudioRate::ALL,
            AudioRate::label,
        );
    }
}

fn gif_section(ui: &mut egui::Ui, sheet: &mut Sheet) {
    let language = words::active(ui);
    section(
        ui,
        words::of(ui, Phrase::Gif),
        &mut sheet.gif_open,
        &mut sheet.gif_on,
    );
    if !sheet.gif_open {
        return;
    }
    choice(
        ui,
        "gif-rate",
        words::of(ui, Phrase::FrameRate),
        &mut sheet.gif_rate,
        &GifRate::ALL,
        GifRate::label,
    );
    choice(
        ui,
        "gif-loop",
        words::of(ui, Phrase::Loop),
        &mut sheet.gif_loop,
        &GifLoop::ALL,
        |item| GifLoop::label(item, language),
    );
}

fn caption_section(ui: &mut egui::Ui, sheet: &mut Sheet) {
    section(
        ui,
        words::of(ui, Phrase::Caption),
        &mut sheet.caption_open,
        &mut sheet.caption_on,
    );
    if !sheet.caption_open {
        return;
    }
    let language = words::active(ui);
    choice(
        ui,
        "caption",
        words::of(ui, Phrase::File),
        &mut sheet.caption,
        &Caption::ALL,
        |item| Caption::label(item, language),
    );
}

fn bitrate_row(ui: &mut egui::Ui, sheet: &mut Sheet) {
    let mut text = sheet.bitrate.clone();
    labeled(ui, words::of(ui, Phrase::Bitrate), |ui, width| {
        ui.horizontal(|ui| {
            let output = TextEdit::singleline(&mut text)
                .id(ui.scope_id().with("bitrate"))
                .frame(field_frame())
                .desired_width((width - 48.0).max(40.0))
                .show(ui);
            ui.label(RichText::new("Mbps").size(12.0).color(theme::FAINT));
            if output.response.changed() {
                sheet.edit_bitrate(&text);
            }
        });
    });
}

fn field_frame() -> Frame {
    Frame::NONE
        .fill(theme::PANEL)
        .corner_radius(10)
        .inner_margin(Margin::symmetric(12, 6))
}

fn section(ui: &mut egui::Ui, title: &str, open: &mut bool, enabled: &mut bool) {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(width, 28.0), Sense::hover());
    let fold = Rect::from_min_max(rect.min, pos2(rect.right() - 40.0, rect.bottom()));
    let fold_response = ui.interact(fold, ui.scope_id().with((title, "fold")), Sense::click());
    if fold_response.clicked() {
        *open = !*open;
    }
    crate::tip::disclosure(&fold_response, title, *open);
    chevron(
        ui.painter(),
        pos2(rect.left() + 4.0, rect.center().y),
        *open,
    );
    ui.painter().text(
        pos2(rect.left() + 16.0, rect.center().y),
        Align2::LEFT_CENTER,
        title,
        FontId::new(12.0, FontFamily::Proportional),
        theme::INK,
    );
    let switch =
        Rect::from_center_size(pos2(rect.right() - 16.0, rect.center().y), vec2(32.0, 18.0));
    let switch_response = ui.interact(
        switch,
        ui.scope_id().with((title, "switch")),
        Sense::click(),
    );
    if switch_response.clicked() {
        *enabled = !*enabled;
    }
    let fill = if *enabled {
        theme::EXPORT
    } else {
        theme::PANEL
    };
    ui.painter().rect_filled(switch, 99.0, fill);
    let thumb = if *enabled {
        switch.right() - 9.0
    } else {
        switch.left() + 9.0
    };
    ui.painter()
        .circle_filled(pos2(thumb, switch.center().y), 7.0, Color32::WHITE);
    let tip = words::named_switch(words::active(ui), title, *enabled);
    crate::tip::show_switch(&switch_response, &tip, *enabled, true);
}

fn chevron(painter: &egui::Painter, center: egui::Pos2, open: bool) {
    let stroke = Stroke::new(1.5, theme::MUTED);
    if open {
        painter.line_segment([center + vec2(-4.0, -1.0), center + vec2(0.0, 3.0)], stroke);
        painter.line_segment([center + vec2(0.0, 3.0), center + vec2(4.0, -1.0)], stroke);
    } else {
        painter.line_segment([center + vec2(-2.0, -4.0), center + vec2(2.0, 0.0)], stroke);
        painter.line_segment([center + vec2(2.0, 0.0), center + vec2(-2.0, 4.0)], stroke);
    }
}

fn labeled(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui, f32)) {
    let width = ui.available_width();
    let field = (width - 124.0).max(80.0);
    ui.horizontal(|ui| {
        ui.set_min_width(width);
        ui.spacing_mut().item_spacing.x = 16.0;
        let name = ui.add_sized(
            vec2(108.0, 32.0),
            egui::Label::new(RichText::new(label).size(12.0).color(theme::MUTED)),
        );
        ui.label_inputs_by(name.id, |ui| add(ui, field));
    });
    ui.add_space(8.0);
}

fn choice<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    salt: &str,
    label: &str,
    current: &mut T,
    options: &[T],
    text: impl Fn(T) -> &'static str,
) {
    labeled(ui, label, |ui, width| {
        egui::ComboBox::from_id_salt((salt, "export"))
            .selected_text(text(*current))
            .width(width)
            .show_ui(ui, |ui| {
                for option in options {
                    ui.selectable_value(current, *option, text(*option));
                }
            });
    });
}

#[cfg(test)]
mod tests {
    use super::{
        AudioCodec, Container, Quality, Rate, Resolution, Sheet, VideoCodec, digits, suggested,
    };
    use cutlass_project::FrameRate;

    #[test]
    fn a_new_dialog_starts_from_the_project_and_folds_gif_and_caption() {
        let sheet = Sheet::open("Vows", FrameRate::Fps30);
        assert_eq!(sheet.name, "Vows");
        assert_eq!(sheet.resolution, Resolution::P1080);
        assert_eq!(sheet.rate, Rate::F30);
        assert_eq!(sheet.quality, Quality::High);
        assert_eq!(sheet.format, Container::Mp4);
        assert_eq!(sheet.codec, VideoCodec::H264);
        assert_eq!(sheet.bitrate, "16");
        assert!(sheet.video_on && sheet.video_open);
        assert!(sheet.audio_on && sheet.audio_open);
        assert!(!sheet.gif_on && !sheet.gif_open);
        assert!(!sheet.caption_on && !sheet.caption_open);
        assert_eq!(Sheet::open("Cut", FrameRate::Fps24).rate, Rate::F24);
        assert_eq!(Sheet::open("Cut", FrameRate::Fps23976).rate, Rate::F24);
        assert_eq!(Sheet::open("Cut", FrameRate::Fps60).rate, Rate::F60);
        assert_eq!(Sheet::open("Cut", FrameRate::Fps5994).rate, Rate::F60);
        assert_eq!(Sheet::open("Cut", FrameRate::Fps25).rate, Rate::F30);
        assert_eq!(Sheet::open("Cut", FrameRate::Fps50).rate, Rate::F30);
    }

    #[test]
    fn format_chooses_the_codec_and_prores_drops_the_bitrate() {
        let mut sheet = Sheet::open("Vows", FrameRate::Fps30);
        sheet.set_format(Container::Webm);
        assert_eq!(sheet.codec, VideoCodec::Vp9);
        assert_eq!(sheet.audio, AudioCodec::Opus);
        assert!(sheet.shows_bitrate());
        sheet.set_format(Container::Mov);
        assert_eq!(sheet.codec, VideoCodec::H264);
        assert_eq!(sheet.audio, AudioCodec::Aac);
        assert!(Container::Mov.video().contains(&VideoCodec::ProRes));
        assert!(!Container::Mp4.video().contains(&VideoCodec::ProRes));
        sheet.codec = VideoCodec::ProRes;
        assert!(!sheet.shows_bitrate());
        sheet.audio = AudioCodec::Pcm;
        assert!(!sheet.shows_audio_bitrate());
    }

    #[test]
    fn quality_replaces_a_typed_bitrate_and_resolution_keeps_it() {
        let mut sheet = Sheet::open("Vows", FrameRate::Fps30);
        assert_eq!(suggested(Resolution::P1080, Quality::High), "16");
        assert_eq!(suggested(Resolution::K8, Quality::Best), "160");
        assert_eq!(suggested(Resolution::P144, Quality::Standard), "0.2");
        sheet.set_resolution(Resolution::P720);
        assert_eq!(sheet.bitrate, "8");
        sheet.edit_bitrate("12 Mbps");
        assert_eq!(sheet.bitrate, "12");
        assert_eq!(digits("1.5.2"), "1.5.2");
        sheet.set_resolution(Resolution::P1080);
        assert_eq!(sheet.bitrate, "12");
        sheet.set_quality(Quality::Best);
        assert_eq!(sheet.bitrate, "40");
        assert!(!sheet.bitrate_custom);
    }
}
