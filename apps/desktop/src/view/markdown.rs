//! The assistant's answers, painted from their markdown: paragraphs,
//! headings, lists, quotes, code and tables. Every word can be selected
//! and copied; a web link opens in the browser; a code block has a copy
//! button.
//!
//! Answers are set in Inter and code in JetBrains Mono, two faces the
//! program already ships for titles. Both vary in weight, so strong words
//! and headings are truly bold rather than only brighter.

use std::ops::Range;
use std::sync::Arc;

use cutlass_agent::markdown::{self, Block, Item, Span, Style};
use egui::epaint::text::VariationCoords;
use egui::text::{CCursor, LayoutJob, TextFormat};
use egui::{
    Align, Color32, Context, CursorIcon, FontData, FontDefinitions, FontFamily, FontId, Frame,
    Galley, Id, Label, Layout, Margin, OpenUrl, Sense, Stroke, Ui, Vec2, vec2,
};

use crate::{tokens, ui};

/// The size of an answer's words.
const BODY: f32 = 13.0;
/// The size of code, a point under the words around it.
const CODE: f32 = 12.0;
/// The space between two blocks.
const BLOCK_GAP: f32 = 6.0;
/// How far a quote's and a list item's blocks sit in.
const INDENT: f32 = 14.0;
const LINK: Color32 = tokens::EXPORT_HOVER;
/// Strong words are white, the brightest the panel draws.
const STRONG: Color32 = Color32::WHITE;
/// A code block's ground, darker than the panel.
const CODE_GROUND: Color32 = tokens::GAP;
/// Inline code's ground, lighter than the panel.
const CODE_INLINE: Color32 = tokens::RAISED;

/// The family answers are set in: Inter, then the window's own faces for
/// what Inter lacks.
const WORDS_FAMILY: &str = "assistant";
/// The family code is set in: JetBrains Mono, then the window's monospace.
const CODE_FAMILY: &str = "assistant-code";
/// The weights words are drawn at, on the faces' `wght` axis.
const REGULAR: f32 = 400.0;
const BOLD: f32 = 620.0;
const HEADING: f32 = 680.0;

/// Adds the answer and code families to the window's fonts, once. Until
/// they are in, answers are set in the window's own faces.
pub fn install_fonts(ctx: &Context) {
    let wanted = FontFamily::Name(WORDS_FAMILY.into());
    if ctx.fonts(|fonts| fonts.definitions().families.contains_key(&wanted)) {
        return;
    }
    let mut definitions = FontDefinitions::default();
    for (family, face, under) in [
        (WORDS_FAMILY, "Inter", FontFamily::Proportional),
        (CODE_FAMILY, "JetBrains Mono", FontFamily::Monospace),
    ] {
        let Some(bytes) = shipped(face) else { continue };
        definitions
            .font_data
            .insert(face.to_owned(), Arc::new(FontData::from_static(bytes)));
        let mut faces = vec![face.to_owned()];
        faces.extend(
            definitions
                .families
                .get(&under)
                .cloned()
                .unwrap_or_default(),
        );
        definitions
            .families
            .insert(FontFamily::Name(family.into()), faces);
    }
    ctx.set_fonts(definitions);
}

/// The bytes of the shipped family `name`, in one file that varies.
fn shipped(name: &str) -> Option<&'static [u8]> {
    cutlass_fonts::FAMILIES
        .iter()
        .find(|family| family.name == name)
        .and_then(|family| family.files.first())
        .map(|(_, bytes)| *bytes)
}

/// `family` when the window has it, or the window's own `instead`.
fn family(ctx: &Context, family: &str, instead: FontFamily) -> FontFamily {
    let named = FontFamily::Name(family.into());
    if ctx.fonts(|fonts| fonts.definitions().families.contains_key(&named)) {
        named
    } else {
        instead
    }
}

/// `text`, the model's markdown, as the panel shows an answer. `id`
/// tells this answer's tables apart from another's.
pub fn answer(ui: &mut Ui, text: &str, id: Id) {
    let blocks = parsed(ui, text);
    stack(ui, &blocks, id, tokens::TEXT);
}

/// The blocks of `text`, read once while the answer is on screen.
fn parsed(ui: &Ui, text: &str) -> Arc<Vec<Block>> {
    ui.memory_mut(|memory| memory.caches.cache::<Parsed>().get(text).clone())
}

type Parsed = egui::cache::FrameCache<Arc<Vec<Block>>, Parse>;

#[derive(Default)]
struct Parse;

impl egui::cache::ComputerMut<&str, Arc<Vec<Block>>> for Parse {
    fn compute(&mut self, text: &str) -> Arc<Vec<Block>> {
        #[cfg(test)]
        PARSES.with(|count| count.set(count.get() + 1));
        Arc::new(markdown::blocks(text))
    }
}

#[cfg(test)]
thread_local! {
    /// How many answers this thread read, so a test can see the cache.
    static PARSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// `blocks` one under another, words in `color`.
fn stack(ui: &mut Ui, blocks: &[Block], id: Id, color: Color32) {
    for (index, block) in blocks.iter().enumerate() {
        if index > 0 {
            ui.add_space(BLOCK_GAP);
        }
        paint(ui, block, id.with(index), color);
    }
}

fn paint(ui: &mut Ui, block: &Block, id: Id, color: Color32) {
    match block {
        Block::Paragraph(spans) => words(ui, spans, BODY, color, REGULAR, true),
        Block::Heading { level, spans } => {
            let size = match level {
                1 => 18.0,
                2 => 16.0,
                3 => 14.5,
                _ => BODY,
            };
            words(ui, spans, size, STRONG, HEADING, true);
        }
        Block::Code { language, text } => code_block(ui, language, text),
        Block::Quote(inner) => quote(ui, inner, id),
        Block::List { first, items } => list(ui, *first, items, id, color),
        Block::Table { head, rows } => table(ui, head, rows, id, color),
        Block::Rule => {
            ui.separator();
        }
    }
}

/// The families an answer is set in: its words' and its code's.
#[derive(Clone)]
struct Faces {
    words: FontFamily,
    code: FontFamily,
}

impl Faces {
    fn of(ctx: &Context) -> Self {
        Self {
            words: family(ctx, WORDS_FAMILY, FontFamily::Proportional),
            code: family(ctx, CODE_FAMILY, FontFamily::Monospace),
        }
    }
}

/// The weight `weight` on the `wght` axis.
fn weight(weight: f32) -> VariationCoords {
    VariationCoords::new([(b"wght", weight)])
}

/// How `style` is drawn at `size`, the words otherwise `color`, and all
/// at least `least` heavy.
fn format(style: &Style, faces: &Faces, size: f32, color: Color32, least: f32) -> TextFormat {
    let mut format = TextFormat {
        font_id: FontId::new(size, faces.words.clone()),
        color,
        coords: weight(least),
        ..TextFormat::default()
    };
    if style.code {
        format.font_id = FontId::new(size - (BODY - CODE), faces.code.clone());
        format.background = CODE_INLINE;
    }
    if style.strong {
        format.coords = weight(least.max(BOLD));
    }
    if style.strong || least > REGULAR {
        format.color = STRONG;
    }
    format.italics = style.emphasis;
    if style.struck {
        format.strikethrough = Stroke::new(1.0, format.color);
    }
    if style.link.is_some() {
        format.color = LINK;
        format.underline = Stroke::new(1.0, LINK);
    }
    format
}

/// `spans` as one job, and where its links are, by character.
fn job(
    spans: &[Span],
    faces: &Faces,
    size: f32,
    color: Color32,
    least: f32,
) -> (LayoutJob, Vec<(Range<usize>, String)>) {
    let mut job = LayoutJob::default();
    let mut links = Vec::new();
    let mut chars = 0;
    for span in spans {
        let count = span.text.chars().count();
        if let Some(url) = &span.style.link {
            links.push((chars..chars + count, url.clone()));
        }
        chars += count;
        job.append(
            &span.text,
            0.0,
            format(&span.style, faces, size, color, least),
        );
    }
    (job, links)
}

/// A run of words that can be selected, at least `least` heavy, wrapped to
/// the width there is when `wrap`, and whose links open when clicked.
fn words(ui: &mut Ui, spans: &[Span], size: f32, color: Color32, least: f32, wrap: bool) {
    let faces = Faces::of(ui.ctx());
    let (mut job, links) = job(spans, &faces, size, color, least);
    job.wrap.max_width = if wrap {
        ui.available_width()
    } else {
        f32::INFINITY
    };
    let galley = ui.ctx().fonts_mut(|fonts| fonts.layout_job(job));
    let mut label = Label::new(galley.clone()).selectable(true);
    if !links.is_empty() {
        label = label.sense(Sense::click());
    }
    let response = ui.add(label);
    let Some(pointer) = response.hover_pos() else {
        return;
    };
    let Some(url) = link_at(&galley, pointer - response.rect.min, &links) else {
        return;
    };
    ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
    if response.clicked() {
        ui.ctx().open_url(OpenUrl::new_tab(url));
    }
    response.on_hover_text_at_pointer(url);
}

/// The link under `pos`, a point from the galley's corner.
fn link_at<'a>(galley: &Galley, pos: Vec2, links: &'a [(Range<usize>, String)]) -> Option<&'a str> {
    if links.is_empty() || !galley.rect.contains(pos.to_pos2()) {
        return None;
    }
    // The cursor falls between two characters: the one under the pointer
    // is the one after the cursor when the pointer is past its edge.
    let cursor = galley.cursor_from_pos(pos).index;
    let edge = galley.pos_from_cursor(CCursor::new(cursor)).min.x;
    let index = if pos.x < edge {
        cursor.0.checked_sub(1)?
    } else {
        cursor.0
    };
    links
        .iter()
        .find(|(range, _)| range.contains(&index))
        .map(|(_, url)| url.as_str())
}

/// Code on a darker ground, with its language and a button that copies it.
fn code_block(ui: &mut Ui, language: &str, text: &str) {
    Frame::new()
        .fill(CODE_GROUND)
        .corner_radius(tokens::RADIUS_SM)
        .inner_margin(Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(language)
                        .size(10.5)
                        .color(tokens::FAINT),
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui
                        .add(ui::Button::new("Copy").text_size(11.0))
                        .on_hover_text("Copy the code")
                        .clicked()
                    {
                        ui.ctx().copy_text(text.to_owned());
                    }
                });
            });
            let code = Faces::of(ui.ctx()).code;
            ui.add(
                Label::new(
                    egui::RichText::new(text)
                        .font(FontId::new(CODE, code))
                        .color(tokens::TEXT),
                )
                .selectable(true),
            );
        });
}

/// Quoted blocks, muted, with a bar down their left.
fn quote(ui: &mut Ui, blocks: &[Block], id: Id) {
    let shown = Frame::new()
        .inner_margin(Margin {
            left: INDENT as i8,
            ..Margin::ZERO
        })
        .show(ui, |ui| stack(ui, blocks, id, tokens::MUTED));
    let rect = shown.response.rect;
    ui.painter().line_segment(
        [
            rect.left_top() + vec2(3.0, 1.0),
            rect.left_bottom() + vec2(3.0, -1.0),
        ],
        Stroke::new(2.0, tokens::RULER_TICK),
    );
}

/// The marker before item `index` of a list counting from `first`.
fn marker(first: Option<u64>, index: usize) -> String {
    match first {
        Some(first) => format!("{}.", first.saturating_add(index as u64)),
        None => "•".to_owned(),
    }
}

fn list(ui: &mut Ui, first: Option<u64>, items: &[Item], id: Id, color: Color32) {
    // Numbers line up on their dots: the widest marker sets the column.
    let font = FontId::new(BODY, Faces::of(ui.ctx()).words);
    let width = (0..items.len())
        .map(|index| {
            let text = marker(first, index);
            ui.ctx()
                .fonts_mut(|fonts| fonts.layout_no_wrap(text, font.clone(), color))
                .size()
                .x
        })
        .fold(INDENT + 2.0, f32::max)
        + 4.0;
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            ui.add_space(BLOCK_GAP / 2.0);
        }
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            let line = ui.ctx().fonts_mut(|fonts| fonts.row_height(&font));
            let (rect, _) = ui.allocate_exact_size(vec2(width, line), Sense::hover());
            match item.task {
                Some(ticked) => task_box(ui, rect, ticked),
                None => {
                    ui.painter().text(
                        rect.right_center() - vec2(4.0, 0.0),
                        egui::Align2::RIGHT_CENTER,
                        marker(first, index),
                        font.clone(),
                        tokens::MUTED,
                    );
                }
            }
            ui.vertical(|ui| stack(ui, &item.blocks, id.with(index), color));
        });
    }
}

/// A task's box, ticked or not, at the right of `rect`.
fn task_box(ui: &Ui, rect: egui::Rect, ticked: bool) {
    let side = 11.0;
    let square = egui::Rect::from_center_size(
        rect.right_center() - vec2(4.0 + side / 2.0, 0.0),
        Vec2::splat(side),
    );
    let painter = ui.painter();
    if ticked {
        painter.rect_filled(square, 2.0, LINK);
        let ink = Stroke::new(1.5, tokens::EXPORT_INK);
        let left = square.left_center() + vec2(2.5, 0.0);
        let bottom = square.center_bottom() - vec2(1.0, 3.0);
        let right = square.right_top() + vec2(-2.5, 3.0);
        painter.line_segment([left, bottom], ink);
        painter.line_segment([bottom, right], ink);
    } else {
        painter.rect_stroke(
            square,
            2.0,
            Stroke::new(1.0, tokens::MUTED),
            egui::StrokeKind::Inside,
        );
    }
}

/// A table in a grid that scrolls sideways when it is wider than the panel.
fn table(ui: &mut Ui, head: &[Vec<Span>], rows: &[Vec<Vec<Span>>], id: Id, color: Color32) {
    ui::Scroll::horizontal(id.with("scroll")).show(ui, |ui| {
        Frame::new()
            .stroke(Stroke::new(1.0, tokens::LANE_LINE))
            .corner_radius(tokens::RADIUS_SM)
            .inner_margin(Margin::symmetric(8, 6))
            .show(ui, |ui| {
                egui::Grid::new(id.with("grid"))
                    .striped(true)
                    .spacing([16.0, 6.0])
                    .show(ui, |ui| {
                        for cell in head {
                            words(ui, cell, BODY, color, BOLD, false);
                        }
                        ui.end_row();
                        for row in rows {
                            for cell in row {
                                words(ui, cell, BODY, color, REGULAR, false);
                            }
                            ui.end_row();
                        }
                    });
            });
    });
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use egui::text::CCursor;
    use egui::{
        Context, Event, FullOutput, Galley, Modifiers, OutputCommand, Pos2, RawInput, Rect, Shape,
        Vec2,
    };

    use super::{PARSES, answer};

    /// One frame of `text` as an answer, after `events`.
    fn frame(ctx: &Context, text: &str, events: Vec<Event>) -> FullOutput {
        let mut output = ctx.run_ui(
            RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(360.0, 600.0))),
                events,
                ..Default::default()
            },
            |ui| {
                super::install_fonts(ui.ctx());
                answer(ui, text, ui.scope_id().with("answer"));
            },
        );
        output.textures_delta.clear();
        output
    }

    fn texts(output: &FullOutput) -> Vec<(Pos2, Arc<Galley>)> {
        fn walk(shape: &Shape, out: &mut Vec<(Pos2, Arc<Galley>)>) {
            match shape {
                Shape::Text(text) => out.push((text.pos, text.galley.clone())),
                Shape::Vec(shapes) => shapes.iter().for_each(|shape| walk(shape, out)),
                _ => {}
            }
        }
        let mut out = Vec::new();
        for clipped in &output.shapes {
            walk(&clipped.shape, &mut out);
        }
        out
    }

    fn lines(output: &FullOutput) -> Vec<String> {
        texts(output)
            .into_iter()
            .map(|(_, galley)| galley.text().to_owned())
            .collect()
    }

    /// Where character `index` of the drawn `words` is on screen.
    fn char_pos(output: &FullOutput, words: &str, index: usize) -> Pos2 {
        let (pos, galley) = texts(output)
            .into_iter()
            .find(|(_, galley)| galley.text() == words)
            .unwrap_or_else(|| panic!("{words} is not drawn: {:?}", lines(output)));
        let left = galley.pos_from_cursor(CCursor::new(index));
        let right = galley.pos_from_cursor(CCursor::new(index + 1));
        pos + Vec2::new((left.min.x + right.min.x) / 2.0, left.center().y)
    }

    fn press(pos: Pos2, pressed: bool) -> Event {
        Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Modifiers::default(),
        }
    }

    /// What a click at `at` asks of the platform.
    fn click(ctx: &Context, text: &str, at: Pos2) -> Vec<OutputCommand> {
        let _ = frame(ctx, text, vec![Event::PointerMoved(at)]);
        let _ = frame(ctx, text, vec![press(at, true)]);
        frame(ctx, text, vec![press(at, false)])
            .platform_output
            .commands
    }

    fn opened(commands: &[OutputCommand]) -> Vec<String> {
        commands
            .iter()
            .filter_map(|command| match command {
                OutputCommand::OpenUrl(open) => Some(open.url.clone()),
                _ => None,
            })
            .collect()
    }

    fn copied(commands: &[OutputCommand]) -> Vec<String> {
        commands
            .iter()
            .filter_map(|command| match command {
                OutputCommand::CopyText(text) => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    const ANSWER: &str = "# Plan\n\nCut **here** and `split`.\n\n- one\n- two\n\n3. three\n\n> quoted\n\n```rust\nlet a = 1;\n```\n\n| Clip | Length |\n|---|---|\n| a | 2 s |";

    #[test]
    fn an_answer_is_drawn_without_its_markup() {
        let ctx = Context::default();
        let lines = lines(&frame(&ctx, ANSWER, Vec::new()));
        for shown in [
            "Plan",
            "Cut here and split.",
            "•",
            "one",
            "two",
            "3.",
            "three",
            "quoted",
            "rust",
            "Copy",
            "let a = 1;",
            "Clip",
            "Length",
            "a",
            "2 s",
        ] {
            assert!(lines.iter().any(|line| line == shown), "{shown}: {lines:?}");
        }
        for line in &lines {
            assert!(
                !line.contains("**")
                    && !line.contains('`')
                    && !line.contains('#')
                    && !line.contains('|'),
                "{line}"
            );
        }
    }

    #[test]
    fn a_web_link_opens_when_clicked_and_plain_words_do_not() {
        let text = "See [the docs](https://example.com/docs) now.";
        let ctx = Context::default();
        let output = frame(&ctx, text, Vec::new());
        let link = char_pos(&output, "See the docs now.", 6);
        let plain = char_pos(&output, "See the docs now.", 1);
        assert_eq!(
            opened(&click(&ctx, text, link)),
            ["https://example.com/docs"]
        );
        assert!(opened(&click(&ctx, text, plain)).is_empty());
    }

    #[test]
    fn a_link_to_anything_but_the_web_or_mail_does_not_open() {
        for text in ["[run](javascript:alert(1))", "[file](file:///etc/passwd)"] {
            let ctx = Context::default();
            let output = frame(&ctx, text, Vec::new());
            let words = lines(&output).remove(0);
            let at = char_pos(&output, &words, 1);
            assert!(opened(&click(&ctx, text, at)).is_empty(), "{text}");
        }
    }

    #[test]
    fn the_copy_button_copies_the_code_alone() {
        let ctx = Context::default();
        let output = frame(&ctx, ANSWER, Vec::new());
        let at = char_pos(&output, "Copy", 1);
        assert_eq!(copied(&click(&ctx, ANSWER, at)), ["let a = 1;"]);
    }

    #[test]
    fn words_can_be_selected_across_blocks() {
        let text = "first part\n\n- second part";
        let ctx = Context::default();
        let output = frame(&ctx, text, Vec::new());
        let start = char_pos(&output, "first part", 0) - Vec2::new(2.0, 0.0);
        let end = char_pos(&output, "second part", 10) + Vec2::new(2.0, 0.0);
        let _ = frame(&ctx, text, vec![Event::PointerMoved(start)]);
        let _ = frame(&ctx, text, vec![press(start, true)]);
        let _ = frame(&ctx, text, vec![Event::PointerMoved(end)]);
        let _ = frame(&ctx, text, vec![press(end, false)]);
        let copied = copied(
            &frame(&ctx, text, vec![Event::Copy])
                .platform_output
                .commands,
        );
        assert_eq!(copied.len(), 1, "{copied:?}");
        assert!(copied[0].starts_with("first part"), "{copied:?}");
        assert!(copied[0].ends_with("second part"), "{copied:?}");
    }

    /// The format of the drawn `words`' section that starts at `at`.
    fn section(output: &FullOutput, words: &str, at: usize) -> egui::text::TextFormat {
        let (_, galley) = texts(output)
            .into_iter()
            .find(|(_, galley)| galley.text() == words)
            .unwrap_or_else(|| panic!("{words} is not drawn"));
        galley
            .job
            .sections
            .iter()
            .find(|section| section.byte_range.start.0 == at)
            .map(|section| section.format.clone())
            .unwrap_or_else(|| panic!("no section at {at}"))
    }

    #[test]
    fn strong_words_and_headings_are_set_heavier_in_the_answer_faces() {
        use egui::FontFamily;

        let ctx = Context::default();
        let _ = frame(&ctx, ANSWER, Vec::new());
        // The faces are in from the second frame of a bare context.
        let output = frame(&ctx, ANSWER, Vec::new());
        let words = FontFamily::Name(super::WORDS_FAMILY.into());
        let code = FontFamily::Name(super::CODE_FAMILY.into());
        let plain = section(&output, "Cut here and split.", 0);
        let strong = section(&output, "Cut here and split.", "Cut ".len());
        let inline = section(&output, "Cut here and split.", "Cut here and ".len());
        let heading = section(&output, "Plan", 0);
        assert_eq!(plain.font_id.family, words);
        assert_eq!(plain.coords, super::weight(super::REGULAR));
        assert_eq!(strong.font_id.family, words);
        assert_eq!(strong.coords, super::weight(super::BOLD));
        assert_eq!(strong.color, super::STRONG);
        assert_eq!(inline.font_id.family, code);
        assert_eq!(heading.coords, super::weight(super::HEADING));
        // Both faces really carry the weight axis the coords name.
        for face in ["Inter", "JetBrains Mono"] {
            let data = egui::FontData::from_static(super::shipped(face).expect(face));
            assert!(
                data.variation_axes().iter().any(|axis| axis.tag == "wght"),
                "{face}"
            );
        }
    }

    #[test]
    fn without_the_faces_an_answer_is_set_in_the_window_faces() {
        let ctx = Context::default();
        let mut output = ctx.run_ui(RawInput::default(), |ui| {
            answer(ui, "**bold** `code`", ui.scope_id().with("answer"))
        });
        output.textures_delta.clear();
        let format = section(&output, "bold code", 0);
        assert_eq!(format.font_id.family, egui::FontFamily::Proportional);
        assert_eq!(
            section(&output, "bold code", "bold ".len()).font_id.family,
            egui::FontFamily::Monospace
        );
    }

    /// `text` as the real renderer paints an answer, in a 360 point column.
    fn rendered(text: &str, height: f32) -> image::RgbaImage {
        let text = text.to_owned();
        let mut harness = egui_kittest::Harness::builder()
            .with_size(Vec2::new(360.0, height))
            .wgpu()
            .build_ui(move |ui| {
                super::install_fonts(ui.ctx());
                ui.painter()
                    .rect_filled(ui.max_rect(), 0.0, crate::tokens::PANEL);
                answer(ui, &text, ui.scope_id().with("answer"));
            });
        harness.run_steps(3);
        harness.render().expect("the renderer draws the answer")
    }

    /// How much light the words put on the panel.
    fn ink(image: &image::RgbaImage) -> u64 {
        let ground = u64::from(crate::tokens::PANEL.g());
        image
            .pixels()
            .map(|pixel| u64::from(pixel.0[1]).saturating_sub(ground))
            .sum()
    }

    /// The answer painted by the real renderer. With `CUTLASS_RENDER_DIR`
    /// set the picture is kept there to look at.
    #[test]
    fn strong_words_are_painted_heavier() {
        let regular = ink(&rendered("Cut here", 40.0));
        let bold = ink(&rendered("Cut **here**", 40.0));
        assert!(regular > 0, "nothing was painted");
        // The weight, not the brighter white, makes most of the difference.
        assert!(bold * 10 > regular * 12, "{bold} against {regular}");
        if let Some(dir) = std::env::var_os("CUTLASS_RENDER_DIR") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            rendered(ANSWER, 520.0)
                .save(dir.join("assistant-answer.png"))
                .unwrap();
        }
    }

    #[test]
    fn an_answer_is_read_once_while_it_stays_on_screen() {
        let ctx = Context::default();
        let before = PARSES.with(std::cell::Cell::get);
        for _ in 0..5 {
            let _ = frame(&ctx, ANSWER, Vec::new());
        }
        assert_eq!(PARSES.with(std::cell::Cell::get) - before, 1);
        let _ = frame(&ctx, "another answer", Vec::new());
        assert_eq!(PARSES.with(std::cell::Cell::get) - before, 2);
    }
}
