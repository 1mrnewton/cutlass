//! The model's answer as blocks a panel paints: paragraphs, headings, lists,
//! quotes, code and tables, each holding runs of styled words.
//!
//! The model writes markdown. Each shell paints these blocks its own way,
//! so the desktop and a phone read an answer the same. The answer is
//! untrusted text: a link keeps its address only when it is a web or mail
//! link, raw HTML is shown as the words it is, and nesting stops deepening
//! after [`MOST_NESTED`] quotes and lists.

use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};

/// The most quotes and lists inside one another that keep their own block.
/// Deeper ones put their words in the deepest block kept.
pub const MOST_NESTED: usize = 8;

/// One block of an answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Block {
    Paragraph(Vec<Span>),
    /// A heading, level 1 the largest, up to 6.
    Heading {
        level: u8,
        spans: Vec<Span>,
    },
    /// A code block: its language as written after the fence, maybe
    /// empty, and its lines without the last line break.
    Code {
        language: String,
        text: String,
    },
    Quote(Vec<Block>),
    /// A list. A numbered one counts up from `first`; a bulleted one has
    /// none.
    List {
        first: Option<u64>,
        items: Vec<Item>,
    },
    /// A table: the head's cells, then each row's.
    Table {
        head: Vec<Vec<Span>>,
        rows: Vec<Vec<Vec<Span>>>,
    },
    /// A line across.
    Rule,
}

/// One item of a list: whether it is a ticked or unticked task, and what
/// it holds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Item {
    pub task: Option<bool>,
    pub blocks: Vec<Block>,
}

/// Words that share one style.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub text: String,
    pub style: Style,
}

/// How a run of words is drawn.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Style {
    pub strong: bool,
    pub emphasis: bool,
    pub struck: bool,
    /// Inline code.
    pub code: bool,
    /// Where the words link to: only an `http`, `https` or `mailto`
    /// address. Words linked elsewhere are drawn as plain words.
    pub link: Option<String>,
}

/// `text`, the model's markdown, as blocks.
#[must_use]
pub fn blocks(text: &str) -> Vec<Block> {
    let options =
        Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
    let mut builder = Builder::default();
    for event in Parser::new_ext(text, options) {
        builder.event(event);
    }
    builder.finish()
}

/// `url` when the panel may open it: a web or mail address.
fn safe_link(url: &str) -> Option<String> {
    let url = url.trim();
    let scheme = url.split_once(':')?.0.to_ascii_lowercase();
    matches!(scheme.as_str(), "http" | "https" | "mailto").then(|| url.to_owned())
}

/// A block or a part of one that is open while the events fill it.
enum Open {
    /// The answer itself, at the bottom.
    Answer(Vec<Block>),
    Quote(Vec<Block>),
    List {
        first: Option<u64>,
        items: Vec<Item>,
    },
    Item(Item),
    Table {
        head: Vec<Vec<Span>>,
        rows: Vec<Vec<Vec<Span>>>,
        row: Vec<Vec<Span>>,
    },
    /// A quote, list or item past [`MOST_NESTED`]: its words go to the
    /// block under it.
    Flat,
}

/// What the words being read go into.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Words {
    Paragraph,
    Heading(u8),
    Cell,
}

struct Builder {
    open: Vec<Open>,
    words: Option<Words>,
    spans: Vec<Span>,
    /// The code block being read: its language and text.
    code: Option<(String, String)>,
    strong: usize,
    emphasis: usize,
    struck: usize,
    /// Each link open around the words, with its address if it is kept.
    links: Vec<Option<String>>,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            open: vec![Open::Answer(Vec::new())],
            words: None,
            spans: Vec::new(),
            code: None,
            strong: 0,
            emphasis: 0,
            struck: 0,
            links: Vec::new(),
        }
    }
}

impl Builder {
    fn event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(text) => {
                if let Some((_, code)) = &mut self.code {
                    code.push_str(&text);
                } else {
                    self.push(&text, false);
                }
            }
            Event::Code(text) | Event::InlineMath(text) => self.push(&text, true),
            Event::DisplayMath(text) => {
                self.end_words();
                self.block(Block::Code {
                    language: String::new(),
                    text: text.trim_end_matches('\n').to_owned(),
                });
            }
            // Raw HTML is shown as written; the panel draws no HTML.
            Event::Html(text) | Event::InlineHtml(text) => self.push(&text, false),
            Event::FootnoteReference(name) => self.push(&format!("[^{name}]"), false),
            Event::SoftBreak => self.push(" ", false),
            Event::HardBreak => self.push("\n", false),
            Event::Rule => {
                self.end_words();
                self.block(Block::Rule);
            }
            Event::TaskListMarker(ticked) => {
                if let Some(Open::Item(item)) = self.open.last_mut() {
                    item.task = Some(ticked);
                }
            }
        }
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {
                self.end_words();
                self.words = Some(Words::Paragraph);
            }
            Tag::Heading { level, .. } => {
                self.end_words();
                // The levels are 1 to 6.
                self.words = Some(Words::Heading(level as u8));
            }
            Tag::CodeBlock(kind) => {
                self.end_words();
                let language = match kind {
                    CodeBlockKind::Fenced(info) => {
                        info.split_whitespace().next().unwrap_or("").to_owned()
                    }
                    CodeBlockKind::Indented => String::new(),
                };
                self.code = Some((language, String::new()));
            }
            Tag::HtmlBlock => {
                self.end_words();
                self.words = Some(Words::Paragraph);
            }
            Tag::BlockQuote(_) => {
                self.end_words();
                let open = self.nested(Open::Quote(Vec::new()));
                self.open.push(open);
            }
            Tag::List(first) => {
                self.end_words();
                let open = self.nested(Open::List {
                    first,
                    items: Vec::new(),
                });
                self.open.push(open);
            }
            Tag::Item => {
                self.end_words();
                let open = if matches!(self.open.last(), Some(Open::List { .. })) {
                    Open::Item(Item::default())
                } else {
                    Open::Flat
                };
                self.open.push(open);
            }
            Tag::Table(_) => {
                self.end_words();
                self.open.push(Open::Table {
                    head: Vec::new(),
                    rows: Vec::new(),
                    row: Vec::new(),
                });
            }
            Tag::TableHead | Tag::TableRow => {
                if let Some(Open::Table { row, .. }) = self.open.last_mut() {
                    row.clear();
                }
            }
            Tag::TableCell => {
                self.spans.clear();
                self.words = Some(Words::Cell);
            }
            Tag::Emphasis => self.emphasis += 1,
            Tag::Strong => self.strong += 1,
            Tag::Strikethrough => self.struck += 1,
            Tag::Link { dest_url, .. } => self.links.push(safe_link(&dest_url)),
            // An image shows its words and links nowhere.
            Tag::Image { .. } => self.links.push(None),
            Tag::FootnoteDefinition(_)
            | Tag::DefinitionList
            | Tag::DefinitionListTitle
            | Tag::DefinitionListDefinition
            | Tag::Superscript
            | Tag::Subscript
            | Tag::MetadataBlock(_) => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph | TagEnd::Heading(_) | TagEnd::HtmlBlock => self.end_words(),
            TagEnd::CodeBlock => {
                if let Some((language, mut text)) = self.code.take() {
                    if text.ends_with('\n') {
                        text.pop();
                    }
                    self.block(Block::Code { language, text });
                }
            }
            TagEnd::BlockQuote(_) | TagEnd::List(_) | TagEnd::Item => {
                self.end_words();
                self.close();
            }
            TagEnd::TableCell => {
                let spans = std::mem::take(&mut self.spans);
                self.words = None;
                if let Some(Open::Table { row, .. }) = self.open.last_mut() {
                    row.push(spans);
                }
            }
            TagEnd::TableHead => {
                if let Some(Open::Table { head, row, .. }) = self.open.last_mut() {
                    *head = std::mem::take(row);
                }
            }
            TagEnd::TableRow => {
                if let Some(Open::Table { rows, row, .. }) = self.open.last_mut() {
                    rows.push(std::mem::take(row));
                }
            }
            TagEnd::Table => self.close(),
            TagEnd::Emphasis => self.emphasis = self.emphasis.saturating_sub(1),
            TagEnd::Strong => self.strong = self.strong.saturating_sub(1),
            TagEnd::Strikethrough => self.struck = self.struck.saturating_sub(1),
            TagEnd::Link | TagEnd::Image => {
                self.links.pop();
            }
            TagEnd::FootnoteDefinition
            | TagEnd::DefinitionList
            | TagEnd::DefinitionListTitle
            | TagEnd::DefinitionListDefinition
            | TagEnd::Superscript
            | TagEnd::Subscript
            | TagEnd::MetadataBlock(_) => {}
        }
    }

    /// `open` when it still fits under [`MOST_NESTED`], or a flat one.
    fn nested(&self, open: Open) -> Open {
        let depth = self
            .open
            .iter()
            .filter(|open| matches!(open, Open::Quote(_) | Open::List { .. }))
            .count();
        if depth < MOST_NESTED {
            open
        } else {
            Open::Flat
        }
    }

    /// Words in the current style; inline code when `code`.
    fn push(&mut self, text: &str, code: bool) {
        if text.is_empty() {
            return;
        }
        if self.words.is_none() {
            // A tight list's item holds words with no paragraph around them.
            self.words = Some(Words::Paragraph);
        }
        let style = Style {
            strong: self.strong > 0,
            emphasis: self.emphasis > 0,
            struck: self.struck > 0,
            code,
            link: self.links.last().cloned().flatten(),
        };
        match self.spans.last_mut() {
            Some(last) if last.style == style => last.text.push_str(text),
            _ => self.spans.push(Span {
                text: text.to_owned(),
                style,
            }),
        }
    }

    /// Ends the paragraph or heading being read, if any.
    fn end_words(&mut self) {
        let Some(words) = self.words else { return };
        if words == Words::Cell {
            return;
        }
        self.words = None;
        let mut spans = std::mem::take(&mut self.spans);
        // An HTML block ends on its line break.
        if let Some(last) = spans.last_mut() {
            let kept = last.text.trim_end_matches('\n').len();
            last.text.truncate(kept);
            if last.text.is_empty() {
                spans.pop();
            }
        }
        match words {
            _ if spans.is_empty() => {}
            Words::Heading(level) => self.block(Block::Heading { level, spans }),
            _ => self.block(Block::Paragraph(spans)),
        }
    }

    /// Puts `block` in the innermost open block that holds blocks.
    fn block(&mut self, block: Block) {
        for open in self.open.iter_mut().rev() {
            match open {
                Open::Answer(blocks) | Open::Quote(blocks) => return blocks.push(block),
                Open::Item(item) => return item.blocks.push(block),
                Open::List { .. } | Open::Table { .. } | Open::Flat => {}
            }
        }
    }

    /// Closes the innermost open block into the one under it.
    fn close(&mut self) {
        if self.open.len() <= 1 {
            return;
        }
        match self.open.pop() {
            Some(Open::Quote(blocks)) => self.block(Block::Quote(blocks)),
            Some(Open::List { first, items }) => self.block(Block::List { first, items }),
            Some(Open::Item(item)) => {
                if let Some(Open::List { items, .. }) = self.open.last_mut() {
                    items.push(item);
                }
            }
            Some(Open::Table { head, rows, .. }) => self.block(Block::Table { head, rows }),
            Some(Open::Answer(_) | Open::Flat) | None => {}
        }
    }

    fn finish(mut self) -> Vec<Block> {
        self.end_words();
        while self.open.len() > 1 {
            self.close();
        }
        match self.open.pop() {
            Some(Open::Answer(blocks)) => blocks,
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Block, Item, MOST_NESTED, Span, Style, blocks};

    fn plain(text: &str) -> Span {
        Span {
            text: text.to_owned(),
            style: Style::default(),
        }
    }

    fn styled(text: &str, style: Style) -> Span {
        Span {
            text: text.to_owned(),
            style,
        }
    }

    #[test]
    fn nothing_is_no_blocks_and_plain_words_are_one_paragraph() {
        assert!(blocks("").is_empty());
        assert!(blocks("  \n\n ").is_empty());
        assert_eq!(
            blocks("The playhead\nis at 1.5 s."),
            [Block::Paragraph(vec![plain("The playhead is at 1.5 s.")])]
        );
        assert_eq!(
            blocks("one\n\ntwo"),
            [
                Block::Paragraph(vec![plain("one")]),
                Block::Paragraph(vec![plain("two")]),
            ]
        );
    }

    #[test]
    fn words_keep_their_styles_and_runs_of_one_style_join() {
        let strong = Style {
            strong: true,
            ..Style::default()
        };
        let both = Style {
            strong: true,
            emphasis: true,
            ..Style::default()
        };
        let code = Style {
            code: true,
            ..Style::default()
        };
        let struck = Style {
            struck: true,
            ..Style::default()
        };
        assert_eq!(
            blocks("Cut **at *two* s** with `split` ~~now~~ &amp; done"),
            [Block::Paragraph(vec![
                plain("Cut "),
                styled("at ", strong.clone()),
                styled("two", both),
                styled(" s", strong),
                plain(" with "),
                styled("split", code),
                plain(" "),
                styled("now", struck),
                plain(" & done"),
            ])]
        );
        // A hard break stays a line break.
        assert_eq!(
            blocks("one  \ntwo"),
            [Block::Paragraph(vec![plain("one\ntwo")])]
        );
    }

    #[test]
    fn headings_rules_and_code_blocks() {
        assert_eq!(
            blocks(
                "# Plan\n\n### Steps\n\n---\n\n```rust extra\nfn a() {}\n\n```\n\n    indented\n"
            ),
            [
                Block::Heading {
                    level: 1,
                    spans: vec![plain("Plan")],
                },
                Block::Heading {
                    level: 3,
                    spans: vec![plain("Steps")],
                },
                Block::Rule,
                Block::Code {
                    language: "rust".to_owned(),
                    text: "fn a() {}\n".to_owned(),
                },
                Block::Code {
                    language: String::new(),
                    text: "indented".to_owned(),
                },
            ]
        );
        // Markdown inside code is code.
        assert_eq!(
            blocks("```\n**not bold**\n```"),
            [Block::Code {
                language: String::new(),
                text: "**not bold**".to_owned(),
            }]
        );
        // An empty heading is left out.
        assert!(blocks("#").is_empty());
    }

    #[test]
    fn lists_number_from_their_first_and_hold_blocks() {
        let item = |text: &str| Item {
            task: None,
            blocks: vec![Block::Paragraph(vec![plain(text)])],
        };
        assert_eq!(
            blocks("- a\n- b\n  1. c\n  2. d"),
            [Block::List {
                first: None,
                items: vec![
                    item("a"),
                    Item {
                        task: None,
                        blocks: vec![
                            Block::Paragraph(vec![plain("b")]),
                            Block::List {
                                first: Some(1),
                                items: vec![item("c"), item("d")],
                            },
                        ],
                    },
                ],
            }]
        );
        assert_eq!(
            blocks("3. three\n\n4. four"),
            [Block::List {
                first: Some(3),
                items: vec![item("three"), item("four")],
            }]
        );
        assert_eq!(
            blocks("- [x] done\n- [ ] to do"),
            [Block::List {
                first: None,
                items: vec![
                    Item {
                        task: Some(true),
                        ..item("done")
                    },
                    Item {
                        task: Some(false),
                        ..item("to do")
                    },
                ],
            }]
        );
    }

    #[test]
    fn quotes_and_tables() {
        assert_eq!(
            blocks("> said\n>\n> - a"),
            [Block::Quote(vec![
                Block::Paragraph(vec![plain("said")]),
                Block::List {
                    first: None,
                    items: vec![Item {
                        task: None,
                        blocks: vec![Block::Paragraph(vec![plain("a")])],
                    }],
                },
            ])]
        );
        let bold = Style {
            strong: true,
            ..Style::default()
        };
        assert_eq!(
            blocks("| Clip | Length |\n|---|--:|\n| **a** | 2 s |\n| b |"),
            [Block::Table {
                head: vec![vec![plain("Clip")], vec![plain("Length")]],
                rows: vec![
                    vec![vec![styled("a", bold)], vec![plain("2 s")]],
                    // A short row is filled with empty cells.
                    vec![vec![plain("b")], vec![]],
                ],
            }]
        );
    }

    #[test]
    fn only_web_and_mail_links_keep_their_address() {
        let link = |url: &str| Style {
            link: Some(url.to_owned()),
            ..Style::default()
        };
        assert_eq!(
            blocks(
                "[docs](https://example.com/a) and <mailto:me@example.com> and [HTTP](HTTP://x.org)"
            ),
            [Block::Paragraph(vec![
                styled("docs", link("https://example.com/a")),
                plain(" and "),
                styled("mailto:me@example.com", link("mailto:me@example.com")),
                plain(" and "),
                styled("HTTP", link("HTTP://x.org")),
            ])]
        );
        for hostile in [
            "[x](javascript:alert(1))",
            "[x](file:///etc/passwd)",
            "[x](../project.json)",
            "[x](data:text/html,hi)",
            "[x](cutlass://delete)",
        ] {
            assert_eq!(
                blocks(hostile),
                [Block::Paragraph(vec![plain("x")])],
                "{hostile}"
            );
        }
        // An image shows its words and links nowhere.
        assert_eq!(
            blocks("![a frame](https://example.com/f.png)"),
            [Block::Paragraph(vec![plain("a frame")])]
        );
    }

    #[test]
    fn html_is_shown_as_written() {
        assert_eq!(
            blocks("a <b>bold</b> word"),
            [Block::Paragraph(vec![plain("a <b>bold</b> word")])]
        );
        assert_eq!(
            blocks("<script>alert(1)</script>\n\nafter"),
            [
                Block::Paragraph(vec![plain("<script>alert(1)</script>")]),
                Block::Paragraph(vec![plain("after")]),
            ]
        );
    }

    /// How many quotes and lists are inside one another in `blocks`.
    fn depth(blocks: &[Block]) -> usize {
        blocks
            .iter()
            .map(|block| match block {
                Block::Quote(inner) => 1 + depth(inner),
                Block::List { items, .. } => {
                    1 + items
                        .iter()
                        .map(|item| depth(&item.blocks))
                        .max()
                        .unwrap_or(0)
                }
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    }

    /// Every word in `blocks`, in order.
    fn words(blocks: &[Block], out: &mut String) {
        for block in blocks {
            match block {
                Block::Paragraph(spans) | Block::Heading { spans, .. } => {
                    for span in spans {
                        out.push_str(&span.text);
                        out.push(' ');
                    }
                }
                Block::Quote(inner) => words(inner, out),
                Block::List { items, .. } => {
                    for item in items {
                        words(&item.blocks, out);
                    }
                }
                _ => {}
            }
        }
    }

    #[test]
    fn deep_nesting_stops_deepening_and_keeps_every_word() {
        let quotes = format!("{} deep", ">".repeat(10_000));
        let parsed = blocks(&quotes);
        assert_eq!(depth(&parsed), MOST_NESTED);
        let mut said = String::new();
        words(&parsed, &mut said);
        assert_eq!(said.trim(), "deep");

        let mut lists = String::new();
        for level in 0..40 {
            lists.push_str(&"  ".repeat(level));
            lists.push_str(&format!("- l{level}\n"));
        }
        let parsed = blocks(&lists);
        assert_eq!(depth(&parsed), MOST_NESTED);
        let mut said = String::new();
        words(&parsed, &mut said);
        let expected: Vec<String> = (0..40).map(|level| format!("l{level}")).collect();
        assert_eq!(said.split_whitespace().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn a_long_answer_parses_in_one_pass() {
        // Hostile shapes that are slow in a naive parser stay linear here.
        let started = std::time::Instant::now();
        for text in [
            "*a ".repeat(20_000),
            "[".repeat(20_000),
            "`".repeat(20_000),
            "| a ".repeat(5_000) + "|\n" + "|---".repeat(5_000).as_str() + "|\n",
        ] {
            let _ = blocks(&text);
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }
}
