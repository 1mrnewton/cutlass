import SwiftUI

/// Add text, or change a text clip: the words on top with a check that
/// closes, then the faces Rust found or the title's styles. The words show
/// on the preview as they are typed: the first letter adds the clip at the
/// playhead, and the whole typing is one undo step. Picking a face or a
/// color applies it at once, as an undo step of its own, and each slider
/// drag is one step. Bold, italic and underline go on the letters selected
/// in the words, or on the whole title when none are.
struct TextPanel: View {
    let model: EditorModel
    /// The clip being changed, or nil to add one.
    let clip: String?
    /// The clip the words go into: `clip`, or the one the first letter added.
    @State private var typed: String?
    let close: () -> Void
    @State private var words: String
    @State private var face: String
    @State private var tab: Tab = .fonts
    /// The letters selected in the words, counted in characters as Rust
    /// counts them; nil when none are.
    @State private var marked: Range<Int>?
    /// What every selected letter has, or the title's own pattern when
    /// none are selected.
    @State private var pattern: TextPattern?
    @State private var query = ""
    /// The faces Rust matched to the search.
    @State private var found: [Font]
    /// The axes the clip's face varies along.
    @State private var axes: [FontAxis] = []
    @FocusState private var typing: Bool
    @FocusState private var searching: Bool

    enum Tab: String, CaseIterable {
        case fonts = "Fonts"
        case styles = "Styles"
    }

    init(model: EditorModel, clip: String?, close: @escaping () -> Void) {
        self.model = model
        self.clip = clip
        self.close = close
        _typed = State(initialValue: clip)
        let text = clip.flatMap { model.text(of: $0) }
        _words = State(initialValue: text?.content ?? "")
        _face = State(initialValue: text?.face ?? model.fonts.first?.name ?? "")
        _found = State(initialValue: model.fonts)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Group {
                    if #available(iOS 18.0, *) {
                        SelectingField(words: $words, marked: $marked)
                    } else {
                        TextField("Enter text", text: $words, axis: .vertical)
                    }
                }
                .lineLimit(1...3)
                .focused($typing)
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .background(Color(white: 0.18), in: RoundedRectangle(cornerRadius: 8))
                .submitLabel(.done)
                Button(action: done) {
                    HugeIcon("done", size: 20)
                        .frame(width: 40, height: 36)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Done")
            }
            Picker("Text", selection: $tab) {
                ForEach(Tab.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            switch tab {
            case .fonts: fonts
            case .styles:
                if let typed {
                    StylesTab(model: model, clip: typed)
                } else {
                    Text("Type some words to style them.")
                        .font(.footnote)
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 16)
        .padding(.top, 12)
        .padding(.bottom, 16)
        .background(Color(white: 0.11))
        .onAppear {
            typing = clip == nil
            read()
        }
        .onChange(of: words) { _, words in
            typed = model.typeText(into: typed, words, face: face)
        }
        .onChange(of: tab) { _, tab in
            if tab == .styles {
                typing = false
                searching = false
            }
        }
        .onChange(of: query) { _, query in found = model.fontsMatching(query) }
        .onChange(of: marked) { _, _ in read() }
        .onChange(of: typing) { _, _ in read() }
        .onChange(of: model.edits) { _, _ in read() }
    }

    /// The letters the buttons act on. The field keeps its selection after
    /// the keyboard leaves it, where nothing shows it, so it counts only
    /// while the words are being typed.
    private var selected: Range<Int>? { typing ? marked : nil }

    /// Read what the buttons and the sliders show from the clip.
    private func read() {
        guard let typed, let style = model.text(of: typed)?.style else {
            pattern = nil
            axes = []
            return
        }
        pattern = selected.flatMap { model.textPattern(of: typed, letters: $0) }
            ?? TextPattern(bold: style.bold, italic: style.italic, underline: style.underline)
        axes = model.fontAxes(of: typed)
    }

    private var fonts: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                letterToggle("bold", on: pattern?.bold, label: "Bold") {
                    setPattern(bold: $0)
                }
                letterToggle("italic", on: pattern?.italic, label: "Italic") {
                    setPattern(italic: $0)
                }
                letterToggle("underline", on: pattern?.underline, label: "Underline") {
                    setPattern(underline: $0)
                }
                search
            }
            if found.isEmpty {
                Text(model.fonts.isEmpty ? "No fonts were found on this device." : "No fonts match.")
                    .lineLimit(1)
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.6))
                    .frame(height: 34)
            } else {
                faces
            }
            ForEach(axes.filter { $0.min < $0.max }, id: \.tag) { axis in
                StyleSlider(
                    label: axis.name, value: axis.value, range: axis.min...axis.max,
                    shown: { "\(Int($0.rounded()))" }
                ) { value, live in
                    if let typed {
                        model.setTextFontAxis(typed, tag: axis.tag, value: value, live: live)
                    }
                }
                .id(face + axis.tag)
            }
        }
    }

    /// The search over the faces. Rust matches the words.
    private var search: some View {
        HStack(spacing: 8) {
            HugeIcon("search", size: 16)
                .foregroundStyle(.white.opacity(0.6))
            TextField("Search fonts", text: $query)
                .font(.footnote)
                .focused($searching)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
                .submitLabel(.search)
            if !query.isEmpty {
                Button { query = "" } label: {
                    HugeIcon("close", size: 14)
                        .foregroundStyle(.white.opacity(0.6))
                        .frame(width: 24, height: 24)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Clear search")
            }
        }
        .padding(.horizontal, 10)
        .frame(height: 36)
        .background(Color(white: 0.18), in: RoundedRectangle(cornerRadius: 8))
    }

    /// The faces in a row. Only the ones on screen are built, so a long
    /// list stays quick.
    private var faces: some View {
        ScrollViewReader { row in
            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 8) {
                    ForEach(found, id: \.name) { font in
                        Button { pick(font.name) } label: {
                            Text(font.label)
                                .font(.footnote)
                                .lineLimit(1)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(
                                    font.name == face ? Palette.accent : Color(white: 0.18),
                                    in: Capsule())
                                .foregroundStyle(font.name == face ? .black : .white)
                        }
                        .accessibilityAddTraits(font.name == face ? .isSelected : [])
                    }
                }
            }
            .frame(height: 34)
            .onAppear { row.scrollTo(face, anchor: .center) }
            .onChange(of: found) { _, found in
                // A search shows its faces from the first; the whole list
                // shows the chosen face again.
                if query.isEmpty {
                    row.scrollTo(face, anchor: .center)
                } else if let first = found.first {
                    row.scrollTo(first.name, anchor: .leading)
                }
            }
        }
    }

    /// A button for one of bold, italic and underline. `on` is nil while
    /// there are no letters to act on.
    private func letterToggle(
        _ icon: String, on: Bool?, label: String, set: @escaping (Bool) -> Void
    ) -> some View {
        let lit = on == true
        return Button { set(!lit) } label: {
            HugeIcon(icon, size: 18)
                .frame(width: 36, height: 36)
                .foregroundStyle(lit ? .black : .white)
                .background(
                    lit ? Color.white : Color(white: 0.2), in: RoundedRectangle(cornerRadius: 8))
        }
        .disabled(on == nil)
        .opacity(on == nil ? 0.4 : 1)
        .accessibilityLabel(label)
        .accessibilityAddTraits(lit ? .isSelected : [])
    }

    /// Change the selected letters, or the title's own pattern when none
    /// are selected, as the desktop's buttons do.
    private func setPattern(bold: Bool? = nil, italic: Bool? = nil, underline: Bool? = nil) {
        guard let typed else { return }
        if let selected {
            model.setTextPattern(
                typed, letters: selected, bold: bold, italic: italic, underline: underline)
        } else if let pattern {
            model.styleText(
                typed,
                .pattern(
                    bold: bold ?? pattern.bold, italic: italic ?? pattern.italic,
                    underline: underline ?? pattern.underline),
                live: false)
        }
    }

    private func pick(_ name: String) {
        face = name
        if let typed { model.setTextFont(typed, face: name) }
    }

    /// Close. The words are already in the clip; blank words added nothing
    /// and leave a clip's words as they were.
    private func done() {
        typing = false
        searching = false
        close()
    }
}

/// The words, in a field that also tells which letters are selected. The
/// selection is counted in characters, as Rust counts a title's letters.
@available(iOS 18.0, *)
private struct SelectingField: View {
    @Binding var words: String
    @Binding var marked: Range<Int>?
    @State private var selection: TextSelection?

    var body: some View {
        TextField("Enter text", text: $words, selection: $selection, axis: .vertical)
            .onChange(of: selection) { _, _ in marked = letters() }
            .onChange(of: words) { _, _ in marked = letters() }
    }

    private func letters() -> Range<Int>? {
        guard case let .selection(range) = selection?.indices, !range.isEmpty,
              range.upperBound <= words.endIndex
        else { return nil }
        let scalars = words.unicodeScalars
        let start = scalars.distance(from: scalars.startIndex, to: range.lowerBound)
        return start..<start + scalars.distance(from: range.lowerBound, to: range.upperBound)
    }
}

/// A title's color and size, format, outline, background, shadow, and glow.
/// Colors, ranges, and the values a part starts at come from Rust.
private struct StylesTab: View {
    let model: EditorModel
    let clip: String
    @State private var part: Part = .text
    @State private var style: TextStyle?

    enum Part: String, CaseIterable {
        case text = "Text"
        case format = "Format"
        case outline = "Outline"
        case background = "Background"
        case shadow = "Shadow"
        case glow = "Glow"
    }

    private static let colors = textColors()
    private static let ranges = textStyleRanges()

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 16) {
                ForEach(Part.allCases, id: \.self) { part in
                    Button { self.part = part } label: {
                        Text(part.rawValue)
                            .font(.footnote.weight(self.part == part ? .semibold : .regular))
                            .foregroundStyle(self.part == part ? .white : .white.opacity(0.55))
                    }
                    .accessibilityAddTraits(self.part == part ? .isSelected : [])
                }
            }
            if let style {
                if part == .format { format(style) } else { swatches(style) }
                // Room for two sliders on every part, so the panel and the
                // preview above it keep their size.
                VStack(spacing: 6) { sliders(style) }
                    .frame(height: 84, alignment: .top)
            }
        }
        .onAppear(perform: read)
        .onChange(of: model.edits) { _, _ in read() }
    }

    private func read() {
        style = model.text(of: clip)?.style
    }

    private func swatches(_ style: TextStyle) -> some View {
        let chosen = chosenColor(style)
        return ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                if part != .text {
                    NoneTile(chosen: chosen.isEmpty, size: 36) { setColor("", style) }
                }
                ForEach(Self.colors, id: \.self) { color in
                    SwatchTile(color: color, chosen: same(color, chosen), size: 36) {
                        setColor(color, style)
                    }
                }
            }
            .padding(4)
        }
    }

    /// Bold, italic and underline, then left, center and right.
    private func format(_ style: TextStyle) -> some View {
        HStack(spacing: 8) {
            toggle("B", on: style.bold, label: "Bold") {
                send(.pattern(bold: !style.bold, italic: style.italic, underline: style.underline),
                     live: false)
            }
            toggle("I", on: style.italic, label: "Italic") {
                send(.pattern(bold: style.bold, italic: !style.italic, underline: style.underline),
                     live: false)
            }
            toggle("U", on: style.underline, label: "Underline") {
                send(.pattern(bold: style.bold, italic: style.italic, underline: !style.underline),
                     live: false)
            }
            Spacer().frame(width: 12)
            ForEach(
                [(0, "text.alignleft", "Left"), (1, "text.aligncenter", "Center"),
                 (2, "text.alignright", "Right")], id: \.0
            ) { alignment, icon, label in
                toggle(icon, symbol: true, on: style.alignment == alignment, label: "Align \(label)") {
                    send(.align(alignment: Int32(alignment)), live: false)
                }
            }
        }
        .padding(4)
        .frame(height: 44)
    }

    private func toggle(
        _ text: String, symbol: Bool = false, on: Bool, label: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Group {
                if symbol {
                    Image(systemName: text).font(.system(size: 15))
                } else {
                    Text(text)
                        .font(.system(size: 16, weight: text == "B" ? .bold : .regular))
                        .italic(text == "I")
                        .underline(text == "U")
                }
            }
            .frame(width: 36, height: 36)
            .foregroundStyle(on ? .black : .white)
            .background(on ? Color.white : Color(white: 0.2), in: RoundedRectangle(cornerRadius: 8))
        }
        .accessibilityLabel(label)
        .accessibilityAddTraits(on ? .isSelected : [])
    }

    @ViewBuilder
    private func sliders(_ style: TextStyle) -> some View {
        let ranges = Self.ranges
        switch part {
        case .format:
            StyleSlider(
                label: "Letters", value: style.letterSpacing,
                range: ranges.letterSpacingMin...ranges.letterSpacingMax, shown: percent
            ) { spacing, live in
                send(.letterSpacing(spacing: spacing), live: live)
            }
            StyleSlider(
                label: "Lines", value: style.lineSpacing,
                range: ranges.lineSpacingMin...ranges.lineSpacingMax, shown: percent
            ) { spacing, live in
                send(.lineSpacing(spacing: spacing), live: live)
            }
        case .text:
            StyleSlider(
                label: "Size", value: style.size, range: ranges.sizeMin...ranges.sizeMax,
                shown: { "\(Int($0.rounded()))" }
            ) { size, live in
                send(.size(size: size), live: live)
            }
        case .outline:
            if !style.outlineColor.isEmpty {
                StyleSlider(
                    label: "Width", value: style.outlineWidth, range: 0...ranges.outlineMax,
                    shown: { percent($0 / ranges.outlineMax) }
                ) { width, live in
                    send(.outline(color: style.outlineColor, width: width), live: live)
                }
            }
        case .background:
            if !style.backgroundColor.isEmpty {
                StyleSlider(
                    label: "Opacity", value: style.backgroundAlpha, range: 0...1, shown: percent
                ) { alpha, live in
                    send(background(style, alpha: alpha), live: live)
                }
                StyleSlider(
                    label: "Corners", value: style.backgroundRound, range: 0...1, shown: percent
                ) { round, live in
                    send(background(style, round: round), live: live)
                }
            }
        case .shadow:
            if !style.shadowColor.isEmpty {
                StyleSlider(
                    label: "Distance", value: style.shadowDistance,
                    range: 0...ranges.shadowDistanceMax,
                    shown: { percent($0 / ranges.shadowDistanceMax) }
                ) { distance, live in
                    send(shadow(style, distance: distance), live: live)
                }
                StyleSlider(
                    label: "Blur", value: style.shadowBlur, range: 0...1, shown: percent
                ) { blur, live in
                    send(shadow(style, blur: blur), live: live)
                }
            }
        case .glow:
            if !style.glowColor.isEmpty {
                StyleSlider(
                    label: "Strength", value: style.glowStrength, range: 0...1, shown: percent
                ) { strength, live in
                    send(glow(style, strength: strength), live: live)
                }
                StyleSlider(
                    label: "Spread", value: style.glowRange, range: 0...1, shown: percent
                ) { range, live in
                    send(glow(style, range: range), live: live)
                }
            }
        }
    }

    private func chosenColor(_ style: TextStyle) -> String {
        switch part {
        case .text: style.color.isEmpty ? "#FFFFFF" : style.color
        case .format: ""
        case .outline: style.outlineColor
        case .background: style.backgroundColor
        case .shadow: style.shadowColor
        case .glow: style.glowColor
        }
    }

    /// Pick a color; a part turned on starts at Rust's values.
    private func setColor(_ color: String, _ style: TextStyle) {
        let ranges = Self.ranges
        switch part {
        case .text:
            send(.color(color: color), live: false)
        case .format:
            break
        case .outline:
            let width = style.outlineWidth > 0 ? style.outlineWidth : ranges.outlineStart
            send(.outline(color: color, width: width), live: false)
        case .background:
            let fresh = style.backgroundColor.isEmpty
            send(
                .background(
                    color: color,
                    alpha: fresh ? ranges.backgroundAlphaStart : style.backgroundAlpha,
                    round: fresh ? ranges.backgroundRoundStart : style.backgroundRound),
                live: false)
        case .shadow:
            let fresh = style.shadowColor.isEmpty
            send(
                .shadow(
                    color: color,
                    distance: fresh ? ranges.shadowDistanceStart : style.shadowDistance,
                    blur: fresh ? ranges.shadowBlurStart : style.shadowBlur),
                live: false)
        case .glow:
            let fresh = style.glowColor.isEmpty
            send(
                .glow(
                    color: color,
                    strength: fresh ? ranges.glowStrengthStart : style.glowStrength,
                    range: fresh ? ranges.glowRangeStart : style.glowRange),
                live: false)
        }
    }

    private func background(
        _ style: TextStyle, alpha: Double? = nil, round: Double? = nil
    ) -> TextStyleChange {
        .background(
            color: style.backgroundColor,
            alpha: alpha ?? style.backgroundAlpha,
            round: round ?? style.backgroundRound)
    }

    private func shadow(
        _ style: TextStyle, distance: Double? = nil, blur: Double? = nil
    ) -> TextStyleChange {
        .shadow(
            color: style.shadowColor,
            distance: distance ?? style.shadowDistance,
            blur: blur ?? style.shadowBlur)
    }

    private func glow(
        _ style: TextStyle, strength: Double? = nil, range: Double? = nil
    ) -> TextStyleChange {
        .glow(
            color: style.glowColor,
            strength: strength ?? style.glowStrength,
            range: range ?? style.glowRange)
    }

    private func send(_ change: TextStyleChange, live: Bool) {
        model.styleText(clip, change, live: live)
    }

    private func same(_ one: String, _ other: String) -> Bool {
        one.caseInsensitiveCompare(other) == .orderedSame
    }
}

private func percent(_ fraction: Double) -> String {
    "\(Int((fraction * 100).rounded()))"
}

/// A labelled slider that reports each move as live and the release as
/// final, so a drag is one undo step. Between drags it follows the title,
/// so an undo moves it back.
struct StyleSlider: View {
    let label: String
    let current: Double
    let range: ClosedRange<Double>
    let shown: (Double) -> String
    let change: (Double, Bool) -> Void
    @State private var value: Double
    @State private var dragging = false

    init(
        label: String, value: Double, range: ClosedRange<Double>,
        shown: @escaping (Double) -> String, change: @escaping (Double, Bool) -> Void
    ) {
        self.label = label
        self.current = value
        self.range = range
        self.shown = shown
        self.change = change
        _value = State(initialValue: min(max(value, range.lowerBound), range.upperBound))
    }

    var body: some View {
        HStack(spacing: 12) {
            Text(label)
                .font(.footnote)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .foregroundStyle(.white.opacity(0.75))
                .frame(width: 72, alignment: .leading)
            Slider(
                value: Binding(
                    get: { value },
                    set: { value = $0; change($0, true) }),
                in: range
            ) { editing in
                dragging = editing
                if !editing { change(value, false) }
            }
            .tint(Palette.accent)
            .accessibilityLabel(label)
            .accessibilityValue(shown(value))
            Text(shown(value))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.white.opacity(0.6))
                .frame(width: 36, alignment: .trailing)
        }
        .onChange(of: current) { _, current in
            if !dragging { value = min(max(current, range.lowerBound), range.upperBound) }
        }
    }
}
