import SwiftUI

/// Add text, or change a text clip: the words on top with a check that
/// closes, then the faces Rust found or the title's styles. The words show
/// on the preview as they are typed: the first letter adds the clip at the
/// playhead, and the whole typing is one undo step. Picking a face or a
/// color applies it at once, as an undo step of its own, and each slider
/// drag is one step.
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
    @FocusState private var typing: Bool

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
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                TextField("Enter text", text: $words, axis: .vertical)
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
        .onAppear { typing = clip == nil }
        .onChange(of: words) { _, words in
            typed = model.typeText(into: typed, words, face: face)
        }
        .onChange(of: tab) { _, tab in
            if tab == .styles { typing = false }
        }
    }

    @ViewBuilder
    private var fonts: some View {
        if model.fonts.isEmpty {
            Text("No fonts were found on this device.")
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.6))
        } else {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(model.fonts, id: \.name) { font in
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
        close()
    }
}

/// A title's color and size, outline, background, and shadow. Colors,
/// ranges, and the values a part starts at come from Rust.
private struct StylesTab: View {
    let model: EditorModel
    let clip: String
    @State private var part: Part = .text
    @State private var style: TextStyle?

    enum Part: String, CaseIterable {
        case text = "Text"
        case outline = "Outline"
        case background = "Background"
        case shadow = "Shadow"
    }

    private static let colors = textColors()
    private static let ranges = textStyleRanges()

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 18) {
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
                swatches(style)
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

    @ViewBuilder
    private func sliders(_ style: TextStyle) -> some View {
        let ranges = Self.ranges
        switch part {
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
        }
    }

    private func chosenColor(_ style: TextStyle) -> String {
        switch part {
        case .text: style.color.isEmpty ? "#FFFFFF" : style.color
        case .outline: style.outlineColor
        case .background: style.backgroundColor
        case .shadow: style.shadowColor
        }
    }

    /// Pick a color; a part turned on starts at Rust's values.
    private func setColor(_ color: String, _ style: TextStyle) {
        let ranges = Self.ranges
        switch part {
        case .text:
            send(.color(color: color), live: false)
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
private struct StyleSlider: View {
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
