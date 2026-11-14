import SwiftUI

/// How a selected shape or pen drawing looks: its fill, outline, corners
/// and shadow. Rust reads and changes each; colors, ranges and the values a
/// part starts at come from Rust too.
struct ShapeStylePanel: View {
    let model: EditorModel
    let clip: String
    let close: () -> Void
    @State private var part: Part = .fill
    @State private var style: ShapeStyle?

    enum Part: String, CaseIterable {
        case fill = "Fill"
        case outline = "Outline"
        case corners = "Corners"
        case shadow = "Shadow"
    }

    private static let colors = textColors()
    private static let ranges = shapeStyleRanges()

    /// A line has no inside to fill and no corners to round.
    private var parts: [Part] {
        style?.line == true ? [.outline, .shadow] : Part.allCases
    }

    var body: some View {
        PanelFrame(title: "Style", close: close) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 16) {
                    ForEach(parts, id: \.self) { part in
                        Button { self.part = part } label: {
                            Text(part.rawValue)
                                .font(.footnote.weight(self.part == part ? .semibold : .regular))
                                .foregroundStyle(self.part == part ? .white : .white.opacity(0.55))
                        }
                        .accessibilityAddTraits(self.part == part ? .isSelected : [])
                    }
                }
                if let style {
                    if part == .corners {
                        Spacer().frame(height: 44)
                    } else {
                        swatches(style)
                    }
                    VStack(spacing: 6) { sliders(style) }
                        .frame(height: 84, alignment: .top)
                }
            }
        }
        .onAppear(perform: read)
        .onChange(of: model.edits) { _, _ in read() }
    }

    private func read() {
        style = model.shapeStyle(of: clip)
        if let style, style.line, part == .fill || part == .corners { part = .outline }
    }

    private func send(_ change: ShapeStyleChange, live: Bool) {
        model.styleShape(clip, change, live: live)
    }

    // MARK: Colors

    /// The part's color now; empty while the part is off.
    private func chosen(_ style: ShapeStyle) -> String {
        switch part {
        case .fill: style.fill
        case .outline: style.strokeWidth > 0 ? style.stroke : ""
        case .shadow: style.shadowOpacity > 0 ? style.shadow : ""
        case .corners: ""
        }
    }

    private func swatches(_ style: ShapeStyle) -> some View {
        let chosen = chosen(style)
        return ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                // A line keeps its outline: it would vanish without it.
                if !(style.line && part == .outline) {
                    NoneTile(chosen: chosen.isEmpty, size: 36) { setColor("", style) }
                }
                ForEach(Self.colors, id: \.self) { color in
                    SwatchTile(
                        color: color, chosen: color.caseInsensitiveCompare(chosen) == .orderedSame,
                        size: 36
                    ) { setColor(color, style) }
                }
            }
            .padding(4)
        }
        .frame(height: 44)
    }

    /// Color the part, switching it on where it starts; an empty color
    /// switches it off.
    private func setColor(_ color: String, _ style: ShapeStyle) {
        let ranges = Self.ranges
        switch part {
        case .fill:
            send(.fill(color: color, opacity: style.fillOpacity > 0 ? style.fillOpacity : 1),
                 live: false)
        case .outline:
            let width = color.isEmpty ? 0 : (style.strokeWidth > 0 ? style.strokeWidth : ranges.strokeStart)
            send(.stroke(
                color: color, width: width,
                opacity: style.strokeOpacity > 0 ? style.strokeOpacity : 1,
                style: style.strokeStyle), live: false)
        case .shadow:
            let on = style.shadowOpacity > 0
            send(.shadow(
                color: color,
                distance: on ? style.shadowDistance : ranges.shadowDistanceStart,
                angle: on ? style.shadowAngle : ranges.shadowAngleStart,
                opacity: color.isEmpty ? 0 : (on ? style.shadowOpacity : ranges.shadowOpacityStart)),
                live: false)
        case .corners:
            break
        }
    }

    // MARK: Sliders

    @ViewBuilder
    private func sliders(_ style: ShapeStyle) -> some View {
        let ranges = Self.ranges
        switch part {
        case .fill:
            if !style.fill.isEmpty {
                StyleSlider(label: "Opacity", value: style.fillOpacity, range: 0...1, shown: percent) {
                    opacity, live in
                    send(.fill(color: style.fill, opacity: opacity), live: live)
                }
            }
        case .outline:
            if style.strokeWidth > 0 {
                StyleSlider(
                    label: "Width", value: style.strokeWidth,
                    range: (style.line ? ranges.penMin : 0.5)...ranges.strokeMax, shown: whole
                ) { width, live in
                    send(.stroke(color: style.stroke, width: width, opacity: style.strokeOpacity,
                                 style: style.strokeStyle), live: live)
                }
                lineStyles(style)
            }
        case .corners:
            if style.cornersMax > 0 {
                StyleSlider(
                    label: "Radius", value: style.corners, range: 0...style.cornersMax, shown: whole
                ) { radius, live in
                    send(.corners(radius: radius), live: live)
                }
            }
        case .shadow:
            if style.shadowOpacity > 0 {
                StyleSlider(
                    label: "Distance", value: style.shadowDistance,
                    range: 0...ranges.shadowDistanceMax, shown: whole
                ) { distance, live in
                    send(.shadow(color: style.shadow, distance: distance, angle: style.shadowAngle,
                                 opacity: style.shadowOpacity), live: live)
                }
                StyleSlider(label: "Opacity", value: style.shadowOpacity, range: 0.05...1, shown: percent) {
                    opacity, live in
                    send(.shadow(color: style.shadow, distance: style.shadowDistance,
                                 angle: style.shadowAngle, opacity: opacity), live: live)
                }
            }
        }
    }

    /// Solid, dashed or dotted.
    private func lineStyles(_ style: ShapeStyle) -> some View {
        HStack(spacing: 8) {
            Text("Line")
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.75))
                .frame(width: 72, alignment: .leading)
            ForEach([(Int32(0), "Solid"), (Int32(1), "Dashed"), (Int32(2), "Dotted")], id: \.0) {
                kind, label in
                let on = style.strokeStyle == kind
                Button {
                    send(.stroke(color: style.stroke, width: style.strokeWidth,
                                 opacity: style.strokeOpacity, style: kind), live: false)
                } label: {
                    Text(label)
                        .font(.caption)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .foregroundStyle(on ? .black : .white)
                        .background(on ? Color.white : Color(white: 0.2), in: Capsule())
                }
                .accessibilityAddTraits(on ? .isSelected : [])
            }
            Spacer()
        }
    }
}

private func percent(_ fraction: Double) -> String {
    "\(Int((fraction * 100).rounded()))"
}

private func whole(_ value: Double) -> String {
    "\(Int(value.rounded()))"
}
