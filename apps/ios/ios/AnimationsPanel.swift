import SwiftUI

/// The selected clip's In and Out animations: a tab for each side, None and
/// that side's presets in a row, and a length slider for the chosen one. A
/// tap sets the side at once, and letting go of the slider sets its length,
/// each as one undo step.
struct AnimationsPanel: View {
    let model: EditorModel
    let close: () -> Void

    @State private var enter = true
    /// The slider's length while a finger is on it.
    @State private var dragging: Double?

    var body: some View {
        PanelFrame(title: "Animations", close: close) {
            VStack(spacing: 12) {
                Picker("Side", selection: $enter) {
                    Text("In").tag(true)
                    Text("Out").tag(false)
                }
                .pickerStyle(.segmented)
                if let animations = model.animations {
                    tiles(chosen: chosen(animations).id)
                    length(animations)
                } else {
                    Text("Select a picture or title to animate it")
                        .font(.footnote)
                        .foregroundStyle(.white.opacity(0.6))
                        .frame(height: 64)
                }
            }
        }
        .onChange(of: enter) { dragging = nil }
    }

    /// The chosen side's id and length; an empty id is None.
    private func chosen(_ animations: ClipAnimations) -> (id: String, us: Int64) {
        enter ? (animations.inId, animations.inUs) : (animations.outId, animations.outUs)
    }

    private func tiles(chosen: String) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                tile(name: "None", preset: nil, chosen: chosen.isEmpty)
                ForEach(model.animationList.filter { $0.enter == enter }, id: \.id) { preset in
                    tile(name: preset.name, preset: preset, chosen: chosen == preset.id)
                }
            }
        }
    }

    private func tile(name: String, preset: AnimationPreset?, chosen: Bool) -> some View {
        Button {
            dragging = nil
            model.setAnimation(enter: enter, id: preset?.id ?? "", durationUs: preset?.durationUs ?? 0)
        } label: {
            AnimationTile(name: name, preset: preset, chosen: chosen)
        }
        .accessibilityLabel(name)
        .accessibilityAddTraits(chosen ? .isSelected : [])
    }

    /// The chosen animation's length, from a tenth of a second to what the
    /// clip has left beside its other side.
    @ViewBuilder
    private func length(_ animations: ClipAnimations) -> some View {
        let side = chosen(animations)
        let other = enter ? animations.outUs : animations.inUs
        let longest = Double(max(animations.lengthUs - other, 100_000)) / 1e6
        if !side.id.isEmpty, longest > 0.1 {
            let shown = dragging ?? Double(side.us) / 1e6
            HStack(spacing: 12) {
                Text(String(format: "%.1fs", shown))
                    .font(.footnote.monospacedDigit())
                    .frame(width: 40, alignment: .leading)
                Slider(
                    value: Binding(get: { shown }, set: { dragging = $0 }),
                    in: 0.1...longest,
                    onEditingChanged: { editing in
                        guard !editing, let seconds = dragging else { return }
                        model.setAnimation(
                            enter: enter, id: side.id, durationUs: Int64((seconds * 10).rounded()) * 100_000)
                        dragging = nil
                    }
                )
                .tint(Palette.accent)
                // A drag the slider never reports ending still lands once
                // the finger rests.
                .task(id: dragging) {
                    guard let seconds = dragging else { return }
                    try? await Task.sleep(for: .milliseconds(400))
                    guard !Task.isCancelled, dragging == seconds else { return }
                    model.setAnimation(enter: enter, id: side.id, durationUs: Int64((seconds * 10).rounded()) * 100_000)
                    dragging = nil
                }
            }
        } else {
            Color.clear.frame(height: 30)
        }
    }
}

/// One animation: a small drawing of its motion, and its name under it.
private struct AnimationTile: View {
    let name: String
    let preset: AnimationPreset?
    let chosen: Bool

    var body: some View {
        VStack(spacing: 6) {
            AnimationGlyph(name: preset?.name ?? "")
                .frame(width: 34, height: 34)
            Text(name)
                .font(.caption2)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .foregroundStyle(chosen ? Palette.accent : Palette.text)
        .frame(width: 64, height: 64)
        .background(
            Color(white: chosen ? 0.2 : 0.15),
            in: RoundedRectangle(cornerRadius: 6)
        )
        .contentShape(Rectangle())
    }
}

/// A fade is a square going from clear to solid, a slide an arrow its way,
/// a zoom two nested squares, and None a cross.
private struct AnimationGlyph: View {
    let name: String

    var body: some View {
        if name.hasPrefix("Fade") {
            RoundedRectangle(cornerRadius: 3)
                .fill(LinearGradient(
                    colors: [.white.opacity(name.hasSuffix("In") ? 0.05 : 0.9),
                             .white.opacity(name.hasSuffix("In") ? 0.9 : 0.05)],
                    startPoint: .leading, endPoint: .trailing
                ))
                .padding(5)
        } else if name.hasPrefix("Slide") {
            Arrow()
                .stroke(style: StrokeStyle(lineWidth: 1.8, lineCap: .round, lineJoin: .round))
                .rotationEffect(turn)
                .padding(6)
        } else if name.hasPrefix("Zoom") {
            ZStack {
                RoundedRectangle(cornerRadius: 2).stroke(lineWidth: 1.5)
                RoundedRectangle(cornerRadius: 2).stroke(lineWidth: 1.5).padding(8)
            }
            .padding(4)
        } else {
            HugeIcon("close", size: 20)
        }
    }

    private var turn: Angle {
        switch name {
        case "Slide Left": .degrees(180)
        case "Slide Up": .degrees(-90)
        case "Slide Down": .degrees(90)
        default: .zero
        }
    }
}

/// An arrow pointing right across its frame.
private struct Arrow: Shape {
    func path(in rect: CGRect) -> Path {
        Path { path in
            path.move(to: CGPoint(x: rect.minX, y: rect.midY))
            path.addLine(to: CGPoint(x: rect.maxX, y: rect.midY))
            path.move(to: CGPoint(x: rect.maxX - rect.width * 0.35, y: rect.minY + rect.height * 0.15))
            path.addLine(to: CGPoint(x: rect.maxX, y: rect.midY))
            path.addLine(to: CGPoint(x: rect.maxX - rect.width * 0.35, y: rect.maxY - rect.height * 0.15))
        }
    }
}
