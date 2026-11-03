import SwiftUI

/// A tool panel over the tool bar: a title, the control, and a check that
/// closes it. A value is sent once the finger lifts, so one drag is one
/// undo step.
struct PanelFrame<Content: View>: View {
    let title: String
    let close: () -> Void
    @ViewBuilder let content: Content

    var body: some View {
        VStack(spacing: 14) {
            ZStack {
                Text(title).font(.subheadline.weight(.semibold))
                HStack {
                    Spacer()
                    Button(action: close) {
                        HugeIcon("done", size: 20)
                            .frame(width: 40, height: 32)
                            .contentShape(Rectangle())
                    }
                    .accessibilityLabel("Done")
                }
            }
            content
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 16)
        .padding(.top, 10)
        .padding(.bottom, 16)
        .background(Color(white: 0.11))
    }
}

/// Clip gain from silence to 1000%, unity in the middle of the slider.
struct VolumePanel: View {
    let model: EditorModel
    let clip: Clip
    let close: () -> Void
    @State private var position: Double

    init(model: EditorModel, clip: Clip, close: @escaping () -> Void) {
        self.model = model
        self.clip = clip
        self.close = close
        _position = State(initialValue: volumePosition(volume: clip.volume))
    }

    var body: some View {
        PanelFrame(title: "Volume", close: close) {
            HStack(spacing: 12) {
                Text("0").font(.caption.monospacedDigit()).foregroundStyle(.white.opacity(0.5))
                Slider(value: $position, in: 0...1) { editing in
                    if !editing { model.setVolume(clip.id, volumeAt(position: position)) }
                }
                .tint(Palette.accent)
                .accessibilityLabel("Volume")
                .accessibilityValue(percent)
                Text("1000").font(.caption.monospacedDigit()).foregroundStyle(.white.opacity(0.5))
            }
            Text(percent)
                .font(.footnote.monospacedDigit())
                .foregroundStyle(.white.opacity(0.75))
        }
        .onChange(of: clip.volume) { _, volume in position = volumePosition(volume: volume) }
    }

    private var percent: String {
        "\(Int((volumeAt(position: position) * 100).rounded()))%"
    }
}

/// Constant speed from 0.1x to 100x on a scale whose marks sit the same
/// distance apart. The clip's new length shows while dragging.
struct SpeedPanel: View {
    let model: EditorModel
    let clip: Clip
    let close: () -> Void
    @State private var position: Double
    /// The clip's length when the panel opened.
    @State private var before: Int64

    static let marks: [Double] = [0.1, 1, 2, 5, 10, 100]
    /// Space at each end of the scale, so its end labels fit.
    private static let inset: CGFloat = 20

    init(model: EditorModel, clip: Clip, close: @escaping () -> Void) {
        self.model = model
        self.clip = clip
        self.close = close
        _position = State(initialValue: speedPosition(speed: clip.speed))
        _before = State(initialValue: clip.durationUs)
    }

    var body: some View {
        PanelFrame(title: "Speed", close: close) {
            HStack(spacing: 6) {
                Text("Duration")
                    .foregroundStyle(.white.opacity(0.55))
                Text(seconds(before))
                    .foregroundStyle(.white.opacity(0.55))
                HugeIcon("arrowRight", size: 14)
                    .foregroundStyle(Palette.accent)
                Text(seconds(newLength))
                Spacer()
                Text(Self.label(speedAt(position: position)))
                    .font(.footnote.weight(.semibold).monospacedDigit())
            }
            .font(.footnote.monospacedDigit())
            scale
        }
        .onChange(of: clip.speed) { _, speed in position = speedPosition(speed: speed) }
    }

    /// The source length stays; the timeline length follows the speed.
    private var newLength: Int64 {
        let source = Double(clip.durationUs) * clip.speed
        return Int64(source / max(speedAt(position: position), 0.01))
    }

    private var scale: some View {
        GeometryReader { geometry in
            let width = geometry.size.width - 2 * Self.inset
            let thumbX = Self.inset + width * position
            ZStack(alignment: .topLeading) {
                Canvas { context, size in
                    let minors = 35
                    for index in 0...minors {
                        let fraction = Double(index) / Double(minors)
                        let major = index % 7 == 0
                        let x = Self.inset + width * fraction
                        var line = Path()
                        line.move(to: CGPoint(x: x, y: major ? 6 : 10))
                        line.addLine(to: CGPoint(x: x, y: major ? 26 : 22))
                        context.stroke(
                            line, with: .color(.white.opacity(major ? 0.8 : 0.35)), lineWidth: 1.5)
                    }
                    for (index, mark) in Self.marks.enumerated() {
                        let x = Self.inset + width * Double(index) / Double(Self.marks.count - 1)
                        context.draw(
                            Text(Self.label(mark)).font(.caption.monospacedDigit())
                                .foregroundStyle(.white.opacity(0.75)),
                            at: CGPoint(x: x, y: size.height - 8))
                    }
                }
                Circle()
                    .strokeBorder(Palette.accent, lineWidth: 3)
                    .background(Circle().fill(Color(white: 0.11)))
                    .frame(width: 24, height: 24)
                    .offset(x: thumbX - 12, y: 4)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { drag in
                        position = Self.snapped(
                            min(max((drag.location.x - Self.inset) / width, 0), 1), width: width)
                    }
                    .onEnded { _ in
                        model.setSpeed(clip.id, speedAt(position: position))
                    }
            )
        }
        .frame(height: 58)
        .accessibilityElement()
        .accessibilityLabel("Speed")
        .accessibilityValue(Self.label(speedAt(position: position)))
        .accessibilityAdjustableAction { direction in
            let step = 0.2 / 7
            position = min(max(position + (direction == .increment ? step : -step), 0), 1)
            model.setSpeed(clip.id, speedAt(position: position))
        }
    }

    /// A position within 8 points of a mark lands on the mark.
    private static func snapped(_ position: Double, width: CGFloat) -> Double {
        let gaps = Double(marks.count - 1)
        let nearest = (position * gaps).rounded() / gaps
        return abs(nearest - position) * width < 8 ? nearest : position
    }

    private func seconds(_ micros: Int64) -> String {
        String(format: "%.1fs", Double(micros) / 1_000_000)
    }

    static func label(_ speed: Double) -> String {
        let text = speed == speed.rounded() ? String(Int(speed)) : String(format: "%g", speed)
        return text + "×"
    }
}
