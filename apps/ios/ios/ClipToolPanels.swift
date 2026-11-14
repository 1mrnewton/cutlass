import SwiftUI

/// Crop the selected video to its middle at a ratio, or show the whole
/// frame. Each tap is one undo step; Rust works out the crop.
struct CropPanel: View {
    let model: EditorModel
    let close: () -> Void
    private let ratios = cropRatios()

    var body: some View {
        PanelFrame(title: "Crop", close: close) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    tile("", label: "Original")
                    ForEach(ratios, id: \.self) { ratio in
                        tile(ratio, label: ratio)
                    }
                }
                .padding(.horizontal, 2)
            }
        }
    }

    private func tile(_ ratio: String, label: String) -> some View {
        let chosen = (model.clipState?.crop ?? "") == ratio
        return Button {
            model.setCrop(ratio)
        } label: {
            VStack(spacing: 6) {
                RoundedRectangle(cornerRadius: 3)
                    .strokeBorder(chosen ? Palette.accent : .white.opacity(0.7), lineWidth: 1.5)
                    .frame(width: shape(ratio).width, height: shape(ratio).height)
                    .frame(width: 44, height: 44)
                Text(label).font(.caption2)
            }
            .frame(width: 60, height: 70)
            .background(chosen ? Color.white.opacity(0.08) : .clear, in: RoundedRectangle(cornerRadius: 8))
        }
        .accessibilityLabel(ratio.isEmpty ? "Original" : "Crop to \(ratio)")
        .accessibilityAddTraits(chosen ? .isSelected : [])
    }

    /// The tile's outline, the ratio's shape inside 36 points.
    private func shape(_ ratio: String) -> CGSize {
        let parts = ratio.split(separator: ":").compactMap { Double($0) }
        guard parts.count == 2, parts[0] > 0, parts[1] > 0 else {
            return CGSize(width: 36, height: 28)
        }
        let wide = parts[0] / parts[1]
        return wide >= 1 ? CGSize(width: 36, height: 36 / wide) : CGSize(width: 36 * wide, height: 36)
    }
}

/// How solid the selected picture is, sent when the finger lifts.
struct OpacityPanel: View {
    let model: EditorModel
    let close: () -> Void
    @State private var value: Double

    init(model: EditorModel, close: @escaping () -> Void) {
        self.model = model
        self.close = close
        _value = State(initialValue: model.clipState?.opacity ?? 1)
    }

    var body: some View {
        PanelFrame(title: "Opacity", close: close) {
            HStack(spacing: 12) {
                Slider(value: $value, in: 0...1) { editing in
                    if !editing { model.setOpacity(value) }
                }
                .tint(Palette.accent)
                .accessibilityLabel("Opacity")
                .accessibilityValue(percent)
                Text(percent)
                    .font(.footnote.monospacedDigit())
                    .frame(width: 44, alignment: .trailing)
            }
        }
        .onChange(of: model.clipState?.opacity) { _, opacity in
            if let opacity { value = opacity }
        }
    }

    private var percent: String { "\(Int((value * 100).rounded()))%" }
}

/// Fade the selected clip in and out. Each slider is sent when the finger
/// lifts; the two together can be no longer than the clip.
struct FadePanel: View {
    let model: EditorModel
    let clip: Clip
    let close: () -> Void
    @State private var fadeIn: Double
    @State private var fadeOut: Double

    init(model: EditorModel, clip: Clip, close: @escaping () -> Void) {
        self.model = model
        self.clip = clip
        self.close = close
        _fadeIn = State(initialValue: Double(model.clipState?.fadeInUs ?? 0) / 1_000_000)
        _fadeOut = State(initialValue: Double(model.clipState?.fadeOutUs ?? 0) / 1_000_000)
    }

    /// Each fade reaches half the clip, and no more than ten seconds.
    private var longest: Double { min(Double(clip.durationUs) / 2_000_000, 10) }

    var body: some View {
        PanelFrame(title: "Fade", close: close) {
            row("Fade in", value: $fadeIn)
            row("Fade out", value: $fadeOut)
        }
        .onChange(of: model.clipState?.fadeInUs) { _, micros in
            if let micros { fadeIn = Double(micros) / 1_000_000 }
        }
        .onChange(of: model.clipState?.fadeOutUs) { _, micros in
            if let micros { fadeOut = Double(micros) / 1_000_000 }
        }
    }

    private func row(_ title: String, value: Binding<Double>) -> some View {
        HStack(spacing: 12) {
            Text(title)
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.7))
                .frame(width: 64, alignment: .leading)
            Slider(value: value, in: 0...max(longest, 0.1)) { editing in
                if !editing {
                    model.setFade(
                        inUs: Int64((fadeIn * 1_000_000).rounded()),
                        outUs: Int64((fadeOut * 1_000_000).rounded()))
                }
            }
            .tint(Palette.accent)
            .accessibilityLabel(title)
            .accessibilityValue(String(format: "%.1f seconds", value.wrappedValue))
            Text(String(format: "%.1fs", value.wrappedValue))
                .font(.footnote.monospacedDigit())
                .frame(width: 40, alignment: .trailing)
        }
    }
}
