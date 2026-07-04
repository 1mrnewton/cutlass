import SwiftUI

/// CapCut-style bottom panel drawn over the editor chrome: grab bar,
/// title row with optional X (cancel) and check (apply), then scrollable
/// content. The preview and timeline stay visible underneath.
struct PanelSheet<Content: View>: View {
    var title: String
    @Binding var height: CGFloat
    var minHeight: CGFloat
    var maxHeight: CGFloat
    /// Picker-style panels (stickers, audio, ...) apply instantly and only
    /// offer the check button.
    var showsCancel = true
    var onCancel: () -> Void = {}
    var onApply: () -> Void
    @ViewBuilder var content: Content

    @State private var heightAnchor: CGFloat?

    var body: some View {
        VStack(spacing: 0) {
            Capsule()
                .fill(Theme.textTertiary)
                .frame(width: 34, height: 4)
                .padding(.top, 8)
                .frame(maxWidth: .infinity)
                .frame(height: 24)
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 2, coordinateSpace: .global)
                        .onChanged { value in
                            let anchor = heightAnchor ?? height
                            heightAnchor = anchor
                            height = min(max(anchor - value.translation.height, minHeight), maxHeight)
                        }
                        .onEnded { _ in
                            heightAnchor = nil
                        }
                )

            ZStack {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white)

                HStack {
                    if showsCancel {
                        Button(action: onCancel) {
                            Image(systemName: "xmark")
                                .font(.system(size: 15, weight: .semibold))
                                .foregroundStyle(Theme.textSecondary)
                                .frame(width: 34, height: 34)
                        }
                        .buttonStyle(.plain)
                    }

                    Spacer()

                    Button(action: onApply) {
                        Image(systemName: "checkmark")
                            .font(.system(size: 15, weight: .bold))
                            .foregroundStyle(.white)
                            .frame(width: 34, height: 34)
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 10)
            }
            .padding(.top, 2)

            ScrollView {
                content
                    .padding(.top, 6)
                    .padding(.bottom, 10)
            }
        }
        .frame(height: height)
        .frame(maxWidth: .infinity)
        .background {
            UnevenRoundedRectangle(
                cornerRadii: RectangleCornerRadii(topLeading: 16, topTrailing: 16),
                style: .continuous
            )
            .fill(Theme.surface)
            .shadow(color: .black.opacity(0.45), radius: 16, y: -4)
            .ignoresSafeArea(edges: .bottom)
        }
        .accessibilityIdentifier("editorPanel")
    }
}

/// Labeled slider with a live value bubble, used across property panels.
struct PanelSlider: View {
    var label: String
    @Binding var value: Double
    var range: ClosedRange<Double>
    /// Renders the bubble text; defaults to a percent-style integer.
    var format: (Double) -> String = { "\(Int(($0 * 100).rounded()))" }

    var body: some View {
        HStack(spacing: 12) {
            Text(label)
                .font(.footnote)
                .foregroundStyle(Theme.textSecondary)
                .frame(width: 82, alignment: .leading)

            Slider(value: $value, in: range)
                .tint(Theme.accent)

            Text(format(value))
                .font(.footnote.weight(.semibold).monospacedDigit())
                .foregroundStyle(.white)
                .frame(width: 44, alignment: .trailing)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 6)
    }
}

/// Square preset tile with a selection ring (effects, filters, masks, ...).
struct PresetTile: View {
    var name: String
    var isSelected: Bool
    var art: MockArt?
    var symbol: String?
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 6) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Theme.surfaceElevated)
                    if let art {
                        MockArtView(art: art, symbolSize: 0)
                            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                    }
                    if let symbol {
                        Image(systemName: symbol)
                            .font(.system(size: 20, weight: .medium))
                            .foregroundStyle(.white)
                    }
                }
                .frame(width: 58, height: 58)
                .overlay {
                    if isSelected {
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .strokeBorder(Theme.accent, lineWidth: 2.5)
                    }
                }

                Text(name)
                    .font(.system(size: 10.5))
                    .foregroundStyle(isSelected ? .white : Theme.textSecondary)
                    .lineLimit(1)
                    .frame(width: 62)
            }
        }
        .buttonStyle(.plain)
    }
}

/// Underlined tab row used inside tabbed panels (text tools, audio, ...).
struct PanelTabs: View {
    var tabs: [String]
    @Binding var selection: Int

    var body: some View {
        HStack(spacing: 22) {
            ForEach(tabs.indices, id: \.self) { index in
                Button {
                    selection = index
                } label: {
                    VStack(spacing: 5) {
                        Text(tabs[index])
                            .font(.footnote.weight(selection == index ? .semibold : .regular))
                            .foregroundStyle(selection == index ? .white : Theme.textTertiary)
                        Capsule()
                            .fill(selection == index ? .white : .clear)
                            .frame(width: 18, height: 2)
                    }
                }
                .buttonStyle(.plain)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 4)
    }
}

#Preview {
    @Previewable @State var height: CGFloat = 320
    VStack {
        Spacer()
        PanelSheet(title: "Aspect ratio", height: $height, minHeight: 280, maxHeight: 500, onCancel: {}, onApply: {}) {
            PanelSlider(label: "Intensity", value: .constant(0.8), range: 0...1)
            HStack(spacing: 10) {
                PresetTile(name: "Vivid", isSelected: true, art: MockData.tileArt(for: "Vivid"), symbol: nil) {}
                PresetTile(name: "None", isSelected: false, art: nil, symbol: "slash.circle") {}
            }
        }
    }
    .background(Theme.background)
}
