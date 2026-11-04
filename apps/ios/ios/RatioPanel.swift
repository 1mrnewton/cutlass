import SwiftUI

/// The canvas ratios in a row, each drawn at its own shape. A tap sets the
/// ratio at once, as one undo step, and the preview takes the new frame.
struct RatioPanel: View {
    let model: EditorModel
    let close: () -> Void

    var body: some View {
        PanelFrame(title: "Ratio", close: close) {
            ScrollViewReader { scroller in
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(model.ratios, id: \.ratio) { ratio in
                            Button { model.setRatio(ratio.ratio) } label: {
                                RatioTile(ratio: ratio)
                            }
                            .id(ratio.ratio)
                            .accessibilityLabel(ratio.label)
                            .accessibilityAddTraits(ratio.chosen ? .isSelected : [])
                        }
                    }
                }
                .onAppear {
                    if let chosen = model.ratios.first(where: \.chosen) {
                        scroller.scrollTo(chosen.ratio, anchor: .center)
                    }
                }
            }
        }
    }
}

/// One ratio: its frame outlined inside a square, and its name under it.
private struct RatioTile: View {
    let ratio: CanvasRatio

    private static let box: CGFloat = 34

    var body: some View {
        VStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 3)
                .stroke(lineWidth: 1.5)
                .frame(width: frame.width, height: frame.height)
                .frame(width: Self.box, height: Self.box)
            Text(ratio.label)
                .font(.caption2)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .foregroundStyle(ratio.chosen ? Palette.accent : Palette.text)
        .frame(width: 60, height: 64)
        .background(
            Color(white: ratio.chosen ? 0.2 : 0.15),
            in: RoundedRectangle(cornerRadius: 6)
        )
        .contentShape(Rectangle())
    }

    /// The ratio's frame fitted in the box, never thinner than a few points.
    private var frame: CGSize {
        let width = CGFloat(max(ratio.width, 1))
        let height = CGFloat(max(ratio.height, 1))
        let scale = Self.box / max(width, height)
        return CGSize(width: max(width * scale, 6), height: max(height * scale, 6))
    }
}
