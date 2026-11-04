import SwiftUI

/// The keyframe curves in a row, each drawn as its shape. A tap eases the
/// span the playhead is in, as one undo step; None makes it straight again.
struct GraphsPanel: View {
    let model: EditorModel
    let close: () -> Void

    /// Rust samples every curve once; they never change.
    private static let curves = graphCurves()

    var body: some View {
        PanelFrame(title: "Graphs", close: close) {
            if let keys = model.keys, keys.inSpan {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        tile(name: "None", curve: nil, chosen: keys.graph.isEmpty)
                        ForEach(Self.curves, id: \.name) { curve in
                            tile(name: curve.name, curve: curve, chosen: keys.graph == curve.name)
                        }
                    }
                }
            } else {
                Text("Put the playhead between two keyframes")
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.6))
                    .frame(height: 64)
            }
        }
    }

    private func tile(name: String, curve: GraphCurve?, chosen: Bool) -> some View {
        Button { model.easeKeyframes(curve?.name ?? "") } label: {
            GraphTile(name: name, curve: curve, chosen: chosen)
        }
        .accessibilityLabel(name)
        .accessibilityAddTraits(chosen ? .isSelected : [])
    }
}

/// One curve in a square, its name under it. No curve draws a slash.
private struct GraphTile: View {
    let name: String
    let curve: GraphCurve?
    let chosen: Bool

    private static let box: CGFloat = 34

    var body: some View {
        VStack(spacing: 6) {
            Group {
                if curve == nil {
                    HugeIcon("close", size: 20)
                } else {
                    shape.stroke(style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))
                }
            }
            .frame(width: Self.box, height: Self.box)
            Text(name)
                .font(.caption2)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .foregroundStyle(chosen ? Palette.accent : Palette.text)
        .frame(width: 60, height: 64)
        .background(
            Color(white: chosen ? 0.2 : 0.15),
            in: RoundedRectangle(cornerRadius: 6)
        )
        .contentShape(Rectangle())
    }

    /// The curve in the box, y up. Rebounds leave the unit square, so the
    /// box fits the curve's own height.
    private var shape: Path {
        Path { path in
            guard let curve, !curve.xs.isEmpty else { return }
            let low = min(curve.ys.min() ?? 0, 0)
            let high = max(curve.ys.max() ?? 1, 1)
            let side = Self.box
            func point(_ index: Int) -> CGPoint {
                CGPoint(
                    x: curve.xs[index] * side,
                    y: side - (curve.ys[index] - low) / (high - low) * side
                )
            }
            path.move(to: point(0))
            for index in 1..<min(curve.xs.count, curve.ys.count) {
                path.addLine(to: point(index))
            }
        }
    }
}
