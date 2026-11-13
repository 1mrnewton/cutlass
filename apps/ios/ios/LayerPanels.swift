import SwiftUI

/// Stickers and shapes: a tab for each, a grid of tiles. A tap puts the
/// sticker or shape at the playhead on a sticker row and selects it; the
/// panel stays so another can follow.
struct StickersPanel: View {
    let model: EditorModel
    let close: () -> Void
    @State private var shapes = false

    private let columns = Array(repeating: GridItem(.flexible(), spacing: 10), count: 5)

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 18) {
                tab("Stickers", on: !shapes) { shapes = false }
                tab("Shapes", on: shapes) { shapes = true }
                Spacer()
                Button(action: close) {
                    HugeIcon("done", size: 20)
                        .frame(width: 40, height: 32)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Done")
            }
            ScrollView {
                LazyVGrid(columns: columns, spacing: 12) {
                    if shapes {
                        ForEach(model.shapeList, id: \.id) { shape in
                            Button { model.addShape(shape.id) } label: {
                                ShapeTile(name: shape.name)
                                    .padding(12)
                                    .aspectRatio(1, contentMode: .fit)
                                    .background(Color(white: 0.18), in: RoundedRectangle(cornerRadius: 8))
                            }
                            .accessibilityLabel(shape.name)
                        }
                    } else {
                        ForEach(model.stickerList, id: \.id) { sticker in
                            Button { model.addSticker(sticker.id) } label: {
                                Group {
                                    if let poster = model.stickerPoster(sticker.id) {
                                        Image(decorative: poster, scale: 1).resizable()
                                    } else {
                                        Color(white: 0.18)
                                    }
                                }
                                .aspectRatio(1, contentMode: .fit)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                                .overlay(alignment: .bottomTrailing) {
                                    if sticker.animated {
                                        Image(systemName: "play.fill")
                                            .font(.system(size: 7))
                                            .padding(3)
                                            .background(.black.opacity(0.5), in: Circle())
                                            .padding(3)
                                    }
                                }
                            }
                            .accessibilityLabel(sticker.name)
                        }
                    }
                }
            }
            .frame(height: 230)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 16)
        .padding(.top, 10)
        .padding(.bottom, 12)
        .background(Color(white: 0.11))
    }

    private func tab(_ label: String, on: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label)
                .font(.subheadline.weight(on ? .semibold : .regular))
                .foregroundStyle(on ? .white : .white.opacity(0.5))
        }
        .accessibilityAddTraits(on ? .isSelected : [])
    }
}

/// A shape's tile, drawn by its name.
private struct ShapeTile: View {
    let name: String

    var body: some View {
        GeometryReader { geometry in
            let size = geometry.size
            path(in: CGRect(origin: .zero, size: size))
                .fill(name == "Line" || name == "Arrow" ? .clear : .white)
                .overlay {
                    path(in: CGRect(origin: .zero, size: size))
                        .stroke(.white, style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
                }
        }
    }

    private func path(in rect: CGRect) -> Path {
        Path { path in
            let (w, h) = (rect.width, rect.height)
            switch name {
            case "Circle":
                path.addEllipse(in: rect)
            case "Triangle":
                path.move(to: CGPoint(x: w / 2, y: 0))
                path.addLine(to: CGPoint(x: w, y: h))
                path.addLine(to: CGPoint(x: 0, y: h))
                path.closeSubpath()
            case "Parallelogram":
                path.move(to: CGPoint(x: w * 0.25, y: h * 0.2))
                path.addLine(to: CGPoint(x: w, y: h * 0.2))
                path.addLine(to: CGPoint(x: w * 0.75, y: h * 0.8))
                path.addLine(to: CGPoint(x: 0, y: h * 0.8))
                path.closeSubpath()
            case "Trapezoid":
                path.move(to: CGPoint(x: w * 0.25, y: h * 0.2))
                path.addLine(to: CGPoint(x: w * 0.75, y: h * 0.2))
                path.addLine(to: CGPoint(x: w, y: h * 0.8))
                path.addLine(to: CGPoint(x: 0, y: h * 0.8))
                path.closeSubpath()
            case "Line":
                path.move(to: CGPoint(x: 0, y: h / 2))
                path.addLine(to: CGPoint(x: w, y: h / 2))
            case "Arrow":
                path.move(to: CGPoint(x: 0, y: h / 2))
                path.addLine(to: CGPoint(x: w, y: h / 2))
                path.move(to: CGPoint(x: w * 0.65, y: h * 0.25))
                path.addLine(to: CGPoint(x: w, y: h / 2))
                path.addLine(to: CGPoint(x: w * 0.65, y: h * 0.75))
            default:
                path.addRect(rect.insetBy(dx: 0, dy: h * 0.12))
            }
        }
    }
}

/// Draw: each finger stroke on the preview becomes a smooth line at the
/// playhead, one undo step each. The panel only says how and closes.
struct DrawPanel: View {
    let close: () -> Void

    var body: some View {
        PanelFrame(title: "Draw", close: close) {
            Text("Draw on the picture with one finger. Each stroke is its own layer.")
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.7))
                .multilineTextAlignment(.center)
                .frame(maxWidth: .infinity)
        }
    }
}

/// The stroke a finger draws on the preview, sent in canvas pixels when
/// the finger lifts.
struct DrawOverlay: View {
    let model: EditorModel
    @State private var stroke: [CGPoint] = []

    var body: some View {
        GeometryReader { geometry in
            let canvas = model.canvasSize ?? CGSize(width: 16, height: 9)
            let scale = min(geometry.size.width / canvas.width, geometry.size.height / canvas.height)
            let origin = CGPoint(
                x: (geometry.size.width - canvas.width * scale) / 2,
                y: (geometry.size.height - canvas.height * scale) / 2
            )
            ZStack {
                Color.clear.contentShape(Rectangle())
                Path { path in
                    guard let first = stroke.first else { return }
                    path.move(to: first)
                    stroke.dropFirst().forEach { path.addLine(to: $0) }
                }
                .stroke(.white, style: StrokeStyle(lineWidth: 4, lineCap: .round, lineJoin: .round))
                .shadow(color: .black.opacity(0.4), radius: 1)
            }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { stroke.append($0.location) }
                    .onEnded { _ in
                        let points = stroke.map {
                            CGPoint(x: ($0.x - origin.x) / scale, y: ($0.y - origin.y) / scale)
                        }
                        stroke = []
                        if points.count > 1 { model.drawStroke(points) }
                    }
            )
        }
        .accessibilityLabel("Drawing area")
    }
}
