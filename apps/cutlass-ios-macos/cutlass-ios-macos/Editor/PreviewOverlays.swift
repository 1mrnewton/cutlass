import SwiftUI

/// Renders every overlay clip visible at the playhead inside the preview
/// frame. The selected overlay gets a dashed manipulation frame: drag to
/// move, corner buttons to delete/duplicate, and a grip that scale-rotates.
struct PreviewOverlayLayer: View {
    var state: EditorState
    var onEditText: (UUID) -> Void = { _ in }

    var body: some View {
        GeometryReader { geo in
            ForEach(state.overlays(at: state.playhead)) { overlay in
                ManipulableOverlay(
                    state: state,
                    clip: overlay,
                    frameSize: geo.size,
                    onEditText: onEditText
                )
            }
        }
        .coordinateSpace(name: "previewFrame")
        .clipped()
    }
}

/// One overlay on the canvas, with selection chrome and gestures.
private struct ManipulableOverlay: View {
    var state: EditorState
    var clip: MockOverlayClip
    var frameSize: CGSize
    var onEditText: (UUID) -> Void

    /// (posX, posY) captured when a move drag starts.
    @State private var moveAnchor: (x: Double, y: Double)?
    /// (scale, rotation, grip vector) captured when a grip drag starts.
    @State private var gripAnchor: (scale: Double, rotation: Double, vector: CGSize)?

    private var isSelected: Bool {
        state.selection == .overlay(clip.id)
    }

    var body: some View {
        OverlayContentView(clip: clip, frameSize: frameSize)
            .padding(9)
            .overlay {
                if isSelected {
                    manipulationChrome
                }
            }
            .scaleEffect(clip.scale)
            .rotationEffect(.degrees(clip.rotationDegrees))
            .position(x: clip.posX * frameSize.width, y: clip.posY * frameSize.height)
            .onTapGesture(count: 2) {
                if clip.kind == .text {
                    state.selection = .overlay(clip.id)
                    onEditText(clip.id)
                }
            }
            .onTapGesture {
                state.selection = isSelected ? nil : .overlay(clip.id)
            }
            .gesture(isSelected ? moveGesture : nil)
    }

    // MARK: Selection chrome

    private var manipulationChrome: some View {
        RoundedRectangle(cornerRadius: 6, style: .continuous)
            .strokeBorder(
                .white,
                style: StrokeStyle(lineWidth: 1.5 / clip.scale, dash: [5 / clip.scale, 4 / clip.scale])
            )
            .overlay(alignment: .topLeading) {
                cornerButton("xmark") { state.deleteSelected() }
                    .offset(x: -11, y: -11)
            }
            .overlay(alignment: .topTrailing) {
                cornerButton("plus.square.on.square") { state.duplicateSelected() }
                    .offset(x: 11, y: -11)
            }
            .overlay(alignment: .bottomTrailing) {
                scaleRotateGrip
                    .offset(x: 11, y: 11)
            }
    }

    /// Corner buttons counter-scale so they stay finger-sized at any zoom.
    private func cornerButton(_ symbol: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Circle()
                .fill(.white)
                .frame(width: 22, height: 22)
                .overlay {
                    Image(systemName: symbol)
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.black)
                }
                .shadow(color: .black.opacity(0.35), radius: 3)
        }
        .buttonStyle(.plain)
        .scaleEffect(1 / clip.scale)
    }

    private var scaleRotateGrip: some View {
        Circle()
            .fill(.white)
            .frame(width: 22, height: 22)
            .overlay {
                Image(systemName: "arrow.up.left.and.arrow.down.right")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.black)
            }
            .shadow(color: .black.opacity(0.35), radius: 3)
            .scaleEffect(1 / clip.scale)
            .highPriorityGesture(gripGesture)
    }

    // MARK: Gestures

    private var moveGesture: some Gesture {
        DragGesture(minimumDistance: 2)
            .onChanged { value in
                let anchor = moveAnchor ?? (clip.posX, clip.posY)
                moveAnchor = anchor
                state.dragOverlay(
                    clip.id,
                    anchorX: anchor.x,
                    anchorY: anchor.y,
                    deltaX: value.translation.width / frameSize.width,
                    deltaY: value.translation.height / frameSize.height
                )
            }
            .onEnded { _ in
                moveAnchor = nil
                state.endGesture()
            }
    }

    /// Dragging the grip away from the overlay's center scales it; circling
    /// the center rotates it. Vectors are measured from the overlay's center
    /// in the preview-frame coordinate space, so the math is exact.
    private var gripGesture: some Gesture {
        DragGesture(minimumDistance: 1, coordinateSpace: .named("previewFrame"))
            .onChanged { value in
                let center = CGPoint(
                    x: clip.posX * frameSize.width,
                    y: clip.posY * frameSize.height
                )

                let anchor: (scale: Double, rotation: Double, vector: CGSize)
                if let gripAnchor {
                    anchor = gripAnchor
                } else {
                    let vector = CGSize(
                        width: value.startLocation.x - center.x,
                        height: value.startLocation.y - center.y
                    )
                    anchor = (clip.scale, clip.rotationDegrees, vector)
                    gripAnchor = anchor
                }

                let current = CGSize(
                    width: value.location.x - center.x,
                    height: value.location.y - center.y
                )
                let startLength = max(hypot(anchor.vector.width, anchor.vector.height), 8)
                let currentLength = max(hypot(current.width, current.height), 4)
                let startAngle = atan2(anchor.vector.height, anchor.vector.width)
                let currentAngle = atan2(current.height, current.width)

                state.transformOverlay(
                    clip.id,
                    anchorScale: anchor.scale,
                    anchorRotation: anchor.rotation,
                    scaleFactor: currentLength / startLength,
                    rotationDelta: (currentAngle - startAngle) * 180 / .pi
                )
            }
            .onEnded { _ in
                gripAnchor = nil
                state.endGesture()
            }
    }
}

/// The visual body of a single overlay clip (text, sticker, or PiP).
struct OverlayContentView: View {
    var clip: MockOverlayClip
    var frameSize: CGSize

    var body: some View {
        switch clip.kind {
        case .text:
            styledText
        case .sticker:
            Image(systemName: clip.symbol ?? "questionmark")
                .font(.system(size: 46))
                .foregroundStyle(.white)
                .shadow(color: .black.opacity(0.35), radius: 4, y: 2)
        case .pip:
            if let art = clip.art {
                MockArtView(art: art, symbolSize: 22)
                    .frame(width: frameSize.width * 0.52, height: frameSize.width * 0.39)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .overlay {
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .strokeBorder(.white.opacity(0.7), lineWidth: 1.5)
                    }
                    .shadow(color: .black.opacity(0.4), radius: 8, y: 4)
            }
        }
    }

    @ViewBuilder
    private var styledText: some View {
        let string = clip.text.isEmpty ? "Enter text" : clip.text
        let base = Text(string)
            .font(TextStyling.font(named: clip.fontName, size: 28))
            .multilineTextAlignment(.center)

        switch clip.textEffect {
        case "Neon":
            base.foregroundStyle(clip.textColor)
                .shadow(color: Theme.accent, radius: 6)
                .shadow(color: Theme.accent.opacity(0.8), radius: 14)
        case "Shadow":
            base.foregroundStyle(clip.textColor)
                .shadow(color: .black.opacity(0.85), radius: 3, x: 0, y: 3)
        case "Outline":
            base.foregroundStyle(clip.textColor)
                .shadow(color: .black, radius: 1, x: 1.2, y: 1.2)
                .shadow(color: .black, radius: 1, x: -1.2, y: -1.2)
                .shadow(color: .black, radius: 1, x: 1.2, y: -1.2)
                .shadow(color: .black, radius: 1, x: -1.2, y: 1.2)
        case "Glow":
            base.foregroundStyle(clip.textColor)
                .shadow(color: .white.opacity(0.9), radius: 8)
        case "Retro":
            base.foregroundStyle(clip.textColor)
                .shadow(color: Color(hex: 0xF97316), radius: 0, x: 3, y: 3)
        case "Gradient":
            base.foregroundStyle(
                LinearGradient(
                    colors: [Color(hex: 0xF472B6), Color(hex: 0x60A5FA)],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
        case "Chrome":
            base.foregroundStyle(
                LinearGradient(
                    colors: [.white, Color(hex: 0x94A3B8), .white],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
        default:
            base.foregroundStyle(clip.text.isEmpty ? clip.textColor.opacity(0.55) : clip.textColor)
        }
    }
}
