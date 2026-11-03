import SwiftUI

/// The selected picture or title's box on the preview, and the fingers
/// that move, pinch, and turn it. A tap selects the clip under it; a tap on
/// the selected title opens its words. Rust does the geometry and the
/// snapping; this only turns points into canvas pixels.
struct StageOverlay: View {
    let model: EditorModel
    /// Open the words of a title.
    let editText: (String) -> Void

    /// Each part of the gesture so far, kept while any finger is down so a
    /// drag that ends during a pinch does not jump.
    @State private var shift: CGSize = .zero
    @State private var pinch: Double = 1
    @State private var turn: Double = 0
    @State private var dragged: CGSize = .zero
    @State private var pinched: Double = 1
    @State private var turned: Double = 0
    @State private var dragging = false
    @State private var twisting = false
    @State private var live = false
    /// A finger of the drag or the pinch is down. Unlike their end
    /// handlers, this also resets when a second finger cancels the drag.
    @GestureState private var dragDown = false
    @GestureState private var twistDown = false
    @GestureState private var cornerDown = false
    /// Where a corner drag started, from the box centre, in view points.
    @State private var cornerStart: CGSize?

    var body: some View {
        GeometryReader { geometry in
            let canvas = model.canvasSize ?? CGSize(width: 16, height: 9)
            let rect = fitted(canvas, in: geometry.size)
            let scale = rect.width / canvas.width
            ZStack(alignment: .topLeading) {
                Color.clear.contentShape(Rectangle())
                if let stage = shown {
                    if live { guides(stage, rect: rect) }
                    box(stage, rect: rect, scale: scale)
                }
            }
            .coordinateSpace(name: StageOverlay.space)
            .gesture(tap(rect: rect, scale: scale))
            .simultaneousGesture(drag(rect: rect, scale: scale))
            .simultaneousGesture(twist(scale: scale))
        }
        .sensoryFeedback(.impact(weight: .light), trigger: snapped) { _, now in now }
        .onChange(of: dragDown) { _, down in
            if !down { endDrag() }
        }
        .onChange(of: twistDown) { _, down in
            if !down { endTwist() }
        }
        .onChange(of: cornerDown) { _, down in
            if !down { endCorner() }
        }
    }

    /// The selected clip's box, while the clip is on screen at the playhead.
    private var shown: Stage? {
        model.selection?.covers(model.playheadUs) == true ? model.stage : nil
    }

    /// Both centre lines together, so a haptic marks each new snap.
    private var snapped: Bool {
        guard live, let stage = shown else { return false }
        return stage.verticalGuide || stage.horizontalGuide
    }

    // MARK: Drawing

    private func box(_ stage: Stage, rect: CGRect, scale: CGFloat) -> some View {
        let size = boxSize(stage, scale: scale)
        return Rectangle()
            .stroke(.white, lineWidth: 1.5)
            .shadow(color: .black.opacity(0.5), radius: 1)
            .frame(width: size.width, height: size.height)
            .allowsHitTesting(false)
            .overlay(alignment: .topLeading) {
                Button { model.deleteClip() } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.black)
                        .frame(width: 20, height: 20)
                        .background(.white, in: Circle())
                        .frame(width: StageOverlay.control, height: StageOverlay.control)
                        .contentShape(Rectangle())
                }
                .offset(x: -StageOverlay.control / 2, y: -StageOverlay.control / 2)
                .accessibilityLabel("Delete clip")
            }
            .overlay(alignment: .bottomTrailing) {
                Image(systemName: "arrow.up.left.and.arrow.down.right")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(.black)
                    .frame(width: 20, height: 20)
                    .background(.white, in: Circle())
                    .frame(width: StageOverlay.control, height: StageOverlay.control)
                    .contentShape(Rectangle())
                    .offset(x: StageOverlay.control / 2, y: StageOverlay.control / 2)
                    .gesture(corner(stage, rect: rect, scale: scale))
                    .accessibilityLabel("Scale and turn")
            }
            .rotationEffect(.degrees(stage.turn))
            .position(
                x: rect.minX + stage.centerX * scale,
                y: rect.minY + stage.centerY * scale
            )
    }

    @ViewBuilder
    private func guides(_ stage: Stage, rect: CGRect) -> some View {
        if stage.verticalGuide {
            Rectangle()
                .fill(Palette.accent)
                .frame(width: 1, height: rect.height)
                .position(x: rect.midX, y: rect.midY)
                .allowsHitTesting(false)
        }
        if stage.horizontalGuide {
            Rectangle()
                .fill(Palette.accent)
                .frame(width: rect.width, height: 1)
                .position(x: rect.midX, y: rect.midY)
                .allowsHitTesting(false)
        }
    }

    // MARK: Gestures

    private func tap(rect: CGRect, scale: CGFloat) -> some Gesture {
        SpatialTapGesture().onEnded { value in
            // Every finger was up before a tap, so a gesture still open
            // lost its end; close it before acting on the tap.
            if live {
                dragging = false
                twisting = false
                cornerStart = nil
                finish()
            }
            let point = canvasPoint(value.location, rect: rect, scale: scale)
            // A tap in the selected title's box, padding too, opens its words.
            if let stage = shown, stage.text, let clip = model.selectedClip,
               contains(stage, point, size: boxSize(stage, scale: scale) / scale) {
                editText(clip)
                return
            }
            model.pick(at: point)
        }
    }

    /// One finger moves the selected picture, or the one it starts on.
    private func drag(rect: CGRect, scale: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 6)
            .updating($dragDown) { _, down, _ in down = true }
            .onChanged { value in
                // The corner handle has this finger.
                guard cornerStart == nil else { return }
                if !dragging {
                    dragging = true
                    let point = canvasPoint(value.startLocation, rect: rect, scale: scale)
                    if !live, !start(at: point, scale: scale) { return }
                }
                guard live else { return }
                shift = dragged + value.translation / scale
                send()
            }
            .onEnded { value in
                if dragging, live { dragged = dragged + value.translation / scale }
                endDrag()
            }
    }

    /// Two fingers pinch and turn the selected picture.
    private func twist(scale: CGFloat) -> some Gesture {
        MagnifyGesture().simultaneously(with: RotateGesture())
            .updating($twistDown) { _, down, _ in down = true }
            .onChanged { value in
                if !twisting {
                    twisting = true
                    if !live, shown != nil, let clip = model.selectedClip {
                        live = model.beginStage(clip)
                    }
                }
                guard live else { return }
                pinch = pinched * Double(value.first?.magnification ?? 1)
                turn = turned + (value.second?.rotation.degrees ?? 0)
                send()
            }
            .onEnded { value in
                if twisting, live {
                    pinched *= Double(value.first?.magnification ?? 1)
                    turned += value.second?.rotation.degrees ?? 0
                }
                endTwist()
            }
    }

    /// The corner handle scales and turns about the box centre with one finger.
    private func corner(_ stage: Stage, rect: CGRect, scale: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 0, coordinateSpace: .named(StageOverlay.space))
            .updating($cornerDown) { _, down, _ in down = true }
            .onChanged { value in
                guard let clip = model.selectedClip else { return }
                let center = CGPoint(
                    x: rect.minX + stage.centerX * scale,
                    y: rect.minY + stage.centerY * scale
                )
                if cornerStart == nil {
                    guard model.beginStage(clip) else { return }
                    live = true
                    cornerStart = value.startLocation - center
                }
                guard let from = cornerStart, from.length > 1 else { return }
                let to = from + value.translation
                pinch = to.length / from.length
                turn = (atan2(to.height, to.width) - atan2(from.height, from.width)) * 180 / .pi
                send()
            }
            .onEnded { _ in endCorner() }
    }

    /// Begin on the selected picture when the finger starts in its box,
    /// else on the picture under the finger.
    private func start(at point: CGPoint, scale: CGFloat) -> Bool {
        let clip: String?
        if let stage = shown, contains(stage, point, size: boxSize(stage, scale: scale) / scale) {
            clip = model.selectedClip
        } else {
            model.pick(at: point)
            clip = shown == nil ? nil : model.selectedClip
        }
        guard let clip else { return false }
        live = model.beginStage(clip)
        return live
    }

    private func send() {
        model.moveStage(shift: shift, pinch: pinch, turn: turn)
    }

    private func endDrag() {
        guard dragging else { return }
        dragging = false
        finish()
    }

    private func endCorner() {
        guard cornerStart != nil else { return }
        cornerStart = nil
        finish()
    }

    private func endTwist() {
        guard twisting else { return }
        twisting = false
        finish()
    }

    /// End once every finger is up.
    private func finish() {
        guard !dragging, !twisting, cornerStart == nil else { return }
        if live { model.endStage() }
        live = false
        shift = .zero
        dragged = .zero
        pinch = 1
        pinched = 1
        turn = 0
        turned = 0
    }

    // MARK: Geometry

    static let space = "stage"
    /// The touch area of the delete button and the corner handle.
    static let control: CGFloat = 32

    /// The box in view points. A title's picture is cut tight to its
    /// letters, so it gets room around them, and at least enough that a
    /// finger on the words misses the controls at its corners.
    private func boxSize(_ stage: Stage, scale: CGFloat) -> CGSize {
        let width = stage.width * scale
        let height = stage.height * scale
        guard stage.text else { return CGSize(width: width, height: height) }
        return CGSize(width: max(width + 16, 72), height: max(height + 16, 52))
    }

    /// The canvas fitted in the preview, as the picture is drawn.
    private func fitted(_ canvas: CGSize, in size: CGSize) -> CGRect {
        let scale = min(size.width / canvas.width, size.height / canvas.height)
        let width = canvas.width * scale
        let height = canvas.height * scale
        return CGRect(
            x: (size.width - width) / 2, y: (size.height - height) / 2,
            width: width, height: height
        )
    }

    private func canvasPoint(_ point: CGPoint, rect: CGRect, scale: CGFloat) -> CGPoint {
        CGPoint(x: (point.x - rect.minX) / scale, y: (point.y - rect.minY) / scale)
    }

    /// Whether `point` is in the box `size` canvas pixels about the clip's centre.
    private func contains(_ stage: Stage, _ point: CGPoint, size: CGSize) -> Bool {
        let radians = stage.turn * .pi / 180
        let dx = point.x - stage.centerX
        let dy = point.y - stage.centerY
        let x = dx * cos(radians) + dy * sin(radians)
        let y = -dx * sin(radians) + dy * cos(radians)
        return abs(x) <= size.width / 2 && abs(y) <= size.height / 2
    }
}

private extension CGSize {
    var length: CGFloat { (width * width + height * height).squareRoot() }

    static func + (lhs: CGSize, rhs: CGSize) -> CGSize {
        CGSize(width: lhs.width + rhs.width, height: lhs.height + rhs.height)
    }

    static func / (lhs: CGSize, rhs: CGFloat) -> CGSize {
        CGSize(width: lhs.width / rhs, height: lhs.height / rhs)
    }
}

private extension CGPoint {
    static func - (lhs: CGPoint, rhs: CGPoint) -> CGSize {
        CGSize(width: lhs.x - rhs.x, height: lhs.y - rhs.y)
    }
}
