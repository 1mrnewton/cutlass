import SwiftUI

/// What a finger on the preview does while the Mask panel is open.
enum MaskTool: Hashable {
    /// Drags the mask in hand by its middle, corners and knobs.
    case handles
    /// Paints a brush mask, or with `erase` rubs it out.
    case brush(erase: Bool)
    /// Puts down the corners of a pen mask's outline.
    case pen
}

/// The selected clip's masks: a shape tile adds one, the chips pick the
/// one in hand, and the controls under them change it. Rust stores each
/// change, and keys it at the playhead when the mask has keys.
struct MaskPanel: View {
    let model: EditorModel
    let close: () -> Void
    @State private var words = ""

    var body: some View {
        PanelFrame(title: "Mask", close: done) {
            VStack(spacing: 12) {
                shapes
                if !model.masks.isEmpty { list }
                if let mask = model.maskInHand { settings(mask) }
            }
        }
        .onAppear { words = model.maskInHand?.text ?? "" }
        .onChange(of: model.maskIndex) { _, _ in
            words = model.maskInHand?.text ?? ""
            model.maskTool = .handles
            model.penPoints = []
        }
    }

    private func done() {
        model.maskTool = .handles
        model.penPoints = []
        close()
    }

    // MARK: Shapes

    private var shapes: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(model.maskShapeList, id: \.key) { shape in
                    Button {
                        model.addMask(shape.key)
                    } label: {
                        VStack(spacing: 6) {
                            Image(systemName: MaskPanel.symbol(shape.key))
                                .font(.system(size: 20))
                                .frame(width: 44, height: 36)
                            Text(shape.label).font(.caption2)
                        }
                        .frame(width: 60, height: 64)
                    }
                    .accessibilityLabel("Add \(shape.label) mask")
                }
            }
            .padding(.horizontal, 2)
        }
    }

    static func symbol(_ shape: String) -> String {
        switch shape {
        case "linear": "line.diagonal"
        case "mirror": "equal"
        case "circle": "circle"
        case "rectangle": "rectangle"
        case "heart": "heart"
        case "star": "star"
        case "text": "textformat"
        case "brush": "paintbrush"
        case "pen": "pencil.tip"
        default: "questionmark"
        }
    }

    // MARK: The list

    private var list: some View {
        HStack(spacing: 8) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(Array(model.masks.enumerated()), id: \.offset) { index, mask in
                        let chosen = index == model.maskIndex
                        Button {
                            model.maskIndex = index
                        } label: {
                            Label("\(index + 1)", systemImage: MaskPanel.symbol(mask.shape))
                                .font(.caption)
                                .padding(.horizontal, 10)
                                .frame(height: 30)
                                .background(
                                    chosen ? Palette.accent.opacity(0.25) : Color.white.opacity(0.08),
                                    in: Capsule())
                                .overlay(Capsule().strokeBorder(chosen ? Palette.accent : .clear))
                        }
                        .accessibilityLabel("Mask \(index + 1), \(mask.shape)")
                        .accessibilityAddTraits(chosen ? .isSelected : [])
                    }
                }
            }
            Button { model.moveMask(up: true) } label: { Image(systemName: "chevron.left") }
                .disabled(model.maskIndex == 0)
                .accessibilityLabel("Move up")
            Button { model.moveMask(up: false) } label: { Image(systemName: "chevron.right") }
                .disabled(model.maskIndex + 1 >= model.masks.count)
                .accessibilityLabel("Move down")
            Button { model.removeMask() } label: { Image(systemName: "trash") }
                .accessibilityLabel("Delete mask")
        }
        .frame(height: 32)
    }

    // MARK: The mask in hand

    @ViewBuilder
    private func settings(_ mask: ClipMask) -> some View {
        HStack {
            Picker("Combine", selection: Binding(
                get: { mask.mode },
                set: { mode in
                    var changed = mask
                    changed.mode = mode
                    model.editMask(changed, live: false)
                }
            )) {
                Text("Add").tag("add")
                Text("Subtract").tag("subtract")
                Text("Intersect").tag("intersect")
            }
            .pickerStyle(.segmented)
            .accessibilityLabel("Combine")
            Toggle("Invert", isOn: Binding(
                get: { mask.invert },
                set: { invert in
                    var changed = mask
                    changed.invert = invert
                    model.editMask(changed, live: false)
                }
            ))
            .toggleStyle(.button)
            .font(.caption)
        }
        StyleSlider(
            label: "Feather", value: mask.feather, range: 0...1,
            shown: { "\(Int(($0 * 100).rounded()))" }
        ) { value, live in
            var changed = mask
            changed.feather = value
            model.editMask(changed, live: live)
        }
        if mask.shape == "rectangle" {
            StyleSlider(
                label: "Round", value: mask.round, range: 0...1,
                shown: { "\(Int(($0 * 100).rounded()))" }
            ) { value, live in
                var changed = mask
                changed.round = value
                model.editMask(changed, live: live)
            }
        }
        StyleSlider(
            label: "Rotate", value: mask.rotation, range: -180...180,
            shown: { "\(Int($0.rounded()))°" }
        ) { value, live in
            var changed = mask
            changed.rotation = value
            model.editMask(changed, live: live)
        }
        switch mask.shape {
        case "text": textRow
        case "brush": brushRow
        case "pen": penRow
        default: EmptyView()
        }
    }

    private var textRow: some View {
        HStack {
            TextField("Text", text: $words)
                .textFieldStyle(.roundedBorder)
                .submitLabel(.done)
                .onSubmit { model.setMaskText(words) }
            Button("Set") { model.setMaskText(words) }
                .disabled(words.isEmpty)
        }
    }

    private var brushRow: some View {
        VStack(spacing: 8) {
            Picker("Brush", selection: Binding(
                get: { model.maskTool },
                set: { model.maskTool = $0 }
            )) {
                Text("Move").tag(MaskTool.handles)
                Text("Paint").tag(MaskTool.brush(erase: false))
                Text("Erase").tag(MaskTool.brush(erase: true))
            }
            .pickerStyle(.segmented)
            .accessibilityLabel("Brush")
            StyleSlider(
                label: "Size", value: model.maskBrushSize, range: 4...200,
                shown: { "\(Int($0.rounded()))" }
            ) { value, _ in
                model.maskBrushSize = value
            }
        }
    }

    private var penRow: some View {
        HStack {
            if model.maskTool == .pen {
                Text("\(model.penPoints.count) points")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.white.opacity(0.7))
                Spacer()
                Button("Cancel") {
                    model.penPoints = []
                    model.maskTool = .handles
                }
                Button("Done") {
                    model.drawMaskOutline(model.penPoints)
                    model.penPoints = []
                    model.maskTool = .handles
                }
                .disabled(model.penPoints.count < 3)
            } else {
                Button {
                    model.penPoints = []
                    model.maskTool = .pen
                } label: {
                    Label("Draw outline", systemImage: "pencil.tip")
                }
                Spacer()
            }
        }
        .font(.footnote)
    }
}

/// The mask in hand on the preview: its box and knobs to drag, or a
/// brush or pen drawing on it. Rust does the geometry; this only turns
/// points into canvas pixels.
struct MaskOverlay: View {
    let model: EditorModel
    /// The brush stroke under the finger, in view points.
    @State private var stroke: [CGPoint] = []
    /// A handle is held.
    @State private var holding = false
    @State private var began = false

    var body: some View {
        GeometryReader { geometry in
            let canvas = model.canvasSize ?? CGSize(width: 16, height: 9)
            let scale = min(geometry.size.width / canvas.width, geometry.size.height / canvas.height)
            let origin = CGPoint(
                x: (geometry.size.width - canvas.width * scale) / 2,
                y: (geometry.size.height - canvas.height * scale) / 2
            )
            let view = { (x: Double, y: Double) in
                CGPoint(x: origin.x + x * scale, y: origin.y + y * scale)
            }
            ZStack(alignment: .topLeading) {
                Color.clear.contentShape(Rectangle())
                if let handles = model.maskHandles {
                    outline(handles, view: view)
                }
                if model.maskTool == .pen {
                    pen(view: view)
                }
                if case .brush = model.maskTool {
                    Path { path in
                        guard let first = stroke.first else { return }
                        path.move(to: first)
                        stroke.dropFirst().forEach { path.addLine(to: $0) }
                    }
                    .stroke(
                        Palette.accent.opacity(0.8),
                        style: StrokeStyle(
                            lineWidth: max(model.maskBrushSize * scale, 1), lineCap: .round,
                            lineJoin: .round))
                }
            }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        let at = CGPoint(
                            x: (value.location.x - origin.x) / scale,
                            y: (value.location.y - origin.y) / scale)
                        switch model.maskTool {
                        case .handles:
                            if !began {
                                began = true
                                let from = CGPoint(
                                    x: (value.startLocation.x - origin.x) / scale,
                                    y: (value.startLocation.y - origin.y) / scale)
                                holding = model.beginMaskDrag(at: from)
                            }
                            if holding { model.dragMask(to: at) }
                        case .brush:
                            stroke.append(value.location)
                        case .pen:
                            break
                        }
                    }
                    .onEnded { value in
                        switch model.maskTool {
                        case .handles:
                            if holding { model.endMaskDrag() }
                        case let .brush(erase):
                            let points = stroke.map {
                                CGPoint(x: ($0.x - origin.x) / scale, y: ($0.y - origin.y) / scale)
                            }
                            stroke = []
                            if !points.isEmpty {
                                model.paintMask(points, size: model.maskBrushSize, erase: erase)
                            }
                        case .pen:
                            if hypot(value.translation.width, value.translation.height) < 10 {
                                model.penPoints.append(CGPoint(
                                    x: (value.location.x - origin.x) / scale,
                                    y: (value.location.y - origin.y) / scale))
                            }
                        }
                        holding = false
                        began = false
                    }
            )
        }
        .accessibilityLabel("Mask area")
    }

    private func outline(
        _ handles: MaskHandles, view: @escaping (Double, Double) -> CGPoint
    ) -> some View {
        let corners = stride(from: 0, to: handles.corners.count - 1, by: 2).map {
            view(handles.corners[$0], handles.corners[$0 + 1])
        }
        let turn = view(handles.turnX, handles.turnY)
        let feather = view(handles.featherX, handles.featherY)
        let middle = view(handles.middleX, handles.middleY)
        return ZStack(alignment: .topLeading) {
            Path { path in
                guard let first = corners.first else { return }
                path.move(to: first)
                corners.dropFirst().forEach { path.addLine(to: $0) }
                path.closeSubpath()
                if handles.line.count == 4 {
                    path.move(to: view(handles.line[0], handles.line[1]))
                    path.addLine(to: view(handles.line[2], handles.line[3]))
                }
            }
            .stroke(.white, style: StrokeStyle(lineWidth: 1.5, dash: [5, 4]))
            .shadow(color: .black.opacity(0.5), radius: 1)
            ForEach(Array(corners.enumerated()), id: \.offset) { _, corner in
                knob(at: corner, color: .white)
            }
            knob(at: turn, color: Palette.accent)
            knob(at: feather, color: .yellow)
            knob(at: middle, color: .white.opacity(0.6))
        }
        .allowsHitTesting(false)
    }

    private func knob(at point: CGPoint, color: Color) -> some View {
        Circle()
            .fill(color)
            .overlay(Circle().stroke(.black.opacity(0.5), lineWidth: 1))
            .frame(width: 12, height: 12)
            .position(point)
    }

    private func pen(view: @escaping (Double, Double) -> CGPoint) -> some View {
        let points = model.penPoints.map { view($0.x, $0.y) }
        return ZStack(alignment: .topLeading) {
            Path { path in
                guard let first = points.first else { return }
                path.move(to: first)
                points.dropFirst().forEach { path.addLine(to: $0) }
            }
            .stroke(Palette.accent, lineWidth: 2)
            ForEach(Array(points.enumerated()), id: \.offset) { _, point in
                knob(at: point, color: Palette.accent)
            }
        }
        .allowsHitTesting(false)
    }
}
