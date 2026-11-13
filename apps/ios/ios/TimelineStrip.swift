import PhotosUI
import SwiftUI

/// The rows under a fixed playhead in the middle of the strip.
///
/// Dragging sideways moves time under the playhead, and a swipe let go
/// slides on and slows to a stop until a tap or drag; a pinch zooms. Picture
/// clips show a filmstrip. A tap selects a clip and a tap on empty space
/// clears it. A selected clip shows its length and a grip at each end to
/// trim it. A long press on a main clip folds the main row into squares to
/// reorder; on any other row it moves the clip in time or to another row.
/// Rust places, snaps and packs when the finger lifts.
struct TimelineStrip: View {
    @Bindable var model: EditorModel
    @Binding var picked: [PhotosPickerItem]
    /// A tap on the button between two main clips, with the clip before it.
    var openCut: (String) -> Void = { _ in }
    @State private var dragStart: Int64?
    @State private var pinchStart: Double?
    /// A clip off main being moved, and how far the finger has gone.
    @State private var moving: (id: String, offset: CGSize)?
    /// The edge being trimmed and how far it has gone, in points.
    @State private var trimming: (id: String, startEdge: Bool, dx: CGFloat)?
    /// A main clip being reordered: where the finger went down and where it is.
    @State private var reordering: (id: String, pressX: CGFloat, fingerX: CGFloat)?

    private let rulerHeight: CGFloat = 26
    private let laneGap: CGFloat = 8
    private let square: CGFloat = 52

    var body: some View {
        GeometryReader { geometry in
            let middle = geometry.size.width / 2
            let origin = middle - x(model.playheadUs)
            ZStack(alignment: .topLeading) {
                Color.clear.contentShape(Rectangle())
                ruler(origin: origin, width: geometry.size.width)
                VStack(alignment: .leading, spacing: laneGap) {
                    ForEach(Array(model.lanes.enumerated()), id: \.offset) { index, lane in
                        // The held clip's row stays under the squares: it
                        // owns the gesture, which ends only while it is there.
                        let folded = lane.main && reordering != nil
                        row(lane, index: index, origin: origin)
                            .opacity(folded ? 0 : 1)
                            .overlay(alignment: .topLeading) {
                                if folded, let reordering {
                                    squares(lane, reordering: reordering)
                                        .allowsHitTesting(false)
                                }
                            }
                            .zIndex(movingIn(lane) ? 1 : 0)
                    }
                }
                .padding(.top, rulerHeight + 6)
                Rectangle()
                    .fill(.white)
                    .frame(width: 2)
                    .offset(x: middle - 1)
                    .allowsHitTesting(false)
                addButton
                    .offset(x: geometry.size.width - 52, y: rulerHeight + 6 + 6)
            }
            .coordinateSpace(.named("strip"))
            .clipped()
            .gesture(scrub)
            .simultaneousGesture(zoom(width: geometry.size.width))
            .onTapGesture {
                model.stopCoast()
                model.selectedClip = nil
            }
            .onAppear { model.refreshStrip(halfWidth: middle) }
            .onChange(of: model.playheadUs) { model.refreshStrip(halfWidth: middle) }
            .onChange(of: model.pixelSize) { model.refreshStrip(halfWidth: middle) }
            .onChange(of: model.lanes) { model.refreshStrip(halfWidth: middle) }
            .onChange(of: geometry.size.width) { model.refreshStrip(halfWidth: middle) }
        }
    }

    // MARK: Time and points

    private func x(_ time: Int64) -> CGFloat {
        CGFloat(Double(time) * model.pixelSize)
    }

    private func micros(_ points: CGFloat) -> Int64 {
        Int64(Double(points) / model.pixelSize)
    }

    /// Points a snap reaches, as time at this zoom.
    private var snapUs: Int64 { micros(10) }

    private func rowHeight(_ lane: Lane) -> CGFloat { lane.main ? 56 : 40 }

    private func movingIn(_ lane: Lane) -> Bool {
        guard let id = moving?.id else { return false }
        return lane.clips.contains { $0.id == id }
    }

    // MARK: Ruler

    /// A label on each major tick and a dot on each minor one, as Rust laid
    /// them out for this zoom.
    private func ruler(origin: CGFloat, width: CGFloat) -> some View {
        Canvas { context, size in
            // A dot is left out within 22 points of a label, or 14 of the
            // dot before it.
            let labels = model.ticks.filter(\.major).map { origin + x($0.timeUs) }
            var last = -CGFloat.infinity
            for tick in model.ticks {
                let position = origin + x(tick.timeUs)
                guard position > -40, position < width + 40 else { continue }
                if !tick.major {
                    if position - last < 14 { continue }
                    if labels.contains(where: { abs($0 - position) < 22 }) { continue }
                    last = position
                }
                if tick.major, let label = tick.label {
                    context.draw(
                        Text(label).font(.system(size: 10).monospacedDigit())
                            .foregroundStyle(.white.opacity(0.55)),
                        at: CGPoint(x: position, y: size.height / 2)
                    )
                } else {
                    let dot = Path(ellipseIn: CGRect(
                        x: position - 1.5, y: size.height / 2 - 1.5, width: 3, height: 3))
                    context.fill(dot, with: .color(.white.opacity(0.3)))
                }
            }
        }
        .frame(height: rulerHeight)
        .allowsHitTesting(false)
    }

    // MARK: Rows

    private func row(_ lane: Lane, index: Int, origin: CGFloat) -> some View {
        let height = rowHeight(lane)
        return ZStack(alignment: .topLeading) {
            Color.clear.frame(height: height)
            ForEach(lane.clips, id: \.id) { clip in
                clipView(clip, lane: lane, index: index, origin: origin, height: height)
            }
            if lane.main {
                ForEach(model.cuts, id: \.outgoingId) { cut in
                    junction(cut)
                        .offset(x: origin + x(cut.atUs) - 11, y: (height - 22) / 2)
                }
            }
            if lane.main && !lane.clips.isEmpty {
                muteButton(lane)
                    .offset(x: origin - 50, y: (height - 40) / 2)
            }
            if lane.main && lane.clips.isEmpty {
                Text("Tap + to add a photo or video")
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.45))
                    .frame(height: height)
                    .offset(x: origin + 12)
            }
        }
        .frame(height: height)
    }

    private func clipView(
        _ clip: Clip, lane: Lane, index: Int, origin: CGFloat, height: CGFloat
    ) -> some View {
        let selected = model.selectedClip == clip.id
        var left = origin + x(clip.startUs)
        var width = x(clip.durationUs)
        var shown = clip.durationUs
        if let trim = trimming, trim.id == clip.id {
            if trim.startEdge {
                let dx = min(trim.dx, width - 4)
                left += dx
                width -= dx
            } else {
                width = max(width + trim.dx, 4)
            }
            shown = micros(width)
        }
        let moved = moving?.id == clip.id ? moving?.offset ?? .zero : .zero
        return ClipBlock(
            clip: clip, kind: lane.kind, selected: selected, height: height,
            tiles: model.films[clip.id] ?? [], thumbs: model.thumbs, wave: model.waves[clip.id],
            trimmedLeft: trimming?.id == clip.id && trimming?.startEdge == true
                ? min(trimming?.dx ?? 0, x(clip.durationUs) - 4) : 0
        )
        .frame(width: max(width - 2, 2), height: height)
        // A clip that shows its name keeps the length below it.
        .overlay(alignment: model.films[clip.id]?.isEmpty == false ? .topLeading : .bottomLeading) {
            if selected {
                Text(String(format: "%.1fs", Double(shown) / 1_000_000))
                    .font(.system(size: 10, weight: .semibold).monospacedDigit())
                    .foregroundStyle(.white)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1)
                    .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 3))
                    .padding(.leading, lane.locked ? 4 : 16)
                    .padding(.vertical, 4)
            }
        }
        .overlay {
            if selected && !lane.locked {
                HStack(spacing: 0) {
                    grip(clip, startEdge: true)
                    Spacer(minLength: 0)
                    grip(clip, startEdge: false)
                }
            }
        }
        .overlay(alignment: .bottomLeading) {
            if !clip.beatsUs.isEmpty {
                ZStack(alignment: .bottomLeading) {
                    ForEach(clip.beatsUs, id: \.self) { time in
                        Circle()
                            .fill(Color(red: 1, green: 0.82, blue: 0.2))
                            .frame(width: 6, height: 6)
                            .offset(x: x(time - clip.startUs) - 3, y: -3)
                    }
                }
                .allowsHitTesting(false)
            }
        }
        .overlay(alignment: .leading) {
            if selected, let keys = model.keys {
                ZStack(alignment: .leading) {
                    ForEach(keys.timesUs, id: \.self) { time in
                        KeyDiamond()
                            .offset(x: x(time - clip.startUs) - 1 - KeyDiamond.side / 2)
                    }
                }
                .allowsHitTesting(false)
            }
        }
        .scaleEffect(moving?.id == clip.id ? 1.04 : 1)
        .shadow(radius: moving?.id == clip.id ? 6 : 0)
        .offset(x: left + 1 + moved.width, y: moved.height)
        .onTapGesture {
            model.stopCoast()
            model.selectedClip = clip.id
        }
        .gesture(lane.locked ? nil : (lane.main ? reorder(clip) : nil))
        .gesture(lane.locked || lane.main ? nil : move(clip, row: index))
    }

    /// The button on a cut between two main clips: white, marked when the
    /// cut has a transition. A tap opens the transitions.
    private func junction(_ cut: Cut) -> some View {
        let chosen = model.selectedCut == cut.outgoingId
        return Button {
            model.stopCoast()
            model.pause()
            model.selectedClip = nil
            model.selectedCut = cut.outgoingId
            openCut(cut.outgoingId)
        } label: {
            Group {
                if cut.transition != nil || !cut.name.isEmpty {
                    Image(systemName: "arrow.left.and.right")
                        .font(.system(size: 9, weight: .bold))
                } else {
                    Rectangle().frame(width: 2, height: 10)
                }
            }
            .foregroundStyle(.black)
            .frame(width: 22, height: 22)
            .background(chosen ? Color.cyan : .white, in: RoundedRectangle(cornerRadius: 5))
            .shadow(color: .black.opacity(0.4), radius: 2)
            .contentShape(Rectangle().inset(by: -6))
        }
        .accessibilityLabel(cut.name.isEmpty ? "Add transition" : "Transition \(cut.name)")
    }

    /// The main row's sound switch, left of time zero; it scrolls away with
    /// the rows.
    private func muteButton(_ lane: Lane) -> some View {
        Button { model.setMuted(lane, !lane.muted) } label: {
            VStack(spacing: 2) {
                HugeIcon(lane.muted ? "volumeOff" : "volume", size: 18)
                Text(lane.muted ? "Muted" : "Sound").font(.system(size: 9))
            }
            .foregroundStyle(.white.opacity(0.85))
            .frame(width: 40, height: 40)
            .background(Color(white: 0.17), in: RoundedRectangle(cornerRadius: 6))
        }
        .accessibilityLabel(lane.muted ? "Unmute clip audio" : "Mute clip audio")
    }

    /// A white tab on one end of the selected clip. Dragging it trims that end.
    private func grip(_ clip: Clip, startEdge: Bool) -> some View {
        UnevenRoundedRectangle(
            topLeadingRadius: startEdge ? 6 : 0, bottomLeadingRadius: startEdge ? 6 : 0,
            bottomTrailingRadius: startEdge ? 0 : 6, topTrailingRadius: startEdge ? 0 : 6
        )
        .fill(.white)
        .frame(width: 12)
        .overlay {
            Capsule().fill(.black.opacity(0.45)).frame(width: 2, height: 16)
        }
        .contentShape(Rectangle().inset(by: -8))
        .highPriorityGesture(
            DragGesture(minimumDistance: 1)
                .onChanged { drag in
                    model.pause()
                    trimming = (clip.id, startEdge, drag.translation.width)
                }
                .onEnded { drag in
                    trimming = nil
                    let edge = startEdge ? clip.startUs : clip.endUs
                    model.trim(
                        clip.id, startEdge: startEdge,
                        timeUs: edge + micros(drag.translation.width), snapUs: snapUs)
                }
        )
        .accessibilityLabel(startEdge ? "Trim start" : "Trim end")
    }

    /// The pinned add button over the main row.
    private var addButton: some View {
        PhotosPicker(
            selection: $picked, matching: .any(of: [.images, .videos]),
            preferredItemEncoding: .current
        ) {
            HugeIcon("plus", size: 22)
                .foregroundStyle(.black)
                .frame(width: 40, height: 40)
                .background(.white, in: RoundedRectangle(cornerRadius: 8))
                .shadow(color: .black.opacity(0.4), radius: 4)
        }
        .disabled(model.isOpening || model.isImporting)
        .accessibilityLabel("Add media")
    }

    // MARK: Move off main

    /// Hold, then drag a clip to another time or row.
    private func move(_ clip: Clip, row: Int) -> some Gesture {
        LongPressGesture(minimumDuration: 0.3)
            .sequenced(before: DragGesture(minimumDistance: 0))
            .onChanged { value in
                guard case .second(true, let drag) = value else { return }
                if moving == nil {
                    model.pause()
                    model.selectedClip = clip.id
                }
                moving = (clip.id, drag?.translation ?? .zero)
            }
            .onEnded { value in
                moving = nil
                guard case .second(true, let drag?) = value else { return }
                model.moveClip(
                    clip.id, row: targetRow(from: row, dy: drag.translation.height),
                    startUs: clip.startUs + micros(drag.translation.width), snapUs: snapUs)
            }
    }

    /// The row a vertical drag of `dy` points from `row` lands on.
    private func targetRow(from row: Int, dy: CGFloat) -> Int {
        let lanes = model.lanes
        guard lanes.indices.contains(row) else { return row }
        var top: CGFloat = 0
        for index in 0..<row { top += rowHeight(lanes[index]) + laneGap }
        let finger = top + rowHeight(lanes[row]) / 2 + dy
        var edge: CGFloat = 0
        for (index, lane) in lanes.enumerated() {
            edge += rowHeight(lane) + laneGap
            if finger < edge { return index }
        }
        return lanes.count
    }

    // MARK: Reorder main

    /// Hold a main clip: the row folds into squares, the held one follows the
    /// finger, and the others make room where it would land.
    private func reorder(_ clip: Clip) -> some Gesture {
        LongPressGesture(minimumDuration: 0.3)
            .sequenced(before: DragGesture(minimumDistance: 0, coordinateSpace: .named("strip")))
            .onChanged { value in
                guard case .second(true, let drag) = value else { return }
                if reordering == nil {
                    model.pause()
                    model.selectedClip = clip.id
                    let x = drag?.startLocation.x ?? 0
                    withAnimation(.easeOut(duration: 0.18)) {
                        reordering = (clip.id, x, x)
                    }
                } else if let drag, let current = reordering {
                    reordering = (current.id, current.pressX, drag.location.x)
                }
            }
            .onEnded { _ in
                guard let current = reordering, let main = model.lanes.first(where: \.main)
                else {
                    reordering = nil
                    return
                }
                let place = landing(current, count: main.clips.count, from: main)
                withAnimation(.easeOut(duration: 0.18)) { reordering = nil }
                if place != main.clips.firstIndex(where: { $0.id == current.id }) {
                    model.reorder(current.id, place: place)
                }
            }
    }

    /// The place the held clip lands in the order without it.
    private func landing(
        _ held: (id: String, pressX: CGFloat, fingerX: CGFloat), count: Int, from lane: Lane
    ) -> Int {
        let start = lane.clips.firstIndex { $0.id == held.id } ?? 0
        let steps = Int(((held.fingerX - held.pressX) / (square + 6)).rounded())
        return min(max(start + steps, 0), max(count - 1, 0))
    }

    private func squares(
        _ lane: Lane, reordering held: (id: String, pressX: CGFloat, fingerX: CGFloat)
    ) -> some View {
        let start = lane.clips.firstIndex { $0.id == held.id } ?? 0
        let place = landing(held, count: lane.clips.count, from: lane)
        let others = lane.clips.filter { $0.id != held.id }
        let slot = { (index: Int) -> CGFloat in
            held.pressX + CGFloat(index - start) * (square + 6) - square / 2
        }
        return ZStack(alignment: .topLeading) {
            Color.clear.frame(height: rowHeight(lane))
            ForEach(Array(others.enumerated()), id: \.element.id) { index, clip in
                SquareClip(clip: clip, tile: model.films[clip.id]?.first, thumbs: model.thumbs)
                    .frame(width: square, height: square)
                    .offset(x: slot(index < place ? index : index + 1), y: 2)
            }
            if let clip = lane.clips.first(where: { $0.id == held.id }) {
                SquareClip(clip: clip, tile: model.films[clip.id]?.first, thumbs: model.thumbs)
                    .frame(width: square, height: square)
                    .overlay {
                        RoundedRectangle(cornerRadius: 6).strokeBorder(.white, lineWidth: 2)
                    }
                    .scaleEffect(1.08)
                    .shadow(radius: 6)
                    .offset(x: held.fingerX - square / 2, y: 2)
                    .zIndex(1)
            }
        }
        .frame(height: rowHeight(lane))
        .animation(.easeOut(duration: 0.15), value: place)
    }

    // MARK: Scrub and zoom

    private var scrub: some Gesture {
        DragGesture(minimumDistance: 4)
            .onChanged { drag in
                guard reordering == nil else { return }
                if dragStart == nil {
                    model.stopCoast()
                    model.pause()
                    dragStart = model.playheadUs
                }
                let start = dragStart ?? model.playheadUs
                model.seek(start - micros(drag.translation.width))
            }
            .onEnded { drag in
                guard dragStart != nil else { return }
                dragStart = nil
                model.coast(from: drag.velocity.width)
            }
    }

    private func zoom(width: CGFloat) -> some Gesture {
        MagnifyGesture()
            .onChanged { pinch in
                let start = pinchStart ?? model.pixelSize
                pinchStart = start
                model.zoom(from: start, by: pinch.magnification, width: width)
            }
            .onEnded { _ in pinchStart = nil }
    }
}

/// One clip: a filmstrip for a picture, else its colour by kind and its
/// name, with a white frame when selected.
struct ClipBlock: View {
    let clip: Clip
    let kind: LaneKind
    let selected: Bool
    let height: CGFloat
    let tiles: [Tile]
    let thumbs: [String: CGImage]
    let wave: Wave?
    /// Points cut off the left while the start edge is dragged.
    let trimmedLeft: CGFloat

    /// The band a video's sound draws in over the foot of its picture.
    static let band: CGFloat = 16

    var body: some View {
        RoundedRectangle(cornerRadius: 6)
            .fill(fill)
            .overlay(alignment: .topLeading) {
                if tiles.isEmpty {
                    Text(clip.name)
                        .font(.caption2.weight(.medium))
                        .foregroundStyle(.white.opacity(0.9))
                        .lineLimit(1)
                        // Clear of the trim grip while selected.
                        .padding(.leading, selected ? 16 : 6)
                        .padding(.trailing, 6)
                        .padding(.top, 4)
                } else {
                    film
                }
            }
            .overlay(alignment: .bottomLeading) { sound }
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay {
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(.white, lineWidth: selected ? 2 : 0)
            }
            .contentShape(Rectangle())
    }

    private var film: some View {
        ZStack(alignment: .topLeading) {
            ForEach(tiles, id: \.left) { tile in
                Group {
                    if let image = thumbs[tile.key] {
                        Image(decorative: image, scale: 1)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                    } else {
                        Color(white: 0.16)
                    }
                }
                .frame(width: tile.width, height: height)
                .clipped()
                .offset(x: tile.left - trimmedLeft)
            }
        }
    }

    /// The clip's waveform: bars from the middle of an audio clip, from the
    /// foot of a video's band. Dim when the row is muted.
    @ViewBuilder
    private var sound: some View {
        if let wave, !wave.levels.isEmpty {
            let under = wave.underPicture
            let tall = under ? Self.band : height
            Canvas { context, size in
                var path = Path()
                let left = CGFloat(wave.first) - trimmedLeft
                for (index, level) in wave.levels.enumerated() {
                    let x = left + CGFloat(index)
                    guard x >= -1, x <= size.width else { continue }
                    let bar = max(CGFloat(level) * (under ? size.height - 2 : size.height * 0.8), 1)
                    let top = under ? size.height - bar : (size.height - bar) / 2
                    path.addRect(CGRect(x: x, y: top, width: 1, height: bar))
                }
                context.fill(path, with: .color(.white.opacity(wave.muted ? 0.25 : 0.7)))
            }
            .frame(height: tall)
            .background(under ? Color.black.opacity(0.45) : Color.clear)
        } else if let wave, wave.underPicture {
            Color.black.opacity(0.45).frame(height: Self.band)
        }
    }

    private var fill: Color {
        switch kind {
        case .video: Color(red: 30 / 255, green: 76 / 255, blue: 81 / 255)
        case .audio: Color(red: 29 / 255, green: 54 / 255, blue: 94 / 255)
        case .text: Color(red: 120 / 255, green: 84 / 255, blue: 40 / 255)
        default: Color(white: 0.3)
        }
    }
}

/// A main clip folded to a square while the row is reordered.
struct SquareClip: View {
    let clip: Clip
    let tile: Tile?
    let thumbs: [String: CGImage]

    var body: some View {
        Group {
            if let key = tile?.key, let image = thumbs[key] {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Color(red: 30 / 255, green: 76 / 255, blue: 81 / 255)
                    .overlay {
                        Text(clip.name)
                            .font(.system(size: 9))
                            .foregroundStyle(.white.opacity(0.8))
                            .lineLimit(2)
                            .padding(3)
                    }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

/// A keyframe's mark on the selected clip.
struct KeyDiamond: View {
    static let side: CGFloat = 11

    var body: some View {
        Rectangle()
            .fill(Color(red: 1, green: 0.3, blue: 0.45))
            .overlay(Rectangle().strokeBorder(.white, lineWidth: 1.5))
            .frame(width: Self.side / 1.414, height: Self.side / 1.414)
            .rotationEffect(.degrees(45))
            .frame(width: Self.side, height: Self.side)
    }
}
