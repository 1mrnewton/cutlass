import SwiftUI

/// The scrubbable timeline: a fixed center playhead over horizontally
/// scrolling content, where scroll offset IS the current time. Renders the
/// ordered lane stack (desktop rules: one kind per lane, audio pinned to the
/// bottom) around the magnetic main track.
struct TimelineView: View {
    var state: EditorState
    /// User-chosen height (from the grab bar above the transport row);
    /// nil = fit content. Clamped to [minHeight, max(natural, maxHeight)].
    var userHeight: CGFloat?
    /// Upper bound from the editor chrome (preview keeps a minimum share).
    var maxTimelineHeight: CGFloat?
    var onAddMedia: () -> Void
    var onTransitionTap: (UUID) -> Void = { _ in }

    /// seconds-per-point captured when a pinch begins.
    @State private var pinchBase: Double?
    /// Live long-press drag-to-reorder on the main track (square sort mode).
    @State private var reorder: ReorderDrag?
    /// Commit-on-release drag of a lane clip or a main clip leaving the track.
    @State private var timelineDrag: TimelineDrag?
    /// Bumped on every successful cross-lane conversion (success haptic).
    @State private var conversionPulse = 0
    /// How far the lane stack is panned up inside its viewport.
    @State private var laneOffset: CGFloat = 0
    /// Playhead + lane offset captured when a timeline pan begins.
    @State private var panAnchor: (playhead: TimeInterval, lane: CGFloat)?
    /// Direction lock for the live pan, decided from the first translation.
    @State private var panMode: PanMode?
    /// Post-flick momentum; cancelled by any new touch or playback.
    @State private var decayTask: Task<Void, Never>?
    @State private var autoscrollTask: Task<Void, Never>?
    /// The timeline's frame in screen space; drag locations arrive in the
    /// global coordinate space, so band/time hit-testing needs this origin.
    @State private var globalFrame: CGRect = .zero

    private enum PanMode {
        case scrub, lanes
    }

    private struct ReorderDrag: Equatable {
        var clipID: UUID
        var fromIndex: Int
        /// Keeps the square strip anchored where the lifted clip's leading
        /// edge was in time layout, so the morph doesn't jump.
        var stripShift: CGFloat
        var translation: CGSize = .zero
        var location: CGPoint = .zero
        var insertionIndex: Int
        /// Finger is over the overlay band: drop converts to a PiP overlay.
        var overOverlayBand = false
    }

    /// Live commit-on-release drag state (lane clips + main clips leaving
    /// the track). One resolution per frame drives ghost, guides, tooltip,
    /// and the release commit.
    private struct TimelineDrag: Equatable {
        var content: DragContent
        var sourceRow: Int
        var anchorStart: TimeInterval
        var length: TimeInterval
        var kind: MockLane.Kind
        var label: String
        var symbol: String
        var art: MockArt?
        var effectKind: MockEffectClip.Kind?
        var audioSeed: Int?
        var translation: CGSize = .zero
        var location: CGPoint = .zero
    }

    /// One row of the rendered lane stack: the lane plus its resolved
    /// vertical slot. The same math sizes the stack and hit-tests fingers,
    /// so targeting can't drift from the layout.
    private struct LaneRow: Identifiable {
        var lane: MockLane
        var y: CGFloat
        var height: CGFloat
        var id: UUID { lane.id }
    }

    private static let rowSpacing: CGFloat = 5
    private static let rulerHeight: CGFloat = 18
    /// Video lanes read like footage; generated-content bars stay slim.
    private static let videoLaneHeight: CGFloat = 44

    private var pointsPerSecond: Double { 1 / state.secondsPerPoint }

    /// Compact height: ruler + main track only.
    static let minHeight: CGFloat = 118

    // MARK: Row geometry (recomputed per render; tiny n)

    private static func rowHeight(for lane: MockLane) -> CGFloat {
        if lane.isMain { return ClipView.height }
        return lane.kind == .video ? videoLaneHeight : LaneClipView.height
    }

    /// Vertical slots for every lane, straight from the ordered lane list.
    private var laneRowLayout: [LaneRow] {
        var rows: [LaneRow] = []
        var y: CGFloat = 0
        for lane in state.lanes {
            let height = Self.rowHeight(for: lane)
            rows.append(LaneRow(lane: lane, y: y, height: height))
            y += height + Self.rowSpacing
        }
        return rows
    }

    private func timelineNaturalHeight(rows: [LaneRow]) -> CGFloat {
        let bottom = rows.last.map { $0.y + $0.height } ?? 0
        return Self.rulerHeight + Self.rowSpacing + bottom + Self.rowSpacing + 16
    }

    private func clampedDisplayHeight(naturalHeight: CGFloat) -> CGFloat {
        let cap = max(maxTimelineHeight ?? naturalHeight, naturalHeight, Self.minHeight)
        return min(max(userHeight ?? naturalHeight, Self.minHeight), cap)
    }

    var body: some View {
        let rows = laneRowLayout
        let naturalHeight = timelineNaturalHeight(rows: rows)
        let displayHeight = clampedDisplayHeight(naturalHeight: naturalHeight)

        GeometryReader { geometry in
            let halfWidth = geometry.size.width / 2
            let contentWidth = max(0, state.duration * pointsPerSecond)
            let lanesViewport = max(displayHeight - Self.rulerHeight - Self.rowSpacing, 40)
            let lanesNatural = naturalHeight - Self.rulerHeight - Self.rowSpacing
            let maxLaneOffset = max(0, lanesNatural - lanesViewport)
            let effectiveLaneOffset = min(laneOffset, maxLaneOffset)

            // topLeading matters: .top would horizontally CENTER the
            // wider-than-viewport content before the -playhead offset
            // applies, silently shifting visual time away from the model.
            ZStack(alignment: .topLeading) {
                // Not a ScrollView: the content offsets by -playhead so the
                // fixed center line IS the playhead, and one drag gesture on
                // the whole surface owns panning (horizontal = scrub,
                // vertical = lane browse). UIScrollView's pan recognizer
                // raced the clip gestures and silently dropped pans that
                // started on clips, which is why scrubbing only worked from
                // the ruler.
                VStack(alignment: .leading, spacing: Self.rowSpacing) {
                    TimeRuler(duration: state.duration, pointsPerSecond: pointsPerSecond)
                        .frame(width: contentWidth, height: Self.rulerHeight, alignment: .leading)
                        .clipped()

                    VStack(alignment: .leading, spacing: Self.rowSpacing) {
                        ForEach(rows) { row in
                            if row.lane.isMain {
                                ZStack(alignment: .leading) {
                                    track
                                    transitionBoundaries
                                }
                                // Band tint above the clips (the background
                                // band is fully covered by them) while a
                                // video-lane clip hovers here.
                                .overlay {
                                    if timelineDrag.map(isMainInsertTarget) == true {
                                        Rectangle()
                                            .fill(Theme.accent.opacity(0.28))
                                            .allowsHitTesting(false)
                                    }
                                }
                                // The lifted square renders above the lane
                                // rows it is dragged across.
                                .zIndex(reorder != nil ? 5 : 0)
                            } else {
                                laneRowView(row, width: contentWidth)
                                    .opacity(reorder == nil || row.lane.kind == .video ? 1 : 0.35)
                                    .zIndex(rowHoldsLiftedClip(row) ? 6 : 0)
                            }
                        }
                    }
                    // Bands pan with the lane stack so they always line up
                    // with their rows.
                    .background(alignment: .top) {
                        laneBackground(rows: rows)
                    }
                    .overlay(alignment: .topLeading) {
                        dragOverlays(rows: rows, contentWidth: contentWidth)
                    }
                    .offset(y: -effectiveLaneOffset)
                    .frame(height: lanesViewport, alignment: .topLeading)
                    .clipped()
                }
                .overlay(alignment: .topLeading) { snapGuide }
                .frame(width: max(contentWidth, 1), alignment: .topLeading)
                .offset(x: halfWidth - state.playhead * pointsPerSecond)
            }
            .frame(width: geometry.size.width, height: displayHeight, alignment: .topLeading)
            .clipped()
            // Chrome sits outside the wide content ZStack: inside it these
            // flexible views would size to the full content width and land
            // offscreen; as overlays they pin to the visible viewport.
            .overlay { playheadLine }
            .overlay(alignment: .top) { readout }
            .overlay(alignment: .bottomTrailing) { magnetToggle }
            .contentShape(Rectangle())
            // Tapping anything that isn't a clip clears the selection.
            .onTapGesture { state.selection = nil }
            .gesture(timelinePanGesture(maxLaneOffset: maxLaneOffset))
            .simultaneousGesture(pinchToZoom)
            .sensoryFeedback(.impact(weight: .light), trigger: state.activeSnapTime) { _, newValue in
                newValue != nil
            }
        }
        .frame(height: displayHeight)
        .background(Theme.timelineBed)
        .clipped()
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("timeline")
        .onGeometryChange(for: CGRect.self) { proxy in
            proxy.frame(in: .global)
        } action: { frame in
            globalFrame = frame
        }
        .onChange(of: displayHeight) { _, newHeight in
            keepMainTrackVisible(displayHeight: newHeight, naturalHeight: naturalHeight)
        }
        .sensoryFeedback(.impact(weight: .medium), trigger: timelineDrag?.content) { _, newValue in
            newValue != nil
        }
        .sensoryFeedback(.success, trigger: conversionPulse)
        .onDisappear {
            decayTask?.cancel()
            autoscrollTask?.cancel()
        }
    }

    /// As the grab bar resizes the timeline, nudge the lane pan by the
    /// minimum needed to keep the main track fully inside the viewport, so
    /// collapsing always lands focused on the track (CapCut behavior). The
    /// user can still pan away afterwards.
    private func keepMainTrackVisible(displayHeight: CGFloat, naturalHeight: CGFloat) {
        let viewport = max(displayHeight - Self.rulerHeight - Self.rowSpacing, 40)
        let maxOffset = max(0, naturalHeight - Self.rulerHeight - Self.rowSpacing - viewport)
        guard let main = laneRowLayout.first(where: { $0.lane.isMain }) else { return }

        var target = min(max(0, laneOffset), maxOffset)
        if target > main.y { target = main.y }
        if target < main.y + main.height - viewport { target = main.y + main.height - viewport }
        laneOffset = min(max(0, target), maxOffset)
    }

    // MARK: Row hit-testing (cross-lane drag targets)

    /// Converts a global (screen) y to the lane stack's coordinate space,
    /// accounting for the ruler and the live lane pan.
    private func laneStackY(fromGlobalY y: CGFloat) -> CGFloat? {
        guard globalFrame != .zero else { return nil }
        let rows = laneRowLayout
        let natural = timelineNaturalHeight(rows: rows)
        let display = clampedDisplayHeight(naturalHeight: natural)
        let viewport = max(display - Self.rulerHeight - Self.rowSpacing, 40)
        let maxOffset = max(0, natural - Self.rulerHeight - Self.rowSpacing - viewport)
        let effectiveOffset = min(laneOffset, maxOffset)
        return y - globalFrame.minY - Self.rulerHeight - Self.rowSpacing + effectiveOffset
    }

    /// The lane-stack row under a stack-space y; -1 above the first row,
    /// `rows.count` below the last (both out-of-range rows resolve drags to
    /// new lanes at the stack's edges). Inter-row gaps split at their
    /// midpoint.
    private func rowIndex(atStackY y: CGFloat, rows: [LaneRow]) -> Int {
        if y < 0 { return -1 }
        for (index, row) in rows.enumerated() {
            if y < row.y + row.height + Self.rowSpacing / 2 { return index }
        }
        return rows.count
    }

    /// Row index under a global (screen) y, if the timeline frame is known.
    private func rowIndex(atGlobalY y: CGFloat) -> Int? {
        guard let stackY = laneStackY(fromGlobalY: y) else { return nil }
        return rowIndex(atStackY: stackY, rows: laneRowLayout)
    }

    private func isMainInsertTarget(_ drag: TimelineDrag) -> Bool {
        guard let resolution = resolution(for: drag) else { return false }
        if case .mainInsert = resolution.landing { return true }
        return false
    }

    /// Is the finger over a video lane (or above the stack, where a new one
    /// would be created)? Main-clip cross-lane drop target.
    private func videoLaneDropTarget(globalY: CGFloat) -> Bool {
        guard let row = rowIndex(atGlobalY: globalY) else { return false }
        if row == -1 { return true }
        let rows = laneRowLayout
        guard rows.indices.contains(row) else { return false }
        return rows[row].lane.kind == .video && !rows[row].lane.isMain
    }

    /// Is the finger over the main track (video-lane clip drop target)?
    private func mainDropTarget(globalY: CGFloat) -> Bool {
        guard let row = rowIndex(atGlobalY: globalY) else { return false }
        let rows = laneRowLayout
        return rows.indices.contains(row) && rows[row].lane.isMain
    }

    /// Timeline time under a global (screen) x; the fixed center is the
    /// playhead.
    private func timeAt(globalX x: CGFloat) -> TimeInterval {
        guard globalFrame.width > 0 else { return state.playhead }
        return max(0, state.playhead + (x - globalFrame.midX) * state.secondsPerPoint)
    }

    private var draggedClipID: UUID? {
        switch timelineDrag?.content {
        case .main(let id):
            return id
        case .lane(.overlay(let id)), .lane(.effect(let id)), .lane(.audio(let id)):
            return id
        case .lane(.main), nil:
            return nil
        }
    }

    // MARK: Commit-on-release drag

    /// One free lane's row: a time-wide bed with its clips offset to their
    /// start times.
    @ViewBuilder
    private func laneRowView(_ row: LaneRow, width: CGFloat) -> some View {
        ZStack(alignment: .topLeading) {
            Color.clear
                .frame(width: max(width, 1), height: row.height)

            switch row.lane.kind {
            case .video, .text, .sticker:
                ForEach(state.overlayClips.filter { $0.laneID == row.lane.id }) { clip in
                    overlayClipView(clip, rowHeight: row.height)
                        .zIndex(clip.id == draggedClipID ? 2 : 0)
                }
            case .effect:
                ForEach(state.effectClips.filter { $0.laneID == row.lane.id }) { clip in
                    effectClipView(clip)
                        .zIndex(clip.id == draggedClipID ? 2 : 0)
                }
            case .audio:
                ForEach(state.audioClips.filter { $0.laneID == row.lane.id }) { clip in
                    audioClipView(clip)
                        .zIndex(clip.id == draggedClipID ? 2 : 0)
                }
            }
        }
        .frame(width: max(width, 1), height: row.height, alignment: .topLeading)
    }

    private func rowHoldsLiftedClip(_ row: LaneRow) -> Bool {
        guard let lifted = draggedClipID else { return false }
        return state.overlayClips.contains { $0.id == lifted && $0.laneID == row.lane.id }
            || state.effectClips.contains { $0.id == lifted && $0.laneID == row.lane.id }
            || state.audioClips.contains { $0.id == lifted && $0.laneID == row.lane.id }
    }

    private func laneDragCallbacks(
        content: DragContent,
        row: Int,
        anchorStart: TimeInterval,
        length: TimeInterval,
        kind: MockLane.Kind,
        label: String,
        symbol: String,
        art: MockArt? = nil,
        effectKind: MockEffectClip.Kind? = nil,
        audioSeed: Int? = nil
    ) -> (begin: () -> Void, changed: (CGSize, CGPoint) -> Void, ended: (CGPoint?) -> Void) {
        (
            begin: {
                state.beginDragGesture()
                state.isPlaying = false
                timelineDrag = TimelineDrag(
                    content: content,
                    sourceRow: row,
                    anchorStart: anchorStart,
                    length: length,
                    kind: kind,
                    label: label,
                    symbol: symbol,
                    art: art,
                    effectKind: effectKind,
                    audioSeed: audioSeed
                )
                startAutoscrollLoop()
            },
            changed: { translation, location in
                updateTimelineDrag(translation: translation, location: location)
            },
            ended: { location in
                endTimelineDrag(location: location)
            }
        )
    }

    private func updateTimelineDrag(translation: CGSize, location: CGPoint) {
        guard var drag = timelineDrag else { return }
        drag.translation = translation
        drag.location = location
        timelineDrag = drag
    }

    private func endTimelineDrag(location: CGPoint?) {
        autoscrollTask?.cancel()
        autoscrollTask = nil
        defer { timelineDrag = nil; state.activeSnapTime = nil }

        guard let drag = timelineDrag else { return }
        let loc = location ?? drag.location
        var hoverRow = rowIndex(atGlobalY: loc.y) ?? drag.sourceRow
        if mainDropTarget(globalY: loc.y) {
            hoverRow = laneRowLayout.firstIndex(where: { $0.lane.isMain }) ?? hoverRow
        } else if videoLaneDropTarget(globalY: loc.y) {
            hoverRow = laneRowLayout.firstIndex(where: { !$0.lane.isMain && $0.lane.kind == .video }) ?? hoverRow
        }
        let desiredStart = max(0, drag.anchorStart + drag.translation.width / pointsPerSecond)
        guard let resolution = state.resolveDrag(
            content: drag.content,
            desiredStart: desiredStart,
            hoverRow: hoverRow
        ) else {
            state.endGesture()
            return
        }
        let changed = !resolution.isNoop
        state.commitDrag(content: drag.content, resolution: resolution)
        if changed { conversionPulse += 1 }
    }

    private func resolution(for drag: TimelineDrag) -> DragResolution? {
        let hoverRow = rowIndex(atGlobalY: drag.location.y) ?? drag.sourceRow
        let desiredStart = max(0, drag.anchorStart + drag.translation.width / pointsPerSecond)
        return state.resolveDrag(content: drag.content, desiredStart: desiredStart, hoverRow: hoverRow)
    }

    private func startAutoscrollLoop() {
        autoscrollTask?.cancel()
        autoscrollTask = Task { @MainActor in
            while !Task.isCancelled, timelineDrag != nil {
                try? await Task.sleep(for: .milliseconds(16))
                guard !Task.isCancelled, var drag = timelineDrag, globalFrame != .zero else { continue }

                let margin: CGFloat = 28
                let maxStep: CGFloat = 14
                var dx: CGFloat = 0
                var dy: CGFloat = 0

                if drag.location.x < globalFrame.minX + margin {
                    let depth = min(1, (globalFrame.minX + margin - drag.location.x) / margin)
                    dx = maxStep * depth
                } else if drag.location.x > globalFrame.maxX - margin {
                    let depth = min(1, (drag.location.x - (globalFrame.maxX - margin)) / margin)
                    dx = -maxStep * depth
                }

                let rows = laneRowLayout
                let natural = timelineNaturalHeight(rows: rows)
                let display = clampedDisplayHeight(naturalHeight: natural)
                let viewport = max(display - Self.rulerHeight - Self.rowSpacing, 40)
                let maxOffset = max(0, natural - Self.rulerHeight - Self.rowSpacing - viewport)
                let stackTop = globalFrame.minY + Self.rulerHeight + Self.rowSpacing
                let stackBottom = stackTop + viewport

                if drag.location.y < stackTop + margin {
                    let depth = min(1, (stackTop + margin - drag.location.y) / margin)
                    dy = maxStep * depth
                } else if drag.location.y > stackBottom - margin {
                    let depth = min(1, (drag.location.y - (stackBottom - margin)) / margin)
                    dy = -maxStep * depth
                }

                if dx != 0 {
                    let before = state.playhead
                    state.playhead = min(max(0, state.playhead - dx * state.secondsPerPoint), state.duration)
                    drag.translation.width += (state.playhead - before) / state.secondsPerPoint
                }
                if dy != 0 {
                    let before = laneOffset
                    laneOffset = min(max(0, laneOffset - dy), maxOffset)
                    drag.translation.height += laneOffset - before
                }
                timelineDrag = drag
            }
        }
    }

    private func overlayClipView(_ clip: MockOverlayClip, rowHeight: CGFloat) -> LaneClipView {
        let style: LaneClipView.Style
        let symbol: String
        let label: String
        switch clip.kind {
        case .text:
            style = .text
            symbol = "textformat"
            label = clip.displayLabel
        case .sticker:
            style = .sticker
            symbol = clip.symbol ?? "face.smiling"
            label = clip.displayLabel
        case .pip:
            // Video-lane clips read like footage: thumbnails + duration, not
            // an "Overlay" identity.
            style = .pip(clip.art)
            symbol = "video.fill"
            label = String(format: "%.1fs", clip.length)
        }
        let row = laneRowLayout.firstIndex(where: { $0.lane.id == clip.laneID }) ?? 0
        let drag = laneDragCallbacks(
            content: .lane(.overlay(clip.id)),
            row: row,
            anchorStart: clip.start,
            length: clip.length,
            kind: clip.laneKind,
            label: label,
            symbol: symbol,
            art: clip.kind == .pip ? clip.art : nil
        )
        return LaneClipView(
            style: style,
            label: label,
            symbol: symbol,
            start: clip.start,
            length: clip.length,
            pointsPerSecond: pointsPerSecond,
            isSelected: state.selection == .overlay(clip.id),
            isMuted: clip.kind == .pip && clip.volume == 0,
            rowHeight: rowHeight,
            isBeingDragged: timelineDrag?.content == .lane(.overlay(clip.id)),
            accessibilityIdentifier: clip.kind == .pip ? "videoLaneClip" : "laneClip",
            onTap: { toggleSelection(.overlay(clip.id)) },
            onTrim: { edge, anchorStart, anchorLength, delta in
                state.trimLaneClip(.overlay(clip.id), edge: edge, anchorStart: anchorStart, anchorLength: anchorLength, by: delta)
            },
            onDragBegin: drag.begin,
            onDragChanged: drag.changed,
            onDragEnded: drag.ended,
            onGestureEnd: { state.endGesture() }
        )
    }

    private func effectClipView(_ clip: MockEffectClip) -> LaneClipView {
        let row = laneRowLayout.firstIndex(where: { $0.lane.id == clip.laneID }) ?? 0
        let drag = laneDragCallbacks(
            content: .lane(.effect(clip.id)),
            row: row,
            anchorStart: clip.start,
            length: clip.length,
            kind: .effect,
            label: clip.displayLabel,
            symbol: clip.kind == .adjust ? "slider.horizontal.3" : "sparkles",
            effectKind: clip.kind
        )
        return LaneClipView(
            style: .effect(clip.kind),
            label: clip.displayLabel,
            symbol: clip.kind == .adjust ? "slider.horizontal.3" : "sparkles",
            start: clip.start,
            length: clip.length,
            pointsPerSecond: pointsPerSecond,
            isSelected: state.selection == .effect(clip.id),
            isBeingDragged: timelineDrag?.content == .lane(.effect(clip.id)),
            onTap: { toggleSelection(.effect(clip.id)) },
            onTrim: { edge, anchorStart, anchorLength, delta in
                state.trimLaneClip(.effect(clip.id), edge: edge, anchorStart: anchorStart, anchorLength: anchorLength, by: delta)
            },
            onDragBegin: drag.begin,
            onDragChanged: drag.changed,
            onDragEnded: drag.ended,
            onGestureEnd: { state.endGesture() }
        )
    }

    private func audioClipView(_ clip: MockAudioClip) -> LaneClipView {
        let row = laneRowLayout.firstIndex(where: { $0.lane.id == clip.laneID }) ?? 0
        let drag = laneDragCallbacks(
            content: .lane(.audio(clip.id)),
            row: row,
            anchorStart: clip.start,
            length: clip.length,
            kind: .audio,
            label: clip.title,
            symbol: clip.symbol,
            audioSeed: clip.waveSeed
        )
        return LaneClipView(
            style: .audio(seed: clip.waveSeed),
            label: clip.title,
            symbol: clip.symbol,
            start: clip.start,
            length: clip.length,
            pointsPerSecond: pointsPerSecond,
            isSelected: state.selection == .audio(clip.id),
            isMuted: clip.volume == 0,
            isBeingDragged: timelineDrag?.content == .lane(.audio(clip.id)),
            onTap: { toggleSelection(.audio(clip.id)) },
            onTrim: { edge, anchorStart, anchorLength, delta in
                state.trimLaneClip(.audio(clip.id), edge: edge, anchorStart: anchorStart, anchorLength: anchorLength, by: delta)
            },
            onDragBegin: drag.begin,
            onDragChanged: drag.changed,
            onDragEnded: drag.ended,
            onGestureEnd: { state.endGesture() }
        )
    }

    private func toggleSelection(_ target: TimelineSelection) {
        state.selection = state.selection == target ? nil : target
    }

    private var track: some View {
        let widths = state.clips.map { CGFloat($0.length * pointsPerSecond) }

        return HStack(spacing: 0) {
            ForEach(Array(state.clips.enumerated()), id: \.element.id) { index, clip in
                let isLifted = reorder?.clipID == clip.id

                ClipView(
                    clip: clip,
                    pointsPerSecond: pointsPerSecond,
                    isSelected: state.selection == .main(clip.id),
                    sortMode: reorder != nil,
                    onTap: { toggleSelection(.main(clip.id)) },
                    onTrim: { edge, anchor, delta in
                        state.trim(clip.id, edge: edge, anchor: anchor, by: delta)
                    },
                    onTrimEnd: { state.endGesture() }
                )
                .scaleEffect(isLifted ? 1.07 : 1)
                .shadow(color: .black.opacity(isLifted ? 0.55 : 0), radius: 10, y: 3)
                .offset(reorderTileOffset(index: index, clip: clip, timeWidths: widths))
                .animation(
                    isLifted ? nil : .easeOut(duration: 0.18),
                    value: reorder?.insertionIndex
                )
                // Morph between time widths and square tiles on lift/drop.
                .animation(.snappy(duration: 0.22), value: reorder != nil)
                .zIndex(isLifted ? 10 : 0)
                .gesture(reorderGesture(clip: clip, index: index, timeWidths: widths))
            }

            Button(action: onAddMedia) {
                RoundedRectangle(cornerRadius: 5, style: .continuous)
                    .fill(Theme.surfaceElevated)
                    .frame(width: 40, height: ClipView.height)
                    .overlay {
                        Image(systemName: "plus")
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(.white)
                    }
            }
            .buttonStyle(.plain)
            .padding(.leading, state.isEmpty ? 0 : 4)
            .opacity(reorder == nil ? 1 : 0)
        }
        .sensoryFeedback(.impact(weight: .medium), trigger: reorder != nil) { _, lifted in
            lifted
        }
    }

    // MARK: Timeline pan (scrub / lane browse)

    /// One direction-locked drag for the whole timeline surface: horizontal
    /// pans scrub (translation -> playhead, with flick momentum), vertical
    /// pans browse the lane stack inside its clipped viewport.
    private func timelinePanGesture(maxLaneOffset: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 10, coordinateSpace: .global)
            .onChanged { value in
                if panAnchor == nil {
                    decayTask?.cancel()
                    state.isPlaying = false
                    panAnchor = (state.playhead, laneOffset)
                    panMode = abs(value.translation.height) > abs(value.translation.width)
                        ? .lanes
                        : .scrub
                }
                guard let anchor = panAnchor else { return }

                switch panMode {
                case .scrub:
                    let time = anchor.playhead - value.translation.width * state.secondsPerPoint
                    state.playhead = min(max(0, time), state.duration)
                case .lanes:
                    laneOffset = min(max(0, anchor.lane - value.translation.height), maxLaneOffset)
                case nil:
                    break
                }
            }
            .onEnded { value in
                let mode = panMode
                panAnchor = nil
                panMode = nil
                guard mode == .scrub else { return }

                // Flick momentum: glide through the distance UIKit predicts
                // beyond the finger-tracked translation, decaying to a stop.
                let extra = value.predictedEndTranslation.width - value.translation.width
                startScrubDecay(distance: -extra * state.secondsPerPoint)
            }
    }

    /// Advances the playhead over `distance` seconds with an ease-out decay,
    /// keeping the model truthful frame by frame (readout ticks, snapping
    /// stays live). Cancelled by any new pan or playback.
    private func startScrubDecay(distance: TimeInterval) {
        guard abs(distance) > 0.05 else { return }
        decayTask?.cancel()
        decayTask = Task { @MainActor in
            let duration: TimeInterval = 0.55
            let start = ContinuousClock.now
            var lastProgress: Double = 0
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(16))
                if Task.isCancelled || state.isPlaying { return }
                let elapsed = min((ContinuousClock.now - start) / .seconds(duration), 1)
                // Ease-out cubic: fast start, glides to rest.
                let progress = 1 - pow(1 - elapsed, 3)
                let step = (progress - lastProgress) * distance
                lastProgress = progress
                state.playhead = min(max(0, state.playhead + step), state.duration)
                if elapsed >= 1 || state.playhead == 0 || state.playhead == state.duration {
                    return
                }
            }
        }
    }

    // MARK: Drag to reorder (square sort mode) / drop onto the overlay band

    /// Long-press lifts a main-track clip and morphs the whole track into
    /// uniform square tiles; horizontal drag slides an insertion gap through
    /// the neighbors, dragging down onto the overlay band converts the clip
    /// to a PiP overlay at the drop time.
    private func reorderGesture(clip: MockClip, index: Int, timeWidths: [CGFloat]) -> some Gesture {
        // Global coordinate space: the lifted clip follows the finger, so a
        // local-space translation would feed back on itself. The tight
        // maximumDistance matters: anything looser swallows slow-starting
        // scrub pans and the timeline "stops scrolling" from the tracks.
        LongPressGesture(minimumDuration: 0.35, maximumDistance: 12)
            .sequenced(before: DragGesture(minimumDistance: 0, coordinateSpace: .global))
            .onChanged { value in
                guard case .second(true, let drag) = value else { return }
                state.isPlaying = false
                var current = reorder ?? ReorderDrag(
                    clipID: clip.id,
                    fromIndex: index,
                    // Anchor the square strip so the lifted tile morphs in
                    // place of the original clip's leading edge.
                    stripShift: timeWidths.prefix(index).reduce(0, +) - CGFloat(index) * ClipView.height,
                    insertionIndex: index
                )
                if let drag {
                    current.translation = drag.translation
                    current.location = drag.location
                    current.overOverlayBand = videoLaneDropTarget(globalY: drag.location.y)
                    current.insertionIndex = insertionIndex(
                        fromIndex: index,
                        translation: drag.translation.width,
                        widths: Array(repeating: ClipView.height, count: timeWidths.count)
                    )
                }
                reorder = current
            }
            .onEnded { value in
                defer { reorder = nil }
                guard case .second(true, _) = value, let final = reorder else { return }
                if final.overOverlayBand {
                    state.beginDragGesture()
                    let hoverRow = rowIndex(atGlobalY: final.location.y) ?? -1
                    let desiredStart = max(0, state.startTime(of: final.clipID) + final.translation.width / pointsPerSecond)
                    if let resolution = state.resolveDrag(
                        content: .main(final.clipID),
                        desiredStart: desiredStart,
                        hoverRow: hoverRow
                    ) {
                        state.commitDrag(content: .main(final.clipID), resolution: resolution)
                        if !resolution.isNoop { conversionPulse += 1 }
                    } else {
                        state.endGesture()
                    }
                } else {
                    state.moveClip(fromIndex: final.fromIndex, toIndex: final.insertionIndex)
                }
            }
    }

    /// Where the dragged clip would land: how many other clips' centers lie
    /// left of the dragged clip's current center.
    private func insertionIndex(fromIndex: Int, translation: CGFloat, widths: [CGFloat]) -> Int {
        let leading = widths.prefix(fromIndex).reduce(0, +)
        let draggedCenter = leading + widths[fromIndex] / 2 + translation

        var slot = 0
        var x: CGFloat = 0
        for (index, width) in widths.enumerated() {
            if index != fromIndex, x + width / 2 < draggedCenter {
                slot += 1
            }
            x += width
        }
        return slot
    }

    /// Tile offsets while sorting: every tile shifts by the strip anchor,
    /// the lifted one follows the finger in 2D, and the others slide to
    /// visualize the removal + insertion gap (gap hidden while the drop
    /// would convert to an overlay instead of reordering).
    private func reorderTileOffset(index: Int, clip: MockClip, timeWidths: [CGFloat]) -> CGSize {
        guard let reorder, timeWidths.indices.contains(reorder.fromIndex) else { return .zero }
        let square = ClipView.height
        if reorder.clipID == clip.id {
            return CGSize(
                width: reorder.stripShift + reorder.translation.width,
                height: reorder.translation.height
            )
        }

        var x = reorder.stripShift
        let postRemovalIndex = index > reorder.fromIndex ? index - 1 : index
        if index > reorder.fromIndex {
            x -= square
        }
        if !reorder.overOverlayBand, postRemovalIndex >= reorder.insertionIndex {
            x += square
        }
        return CGSize(width: x, height: 0)
    }

    /// Small boundary buttons between adjacent main-track clips; accent when
    /// a transition is set. Hidden while a main clip is selected so they
    /// don't fight the trim handles.
    @ViewBuilder
    private var transitionBoundaries: some View {
        if state.clips.count > 1, state.selectedClip == nil, reorder == nil {
            ForEach(state.clips.dropLast()) { clip in
                let isSet = clip.transitionAfter != nil
                let boundaryX = (state.startTime(of: clip.id) + clip.length) * pointsPerSecond

                Button {
                    onTransitionTap(clip.id)
                } label: {
                    RoundedRectangle(cornerRadius: 5, style: .continuous)
                        .fill(isSet ? Theme.accent : .white)
                        .frame(width: 20, height: 20)
                        .overlay {
                            Image(systemName: "arrow.left.arrow.right")
                                .font(.system(size: 8, weight: .heavy))
                                .foregroundStyle(isSet ? .white : .black)
                        }
                        .shadow(color: .black.opacity(0.4), radius: 2)
                }
                .buttonStyle(.plain)
                .offset(x: boundaryX - 10)
            }
        }
    }

    /// Full-width dim bands behind the scrolling rows, one per lane (ruler
    /// excluded; it sits above the lane scroll view). During cross-lane
    /// drags the hovered target band tints accent and bystander bands dim.
    private func laneBackground(rows: [LaneRow]) -> some View {
        let sortActive = reorder != nil
        let videoTargeted = reorder?.overOverlayBand == true
        let highlightMain = timelineDrag.map(isMainInsertTarget) == true

        return VStack(spacing: Self.rowSpacing) {
            ForEach(rows) { row in
                if row.lane.isMain {
                    Rectangle().fill(Theme.trackEmpty)
                        .overlay {
                            if highlightMain {
                                Theme.accent.opacity(0.3)
                            }
                        }
                        .frame(height: row.height)
                } else {
                    let highlighted = row.lane.kind == .video && videoTargeted
                    Rectangle()
                        .fill(Theme.trackEmpty.opacity(highlighted ? 0.95 : 0.7))
                        .overlay {
                            if highlighted {
                                Theme.accent.opacity(0.32)
                            }
                        }
                        .frame(height: row.height)
                        .opacity(sortActive && !highlighted ? 0.35 : 1)
                }
            }
        }
    }

    // MARK: Drag overlays (ghost, floater, guides — desktop snap.rs visuals)

    @ViewBuilder
    private func dragOverlays(rows: [LaneRow], contentWidth: CGFloat) -> some View {
        if let drag = timelineDrag, let resolution = resolution(for: drag) {
            let ghostWidth = drag.length * pointsPerSecond

            // Full-height vertical snap line.
            if let snapTime = resolution.snapTime {
                Rectangle()
                    .fill(Theme.accent)
                    .frame(width: 1.5)
                    .frame(maxHeight: .infinity)
                    .offset(x: snapTime * pointsPerSecond - 0.75)
                    .allowsHitTesting(false)
            }

            switch resolution.landing {
            case .land(let laneID, let start):
                if let row = rows.first(where: { $0.lane.id == laneID }) {
                    dragGhostRect(
                        x: start * pointsPerSecond,
                        y: row.y,
                        width: ghostWidth,
                        height: row.height,
                        accent: false
                    )
                }
            case .newLane(let row, let start):
                let gapY = gapY(forRow: row, rows: rows)
                Rectangle()
                    .fill(Theme.accent)
                    .frame(width: max(contentWidth, 1), height: 2)
                    .offset(y: gapY - 1)
                    .allowsHitTesting(false)
                dragGhostRect(
                    x: start * pointsPerSecond,
                    y: gapY - Self.videoLaneHeight * 0.25,
                    width: ghostWidth,
                    height: Self.videoLaneHeight * 0.5,
                    accent: true
                )
            case .mainInsert(_, let caretTime):
                if let main = rows.first(where: { $0.lane.isMain }) {
                    RoundedRectangle(cornerRadius: 1.5, style: .continuous)
                        .fill(Theme.accent)
                        .frame(width: 3, height: main.height)
                        .offset(x: caretTime * pointsPerSecond - 1.5, y: main.y)
                        .allowsHitTesting(false)
                }
            }

            // Floating copy following the finger.
            if let source = rows.indices.contains(drag.sourceRow) ? rows[drag.sourceRow] : nil {
                dragFloater(drag: drag, row: source, width: ghostWidth)
                    .offset(
                        x: drag.anchorStart * pointsPerSecond + drag.translation.width,
                        y: source.y + drag.translation.height
                    )
            }

            // Landing timecode above the floater.
            dragTooltip(for: drag, resolution: resolution)
                .offset(
                    x: drag.anchorStart * pointsPerSecond + drag.translation.width,
                    y: (rows.indices.contains(drag.sourceRow) ? rows[drag.sourceRow].y : 0) + drag.translation.height - 22
                )
        }
    }

    private func gapY(forRow row: Int, rows: [LaneRow]) -> CGFloat {
        if row <= 0 { return -(Self.rowSpacing / 2) }
        if row >= rows.count {
            return (rows.last?.y ?? 0) + (rows.last?.height ?? 0) + Self.rowSpacing / 2
        }
        let above = rows[row - 1]
        return above.y + above.height + Self.rowSpacing / 2
    }

    private func dragGhostRect(x: CGFloat, y: CGFloat, width: CGFloat, height: CGFloat, accent: Bool) -> some View {
        RoundedRectangle(cornerRadius: 6, style: .continuous)
            .fill(dragGhostFill(accent: accent))
            .overlay {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(accent ? Theme.accent : .white, lineWidth: 1)
            }
            .frame(width: max(width, 6), height: height)
            .offset(x: x, y: y)
            .allowsHitTesting(false)
    }

    private func dragGhostFill(accent: Bool) -> Color {
        accent ? Theme.accent.opacity(0.3) : Color.white.opacity(0.28)
    }

    @ViewBuilder
    private func dragFloater(drag: TimelineDrag, row: LaneRow, width: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: 6, style: .continuous)
            .fill(floaterColor(for: drag))
            .overlay(alignment: .leading) {
                HStack(spacing: 4) {
                    Image(systemName: drag.symbol)
                        .font(.system(size: 9, weight: .bold))
                    Text(drag.label)
                        .font(.system(size: 10, weight: .medium))
                        .lineLimit(1)
                }
                .foregroundStyle(.white.opacity(0.95))
                .padding(.horizontal, 6)
            }
            .overlay {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(.white, lineWidth: 1)
            }
            .frame(width: max(width, 6), height: row.height)
            .opacity(0.85)
            .shadow(color: .black.opacity(0.45), radius: 8, y: 3)
            .allowsHitTesting(false)
    }

    private func floaterColor(for drag: TimelineDrag) -> Color {
        switch drag.kind {
        case .video: return Color(hex: 0x155E75)
        case .text: return Color(hex: 0x4338CA)
        case .sticker: return Color(hex: 0x9D174D)
        case .effect: return Color(hex: 0x7C3AED)
        case .audio: return Color(hex: 0x0C2733)
        }
    }

    private func dragTooltip(for drag: TimelineDrag, resolution: DragResolution) -> some View {
        let time: TimeInterval
        switch resolution.landing {
        case .land(_, let start):
            time = start
        case .newLane(_, let start):
            time = start
        case .mainInsert(_, let caretTime):
            time = caretTime
        }
        return Text(time.timecode)
            .font(.caption2.weight(.semibold).monospacedDigit())
            .foregroundStyle(.white)
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(Color.black.opacity(0.82), in: RoundedRectangle(cornerRadius: 4, style: .continuous))
            .fixedSize()
            .allowsHitTesting(false)
    }

    // MARK: Overlays

    /// Yellow guide line through all lanes while a gesture is locked onto a
    /// snap candidate.
    @ViewBuilder
    private var snapGuide: some View {
        if let time = state.activeSnapTime {
            Rectangle()
                .fill(Color(hex: 0xFACC15))
                .frame(width: 1.5)
                .frame(maxHeight: .infinity)
                .offset(x: time * pointsPerSecond - 0.75)
                .allowsHitTesting(false)
        }
    }

    /// Magnet chip pinned to the timeline's corner; accent when snapping is
    /// on, dim when free-dragging.
    private var magnetToggle: some View {
        Button {
            state.magnetEnabled.toggle()
        } label: {
            Image(systemName: state.magnetEnabled ? "dot.squareshape.split.2x2" : "squareshape.split.2x2.dotted")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(state.magnetEnabled ? .white : Theme.textTertiary)
                .frame(width: 26, height: 26)
                .background(
                    state.magnetEnabled ? AnyShapeStyle(Theme.accent) : AnyShapeStyle(Theme.surfaceElevated),
                    in: RoundedRectangle(cornerRadius: 7, style: .continuous)
                )
                .shadow(color: .black.opacity(0.35), radius: 3)
        }
        .buttonStyle(.plain)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
        .padding(.trailing, 8)
        .padding(.bottom, 8)
    }

    private var playheadLine: some View {
        RoundedRectangle(cornerRadius: 1)
            .fill(.white)
            .frame(width: 2)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
            .allowsHitTesting(false)
    }

    private var readout: some View {
        HStack(spacing: 3) {
            Text(state.playhead.timecode)
                .foregroundStyle(.white)
            Text("/ \(state.duration.timecode)")
                .foregroundStyle(Theme.textTertiary)
        }
        .font(.caption2.weight(.semibold).monospacedDigit())
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(Theme.timelineBed.opacity(0.9), in: Capsule())
        .frame(maxWidth: .infinity, alignment: .trailing)
        .padding(.trailing, 8)
        .allowsHitTesting(false)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("playheadReadout")
    }

    // MARK: Gestures and sync

    private var pinchToZoom: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                let base = pinchBase ?? state.secondsPerPoint
                pinchBase = base
                // More magnification = fewer seconds per point (zoom in).
                let pps = min(max((1 / base) * value.magnification, 10), 150)
                state.secondsPerPoint = 1 / pps
            }
            .onEnded { _ in
                pinchBase = nil
            }
    }
}

#Preview {
    let state = EditorState()
    let _ = state.startProject(with: FixtureLibrary.sampleTimeline)
    let _ = state.addSticker(symbol: "heart.fill")
    let _ = state.addAudio(kind: .music, title: "Slow Morning", duration: 30)
    return TimelineView(state: state, onAddMedia: {})
        .background(Theme.background)
}
