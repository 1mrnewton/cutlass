import SwiftUI

/// The rows under a fixed playhead in the middle of the strip.
///
/// Dragging sideways moves time under the playhead; a pinch changes how many
/// points a second takes. A tap on a clip selects it, and a tap on empty
/// space clears the selection.
struct TimelineStrip: View {
    @Bindable var model: EditorModel
    @State private var pointsPerSecond: CGFloat = 60
    @State private var dragStart: Int64?
    @State private var pinchStart: CGFloat?

    private let rulerHeight: CGFloat = 22
    private let laneGap: CGFloat = 6

    var body: some View {
        GeometryReader { geometry in
            let middle = geometry.size.width / 2
            let origin = middle - x(model.playheadUs)
            ZStack(alignment: .topLeading) {
                Color.clear.contentShape(Rectangle())
                ruler(width: geometry.size.width, origin: origin)
                VStack(alignment: .leading, spacing: laneGap) {
                    ForEach(Array(model.lanes.enumerated()), id: \.offset) { _, lane in
                        row(lane, origin: origin)
                    }
                }
                .padding(.top, rulerHeight + 8)
                Rectangle()
                    .fill(.white)
                    .frame(width: 2)
                    .offset(x: middle - 1)
                    .allowsHitTesting(false)
            }
            .clipped()
            .gesture(scrub)
            .simultaneousGesture(zoom)
            .onTapGesture { model.selectedClip = nil }
        }
    }

    private func x(_ time: Int64) -> CGFloat {
        CGFloat(time) / 1_000_000 * pointsPerSecond
    }

    private func ruler(width: CGFloat, origin: CGFloat) -> some View {
        Canvas { context, size in
            let step = rulerStep
            let first = max(0, Int((-origin / pointsPerSecond) / step))
            let last = Int(((width - origin) / pointsPerSecond) / step) + 1
            guard first <= last else { return }
            for index in first...last {
                let seconds = Double(index) * step
                let position = origin + CGFloat(seconds) * pointsPerSecond
                var tick = Path()
                tick.move(to: CGPoint(x: position, y: size.height - 6))
                tick.addLine(to: CGPoint(x: position, y: size.height))
                context.stroke(tick, with: .color(.white.opacity(0.35)), lineWidth: 1)
                context.draw(
                    Text(Self.clock(seconds)).font(.system(size: 10).monospacedDigit())
                        .foregroundStyle(.white.opacity(0.5)),
                    at: CGPoint(x: position, y: 7)
                )
            }
        }
        .frame(height: rulerHeight)
        .allowsHitTesting(false)
    }

    /// Seconds between labels, so labels stay about 80 points apart.
    private var rulerStep: Double {
        let wanted = 80 / Double(pointsPerSecond)
        return [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600].first { $0 >= wanted } ?? 1200
    }

    static func clock(_ seconds: Double) -> String {
        let whole = Int(seconds)
        let fraction = seconds - Double(whole)
        let text = String(format: "%02d:%02d", whole / 60, whole % 60)
        return fraction > 0 ? text + String(format: ".%d", Int(fraction * 10)) : text
    }

    private func row(_ lane: Lane, origin: CGFloat) -> some View {
        let height: CGFloat = lane.main ? 56 : 40
        return ZStack(alignment: .topLeading) {
            Color.clear.frame(height: height)
            ForEach(lane.clips, id: \.id) { clip in
                ClipBlock(
                    clip: clip,
                    kind: lane.kind,
                    selected: model.selectedClip == clip.id,
                    height: height
                )
                .frame(width: max(x(clip.durationUs) - 2, 2), height: height)
                .offset(x: origin + x(clip.startUs) + 1)
                .onTapGesture { model.selectedClip = clip.id }
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

    private var scrub: some Gesture {
        DragGesture(minimumDistance: 4)
            .onChanged { drag in
                if dragStart == nil {
                    model.pause()
                    dragStart = model.playheadUs
                }
                let start = dragStart ?? model.playheadUs
                let moved = Int64(-drag.translation.width / pointsPerSecond * 1_000_000)
                model.seek(start + moved)
            }
            .onEnded { _ in dragStart = nil }
    }

    private var zoom: some Gesture {
        MagnifyGesture()
            .onChanged { pinch in
                let start = pinchStart ?? pointsPerSecond
                pinchStart = start
                pointsPerSecond = min(max(start * pinch.magnification, 8), 600)
            }
            .onEnded { _ in pinchStart = nil }
    }
}

/// One clip: its color by kind, its name, and a white edge when selected.
struct ClipBlock: View {
    let clip: Clip
    let kind: LaneKind
    let selected: Bool
    let height: CGFloat

    var body: some View {
        RoundedRectangle(cornerRadius: 6)
            .fill(fill)
            .overlay(alignment: .topLeading) {
                Text(clip.name)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.white.opacity(0.9))
                    .lineLimit(1)
                    .padding(.horizontal, 6)
                    .padding(.top, 4)
            }
            .overlay {
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(.white, lineWidth: selected ? 2 : 0)
            }
            .contentShape(Rectangle())
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
