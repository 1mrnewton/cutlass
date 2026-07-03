import SwiftUI

/// The scrubbable timeline: a fixed center playhead over horizontally
/// scrolling content (ruler + clip track), where scroll offset IS the
/// current time. Dim full-width lanes above/below hint at future tracks.
struct TimelineView: View {
    var state: EditorState
    var onAddMedia: () -> Void

    @State private var scrollPosition = ScrollPosition(edge: .leading)
    @State private var scrollPhase: ScrollPhase = .idle
    /// seconds-per-point captured when a pinch begins.
    @State private var pinchBase: Double?

    private static let rowSpacing: CGFloat = 6
    private static let rulerHeight: CGFloat = 18
    private static let emptyLaneHeight: CGFloat = 24

    private var pointsPerSecond: Double { 1 / state.secondsPerPoint }

    private var isUserScrolling: Bool {
        scrollPhase == .tracking || scrollPhase == .interacting || scrollPhase == .decelerating
    }

    var body: some View {
        GeometryReader { geometry in
            let halfWidth = geometry.size.width / 2

            ZStack(alignment: .top) {
                laneBackground

                ScrollView(.horizontal, showsIndicators: false) {
                    VStack(alignment: .leading, spacing: Self.rowSpacing) {
                        TimeRuler(duration: state.duration, pointsPerSecond: pointsPerSecond)
                            .frame(
                                width: max(0, state.duration * pointsPerSecond),
                                height: Self.rulerHeight,
                                alignment: .leading
                            )
                            .clipped()

                        Color.clear.frame(height: Self.emptyLaneHeight)

                        track
                    }
                    .padding(.horizontal, halfWidth)
                }
                .scrollPosition($scrollPosition)
                .onScrollGeometryChange(for: CGFloat.self) { scrollGeometry in
                    scrollGeometry.contentOffset.x
                } action: { _, offset in
                    guard isUserScrolling else { return }
                    state.isPlaying = false
                    state.playhead = min(max(0, offset * state.secondsPerPoint), state.duration)
                }
                .onScrollPhaseChange { _, newPhase in
                    scrollPhase = newPhase
                }

                playheadLine
                readout
            }
            // Tapping anything that isn't a clip clears the selection.
            .onTapGesture { state.selectedClipID = nil }
            .simultaneousGesture(pinchToZoom)
            .onChange(of: state.playhead) {
                syncScrollToPlayhead()
            }
            .onChange(of: state.secondsPerPoint) {
                syncScrollToPlayhead(force: true)
            }
            .onAppear {
                syncScrollToPlayhead(force: true)
            }
        }
        .frame(height: timelineHeight)
        .background(Theme.timelineBed)
        .clipped()
    }

    private var timelineHeight: CGFloat {
        Self.rulerHeight
            + Self.emptyLaneHeight * 2
            + ClipView.height
            + Self.rowSpacing * 3
            + 16
    }

    // MARK: Rows

    private var track: some View {
        HStack(spacing: 0) {
            ForEach(state.clips) { clip in
                ClipView(
                    clip: clip,
                    pointsPerSecond: pointsPerSecond,
                    isSelected: state.selectedClipID == clip.id,
                    onTap: {
                        state.selectedClipID = state.selectedClipID == clip.id ? nil : clip.id
                    },
                    onTrim: { edge, anchor, delta in
                        state.trim(clip.id, edge: edge, anchor: anchor, by: delta)
                    },
                    onTrimEnd: { state.endTrim() }
                )
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
        }
    }

    /// Full-width dim bands behind the scrolling rows: an empty lane above,
    /// the main track bed, and an empty lane below.
    private var laneBackground: some View {
        VStack(spacing: Self.rowSpacing) {
            Color.clear.frame(height: Self.rulerHeight)
            Rectangle().fill(Theme.trackEmpty.opacity(0.55)).frame(height: Self.emptyLaneHeight)
            Rectangle().fill(Theme.trackEmpty).frame(height: ClipView.height)
            Rectangle().fill(Theme.trackEmpty.opacity(0.55)).frame(height: Self.emptyLaneHeight)
        }
    }

    // MARK: Overlays

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

    /// Keeps the scroll offset in lockstep with the playhead whenever the
    /// change didn't originate from the user's finger.
    private func syncScrollToPlayhead(force: Bool = false) {
        guard force || !isUserScrolling else { return }
        scrollPosition.scrollTo(x: state.playhead * pointsPerSecond)
    }
}

#Preview {
    let state = EditorState()
    let _ = state.startProject(with: Array(MockData.libraryItems.prefix(4)))
    return TimelineView(state: state, onAddMedia: {})
        .background(Theme.background)
}
