//
//  TimelineView.swift
//  cutlass-ios-macos
//
//  Renders tracks/clips from the engine's ProjectState as proportional blocks,
//  with a playhead, selection, and gesture -> command wiring (scrub, move, trim).
//  Selection is Swift-side only; every mutation goes through the command bridge.
//

import SwiftUI
import CutlassMobile

struct TimelineView: View {
    @ObservedObject var model: EditorModel

    /// Horizontal zoom: screen points per second of timeline.
    private let pointsPerSecond: Double = 80
    private let trackHeight: CGFloat = 52
    private let rulerHeight: CGFloat = 26

    private var fps: Double { model.state?.fps.asDouble ?? 30 }
    private var pointsPerTick: Double { fps > 0 ? pointsPerSecond / fps : 0 }

    private func x(forTicks ticks: Int64) -> CGFloat { CGFloat(Double(ticks) * pointsPerTick) }
    private func ticks(forPoints points: CGFloat) -> Int64 {
        pointsPerTick > 0 ? Int64(Double(points) / pointsPerTick) : 0
    }

    private var contentWidth: CGFloat {
        max(CGFloat(model.durationSeconds * pointsPerSecond) + 160, 600)
    }

    // Display top-to-bottom: state stacks bottom-to-top, so reverse for the UI.
    private var displayTracks: [ProjectState.Track] {
        (model.state?.tracks ?? []).reversed()
    }

    var body: some View {
        ScrollView([.horizontal, .vertical]) {
            ZStack(alignment: .topLeading) {
                VStack(alignment: .leading, spacing: 6) {
                    ruler
                    ForEach(displayTracks) { track in
                        trackRow(track)
                    }
                    if displayTracks.isEmpty {
                        Text("Import a video to start")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .padding()
                    }
                }
                playhead
            }
            .frame(width: contentWidth, alignment: .topLeading)
            .padding(8)
        }
        .background(Color(white: 0.1))
    }

    private var ruler: some View {
        Rectangle()
            .fill(Color(white: 0.16))
            .frame(width: contentWidth, height: rulerHeight)
            .overlay(alignment: .leading) {
                // Second ticks.
                let seconds = Int(model.durationSeconds.rounded(.up)) + 1
                ForEach(0..<max(seconds, 1), id: \.self) { s in
                    Rectangle()
                        .fill(Color.white.opacity(0.25))
                        .frame(width: 1, height: rulerHeight)
                        .offset(x: CGFloat(Double(s) * pointsPerSecond))
                }
            }
            .contentShape(Rectangle())
            .gesture(scrubGesture)
    }

    private func trackRow(_ track: ProjectState.Track) -> some View {
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 6)
                .fill(Color(white: 0.18))
                .frame(width: contentWidth, height: trackHeight)
            ForEach(track.clips) { clip in
                ClipBlock(
                    clip: clip,
                    pointsPerTick: pointsPerTick,
                    height: trackHeight - 8,
                    isSelected: model.selectedClip == clip.id,
                    onSelect: { model.selectedClip = clip.id },
                    onMove: { newStart in
                        Task { await model.move(clip: clip.id, toTrack: track.id, startTicks: newStart) }
                    },
                    onTrim: { newStart, newDuration in
                        Task { await model.trim(clip: clip.id, startTicks: newStart, durationTicks: newDuration) }
                    },
                    ticksFromPoints: { ticks(forPoints: $0) }
                )
                .padding(.top, 4)
            }
        }
        .frame(height: trackHeight)
    }

    private var playhead: some View {
        Rectangle()
            .fill(Color.red)
            .frame(width: 2)
            .frame(maxHeight: .infinity)
            .offset(x: x(forTicks: Int64((model.playheadSeconds * fps).rounded())) + 8)
            .allowsHitTesting(false)
    }

    private var scrubGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                model.pause()
                let seconds = max(0, min(Double(value.location.x) / pointsPerSecond, model.durationSeconds))
                model.playheadSeconds = seconds
                model.requestRender(at: seconds)
            }
    }
}

/// One clip rectangle with tap-to-select, drag-to-move, and trim handles.
private struct ClipBlock: View {
    let clip: ProjectState.Clip
    let pointsPerTick: Double
    let height: CGFloat
    let isSelected: Bool
    let onSelect: () -> Void
    let onMove: (Int64) -> Void
    let onTrim: (Int64, Int64) -> Void
    let ticksFromPoints: (CGFloat) -> Int64

    @State private var moveDelta: CGFloat = 0
    @State private var leadDelta: CGFloat = 0
    @State private var trailDelta: CGFloat = 0

    private var baseX: CGFloat { CGFloat(Double(clip.start) * pointsPerTick) }
    private var baseW: CGFloat { CGFloat(Double(clip.duration) * pointsPerTick) }

    var body: some View {
        let x = baseX + moveDelta + leadDelta
        let w = max(10, baseW - leadDelta + trailDelta)

        ZStack(alignment: .leading) {
            RoundedRectangle(cornerRadius: 6)
                .fill(clip.kind == "media" ? Color.blue.opacity(0.85) : Color.purple.opacity(0.85))
            Text(clip.label)
                .font(.caption2)
                .lineLimit(1)
                .truncationMode(.middle)
                .foregroundStyle(.white)
                .padding(.horizontal, 8)
            RoundedRectangle(cornerRadius: 6)
                .stroke(isSelected ? Color.yellow : Color.black.opacity(0.3), lineWidth: isSelected ? 2 : 1)

            if isSelected {
                HStack {
                    trimHandle(leading: true)
                    Spacer()
                    trimHandle(leading: false)
                }
            }
        }
        .frame(width: w, height: height)
        .offset(x: x)
        .contentShape(Rectangle())
        .onTapGesture { onSelect() }
        .gesture(
            DragGesture(minimumDistance: 8)
                .onChanged { value in
                    if !isSelected { onSelect() }
                    moveDelta = value.translation.width
                }
                .onEnded { value in
                    let delta = ticksFromPoints(value.translation.width)
                    moveDelta = 0
                    onMove(max(0, clip.start + delta))
                }
        )
    }

    private func trimHandle(leading: Bool) -> some View {
        RoundedRectangle(cornerRadius: 3)
            .fill(Color.yellow)
            .frame(width: 10, height: height * 0.7)
            .padding(.horizontal, 2)
            .contentShape(Rectangle())
            .highPriorityGesture(
                DragGesture(minimumDistance: 2)
                    .onChanged { value in
                        if leading { leadDelta = value.translation.width }
                        else { trailDelta = value.translation.width }
                    }
                    .onEnded { value in
                        let delta = ticksFromPoints(value.translation.width)
                        if leading {
                            leadDelta = 0
                            let newStart = clip.start + delta
                            let newDuration = clip.duration - delta
                            onTrim(max(0, newStart), max(1, newDuration))
                        } else {
                            trailDelta = 0
                            onTrim(clip.start, max(1, clip.duration + delta))
                        }
                    }
            )
    }
}
