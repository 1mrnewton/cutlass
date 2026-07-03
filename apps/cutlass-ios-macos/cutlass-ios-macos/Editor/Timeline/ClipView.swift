import SwiftUI

/// One clip on the track: repeating thumbnail tiles with a fake waveform
/// strip when the source has audio. Width is time-accurate
/// (length x points-per-second) so scrubbing lines up with the ruler.
/// Selected clips grow a white border with draggable trim handles.
struct ClipView: View {
    var clip: MockClip
    var pointsPerSecond: Double
    var isSelected: Bool = false
    var onTap: () -> Void = {}
    var onTrim: (EditorState.TrimEdge, MockClip, Double) -> Void = { _, _, _ in }
    var onTrimEnd: () -> Void = {}

    /// Clip snapshot from the moment a trim drag started.
    @State private var trimAnchor: MockClip?

    static let height: CGFloat = 66
    private static let waveformHeight: CGFloat = 20
    private static let handleWidth: CGFloat = 16

    private var width: CGFloat {
        max(2, clip.length * pointsPerSecond)
    }

    var body: some View {
        VStack(spacing: 0) {
            thumbnailStrip(height: clip.hasAudio ? Self.height - Self.waveformHeight : Self.height)
            if clip.hasAudio {
                WaveformStrip(seed: clip.id.hashValue)
                    .frame(height: Self.waveformHeight)
            }
        }
        .frame(width: width, height: Self.height)
        .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
        // Hairline separator so adjacent clips read as distinct without
        // shifting the time math the way real HStack spacing would.
        .overlay(alignment: .trailing) {
            if !isSelected {
                Rectangle()
                    .fill(Theme.timelineBed)
                    .frame(width: 1)
            }
        }
        .overlay {
            if isSelected {
                selectionChrome
            }
        }
        .contentShape(Rectangle())
        .onTapGesture(perform: onTap)
    }

    private func thumbnailStrip(height: CGFloat) -> some View {
        let tileCount = max(1, Int((width / height).rounded(.up)))
        return HStack(spacing: 0) {
            ForEach(0..<tileCount, id: \.self) { index in
                MockArtView(art: clip.art, symbolSize: 15)
                    .frame(width: height, height: height)
                    // Alternate slight shading to fake distinct frames.
                    .overlay(Color.black.opacity(index.isMultiple(of: 2) ? 0 : 0.10))
                    .clipped()
            }
        }
        .frame(width: width, alignment: .leading)
        .clipped()
    }

    // MARK: Selection + trim handles

    private var selectionChrome: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 5, style: .continuous)
                .strokeBorder(.white, lineWidth: 2.5)

            HStack(spacing: 0) {
                handle(.leading)
                Spacer(minLength: 0)
                handle(.trailing)
            }
        }
    }

    private func handle(_ edge: EditorState.TrimEdge) -> some View {
        let corners: RectangleCornerRadii = edge == .leading
            ? RectangleCornerRadii(topLeading: 5, bottomLeading: 5)
            : RectangleCornerRadii(bottomTrailing: 5, topTrailing: 5)

        return UnevenRoundedRectangle(cornerRadii: corners, style: .continuous)
            .fill(.white)
            .frame(width: Self.handleWidth, height: Self.height)
            .overlay {
                Capsule()
                    .fill(.black.opacity(0.55))
                    .frame(width: 3, height: 16)
            }
            // minimumDistance 0 claims the touch immediately so the
            // surrounding ScrollView can't steal horizontal drags.
            .highPriorityGesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        let anchor = trimAnchor ?? clip
                        trimAnchor = anchor
                        onTrim(edge, anchor, value.translation.width / pointsPerSecond)
                    }
                    .onEnded { _ in
                        trimAnchor = nil
                        onTrimEnd()
                    }
            )
    }
}

/// Deterministic pseudo-random waveform bars, seeded per clip so the shape
/// is stable across renders.
private struct WaveformStrip: View {
    var seed: Int

    var body: some View {
        Canvas { context, size in
            let phase = Double(seed % 977) * 0.13
            let barWidth: CGFloat = 2
            let gap: CGFloat = 1.5
            var x: CGFloat = 1

            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(hex: 0x0C2733)))

            while x < size.width - 1 {
                let t = Double(x)
                let envelope = 0.35 + 0.65 * abs(sin(t * 0.021 + phase))
                let detail = 0.45 + 0.55 * abs(sin(t * 0.29 + phase * 3))
                let barHeight = max(1.5, size.height * envelope * detail * 0.92)
                let rect = CGRect(
                    x: x,
                    y: (size.height - barHeight) / 2,
                    width: barWidth,
                    height: barHeight
                )
                context.fill(Path(roundedRect: rect, cornerRadius: 1), with: .color(Theme.waveform))
                x += barWidth + gap
            }
        }
    }
}

#Preview {
    HStack(spacing: 0) {
        ClipView(clip: MockClip(from: MockData.libraryItems[1]), pointsPerSecond: 44, isSelected: true)
        ClipView(clip: MockClip(from: MockData.libraryItems[0]), pointsPerSecond: 44)
    }
    .padding()
    .background(Theme.timelineBed)
}
