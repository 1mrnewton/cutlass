import SwiftUI

/// Mock preview: renders the art of the clip under the playhead inside the
/// project's aspect ratio, over the chosen canvas background (blurred copy
/// or solid color). Empty timelines show a black canvas.
struct PreviewCanvas: View {
    var state: EditorState
    var onEditText: (UUID) -> Void = { _ in }

    var body: some View {
        ZStack {
            Color.black

            if let clip = state.clip(at: state.playhead) {
                canvasBackground(for: clip)
            }

            frameContent
                .aspectRatio(state.aspect.ratio, contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: 4))
                .padding(.vertical, 10)
        }
        .clipped()
    }

    /// The letterboxed project frame: footage (when the playhead is over the
    /// main track) plus overlay clips and the active-effect badge.
    private var frameContent: some View {
        ZStack {
            if let clip = state.clip(at: state.playhead) {
                MockArtView(art: clip.art, symbolSize: 64)
                    .opacity(clip.opacity)
            } else if state.background.kind == .color {
                state.background.color
            } else {
                Color.black.opacity(0.6)
            }
        }
        .overlay { PreviewOverlayLayer(state: state, onEditText: onEditText) }
        .overlay(alignment: .topLeading) { effectBadge }
    }

    @ViewBuilder
    private var effectBadge: some View {
        let active = state.effects(at: state.playhead)
        if let effect = active.last {
            Label(effect.displayLabel, systemImage: "sparkles")
                .font(.caption2.weight(.medium))
                .foregroundStyle(.white.opacity(0.9))
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.black.opacity(0.45), in: Capsule())
                .padding(8)
        }
    }

    @ViewBuilder
    private func canvasBackground(for clip: MockClip) -> some View {
        switch state.background.kind {
        case .blur:
            if state.background.blurStrength > 0 {
                MockArtView(art: clip.art, symbolSize: 0)
                    .blur(radius: 12 + 50 * state.background.blurStrength)
                    .opacity(0.55)
                    .clipped()
            }
        case .color:
            state.background.color
        }
    }
}

#Preview {
    let state = EditorState()
    let _ = state.startProject(with: Array(FixtureLibrary.sampleTimeline.prefix(3)))
    return PreviewCanvas(state: state)
        .frame(height: 420)
}
