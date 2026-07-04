import SwiftUI

/// The row between preview and timeline: split/keyframe tools, frame-step
/// and play controls, undo/redo.
struct TransportControls: View {
    var state: EditorState
    var canUndo = false
    var canRedo = false
    var onSplit: () -> Void = {}
    var onUndo: () -> Void = {}
    var onRedo: () -> Void = {}

    /// The playhead sits on a keyframe of the selected clip (within tolerance).
    private var keyframeActive: Bool {
        guard let clip = state.selectedClip else { return false }
        let local = state.playhead - state.startTime(of: clip.id)
        return clip.keyframes.contains { abs($0 - local) < 0.15 }
    }

    var body: some View {
        HStack(spacing: 0) {
            HStack(spacing: 24) {
                iconButton("arrow.right.and.line.vertical.and.arrow.left", enabled: !state.isEmpty, action: onSplit)
                Button {
                    state.toggleKeyframeAtPlayhead()
                } label: {
                    Image(systemName: keyframeActive ? "diamond.fill" : "diamond")
                        .font(.system(size: 17, weight: .medium))
                        .foregroundStyle(
                            state.selectedClip == nil
                                ? Theme.textTertiary
                                : keyframeActive ? Theme.accent : .white
                        )
                        .frame(width: 30, height: 30)
                }
                .buttonStyle(.plain)
                .disabled(state.selectedClip == nil)
            }

            Spacer()

            HStack(spacing: 26) {
                iconButton("backward.frame", enabled: !state.isEmpty) {
                    state.stepFrame(by: -1)
                }
                iconButton(state.isPlaying ? "pause.fill" : "play.fill", size: 21, enabled: !state.isEmpty) {
                    state.isPlaying.toggle()
                }
                iconButton("forward.frame", enabled: !state.isEmpty) {
                    state.stepFrame(by: 1)
                }
            }

            Spacer()

            HStack(spacing: 24) {
                iconButton("arrow.uturn.backward", enabled: canUndo, action: onUndo)
                iconButton("arrow.uturn.forward", enabled: canRedo, action: onRedo)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
    }

    private func iconButton(
        _ symbol: String,
        size: CGFloat = 17,
        enabled: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: size, weight: .medium))
                .foregroundStyle(enabled ? .white : Theme.textTertiary)
                .frame(width: 30, height: 30)
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
    }
}

#Preview {
    let state = EditorState()
    let _ = state.startProject(with: Array(FixtureLibrary.sampleTimeline.prefix(2)))
    return TransportControls(state: state)
        .background(Theme.background)
}
