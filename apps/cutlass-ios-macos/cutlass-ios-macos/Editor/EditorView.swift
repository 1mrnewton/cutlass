import SwiftUI

/// The editor screen: top chrome, preview, transport, timeline bed, and the
/// context-sensitive bottom toolbar.
struct EditorView: View {
    var state: EditorState
    var onHome: () -> Void
    var onAddMedia: () -> Void
    var onReplaceMedia: () -> Void

    @State private var exportPresented = false

    var body: some View {
        VStack(spacing: 0) {
            EditorTopBar(
                exportEnabled: !state.isEmpty,
                onHome: onHome,
                onExport: { exportPresented = true }
            )

            PreviewCanvas(state: state)
                .frame(maxHeight: .infinity)

            TransportControls(state: state, onSplit: { state.splitAtPlayhead() })

            TimelineView(state: state, onAddMedia: onAddMedia)

            if state.selectedClip != nil {
                ClipToolbar(
                    onAdd: onAddMedia,
                    onSplit: { state.splitAtPlayhead() },
                    onDelete: { state.deleteSelected() },
                    onDuplicate: { state.duplicateSelected() },
                    onReplace: onReplaceMedia
                )
            } else {
                MediaToolbar(onAddMedia: onAddMedia)
            }
        }
        .background(Theme.background)
        .sheet(isPresented: $exportPresented) {
            ExportStubSheet(duration: state.duration)
        }
    }
}

/// Mock export sheet; rendering is out of scope for the UI build.
private struct ExportStubSheet: View {
    var duration: TimeInterval
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "square.and.arrow.up")
                .font(.system(size: 34, weight: .medium))
                .foregroundStyle(Theme.accent)

            Text("Export")
                .font(.title2.bold())
                .foregroundStyle(.white)

            Text("Exporting a \(duration.timecode) video isn't wired up in this UI preview yet.")
                .font(.subheadline)
                .foregroundStyle(Theme.textSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)

            Button("Done") { dismiss() }
                .font(.headline)
                .foregroundStyle(.white)
                .padding(.horizontal, 40)
                .padding(.vertical, 12)
                .background(Theme.accent, in: Capsule())
                .buttonStyle(.plain)
                .padding(.top, 6)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.surface)
        .presentationDetents([.medium])
    }
}

#Preview {
    let state = EditorState()
    let _ = state.startProject(with: Array(MockData.libraryItems.prefix(3)))
    return EditorView(state: state, onHome: {}, onAddMedia: {}, onReplaceMedia: {})
}
