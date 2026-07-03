import SwiftUI

/// Letterboxed mock preview: renders the art of the clip under the playhead
/// as a sharp 9:16 frame over a blurred full-bleed copy, like real editors
/// pillarbox portrait video. Empty timelines show a black canvas.
struct PreviewCanvas: View {
    var state: EditorState

    var body: some View {
        ZStack {
            Color.black

            if let clip = state.clip(at: state.playhead) {
                MockArtView(art: clip.art, symbolSize: 0)
                    .blur(radius: 46)
                    .opacity(0.55)
                    .clipped()

                MockArtView(art: clip.art, symbolSize: 64)
                    .aspectRatio(9.0 / 16.0, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                    .padding(.vertical, 10)
            }
        }
        .clipped()
    }
}

#Preview {
    let state = EditorState()
    let _ = state.startProject(with: Array(MockData.libraryItems.prefix(3)))
    return PreviewCanvas(state: state)
        .frame(height: 420)
}
