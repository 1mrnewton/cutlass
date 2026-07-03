import SwiftUI

/// Top-level navigation: Home <-> Editor, with the media picker presented as a
/// full-screen sheet from either the home screen or an empty editor.
struct RootView: View {
    private enum Screen {
        case home
        case editor
    }

    @State private var screen: Screen = .home
    @State private var pickerPresented = false
    @State private var editorClips: [MockClip] = []

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()

            switch screen {
            case .home:
                HomeView(
                    onNewProject: { pickerPresented = true },
                    onBlankProject: {
                        editorClips = []
                        screen = .editor
                    }
                )
            case .editor:
                EditorView(clips: editorClips, onHome: { screen = .home })
            }
        }
        .preferredColorScheme(.dark)
        #if os(macOS)
        .sheet(isPresented: $pickerPresented) { picker }
        #else
        .fullScreenCover(isPresented: $pickerPresented) { picker }
        #endif
    }

    private var picker: some View {
        MediaPickerView(
            onCancel: { pickerPresented = false },
            onDone: { items in
                editorClips = items.map(MockClip.init(from:))
                pickerPresented = false
                screen = .editor
            }
        )
    }
}

#Preview {
    RootView()
}
