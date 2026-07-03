import SwiftUI

/// Top-level navigation: Home <-> Editor, with the media picker presented as a
/// full-screen sheet either to start a project or to append to the timeline.
struct RootView: View {
    private enum Screen {
        case home
        case editor
    }

    @State private var screen: Screen = .home
    @State private var pickerPresented = false
    @State private var editorState = EditorState()

    /// Dev shortcut: `-startScreen picker|editor` (e.g. via `simctl launch`)
    /// jumps straight to a screen so states deep in the flow are easy to
    /// reach while iterating on the mock UI.
    init() {
        let arguments = ProcessInfo.processInfo.arguments
        guard let flag = arguments.firstIndex(of: "-startScreen"),
              arguments.indices.contains(flag + 1)
        else { return }

        switch arguments[flag + 1] {
        case "picker":
            _pickerPresented = State(initialValue: true)
        case "editor":
            let state = EditorState()
            state.startProject(with: Array(MockData.libraryItems.prefix(4)))
            _editorState = State(initialValue: state)
            _screen = State(initialValue: .editor)
        default:
            break
        }
    }

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()

            switch screen {
            case .home:
                HomeView(
                    onNewProject: { pickerPresented = true },
                    onBlankProject: {
                        editorState.startProject(with: [])
                        screen = .editor
                    }
                )
            case .editor:
                EditorView(
                    state: editorState,
                    onHome: { screen = .home },
                    onAddMedia: { pickerPresented = true }
                )
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
                if screen == .editor {
                    editorState.appendMedia(items)
                } else {
                    editorState.startProject(with: items)
                    screen = .editor
                }
                pickerPresented = false
            }
        )
    }
}

#Preview {
    RootView()
}
