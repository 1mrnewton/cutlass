import SwiftUI

/// Top-level navigation: Home <-> Editor, with the media picker presented as a
/// full-screen sheet either to start a project or to append to the timeline.
struct RootView: View {
    private enum Screen {
        case home
        case editor
    }

    /// What the picker result should do when it lands.
    private enum PickerIntent {
        case newProject
        case appendToTimeline
        case replaceSelectedClip
    }

    @State private var screen: Screen = .home
    @State private var pickerIntent: PickerIntent?
    @State private var editorState = EditorState()

    private var pickerPresented: Binding<Bool> {
        Binding(
            get: { pickerIntent != nil },
            set: { if !$0 { pickerIntent = nil } }
        )
    }

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
            _pickerIntent = State(initialValue: .newProject)
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
                    onNewProject: { pickerIntent = .newProject },
                    onBlankProject: {
                        editorState.startProject(with: [])
                        screen = .editor
                    }
                )
            case .editor:
                EditorView(
                    state: editorState,
                    onHome: { screen = .home },
                    onAddMedia: { pickerIntent = .appendToTimeline },
                    onReplaceMedia: { pickerIntent = .replaceSelectedClip }
                )
            }
        }
        .preferredColorScheme(.dark)
        #if os(macOS)
        .sheet(isPresented: pickerPresented) { picker }
        #else
        .fullScreenCover(isPresented: pickerPresented) { picker }
        #endif
    }

    private var picker: some View {
        MediaPickerView(
            onCancel: { pickerIntent = nil },
            onDone: { items in
                switch pickerIntent {
                case .appendToTimeline:
                    editorState.appendMedia(items)
                case .replaceSelectedClip:
                    if let item = items.first {
                        editorState.replaceSelected(with: item)
                    }
                case .newProject, nil:
                    editorState.startProject(with: items)
                    screen = .editor
                }
                pickerIntent = nil
            }
        )
    }
}

#Preview {
    RootView()
}
