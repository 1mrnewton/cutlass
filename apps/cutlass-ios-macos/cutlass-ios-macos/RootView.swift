import CutlassMobile
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
        case addOverlay
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
            state.startProject(with: FixtureLibrary.sampleTimeline)
            _editorState = State(initialValue: state)
            _screen = State(initialValue: .editor)
        case "editorLanes":
            // Editor with every lane populated (video PiP above main, sticker,
            // effect, audio at the bottom), used by UI tests to exercise each
            // timeline row and kind-preserving cross-lane drags. Seeded as
            // real engine intents (FIFO, so times are explicit rather than
            // playhead-derived).
            let state = EditorState()
            state.startProject(with: FixtureLibrary.sampleTimeline)
            state.seedIntents([
                .addSticker(atSeconds: 0),
                .addEffect(kind: "effect", atSeconds: 0),
            ])
            if let audio = FixtureLibrary.audio {
                state.seedIntents([.addAudio(path: audio.path, atSeconds: 0)])
            }
            // Insert past the sticker so the overlay lane stays one row tall.
            if let pip = FixtureLibrary.shortVideo {
                state.seedIntents([.addPip(path: pip.path, atSeconds: 5)])
            }
            state.selection = nil
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
                    },
                    onOpenProject: { _ in
                        // Mock "reopen": seed the editor with sample media
                        // (the real project store lands with persistence).
                        editorState.startProject(with: FixtureLibrary.sampleTimeline)
                        screen = .editor
                    }
                )
            case .editor:
                EditorView(
                    state: editorState,
                    onHome: {
                        editorState.isPlaying = false
                        screen = .home
                    },
                    onAddMedia: { pickerIntent = .appendToTimeline },
                    onAddOverlay: { pickerIntent = .addOverlay },
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
            onDone: { urls in
                switch pickerIntent {
                case .appendToTimeline:
                    editorState.appendMedia(urls)
                case .replaceSelectedClip:
                    if let url = urls.first {
                        editorState.replaceSelected(with: url)
                    }
                case .addOverlay:
                    if let url = urls.first {
                        editorState.addPip(from: url)
                    }
                case .newProject, nil:
                    editorState.startProject(with: urls)
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
