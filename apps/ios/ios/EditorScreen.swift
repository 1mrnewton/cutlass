import PhotosUI
import SwiftUI

/// The editor: preview on top, the transport, the timeline, then the tools.
struct EditorScreen: View {
    let name: String
    let home: () -> Void
    @State private var model: EditorModel
    @State private var picked: [PhotosPickerItem] = []
    @State private var panel: Panel?

    /// The tool panel open over the tool bar.
    enum Panel { case volume, speed }

    init(project: ProjectSummary, home: @escaping () -> Void) {
        name = project.name
        self.home = home
        _model = State(initialValue: EditorModel(project: project.path))
    }

    var body: some View {
        VStack(spacing: 0) {
            TitleBar(title: name, background: Color(white: 0.06)) {
                Button {
                    model.pause()
                    home()
                } label: {
                    Image(systemName: "house")
                        .font(.title3)
                        .foregroundStyle(Palette.text)
                        .frame(width: 44, height: 44)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Home")
            }
            preview
            transport
            TimelineStrip(model: model, picked: $picked)
                .frame(maxHeight: .infinity)
                .background(Color(white: 0.09))
            bottom
        }
        .animation(.easeOut(duration: 0.2), value: panel)
        .animation(.easeOut(duration: 0.15), value: model.selectedClip == nil)
        .background(Color(white: 0.06))
        .overlay(alignment: .top) { notice }
        .onChange(of: picked) { _, items in importPicked(items) }
        .onChange(of: model.selectedClip) { _, id in if id == nil { panel = nil } }
        .onDisappear { model.pause() }
    }

    private var preview: some View {
        ZStack {
            Color.black
            if let picture = model.picture {
                Image(decorative: picture, scale: 1)
                    .resizable()
                    .aspectRatio(model.canvasAspect, contentMode: .fit)
            }
            if model.isOpening || model.isImporting {
                ProgressView().tint(.white)
            }
        }
        .frame(maxWidth: .infinity)
        .aspectRatio(16 / 10, contentMode: .fit)
    }

    private var transport: some View {
        HStack {
            Text(clock(model.playheadUs) + " / " + clock(model.durationUs))
                .font(.footnote.monospacedDigit())
                .foregroundStyle(.white.opacity(0.7))
            Spacer()
            Button {
                model.togglePlay()
            } label: {
                Image(systemName: model.isPlaying ? "pause.fill" : "play.fill")
                    .font(.title3)
            }
            .disabled(!model.hasClips)
            .accessibilityLabel(model.isPlaying ? "Pause" : "Play")
            Spacer()
            Button { model.undo() } label: { Image(systemName: "arrow.uturn.backward") }
                .accessibilityLabel("Undo")
            Button { model.redo() } label: { Image(systemName: "arrow.uturn.forward") }
                .accessibilityLabel("Redo")
                .padding(.leading, 16)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    /// A panel when one is open, the clip tools while a clip is selected,
    /// else the root tools.
    @ViewBuilder
    private var bottom: some View {
        if let panel, let clip = model.selection {
            Group {
                switch panel {
                case .volume: VolumePanel(model: model, clip: clip) { self.panel = nil }
                case .speed: SpeedPanel(model: model, clip: clip) { self.panel = nil }
                }
            }
            .transition(.move(edge: .bottom))
        } else if model.selection != nil {
            clipTools
        } else {
            rootTools
        }
    }

    private var rootTools: some View {
        toolBar {
            Button { model.editUnderPlayhead() } label: { tool("Edit", "scissors") }
                .disabled(!model.hasClips)
            PhotosPicker(
                selection: $picked, matching: .any(of: [.images, .videos]),
                preferredItemEncoding: .current
            ) {
                tool("Add", "plus.rectangle.on.rectangle")
            }
            .disabled(model.isOpening || model.isImporting)
        }
    }

    private var clipTools: some View {
        HStack(spacing: 0) {
            Button { model.selectedClip = nil } label: {
                Image(systemName: "chevron.left")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.white)
                    .frame(width: 40, height: 52)
                    .background(Color(white: 0.17), in: RoundedRectangle(cornerRadius: 6))
            }
            .accessibilityLabel("Back")
            .padding(.leading, 8)
            toolBar {
                Button { model.split() } label: { tool("Split", "square.split.2x1") }
                Button { panel = .volume } label: { tool("Volume", "speaker.wave.2") }
                Button { panel = .speed } label: { tool("Speed", "gauge.with.dots.needle.67percent") }
                Button { model.deleteClip() } label: { tool("Delete", "trash") }
            }
        }
        .background(Color(white: 0.11))
    }

    private func toolBar<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 4) { content() }
                .padding(.horizontal, 8)
        }
        .foregroundStyle(.white)
        .padding(.top, 10)
        .padding(.bottom, 4)
        .frame(maxWidth: .infinity)
        .background(Color(white: 0.11))
    }

    private func tool(_ title: String, _ symbol: String) -> some View {
        VStack(spacing: 4) {
            Image(systemName: symbol).font(.title3)
            Text(title).font(.caption2)
        }
        .frame(width: 68, height: 52)
        .contentShape(Rectangle())
    }

    @ViewBuilder
    private var notice: some View {
        if let notice = model.notice {
            Text(notice.text)
                .font(.footnote.weight(.medium))
                .lineLimit(2)
                .foregroundStyle(.black)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.white, in: Capsule())
                .padding(.top, 8)
                .transition(.move(edge: .top).combined(with: .opacity))
                .task(id: notice.id) {
                    try? await Task.sleep(for: .seconds(2.5))
                    if model.notice == notice { withAnimation { model.clearNotice() } }
                }
        }
    }

    private func importPicked(_ items: [PhotosPickerItem]) {
        guard !items.isEmpty else { return }
        picked = []
        model.beginImport()
        Task {
            for item in items {
                do {
                    let file = try await PickedFile.load(item)
                    model.importFile(at: file)
                    try? FileManager.default.removeItem(at: file.deletingLastPathComponent())
                } catch {
                    model.show(error)
                }
            }
            model.endImport()
        }
    }

    private func clock(_ micros: Int64) -> String {
        TimelineStrip.clock(Double(micros / 100_000) / 10)
    }
}
