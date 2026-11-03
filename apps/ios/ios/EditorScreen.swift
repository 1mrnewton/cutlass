import PhotosUI
import SwiftUI

/// The editor: preview on top, the transport, the timeline, then the tools.
struct EditorScreen: View {
    let name: String
    @State private var model: EditorModel
    @State private var picked: [PhotosPickerItem] = []

    init(project: ProjectSummary) {
        name = project.name
        _model = State(initialValue: EditorModel(project: project.path))
    }

    var body: some View {
        VStack(spacing: 0) {
            preview
            transport
            TimelineStrip(model: model)
                .frame(maxHeight: .infinity)
                .background(Color(white: 0.09))
            tools
        }
        .background(Color(white: 0.06))
        .overlay(alignment: .top) { notice }
        .onChange(of: picked) { _, items in importPicked(items) }
        .onDisappear { model.pause() }
        .navigationTitle(name)
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(Color(white: 0.06), for: .navigationBar)
        .toolbarBackground(.visible, for: .navigationBar)
    }

    private var preview: some View {
        ZStack {
            Color.black
            if let picture = model.picture {
                Image(decorative: picture, scale: 1)
                    .resizable()
                    .aspectRatio(model.canvasAspect, contentMode: .fit)
            }
            if model.isImporting {
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

    private var tools: some View {
        HStack {
            PhotosPicker(selection: $picked, matching: .any(of: [.images, .videos])) {
                tool("Add", "plus")
            }
            .disabled(model.isImporting)
            Button { model.split() } label: { tool("Split", "scissors") }
                .disabled(!model.hasClips)
            Button { model.deleteClip() } label: { tool("Delete", "trash") }
                .disabled(!model.hasClips)
        }
        .foregroundStyle(.white)
        .padding(.top, 10)
        .padding(.bottom, 4)
        .background(Color(white: 0.11))
    }

    private func tool(_ title: String, _ symbol: String) -> some View {
        VStack(spacing: 4) {
            Image(systemName: symbol).font(.title3)
            Text(title).font(.caption2)
        }
        .frame(maxWidth: .infinity)
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
                    let file = try await PickedMedia.load(item)
                    model.importFile(at: file.url, as: file.media)
                    try? FileManager.default.removeItem(at: file.url.deletingLastPathComponent())
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
