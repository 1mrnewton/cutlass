//
//  EditorView.swift
//  cutlass-ios-macos
//
//  The v1 editor: preview viewport + transport + timeline, driven entirely
//  through the JSON command/state bridge. All engine access is funneled through
//  a serial `SessionActor`, so renders (off the main thread) never overlap edits.
//

import Combine
import SwiftUI
import PhotosUI
import CutlassMobile

/// Serializes every engine call. Actor isolation guarantees the native session
/// (which is `&mut` for every op) is only ever touched one call at a time.
actor SessionActor {
    private let session: CutlassSession

    init(session: CutlassSession) {
        self.session = session
    }

    func state() -> ProjectState? { session.state() }
    func apply(_ command: Command) -> CommandResult? { session.apply(command) }
    func apply(_ edit: EditCommand) -> CommandResult? { session.apply(edit) }
    func undo() -> CommandResult? { session.undo() }
    func redo() -> CommandResult? { session.redo() }
    func importMedia(path: String) -> ImportResult? { session.importMedia(path: path) }
    func save(path: String) -> Bool { session.save(path: path) }
    func render(at seconds: Double) -> FrameBox { FrameBox(image: session.render(atSeconds: seconds)) }
}

@MainActor
final class EditorModel: ObservableObject {
    @Published var state: ProjectState?
    @Published var frame: CGImage?
    @Published var selectedClip: UInt64?
    @Published var playheadSeconds: Double = 0
    @Published var isPlaying = false
    @Published var status: String?

    let projectURL: URL
    private let actor: SessionActor
    private let isNew: Bool

    private var renderTask: Task<Void, Never>?
    private var playbackTask: Task<Void, Never>?

    init?(projectURL: URL, cacheDir: String, create: Bool, fpsNum: Int32 = 30, fpsDen: Int32 = 1) {
        self.projectURL = projectURL
        self.isNew = create
        let session: CutlassSession?
        if create {
            session = CutlassSession.create(fpsNum: fpsNum, fpsDen: fpsDen, cacheDir: cacheDir)
        } else {
            session = CutlassSession.open(path: projectURL.path, cacheDir: cacheDir)
        }
        guard let session else { return nil }
        self.actor = SessionActor(session: session)
    }

    var fps: Double { state.map { $0.fps.asDouble } ?? 30 }
    var durationSeconds: Double {
        guard let state, fps > 0 else { return 0 }
        return Double(state.durationTicks) / fps
    }

    /// Initial load: persist a brand-new project, then pull state + first frame.
    func start() async {
        if isNew { _ = await actor.save(path: projectURL.path) }
        await refresh()
    }

    func refresh() async {
        state = await actor.state()
        requestRender(at: playheadSeconds)
    }

    /// Render at `seconds`, latest-wins (the actor serializes the work).
    func requestRender(at seconds: Double) {
        renderTask?.cancel()
        renderTask = Task { [weak self] in
            guard let self else { return }
            let box = await self.actor.render(at: seconds)
            if Task.isCancelled { return }
            if let image = box.image { self.frame = image }
        }
    }

    // MARK: Transport

    func togglePlay() {
        if isPlaying { pause() } else { play() }
    }

    private func play() {
        guard !isPlaying, durationSeconds > 0 else { return }
        isPlaying = true
        let step = 1.0 / max(fps, 1)
        // Created inside a @MainActor method, so this task is main-actor isolated:
        // UI/state reads are direct; only the actor render call suspends.
        playbackTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled && self.isPlaying {
                let total = self.durationSeconds
                if self.playheadSeconds >= total {
                    self.isPlaying = false
                    break
                }
                let box = await self.actor.render(at: self.playheadSeconds)
                if Task.isCancelled { break }
                if let image = box.image { self.frame = image }
                self.playheadSeconds = min(self.playheadSeconds + step, total)
                try? await Task.sleep(nanoseconds: UInt64(step * 1_000_000_000))
            }
        }
    }

    func pause() {
        isPlaying = false
        playbackTask?.cancel()
        playbackTask = nil
    }

    // MARK: Edits

    private func ticks(forSeconds seconds: Double) -> Int64 {
        Int64((seconds * fps).rounded())
    }

    private var projectRate: Rational { state?.fps ?? Rational(num: 30, den: 1) }

    func importMovie(at url: URL) async {
        guard let imp = await actor.importMedia(path: url.path), imp.ok,
              let media = imp.media, let mediaFps = imp.fps, let dur = imp.durationTicks
        else {
            status = "Import failed"
            return
        }
        // Refresh so we know the current tracks / end of timeline.
        let snapshot = await actor.state()
        state = snapshot
        guard let track = snapshot?.tracks.first(where: { $0.kind == "video" })?.id
            ?? snapshot?.tracks.first?.id
        else {
            status = "No track to place on"
            return
        }
        let startTicks = snapshot?.durationTicks ?? 0
        let source = TimeRange(
            start: RationalTime(value: 0, rate: mediaFps),
            duration: RationalTime(value: dur, rate: mediaFps)
        )
        let cmd = EditCommand.addClip(
            track: track,
            media: media,
            source: source,
            start: RationalTime(value: startTicks, rate: projectRate)
        )
        let res = await actor.apply(cmd)
        if let res, res.ok {
            selectedClip = res.createdClip
        } else {
            status = res?.error ?? "Add clip failed"
        }
        await refresh()
    }

    func splitAtPlayhead() async {
        guard let clip = selectedClip else { return }
        let at = RationalTime(value: ticks(forSeconds: playheadSeconds), rate: projectRate)
        let res = await actor.apply(.splitClip(clip: clip, at: at))
        if let res, res.ok {
            if let created = res.createdClip { selectedClip = created }
        } else {
            status = res?.error ?? "Split failed"
        }
        await refresh()
    }

    func deleteSelected() async {
        guard let clip = selectedClip else { return }
        let res = await actor.apply(.removeClip(clip: clip))
        if res?.ok == true { selectedClip = nil } else { status = res?.error }
        await refresh()
    }

    func move(clip: UInt64, toTrack: UInt64, startTicks: Int64) async {
        let res = await actor.apply(.moveClip(
            clip: clip,
            toTrack: toTrack,
            start: RationalTime(value: max(0, startTicks), rate: projectRate)
        ))
        if res?.ok != true { status = res?.error }
        await refresh()
    }

    func trim(clip: UInt64, startTicks: Int64, durationTicks: Int64) async {
        let timeline = TimeRange(
            start: RationalTime(value: max(0, startTicks), rate: projectRate),
            duration: RationalTime(value: max(1, durationTicks), rate: projectRate)
        )
        let res = await actor.apply(.trimClip(clip: clip, timeline: timeline))
        if res?.ok != true { status = res?.error }
        await refresh()
    }

    func undo() async {
        let res = await actor.undo()
        if res?.ok == true { selectedClip = nil }
        await refresh()
    }

    func redo() async {
        let res = await actor.redo()
        if res?.ok == true { selectedClip = nil }
        await refresh()
    }

    // MARK: Persistence

    func save() async {
        _ = await actor.save(path: projectURL.path)
        if let image = frame {
            let thumb = projectURL.deletingPathExtension().appendingPathExtension("png")
            writePNG(image, to: thumb)
        }
    }
}

struct EditorView: View {
    @StateObject private var model: EditorModel
    @Environment(\.scenePhase) private var scenePhase
    @State private var pickerItem: PhotosPickerItem?

    init?(projectURL: URL, cacheDir: String, create: Bool, fpsNum: Int32 = 30, fpsDen: Int32 = 1) {
        guard let model = EditorModel(
            projectURL: projectURL, cacheDir: cacheDir, create: create,
            fpsNum: fpsNum, fpsDen: fpsDen
        ) else { return nil }
        _model = StateObject(wrappedValue: model)
    }

    var body: some View {
        VStack(spacing: 0) {
            preview
            transport
            Divider()
            TimelineView(model: model)
                .frame(maxHeight: 260)
        }
        .navigationTitle(model.state?.name ?? "Editor")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .toolbar { toolbar }
        .task { await model.start() }
        .onDisappear { Task { await model.save() } }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active { Task { await model.save() } }
        }
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            Task { await loadAndImport(item) }
        }
        .overlay(alignment: .bottom) {
            if let status = model.status {
                Text(status)
                    .font(.caption)
                    .padding(8)
                    .background(.thinMaterial, in: Capsule())
                    .padding(.bottom, 8)
            }
        }
    }

    private var preview: some View {
        ZStack {
            Color.black
            if let frame = model.frame {
                Image(frame, scale: 1, label: Text("Preview"))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            } else {
                ProgressView().tint(.white)
            }
        }
        .frame(maxWidth: .infinity)
        .frame(minHeight: 200, maxHeight: 360)
    }

    private var transport: some View {
        HStack(spacing: 12) {
            Button(action: { model.togglePlay() }) {
                Image(systemName: model.isPlaying ? "pause.fill" : "play.fill")
                    .frame(width: 28)
            }
            Slider(
                value: Binding(
                    get: { model.playheadSeconds },
                    set: { newValue in
                        model.pause()
                        model.playheadSeconds = newValue
                        model.requestRender(at: newValue)
                    }
                ),
                in: 0...max(model.durationSeconds, 0.0001)
            )
            Text(String(format: "%.2f / %.2f s", model.playheadSeconds, model.durationSeconds))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 110, alignment: .trailing)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .buttonStyle(.borderless)
    }

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            PhotosPicker(selection: $pickerItem, matching: .videos, preferredItemEncoding: .current) {
                Image(systemName: "plus.rectangle.on.rectangle")
            }
            Button { Task { await model.splitAtPlayhead() } } label: {
                Image(systemName: "scissors")
            }
            .disabled(model.selectedClip == nil)
            Button { Task { await model.deleteSelected() } } label: {
                Image(systemName: "trash")
            }
            .disabled(model.selectedClip == nil)
            Button { Task { await model.undo() } } label: {
                Image(systemName: "arrow.uturn.backward")
            }
            .disabled(!(model.state?.canUndo ?? false))
            Button { Task { await model.redo() } } label: {
                Image(systemName: "arrow.uturn.forward")
            }
            .disabled(!(model.state?.canRedo ?? false))
        }
    }

    private func loadAndImport(_ item: PhotosPickerItem) async {
        do {
            guard let movie = try await item.loadTransferable(type: PickedMovie.self) else {
                model.status = "Couldn't load that video"
                return
            }
            await model.importMovie(at: movie.url)
        } catch {
            model.status = "Import error: \(error.localizedDescription)"
        }
        pickerItem = nil
    }
}
