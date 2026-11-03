import CoreGraphics
import Foundation
import Observation
import os

/// The open project as the screens see it.
///
/// Every edit goes through the Rust editor, which owns the draft and its
/// history. This model keeps what the screens draw: the rows, the length,
/// the playhead, and the picture at the playhead. A picture is drawn off the
/// main thread; while one draws, only the newest time asked for waits.
@MainActor
@Observable
final class EditorModel {
    private(set) var lanes: [Lane] = []
    private(set) var durationUs: Int64 = 0
    private(set) var playheadUs: Int64 = 0
    private(set) var picture: CGImage?
    private(set) var canvasAspect: CGFloat = 16 / 9
    private(set) var isPlaying = false
    private(set) var isImporting = false
    private(set) var notice: Notice?
    var selectedClip: String?

    struct Notice: Equatable {
        let id = UUID()
        let text: String
    }

    private static let log = Logger(subsystem: "cutlass", category: "editor")
    private let editor: Editor?
    private var drawing = false
    private var waiting: Int64?
    private var playback: Task<Void, Never>?

    /// Open the project folder at `project`.
    init(project: String) {
        do {
            try FileManager.default.createDirectory(
                at: Folders.bundle, withIntermediateDirectories: true)
            editor = try Editor.open(
                project: project,
                bundle: Folders.bundle.path(percentEncoded: false),
                cache: Folders.cache.path(percentEncoded: false)
            )
        } catch {
            editor = nil
            show(error)
        }
        reload()
    }

    var hasClips: Bool { lanes.contains { !$0.clips.isEmpty } }

    /// The clip a split cuts: the selection when it covers the playhead,
    /// else the main-lane clip under it.
    var splitTarget: Clip? {
        if let clip = selection, clip.covers(playheadUs) { return clip }
        return clipUnderPlayhead
    }

    /// The clip a delete removes: the selection wherever it is, else the
    /// main-lane clip under the playhead.
    var deleteTarget: Clip? { selection ?? clipUnderPlayhead }

    private var selection: Clip? {
        guard let id = selectedClip else { return nil }
        return lanes.lazy.flatMap(\.clips).first { $0.id == id }
    }

    private var clipUnderPlayhead: Clip? {
        lanes.first(where: \.main)?.clips.first { $0.covers(playheadUs) }
    }

    // MARK: Playhead

    func seek(_ time: Int64) {
        playheadUs = min(max(time, 0), durationUs)
        draw()
    }

    func togglePlay() {
        isPlaying ? pause() : play()
    }

    func play() {
        guard durationUs > 0 else { return }
        if playheadUs >= durationUs { playheadUs = 0 }
        isPlaying = true
        let from = playheadUs
        let started = Date.now
        playback = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let elapsed = Int64(Date.now.timeIntervalSince(started) * 1_000_000)
                let time = from + elapsed
                if time >= durationUs {
                    seek(durationUs)
                    pause()
                    return
                }
                seek(time)
                try? await Task.sleep(for: .milliseconds(16))
            }
        }
    }

    func pause() {
        playback?.cancel()
        playback = nil
        isPlaying = false
    }

    // MARK: Edits

    func importFile(at url: URL, as media: Media) {
        edit {
            let material = try $0.import(path: url.path(percentEncoded: false), media: media)
            let clip = try $0.append(materialId: material)
            selectedClip = clip
        }
    }

    func beginImport() { isImporting = true }
    func endImport() { isImporting = false }

    func split() {
        guard let clip = splitTarget else {
            say("Move the playhead over a clip to split it")
            return
        }
        edit { _ = try $0.split(clipId: clip.id, atUs: playheadUs) }
    }

    func deleteClip() {
        guard let clip = deleteTarget else {
            say("Select a clip to delete")
            return
        }
        edit { try $0.remove(clipId: clip.id) }
        selectedClip = nil
    }

    func undo() { edit { try $0.undo() } }
    func redo() { edit { try $0.redo() } }

    func clearNotice() { notice = nil }

    func say(_ text: String) {
        Self.log.notice("\(text, privacy: .public)")
        notice = Notice(text: text)
    }

    func show(_ error: Error) {
        if case let EditorError.Failed(message) = error {
            say(message)
        } else {
            say(error.localizedDescription)
        }
    }

    /// Run one edit, save the project, and redraw. A refusal shows its reason
    /// and leaves the project as it was.
    private func edit(_ change: (Editor) throws -> Void) {
        guard let editor else { return }
        do {
            try change(editor)
            try editor.save()
        } catch {
            show(error)
        }
        reload()
    }

    private func reload() {
        guard let editor else { return }
        do {
            lanes = try editor.lanes()
            durationUs = try editor.durationUs()
        } catch {
            show(error)
        }
        if let id = selectedClip, !lanes.contains(where: { $0.clips.contains { $0.id == id } }) {
            selectedClip = nil
        }
        playheadUs = min(playheadUs, durationUs)
        draw()
    }

    // MARK: Picture

    private func draw() {
        waiting = playheadUs
        guard !drawing, let editor, let time = waiting else { return }
        waiting = nil
        drawing = true
        Task.detached(priority: .userInitiated) {
            let result = Result { try editor.frame(timeUs: time) }
            await MainActor.run {
                switch result {
                case let .success(frame):
                    if let image = frame.image {
                        self.picture = image
                        self.canvasAspect = CGFloat(frame.width) / CGFloat(max(frame.height, 1))
                    }
                case let .failure(error):
                    self.show(error)
                }
                self.drawing = false
                if self.waiting != nil { self.draw() }
            }
        }
    }
}

extension Clip {
    var endUs: Int64 { startUs + durationUs }

    func covers(_ time: Int64) -> Bool {
        time >= startUs && time < endUs
    }
}

extension Frame {
    /// The RGBA bytes as an image, or `nil` when the size and bytes disagree.
    nonisolated var image: CGImage? {
        let width = Int(self.width)
        let height = Int(self.height)
        guard width > 0, height > 0, rgba.count == width * height * 4,
              let provider = CGDataProvider(data: rgba as CFData)
        else { return nil }
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }
}
