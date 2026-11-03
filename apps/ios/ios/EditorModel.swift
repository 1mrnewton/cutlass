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
    private(set) var isOpening = true
    private(set) var isImporting = false
    private(set) var notice: Notice?
    var selectedClip: String?

    /// Points per microsecond on the timeline. Rust holds it inside what
    /// the cut allows.
    private(set) var pixelSize: Double = 60 / 1_000_000
    /// Ruler ticks and filmstrip tiles for the window around the playhead.
    private(set) var ticks: [Tick] = []
    private(set) var films: [String: [Tile]] = [:]
    /// Each clip's sound over the window read, by clip id.
    private(set) var waves: [String: Wave] = [:]
    /// Filmstrip pictures by tile key.
    private(set) var thumbs: [String: CGImage] = [:]

    struct Notice: Equatable {
        let id = UUID()
        let text: String
    }

    private static let log = Logger(subsystem: "cutlass", category: "editor")
    /// Where the system keeps the faces text can be set in.
    nonisolated private static let fontFolders = ["/System/Library/Fonts"]
    /// The faces text can be set in, as Rust found them.
    private(set) var fonts: [Font] = []
    private var editor: Editor?
    private var drawing = false
    private var waiting: Int64?
    private var playback: Task<Void, Never>?
    /// The strip sliding on after a swipe.
    private var coasting: Task<Void, Never>?
    /// The window the ticks and tiles cover, and the edit they were read for.
    private var stripWindow: (from: Int64, to: Int64, pixelSize: Double, edit: Int)?
    private var edits = 0
    private var stripRetry: Task<Void, Never>?
    private var fetching: Set<String> = []

    /// Open the project folder at `project`. Rust reads it off the main
    /// thread; the screen shows a spinner until it is open.
    init(project: String) {
        let root = Places.root
        let bundle = Places.bundle
        Task.detached(priority: .userInitiated) { [weak self] in
            let opened = Result {
                let editor = try Editor.open(root: root, bundle: bundle, project: project)
                try editor.setFontFolders(folders: Self.fontFolders)
                return editor
            }
            await MainActor.run {
                guard let self else { return }
                switch opened {
                case let .success(editor):
                    self.editor = editor
                    self.fonts = (try? editor.fonts()) ?? []
                    self.reload()
                case let .failure(error):
                    self.show(error)
                }
                self.isOpening = false
            }
        }
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

    /// The selected clip as the project holds it now.
    var selection: Clip? {
        guard let id = selectedClip else { return nil }
        return lanes.lazy.flatMap(\.clips).first { $0.id == id }
    }

    /// The tool bar for the selected clip, as Rust picks it for its row.
    var selectionBar: ToolBar? {
        guard let editor, let id = selectedClip else { return nil }
        return try? editor.clipBar(clipId: id)
    }

    private var clipUnderPlayhead: Clip? {
        lanes.first(where: \.main)?.clips.first { $0.covers(playheadUs) }
    }

    // MARK: Playhead

    func seek(_ time: Int64) {
        playheadUs = min(max(time, 0), durationUs)
        try? editor?.seek(timeUs: playheadUs)
        draw()
    }

    /// Let the strip slide on after a swipe let go at `velocity` points a
    /// second, finger right being back in time. Rust steps the slowing.
    func coast(from velocity: Double) {
        stopCoast()
        var speed = flingStart(velocity: velocity)
        guard speed != 0 else { return }
        coasting = Task { [weak self] in
            var last = ContinuousClock.now
            while speed != 0 {
                try? await Task.sleep(for: .milliseconds(16))
                guard let self, !Task.isCancelled else { return }
                let now = ContinuousClock.now
                let elapsed = (now - last).components
                last = now
                let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
                let step = flingStep(velocity: speed, elapsed: seconds)
                speed = step.velocity
                let target = playheadUs - Int64(step.distance / pixelSize)
                seek(target)
                if target <= 0 || target >= durationUs { return }
            }
        }
    }

    /// Stop a coast where it is, as a finger landing on the strip does.
    func stopCoast() {
        coasting?.cancel()
        coasting = nil
    }

    func togglePlay() {
        isPlaying ? pause() : play()
    }

    /// Play with sound. Rust moves the playhead with the heard audio; each
    /// screen frame asks it where the playhead is, off the main thread, and
    /// draws that picture.
    func play() {
        guard let editor, durationUs > 0 else { return }
        stopCoast()
        do {
            try editor.play()
        } catch {
            show(error)
            return
        }
        isPlaying = true
        playback = Task.detached(priority: .userInitiated) { [weak self] in
            var last = ContinuousClock.now
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(16))
                let now = ContinuousClock.now
                let elapsed = (now - last).components
                last = now
                let micros = elapsed.seconds * 1_000_000 + elapsed.attoseconds / 1_000_000_000_000
                guard let time = try? editor.tick(elapsedUs: micros),
                      let playing = try? editor.isPlaying()
                else {
                    await MainActor.run { self?.stopped() }
                    return
                }
                let going = await MainActor.run { () -> Bool in
                    guard let self, self.isPlaying else { return false }
                    self.playheadUs = time
                    self.draw()
                    if !playing { self.stopped() }
                    return playing
                }
                if !going { return }
            }
        }
    }

    func pause() {
        try? editor?.pause()
        stopped()
    }

    private func stopped() {
        playback?.cancel()
        playback = nil
        isPlaying = false
    }

    // MARK: Edits

    func importFile(at url: URL) {
        edit {
            let material = try $0.import(path: url.path(percentEncoded: false))
            let clip = try $0.append(materialId: material)
            selectedClip = clip
        }
    }

    func beginImport() { isImporting = true }
    func endImport() { isImporting = false }

    /// Select the main clip under the playhead, for its tools.
    func editUnderPlayhead() {
        guard let clip = clipUnderPlayhead ?? lanes.first(where: \.main)?.clips.last else {
            say("Add a photo or video first")
            return
        }
        selectedClip = clip.id
    }

    func split() {
        guard let clip = splitTarget else {
            say("Move the playhead over a clip to split it")
            return
        }
        edit { _ = try $0.split(clipId: clip.id, atUs: playheadUs) }
    }

    // MARK: Effects

    /// The effects that ship in the app, in picker order.
    let effects: [EffectChoice] = effectChoices()

    /// Add `effect` at the playhead and select it.
    func addEffect(_ effect: String) {
        var added: String?
        edit { added = try $0.addEffect(effectId: effect, atUs: playheadUs) }
        if let added { selectedClip = added }
    }

    // MARK: Text

    /// Add `content` at the playhead in `face` and select it.
    func addText(_ content: String, face: String) {
        var added: String?
        edit { added = try $0.addText(content: content, face: face, atUs: playheadUs) }
        if let added { selectedClip = added }
    }

    func setText(_ clip: String, _ content: String) {
        edit { try $0.setText(clipId: clip, content: content) }
    }

    func setTextFont(_ clip: String, face: String) {
        edit { try $0.setTextFont(clipId: clip, face: face) }
    }

    /// What `clip` says, or nil when it is not a text clip.
    func text(of clip: String) -> TextInfo? {
        try? editor?.text(clipId: clip)
    }

    func deleteClip() {
        guard let clip = deleteTarget else {
            say("Select a clip to delete")
            return
        }
        edit { try $0.remove(clipId: clip.id) }
        selectedClip = nil
    }

    /// Drop `clip` on `row` near `startUs`. It may snap within `snapUs` and
    /// packs in on a magnetic row.
    func moveClip(_ clip: String, row: Int, startUs: Int64, snapUs: Int64) {
        edit {
            _ = try $0.moveClip(
                clipId: clip, row: UInt32(max(row, 0)), startUs: max(startUs, 0),
                snapUs: snapUs, playheadUs: playheadUs)
        }
    }

    /// Drag the start or end edge of `clip` to `timeUs`.
    func trim(_ clip: String, startEdge: Bool, timeUs: Int64, snapUs: Int64) {
        edit {
            _ = try $0.trim(
                clipId: clip, startEdge: startEdge, timeUs: timeUs, snapUs: snapUs,
                playheadUs: playheadUs)
        }
    }

    /// Set the clip's linear gain: 1 is unity and 0 is silence.
    func setVolume(_ clip: String, _ volume: Double) {
        edit { try $0.setVolume(clipId: clip, volume: volume) }
    }

    /// Play the clip at a constant speed. Its length follows.
    func setSpeed(_ clip: String, _ speed: Double) {
        edit { try $0.setSpeed(clipId: clip, speed: speed) }
    }

    /// Put a main clip at `place` in the main row's order.
    func reorder(_ clip: String, place: Int) {
        edit { try $0.reorder(clipId: clip, place: UInt32(max(place, 0))) }
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
        edits += 1
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

    // MARK: Strip

    /// Zoom by `factor` for a strip `width` points wide.
    func zoom(from start: Double, by factor: Double, width: Double) {
        guard let editor, let size = try? editor.zoom(
            pixelSize: start, factor: factor, laneWidth: width)
        else { return }
        pixelSize = size
    }

    /// Read the ticks and tiles when the strip, `halfWidth` points each side
    /// of the playhead, leaves the window last read, the zoom changes, or
    /// the cut changes. The window reaches a screen past each side.
    func refreshStrip(halfWidth: Double) {
        guard let editor, halfWidth > 0 else { return }
        lastHalfWidth = halfWidth
        let reach = Int64(halfWidth / pixelSize)
        let from = playheadUs - reach
        let to = playheadUs + reach
        if let window = stripWindow, window.pixelSize == pixelSize, window.edit == edits,
           window.from <= from, window.to >= to
        {
            return
        }
        let wide = (from: max(playheadUs - 3 * reach, 0), to: playheadUs + 3 * reach)
        stripWindow = (wide.from, wide.to, pixelSize, edits)
        do {
            ticks = try editor.ruler(pixelSize: pixelSize, fromUs: wide.from, toUs: wide.to)
            let read = try editor.film(
                fromUs: wide.from, toUs: wide.to, pixelSize: pixelSize, height: Self.filmHeight)
            films = Dictionary(read.map { ($0.clipId, $0.tiles) }, uniquingKeysWith: { a, _ in a })
            let sound = try editor.waves(fromUs: wide.from, toUs: wide.to, pixelSize: pixelSize)
            waves = Dictionary(sound.map { ($0.clipId, $0) }, uniquingKeysWith: { a, _ in a })
        } catch {
            show(error)
            return
        }
        fetchThumbs()
    }

    /// Tile height the filmstrip asks for, in points.
    static let filmHeight: Double = 52

    /// Fetch the stored pictures not yet held. While some are still being
    /// decoded, read the tiles again shortly.
    private func fetchThumbs() {
        guard let editor else { return }
        let tiles = films.values.joined()
        let wanted = tiles.filter { $0.ready && thumbs[$0.key] == nil && !fetching.contains($0.key) }
        for tile in wanted {
            fetching.insert(tile.key)
            let key = tile.key
            Task.detached(priority: .utility) {
                let image = (try? editor.thumb(key: key))??.image
                await MainActor.run {
                    self.fetching.remove(key)
                    if let image { self.thumbs[key] = image }
                }
            }
        }
        if thumbs.count > 600 { thumbs.removeAll() }
        stripRetry?.cancel()
        if tiles.contains(where: { !$0.ready }) || waves.values.contains(where: { !$0.ready }) {
            stripRetry = Task { [weak self] in
                try? await Task.sleep(for: .milliseconds(250))
                guard !Task.isCancelled, let self else { return }
                self.stripWindow = nil
                self.refreshStrip(halfWidth: self.lastHalfWidth)
            }
        }
    }

    /// The half width the strip last drew at, for a retry.
    var lastHalfWidth: Double = 0

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
