import AVFAudio
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
    /// The cuts between touching main clips, with their transitions.
    private(set) var cuts: [Cut] = []
    /// The cut whose transition the panel edits, by the clip before it.
    var selectedCut: String?
    private(set) var durationUs: Int64 = 0
    private(set) var playheadUs: Int64 = 0 {
        didSet { if playheadUs != oldValue, selectedClip != nil { refreshStage() } }
    }
    private(set) var picture: CGImage?
    private(set) var canvasAspect: CGFloat = 16 / 9
    /// The canvas ratios, Original first, as Rust sizes them for this project.
    private(set) var ratios: [CanvasRatio] = []
    /// What fills the canvas around the main clips.
    private(set) var background: Background = .none
    private(set) var isPlaying = false
    private(set) var isOpening = true
    private(set) var isImporting = false
    private(set) var notice: Notice?
    var selectedClip: String? {
        didSet { if selectedClip != oldValue { refreshStage(now: true) } }
    }
    /// The selected picture clip's box on the canvas, as the picture on
    /// screen shows it. Text and sound have none.
    private(set) var stage: Stage?
    /// The box for the draft as it is now. After an edit it waits for the
    /// picture drawn from that draft, so the box never runs ahead of it.
    private var placed: (clip: String?, stage: Stage?) = (nil, nil)
    /// The selected clip's keyframes, and whether one is under the
    /// playhead. Clips that take none have nil.
    private(set) var keys: Keyframes?
    /// The selected clip's in and out animations. Clips that take none
    /// have nil.
    private(set) var animations: ClipAnimations?
    /// The Animations panel's presets; they never change.
    let animationList = animationPresets()
    /// How the selected clip stands at the playhead, for its clip tools.
    private(set) var clipState: ClipState?
    /// Whether clips on other rows move and go with the main clip under them.
    private(set) var linkage = false
    /// The Filters panel's filters; they never change.
    let filterList = filterChoices()
    /// The filter the selected clip keeps, or a selected filter layer's own.
    private(set) var clipFilter: ClipFilter?
    /// The selected picture's or adjustment layer's sliders.
    private(set) var adjustSliders: [AdjustSlider]?
    /// Filter tile pictures by filter id, drawn by Rust once each.
    private(set) var posters: [String: CGImage] = [:]

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
    /// The Rust editor once open. Hidden while an export holds its thread,
    /// so nothing here waits on it.
    private var editor: Editor? { exporting ? nil : opened }
    private var opened: Editor?
    private(set) var exporting = false
    private var drawing = false
    private var waiting: Int64?
    private var playback: Task<Void, Never>?
    /// The strip sliding on after a swipe.
    private var coasting: Task<Void, Never>?
    /// The window the ticks and tiles cover, and the edit they were read for.
    private var stripWindow: (from: Int64, to: Int64, pixelSize: Double, edit: Int)?
    /// Counts the project's changes, so a panel showing a clip rereads it.
    private(set) var edits = 0
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
                    self.opened = editor
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
    ///
    /// The audio session is made active again first, off the main thread:
    /// iOS lets it go when another app or a video preview takes the sound,
    /// and Rust starts its stopped output again on play.
    func play() {
        guard let editor, durationUs > 0 else { return }
        stopCoast()
        isPlaying = true
        playback = Task.detached(priority: .userInitiated) { [weak self] in
            try? AVAudioSession.sharedInstance().setActive(true)
            // Paused while the session woke.
            if Task.isCancelled { return }
            do {
                try editor.play()
            } catch {
                await MainActor.run {
                    self?.show(error)
                    self?.stopped()
                }
                return
            }
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

    // MARK: Export

    /// Stop playing and hand the editor to an export. Every other call
    /// waits until [`endExport`] gives it back.
    func beginExport() -> Editor? {
        guard let editor, durationUs > 0 else {
            say("Add a photo or video first")
            return nil
        }
        stopCoast()
        pause()
        exporting = true
        return editor
    }

    func endExport() {
        exporting = false
        reload()
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

    /// Run `effect` over the selected picture alone, for its span.
    func addClipEffect(_ effect: String) {
        guard let clip = selectedClip else { return }
        pause()
        edit { _ = try $0.addClipEffect(clipId: clip, effectId: effect) }
    }

    // MARK: Filters and adjust

    /// Pick `filter` (empty for none): on the selected picture or filter
    /// layer it takes the place of the one there; with nothing selected a
    /// filter layer goes at the playhead and is selected.
    func pickFilter(_ filter: String) {
        pause()
        if let clip = selectedClip {
            var placed: String?
            edit { placed = try $0.setClipFilter(clipId: clip, filterId: filter) }
            // A filter layer swapped in place is a new clip: select it.
            if let placed, selectedClip == nil { selectedClip = placed }
        } else if !filter.isEmpty {
            var added: String?
            edit { added = try $0.addFilter(filterId: filter, atUs: playheadUs) }
            if let added { selectedClip = added }
        }
    }

    /// How strong the kept filter is, 0 to 1.
    func setFilterStrength(_ value: Double) {
        guard let layer = clipFilter?.layerId else { return }
        edit { try $0.setStrength(clipId: layer, value: value) }
    }

    /// Put an adjustment layer at the playhead and select it.
    func addAdjustment() {
        pause()
        var added: String?
        edit { added = try $0.addAdjustment(atUs: playheadUs) }
        if let added { selectedClip = added }
    }

    func setAdjust(_ key: String, _ value: Double) {
        guard let clip = selectedClip else { return }
        edit { try $0.setAdjust(clipId: clip, key: key, value: value) }
    }

    /// The tile picture for `filter`, drawn off the main thread once.
    func poster(_ filter: String) -> CGImage? {
        if let image = posters[filter] { return image }
        if !drawingPosters.contains(filter) {
            drawingPosters.insert(filter)
            Task.detached(priority: .utility) {
                let image = filterPoster(filterId: filter, width: 120, height: 120)?.image
                await MainActor.run {
                    if let image { self.posters[filter] = image }
                }
            }
        }
        return nil
    }

    private var drawingPosters: Set<String> = []

    // MARK: Text

    /// Type into `clip` as the words change, or add a title at the playhead
    /// in `face` once the words are not blank, and select it. The whole
    /// typing is one undo step. Returns the clip, or nil while none exists.
    func typeText(into clip: String?, _ content: String, face: String) -> String? {
        var typed = clip
        edit {
            typed = try $0.typeText(clipId: clip, content: content, face: face, atUs: playheadUs)
        }
        if let typed { selectedClip = typed }
        return typed
    }

    func setTextFont(_ clip: String, face: String) {
        edit { try $0.setTextFont(clipId: clip, face: face) }
    }

    /// Change how `clip` looks. Changes while `live` join one undo step and
    /// are saved when the last one, not live, arrives.
    func styleText(_ clip: String, _ change: TextStyleChange, live: Bool) {
        edit(save: !live) { try $0.styleText(clipId: clip, change: change, live: live) }
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

    /// Give the canvas a ratio. The clips keep their place and fit the new frame.
    func setRatio(_ ratio: String) {
        edit { try $0.setRatio(ratio: ratio) }
    }

    /// Fill the canvas around the main clips.
    func setBackground(_ background: Background) {
        edit { try $0.setBackground(background: background) }
    }

    /// Play the clip at a constant speed. Its length follows.
    func setSpeed(_ clip: String, _ speed: Double) {
        edit { try $0.setSpeed(clipId: clip, speed: speed) }
    }

    // MARK: Clip tools

    /// Put a copy of the selected clip right after it, and select the copy.
    func duplicate() {
        guard let clip = selectedClip else { return say("Select a clip to duplicate") }
        edit { selectedClip = try $0.duplicate(clipId: clip) }
    }

    func toggleMirror() {
        guard let clip = selectedClip else { return }
        edit { try $0.toggleMirror(clipId: clip) }
    }

    func turn() {
        guard let clip = selectedClip else { return }
        edit { try $0.turn(clipId: clip) }
    }

    func toggleReverse() {
        guard let clip = selectedClip else { return }
        pause()
        let backward = !(clipState?.reversed ?? false)
        edit { try $0.toggleReverse(clipId: clip) }
        say(backward ? "The clip plays backward" : "The clip plays forward")
    }

    /// Hold the frame under the playhead as a still there, and select it.
    func freeze() {
        guard let clip = selectedClip else { return }
        pause()
        let at = playheadUs
        edit { selectedClip = try $0.freeze(clipId: clip, atUs: at) }
    }

    /// Put the video's sound on its own clip, or give it back when it was
    /// already taken out.
    func extractOrRecoverAudio() {
        guard let clip = selectedClip else { return }
        if clipState?.extracted == true {
            edit { try $0.recoverAudio(clipId: clip) }
            say("The clip has its sound again")
        } else {
            edit { try $0.extractAudio(clipId: clip) }
        }
    }

    /// Mark a beat at the playhead on the selected sound, or take it off.
    func toggleBeat() {
        guard let clip = selectedClip else { return }
        pause()
        let at = playheadUs
        edit { try $0.toggleBeat(clipId: clip, atUs: at) }
    }

    /// Move the selected picture from its row onto main.
    func toMain() {
        guard let clip = selectedClip else { return }
        edit { try $0.toMain(clipId: clip) }
    }

    /// Crop the selected video to its middle at `ratio`; empty shows it all.
    func setCrop(_ ratio: String) {
        guard let clip = selectedClip else { return }
        edit { try $0.setCrop(clipId: clip, ratio: ratio) }
    }

    func setOpacity(_ opacity: Double) {
        guard let clip = selectedClip else { return }
        edit { try $0.setOpacity(clipId: clip, opacity: opacity) }
    }

    func setFade(inUs: Int64, outUs: Int64) {
        guard let clip = selectedClip else { return }
        edit { try $0.setFade(clipId: clip, fadeInUs: inUs, fadeOutUs: outUs) }
    }

    /// Turn linkage on or off, saying what it means.
    func toggleLinkage() {
        let on = !linkage
        edit { try $0.setLinkageEnabled(on: on) }
        say(on
            ? "Text, stickers and effects move with the main clip under them"
            : "Text, stickers and effects no longer move or delete with main clips")
    }

    /// Mute or unmute every clip on a row.
    func setMuted(_ lane: Lane, _ muted: Bool) {
        edit { try $0.setRowMuted(rowId: lane.id, muted: muted) }
    }

    /// Bring in a picked photo or video as an overlay at the playhead.
    func addOverlay(at url: URL) {
        let at = playheadUs
        edit {
            let material = try $0.import(path: url.path(percentEncoded: false))
            selectedClip = try $0.addOverlay(materialId: material, atUs: at)
        }
    }

    /// Bring in a sound file from Files at the end of the sound row.
    func importSound(at url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        edit {
            let material = try $0.import(path: url.path(percentEncoded: false))
            selectedClip = try $0.append(materialId: material)
        }
    }

    /// Bring in the sound of a picked video at the end of the sound row.
    func extractSound(at url: URL) {
        edit {
            let material = try $0.importSoundOf(path: url.path(percentEncoded: false))
            selectedClip = try $0.append(materialId: material)
        }
    }

    /// Play the picked file in the selected clip in place of its media.
    func replaceSelected(with url: URL) {
        guard let clip = selectedClip else { return }
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        edit { try $0.replaceFile(clipId: clip, path: url.path(percentEncoded: false)) }
    }

    /// Bring in an SRT or LRC file from Files as captions.
    func importCaptions(at url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            let text = try String(contentsOf: url, encoding: .utf8)
            edit { try $0.importSubtitles(fileName: url.lastPathComponent, text: text) }
        } catch {
            show(error)
        }
    }

    // MARK: Stickers and drawing

    let stickerList = stickerChoices()
    let shapeList = shapeChoices()

    /// Put a sticker at the playhead and select it.
    func addSticker(_ id: String) {
        let at = playheadUs
        edit { selectedClip = try $0.addSticker(stickerId: id, atUs: at) }
    }

    /// Put a shape at the playhead and select it.
    func addShape(_ id: String) {
        let at = playheadUs
        edit { selectedClip = try $0.addShape(shapeId: id, atUs: at) }
    }

    /// The pen's color, `#RRGGBB`, and weight in canvas pixels.
    var penColor = "#FFFFFF"
    var penWidth = shapeStyleRanges().penStart

    /// Put a finger's stroke, in canvas pixels, at the playhead, in the
    /// pen's color and weight.
    func drawStroke(_ points: [CGPoint]) {
        let at = playheadUs
        let trace = points.map { CanvasPoint(x: $0.x, y: $0.y) }
        let (color, width) = (penColor, penWidth)
        edit { _ = try $0.drawStroke(trace: trace, atUs: at, color: color, width: width) }
    }

    /// How `clip` looks, or nil when it is not a shape.
    func shapeStyle(of clip: String) -> ShapeStyle? {
        (try? editor?.shapeStyle(clipId: clip)) ?? nil
    }

    /// Change how shape `clip` looks. Changes while `live` join one undo
    /// step and are saved when the last one, not live, arrives.
    func styleShape(_ clip: String, _ change: ShapeStyleChange, live: Bool) {
        edit(save: !live) { try $0.styleShape(clipId: clip, change: change, live: live) }
    }

    /// A sticker's shelf tile, drawn once off the main thread.
    func stickerPoster(_ id: String) -> CGImage? {
        let key = "sticker:" + id
        if let image = posters[key] { return image }
        if !drawingPosters.contains(key) {
            drawingPosters.insert(key)
            Task.detached(priority: .utility) {
                let image = cutlassStickerPoster(id)
                await MainActor.run {
                    if let image { self.posters[key] = image }
                }
            }
        }
        return nil
    }

    // MARK: Transitions

    let transitionList = transitionChoices()
    let transitionTabs = transitionCategories()

    /// The cut the transition panel edits.
    var cut: Cut? { cuts.first { $0.outgoingId == selectedCut } }

    /// Put transition `id` on the selected cut, or take it off for nil.
    func setTransition(_ id: String?) {
        guard let outgoing = selectedCut else { return }
        pause()
        edit {
            if let id {
                try $0.setTransition(outgoingId: outgoing, transitionId: id)
            } else {
                try $0.removeTransition(outgoingId: outgoing)
            }
        }
        // Play the cut so the transition shows.
        if let cut { seek(max(cut.atUs - cut.durationUs, 0)) }
    }

    func setTransitionLength(_ us: Int64) {
        guard let outgoing = selectedCut else { return }
        edit { try $0.setTransitionLength(outgoingId: outgoing, durationUs: us) }
    }

    /// Put the selected cut's transition on every cut of the main row.
    func transitionEverywhere() {
        guard let outgoing = selectedCut else { return }
        edit { try $0.transitionEverywhere(outgoingId: outgoing) }
        say("Applied to every cut")
    }

    // MARK: Stage

    /// The canvas size in pixels, from the chosen ratio.
    var canvasSize: CGSize? {
        guard let ratio = ratios.first(where: \.chosen), ratio.width > 0, ratio.height > 0 else {
            return nil
        }
        return CGSize(width: Double(ratio.width), height: Double(ratio.height))
    }

    /// Select the topmost picture under the canvas point, or clear the
    /// selection when there is none.
    func pick(at point: CGPoint) {
        guard let editor else { return }
        selectedClip = try? editor.stagePick(atUs: playheadUs, x: point.x, y: point.y)
    }

    /// Start moving `clip` on the preview, selecting it. False when it
    /// cannot move there.
    func beginStage(_ clip: String) -> Bool {
        guard let editor else { return false }
        pause()
        do {
            try editor.stageBegin(clipId: clip, atUs: playheadUs)
        } catch {
            return false
        }
        selectedClip = clip
        return true
    }

    /// Place the clip being moved. Rust snaps it and keeps one undo step.
    func moveStage(shift: CGSize, pinch: Double, turn: Double) {
        guard let editor else { return }
        do {
            let moved = try editor.stageMove(
                shiftX: shift.width, shiftY: shift.height, pinch: pinch, turn: turn
            )
            placed = (selectedClip, moved)
        } catch {
            show(error)
            return
        }
        draw()
    }

    /// End the preview gesture and save.
    func endStage() {
        guard let editor else { return }
        edit { _ in try editor.stageEnd() }
    }

    /// Read the selected box again. `now` shows it at once, for a new
    /// selection; otherwise it waits for the next picture.
    private func refreshStage(now: Bool = false) {
        let at = playheadUs
        let box = selectedClip.flatMap { clip in try? editor?.stage(clipId: clip, atUs: at) }
        placed = (selectedClip, box)
        if now { stage = box }
        keys = selectedClip.flatMap { clip in try? editor?.keyframes(clipId: clip, atUs: at) }
        animations = selectedClip.flatMap { clip in try? editor?.animations(clipId: clip) }
        clipState = selectedClip.flatMap { clip in try? editor?.clipState(clipId: clip, atUs: at) }
        clipFilter = selectedClip.flatMap { clip in try? editor?.clipFilter(clipId: clip) }
        adjustSliders = selectedClip.flatMap { clip in try? editor?.adjustSliders(clipId: clip) }
    }

    /// The keyframe button: remove the keyframe under the playhead, else key
    /// the selected clip there.
    func toggleKeyframe() {
        guard let clip = selectedClip else { return }
        pause()
        let at = playheadUs
        edit { try $0.toggleKeyframe(clipId: clip, atUs: at) }
    }

    /// Ease the span the playhead is in along the named curve; an empty
    /// name makes it straight.
    func easeKeyframes(_ graph: String) {
        guard let clip = selectedClip else { return }
        pause()
        let at = playheadUs
        edit { try $0.easeKeyframes(clipId: clip, atUs: at, graph: graph) }
    }

    /// Give the selected clip the in (`enter`) or out animation `id`, or
    /// clear that side with an empty id, as one undo step.
    func setAnimation(enter: Bool, id: String, durationUs: Int64) {
        guard let clip = selectedClip else { return }
        pause()
        edit { try $0.setAnimation(clipId: clip, enter: enter, id: id, durationUs: durationUs) }
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
    private func edit(save: Bool = true, _ change: (Editor) throws -> Void) {
        guard let editor else { return }
        do {
            try change(editor)
            if save { try editor.save() }
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
            cuts = try editor.cuts()
            durationUs = try editor.durationUs()
            ratios = try editor.ratios()
            background = try editor.background()
            linkage = try editor.linkageEnabled()
        } catch {
            show(error)
        }
        if let id = selectedClip, !lanes.contains(where: { $0.clips.contains { $0.id == id } }) {
            selectedClip = nil
        }
        if let id = selectedCut, !cuts.contains(where: { $0.outgoingId == id }) {
            selectedCut = nil
        }
        playheadUs = min(playheadUs, durationUs)
        refreshStage()
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
        // About 180 KB a tile at the worker's height.
        if thumbs.count > 300 { thumbs.removeAll() }
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
        let box = placed
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
                if box.clip == self.selectedClip { self.stage = box.stage }
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

/// A sticker's shelf tile. Kept outside the model, whose own
/// `stickerPoster` would hide the Rust call of the same name.
nonisolated private func cutlassStickerPoster(_ id: String) -> CGImage? {
    stickerPoster(stickerId: id, side: 128)?.image
}
