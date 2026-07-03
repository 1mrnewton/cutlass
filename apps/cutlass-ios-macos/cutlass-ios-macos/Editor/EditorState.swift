import SwiftUI

/// One undoable state of the whole mock timeline.
nonisolated struct TimelineSnapshot: Equatable {
    var clips: [MockClip] = []
    var overlays: [MockOverlayClip] = []
    var effects: [MockEffectClip] = []
    var audios: [MockAudioClip] = []
    var aspect: AspectRatio = .original
    var background = CanvasBackground()
}

/// In-memory state for the mock editor: a sequential main video track plus
/// free-floating overlay (text/sticker/PiP), effect, and audio lanes.
/// All edits are pure array/state manipulation; nothing touches the engine.
@Observable
final class EditorState {
    var clips: [MockClip] = []
    var overlayClips: [MockOverlayClip] = []
    var effectClips: [MockEffectClip] = []
    var audioClips: [MockAudioClip] = []
    var aspect: AspectRatio = .original
    var background = CanvasBackground()

    var playhead: TimeInterval = 0
    var selection: TimelineSelection?

    var isPlaying = false {
        didSet {
            guard oldValue != isPlaying else { return }
            if isPlaying {
                startPlayback()
            } else {
                playbackTask?.cancel()
                playbackTask = nil
            }
        }
    }

    /// Timeline zoom: how many seconds one point of track width represents.
    var secondsPerPoint: Double = 1.0 / 44.0

    private var undoStack: [TimelineSnapshot] = []
    private var redoStack: [TimelineSnapshot] = []
    private var playbackTask: Task<Void, Never>?

    var canUndo: Bool { !undoStack.isEmpty }
    var canRedo: Bool { !redoStack.isEmpty }

    var isEmpty: Bool {
        clips.isEmpty && overlayClips.isEmpty && effectClips.isEmpty && audioClips.isEmpty
    }

    /// End of the sequential main track.
    var mainDuration: TimeInterval {
        clips.reduce(0) { $0 + $1.length }
    }

    /// End of the whole timeline including floating lane clips.
    var duration: TimeInterval {
        var end = mainDuration
        for clip in overlayClips { end = max(end, clip.start + clip.length) }
        for clip in effectClips { end = max(end, clip.start + clip.length) }
        for clip in audioClips { end = max(end, clip.start + clip.length) }
        return end
    }

    // MARK: Selection accessors

    var selectedClip: MockClip? {
        guard case .main(let id) = selection else { return nil }
        return clips.first { $0.id == id }
    }

    var selectedOverlay: MockOverlayClip? {
        guard case .overlay(let id) = selection else { return nil }
        return overlayClips.first { $0.id == id }
    }

    var selectedEffect: MockEffectClip? {
        guard case .effect(let id) = selection else { return nil }
        return effectClips.first { $0.id == id }
    }

    var selectedAudio: MockAudioClip? {
        guard case .audio(let id) = selection else { return nil }
        return audioClips.first { $0.id == id }
    }

    // MARK: Time <-> clip mapping

    /// Timeline start time of the given main-track clip.
    func startTime(of id: MockClip.ID) -> TimeInterval {
        var start: TimeInterval = 0
        for clip in clips {
            if clip.id == id { break }
            start += clip.length
        }
        return start
    }

    /// The main-track clip under a timeline position. Holds the last frame at
    /// the exact end of the main track; nil past it (lane content may extend
    /// further and renders over the canvas background).
    func clip(at time: TimeInterval) -> MockClip? {
        var start: TimeInterval = 0
        for clip in clips {
            let end = start + clip.length
            if time < end { return clip }
            start = end
        }
        return time <= start + 0.001 ? clips.last : nil
    }

    /// Overlay clips visible at a timeline position (text, stickers, PiP).
    func overlays(at time: TimeInterval) -> [MockOverlayClip] {
        overlayClips.filter { time >= $0.start && time < $0.start + $0.length }
    }

    /// Effect bars active at a timeline position.
    func effects(at time: TimeInterval) -> [MockEffectClip] {
        effectClips.filter { time >= $0.start && time < $0.start + $0.length }
    }

    // MARK: Snapshots

    private var snapshot: TimelineSnapshot {
        get {
            TimelineSnapshot(
                clips: clips,
                overlays: overlayClips,
                effects: effectClips,
                audios: audioClips,
                aspect: aspect,
                background: background
            )
        }
        set {
            clips = newValue.clips
            overlayClips = newValue.overlays
            effectClips = newValue.effects
            audioClips = newValue.audios
            aspect = newValue.aspect
            background = newValue.background
        }
    }

    private func pushUndoSnapshot() {
        // While a panel session is open the session snapshot owns undo; ops
        // triggered from inside the panel fold into one step on Apply.
        guard panelSnapshot == nil else { return }
        undoStack.append(snapshot)
        if undoStack.count > 50 {
            undoStack.removeFirst()
        }
        redoStack = []
    }

    // MARK: Project lifecycle

    func startProject(with items: [MockMediaItem]) {
        isPlaying = false
        snapshot = TimelineSnapshot(clips: items.map(MockClip.init(from:)))
        playhead = 0
        selection = nil
        undoStack = []
        redoStack = []
    }

    func appendMedia(_ items: [MockMediaItem]) {
        guard !items.isEmpty else { return }
        pushUndoSnapshot()
        clips.append(contentsOf: items.map(MockClip.init(from:)))
    }

    // MARK: Transport

    func stepFrame(by direction: Double) {
        isPlaying = false
        playhead = min(max(0, playhead + direction / 30.0), duration)
    }

    /// Advances the playhead in wall-clock time until the timeline ends or
    /// playback is stopped (pause button, scrubbing, frame step).
    private func startPlayback() {
        if playhead >= duration {
            playhead = 0
        }
        playbackTask = Task { [weak self] in
            guard let self else { return }
            var lastTick = Date.now
            while !Task.isCancelled, self.isPlaying {
                try? await Task.sleep(for: .milliseconds(16))
                guard !Task.isCancelled else { return }

                let now = Date.now
                self.playhead = min(self.playhead + now.timeIntervalSince(lastTick), self.duration)
                lastTick = now

                if self.playhead >= self.duration {
                    self.isPlaying = false
                }
            }
        }
    }

    // MARK: Undo / redo

    func undo() {
        guard let previous = undoStack.popLast() else { return }
        isPlaying = false
        redoStack.append(snapshot)
        snapshot = previous
        reconcileAfterHistoryJump()
    }

    func redo() {
        guard let next = redoStack.popLast() else { return }
        isPlaying = false
        undoStack.append(snapshot)
        snapshot = next
        reconcileAfterHistoryJump()
    }

    private func reconcileAfterHistoryJump() {
        switch selection {
        case .main(let id) where !clips.contains(where: { $0.id == id }),
             .overlay(let id) where !overlayClips.contains(where: { $0.id == id }),
             .effect(let id) where !effectClips.contains(where: { $0.id == id }),
             .audio(let id) where !audioClips.contains(where: { $0.id == id }):
            selection = nil
        default:
            break
        }
        clampPlayhead()
    }

    // MARK: Panel edit sessions

    /// Property panels mutate state live so the preview reacts; the session
    /// snapshot makes Cancel restore and Apply undoable as one step.
    private var panelSnapshot: TimelineSnapshot?

    func beginPanelSession() {
        panelSnapshot = snapshot
    }

    func commitPanelSession() {
        if let before = panelSnapshot, before != snapshot {
            undoStack.append(before)
            redoStack = []
        }
        panelSnapshot = nil
    }

    func cancelPanelSession() {
        if let before = panelSnapshot {
            snapshot = before
        }
        panelSnapshot = nil
    }

    // MARK: Targeted mutation helpers (used by property panels)

    func updateSelectedClip(_ mutate: (inout MockClip) -> Void) {
        guard case .main(let id) = selection,
              let index = clips.firstIndex(where: { $0.id == id })
        else { return }
        mutate(&clips[index])
    }

    func updateSelectedOverlay(_ mutate: (inout MockOverlayClip) -> Void) {
        guard case .overlay(let id) = selection,
              let index = overlayClips.firstIndex(where: { $0.id == id })
        else { return }
        mutate(&overlayClips[index])
    }

    func updateSelectedEffect(_ mutate: (inout MockEffectClip) -> Void) {
        guard case .effect(let id) = selection,
              let index = effectClips.firstIndex(where: { $0.id == id })
        else { return }
        mutate(&effectClips[index])
    }

    func updateSelectedAudio(_ mutate: (inout MockAudioClip) -> Void) {
        guard case .audio(let id) = selection,
              let index = audioClips.firstIndex(where: { $0.id == id })
        else { return }
        mutate(&audioClips[index])
    }

    /// Changing speed rescales the clip's timeline length so the same source
    /// content plays faster or slower.
    func setSelectedSpeed(_ newSpeed: Double) {
        updateSelectedClip { clip in
            let content = clip.length * clip.speed
            clip.speed = newSpeed
            clip.length = max(MockClip.minDuration, content / newSpeed)
        }
    }

    // MARK: Adding lane content (all insert at the playhead and select)

    @discardableResult
    func addTextClip(text: String = "") -> UUID {
        pushUndoSnapshot()
        var clip = MockOverlayClip(kind: .text, start: insertionTime, length: 3)
        clip.text = text
        clip.posY = 0.62
        overlayClips.append(clip)
        selection = .overlay(clip.id)
        return clip.id
    }

    @discardableResult
    func addSticker(symbol: String) -> UUID {
        pushUndoSnapshot()
        var clip = MockOverlayClip(kind: .sticker, start: insertionTime, length: 3)
        clip.symbol = symbol
        clip.posY = 0.35
        overlayClips.append(clip)
        selection = .overlay(clip.id)
        return clip.id
    }

    @discardableResult
    func addPip(from item: MockMediaItem) -> UUID {
        pushUndoSnapshot()
        let length = item.videoDuration ?? MockClip.photoDefaultDuration
        var clip = MockOverlayClip(kind: .pip, start: insertionTime, length: length)
        clip.art = item.art
        clip.sourceDuration = item.videoDuration ?? MockClip.photoMaxDuration
        clip.scale = 0.5
        clip.posY = 0.32
        overlayClips.append(clip)
        selection = .overlay(clip.id)
        return clip.id
    }

    @discardableResult
    func addEffectClip(name: String, kind: MockEffectClip.Kind) -> UUID {
        pushUndoSnapshot()
        let clip = MockEffectClip(kind: kind, name: name, start: insertionTime, length: 3)
        effectClips.append(clip)
        selection = .effect(clip.id)
        return clip.id
    }

    @discardableResult
    func addAudio(kind: MockAudioClip.Kind, title: String, duration: TimeInterval) -> UUID {
        pushUndoSnapshot()
        let clip = MockAudioClip(
            kind: kind,
            title: title,
            start: insertionTime,
            length: duration,
            sourceDuration: duration
        )
        audioClips.append(clip)
        selection = .audio(clip.id)
        return clip.id
    }

    /// Lane content lands at the playhead, clamped inside the timeline.
    private var insertionTime: TimeInterval {
        max(0, min(playhead, max(duration - 0.5, 0)))
    }

    // MARK: Edit operations

    /// Reorders a main-track clip (drag-to-reorder drop).
    func moveClip(fromIndex: Int, toIndex: Int) {
        guard clips.indices.contains(fromIndex),
              clips.indices.contains(toIndex),
              fromIndex != toIndex
        else { return }
        pushUndoSnapshot()
        let clip = clips.remove(at: fromIndex)
        clips.insert(clip, at: toIndex)
    }

    /// Splits whatever is selected at the playhead; with no selection, splits
    /// the main-track clip under the playhead.
    func splitAtPlayhead() {
        switch selection {
        case .overlay(let id):
            if let index = overlayClips.firstIndex(where: { $0.id == id }),
               let (left, right) = splitRange(overlayClips[index].start, overlayClips[index].length) {
                pushUndoSnapshot()
                overlayClips[index].length = left
                var second = overlayClips[index]
                second.id = UUID()
                second.start = playhead
                second.length = right
                overlayClips.insert(second, at: index + 1)
            }
        case .effect(let id):
            if let index = effectClips.firstIndex(where: { $0.id == id }),
               let (left, right) = splitRange(effectClips[index].start, effectClips[index].length) {
                pushUndoSnapshot()
                effectClips[index].length = left
                var second = effectClips[index]
                second.id = UUID()
                second.start = playhead
                second.length = right
                effectClips.insert(second, at: index + 1)
            }
        case .audio(let id):
            if let index = audioClips.firstIndex(where: { $0.id == id }),
               let (left, right) = splitRange(audioClips[index].start, audioClips[index].length) {
                pushUndoSnapshot()
                audioClips[index].length = left
                var second = audioClips[index]
                second.id = UUID()
                second.start = playhead
                second.length = right
                second.waveSeed = Int.random(in: 0..<10_000)
                audioClips.insert(second, at: index + 1)
            }
        default:
            splitMainAtPlayhead()
        }
    }

    /// Left/right lengths if the playhead splits the range non-degenerately.
    private func splitRange(_ start: TimeInterval, _ length: TimeInterval) -> (TimeInterval, TimeInterval)? {
        let local = playhead - start
        guard local >= MockClip.minDuration, local <= length - MockClip.minDuration else { return nil }
        return (local, length - local)
    }

    private func splitMainAtPlayhead() {
        guard let clip = clip(at: playhead),
              let index = clips.firstIndex(where: { $0.id == clip.id })
        else { return }

        let local = playhead - startTime(of: clip.id)
        guard local >= MockClip.minDuration, local <= clip.length - MockClip.minDuration
        else { return }

        pushUndoSnapshot()

        var left = clip
        left.length = local
        left.transitionAfter = nil

        var right = clip
        right.id = UUID()
        right.trimStart = clip.trimStart + local
        right.length = clip.length - local

        clips.replaceSubrange(index...index, with: [left, right])
        if selection == .main(clip.id) {
            selection = .main(left.id)
        }
    }

    func deleteSelected() {
        switch selection {
        case .main(let id):
            pushUndoSnapshot()
            clips.removeAll { $0.id == id }
        case .overlay(let id):
            pushUndoSnapshot()
            overlayClips.removeAll { $0.id == id }
        case .effect(let id):
            pushUndoSnapshot()
            effectClips.removeAll { $0.id == id }
        case .audio(let id):
            pushUndoSnapshot()
            audioClips.removeAll { $0.id == id }
        case nil:
            return
        }
        selection = nil
        clampPlayhead()
    }

    /// Inserts a copy of the selected clip right after it (in time for lane
    /// clips, in order for main-track clips).
    func duplicateSelected() {
        switch selection {
        case .main(let id):
            guard let index = clips.firstIndex(where: { $0.id == id }) else { return }
            pushUndoSnapshot()
            var copy = clips[index]
            copy.id = UUID()
            clips.insert(copy, at: index + 1)
        case .overlay(let id):
            guard let index = overlayClips.firstIndex(where: { $0.id == id }) else { return }
            pushUndoSnapshot()
            var copy = overlayClips[index]
            copy.id = UUID()
            copy.start += copy.length
            overlayClips.append(copy)
            selection = .overlay(copy.id)
        case .effect(let id):
            guard let index = effectClips.firstIndex(where: { $0.id == id }) else { return }
            pushUndoSnapshot()
            var copy = effectClips[index]
            copy.id = UUID()
            copy.start += copy.length
            effectClips.append(copy)
            selection = .effect(copy.id)
        case .audio(let id):
            guard let index = audioClips.firstIndex(where: { $0.id == id }) else { return }
            pushUndoSnapshot()
            var copy = audioClips[index]
            copy.id = UUID()
            copy.start += copy.length
            copy.waveSeed = Int.random(in: 0..<10_000)
            audioClips.append(copy)
            selection = .audio(copy.id)
        case nil:
            return
        }
    }

    /// Swaps the selected clip's source for a picked library item, keeping
    /// its slot on the timeline. Works for main clips and PiP overlays.
    func replaceSelected(with item: MockMediaItem) {
        switch selection {
        case .main(let id):
            guard let index = clips.firstIndex(where: { $0.id == id }) else { return }
            pushUndoSnapshot()
            let replacement = MockClip(from: item)
            clips[index] = replacement
            selection = .main(replacement.id)
        case .overlay(let id):
            guard let index = overlayClips.firstIndex(where: { $0.id == id }),
                  overlayClips[index].kind == .pip
            else { return }
            pushUndoSnapshot()
            overlayClips[index].art = item.art
            overlayClips[index].sourceDuration = item.videoDuration ?? MockClip.photoMaxDuration
            overlayClips[index].length = min(
                overlayClips[index].length,
                overlayClips[index].sourceDuration ?? .greatestFiniteMagnitude
            )
        default:
            return
        }
        clampPlayhead()
    }

    // MARK: Quick ops

    func setTransition(after clipID: MockClip.ID, _ transition: MockTransition?) {
        guard let index = clips.firstIndex(where: { $0.id == clipID }) else { return }
        pushUndoSnapshot()
        clips[index].transitionAfter = transition
    }

    /// Stamps or removes a keyframe on the selected main clip at the playhead.
    func toggleKeyframeAtPlayhead() {
        guard case .main(let id) = selection,
              let index = clips.firstIndex(where: { $0.id == id })
        else { return }

        let local = playhead - startTime(of: id)
        guard local >= 0, local <= clips[index].length else { return }

        pushUndoSnapshot()
        if let existing = clips[index].keyframes.firstIndex(where: { abs($0 - local) < 0.15 }) {
            clips[index].keyframes.remove(at: existing)
        } else {
            clips[index].keyframes.append(local)
            clips[index].keyframes.sort()
        }
    }

    func reverseSelected() {
        guard case .main = selection else { return }
        pushUndoSnapshot()
        updateSelectedClip { $0.isReversed.toggle() }
    }

    /// Inserts a 3-second freeze-frame segment at the playhead inside the
    /// selected clip (or before/after it when the playhead sits at an edge).
    func freezeFrame() {
        guard case .main(let id) = selection,
              let index = clips.firstIndex(where: { $0.id == id })
        else { return }

        let clip = clips[index]
        let local = playhead - startTime(of: id)
        guard local >= -0.001, local <= clip.length + 0.001 else { return }

        pushUndoSnapshot()

        var freeze = clip
        freeze.id = UUID()
        freeze.isFreeze = true
        freeze.hasAudio = false
        freeze.length = 3
        freeze.speed = 1
        freeze.keyframes = []
        freeze.transitionAfter = nil

        if local < MockClip.minDuration {
            clips.insert(freeze, at: index)
        } else if local > clip.length - MockClip.minDuration {
            clips.insert(freeze, at: index + 1)
        } else {
            var left = clip
            left.length = local
            left.transitionAfter = nil
            var right = clip
            right.id = UUID()
            right.trimStart = clip.trimStart + local
            right.length = clip.length - local
            clips.replaceSubrange(index...index, with: [left, freeze, right])
        }
    }

    /// Adds an "extracted audio" lane clip aligned with the selected clip and
    /// mutes the original.
    func extractAudio() {
        guard case .main(let id) = selection,
              let index = clips.firstIndex(where: { $0.id == id }),
              clips[index].hasAudio
        else { return }

        pushUndoSnapshot()
        let clip = clips[index]
        let extracted = MockAudioClip(
            kind: .extracted,
            title: "Extracted audio",
            start: startTime(of: id),
            length: clip.length,
            sourceDuration: clip.length
        )
        audioClips.append(extracted)
        clips[index].volume = 0
    }

    // MARK: Trim / move gestures (main + lane clips)

    enum TrimEdge {
        case leading
        case trailing
    }

    /// Snapshot taken at the first update of a trim/move gesture; committed
    /// to the undo stack when the gesture ends.
    private var gestureSnapshot: TimelineSnapshot?

    private func beginGestureIfNeeded() {
        if gestureSnapshot == nil {
            gestureSnapshot = snapshot
        }
    }

    func endGesture() {
        if let before = gestureSnapshot, before != snapshot {
            undoStack.append(before)
            redoStack = []
        }
        gestureSnapshot = nil
        clampPlayhead()
    }

    /// Applies a trim drag to a main-track clip. `anchor` is the clip as it
    /// was when the drag began, so updates compute from absolute math.
    func trim(_ id: MockClip.ID, edge: TrimEdge, anchor: MockClip, by deltaSeconds: Double) {
        guard let index = clips.firstIndex(where: { $0.id == id }) else { return }
        beginGestureIfNeeded()

        var clip = anchor
        switch edge {
        case .leading:
            let delta = min(
                max(deltaSeconds, -anchor.trimStart),
                anchor.length - MockClip.minDuration
            )
            clip.trimStart = anchor.trimStart + delta
            clip.length = anchor.length - delta
        case .trailing:
            let maxLength = anchor.sourceDuration - anchor.trimStart
            clip.length = min(
                max(anchor.length + deltaSeconds, MockClip.minDuration),
                maxLength
            )
        }
        clips[index] = clip
    }

    /// Shared trim math for free-floating lane clips: leading trims move the
    /// start, trailing trims change the length.
    private func trimmedRange(
        edge: TrimEdge,
        start: TimeInterval,
        length: TimeInterval,
        delta: TimeInterval,
        maxLength: TimeInterval?
    ) -> (start: TimeInterval, length: TimeInterval) {
        switch edge {
        case .leading:
            let clamped = min(max(delta, -start), length - MockClip.minDuration)
            return (start + clamped, length - clamped)
        case .trailing:
            var newLength = max(length + delta, MockClip.minDuration)
            if let maxLength {
                newLength = min(newLength, maxLength)
            }
            return (start, newLength)
        }
    }

    /// Applies a trim drag to a lane clip, anchored at the range captured
    /// when the gesture began. PiP and audio clips clamp to their source.
    func trimLaneClip(
        _ target: TimelineSelection,
        edge: TrimEdge,
        anchorStart: TimeInterval,
        anchorLength: TimeInterval,
        by delta: TimeInterval
    ) {
        beginGestureIfNeeded()
        switch target {
        case .overlay(let id):
            guard let index = overlayClips.firstIndex(where: { $0.id == id }) else { return }
            let limit = overlayClips[index].kind == .pip ? overlayClips[index].sourceDuration : nil
            let range = trimmedRange(edge: edge, start: anchorStart, length: anchorLength, delta: delta, maxLength: limit)
            overlayClips[index].start = range.start
            overlayClips[index].length = range.length
        case .effect(let id):
            guard let index = effectClips.firstIndex(where: { $0.id == id }) else { return }
            let range = trimmedRange(edge: edge, start: anchorStart, length: anchorLength, delta: delta, maxLength: nil)
            effectClips[index].start = range.start
            effectClips[index].length = range.length
        case .audio(let id):
            guard let index = audioClips.firstIndex(where: { $0.id == id }) else { return }
            let range = trimmedRange(edge: edge, start: anchorStart, length: anchorLength, delta: delta, maxLength: audioClips[index].sourceDuration)
            audioClips[index].start = range.start
            audioClips[index].length = range.length
        case .main:
            break
        }
    }

    /// Canvas drag of an overlay clip to a new normalized position.
    func dragOverlay(_ id: UUID, anchorX: Double, anchorY: Double, deltaX: Double, deltaY: Double) {
        guard let index = overlayClips.firstIndex(where: { $0.id == id }) else { return }
        beginGestureIfNeeded()
        overlayClips[index].posX = min(max(anchorX + deltaX, 0.03), 0.97)
        overlayClips[index].posY = min(max(anchorY + deltaY, 0.03), 0.97)
    }

    /// Scale/rotate an overlay from its corner grip.
    func transformOverlay(_ id: UUID, anchorScale: Double, anchorRotation: Double, scaleFactor: Double, rotationDelta: Double) {
        guard let index = overlayClips.firstIndex(where: { $0.id == id }) else { return }
        beginGestureIfNeeded()
        overlayClips[index].scale = min(max(anchorScale * scaleFactor, 0.25), 4)
        overlayClips[index].rotationDegrees = anchorRotation + rotationDelta
    }

    /// Horizontal drag of a lane clip to a new start time.
    func moveLaneClip(_ selectionCase: TimelineSelection, anchorStart: TimeInterval, by delta: TimeInterval) {
        beginGestureIfNeeded()
        let newStart = max(0, anchorStart + delta)
        switch selectionCase {
        case .overlay(let id):
            if let index = overlayClips.firstIndex(where: { $0.id == id }) {
                overlayClips[index].start = newStart
            }
        case .effect(let id):
            if let index = effectClips.firstIndex(where: { $0.id == id }) {
                effectClips[index].start = newStart
            }
        case .audio(let id):
            if let index = audioClips.firstIndex(where: { $0.id == id }) {
                audioClips[index].start = newStart
            }
        case .main:
            break
        }
    }

    private func clampPlayhead() {
        playhead = min(max(0, playhead), duration)
    }
}
