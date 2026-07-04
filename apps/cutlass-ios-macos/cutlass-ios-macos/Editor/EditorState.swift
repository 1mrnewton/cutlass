import CutlassMobile
import SwiftUI

/// One snapshot of the timeline projection (used as the panel-session
/// baseline for cancel/diff, and by gesture bookkeeping).
nonisolated struct TimelineSnapshot: Equatable {
    var clips: [MockClip] = []
    var lanes: [MockLane] = [MockLane(kind: .video, isMain: true)]
    var overlays: [MockOverlayClip] = []
    var effects: [MockEffectClip] = []
    var audios: [MockAudioClip] = []
    var aspect: AspectRatio = .original
    var background = CanvasBackground()
}

/// What a timeline drag is carrying: a clip lifted off the main track or a
/// free-floating lane clip.
nonisolated enum DragContent: Equatable {
    case main(UUID)
    case lane(TimelineSelection)
}

/// Where a live drag would land if released right now. One resolution per
/// drag frame drives the landing ghost, the snap guide, the tooltip, and the
/// release commit, so the preview can never disagree with the drop (ported
/// from the desktop `resolve_clip_drag`).
nonisolated struct DragResolution: Equatable {
    enum Landing: Equatable {
        /// A free span on an existing lane.
        case land(laneID: UUID, start: TimeInterval)
        /// A new lane inserted at this row of the lane stack (hovering a
        /// foreign kind, a conflicting span, or outside the stack).
        case newLane(row: Int, start: TimeInterval)
        /// Magnetic insertion between main-track clips; commit shifts later
        /// clips right. The caret renders at `caretTime` in current space.
        case mainInsert(index: Int, caretTime: TimeInterval)
    }

    var landing: Landing
    /// Lane kind of the dragged content (styles the ghost / new lane).
    var kind: MockLane.Kind
    var length: TimeInterval
    /// Vertical guide line while a magnet candidate is engaged.
    var snapTime: TimeInterval?
    /// Releasing here recreates the current arrangement; commit skips it.
    var isNoop = false
}

/// Editor state riding the Rust engine.
///
/// The engine (`CutlassSession`) owns the project, all edit validation, and
/// the undo history. This class holds a *projection* of the engine's
/// `ui_state` in the same view-model arrays the views always rendered, plus
/// pure-UI state (playhead, selection, zoom).
///
/// Mutations follow two shapes:
/// - **Taps** (split, delete, add text, …) send an intent to the engine and
///   re-project from `ui_state` when it answers (~ms).
/// - **Continuous gestures** (trims, drags) mutate the local arrays live for
///   a 60fps preview, then commit one intent on release — the engine's answer
///   snaps the projection to the truth (commit-on-release, same as desktop).
@Observable
final class EditorState {
    var clips: [MockClip] = []
    /// The lane stack, top row first. Exactly one lane `isMain` (the
    /// sequential track backed by `clips`); audio lanes always sit last.
    var lanes: [MockLane] = [MockLane(kind: .video, isMain: true)]
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

    /// Magnet toggle: when on, trims and lane moves lock onto the playhead,
    /// clip edges, and timeline bounds.
    var magnetEnabled = true
    /// Time a live gesture is currently locked onto (drives the yellow
    /// guide line + snap haptic); nil when not snapped.
    var activeSnapTime: TimeInterval?

    /// Engine history flags (from the last `ui_state` refresh).
    private(set) var canUndo = false
    private(set) var canRedo = false

    private var playbackTask: Task<Void, Never>?

    // MARK: Engine session plumbing

    /// The engine session; nil only before the first project starts.
    @ObservationIgnored private var session: CutlassSession?
    /// Bumps when the session is replaced; in-flight ops from the old
    /// session check it and drop their results.
    @ObservationIgnored private var sessionGeneration = 0
    /// FIFO chain serializing engine ops so refreshes apply in issue order.
    @ObservationIgnored private var opChain: Task<Void, Never>?
    /// Bumps per enqueued op; `waitForEngine` uses it to detect a moved tail.
    @ObservationIgnored private var opCounter = 0
    /// Engine id -> stable SwiftUI identity.
    @ObservationIgnored private var idMap = EngineIDMap()
    /// Engine clip id -> hosting lane kind (from the last refresh).
    @ObservationIgnored private var clipLaneKinds: [UInt64: MockLane.Kind] = [:]
    /// Revision of the last applied refresh; older snapshots are dropped.
    @ObservationIgnored private var appliedRevision: UInt64 = 0
    /// Refresh deferred because a live gesture owns the arrays right now.
    @ObservationIgnored private var deferredRefresh: UiState?
    /// On-disk media home for the current project (picker copies, freeze
    /// stills). Fresh per session until the project store lands (Phase H).
    @ObservationIgnored private(set) var mediaStore = ProjectMediaStore(projectID: UUID())

    init() {
        createSession()
    }

    /// Bring up a fresh engine session at the head of the op chain. GPU/
    /// renderer init happens off the main thread; ops enqueued meanwhile run
    /// after it in FIFO order.
    private func createSession() {
        sessionGeneration += 1
        let generation = sessionGeneration
        session = nil
        opCounter += 1
        opChain = Task { @MainActor [weak self] in
            let created = await Task.detached(priority: .userInitiated) {
                CutlassSession.create()
            }.value
            guard let self, self.sessionGeneration == generation else { return }
            self.session = created
        }
    }

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

    // MARK: Engine op pipeline

    /// Append an engine op to the FIFO chain. `body` runs on the main actor,
    /// awaits the session actor for the engine work, and applies the refresh.
    /// The session resolves when the op *runs* (session creation is itself
    /// the head of the chain).
    private func enqueue(_ body: @escaping @MainActor (CutlassSession) async throws -> Void) {
        let generation = sessionGeneration
        let previous = opChain
        opCounter += 1
        opChain = Task { @MainActor [weak self] in
            await previous?.value
            guard let self, self.sessionGeneration == generation,
                let session = self.session
            else { return }
            do {
                try await body(session)
            } catch {
                // Engine rejection: state is unchanged in Rust; re-project so
                // any optimistic local mutation snaps back to the truth.
                print("cutlass: engine op failed: \(error)")
                if let state = try? await session.uiState(),
                    self.sessionGeneration == generation
                {
                    self.applyRefresh(state)
                }
            }
        }
    }

    /// Run one intent, refresh, then hand the result to `onResult` (invoked
    /// after the refresh so created entities are already projected).
    private func runIntent(
        _ intent: Intent, onResult: (@MainActor (IntentResult) -> Void)? = nil
    ) {
        enqueue { [weak self] session in
            let result = try await session.run(intent)
            let state = try await session.uiState()
            guard let self else { return }
            self.noteEngineOpDuringPanel()
            self.applyRefresh(state)
            onResult?(result)
        }
    }

    /// Run a create-intent whose optimistic placeholder is already on screen:
    /// the engine clip adopts the placeholder's UUID so views (and the
    /// selection) holding it never see an identity swap. A rejected create
    /// removes its placeholder (nobody will ever confirm it).
    private func runCreateIntent(_ intent: Intent, placeholder: UUID) {
        enqueue { [weak self] session in
            let result: IntentResult
            do {
                result = try await session.run(intent)
            } catch {
                self?.removePlaceholder(placeholder)
                throw error
            }
            guard let self else { return }
            if let clip = result.clip {
                self.idMap.adopted[clip] = placeholder
            }
            let state = try await session.uiState()
            self.noteEngineOpDuringPanel()
            self.applyRefresh(state)
        }
    }

    /// Drop an optimistic clip whose engine create was rejected.
    private func removePlaceholder(_ id: UUID) {
        overlayClips.removeAll { $0.id == id && $0.engineID == nil }
        effectClips.removeAll { $0.id == id && $0.engineID == nil }
        audioClips.removeAll { $0.id == id && $0.engineID == nil }
        pruneEmptyLanes()
        reconcileSelection()
    }

    /// Apply one raw wire command + refresh.
    private func runCommand(_ command: Command) {
        enqueue { [weak self] session in
            try await session.apply(command)
            let state = try await session.uiState()
            guard let self else { return }
            self.noteEngineOpDuringPanel()
            self.applyRefresh(state)
        }
    }

    /// Re-project the arrays from a `ui_state` snapshot. Deferred while a
    /// live gesture owns the arrays; stale snapshots are dropped.
    private func applyRefresh(_ state: UiState) {
        guard state.revision >= appliedRevision else { return }
        if liveGesture != nil || dragInProgress {
            deferredRefresh = state
            return
        }
        appliedRevision = state.revision
        var projection = EngineBridge.project(state, previous: currentProjection(), ids: idMap)
        carryUnconfirmedPlaceholders(into: &projection)

        var newClips = projection.clips
        var newOverlays = projection.overlays
        var newEffects = projection.effects
        var newAudios = projection.audios
        // While a panel session edits the selected clip, its local (not yet
        // committed) values win over the engine's — commit sends the diff.
        if panelBaseline != nil, let selection {
            preserveLocal(selection, in: &newClips, &newOverlays, &newEffects, &newAudios)
        }

        clips = newClips
        lanes = projection.lanes
        overlayClips = newOverlays
        effectClips = newEffects
        audioClips = newAudios
        clipLaneKinds = projection.clipLane
        canUndo = projection.canUndo
        canRedo = projection.canRedo
        if panelBaseline == nil {
            aspect = projection.aspect
            if background.kind == .color, let color = projection.canvasBackground {
                background.color = color
            }
        }
        reconcileSelection()
        clampPlayhead()
    }

    private func currentProjection() -> EngineProjection {
        var projection = EngineProjection()
        projection.clips = clips
        projection.overlays = overlayClips
        projection.effects = effectClips
        projection.audios = audioClips
        return projection
    }

    /// Keep optimistic clips whose create-intent is still queued behind this
    /// refresh (engineID nil, absent from the engine state): they stay on
    /// screen, and their styling is still around when the create adopts them.
    /// Their optimistic lanes ride along at the kind's default row.
    private func carryUnconfirmedPlaceholders(into projection: inout EngineProjection) {
        var carriedLanes: [UUID] = []

        for overlay in overlayClips
        where overlay.engineID == nil && !projection.overlays.contains(where: { $0.id == overlay.id }) {
            projection.overlays.append(overlay)
            carriedLanes.append(overlay.laneID)
        }
        for effect in effectClips
        where effect.engineID == nil && !projection.effects.contains(where: { $0.id == effect.id }) {
            projection.effects.append(effect)
            carriedLanes.append(effect.laneID)
        }
        for audio in audioClips
        where audio.engineID == nil && !projection.audios.contains(where: { $0.id == audio.id }) {
            projection.audios.append(audio)
            carriedLanes.append(audio.laneID)
        }

        for laneID in carriedLanes where !projection.lanes.contains(where: { $0.id == laneID }) {
            guard let lane = lanes.first(where: { $0.id == laneID }) else { continue }
            let mainRow = projection.lanes.firstIndex(where: \.isMain) ?? 0
            let audioFloor =
                projection.lanes.firstIndex(where: { $0.kind == .audio }) ?? projection.lanes.count
            let row =
                switch lane.kind {
                case .video: mainRow
                case .audio: projection.lanes.count
                case .text, .sticker, .effect: audioFloor
                }
            projection.lanes.insert(lane, at: min(row, projection.lanes.count))
        }
    }

    /// Copy the selected item's local struct over the freshly projected one
    /// (keeping the projection's identity fields).
    private func preserveLocal(
        _ selection: TimelineSelection,
        in clips: inout [MockClip], _ overlays: inout [MockOverlayClip],
        _ effects: inout [MockEffectClip], _ audios: inout [MockAudioClip]
    ) {
        switch selection {
        case .main(let id):
            guard let local = self.clips.first(where: { $0.id == id }),
                let index = clips.firstIndex(where: { $0.id == id })
            else { return }
            var kept = local
            kept.engineID = clips[index].engineID
            clips[index] = kept
        case .overlay(let id):
            guard let local = overlayClips.first(where: { $0.id == id }),
                let index = overlays.firstIndex(where: { $0.id == id })
            else { return }
            var kept = local
            kept.engineID = overlays[index].engineID
            kept.laneID = overlays[index].laneID
            overlays[index] = kept
        case .effect(let id):
            guard let local = effectClips.first(where: { $0.id == id }),
                let index = effects.firstIndex(where: { $0.id == id })
            else { return }
            var kept = local
            kept.engineID = effects[index].engineID
            kept.laneID = effects[index].laneID
            effects[index] = kept
        case .audio(let id):
            guard let local = audioClips.first(where: { $0.id == id }),
                let index = audios.firstIndex(where: { $0.id == id })
            else { return }
            var kept = local
            kept.engineID = audios[index].engineID
            kept.laneID = audios[index].laneID
            audios[index] = kept
        }
    }

    private func flushDeferredRefresh() {
        guard let state = deferredRefresh else { return }
        deferredRefresh = nil
        applyRefresh(state)
    }

    /// The engine id behind a selection (nil for optimistic placeholders the
    /// engine hasn't confirmed yet).
    private func engineID(of target: TimelineSelection) -> UInt64? {
        switch target {
        case .main(let id): clips.first { $0.id == id }?.engineID
        case .overlay(let id): overlayClips.first { $0.id == id }?.engineID
        case .effect(let id): effectClips.first { $0.id == id }?.engineID
        case .audio(let id): audioClips.first { $0.id == id }?.engineID
        }
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
        // Composite by lane order: bottom rows draw first, the top row wins
        // (desktop z = row order). Callers render the array in order.
        let rowOf = Dictionary(uniqueKeysWithValues: lanes.enumerated().map { ($0.element.id, $0.offset) })
        return overlayClips
            .filter { time >= $0.start && time < $0.start + $0.length }
            .sorted { (rowOf[$0.laneID] ?? .max) > (rowOf[$1.laneID] ?? .max) }
    }

    /// Effect bars active at a timeline position.
    func effects(at time: TimeInterval) -> [MockEffectClip] {
        effectClips.filter { time >= $0.start && time < $0.start + $0.length }
    }

    // MARK: Lane stack

    /// Row index of the main track in the lane stack.
    var mainLaneRow: Int {
        lanes.firstIndex(where: \.isMain) ?? 0
    }

    /// Row index of the first audio lane (== the audio floor); `lanes.count`
    /// when there is none.
    var audioFloorRow: Int {
        lanes.firstIndex(where: { $0.kind == .audio }) ?? lanes.count
    }

    /// Time spans of every clip on a lane, excluding one clip id.
    private func spans(on laneID: UUID, excluding excluded: UUID? = nil) -> [(start: TimeInterval, end: TimeInterval)] {
        var result: [(start: TimeInterval, end: TimeInterval)] = []
        for clip in overlayClips where clip.laneID == laneID && clip.id != excluded {
            result.append((clip.start, clip.start + clip.length))
        }
        for clip in effectClips where clip.laneID == laneID && clip.id != excluded {
            result.append((clip.start, clip.start + clip.length))
        }
        for clip in audioClips where clip.laneID == laneID && clip.id != excluded {
            result.append((clip.start, clip.start + clip.length))
        }
        return result
    }

    /// Whether `[start, start+length)` overlaps no clip on the lane
    /// (touching edges are fine, mirroring the engine's overlap rule).
    func spanIsFree(on laneID: UUID, start: TimeInterval, length: TimeInterval, excluding excluded: UUID? = nil) -> Bool {
        let end = start + length
        let epsilon = 0.001
        return !spans(on: laneID, excluding: excluded).contains { span in
            start < span.end - epsilon && span.start < end - epsilon
        }
    }

    /// Audio floor invariant (desktop `enforce_audio_floor`): every audio
    /// lane sinks below every visual lane, both groups keeping their order.
    private func enforceAudioFloor() {
        let visual = lanes.filter { $0.kind != .audio }
        let audio = lanes.filter { $0.kind == .audio }
        let ordered = visual + audio
        if ordered.map(\.id) != lanes.map(\.id) {
            lanes = ordered
        }
    }

    /// Drops lanes that no longer host any clip (the main track always
    /// stays, even when empty). Only used for optimistic gesture previews;
    /// the engine refresh is the real pruner.
    private func pruneEmptyLanes() {
        lanes.removeAll { lane in
            !lane.isMain && lane.kind != .audio && spans(on: lane.id).isEmpty
        }
        // Audio lanes prune too, but only ever from the bottom block.
        lanes.removeAll { lane in
            lane.kind == .audio && spans(on: lane.id).isEmpty
        }
    }

    /// A lane of `kind` whose span at [start, start+length) is free —
    /// `preferred` first, then top-to-bottom — or a brand-new lane inserted
    /// at the kind's default row (video above main, generated kinds just
    /// above the audio floor, audio at the very bottom). Mirrors the engine's
    /// `host_lane_plan`; used for optimistic placeholders.
    @discardableResult
    func hostLane(for kind: MockLane.Kind, start: TimeInterval, length: TimeInterval, preferred: UUID? = nil) -> UUID {
        if let preferred,
           let lane = lanes.first(where: { $0.id == preferred }),
           !lane.isMain, lane.kind == kind,
           spanIsFree(on: lane.id, start: start, length: length) {
            return lane.id
        }
        if let lane = lanes.first(where: { !$0.isMain && $0.kind == kind && spanIsFree(on: $0.id, start: start, length: length) }) {
            return lane.id
        }
        let lane = MockLane(kind: kind)
        let row: Int
        switch kind {
        case .video: row = mainLaneRow
        case .audio: row = lanes.count
        case .text, .sticker, .effect: row = audioFloorRow
        }
        lanes.insert(lane, at: min(row, lanes.count))
        enforceAudioFloor()
        return lane.id
    }

    // MARK: Drag resolution (ported from the desktop resolve_clip_drag)

    /// (kind, length, origin lane id or nil for main, origin start) of the
    /// dragged content; nil when the ids are stale.
    private func dragProfile(of content: DragContent) -> (kind: MockLane.Kind, length: TimeInterval, laneID: UUID?, start: TimeInterval, clipID: UUID)? {
        switch content {
        case .main(let id):
            guard let clip = clips.first(where: { $0.id == id }) else { return nil }
            return (.video, clip.length, nil, startTime(of: id), id)
        case .lane(.overlay(let id)):
            guard let clip = overlayClips.first(where: { $0.id == id }) else { return nil }
            return (clip.laneKind, clip.length, clip.laneID, clip.start, id)
        case .lane(.effect(let id)):
            guard let clip = effectClips.first(where: { $0.id == id }) else { return nil }
            return (.effect, clip.length, clip.laneID, clip.start, id)
        case .lane(.audio(let id)):
            guard let clip = audioClips.first(where: { $0.id == id }) else { return nil }
            return (.audio, clip.length, clip.laneID, clip.start, id)
        case .lane(.main):
            return nil
        }
    }

    /// Resolves where a drag would land if released right now.
    ///
    /// - `desiredStart`: the floater's leading edge in timeline seconds.
    /// - `hoverRow`: lane-stack row under the finger; may be out of range
    ///   (above the first or below the last row).
    ///
    /// Policy (desktop `snap.rs`): the main row takes video insertions
    /// between clips; a same-kind lane with a free span lands there (magnet
    /// pulling both edges, and a snap that *causes* a conflict is dropped in
    /// favor of the free unsnapped spot); everything else — foreign kind,
    /// conflicting span, out of range — resolves to a new lane at the row,
    /// clamped so nothing but audio ever enters the audio floor.
    func resolveDrag(content: DragContent, desiredStart: TimeInterval, hoverRow: Int) -> DragResolution? {
        guard let profile = dragProfile(of: content) else { return nil }
        let desired = max(0, desiredStart)

        // Main-track magnet: a video clip over the main row inserts between
        // clips (midpoint rule); the commit opens the hole.
        if profile.kind == .video, lanes.indices.contains(hoverRow), lanes[hoverRow].isMain {
            var excludedIndex: Int?
            if case .main(let id) = content {
                excludedIndex = clips.firstIndex(where: { $0.id == id })
            }
            let insertion = mainInsertion(desired: desired, excludingIndex: excludedIndex)
            return DragResolution(
                landing: .mainInsert(index: insertion.index, caretTime: insertion.caretTime),
                kind: profile.kind,
                length: profile.length,
                snapTime: nil,
                isNoop: insertion.noop
            )
        }

        let exclusion: TimelineSelection? = {
            if case .lane(let selection) = content { return selection }
            return nil
        }()
        let candidates = laneSnapCandidates(excluding: exclusion)
        let snap = snappedDragStart(desired: desired, length: profile.length, candidates: candidates)

        if lanes.indices.contains(hoverRow) {
            let lane = lanes[hoverRow]
            if lane.kind == profile.kind, !lane.isMain {
                let sameSpot = lane.id == profile.laneID && abs(snap.start - profile.start) < 0.001
                if spanIsFree(on: lane.id, start: snap.start, length: profile.length, excluding: profile.clipID) {
                    return DragResolution(
                        landing: .land(laneID: lane.id, start: snap.start),
                        kind: profile.kind,
                        length: profile.length,
                        snapTime: snap.snapTime,
                        isNoop: sameSpot
                    )
                }
                // The snap pulled us into a conflict the raw position doesn't
                // have — prefer landing free without the magnet.
                if snap.snapTime != nil, snap.start != desired,
                   spanIsFree(on: lane.id, start: desired, length: profile.length, excluding: profile.clipID) {
                    return DragResolution(
                        landing: .land(laneID: lane.id, start: desired),
                        kind: profile.kind,
                        length: profile.length,
                        snapTime: nil,
                        isNoop: lane.id == profile.laneID && abs(desired - profile.start) < 0.001
                    )
                }
            }
        }

        // Foreign kind, conflicting span, or outside the stack: a new lane
        // inserted at the hovered row, clamped around the audio floor.
        let row: Int
        if profile.kind == .audio {
            row = min(max(hoverRow, audioFloorRow), lanes.count)
        } else {
            row = min(max(hoverRow, 0), audioFloorRow)
        }
        return DragResolution(
            landing: .newLane(row: row, start: snap.start),
            kind: profile.kind,
            length: profile.length,
            snapTime: snap.snapTime
        )
    }

    /// Marks the start of a commit-on-release drag (defers engine refreshes
    /// so they can't stomp the floater's source arrays mid-drag).
    func beginDragGesture() {
        dragInProgress = true
    }

    /// Whether a cross-lane drag currently owns the arrays.
    @ObservationIgnored private var dragInProgress = false

    /// Applies a drag resolution on release: optimistic local placement for
    /// instant feedback, then one engine intent (`move_lane` /
    /// `insert_into_main`) whose refresh snaps to the truth.
    func commitDrag(content: DragContent, resolution: DragResolution) {
        activeSnapTime = nil
        defer {
            dragInProgress = false
            flushDeferredRefresh()
        }
        guard !resolution.isNoop, let profile = dragProfile(of: content) else {
            clampPlayhead()
            return
        }
        let draggedEngineID = engineID(of: dragSelection(of: content))

        switch resolution.landing {
        case .land(let laneID, let start):
            let laneEngineID = lanes.first(where: { $0.id == laneID })?.engineID
            place(content, onLane: laneID, at: start)
            if let draggedEngineID {
                runIntent(.moveLane(clip: draggedEngineID, track: laneEngineID, startSeconds: max(0, start)))
            }
        case .newLane(let row, let start):
            let lane = MockLane(kind: resolution.kind)
            lanes.insert(lane, at: min(max(row, 0), lanes.count))
            enforceAudioFloor()
            place(content, onLane: lane.id, at: start)
            if let draggedEngineID {
                runIntent(.moveLane(clip: draggedEngineID, track: nil, startSeconds: max(0, start)))
            }
        case .mainInsert(let index, _):
            insertIntoMain(content, at: index)
            if let draggedEngineID {
                runIntent(.insertIntoMain(clip: draggedEngineID, index: index))
            }
        }

        pruneEmptyLanes()
        clampPlayhead()
        _ = profile
    }

    private func dragSelection(of content: DragContent) -> TimelineSelection {
        switch content {
        case .main(let id): .main(id)
        case .lane(let selection): selection
        }
    }

    /// Moves the dragged content onto an existing lane at `start`
    /// (optimistic placement — the engine's `move_lane` is the commit).
    /// Main-track clips leave the sequential track and become free
    /// video-lane clips, keeping their identity, look, and audio.
    private func place(_ content: DragContent, onLane laneID: UUID, at start: TimeInterval) {
        switch content {
        case .main(let id):
            guard let index = clips.firstIndex(where: { $0.id == id }) else { return }
            let clip = clips.remove(at: index)
            var lifted = MockOverlayClip(kind: .pip, laneID: laneID, start: max(0, start), length: clip.length)
            lifted.id = clip.id
            lifted.engineID = clip.engineID
            lifted.art = clip.art
            lifted.sourceDuration = clip.sourceDuration
            lifted.pipHasAudio = clip.hasAudio
            lifted.volume = clip.volume
            // Full-frame: leaving the main track must not shrink the clip.
            lifted.scale = 1
            lifted.posX = 0.5
            lifted.posY = 0.5
            overlayClips.append(lifted)
            selection = .overlay(lifted.id)
        case .lane(.overlay(let id)):
            guard let index = overlayClips.firstIndex(where: { $0.id == id }) else { return }
            overlayClips[index].laneID = laneID
            overlayClips[index].start = max(0, start)
        case .lane(.effect(let id)):
            guard let index = effectClips.firstIndex(where: { $0.id == id }) else { return }
            effectClips[index].laneID = laneID
            effectClips[index].start = max(0, start)
        case .lane(.audio(let id)):
            guard let index = audioClips.firstIndex(where: { $0.id == id }) else { return }
            audioClips[index].laneID = laneID
            audioClips[index].start = max(0, start)
        case .lane(.main):
            break
        }
    }

    /// Optimistically inserts the dragged content into the main track at
    /// `index` (already in post-removal space for reorders).
    private func insertIntoMain(_ content: DragContent, at index: Int) {
        switch content {
        case .main(let id):
            guard let from = clips.firstIndex(where: { $0.id == id }) else { return }
            let clip = clips.remove(at: from)
            clips.insert(clip, at: min(max(index, 0), clips.count))
        case .lane(.overlay(let id)):
            guard let overlayIndex = overlayClips.firstIndex(where: { $0.id == id }),
                  overlayClips[overlayIndex].kind == .pip,
                  let art = overlayClips[overlayIndex].art
            else { return }
            let overlay = overlayClips.remove(at: overlayIndex)
            var clip = MockClip(
                art: art,
                sourceDuration: overlay.sourceDuration ?? overlay.length,
                length: overlay.length,
                hasAudio: overlay.pipHasAudio
            )
            clip.id = overlay.id
            clip.engineID = overlay.engineID
            clips.insert(clip, at: min(max(index, 0), clips.count))
            selection = .main(clip.id)
        case .lane:
            break
        }
    }

    /// An insertion slot on the gapless main track for content whose left
    /// edge sits at `desired`: before the first clip whose midpoint lies
    /// right of it (crossing a clip's middle flips the caret to its other
    /// side), else after the last clip.
    struct MainInsertion {
        /// Array insertion index, in post-removal space for reorders.
        var index: Int
        /// Caret position in the track's current visual space.
        var caretTime: TimeInterval
        /// The slot is exactly where the excluded clip already sits.
        var noop: Bool
    }

    func mainInsertion(desired: TimeInterval, excludingIndex: Int? = nil) -> MainInsertion {
        var spans: [(start: TimeInterval, end: TimeInterval)] = []
        var excludedStart: TimeInterval?
        var boundary: TimeInterval = 0
        for (position, clip) in clips.enumerated() {
            let start = boundary
            boundary += clip.length
            if position == excludingIndex {
                excludedStart = start
            } else {
                spans.append((start, boundary))
            }
        }

        let index = spans.firstIndex { desired < ($0.start + $0.end) / 2 } ?? spans.count
        let noop = excludingIndex.map { $0 == index } ?? false

        let caretTime: TimeInterval
        if noop, let excludedStart {
            caretTime = excludedStart
        } else if index < spans.count {
            caretTime = spans[index].start
        } else {
            caretTime = spans.last?.end ?? 0
        }
        return MainInsertion(index: index, caretTime: caretTime, noop: noop)
    }

    /// Both edges of the dragged span magnet to the nearest candidate; the
    /// closest edge wins. A snap clamped away (below t=0) drops its guide.
    private func snappedDragStart(
        desired: TimeInterval,
        length: TimeInterval,
        candidates: [TimeInterval]
    ) -> (start: TimeInterval, snapTime: TimeInterval?) {
        guard magnetEnabled else { return (desired, nil) }
        let threshold = 8 * secondsPerPoint
        let end = desired + length
        var best: (distance: TimeInterval, start: TimeInterval, line: TimeInterval)?

        for candidate in candidates {
            let leading = abs(candidate - desired)
            if leading <= threshold, best.map({ leading < $0.distance }) ?? true {
                best = (leading, candidate, candidate)
            }
            let trailing = abs(candidate - end)
            if trailing <= threshold, best.map({ trailing < $0.distance }) ?? true {
                best = (trailing, candidate - length, candidate)
            }
        }

        guard let best else { return (desired, nil) }
        let start = max(0, best.start)
        return (start, start == best.start ? best.line : nil)
    }

    // MARK: Project lifecycle

    /// Fresh engine session; picked files (staged copies or bundled samples)
    /// are adopted into the media store and ripple-appended onto main.
    func startProject(with urls: [URL]) {
        isPlaying = false
        resetSession()
        let paths = urls.map { mediaStore.adopt($0).path }
        if !paths.isEmpty {
            runIntent(.appendMain(paths: paths))
        }
    }

    func appendMedia(_ urls: [URL]) {
        let paths = urls.map { mediaStore.adopt($0).path }
        guard !paths.isEmpty else { return }
        runIntent(.appendMain(paths: paths))
    }

    /// Queues raw intents in order (UI-test seeding; each refreshes the
    /// projection when it lands).
    func seedIntents(_ intents: [Intent]) {
        for intent in intents {
            runIntent(intent)
        }
    }

    /// Suspends until every queued engine op (and its refresh) has landed.
    /// Used by tests; the app never blocks on the chain.
    func waitForEngine() async {
        var seen = -1
        while seen != opCounter {
            seen = opCounter
            await opChain?.value
        }
    }

    private func resetSession() {
        createSession()
        mediaStore = ProjectMediaStore(projectID: UUID())
        idMap = EngineIDMap()
        clipLaneKinds = [:]
        appliedRevision = 0
        deferredRefresh = nil
        panelBaseline = nil
        panelEngineOps = 0
        liveGesture = nil
        dragInProgress = false

        clips = []
        lanes = [MockLane(kind: .video, isMain: true)]
        overlayClips = []
        effectClips = []
        audioClips = []
        aspect = .original
        background = CanvasBackground()
        canUndo = false
        canRedo = false
        playhead = 0
        selection = nil
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

    // MARK: Undo / redo (engine history)

    func undo() {
        guard canUndo else { return }
        isPlaying = false
        enqueue { [weak self] session in
            _ = await session.undo()
            let state = try await session.uiState()
            self?.applyRefresh(state)
        }
    }

    func redo() {
        guard canRedo else { return }
        isPlaying = false
        enqueue { [weak self] session in
            _ = await session.redo()
            let state = try await session.uiState()
            self?.applyRefresh(state)
        }
    }

    private func reconcileSelection() {
        switch selection {
        case .main(let id) where !clips.contains(where: { $0.id == id }),
             .overlay(let id) where !overlayClips.contains(where: { $0.id == id }),
             .effect(let id) where !effectClips.contains(where: { $0.id == id }),
             .audio(let id) where !audioClips.contains(where: { $0.id == id }):
            selection = nil
        default:
            break
        }
    }

    // MARK: Panel edit sessions

    /// Property panels mutate the local projection live so the preview
    /// reacts; the engine stays untouched until Apply, which sends the diff
    /// as intents. Cancel restores the baseline (plus undoes any structural
    /// ops — transition taps, adds — that went to the engine mid-session).
    @ObservationIgnored private var panelBaseline: TimelineSnapshot?
    /// Engine history entries recorded while the panel session was open.
    @ObservationIgnored private var panelEngineOps = 0

    private func noteEngineOpDuringPanel() {
        if panelBaseline != nil {
            panelEngineOps += 1
        }
    }

    func beginPanelSession() {
        panelBaseline = TimelineSnapshot(
            clips: clips,
            lanes: lanes,
            overlays: overlayClips,
            effects: effectClips,
            audios: audioClips,
            aspect: aspect,
            background: background
        )
        panelEngineOps = 0
    }

    func commitPanelSession() {
        guard let baseline = panelBaseline else { return }
        panelBaseline = nil
        panelEngineOps = 0
        commitPanelDiff(from: baseline)
    }

    func cancelPanelSession() {
        guard let baseline = panelBaseline else { return }
        panelBaseline = nil

        clips = baseline.clips
        lanes = baseline.lanes
        overlayClips = baseline.overlays
        effectClips = baseline.effects
        audioClips = baseline.audios
        aspect = baseline.aspect
        background = baseline.background

        // Structural ops that already reached the engine mid-session revert
        // through its history, one undo per recorded op.
        let ops = panelEngineOps
        panelEngineOps = 0
        if ops > 0 {
            enqueue { [weak self] session in
                for _ in 0..<ops {
                    _ = await session.undo()
                }
                let state = try await session.uiState()
                self?.applyRefresh(state)
            }
        }
        reconcileSelection()
        clampPlayhead()
    }

    /// Send every engine-mapped change between the panel baseline and the
    /// current local state as intents (the mock-only styling fields stay
    /// local until their models land in Phase I).
    private func commitPanelDiff(from baseline: TimelineSnapshot) {
        // Canvas.
        let backgroundColorChanged =
            background.kind == .color
            && (baseline.background.color != background.color || baseline.background.kind != .color)
        if aspect != baseline.aspect || backgroundColorChanged {
            let rgba = background.kind == .color ? background.color.engineRGB : [0, 0, 0]
            runCommand(.setCanvas(aspect: aspect.wireName, background: rgba))
        }

        // Selected main clip.
        if let clip = selectedClip, let engineID = clip.engineID {
            let before = baseline.clips.first { $0.id == clip.id }
            if clip.speed != before?.speed ?? clip.speed
                || clip.isReversed != before?.isReversed ?? clip.isReversed
            {
                runIntent(.setSpeed(clip: engineID, speed: clip.speed, reversed: clip.isReversed))
            }
            if clip.volume != before?.volume ?? clip.volume {
                runIntent(
                    .setAudio(
                        clip: engineID, volume: Float(clip.volume),
                        fadeInSeconds: clip.fadeIn, fadeOutSeconds: clip.fadeOut))
            }
            if clip.opacity != before?.opacity ?? clip.opacity {
                runIntent(
                    .setTransform(
                        clip: engineID,
                        posX: Float(clip.posX), posY: Float(clip.posY),
                        scale: Float(clip.scale),
                        rotationDegrees: Float(clip.rotationDegrees),
                        opacity: Float(clip.opacity)))
            }
        }

        // Selected overlay (text content/style, PiP volume, placement).
        if let overlay = selectedOverlay, let engineID = overlay.engineID {
            let before = baseline.overlays.first { $0.id == overlay.id }
            if overlay.kind == .text,
                overlay.text != before?.text ?? overlay.text
                    || overlay.fontName != before?.fontName ?? overlay.fontName
                    || overlay.textColor != before?.textColor ?? overlay.textColor
            {
                var style = TextStyle()
                style.font = overlay.fontName == "Default" ? "" : overlay.fontName
                style.fill = overlay.textColor.engineRGBA
                runCommand(.setGenerator(clip: engineID, generator: .text(content: overlay.text, style: style)))
            }
            if overlay.kind == .pip, overlay.volume != before?.volume ?? overlay.volume {
                runIntent(.setAudio(clip: engineID, volume: Float(overlay.volume)))
            }
            let placementChanged =
                before == nil
                || overlay.posX != before!.posX || overlay.posY != before!.posY
                || overlay.scale != before!.scale
                || overlay.rotationDegrees != before!.rotationDegrees
                || overlay.opacity != before!.opacity
            if placementChanged {
                runIntent(
                    .setTransform(
                        clip: engineID,
                        posX: Float(overlay.posX), posY: Float(overlay.posY),
                        scale: Float(overlay.scale),
                        rotationDegrees: Float(overlay.rotationDegrees),
                        opacity: Float(overlay.opacity)))
            }
        }

        // Selected audio clip (volume + fades).
        if let audio = selectedAudio, let engineID = audio.engineID {
            let before = baseline.audios.first { $0.id == audio.id }
            if audio.volume != before?.volume ?? audio.volume
                || audio.fadeIn != before?.fadeIn ?? audio.fadeIn
                || audio.fadeOut != before?.fadeOut ?? audio.fadeOut
            {
                runIntent(
                    .setAudio(
                        clip: engineID, volume: Float(audio.volume),
                        fadeInSeconds: audio.fadeIn, fadeOutSeconds: audio.fadeOut))
            }
        }
    }

    // MARK: Targeted mutation helpers (used by property panels)

    // These mutate the local projection only; the panel session commit sends
    // the engine-mapped diff.

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
    /// content plays faster or slower (live preview; committed on Apply).
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
        let start = insertionTime
        var clip = MockOverlayClip(kind: .text, laneID: hostLane(for: .text, start: start, length: 3), start: start, length: 3)
        clip.text = text
        overlayClips.append(clip)
        selection = .overlay(clip.id)
        runCreateIntent(.addText(text: text, atSeconds: start), placeholder: clip.id)
        return clip.id
    }

    @discardableResult
    func addSticker(symbol: String) -> UUID {
        let start = insertionTime
        var clip = MockOverlayClip(kind: .sticker, laneID: hostLane(for: .sticker, start: start, length: 3), start: start, length: 3)
        clip.symbol = symbol
        overlayClips.append(clip)
        selection = .overlay(clip.id)
        runCreateIntent(.addSticker(atSeconds: start), placeholder: clip.id)
        return clip.id
    }

    @discardableResult
    func addPip(from url: URL) -> UUID {
        // Placeholder length is provisional (the engine's refresh lands the
        // real one); the drop pose matches the engine's PiP placement.
        let length = MockClip.photoDefaultDuration
        let start = insertionTime
        var clip = MockOverlayClip(kind: .pip, laneID: hostLane(for: .video, start: start, length: length), start: start, length: length)
        clip.sourceDuration = length
        clip.scale = 0.5
        clip.posY = 0.32
        overlayClips.append(clip)
        selection = .overlay(clip.id)
        runCreateIntent(
            .addPip(path: mediaStore.adopt(url).path, atSeconds: start), placeholder: clip.id)
        return clip.id
    }

    @discardableResult
    func addEffectClip(name: String, kind: MockEffectClip.Kind) -> UUID {
        let start = insertionTime
        let clip = MockEffectClip(kind: kind, laneID: hostLane(for: .effect, start: start, length: 3), name: name, start: start, length: 3)
        effectClips.append(clip)
        selection = .effect(clip.id)
        let laneKind =
            switch kind {
            case .effect: "effect"
            case .filter: "filter"
            case .adjust: "adjustment"
            }
        runCreateIntent(.addEffect(kind: laneKind, atSeconds: start), placeholder: clip.id)
        return clip.id
    }

    @discardableResult
    func addAudio(kind: MockAudioClip.Kind, title: String, duration: TimeInterval) -> UUID {
        let start = insertionTime
        let clip = MockAudioClip(
            kind: kind,
            laneID: hostLane(for: .audio, start: start, length: duration),
            title: title,
            start: start,
            length: duration,
            sourceDuration: duration
        )
        audioClips.append(clip)
        selection = .audio(clip.id)
        if let url = FixtureLibrary.audio {
            runCreateIntent(.addAudio(path: url.path, atSeconds: start), placeholder: clip.id)
        }
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
        let engineID = clips[fromIndex].engineID
        let clip = clips.remove(at: fromIndex)
        clips.insert(clip, at: toIndex)
        if let engineID {
            runIntent(.insertIntoMain(clip: engineID, index: toIndex))
        }
    }

    // MARK: Cross-lane moves (drag a clip onto another lane)

    /// Pulls a clip off the main track onto a free video lane at `start`
    /// (snap-aware). It stays a full-frame video clip — same art, audio, and
    /// scale — just free-floating now.
    func moveMainClipToLane(_ id: UUID, at start: TimeInterval) {
        guard let clip = clips.first(where: { $0.id == id }) else { return }

        var dropStart = max(0, start)
        if let snapped = snapTime(near: dropStart, candidates: laneSnapCandidates(excluding: nil)) {
            dropStart = max(0, snapped)
        }
        guard let profile = dragProfile(of: .main(id)) else { return }
        let laneID = hostLane(for: .video, start: dropStart, length: profile.length)
        let laneEngineID = lanes.first(where: { $0.id == laneID })?.engineID
        place(.main(id), onLane: laneID, at: dropStart)
        pruneEmptyLanes()
        clampPlayhead()
        if let engineID = clip.engineID {
            runIntent(.moveLane(clip: engineID, track: laneEngineID, startSeconds: dropStart))
        }
    }

    /// Drops a video-lane clip into the main track at the midpoint-rule
    /// insertion slot nearest `time`.
    func moveLaneClipToMain(_ id: UUID, at time: TimeInterval) {
        guard let clip = overlayClips.first(where: { $0.id == id }), clip.kind == .pip else { return }
        let index = mainInsertion(desired: time).index
        insertIntoMain(.lane(.overlay(id)), at: index)
        pruneEmptyLanes()
        clampPlayhead()
        if let engineID = clip.engineID {
            runIntent(.insertIntoMain(clip: engineID, index: index))
        }
    }

    /// Splits whatever is selected at the playhead; with no selection, splits
    /// the main-track clip under the playhead. The engine validates the
    /// position (both pieces must stay at least one frame long).
    func splitAtPlayhead() {
        let target: TimelineSelection?
        switch selection {
        case .overlay, .effect, .audio:
            target = selection
        default:
            target = clip(at: playhead).map { .main($0.id) }
        }
        guard let target, let engineID = engineID(of: target) else { return }
        runIntent(.split(clip: engineID, seconds: playhead)) { [weak self] result in
            // Keep the left piece selected, like the mock did.
            guard let self, case .main = target else { return }
            _ = result
            self.reconcileSelection()
        }
    }

    func deleteSelected() {
        guard let target = selection, let engineID = engineID(of: target) else { return }
        selection = nil
        if case .main = target {
            runCommand(.rippleDelete(clip: engineID))
        } else {
            runCommand(.removeClip(clip: engineID))
        }
    }

    /// Inserts a copy of the selected clip right after it (ripple on main,
    /// first free slot on a lane).
    func duplicateSelected() {
        guard let target = selection, let engineID = engineID(of: target) else { return }
        runIntent(.duplicate(clip: engineID)) { [weak self] result in
            guard let self, let newID = result.clip else { return }
            let uuid = self.idMap.clip(newID)
            switch target {
            case .main: self.selection = .main(uuid)
            case .overlay: self.selection = .overlay(uuid)
            case .effect: self.selection = .effect(uuid)
            case .audio: self.selection = .audio(uuid)
            }
        }
    }

    /// Swaps the selected clip's source for a picked library item, keeping
    /// its slot on the timeline. Works for main clips and PiP overlays.
    func replaceSelected(with url: URL) {
        guard let target = selection, let engineID = engineID(of: target) else { return }
        switch target {
        case .main, .overlay:
            runIntent(.replaceMedia(clip: engineID, path: mediaStore.adopt(url).path))
        default:
            return
        }
    }

    // MARK: Quick ops

    func setTransition(after clipID: MockClip.ID, _ transition: MockTransition?) {
        guard let index = clips.firstIndex(where: { $0.id == clipID }),
            let engineID = clips[index].engineID
        else { return }
        clips[index].transitionAfter = transition
        runIntent(
            .setTransition(
                clip: engineID,
                transitionID: transition.map { TransitionMap.engineID(forStyle: $0.style) ?? "crossfade" },
                durationSeconds: transition?.duration ?? 0))
    }

    /// Stamps the same transition on every interior boundary.
    func applyTransitionToAll(_ transition: MockTransition?) {
        guard clips.count > 1 else { return }
        for index in clips.indices.dropLast() {
            clips[index].transitionAfter = transition
            guard let engineID = clips[index].engineID else { continue }
            runIntent(
                .setTransition(
                    clip: engineID,
                    transitionID: transition.map { TransitionMap.engineID(forStyle: $0.style) ?? "crossfade" },
                    durationSeconds: transition?.duration ?? 0))
        }
    }

    /// Stamps or removes a transform keyframe on the selected main clip at
    /// the playhead (the engine's diamond toggle).
    func toggleKeyframeAtPlayhead() {
        guard case .main(let id) = selection,
              let clip = clips.first(where: { $0.id == id }),
              let engineID = clip.engineID
        else { return }

        let local = playhead - startTime(of: id)
        guard local >= 0, local <= clip.length else { return }
        runIntent(.toggleTransformKeyframe(clip: engineID, seconds: playhead))
    }

    func reverseSelected() {
        guard case .main = selection, let clip = selectedClip, let engineID = clip.engineID
        else { return }
        updateSelectedClip { $0.isReversed.toggle() }
        runIntent(.setSpeed(clip: engineID, speed: clip.speed, reversed: !clip.isReversed))
    }

    /// Inserts a 3-second still of the frame under the playhead into the
    /// selected main clip (split when mid-clip). The engine extracts the
    /// frame, writes the PNG into the media store, and ripples it in as one
    /// undo step.
    func freezeFrame() {
        guard case .main(let id) = selection,
            let clip = clips.first(where: { $0.id == id }),
            let engineID = clip.engineID, !clip.isFreeze
        else { return }
        let local = playhead - startTime(of: id)
        guard local >= -0.001, local <= clip.length + 0.001 else { return }
        runIntent(
            .freeze(
                clip: engineID, seconds: playhead,
                pngPath: mediaStore.freezeFrameURL().path))
    }

    /// CapCut "extract audio": linked audio clip on an audio lane; the
    /// original's own sound goes silent via the link.
    func extractAudio() {
        guard case .main = selection, let clip = selectedClip, clip.hasAudio,
            let engineID = clip.engineID
        else { return }
        runIntent(.extractAudio(clip: engineID))
    }

    // MARK: Snap engine

    /// Times a dragged lane edge can lock onto: timeline bounds, the
    /// playhead, main-track boundaries, and every other lane clip's edges.
    private func laneSnapCandidates(excluding target: TimelineSelection?) -> [TimeInterval] {
        var times: [TimeInterval] = [0, playhead]
        var boundary: TimeInterval = 0
        for clip in clips {
            boundary += clip.length
            times.append(boundary)
        }
        for clip in overlayClips where target != .overlay(clip.id) {
            times.append(clip.start)
            times.append(clip.start + clip.length)
        }
        for clip in effectClips where target != .effect(clip.id) {
            times.append(clip.start)
            times.append(clip.start + clip.length)
        }
        for clip in audioClips where target != .audio(clip.id) {
            times.append(clip.start)
            times.append(clip.start + clip.length)
        }
        return times
    }

    /// Candidates for a main-track trailing trim: boundaries at or after the
    /// dragged edge shift with the trim, so only earlier ones are stable.
    private func mainSnapCandidates(beforeClipAt index: Int) -> [TimeInterval] {
        var times: [TimeInterval] = [0, playhead]
        var boundary: TimeInterval = 0
        for clip in clips.prefix(index) {
            boundary += clip.length
            times.append(boundary)
        }
        for clip in overlayClips {
            times.append(clip.start)
            times.append(clip.start + clip.length)
        }
        for clip in effectClips {
            times.append(clip.start)
            times.append(clip.start + clip.length)
        }
        for clip in audioClips {
            times.append(clip.start)
            times.append(clip.start + clip.length)
        }
        return times
    }

    /// Nearest candidate within the (zoom-aware) threshold, or nil.
    private func snapTime(near time: TimeInterval, candidates: [TimeInterval]) -> TimeInterval? {
        guard magnetEnabled else { return nil }
        let threshold = 8 * secondsPerPoint
        guard let best = candidates.min(by: { abs($0 - time) < abs($1 - time) }),
              abs(best - time) <= threshold
        else { return nil }
        return best
    }

    // MARK: Trim / move gestures (main + lane clips)

    enum TrimEdge {
        case leading
        case trailing
    }

    /// The continuous gesture currently mutating the local arrays. Live
    /// updates stay local for 60fps; `endGesture` commits one intent.
    private enum LiveGesture {
        case mainTrim(id: UUID, edge: TrimEdge, anchor: MockClip)
        /// Lane trim or horizontal move: commits final start+length.
        case laneAdjust(target: TimelineSelection, anchorStart: TimeInterval, anchorLength: TimeInterval)
        /// Canvas drag / pinch of an overlay: commits the final transform.
        case overlayTransform(id: UUID)
    }

    @ObservationIgnored private var liveGesture: LiveGesture?

    /// Commit-on-release: sends the gesture's final geometry as one engine
    /// intent (one undo step), then lets deferred refreshes land.
    func endGesture() {
        let gesture = liveGesture
        liveGesture = nil
        activeSnapTime = nil
        defer { flushDeferredRefresh() }

        switch gesture {
        case nil:
            break
        case .mainTrim(let id, let edge, let anchor):
            guard let clip = clips.first(where: { $0.id == id }), let engineID = clip.engineID
            else { break }
            let delta: TimeInterval
            switch edge {
            case .leading: delta = clip.trimStart - anchor.trimStart
            case .trailing: delta = clip.length - anchor.length
            }
            guard abs(delta) > 0.0005 else { break }
            runIntent(
                .rippleTrimMain(
                    clip: engineID,
                    edge: edge == .leading ? "leading" : "trailing",
                    deltaSeconds: delta))
        case .laneAdjust(let target, let anchorStart, let anchorLength):
            guard let geometry = laneGeometry(of: target), let engineID = engineID(of: target)
            else { break }
            let moved =
                abs(geometry.start - anchorStart) > 0.0005
                || abs(geometry.length - anchorLength) > 0.0005
            guard moved else { break }
            runIntent(
                .trimLane(
                    clip: engineID,
                    startSeconds: geometry.start,
                    lengthSeconds: geometry.length))
        case .overlayTransform(let id):
            guard let overlay = overlayClips.first(where: { $0.id == id }),
                let engineID = overlay.engineID
            else { break }
            runIntent(
                .setTransform(
                    clip: engineID,
                    posX: Float(overlay.posX), posY: Float(overlay.posY),
                    scale: Float(overlay.scale),
                    rotationDegrees: Float(overlay.rotationDegrees),
                    opacity: Float(overlay.opacity)))
        }
        clampPlayhead()
    }

    private func laneGeometry(of target: TimelineSelection) -> (start: TimeInterval, length: TimeInterval)? {
        switch target {
        case .overlay(let id):
            overlayClips.first(where: { $0.id == id }).map { ($0.start, $0.length) }
        case .effect(let id):
            effectClips.first(where: { $0.id == id }).map { ($0.start, $0.length) }
        case .audio(let id):
            audioClips.first(where: { $0.id == id }).map { ($0.start, $0.length) }
        case .main:
            nil
        }
    }

    /// Applies a trim drag to a main-track clip. `anchor` is the clip as it
    /// was when the drag began, so updates compute from absolute math.
    func trim(_ id: MockClip.ID, edge: TrimEdge, anchor: MockClip, by deltaSeconds: Double) {
        guard let index = clips.firstIndex(where: { $0.id == id }) else { return }
        if liveGesture == nil {
            liveGesture = .mainTrim(id: id, edge: edge, anchor: anchor)
        }

        var clip = anchor
        switch edge {
        case .leading:
            // The clip's timeline start is pinned by the clips before it, so
            // there is no stable edge to snap; just clamp to the source.
            activeSnapTime = nil
            let delta = min(
                max(deltaSeconds, -anchor.trimStart),
                anchor.length - MockClip.minDuration
            )
            clip.trimStart = anchor.trimStart + delta
            clip.length = anchor.length - delta
        case .trailing:
            let start = startTime(of: id)
            let maxLength = anchor.sourceDuration - anchor.trimStart
            var newLength = min(
                max(anchor.length + deltaSeconds, MockClip.minDuration),
                maxLength
            )
            if let snapped = snapTime(
                near: start + newLength,
                candidates: mainSnapCandidates(beforeClipAt: index)
            ), snapped - start >= MockClip.minDuration, snapped - start <= maxLength {
                newLength = snapped - start
                activeSnapTime = snapped
            } else {
                activeSnapTime = nil
            }
            clip.length = newLength
        }
        clips[index] = clip
    }

    /// The lane and clip id behind a lane-clip selection (nil for main).
    private func laneAddress(of target: TimelineSelection) -> (laneID: UUID, clipID: UUID)? {
        switch target {
        case .overlay(let id):
            return overlayClips.first(where: { $0.id == id }).map { ($0.laneID, id) }
        case .effect(let id):
            return effectClips.first(where: { $0.id == id }).map { ($0.laneID, id) }
        case .audio(let id):
            return audioClips.first(where: { $0.id == id }).map { ($0.laneID, id) }
        case .main:
            return nil
        }
    }

    /// Shared trim math for free-floating lane clips: leading trims move the
    /// start (end pinned), trailing trims change the length. Edges clamp to
    /// the lane's neighbor clips so a trim can never create an overlap, and
    /// snap to timeline landmarks when the magnet is on — a snap the clamp
    /// rejects is dropped rather than shown lying.
    private func trimmedRange(
        target: TimelineSelection,
        edge: TrimEdge,
        start: TimeInterval,
        length: TimeInterval,
        delta: TimeInterval,
        maxLength: TimeInterval?
    ) -> (start: TimeInterval, length: TimeInterval) {
        let candidates = laneSnapCandidates(excluding: target)
        let end = start + length

        // Nearest same-lane neighbors around the anchored range.
        var previousEnd: TimeInterval = 0
        var nextStart: TimeInterval = .greatestFiniteMagnitude
        if let address = laneAddress(of: target) {
            for span in spans(on: address.laneID, excluding: address.clipID) {
                if span.end <= start + 0.001 {
                    previousEnd = max(previousEnd, span.end)
                }
                if span.start >= end - 0.001 {
                    nextStart = min(nextStart, span.start)
                }
            }
        }

        switch edge {
        case .leading:
            let minStart = max(previousEnd, max(0, maxLength.map { end - $0 } ?? 0))
            let maxStart = end - MockClip.minDuration
            var newStart = min(max(start + delta, minStart), maxStart)
            if let snapped = snapTime(near: newStart, candidates: candidates),
               snapped >= minStart, snapped <= maxStart {
                newStart = snapped
                activeSnapTime = snapped
            } else {
                activeSnapTime = nil
            }
            return (newStart, end - newStart)
        case .trailing:
            let lengthCap = min(maxLength ?? .greatestFiniteMagnitude, nextStart - start)
            var newLength = min(max(length + delta, MockClip.minDuration), lengthCap)
            if let snapped = snapTime(near: start + newLength, candidates: candidates),
               snapped - start >= MockClip.minDuration,
               snapped - start <= lengthCap {
                newLength = snapped - start
                activeSnapTime = snapped
            } else {
                activeSnapTime = nil
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
        if liveGesture == nil {
            liveGesture = .laneAdjust(target: target, anchorStart: anchorStart, anchorLength: anchorLength)
        }
        switch target {
        case .overlay(let id):
            guard let index = overlayClips.firstIndex(where: { $0.id == id }) else { return }
            let limit = overlayClips[index].kind == .pip ? overlayClips[index].sourceDuration : nil
            let range = trimmedRange(target: target, edge: edge, start: anchorStart, length: anchorLength, delta: delta, maxLength: limit)
            overlayClips[index].start = range.start
            overlayClips[index].length = range.length
        case .effect(let id):
            guard let index = effectClips.firstIndex(where: { $0.id == id }) else { return }
            let range = trimmedRange(target: target, edge: edge, start: anchorStart, length: anchorLength, delta: delta, maxLength: nil)
            effectClips[index].start = range.start
            effectClips[index].length = range.length
        case .audio(let id):
            guard let index = audioClips.firstIndex(where: { $0.id == id }) else { return }
            let range = trimmedRange(target: target, edge: edge, start: anchorStart, length: anchorLength, delta: delta, maxLength: audioClips[index].sourceDuration)
            audioClips[index].start = range.start
            audioClips[index].length = range.length
        case .main:
            break
        }
    }

    /// Canvas drag of an overlay clip to a new normalized position; gently
    /// snaps to the frame center on each axis.
    func dragOverlay(_ id: UUID, anchorX: Double, anchorY: Double, deltaX: Double, deltaY: Double) {
        guard let index = overlayClips.firstIndex(where: { $0.id == id }) else { return }
        if liveGesture == nil {
            liveGesture = .overlayTransform(id: id)
        }
        var x = min(max(anchorX + deltaX, 0.03), 0.97)
        var y = min(max(anchorY + deltaY, 0.03), 0.97)
        if magnetEnabled {
            if abs(x - 0.5) < 0.02 { x = 0.5 }
            if abs(y - 0.5) < 0.02 { y = 0.5 }
        }
        overlayClips[index].posX = x
        overlayClips[index].posY = y
    }

    /// Scale/rotate an overlay from its corner grip.
    func transformOverlay(_ id: UUID, anchorScale: Double, anchorRotation: Double, scaleFactor: Double, rotationDelta: Double) {
        guard let index = overlayClips.firstIndex(where: { $0.id == id }) else { return }
        if liveGesture == nil {
            liveGesture = .overlayTransform(id: id)
        }
        overlayClips[index].scale = min(max(anchorScale * scaleFactor, 0.25), 4)
        overlayClips[index].rotationDegrees = anchorRotation + rotationDelta
    }

    /// Horizontal drag of a lane clip to a new start time; either edge can
    /// lock onto a snap candidate.
    func moveLaneClip(_ selectionCase: TimelineSelection, anchorStart: TimeInterval, by delta: TimeInterval) {
        let length: TimeInterval
        switch selectionCase {
        case .overlay(let id):
            guard let clip = overlayClips.first(where: { $0.id == id }) else { return }
            length = clip.length
        case .effect(let id):
            guard let clip = effectClips.first(where: { $0.id == id }) else { return }
            length = clip.length
        case .audio(let id):
            guard let clip = audioClips.first(where: { $0.id == id }) else { return }
            length = clip.length
        case .main:
            return
        }
        if liveGesture == nil {
            liveGesture = .laneAdjust(target: selectionCase, anchorStart: anchorStart, anchorLength: length)
        }

        var newStart = max(0, anchorStart + delta)
        let candidates = laneSnapCandidates(excluding: selectionCase)
        if let snapped = snapTime(near: newStart, candidates: candidates) {
            newStart = snapped
            activeSnapTime = snapped
        } else if let snapped = snapTime(near: newStart + length, candidates: candidates),
                  snapped - length >= 0 {
            newStart = snapped - length
            activeSnapTime = snapped
        } else {
            activeSnapTime = nil
        }

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
