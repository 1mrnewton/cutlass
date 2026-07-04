import Testing

@testable import cutlass_ios_macos

/// Engine-backed coverage for every `EditorState` edit op: each test drives
/// the public API the views call, waits for the engine round trip, and
/// asserts on the re-projected view-model arrays (`ui_state` truth).
///
/// Media comes from the bundled fixtures (`demo1.mp4` 4s, `demo2.mp4` 6s,
/// `tone.m4a` 8s) — the same files the mock picker resolves to.
struct EditorStateEngineTests {
    /// Photo-flavored item (resolves to demo1.mp4, 4s) and video-flavored
    /// item (resolves to demo2.mp4, 6s).
    private var photoItem: MockMediaItem { MockData.libraryItems[0] }
    private var videoItem: MockMediaItem { MockData.libraryItems[1] }

    private func makeProject(_ items: [MockMediaItem]) async -> EditorState {
        let state = EditorState()
        state.startProject(with: items)
        await state.waitForEngine()
        return state
    }

    private func near(
        _ value: Double, _ expected: Double, tolerance: Double = 0.05
    ) -> Bool {
        abs(value - expected) <= tolerance
    }

    // MARK: Project lifecycle

    @Test func startProjectAppendsPicksToMain() async throws {
        let state = await makeProject([photoItem, videoItem])

        #expect(state.clips.count == 2)
        #expect(state.clips.allSatisfy { $0.engineID != nil })
        #expect(near(state.clips[0].length, 4))
        #expect(near(state.clips[1].length, 6))
        #expect(near(state.mainDuration, 10))
        #expect(state.lanes.count == 1)
        #expect(state.lanes[0].isMain)
        #expect(state.canUndo, "the append intent is one undo step")
        #expect(!state.canRedo)
        // Engine media backs every clip.
        #expect(state.clips[0].mediaPath?.hasSuffix("demo1.mp4") == true)
        #expect(state.clips[0].hasAudio, "fixture videos carry AAC audio")
    }

    @Test func appendMediaExtendsTheTimeline() async throws {
        let state = await makeProject([photoItem])
        state.appendMedia([videoItem])
        await state.waitForEngine()

        #expect(state.clips.count == 2)
        #expect(near(state.mainDuration, 10))
    }

    @Test func undoToEmptyAndRedoBack() async throws {
        let state = await makeProject([photoItem])
        #expect(state.clips.count == 1)

        state.undo()
        await state.waitForEngine()
        #expect(state.clips.isEmpty)
        #expect(!state.canUndo)
        #expect(state.canRedo)

        state.redo()
        await state.waitForEngine()
        #expect(state.clips.count == 1)
    }

    // MARK: Structural ops

    @Test func splitAtPlayheadMakesTwoPieces() async throws {
        let state = await makeProject([photoItem])
        state.playhead = 2

        state.splitAtPlayhead()
        await state.waitForEngine()
        #expect(state.clips.count == 2)
        #expect(near(state.clips[0].length, 2))
        #expect(near(state.clips[1].length, 2))
        #expect(near(state.mainDuration, 4), "split never changes duration")

        // One undo step restores the single clip; redo re-splits.
        state.undo()
        await state.waitForEngine()
        #expect(state.clips.count == 1)
        #expect(near(state.clips[0].length, 4))
        state.redo()
        await state.waitForEngine()
        #expect(state.clips.count == 2)
    }

    @Test func rippleTrimTrailingEdge() async throws {
        let state = await makeProject([photoItem, videoItem])
        let anchor = state.clips[0]

        state.trim(anchor.id, edge: .trailing, anchor: anchor, by: -1)
        #expect(near(state.clips[0].length, 3), "local preview trims live")
        state.endGesture()
        await state.waitForEngine()

        #expect(near(state.clips[0].length, 3), "engine confirms the trim")
        #expect(near(state.mainDuration, 9), "later clips ripple left")
    }

    @Test func rippleTrimLeadingEdgeConsumesSource() async throws {
        let state = await makeProject([photoItem])
        let anchor = state.clips[0]

        state.trim(anchor.id, edge: .leading, anchor: anchor, by: 1)
        state.endGesture()
        await state.waitForEngine()

        #expect(near(state.clips[0].length, 3))
        #expect(near(state.clips[0].trimStart, 1), "in-point advanced")
    }

    @Test func reorderMainClips() async throws {
        let state = await makeProject([photoItem, videoItem])
        let first = state.clips[0].engineID

        state.moveClip(fromIndex: 0, toIndex: 1)
        await state.waitForEngine()

        #expect(near(state.clips[0].length, 6))
        #expect(near(state.clips[1].length, 4))
        #expect(state.clips[1].engineID == first, "identity travels with the clip")
    }

    @Test func deleteSelectedRipples() async throws {
        let state = await makeProject([photoItem, videoItem])
        state.selection = .main(state.clips[0].id)

        state.deleteSelected()
        await state.waitForEngine()

        #expect(state.clips.count == 1)
        #expect(near(state.mainDuration, 6), "survivor slides to t=0")
        #expect(state.selection == nil)
    }

    @Test func duplicateSelectsTheCopy() async throws {
        let state = await makeProject([photoItem])
        state.selection = .main(state.clips[0].id)

        state.duplicateSelected()
        await state.waitForEngine()

        #expect(state.clips.count == 2)
        #expect(near(state.clips[1].length, 4))
        #expect(state.clips[0].engineID != state.clips[1].engineID)
        #expect(state.selection == .main(state.clips[1].id), "copy is selected")
    }

    @Test func replaceSwapsTheSourceKeepingTheSlot() async throws {
        let state = await makeProject([photoItem, videoItem])
        state.selection = .main(state.clips[0].id)

        state.replaceSelected(with: videoItem)
        await state.waitForEngine()

        #expect(state.clips.count == 2)
        #expect(state.clips[0].mediaPath?.hasSuffix("demo2.mp4") == true)
        #expect(near(state.clips[0].length, 4), "slot length is preserved")
        #expect(near(state.clips[0].sourceDuration, 6), "new source window")
    }

    // MARK: Lane content

    @Test func addTextAdoptsThePlaceholderIdentity() async throws {
        let state = await makeProject([photoItem])
        let id = state.addTextClip(text: "Hello")
        #expect(state.overlayClips.count == 1, "optimistic placeholder shows now")
        #expect(state.overlayClips[0].engineID == nil)

        await state.waitForEngine()
        #expect(state.overlayClips.count == 1)
        #expect(state.overlayClips[0].id == id, "engine clip kept the placeholder UUID")
        #expect(state.overlayClips[0].engineID != nil)
        #expect(state.overlayClips[0].kind == .text)
        #expect(state.overlayClips[0].text == "Hello")
        #expect(state.lanes.contains { $0.kind == .text && !$0.isMain })
        #expect(state.selection == .overlay(id), "selection survives the refresh")
    }

    @Test func addStickerEffectAndAudioLand() async throws {
        let state = await makeProject([photoItem])
        let sticker = state.addSticker(symbol: "heart.fill")
        let effect = state.addEffectClip(name: "Blur", kind: .effect)
        let audio = state.addAudio(kind: .music, title: "Tone", duration: 8)
        await state.waitForEngine()

        #expect(state.overlayClips.count == 1)
        #expect(state.overlayClips[0].id == sticker)
        #expect(state.overlayClips[0].engineID != nil)
        #expect(state.overlayClips[0].symbol == "heart.fill", "panel pick survives refresh")

        #expect(state.effectClips.count == 1)
        #expect(state.effectClips[0].id == effect)
        #expect(state.effectClips[0].engineID != nil)
        #expect(state.effectClips[0].name == "Blur")

        #expect(state.audioClips.count == 1)
        #expect(state.audioClips[0].id == audio)
        #expect(state.audioClips[0].engineID != nil)
        #expect(near(state.audioClips[0].length, 8), "tone.m4a is 8s")

        // Lane stack order: main on top, generated kinds between, audio floor.
        #expect(state.lanes.first?.isMain == true)
        #expect(state.lanes.last?.kind == .audio)
    }

    @Test func addPipGetsTheDropPose() async throws {
        let state = await makeProject([videoItem])
        state.playhead = 1
        let id = state.addPip(from: photoItem)
        await state.waitForEngine()

        #expect(state.overlayClips.count == 1)
        let pip = try #require(state.overlayClips.first { $0.id == id })
        #expect(pip.engineID != nil)
        #expect(pip.kind == .pip)
        #expect(near(pip.start, 1))
        #expect(near(pip.length, 4), "pip spans its media length")
        #expect(near(pip.scale, 0.5), "CapCut drop pose")
        #expect(near(pip.posY, 0.32))
        // The pip lane sits above the main row.
        let pipLaneRow = state.lanes.firstIndex { $0.id == pip.laneID }
        #expect(pipLaneRow != nil && pipLaneRow! < state.mainLaneRow)
    }

    @Test func laneTrimAndMoveCommitOnRelease() async throws {
        let state = await makeProject([videoItem])
        let id = state.addAudio(kind: .music, title: "Tone", duration: 8)
        await state.waitForEngine()
        let target = TimelineSelection.audio(id)

        state.trimLaneClip(target, edge: .trailing, anchorStart: 0, anchorLength: 8, by: -2)
        state.endGesture()
        await state.waitForEngine()
        #expect(near(state.audioClips[0].length, 6))

        state.moveLaneClip(target, anchorStart: 0, by: 1.2)
        state.endGesture()
        await state.waitForEngine()
        #expect(near(state.audioClips[0].start, 1.2))
        #expect(near(state.audioClips[0].length, 6), "move keeps the length")
    }

    // MARK: Cross-lane moves

    @Test func mainClipLiftsToALaneAndBack() async throws {
        let state = await makeProject([photoItem, videoItem])
        let lifted = state.clips[0].id

        state.moveMainClipToLane(lifted, at: 8)
        await state.waitForEngine()
        #expect(state.clips.count == 1)
        #expect(near(state.mainDuration, 6), "main closes the hole")
        #expect(state.overlayClips.count == 1)
        let pip = state.overlayClips[0]
        #expect(pip.id == lifted, "same identity across the lane change")
        #expect(pip.kind == .pip)
        #expect(near(pip.start, 8))

        state.moveLaneClipToMain(lifted, at: 100)
        await state.waitForEngine()
        #expect(state.overlayClips.isEmpty)
        #expect(state.clips.count == 2)
        #expect(state.clips[1].id == lifted)
        #expect(near(state.mainDuration, 10))
    }

    // MARK: Quick ops

    @Test func transitionsPersistAndClear() async throws {
        let state = await makeProject([photoItem, videoItem])
        let first = state.clips[0].id

        state.setTransition(after: first, MockTransition(style: "Fade", duration: 0.5))
        await state.waitForEngine()
        let applied = try #require(state.clips[0].transitionAfter)
        #expect(applied.style == "Fade")
        #expect(near(applied.duration, 0.5))

        state.setTransition(after: first, nil)
        await state.waitForEngine()
        #expect(state.clips[0].transitionAfter == nil)
    }

    @Test func keyframeToggleStampsAndRemoves() async throws {
        let state = await makeProject([photoItem])
        state.selection = .main(state.clips[0].id)
        state.playhead = 1

        state.toggleKeyframeAtPlayhead()
        await state.waitForEngine()
        #expect(state.clips[0].keyframes.contains { near($0, 1) })

        state.toggleKeyframeAtPlayhead()
        await state.waitForEngine()
        #expect(state.clips[0].keyframes.isEmpty)
    }

    @Test func extractAudioLandsALinkedAudioClip() async throws {
        let state = await makeProject([videoItem])
        state.selection = .main(state.clips[0].id)

        state.extractAudio()
        await state.waitForEngine()

        #expect(state.audioClips.count == 1)
        #expect(state.audioClips[0].engineID != nil)
        #expect(near(state.audioClips[0].start, 0), "aligned with its video")
        #expect(state.audioClips[0].kind == .extracted)
        #expect(state.clips.count == 1, "original stays on main")
    }

    @Test func freezeFrameSplitsAroundAStill() async throws {
        let state = await makeProject([photoItem])
        state.selection = .main(state.clips[0].id)
        state.playhead = 2

        state.freezeFrame()
        await state.waitForEngine()

        #expect(state.clips.count == 3, "left half + still + right half")
        #expect(near(state.clips[0].length, 2))
        #expect(state.clips[1].isFreeze, "the still wears the snowflake")
        #expect(near(state.clips[1].length, 3))
        #expect(!state.clips[1].hasAudio)
        #expect(near(state.clips[2].length, 2))
        #expect(near(state.mainDuration, 7))

        state.undo()
        await state.waitForEngine()
        #expect(state.clips.count == 1, "the freeze undoes as one step")
        #expect(near(state.mainDuration, 4))
    }

    @Test func reverseSelectedRoundTrips() async throws {
        let state = await makeProject([photoItem])
        state.selection = .main(state.clips[0].id)

        state.reverseSelected()
        await state.waitForEngine()
        #expect(state.clips[0].isReversed)

        state.reverseSelected()
        await state.waitForEngine()
        #expect(!state.clips[0].isReversed)
    }

    // MARK: Panel sessions

    @Test func panelSpeedCommitRetimesTheClip() async throws {
        let state = await makeProject([videoItem])
        state.selection = .main(state.clips[0].id)

        state.beginPanelSession()
        state.setSelectedSpeed(2)
        #expect(near(state.clips[0].length, 3), "local preview rescales")
        state.commitPanelSession()
        await state.waitForEngine()

        #expect(near(state.clips[0].speed, 2))
        #expect(near(state.clips[0].length, 3), "engine confirms 6s content at 2x")
    }

    @Test func panelVolumeCommitAndCancel() async throws {
        let state = await makeProject([photoItem])
        state.selection = .main(state.clips[0].id)

        // Cancel: local change reverts, engine never sees it.
        state.beginPanelSession()
        state.updateSelectedClip { $0.volume = 0.2 }
        state.cancelPanelSession()
        await state.waitForEngine()
        #expect(near(state.clips[0].volume, 1))

        // Commit: the diff lands as one intent.
        state.beginPanelSession()
        state.updateSelectedClip { $0.volume = 0.4 }
        state.commitPanelSession()
        await state.waitForEngine()
        #expect(near(state.clips[0].volume, 0.4))

        // The volume edit is one undo step.
        state.undo()
        await state.waitForEngine()
        #expect(near(state.clips[0].volume, 1))
    }

    @Test func panelCanvasAspectCommits() async throws {
        let state = await makeProject([photoItem])

        state.beginPanelSession()
        state.aspect = .vertical
        state.commitPanelSession()
        await state.waitForEngine()

        #expect(state.aspect == .vertical, "engine round-trips 9:16")
        state.undo()
        await state.waitForEngine()
        #expect(state.aspect == .original)
    }

    @Test func panelOverlayTransformCommits() async throws {
        let state = await makeProject([photoItem])
        let id = state.addTextClip(text: "Move me")
        await state.waitForEngine()
        state.selection = .overlay(id)

        state.beginPanelSession()
        state.updateSelectedOverlay {
            $0.posX = 0.25
            $0.posY = 0.75
            $0.scale = 1.5
            $0.opacity = 0.5
        }
        state.commitPanelSession()
        await state.waitForEngine()

        let overlay = try #require(state.overlayClips.first { $0.id == id })
        #expect(near(overlay.posX, 0.25))
        #expect(near(overlay.posY, 0.75))
        #expect(near(overlay.scale, 1.5))
        #expect(near(overlay.opacity, 0.5))
    }

    // MARK: Gesture transforms

    @Test func overlayDragCommitsTheFinalPose() async throws {
        let state = await makeProject([photoItem])
        let id = state.addTextClip(text: "Drag")
        await state.waitForEngine()

        state.dragOverlay(id, anchorX: 0.5, anchorY: 0.5, deltaX: 0.2, deltaY: -0.1)
        state.endGesture()
        await state.waitForEngine()

        let overlay = try #require(state.overlayClips.first { $0.id == id })
        #expect(near(overlay.posX, 0.7))
        #expect(near(overlay.posY, 0.4))
    }
}
