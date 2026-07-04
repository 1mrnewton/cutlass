import XCTest

@testable import CutlassMobile

/// The Phase I look surface end-to-end: catalogs from Rust, look commands
/// through the wire, and the `ui_state` fields the panels read back.
final class LookTests: XCTestCase {
    private func makeSession() throws -> CutlassSession {
        try XCTUnwrap(CutlassSession.create(), "session needs a working GPU")
    }

    /// A 2-second red solid on a sticker lane (visual, media-free).
    private func addSolid(_ session: CutlassSession) async throws -> UInt64 {
        let created = try await session.apply(.addTrack(kind: "Sticker", name: "Solids"))
        let track = try XCTUnwrap(created.editedID)
        let added = try await session.apply(
            .addGenerated(
                track: track,
                generator: .solidColor(rgba: [255, 0, 0, 255]),
                startTicks: 0, durationTicks: 60, fps: .fps30))
        return try XCTUnwrap(added.editedID)
    }

    private func clip(_ session: CutlassSession, _ id: UInt64) async throws -> UiClip {
        let state = try await session.uiState()
        for lane in state.lanes {
            if let clip = lane.clips.first(where: { $0.id == id }) { return clip }
        }
        return try XCTUnwrap(nil, "clip \(id) not in ui_state")
    }

    /// Mask / stabilize / animation demand a media video clip — the shapes the
    /// app's clip panels send, end to end into the decoded `UiClip`.
    func testMediaLookFieldsDecodeFromUiState() async throws {
        let session = try makeSession()
        let sample = try XCTUnwrap(
            Bundle.module.url(forResource: "sample", withExtension: "mp4"),
            "bundled sample.mp4")
        let result = try await session.run(.appendMain(paths: [sample.path]))
        let id = try XCTUnwrap(result.clips.first)

        try await session.apply(.setClipMask(clip: id, mask: UiMask(kind: "circle")))
        try await session.apply(.setClipStabilize(clip: id, level: "smooth"))
        try await session.apply(.setClipAnimation(clip: id, slot: "in", animationID: "fade_in"))

        let state = try await clip(session, id)
        XCTAssertEqual(state.mask?.kind, "circle")
        XCTAssertEqual(state.stabilize, "smooth")
        XCTAssertEqual(state.animationIn, "fade_in")
    }

    func testCatalogsLoadFromRust() {
        let catalogs = Catalogs.shared
        XCTAssertFalse(catalogs.masks.isEmpty)
        XCTAssertFalse(catalogs.filters.isEmpty)
        XCTAssertFalse(catalogs.animations.isEmpty)
        XCTAssertFalse(catalogs.textEffects.isEmpty)
        XCTAssertFalse(catalogs.speedPresets.isEmpty)
        XCTAssertFalse(catalogs.stabilizeLevels.isEmpty)
        XCTAssertFalse(catalogs.audioRoles.isEmpty)
        XCTAssertFalse(catalogs.effects.isEmpty)
        XCTAssertFalse(catalogs.transitions.isEmpty)

        XCTAssertTrue(catalogs.masks.contains(where: { $0.id == "circle" }))
        XCTAssertFalse(catalogs.effects[0].params.isEmpty, "effects carry param specs")

        let entrances = catalogs.animations(slot: "in", includeTextOnly: false)
        XCTAssertFalse(entrances.isEmpty)
        XCTAssertTrue(entrances.allSatisfy { $0.slot == "in" && !$0.textOnly })
    }

    func testLookCommandsRoundTripThroughUiState() async throws {
        let session = try makeSession()
        let id = try await addSolid(session)

        // Untouched clips report no look.
        var state = try await clip(session, id)
        XCTAssertNil(state.filter)
        XCTAssertTrue(state.adjustments.isNeutral)
        XCTAssertNil(state.animationIn)

        try await session.apply(.setClipFilter(clip: id, filter: UiFilter(id: "vivid")))
        try await session.apply(
            .setClipAdjustments(clip: id, adjust: UiAdjust(brightness: 0.25, temperature: -0.5)))
        try await session.apply(.setClipAnimation(clip: id, slot: "in", animationID: "fade_in"))

        state = try await clip(session, id)
        XCTAssertEqual(state.filter, UiFilter(id: "vivid", intensity: 0.8))
        XCTAssertEqual(state.adjustments.brightness, 0.25)
        XCTAssertEqual(state.adjustments.temperature, -0.5)
        XCTAssertEqual(state.animationIn, "fade_in")

        // A combo evicts the entrance (engine rule, one undo step).
        try await session.apply(.setClipAnimation(clip: id, slot: "combo", animationID: "pulse"))
        state = try await clip(session, id)
        XCTAssertNil(state.animationIn)
        XCTAssertEqual(state.animationCombo, "pulse")

        // Clearing works with nil payloads.
        try await session.apply(.setClipFilter(clip: id, filter: nil))
        try await session.apply(.setClipAdjustments(clip: id, adjust: UiAdjust()))
        state = try await clip(session, id)
        XCTAssertNil(state.filter)
        XCTAssertTrue(state.adjustments.isNeutral)

        // Unknown catalog ids are engine-rejected with a typed error.
        do {
            try await session.apply(.setClipFilter(clip: id, filter: UiFilter(id: "nope")))
            XCTFail("expected a model error")
        } catch let error as CutlassError {
            XCTAssertEqual(error.kind, "model")
        }
    }

    func testMaskRequiresMediaButTextEffectBakesOnTitles() async throws {
        let session = try makeSession()
        let solid = try await addSolid(session)

        // Masks need media-backed pixels; the solid bounces.
        do {
            try await session.apply(.setClipMask(clip: solid, mask: UiMask(kind: "circle")))
            XCTFail("expected a model error")
        } catch let error as CutlassError {
            XCTAssertEqual(error.kind, "model")
        }

        // A text effect preset bakes stroke/shadow/background onto the style.
        let result = try await session.run(.addText(text: "Title", atSeconds: 0))
        let textClip = try XCTUnwrap(result.clip)
        var style = try await clip(session, textClip).textStyle ?? TextStyle()
        XCTAssertNil(style.stroke)

        style.effectPreset = "neon"
        try await session.apply(
            .setGenerator(clip: textClip, generator: .text(content: "Title", style: style)))

        let bakedStyle = try await clip(session, textClip).textStyle
        let baked = try XCTUnwrap(bakedStyle)
        XCTAssertEqual(baked.effectPreset, "neon")
        XCTAssertNotNil(baked.stroke, "preset bakes a stroke")

        // Unknown presets bounce.
        style.effectPreset = "sparkle-9000"
        do {
            try await session.apply(
                .setGenerator(clip: textClip, generator: .text(content: "Title", style: style)))
            XCTFail("expected a model error")
        } catch let error as CutlassError {
            XCTAssertEqual(error.kind, "model")
        }
    }

    func testSpeedPresetIntentSurfacesInUiState() async throws {
        let session = try makeSession()
        // Speed presets need a media clip; use the bundled sample.
        let sample = try XCTUnwrap(
            Bundle.module.url(forResource: "sample", withExtension: "mp4"),
            "bundled sample.mp4")
        let result = try await session.run(.appendMain(paths: [sample.path]))
        let id = try XCTUnwrap(result.clips.first)

        var preset = try await clip(session, id).speedPreset
        XCTAssertNil(preset)

        try await session.run(.setSpeedPreset(clip: id, preset: "ramp_up"))
        preset = try await clip(session, id).speedPreset
        XCTAssertEqual(preset, "ramp_up")

        try await session.run(.setSpeedPreset(clip: id, preset: nil))
        preset = try await clip(session, id).speedPreset
        XCTAssertNil(preset)
    }
}
