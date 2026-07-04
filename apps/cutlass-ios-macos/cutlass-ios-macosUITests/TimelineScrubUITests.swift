//
//  TimelineScrubUITests.swift
//  cutlass-ios-macosUITests
//
//  Scrub-from-anywhere regression tests: dragging horizontally on any
//  timeline row (ruler, main clips, lanes, empty bands) must move the
//  playhead. XCUITest drags use real UIKit touch delivery, so gesture
//  arbitration matches a human finger (unlike synthesized HID swipes).
//

import XCTest

final class TimelineScrubUITests: XCTestCase {

    private static let rowSpacing: CGFloat = 5
    private static let rulerHeight: CGFloat = 18

    /// editorLanes seed order (top → bottom): video (pip), main, sticker,
    /// effect, audio — matches `RootView`'s `-startScreen editorLanes`.
    private static let editorLaneHeights: [CGFloat] = [44, 66, 30, 30, 30]

    override func setUpWithError() throws {
        continueAfterFailure = true
    }

    @MainActor
    private func launchEditor(_ variant: String) -> (XCUIApplication, XCUIElement) {
        let app = XCUIApplication()
        app.launchArguments = ["-startScreen", variant]
        app.launch()
        let readout = app.staticTexts["playheadReadout"]
        XCTAssertTrue(readout.waitForExistence(timeout: 5), "playhead readout should exist")
        return (app, readout)
    }

    @MainActor
    private func drag(_ app: XCUIApplication, y: CGFloat, fromX: CGFloat = 0.75, toX: CGFloat = 0.35) {
        let window = app.windows.firstMatch
        let height = window.frame.height
        let from = window.coordinate(withNormalizedOffset: CGVector(dx: fromX, dy: y / height))
        let to = window.coordinate(withNormalizedOffset: CGVector(dx: toX, dy: y / height))
        from.press(forDuration: 0.05, thenDragTo: to, withVelocity: 400, thenHoldForDuration: 0.1)
        Thread.sleep(forTimeInterval: 1.2)
    }

    /// Vertical center of a lane row inside the timeline, derived from the
    /// ordered lane list so coordinates stay honest as rows are added.
    @MainActor
    private func rowCenterY(timeline: XCUIElement, rowIndex: Int, heights: [CGFloat]) -> CGFloat {
        var stackY: CGFloat = 0
        for index in 0..<rowIndex {
            stackY += heights[index] + Self.rowSpacing
        }
        stackY += heights[rowIndex] / 2
        return timeline.frame.minY + Self.rulerHeight + Self.rowSpacing + stackY
    }

    @MainActor
    private func windowPoint(_ app: XCUIApplication, x: CGFloat, y: CGFloat) -> XCUICoordinate {
        let window = app.windows.firstMatch
        return window.coordinate(withNormalizedOffset: CGVector(
            dx: x,
            dy: y / window.frame.height
        ))
    }

    /// Screen x for a timeline time with the playhead centered (seed uses t=0).
    @MainActor
    private func timelineTimeX(_ timeline: XCUIElement, time: TimeInterval) -> CGFloat {
        timeline.frame.midX + time * 44
    }

    /// Video-lane PiP bars (kind-preserving cross-lane targets).
    @MainActor
    private func videoLaneClips(in timeline: XCUIElement) -> XCUIElementQuery {
        timeline.descendants(matching: .any).matching(identifier: "videoLaneClip")
    }

    /// Default editor (main track only): scrubbing must work from the ruler
    /// and the main clip row.
    @MainActor
    func testScrubFromEveryTimelineRegion() throws {
        let (app, readout) = launchEditor("editor")

        let timeline = app.otherElements["timeline"]
        XCTAssertTrue(timeline.waitForExistence(timeout: 3), "timeline element should exist")

        let regions: [(name: String, y: CGFloat)] = [
            ("ruler", timeline.frame.minY + Self.rulerHeight / 2),
            ("main clip upper", rowCenterY(timeline: timeline, rowIndex: 0, heights: [66])),
            ("main clip lower", rowCenterY(timeline: timeline, rowIndex: 0, heights: [66]) + 20),
        ]

        for region in regions {
            let before = readout.label
            drag(app, y: region.y)
            XCTAssertNotEqual(
                before, readout.label,
                "scrubbing from \(region.name) (y=\(region.y)pt) should move the playhead"
            )
        }
    }

    /// Collapsed timeline: a vertical drag on the lane area must pan the
    /// lane stack (not scrub, not trigger toolbar buttons below), and the
    /// playhead must hold still.
    @MainActor
    func testVerticalLanePanWhileCollapsed() throws {
        let (app, readout) = launchEditor("editorLanes")

        let timeline = app.otherElements["timeline"]
        XCTAssertTrue(timeline.waitForExistence(timeout: 3), "timeline element should exist")

        // Collapse via the grab bar: drag it down well past minHeight.
        let handle = app.otherElements["timelineHeightHandle"]
        XCTAssertTrue(handle.exists, "height handle should exist")
        let handleStart = handle.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        handleStart.press(forDuration: 0.05, thenDragTo: handleStart.withOffset(CGVector(dx: 0, dy: 400)))
        Thread.sleep(forTimeInterval: 0.5)

        let frame = timeline.frame
        XCTAssertLessThan(frame.height, 130, "timeline should be collapsed near minHeight")

        // Vertical drag up from the main-track row: pans lanes, playhead frozen.
        let before = readout.label
        let window = app.windows.firstMatch
        let midY = frame.minY + frame.height * 0.6
        let from = window.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: midY / window.frame.height))
        from.press(forDuration: 0.05, thenDragTo: from.withOffset(CGVector(dx: 0, dy: -60)), withVelocity: 300, thenHoldForDuration: 0.1)
        Thread.sleep(forTimeInterval: 0.8)

        XCTAssertEqual(before, readout.label, "vertical lane pan should not move the playhead")
        // "Faces" only exists inside the stickers panel; the toolbar label
        // "Stickers" itself always exists, so it can't be the probe.
        XCTAssertFalse(
            app.staticTexts["Faces"].exists,
            "vertical pan on the timeline must not open panels from the toolbar below"
        )
    }

    /// Video-lane PiP clips keep a footage-style duration label (not
    /// "Overlay"); cross-lane main→video moves are covered by
    /// `testDropOnOccupiedSpanCreatesNewLane`.
    @MainActor
    func testVideoLaneClipKindPreserved() throws {
        let (app, _) = launchEditor("editorLanes")

        let timeline = app.otherElements["timeline"]
        XCTAssertTrue(timeline.waitForExistence(timeout: 3), "timeline element should exist")

        let pip = videoLaneClips(in: timeline).element(boundBy: 0)
        XCTAssertTrue(pip.waitForExistence(timeout: 4), "seed PiP should exist on the video lane")
        XCTAssertFalse(pip.label.contains("Overlay"), "video-lane clips must not read as Overlay")
    }

    /// Dropping onto an occupied span on a same-kind lane inserts a new lane
    /// at the hovered row instead of nudging or rejecting the drop.
    @MainActor
    func testDropOnOccupiedSpanCreatesNewLane() throws {
        let (app, _) = launchEditor("editorLanes")

        let timeline = app.otherElements["timeline"]
        XCTAssertTrue(timeline.waitForExistence(timeout: 3), "timeline element should exist")
        let heightBefore = timeline.frame.height

        let pip = videoLaneClips(in: timeline).element(boundBy: 0)
        XCTAssertTrue(pip.waitForExistence(timeout: 2), "seed PiP should exist for overlap targeting")

        let mainY = rowCenterY(timeline: timeline, rowIndex: 1, heights: Self.editorLaneHeights)
        let pipTarget = pip.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))

        // Drop a main clip directly onto the existing PiP bar (same span).
        windowPoint(app, x: 0.62, y: mainY).press(
            forDuration: 0.8,
            thenDragTo: pipTarget,
            withVelocity: 80,
            thenHoldForDuration: 0.5
        )
        Thread.sleep(forTimeInterval: 1.0)

        XCTAssertGreaterThan(
            timeline.frame.height, heightBefore + 30,
            "dropping onto an occupied video-lane span should insert a new lane row"
        )
    }

    /// Opening a property panel overlays the timeline instead of pushing it
    /// up — the timeline frame must stay put.
    @MainActor
    func testPanelOverlayDoesNotMoveTimeline() throws {
        let (app, _) = launchEditor("editorLanes")

        let timeline = app.otherElements["timeline"]
        XCTAssertTrue(timeline.waitForExistence(timeout: 3), "timeline element should exist")
        let frameBefore = timeline.frame

        let stickers = app.buttons.matching(
            NSPredicate(format: "label CONTAINS 'Stickers'")
        ).firstMatch
        if !stickers.waitForExistence(timeout: 2) {
            app.scrollViews.firstMatch.swipeLeft()
        }
        XCTAssertTrue(stickers.waitForExistence(timeout: 2), "Stickers toolbar item should exist")
        stickers.tap()
        XCTAssertTrue(
            app.descendants(matching: .any).matching(identifier: "editorPanel").firstMatch
                .waitForExistence(timeout: 3),
            "stickers panel should open as an overlay"
        )

        Thread.sleep(forTimeInterval: 0.25)
        let frameAfter = timeline.frame
        XCTAssertEqual(frameBefore.minY, frameAfter.minY, accuracy: 4)
        XCTAssertEqual(frameBefore.height, frameAfter.height, accuracy: 4)
    }

    /// Dragging the timeline height handle up expands past the natural
    /// content height cap.
    @MainActor
    func testTimelineHeightHandleExpandsPastNaturalCap() throws {
        let (app, _) = launchEditor("editorLanes")

        let timeline = app.otherElements["timeline"]
        XCTAssertTrue(timeline.waitForExistence(timeout: 3), "timeline element should exist")
        let naturalHeight = timeline.frame.height

        let handle = app.otherElements["timelineHeightHandle"]
        XCTAssertTrue(handle.exists, "height handle should exist")
        let start = handle.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        start.press(forDuration: 0.05, thenDragTo: start.withOffset(CGVector(dx: 0, dy: -220)))
        Thread.sleep(forTimeInterval: 0.6)

        XCTAssertGreaterThan(
            timeline.frame.height, naturalHeight + 20,
            "dragging the height handle up should grow the timeline beyond its natural height"
        )
    }

    /// Fully-populated lanes: scrubbing must work when the drag starts on
    /// lane clips themselves. Row centers come from the lane list layout.
    @MainActor
    func testScrubFromPopulatedLanes() throws {
        let (app, readout) = launchEditor("editorLanes")

        let timeline = app.otherElements["timeline"]
        XCTAssertTrue(timeline.waitForExistence(timeout: 3), "timeline element should exist")

        let rows: [(name: String, index: Int)] = [
            ("video lane clip", 0),
            ("main clip", 1),
            ("sticker lane clip", 2),
            ("effects lane clip", 3),
            ("audio lane clip", 4),
        ]

        for row in rows {
            let y = rowCenterY(timeline: timeline, rowIndex: row.index, heights: Self.editorLaneHeights)
            guard y < timeline.frame.maxY - 4 else { continue }
            let before = readout.label
            drag(app, y: y)
            XCTAssertNotEqual(
                before, readout.label,
                "scrubbing from \(row.name) (y=\(y)pt) should move the playhead"
            )
        }
    }
}
