import CoreGraphics
import XCTest

@testable import CutlassMobile

final class CutlassMobileTests: XCTestCase {
    /// End-to-end: drives the Rust compositor on a real GPU device and checks we
    /// get a correctly sized image back. Validates the whole FFI/render/readback
    /// path (the same code that runs on iOS).
    func testRenderDemoProducesImage() throws {
        guard let image = CutlassMobile.renderDemo(width: 320, height: 568) else {
            XCTFail("renderDemo returned nil — GPU device/compositor unavailable")
            return
        }
        XCTAssertEqual(image.width, 320)
        XCTAssertEqual(image.height, 568)
        XCTAssertEqual(image.bitsPerPixel, 32)
    }

    func testDegenerateSizeReturnsNil() {
        XCTAssertNil(CutlassMobile.renderDemo(width: 0, height: 100))
    }

    /// Decodes the bundled `sample.mp4` through the platform decoder + FFI,
    /// validating resource bundling and the file-frame path end-to-end.
    func testRenderBundledSampleProducesImage() throws {
        guard let image = CutlassMobile.renderBundledSampleFrame(maxWidth: 320, maxHeight: 240)
        else {
            XCTFail("renderBundledSampleFrame returned nil — decode or GPU unavailable")
            return
        }
        XCTAssertLessThanOrEqual(image.width, 320)
        XCTAssertLessThanOrEqual(image.height, 240)
        XCTAssertEqual(image.bitsPerPixel, 32)
    }

    /// Opens the synthetic preview, then renders at two times and confirms the
    /// scrub actually changes the frame — the core interactive-preview contract.
    func testDemoPreviewScrubsChangesFrame() throws {
        guard let preview = CutlassPreview.demo() else {
            XCTFail("CutlassPreview.demo returned nil — GPU unavailable")
            return
        }
        XCTAssertEqual(preview.durationSeconds, 6.0, accuracy: 1e-6)

        guard let first = preview.frame(atSeconds: 0.0),
              let last = preview.frame(atSeconds: preview.durationSeconds - 0.1)
        else {
            XCTFail("preview produced no frame")
            return
        }
        XCTAssertGreaterThan(first.width, 0)
        XCTAssertEqual(first.bitsPerPixel, 32)
        XCTAssertNotEqual(
            centerPixel(first), centerPixel(last),
            "scrubbing should change the composited frame"
        )
    }

    /// Read the center pixel's RGBA bytes from a `CGImage`.
    private func centerPixel(_ image: CGImage) -> [UInt8] {
        guard let data = image.dataProvider?.data,
              let ptr = CFDataGetBytePtr(data)
        else { return [] }
        let bpr = image.bytesPerRow
        let x = image.width / 2
        let y = image.height / 2
        let idx = y * bpr + x * 4
        return [ptr[idx], ptr[idx + 1], ptr[idx + 2], ptr[idx + 3]]
    }
}
