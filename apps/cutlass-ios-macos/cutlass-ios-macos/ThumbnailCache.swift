import CoreGraphics
import CutlassMobile
import Foundation

/// Process-wide cache of engine-rendered thumbnails: filmstrip tiles and
/// picker cells, keyed by (file, source slot, pixel height).
///
/// Each file gets its own `Thumbnailer` — a private decoder + GPU queue in
/// Rust — so thumbnail work never contends with the editing session. Decoding
/// happens on the thumbnailer actors; this class only coordinates, so the
/// main actor stays free for scrolling. `cached(_:)` is the synchronous fast
/// path views draw from; `thumbnail(_:)` fetches on miss.
@MainActor
final class ThumbnailCache {
    static let shared = ThumbnailCache()

    struct Slot: Hashable {
        var path: String
        /// Half-second bucket of source time (stills always use 0).
        var bucket: Int
        /// Requested pixel height (thumbs are height-bound).
        var height: Int

        init(path: String, seconds: Double, height: Int) {
            self.path = path
            self.bucket = Int((seconds * 2).rounded())
            self.height = height
        }
    }

    private var images: [Slot: CGImage] = [:]
    /// Insertion order for crude FIFO eviction.
    private var imageOrder: [Slot] = []
    private var inflight: [Slot: Task<CGImage?, Never>] = [:]
    private var thumbnailers: [String: Thumbnailer] = [:]
    private var thumbnailerOrder: [String] = []
    private var durations: [String: Double] = [:]

    /// ~600 thumbs ≈ 30 MB at 128 px — enough for a long timeline plus the
    /// picker grid without unbounded growth.
    private let imageCap = 600
    /// Open decoder pipelines (GPU queue + decode cache each).
    private let thumbnailerCap = 6

    /// Synchronous cache probe; views draw hits without a task hop.
    func cached(path: String, seconds: Double, height: Int) -> CGImage? {
        images[Slot(path: path, seconds: seconds, height: height)]
    }

    /// The thumbnail for a source slot, fetching (or joining the in-flight
    /// fetch) on miss. Returns nil when the file can't be decoded.
    func thumbnail(path: String, seconds: Double, height: Int) async -> CGImage? {
        let slot = Slot(path: path, seconds: seconds, height: height)
        if let hit = images[slot] { return hit }
        if let task = inflight[slot] { return await task.value }

        guard let thumbnailer = await opened(path) else { return nil }
        let task = Task {
            // Height-bound; the width bound just guards degenerate aspect
            // ratios (fit never upscales).
            await thumbnailer.thumbnail(
                atSeconds: seconds, maxWidth: height * 4, maxHeight: height)
        }
        inflight[slot] = task
        let image = await task.value
        inflight[slot] = nil
        if let image {
            store(image, at: slot)
        }
        return image
    }

    /// Media length in seconds (probed by the thumbnailer on first ask);
    /// nil for files the engine can't open.
    func duration(path: String) async -> Double? {
        if let known = durations[path] { return known }
        guard let thumbnailer = await opened(path) else { return nil }
        let duration = await thumbnailer.durationSeconds()
        durations[path] = duration
        return duration
    }

    private func store(_ image: CGImage, at slot: Slot) {
        if images[slot] == nil {
            imageOrder.append(slot)
        }
        images[slot] = image
        while imageOrder.count > imageCap {
            images[imageOrder.removeFirst()] = nil
        }
    }

    /// The thumbnailer for `path`, opening it (off the main actor) on first
    /// use and closing the oldest pipeline over the cap.
    private func opened(_ path: String) async -> Thumbnailer? {
        if let open = thumbnailers[path] { return open }
        let opened = await Task.detached(priority: .utility) {
            Thumbnailer.open(path: path)
        }.value
        guard let opened else { return nil }
        // A concurrent open may have landed while we awaited; keep the first.
        if let raced = thumbnailers[path] { return raced }
        thumbnailers[path] = opened
        thumbnailerOrder.append(path)
        while thumbnailerOrder.count > thumbnailerCap {
            thumbnailers[thumbnailerOrder.removeFirst()] = nil
        }
        return opened
    }
}
