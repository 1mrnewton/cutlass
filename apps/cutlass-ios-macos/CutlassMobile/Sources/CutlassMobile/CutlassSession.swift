import CoreGraphics
import Foundation
import CutlassMobileFFI

// MARK: - Wire value types (mirror cutlass-models serde shapes)

/// A rational number `num/den` — matches `cutlass_models::Rational`.
public struct Rational: Codable, Hashable, Sendable {
    public var num: Int32
    public var den: Int32
    public init(num: Int32, den: Int32) {
        self.num = num
        self.den = den
    }
    public var asDouble: Double { den == 0 ? 0 : Double(num) / Double(den) }
}

/// A time value in `rate` ticks — matches `cutlass_models::RationalTime`.
public struct RationalTime: Codable, Hashable, Sendable {
    public var value: Int64
    public var rate: Rational
    public init(value: Int64, rate: Rational) {
        self.value = value
        self.rate = rate
    }
}

/// A `[start, start+duration)` window — matches `cutlass_models::TimeRange`.
public struct TimeRange: Codable, Hashable, Sendable {
    public var start: RationalTime
    public var duration: RationalTime
    public init(start: RationalTime, duration: RationalTime) {
        self.start = start
        self.duration = duration
    }
}

// MARK: - Commands

/// A timeline edit. Encodes to the externally-tagged JSON the Rust
/// `EditCommand` expects, e.g. `{"SplitClip": {"clip": 3, "at": {...}}}`.
public enum EditCommand: Encodable, Sendable {
    case addClip(track: UInt64, media: UInt64, source: TimeRange, start: RationalTime)
    case splitClip(clip: UInt64, at: RationalTime)
    case trimClip(clip: UInt64, timeline: TimeRange)
    case moveClip(clip: UInt64, toTrack: UInt64, start: RationalTime)
    case removeClip(clip: UInt64)
    case rippleDelete(clip: UInt64)

    public func encode(to encoder: Encoder) throws {
        var outer = encoder.container(keyedBy: StringKey.self)
        switch self {
        case let .addClip(track, media, source, start):
            var c = outer.nestedContainer(keyedBy: StringKey.self, forKey: "AddClip")
            try c.encode(track, forKey: "track")
            try c.encode(media, forKey: "media")
            try c.encode(source, forKey: "source")
            try c.encode(start, forKey: "start")
        case let .splitClip(clip, at):
            var c = outer.nestedContainer(keyedBy: StringKey.self, forKey: "SplitClip")
            try c.encode(clip, forKey: "clip")
            try c.encode(at, forKey: "at")
        case let .trimClip(clip, timeline):
            var c = outer.nestedContainer(keyedBy: StringKey.self, forKey: "TrimClip")
            try c.encode(clip, forKey: "clip")
            try c.encode(timeline, forKey: "timeline")
        case let .moveClip(clip, toTrack, start):
            var c = outer.nestedContainer(keyedBy: StringKey.self, forKey: "MoveClip")
            try c.encode(clip, forKey: "clip")
            try c.encode(toTrack, forKey: "to_track")
            try c.encode(start, forKey: "start")
        case let .removeClip(clip):
            var c = outer.nestedContainer(keyedBy: StringKey.self, forKey: "RemoveClip")
            try c.encode(clip, forKey: "clip")
        case let .rippleDelete(clip):
            var c = outer.nestedContainer(keyedBy: StringKey.self, forKey: "RippleDelete")
            try c.encode(clip, forKey: "clip")
        }
    }
}

/// Top-level command — matches `cutlass_commands::Command`. Encodes to
/// `{"Edit": <EditCommand>}`.
public enum Command: Encodable, Sendable {
    case edit(EditCommand)

    public func encode(to encoder: Encoder) throws {
        var outer = encoder.container(keyedBy: StringKey.self)
        switch self {
        case let .edit(edit):
            try outer.encode(edit, forKey: "Edit")
        }
    }
}

/// A `CodingKey` backed by an arbitrary string, for emitting the externally
/// tagged enum shapes serde produces.
struct StringKey: CodingKey, ExpressibleByStringLiteral {
    var stringValue: String
    var intValue: Int? { nil }
    init(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { nil }
    init(stringLiteral value: String) { self.stringValue = value }
}

// MARK: - Results / read-model

/// The outcome of an `apply`/`undo`/`redo`.
public struct CommandResult: Codable, Sendable {
    public var ok: Bool
    public var error: String?
    public var createdClip: UInt64?
    public var createdTrack: UInt64?
    public var media: UInt64?
    public var revision: UInt64
    public var dirty: Bool
    public var canUndo: Bool
    public var canRedo: Bool

    enum CodingKeys: String, CodingKey {
        case ok, error, revision, dirty, media
        case createdClip = "created_clip"
        case createdTrack = "created_track"
        case canUndo = "can_undo"
        case canRedo = "can_redo"
    }
}

/// Metadata for a freshly imported media file (enough to build an `AddClip`).
public struct ImportResult: Codable, Sendable {
    public var ok: Bool
    public var error: String?
    public var media: UInt64?
    public var fps: Rational?
    public var durationTicks: Int64?
    public var width: UInt32?
    public var height: UInt32?
    public var isImage: Bool?
    public var hasAudio: Bool?

    enum CodingKeys: String, CodingKey {
        case ok, error, media, fps, width, height
        case durationTicks = "duration_ticks"
        case isImage = "is_image"
        case hasAudio = "has_audio"
    }
}

/// The curated project read-model the UI renders from — matches the Rust
/// `ProjectStateDto`.
public struct ProjectState: Codable, Sendable {
    public var name: String
    public var fps: Rational
    public var durationTicks: Int64
    public var revision: UInt64
    public var dirty: Bool
    public var canUndo: Bool
    public var canRedo: Bool
    public var canvas: Canvas
    public var media: [Media]
    public var tracks: [Track]

    enum CodingKeys: String, CodingKey {
        case name, fps, revision, dirty, canvas, media, tracks
        case durationTicks = "duration_ticks"
        case canUndo = "can_undo"
        case canRedo = "can_redo"
    }

    public struct Canvas: Codable, Sendable {
        public var aspect: String
        public var background: [UInt8]
    }

    public struct Media: Codable, Sendable, Identifiable {
        public var id: UInt64
        public var path: String
        public var fps: Rational
        public var durationTicks: Int64
        public var width: UInt32
        public var height: UInt32
        public var isImage: Bool
        public var hasAudio: Bool

        enum CodingKeys: String, CodingKey {
            case id, path, fps, width, height
            case durationTicks = "duration_ticks"
            case isImage = "is_image"
            case hasAudio = "has_audio"
        }
    }

    public struct Track: Codable, Sendable, Identifiable {
        public var id: UInt64
        public var kind: String
        public var name: String
        public var enabled: Bool
        public var muted: Bool
        public var locked: Bool
        public var clips: [Clip]
    }

    public struct Clip: Codable, Sendable, Identifiable {
        public var id: UInt64
        public var kind: String
        public var start: Int64
        public var duration: Int64
        public var mediaId: UInt64?
        public var label: String

        enum CodingKeys: String, CodingKey {
            case id, kind, start, duration, label
            case mediaId = "media_id"
        }
    }
}

// MARK: - Session

/// Typed Swift front door to the editor engine, over the JSON command/state
/// bridge. Owns the native `CutlassSession` handle.
///
/// Not thread-safe: every method maps to a `&mut` engine call. Render off the
/// main thread, but serialize calls on one instance.
public final class CutlassSession {
    private let handle: OpaquePointer

    private init(handle: OpaquePointer) {
        self.handle = handle
    }

    /// Create a new, empty project at `fps` with one video track. `cacheDir`
    /// should be a writable app directory (e.g. caches); `nil` uses a temp dir.
    public static func create(
        fpsNum: Int32 = 30,
        fpsDen: Int32 = 1,
        cacheDir: String?
    ) -> CutlassSession? {
        let handle = withOptionalBytes(cacheDir) { ptr, len in
            cutlass_session_new(fpsNum, fpsDen, ptr, len)
        }
        guard let handle else { return nil }
        return CutlassSession(handle: handle)
    }

    /// Reopen a `.cutlass` project file. `nil` on failure.
    public static func open(path: String, cacheDir: String?) -> CutlassSession? {
        let handle = withBytes(path) { pPtr, pLen in
            withOptionalBytes(cacheDir) { cPtr, cLen in
                cutlass_session_open(pPtr, pLen, cPtr, cLen)
            }
        }
        guard let handle else { return nil }
        return CutlassSession(handle: handle)
    }

    /// Save to a `.cutlass` file. Returns `true` on success.
    @discardableResult
    public func save(path: String) -> Bool {
        withBytes(path) { ptr, len in
            cutlass_session_save(handle, ptr, len)
        }
    }

    /// Apply a command, returning the result (selection ids + toolbar flags).
    @discardableResult
    public func apply(_ command: Command) -> CommandResult? {
        guard let data = try? JSONEncoder().encode(command) else { return nil }
        let ptr = data.withUnsafeBytes { raw -> UnsafeMutablePointer<CChar>? in
            let bytes = raw.bindMemory(to: UInt8.self)
            return cutlass_session_apply(handle, bytes.baseAddress, bytes.count)
        }
        return decodeAndFree(ptr)
    }

    /// Convenience: apply a timeline edit.
    @discardableResult
    public func apply(_ edit: EditCommand) -> CommandResult? {
        apply(.edit(edit))
    }

    @discardableResult
    public func undo() -> CommandResult? {
        decodeAndFree(cutlass_session_undo(handle))
    }

    @discardableResult
    public func redo() -> CommandResult? {
        decodeAndFree(cutlass_session_redo(handle))
    }

    /// Probe + register a media file, returning its id and metadata.
    public func importMedia(path: String) -> ImportResult? {
        let ptr = withBytes(path) { p, len in
            cutlass_session_import(handle, p, len)
        }
        return decodeAndFree(ptr)
    }

    /// The current curated project read-model.
    public func state() -> ProjectState? {
        decodeAndFree(cutlass_session_state_json(handle))
    }

    /// Render the preview frame at `seconds` (clamped to the timeline). Call
    /// off the main thread; do not call concurrently on the same instance.
    public func render(atSeconds seconds: Double) -> CGImage? {
        CutlassMobile.makeImage(cutlass_session_render(handle, seconds))
    }

    deinit {
        cutlass_session_close(handle)
    }

    private func decodeAndFree<T: Decodable>(_ ptr: UnsafeMutablePointer<CChar>?) -> T? {
        guard let ptr else { return nil }
        defer { cutlass_string_free(ptr) }
        let json = String(cString: ptr)
        return try? JSONDecoder().decode(T.self, from: Data(json.utf8))
    }
}

// MARK: - Helpers

/// Run `body` with the UTF-8 bytes of `s` as a `(ptr, len)` pair.
private func withBytes<R>(_ s: String, _ body: (UnsafePointer<UInt8>?, Int) -> R) -> R {
    let bytes = Array(s.utf8)
    return bytes.withUnsafeBufferPointer { body($0.baseAddress, $0.count) }
}

/// Like `withBytes`, but passes `(nil, 0)` for a `nil` string.
private func withOptionalBytes<R>(_ s: String?, _ body: (UnsafePointer<UInt8>?, Int) -> R) -> R {
    guard let s else { return body(nil, 0) }
    return withBytes(s, body)
}
