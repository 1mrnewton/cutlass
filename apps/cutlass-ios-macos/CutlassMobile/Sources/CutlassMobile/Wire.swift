import Foundation

// The JSON wire protocol shared with the Rust engine.
//
// Everything the session FFI speaks is JSON: commands and intents go in, a
// response envelope comes back, and `ui_state` returns the whole presentation
// tree. The Rust side is the source of truth for this shape (see
// `crates/cutlass-mobile/src/ui_state.rs` and `intents.rs`); the types here
// are deliberately thin mirrors, decoded with `.convertFromSnakeCase`.

// MARK: - JSON value

/// An arbitrary JSON tree — the escape hatch for payloads whose shape varies
/// by command (outcome values, intent results) without inventing a type per
/// variant.
public enum JSONValue: Codable, Equatable, Sendable {
    case null
    case bool(Bool)
    case int(Int64)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int64.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: JSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }

    public subscript(key: String) -> JSONValue? {
        if case .object(let fields) = self { return fields[key] }
        return nil
    }

    public var uint64Value: UInt64? {
        if case .int(let value) = self, value >= 0 { return UInt64(value) }
        if case .number(let value) = self, value >= 0 { return UInt64(value) }
        return nil
    }

    public var boolValue: Bool? {
        if case .bool(let value) = self { return value }
        return nil
    }
}

// MARK: - Shared scalars

/// A rational number as the engine writes it (`{"num": 30, "den": 1}`) —
/// frame rates, mostly.
public struct Fraction: Codable, Equatable, Sendable {
    public var num: Int32
    public var den: Int32

    public init(num: Int32, den: Int32) {
        self.num = num
        self.den = den
    }

    public static let fps30 = Fraction(num: 30, den: 1)
    public static let fps60 = Fraction(num: 60, den: 1)

    public var doubleValue: Double {
        den == 0 ? 0 : Double(num) / Double(den)
    }
}

// MARK: - Errors

/// A failure reported by the engine (or the FFI protocol itself). `kind` is a
/// stable vocabulary: `model | time | render | decode | io | import | export |
/// missing_media | unsupported | protocol | cancelled`.
public struct CutlassError: Error, Codable, Equatable, Sendable {
    public let kind: String
    public let message: String

    public init(kind: String, message: String) {
        self.kind = kind
        self.message = message
    }

    /// The session handle was gone or the response wasn't parseable — a bug,
    /// not an engine rejection.
    static func protocolError(_ message: String) -> CutlassError {
        CutlassError(kind: "protocol", message: message)
    }
}

// MARK: - Response envelope

/// Every string-returning session call answers with
/// `{"ok": <payload>, "revision": n}` or `{"err": {"kind", "message"}}`.
struct ResponseEnvelope: Decodable {
    var ok: JSONValue?
    var revision: UInt64?
    var err: CutlassError?
}

/// The result of applying one raw command: the outcome's adjacent tag plus
/// its payload (`{"type": "Edited", "value": {"type": "Created", "id": 7}}`).
public struct ApplyOutcome: Equatable, Sendable {
    /// `Edited | Imported | Saved | Opened | Loaded | Relinked | Exported |
    /// SavedTemplate | AppliedTemplate | RemovedMedia`.
    public let type: String
    /// Variant payload; for `Edited` this is the edit outcome object.
    public let value: JSONValue?
    /// Session revision after the command.
    public let revision: UInt64

    /// Convenience for `Edited` outcomes carrying a created/updated id.
    public var editedID: UInt64? {
        value?["id"]?.uint64Value
    }
}

/// What an intent handed back: ids of whatever it created or touched. Fields
/// are optional because each intent reports a different subset.
public struct IntentResult: Equatable, Sendable {
    public let clip: UInt64?
    public let track: UInt64?
    public let media: UInt64?
    /// `append_main` reports every appended clip.
    public let clips: [UInt64]
    /// `toggle_transform_keyframe`: whether a keyframe now exists at the time.
    public let keyframed: Bool?
    /// Session revision after the intent.
    public let revision: UInt64
    /// The raw payload for anything not surfaced above.
    public let raw: JSONValue

    init(payload: JSONValue, revision: UInt64) {
        self.clip = payload["clip"]?.uint64Value
        self.track = payload["track"]?.uint64Value
        self.media = payload["media"]?.uint64Value
        if case .array(let items)? = payload["clips"] {
            self.clips = items.compactMap(\.uint64Value)
        } else {
            self.clips = []
        }
        self.keyframed = payload["keyframed"]?.boolValue
        self.revision = revision
        self.raw = payload
    }
}

// MARK: - UI state

/// The whole editor presentation for one revision — what `EditorState`
/// re-renders from after every mutation.
public struct UiState: Decodable, Sendable {
    public let revision: UInt64
    public let dirty: Bool
    public let canUndo: Bool
    public let canRedo: Bool
    public let name: String
    public let fps: Fraction
    /// End of the whole timeline (all lanes), in seconds.
    public let durationSeconds: Double
    /// End of the sequential main track, in seconds.
    public let mainDurationSeconds: Double
    public let canvas: UiCanvas
    /// Lane rows in UI order, top row first.
    public let lanes: [UiLane]

    /// The magnetic main lane (nil only for projects with no video track).
    public var mainLane: UiLane? {
        lanes.first(where: \.isMain)
    }
}

public struct UiCanvas: Decodable, Sendable {
    /// Aspect preset: `auto | 16:9 | 9:16 | 1:1 | 4:5 | 21:9`.
    public let aspect: String
    /// Opaque background color `[r, g, b]`.
    public let background: [UInt8]
    /// Resolved composite size in pixels.
    public let width: UInt32
    public let height: UInt32
}

/// One lane row. `kind` ∈ `video | text | sticker | effect | filter |
/// adjustment | audio`.
public struct UiLane: Decodable, Sendable, Identifiable {
    public let id: UInt64
    public let kind: String
    public let isMain: Bool
    public let enabled: Bool
    public let muted: Bool
    public let locked: Bool
    /// Clips ordered by start time.
    public let clips: [UiClip]
}

/// One placed clip. `kind` ∈ `video | image | audio | text | sticker | shape
/// | solid | effect | filter | adjustment`.
public struct UiClip: Decodable, Sendable, Identifiable {
    public let id: UInt64
    public let kind: String
    /// Display label: media file stem, text content, or the kind name.
    public let label: String
    public let startSeconds: Double
    public let lengthSeconds: Double

    // Media-backed fields (absent for generated clips).
    public let media: UInt64?
    public let path: String?
    /// In-point of the visible source window, in seconds of source time.
    public let trimInSeconds: Double?
    /// Full length of the backing media, in seconds.
    public let sourceDurationSeconds: Double?
    public let hasAudio: Bool
    public let isImage: Bool

    public let speed: Double
    public let reversed: Bool
    public let volume: Float
    public let fadeInSeconds: Double
    public let fadeOutSeconds: Double

    /// Canvas placement in UI coordinates (0..1, 0.5 = center); nil on audio
    /// lanes.
    public let transform: UiTransform?
    private let keyframeSeconds: [Double]?
    /// Clip-relative times (seconds) of transform keyframes — the timeline's
    /// diamonds. (Absent on the wire when empty.)
    public var keyframes: [Double] { keyframeSeconds ?? [] }

    /// Text clips: content + style.
    public let text: String?
    public let textStyle: TextStyle?
    /// Solid / shape fill.
    public let rgba: [UInt8]?

    private let effects: [String]?
    /// Effect-chain catalog ids, in order. (Absent on the wire when empty.)
    public var effectIDs: [String] { effects ?? [] }
    /// Transition at this clip's right junction (main track).
    public let transitionAfter: UiTransition?
    /// Link-group id (e.g. video + its extracted audio).
    public let link: UInt64?

    // Look properties (persist-only this milestone; absent = unset).
    public let mask: UiMask?
    public let chromaKey: UiChromaKey?
    /// `recommended | smooth | max_smooth`.
    public let stabilize: String?
    public let filter: UiFilter?
    private let adjust: UiAdjust?
    /// Color grade sliders (neutral when the wire omits the object).
    public var adjustments: UiAdjust { adjust ?? UiAdjust() }
    /// Animation catalog ids per slot (a combo excludes in/out).
    public let animationIn: String?
    public let animationOut: String?
    public let animationCombo: String?
    /// `music | sfx | voiceover | extracted`.
    public let audioRole: String?
    /// Speed-preset catalog id when the clip's curve matches one exactly.
    public let speedPreset: String?
}

// MARK: - Clip looks

/// A shaped alpha mask over a clip. Fields at their defaults are elided on
/// the wire, so decoding fills them back in.
public struct UiMask: Codable, Equatable, Sendable {
    /// `linear | mirror | circle | rectangle | heart | star`.
    public var kind: String
    /// Edge softness, 0 (hard) … 1.
    public var feather: Float
    /// Keep the outside instead of the inside.
    public var invert: Bool

    public init(kind: String, feather: Float = 0, invert: Bool = false) {
        self.kind = kind
        self.feather = feather
        self.invert = invert
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        kind = try container.decode(String.self, forKey: .kind)
        feather = try container.decodeIfPresent(Float.self, forKey: .feather) ?? 0
        invert = try container.decodeIfPresent(Bool.self, forKey: .invert) ?? false
    }
}

/// Green-screen keying: pixels near `rgb` turn transparent.
public struct UiChromaKey: Codable, Equatable, Sendable {
    /// Key color, opaque `[r, g, b]`.
    public var rgb: [UInt8]
    /// Keying strength (tolerance), 0…1.
    public var strength: Float
    /// Shadow retention, 0…1.
    public var shadow: Float

    public init(rgb: [UInt8], strength: Float = 0, shadow: Float = 0) {
        self.rgb = rgb
        self.strength = strength
        self.shadow = shadow
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        rgb = try container.decode([UInt8].self, forKey: .rgb)
        strength = try container.decodeIfPresent(Float.self, forKey: .strength) ?? 0
        shadow = try container.decodeIfPresent(Float.self, forKey: .shadow) ?? 0
    }
}

/// A filter preset applied to a clip.
public struct UiFilter: Codable, Equatable, Sendable {
    /// Catalog id (`Catalogs.filters`).
    public var id: String
    /// Blend over the original, 0…1. The wire omits the default 0.8.
    public var intensity: Float

    public init(id: String, intensity: Float = 0.8) {
        self.id = id
        self.intensity = intensity
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        intensity = try container.decodeIfPresent(Float.self, forKey: .intensity) ?? 0.8
    }
}

/// Manual color grade: signed strengths in −1…1, 0 = neutral. Sliders at
/// neutral are elided on the wire.
public struct UiAdjust: Codable, Equatable, Sendable {
    public var brightness: Float
    public var contrast: Float
    public var saturation: Float
    public var exposure: Float
    public var temperature: Float

    public init(
        brightness: Float = 0, contrast: Float = 0, saturation: Float = 0,
        exposure: Float = 0, temperature: Float = 0
    ) {
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.exposure = exposure
        self.temperature = temperature
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        brightness = try container.decodeIfPresent(Float.self, forKey: .brightness) ?? 0
        contrast = try container.decodeIfPresent(Float.self, forKey: .contrast) ?? 0
        saturation = try container.decodeIfPresent(Float.self, forKey: .saturation) ?? 0
        exposure = try container.decodeIfPresent(Float.self, forKey: .exposure) ?? 0
        temperature = try container.decodeIfPresent(Float.self, forKey: .temperature) ?? 0
    }

    public var isNeutral: Bool { self == UiAdjust() }
}

public struct UiTransform: Decodable, Equatable, Sendable {
    public let posX: Float
    public let posY: Float
    public let scale: Float
    public let rotationDegrees: Float
    public let opacity: Float
}

public struct UiTransition: Decodable, Equatable, Sendable {
    public let id: String
    public let durationSeconds: Double
}

// MARK: - Text style

/// Mirror of the model's `TextStyle` wire shape — both decoded from `ui_state`
/// and encoded back through `SetGenerator` when the text panel edits a title.
public struct TextStyle: Codable, Equatable, Sendable {
    public var font: String
    public var size: Float
    public var bold: Bool
    public var italic: Bool
    public var underline: Bool
    /// `Normal | Upper | Lower | Title`.
    public var `case`: String
    /// Fill RGBA, 0-255.
    public var fill: [UInt8]
    public var letterSpacing: Float
    public var lineSpacing: Float
    /// `Left | Center | Right`.
    public var alignH: String
    /// `Top | Middle | Bottom`.
    public var alignV: String
    public var wrap: Bool
    public var stroke: TextStroke?
    public var background: TextBackground?
    public var shadow: TextShadow?
    /// Text-effect preset id (`Catalogs.textEffects`). Setting it bakes the
    /// preset's stroke/shadow/background onto the style engine-side.
    public var effectPreset: String?

    /// The engine's defaults (white 90px system font, centered, wrapping).
    public init() {
        font = ""
        size = 90
        bold = false
        italic = false
        underline = false
        `case` = "Normal"
        fill = [255, 255, 255, 255]
        letterSpacing = 0
        lineSpacing = 1.2
        alignH = "Center"
        alignV = "Middle"
        wrap = true
    }
}

public struct TextStroke: Codable, Equatable, Sendable {
    public var rgba: [UInt8]
    public var width: Float

    public init(rgba: [UInt8] = [0, 0, 0, 255], width: Float = 6) {
        self.rgba = rgba
        self.width = width
    }
}

public struct TextBackground: Codable, Equatable, Sendable {
    public var rgba: [UInt8]
    /// Corner rounding, 0 (square) … 1 (pill).
    public var radius: Float

    public init(rgba: [UInt8] = [0, 0, 0, 255], radius: Float = 0) {
        self.rgba = rgba
        self.radius = radius
    }
}

public struct TextShadow: Codable, Equatable, Sendable {
    public var rgba: [UInt8]
    /// Blur radius as a fraction of the font size, 0…1.
    public var blur: Float
    /// Offset distance in reference pixels.
    public var distance: Float

    public init(rgba: [UInt8] = [0, 0, 0, 230], blur: Float = 0.15, distance: Float = 5) {
        self.rgba = rgba
        self.blur = blur
        self.distance = distance
    }
}

// MARK: - Coders

enum WireCoding {
    /// Decoder for typed wire structs (`UiState`, …): converts the engine's
    /// snake_case keys to Swift camelCase.
    static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()

    /// Encoder for typed wire structs sent to the engine (snake_case keys).
    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return encoder
    }()

    /// Coders for raw `JSONValue` trees. Key strategies also rewrite
    /// *dictionary* keys, which would corrupt pass-through payloads (variant
    /// tags like `"SolidColor"`, already-snake intent fields) — so trees that
    /// carry their keys verbatim must use these.
    static let plainDecoder = JSONDecoder()
    static let plainEncoder = JSONEncoder()
}
