import CutlassMobileFFI
import Foundation

// The preset vocabularies the panels render — effects, transitions, masks,
// filters, animations, text effects, speed presets, stabilize levels, audio
// roles — sourced from the same Rust catalogs the engine validates against.
// Static data: loaded once per process from `cutlass_catalogs()`.

/// An `{id, label}` catalog entry (masks, filters, transitions, …).
public struct CatalogEntry: Decodable, Equatable, Sendable, Identifiable {
    public let id: String
    public let label: String

    public init(id: String, label: String) {
        self.id = id
        self.label = label
    }
}

/// One effect with its ordered scalar parameters.
public struct EffectCatalogEntry: Decodable, Equatable, Sendable, Identifiable {
    public struct Param: Decodable, Equatable, Sendable {
        public let name: String
        public let label: String
        public let `default`: Float
        public let min: Float
        public let max: Float
    }

    public let id: String
    public let label: String
    public let params: [Param]
}

/// One animation preset. `slot` ∈ `in | out | combo`; `textOnly` presets are
/// rejected on non-text clips.
public struct AnimationCatalogEntry: Decodable, Equatable, Sendable, Identifiable {
    public let id: String
    public let label: String
    public let slot: String
    public let textOnly: Bool
}

/// Every preset vocabulary, decoded from the Rust catalogs once per process.
public struct Catalogs: Decodable, Sendable {
    public let effects: [EffectCatalogEntry]
    public let transitions: [CatalogEntry]
    public let masks: [CatalogEntry]
    public let filters: [CatalogEntry]
    public let animations: [AnimationCatalogEntry]
    public let textEffects: [CatalogEntry]
    public let speedPresets: [CatalogEntry]
    public let stabilizeLevels: [CatalogEntry]
    public let audioRoles: [CatalogEntry]

    /// The process-wide catalogs. Static Rust data — never fails after the
    /// library links; falls back to empty lists if the FFI misbehaves.
    public static let shared: Catalogs = load()

    /// Animations for one slot, text-only entries filtered unless asked for.
    public func animations(slot: String, includeTextOnly: Bool) -> [AnimationCatalogEntry] {
        animations.filter { $0.slot == slot && (includeTextOnly || !$0.textOnly) }
    }

    private static func load() -> Catalogs {
        guard let pointer = cutlass_catalogs() else { return .empty }
        defer { cutlass_string_free(pointer) }
        let json = Data(String(cString: pointer).utf8)
        return (try? WireCoding.decoder.decode(Catalogs.self, from: json)) ?? .empty
    }

    private static let empty = Catalogs(
        effects: [], transitions: [], masks: [], filters: [], animations: [],
        textEffects: [], speedPresets: [], stabilizeLevels: [], audioRoles: [])

    private init(
        effects: [EffectCatalogEntry], transitions: [CatalogEntry], masks: [CatalogEntry],
        filters: [CatalogEntry], animations: [AnimationCatalogEntry],
        textEffects: [CatalogEntry], speedPresets: [CatalogEntry],
        stabilizeLevels: [CatalogEntry], audioRoles: [CatalogEntry]
    ) {
        self.effects = effects
        self.transitions = transitions
        self.masks = masks
        self.filters = filters
        self.animations = animations
        self.textEffects = textEffects
        self.speedPresets = speedPresets
        self.stabilizeLevels = stabilizeLevels
        self.audioRoles = audioRoles
    }
}
