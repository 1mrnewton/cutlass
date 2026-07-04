import SwiftUI

// Mock-only data model for the UI build: no engine, no FFI, no real media.
// Every "thumbnail" is a gradient plus an SF Symbol stand-in.
//
// The target defaults to MainActor isolation; these are plain value types
// used from anywhere, so they opt out.

/// Placeholder artwork used anywhere real media would render.
nonisolated struct MockArt: Hashable {
    var top: Color
    var bottom: Color
    var symbol: String?

    var gradient: LinearGradient {
        LinearGradient(colors: [top, bottom], startPoint: .topLeading, endPoint: .bottomTrailing)
    }
}

/// A saved project card on the home screen.
nonisolated struct MockProject: Identifiable, Hashable {
    var id = UUID()
    var name: String
    var dateLabel: String
    var duration: TimeInterval
    var art: MockArt
}

/// A template card in the home screen carousels.
nonisolated struct MockTemplate: Identifiable, Hashable {
    var id = UUID()
    var caption: String?
    var art: MockArt
}

nonisolated struct MockTemplateSection: Identifiable, Hashable {
    var id = UUID()
    var title: String
    var subtitle: String
    var templates: [MockTemplate]
}

/// An item in the mock photo-library picker.
nonisolated struct MockMediaItem: Identifiable, Hashable {
    enum Kind: Hashable {
        case photo
        case video(duration: TimeInterval)
    }

    var id = UUID()
    var kind: Kind
    var art: MockArt

    var videoDuration: TimeInterval? {
        if case .video(let duration) = kind { return duration }
        return nil
    }
}

/// Mock color-grade values edited in the Adjust panel.
nonisolated struct AdjustValues: Hashable {
    var brightness: Double = 0
    var contrast: Double = 0
    var saturation: Double = 0
    var exposure: Double = 0
    var temperature: Double = 0

    var isNeutral: Bool { self == AdjustValues() }
}

/// A transition applied at the boundary after a main-track clip.
nonisolated struct MockTransition: Hashable {
    var style: String
    var duration: TimeInterval = 0.5
}

/// One row of the timeline's ordered lane stack (top to bottom). Mirrors the
/// desktop `Track`/`TrackKind` rules: a lane holds one content kind, clips on
/// a lane never overlap, audio lanes are pinned to the bottom (audio floor),
/// and every other lane stacks freely around the main track.
nonisolated struct MockLane: Identifiable, Hashable {
    enum Kind: Hashable {
        case video
        case text
        case sticker
        case effect
        case audio
    }

    var id = UUID()
    var kind: Kind
    /// The magnetic sequential track new media appends to; exactly one lane
    /// has this set and it can never be removed.
    var isMain = false
    /// Rust `TrackId` when this lane mirrors an engine track (nil only for
    /// lanes created optimistically mid-gesture, before the engine confirms).
    var engineID: UInt64?
}

/// A clip on the main (sequential) video track, plus every mock style value
/// the property panels can edit.
nonisolated struct MockClip: Identifiable, Hashable {
    /// Still images can be stretched up to this long.
    static let photoMaxDuration: TimeInterval = 30
    static let photoDefaultDuration: TimeInterval = 3
    static let minDuration: TimeInterval = 0.5

    var id = UUID()
    /// Rust `ClipId` when this clip mirrors an engine clip.
    var engineID: UInt64?
    /// Source file backing the clip (engine media path).
    var mediaPath: String?
    var art: MockArt
    /// Full length of the underlying source media.
    var sourceDuration: TimeInterval
    /// Trim in-point within the source.
    var trimStart: TimeInterval = 0
    /// Trimmed length shown on the timeline.
    var length: TimeInterval
    var hasAudio: Bool
    /// Freeze-frame segments render as stills and carry no audio.
    var isFreeze = false

    // MARK: Mock style values (design-only; wired to panels)

    /// 0...2, 1 = 100%. Zero shows a mute badge on the clip.
    var volume: Double = 1
    /// Audio fades (engine-backed; no main-clip UI yet, but kept so panel
    /// commits round-trip the engine values faithfully).
    var fadeIn: TimeInterval = 0
    var fadeOut: TimeInterval = 0
    /// 0.1...10, 1 = normal. Changing it rescales `length`.
    var speed: Double = 1
    var speedCurve: String?
    var opacity: Double = 1
    /// Canvas placement (engine transform; main clips are usually full-frame
    /// so no UI edits these — carried so opacity edits don't reset them).
    var posX: Double = 0.5
    var posY: Double = 0.5
    var scale: Double = 1
    var rotationDegrees: Double = 0
    var filterName: String?
    var filterIntensity: Double = 0.8
    var adjust = AdjustValues()
    var animationIn: String?
    var animationOut: String?
    var animationCombo: String?
    var maskName: String?
    var cropPreset: String?
    /// 0 = chroma key off.
    var chromaStrength: Double = 0
    var chromaShadow: Double = 0
    var chromaColor: Color?
    var stabilizeLevel: String?
    var isReversed = false
    /// Keyframe times local to the clip (seconds from its leading edge).
    var keyframes: [TimeInterval] = []
    var transitionAfter: MockTransition?

    init(from item: MockMediaItem) {
        art = item.art
        switch item.kind {
        case .photo:
            sourceDuration = Self.photoMaxDuration
            length = Self.photoDefaultDuration
            hasAudio = false
        case .video(let duration):
            sourceDuration = duration
            length = duration
            hasAudio = true
        }
    }

    /// Direct init (the custom `init(from:)` suppressed the memberwise one),
    /// used when a clip is rebuilt from another lane's content.
    init(art: MockArt, sourceDuration: TimeInterval, length: TimeInterval, hasAudio: Bool) {
        self.art = art
        self.sourceDuration = sourceDuration
        self.length = length
        self.hasAudio = hasAudio
    }
}

/// A clip on a floating lane below/above the main track: text, sticker, or
/// picture-in-picture. Position is a normalized canvas center.
nonisolated struct MockOverlayClip: Identifiable, Hashable {
    enum Kind: Hashable {
        case text
        case sticker
        case pip
    }

    var id = UUID()
    /// Rust `ClipId` when this clip mirrors an engine clip.
    var engineID: UInt64?
    var kind: Kind
    /// The lane (row) this clip lives on; always a lane of `laneKind`.
    var laneID = UUID()
    var start: TimeInterval
    var length: TimeInterval

    /// Which lane kind can host this clip (desktop `accepts_content`).
    var laneKind: MockLane.Kind {
        switch kind {
        case .text: return .text
        case .sticker: return .sticker
        case .pip: return .video
        }
    }

    // Text
    var text = ""
    var fontName = "Default"
    var textColor: Color = .white
    var textEffect: String?
    var animation: String?

    // Sticker
    var symbol: String?

    // Picture-in-picture
    var art: MockArt?
    var sourceDuration: TimeInterval?
    /// Whether the PiP's source media carries audio (photos don't); kept so
    /// converting to a main clip round-trips faithfully.
    var pipHasAudio = false
    var volume: Double = 1

    // Canvas placement (normalized 0...1 center within the frame)
    var posX: Double = 0.5
    var posY: Double = 0.5
    var scale: Double = 1
    var rotationDegrees: Double = 0
    var opacity: Double = 1

    var displayLabel: String {
        switch kind {
        case .text: return text.isEmpty ? "Text" : text
        case .sticker: return "Sticker"
        case .pip: return "Overlay"
        }
    }
}

/// A full-width bar on the effects lane: an effect, filter, or adjust layer
/// applied over a time range.
nonisolated struct MockEffectClip: Identifiable, Hashable {
    enum Kind: Hashable {
        case effect
        case filter
        case adjust
    }

    var id = UUID()
    /// Rust `ClipId` when this clip mirrors an engine clip.
    var engineID: UInt64?
    var kind: Kind
    /// The effect lane (row) this bar lives on.
    var laneID = UUID()
    var name: String
    var start: TimeInterval
    var length: TimeInterval
    var intensity: Double = 0.8

    var displayLabel: String {
        switch kind {
        case .effect: return name
        case .filter: return "Filter: \(name)"
        case .adjust: return "Adjust"
        }
    }
}

/// A clip on the audio lane.
nonisolated struct MockAudioClip: Identifiable, Hashable {
    enum Kind: Hashable {
        case music
        case soundFX
        case voiceover
        case extracted
    }

    var id = UUID()
    /// Rust `ClipId` when this clip mirrors an engine clip.
    var engineID: UInt64?
    var kind: Kind
    /// The audio lane (row) this clip lives on; audio lanes stay at the
    /// bottom of the stack (audio floor).
    var laneID = UUID()
    var title: String
    var start: TimeInterval
    var length: TimeInterval
    var sourceDuration: TimeInterval
    var volume: Double = 1
    var fadeIn: TimeInterval = 0
    var fadeOut: TimeInterval = 0
    var waveSeed = Int.random(in: 0..<10_000)

    var symbol: String {
        switch kind {
        case .music: return "music.note"
        case .soundFX: return "waveform"
        case .voiceover: return "mic.fill"
        case .extracted: return "arrow.turn.down.right"
        }
    }
}

/// A mock song / sound effect row in the audio browser.
nonisolated struct MockSong: Identifiable, Hashable {
    var id = UUID()
    var title: String
    var artist: String
    var duration: TimeInterval
}

/// Canvas aspect ratio presets — exactly the engine's `CanvasAspect` set
/// (`original` = the engine's `auto`, following the footage).
nonisolated enum AspectRatio: String, CaseIterable, Hashable {
    case original = "Original"
    case wide = "16:9"
    case vertical = "9:16"
    case square = "1:1"
    case portrait = "4:5"
    case cinema = "21:9"

    /// width / height. `original` follows the mock footage (9:16).
    var ratio: CGFloat {
        switch self {
        case .original, .vertical: return 9.0 / 16.0
        case .wide: return 16.0 / 9.0
        case .square: return 1
        case .portrait: return 4.0 / 5.0
        case .cinema: return 21.0 / 9.0
        }
    }

    var symbol: String {
        switch self {
        case .original: return "sparkles.rectangle.stack"
        case .wide: return "rectangle"
        case .vertical: return "rectangle.portrait"
        case .square: return "square"
        case .portrait: return "rectangle.portrait"
        case .cinema: return "pano"
        }
    }

    /// `CanvasAspect` wire name for `SetCanvas`.
    var wireName: String {
        self == .original ? "auto" : rawValue
    }

    /// The preset for a `ui_state` canvas aspect wire name.
    static func from(wireName: String) -> AspectRatio {
        allCases.first { $0.wireName == wireName } ?? .original
    }
}

/// What fills the canvas behind pillarboxed footage.
nonisolated struct CanvasBackground: Hashable {
    enum Kind: Hashable {
        case blur
        case color
    }

    var kind: Kind = .blur
    /// 0...1, used when kind == .blur.
    var blurStrength: Double = 0.55
    /// Used when kind == .color.
    var color: Color = .black
}

/// Unified timeline selection across every lane.
nonisolated enum TimelineSelection: Hashable {
    case main(UUID)
    case overlay(UUID)
    case effect(UUID)
    case audio(UUID)
}

nonisolated extension TimeInterval {
    /// "1:34"-style label used on badges and the timeline ruler.
    var timecode: String {
        let total = Int(self.rounded())
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}
