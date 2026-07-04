import CutlassMobile
import SwiftUI

// Projection of the engine's `ui_state` tree into the view-model types the
// editor views already render (`MockClip`, `MockLane`, …).
//
// The Rust side owns all editing logic and lane policy; this file only
// reshapes its presentation state and carries over the styling fields the
// model doesn't persist yet (sticker symbols, filter names, …) so panel
// picks survive refreshes until Phase I lands them in the model.

// MARK: - Engine id <-> SwiftUI identity

/// Deterministic UUIDs for engine entities, so SwiftUI identity (and the
/// selection) is stable across `ui_state` refreshes. Clip and track ids are
/// separate engine counters, hence the namespace byte.
nonisolated struct EngineIDMap {
    /// Engine clip ids that must keep a pre-existing UUID: clips created
    /// optimistically in Swift (placeholder shown during the FFI round trip)
    /// adopt the placeholder's UUID so views holding it never see an
    /// identity swap.
    var adopted: [UInt64: UUID] = [:]

    func clip(_ id: UInt64) -> UUID {
        adopted[id] ?? Self.uuid(namespace: 0x01, id)
    }

    func lane(_ id: UInt64) -> UUID {
        Self.uuid(namespace: 0x02, id)
    }

    private static func uuid(namespace: UInt8, _ id: UInt64) -> UUID {
        // "CE" prefix + namespace + big-endian id in the last 8 bytes.
        UUID(uuid: (
            0x43, 0x45, namespace, 0, 0, 0, 0, 0,
            UInt8(truncatingIfNeeded: id >> 56),
            UInt8(truncatingIfNeeded: id >> 48),
            UInt8(truncatingIfNeeded: id >> 40),
            UInt8(truncatingIfNeeded: id >> 32),
            UInt8(truncatingIfNeeded: id >> 24),
            UInt8(truncatingIfNeeded: id >> 16),
            UInt8(truncatingIfNeeded: id >> 8),
            UInt8(truncatingIfNeeded: id)
        ))
    }
}

// MARK: - Transition catalog mapping

/// UI style names <-> engine transition catalog ids. The mock panel's fancier
/// styles (Zoom, Spin, Blur) don't exist in the engine catalog yet; they map
/// onto the nearest real transition so the choice persists honestly.
nonisolated enum TransitionMap {
    private static let uiToEngine: [String: String] = [
        "Fade": "crossfade",
        "Dissolve": "dip_to_black",
        "Slide": "slide",
        "Wipe": "wipe_left",
        "Zoom": "dip_to_white",
        "Spin": "wipe_up",
        "Blur": "crossfade",
    ]

    /// Canonical UI style per engine id (several UI styles can share an
    /// engine transition; the projection shows the canonical one).
    private static let engineToUI: [String: String] = [
        "crossfade": "Fade",
        "dip_to_black": "Dissolve",
        "slide": "Slide",
        "wipe_left": "Wipe",
        "dip_to_white": "Zoom",
        "wipe_up": "Spin",
    ]

    static func engineID(forStyle style: String) -> String? {
        uiToEngine[style]
    }

    static func style(forEngineID id: String) -> String {
        engineToUI[id] ?? "Fade"
    }
}

// MARK: - Media fixtures (until the PhotosUI picker lands in Phase E)

/// Real media files bundled with the app: the picker's Samples tab, preview
/// seeds, and tests all import these through the engine.
nonisolated enum FixtureLibrary {
    /// 6-second video (AAC audio).
    static var video: URL? {
        Bundle.main.url(forResource: "demo2", withExtension: "mp4")
    }

    /// 4-second video (AAC audio).
    static var shortVideo: URL? {
        Bundle.main.url(forResource: "demo1", withExtension: "mp4")
    }

    /// Still photo (imports as a 5s image clip).
    static var photo: URL? {
        Bundle.main.url(forResource: "photo", withExtension: "png")
    }

    /// Audio-only file backing song / sound-effect / voiceover picks.
    static var audio: URL? {
        Bundle.main.url(forResource: "tone", withExtension: "m4a")
    }

    /// Default dev/preview timeline: video, video, photo, video.
    static var sampleTimeline: [URL] {
        [shortVideo, video, photo, video].compactMap(\.self)
    }

    /// Everything the picker's Samples tab offers.
    static var samples: [URL] {
        [shortVideo, video, photo, audio].compactMap(\.self)
    }
}

// MARK: - Wire -> mock projection

/// The full projection of one `ui_state` snapshot, plus id lookups the ops
/// need to translate a `TimelineSelection` back into engine ids.
nonisolated struct EngineProjection {
    var clips: [MockClip] = []
    var lanes: [MockLane] = [MockLane(kind: .video, isMain: true)]
    var overlays: [MockOverlayClip] = []
    var effects: [MockEffectClip] = []
    var audios: [MockAudioClip] = []
    var aspect: AspectRatio = .original
    var canvasBackground: Color?
    var canUndo = false
    var canRedo = false
    var revision: UInt64 = 0

    /// engine clip id -> the lane kind hosting it (drives delete semantics
    /// and selection re-anchoring).
    var clipLane: [UInt64: MockLane.Kind] = [:]
}

nonisolated enum EngineBridge {
    /// Reshape a `ui_state` tree into the view-model arrays, carrying mock
    /// styling over from the previous projection by engine id.
    static func project(
        _ state: UiState, previous: EngineProjection, ids: EngineIDMap
    ) -> EngineProjection {
        var result = EngineProjection()
        result.canUndo = state.canUndo
        result.canRedo = state.canRedo
        result.revision = state.revision
        result.aspect = AspectRatio.from(wireName: state.canvas.aspect)
        result.canvasBackground = color(state.canvas.background)

        // Old items are matched by engine id, falling back to UUID so a
        // placeholder created optimistically in Swift (engineID nil until the
        // engine confirms) still hands its styling to the adopted clip.
        let oldClips = index(previous.clips)
        let oldOverlays = index(previous.overlays)
        let oldEffects = index(previous.effects)
        let oldAudios = index(previous.audios)
        let oldClipsByUUID = Dictionary(uniqueKeysWithValues: previous.clips.map { ($0.id, $0) })
        let oldOverlaysByUUID = Dictionary(
            uniqueKeysWithValues: previous.overlays.map { ($0.id, $0) })
        let oldEffectsByUUID = Dictionary(
            uniqueKeysWithValues: previous.effects.map { ($0.id, $0) })
        let oldAudiosByUUID = Dictionary(uniqueKeysWithValues: previous.audios.map { ($0.id, $0) })

        var lanes: [MockLane] = []
        for lane in state.lanes {
            let kind = laneKind(lane.kind)
            let laneUUID = ids.lane(lane.id)
            lanes.append(
                MockLane(
                    id: laneUUID,
                    kind: kind,
                    isMain: lane.isMain,
                    engineID: lane.id
                ))

            for clip in lane.clips {
                result.clipLane[clip.id] = kind
                let uuid = ids.clip(clip.id)
                if lane.isMain {
                    let old = oldClips[clip.id] ?? oldClipsByUUID[uuid]
                    result.clips.append(mainClip(clip, uuid: uuid, old: old))
                    continue
                }
                let oldOverlay = oldOverlays[clip.id] ?? oldOverlaysByUUID[uuid]
                switch lane.kind {
                case "video":
                    result.overlays.append(
                        pipOverlay(clip, uuid: uuid, laneID: laneUUID, old: oldOverlay))
                case "text":
                    result.overlays.append(
                        textOverlay(clip, uuid: uuid, laneID: laneUUID, old: oldOverlay))
                case "sticker":
                    result.overlays.append(
                        stickerOverlay(clip, uuid: uuid, laneID: laneUUID, old: oldOverlay))
                case "effect", "filter", "adjustment":
                    result.effects.append(
                        effectClip(
                            clip, uuid: uuid, lane: lane.kind, laneID: laneUUID,
                            old: oldEffects[clip.id] ?? oldEffectsByUUID[uuid]))
                case "audio":
                    result.audios.append(
                        audioClip(
                            clip, uuid: uuid, laneID: laneUUID,
                            old: oldAudios[clip.id] ?? oldAudiosByUUID[uuid]))
                default:
                    break
                }
            }
        }
        if !lanes.contains(where: \.isMain) {
            lanes.insert(MockLane(kind: .video, isMain: true), at: 0)
        }
        result.lanes = lanes
        return result
    }

    // MARK: Per-kind builders

    private static func mainClip(_ clip: UiClip, uuid: UUID, old: MockClip?) -> MockClip {
        var result = MockClip(
            art: old?.art ?? art(for: clip),
            sourceDuration: clip.sourceDurationSeconds ?? clip.lengthSeconds,
            length: clip.lengthSeconds,
            hasAudio: clip.hasAudio
        )
        result.id = uuid
        result.engineID = clip.id
        result.mediaPath = clip.path
        result.trimStart = clip.trimInSeconds ?? 0
        result.volume = Double(clip.volume)
        result.fadeIn = clip.fadeInSeconds
        result.fadeOut = clip.fadeOutSeconds
        result.speed = clip.speed
        result.isReversed = clip.reversed
        result.keyframes = clip.keyframes
        // Freeze stills are the images the freeze intent wrote (named by the
        // media store); ordinary photo picks don't wear the snowflake.
        result.isFreeze = clip.isImage && clip.label.hasPrefix("freeze-")
        if let transform = clip.transform {
            result.posX = Double(transform.posX)
            result.posY = Double(transform.posY)
            result.scale = Double(transform.scale)
            result.rotationDegrees = Double(transform.rotationDegrees)
            result.opacity = Double(transform.opacity)
        }
        if let transition = clip.transitionAfter {
            result.transitionAfter = MockTransition(
                style: TransitionMap.style(forEngineID: transition.id),
                duration: transition.durationSeconds
            )
        }

        if let old {
            result.filterName = old.filterName
            result.filterIntensity = old.filterIntensity
            result.adjust = old.adjust
            result.animationIn = old.animationIn
            result.animationOut = old.animationOut
            result.animationCombo = old.animationCombo
            result.maskName = old.maskName
            result.cropPreset = old.cropPreset
            result.chromaStrength = old.chromaStrength
            result.chromaShadow = old.chromaShadow
            result.chromaColor = old.chromaColor
            result.stabilizeLevel = old.stabilizeLevel
            result.speedCurve = old.speedCurve
        }
        return result
    }

    private static func pipOverlay(_ clip: UiClip, uuid: UUID, laneID: UUID, old: MockOverlayClip?) -> MockOverlayClip {
        var result = base(clip, uuid: uuid, kind: .pip, laneID: laneID)
        result.art = old?.art ?? art(for: clip)
        result.sourceDuration = clip.sourceDurationSeconds
        result.pipHasAudio = clip.hasAudio
        result.volume = Double(clip.volume)
        return result
    }

    private static func textOverlay(_ clip: UiClip, uuid: UUID, laneID: UUID, old: MockOverlayClip?) -> MockOverlayClip {
        var result = base(clip, uuid: uuid, kind: .text, laneID: laneID)
        result.text = clip.text ?? ""
        if let style = clip.textStyle {
            result.textColor = color(style.fill)
            result.fontName = style.font.isEmpty ? "Default" : style.font
        }
        result.textEffect = old?.textEffect
        result.animation = old?.animation
        return result
    }

    private static func stickerOverlay(_ clip: UiClip, uuid: UUID, laneID: UUID, old: MockOverlayClip?) -> MockOverlayClip {
        var result = base(clip, uuid: uuid, kind: .sticker, laneID: laneID)
        result.symbol = old?.symbol ?? "face.smiling.inverse"
        if let rgba = clip.rgba {
            // Solid / shape fills render as flat color art (engine test seeds).
            let fill = color(rgba)
            result.art = MockArt(top: fill, bottom: fill, symbol: nil)
        }
        return result
    }

    private static func effectClip(_ clip: UiClip, uuid: UUID, lane: String, laneID: UUID, old: MockEffectClip?) -> MockEffectClip {
        let kind: MockEffectClip.Kind =
            switch lane {
            case "filter": .filter
            case "adjustment": .adjust
            default: .effect
            }
        var result = MockEffectClip(
            kind: kind,
            laneID: laneID,
            name: old?.name ?? clip.label.capitalized,
            start: clip.startSeconds,
            length: clip.lengthSeconds
        )
        result.id = uuid
        result.engineID = clip.id
        result.intensity = old?.intensity ?? 0.8
        return result
    }

    private static func audioClip(_ clip: UiClip, uuid: UUID, laneID: UUID, old: MockAudioClip?) -> MockAudioClip {
        var result = MockAudioClip(
            kind: old?.kind ?? (clip.link != nil ? .extracted : .music),
            laneID: laneID,
            title: old?.title ?? clip.label,
            start: clip.startSeconds,
            length: clip.lengthSeconds,
            sourceDuration: clip.sourceDurationSeconds ?? clip.lengthSeconds
        )
        result.id = uuid
        result.engineID = clip.id
        result.volume = Double(clip.volume)
        result.fadeIn = clip.fadeInSeconds
        result.fadeOut = clip.fadeOutSeconds
        result.waveSeed = Int(truncatingIfNeeded: clip.id &* 2_654_435_761) % 10_000
        return result
    }

    private static func base(_ clip: UiClip, uuid: UUID, kind: MockOverlayClip.Kind, laneID: UUID) -> MockOverlayClip {
        var result = MockOverlayClip(
            kind: kind,
            laneID: laneID,
            start: clip.startSeconds,
            length: clip.lengthSeconds
        )
        result.id = uuid
        result.engineID = clip.id
        if let transform = clip.transform {
            result.posX = Double(transform.posX)
            result.posY = Double(transform.posY)
            result.scale = Double(transform.scale)
            result.rotationDegrees = Double(transform.rotationDegrees)
            result.opacity = Double(transform.opacity)
        }
        return result
    }

    // MARK: Small helpers

    private static func laneKind(_ wire: String) -> MockLane.Kind {
        switch wire {
        case "text": .text
        case "sticker": .sticker
        case "effect", "filter", "adjustment": .effect
        case "audio": .audio
        default: .video
        }
    }

    /// Deterministic placeholder art for a media clip (real thumbnails come
    /// with the Phase E/F filmstrips).
    private static func art(for clip: UiClip) -> MockArt {
        var art = MockData.tileArt(for: clip.label)
        art.symbol = clip.kind == "image" ? "photo" : "film"
        return art
    }

    private static func color(_ rgb: [UInt8]) -> Color {
        Color(
            red: Double(rgb.count > 0 ? rgb[0] : 0) / 255,
            green: Double(rgb.count > 1 ? rgb[1] : 0) / 255,
            blue: Double(rgb.count > 2 ? rgb[2] : 0) / 255
        )
    }

    private static func index<T>(_ items: [T]) -> [UInt64: T] {
        var result: [UInt64: T] = [:]
        for item in items {
            let id: UInt64? =
                switch item {
                case let clip as MockClip: clip.engineID
                case let clip as MockOverlayClip: clip.engineID
                case let clip as MockEffectClip: clip.engineID
                case let clip as MockAudioClip: clip.engineID
                default: nil
                }
            if let id {
                result[id] = item
            }
        }
        return result
    }
}

// MARK: - Color -> wire bytes

nonisolated extension Color {
    /// `[r, g, b, a]` 0-255, for the engine's text/solid fills.
    var engineRGBA: [UInt8] {
        let components = resolvedComponents
        return [
            UInt8((components.r * 255).rounded()),
            UInt8((components.g * 255).rounded()),
            UInt8((components.b * 255).rounded()),
            UInt8((components.a * 255).rounded()),
        ]
    }

    /// `[r, g, b]` 0-255, for the opaque canvas background.
    var engineRGB: [UInt8] {
        Array(engineRGBA.prefix(3))
    }

    private var resolvedComponents: (r: Double, g: Double, b: Double, a: Double) {
        #if canImport(UIKit)
        var r: CGFloat = 0
        var g: CGFloat = 0
        var b: CGFloat = 0
        var a: CGFloat = 0
        UIColor(self).getRed(&r, green: &g, blue: &b, alpha: &a)
        return (min(max(r, 0), 1), min(max(g, 0), 1), min(max(b, 0), 1), min(max(a, 0), 1))
        #else
        guard let color = NSColor(self).usingColorSpace(.deviceRGB) else {
            return (1, 1, 1, 1)
        }
        return (
            min(max(color.redComponent, 0), 1),
            min(max(color.greenComponent, 0), 1),
            min(max(color.blueComponent, 0), 1),
            min(max(color.alphaComponent, 0), 1)
        )
        #endif
    }
}
