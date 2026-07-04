import Foundation

// Builders for the two JSON payloads the session accepts.
//
// `Intent` covers the gesture-level operations (grouped, atomic, the normal
// path for UI edits); `Command` is the raw engine vocabulary for the few
// operations without an intent (track toggles, canvas, text editing, project
// I/O) plus an escape hatch for anything new. Both just assemble the JSON the
// Rust side deserializes — validation happens in the engine, once, for every
// platform.

/// A raw wire command (`{"type": "...", ...}`). Constructors cover what the
/// iOS editor needs; `raw` passes anything else straight through.
public struct Command: Sendable {
    let object: [String: JSONValue]

    private init(type: String, _ fields: [String: JSONValue] = [:]) {
        var object = fields
        object["type"] = .string(type)
        self.object = object
    }

    /// Escape hatch: a full command object (must include `"type"`).
    public static func raw(_ object: [String: JSONValue]) -> Command {
        Command(object: object)
    }

    private init(object: [String: JSONValue]) {
        self.object = object
    }

    // Project I/O.

    public static func importMedia(path: String) -> Command {
        Command(type: "Import", ["path": .string(path)])
    }

    public static func save(path: String) -> Command {
        Command(type: "Save", ["path": .string(path)])
    }

    public static func load(path: String) -> Command {
        Command(type: "Load", ["path": .string(path)])
    }

    public static func export(path: String) -> Command {
        Command(type: "Export", ["path": .string(path)])
    }

    public static func relinkMedia(media: UInt64, path: String) -> Command {
        Command(type: "RelinkMedia", ["media": .int(Int64(media)), "path": .string(path)])
    }

    // Track structure.

    /// `kind` ∈ `Video | Audio | Text | Sticker | Effect | Filter |
    /// Adjustment` (the engine's `TrackKind` names).
    public static func addTrack(kind: String, name: String, index: Int? = nil) -> Command {
        var fields: [String: JSONValue] = ["kind": .string(kind), "name": .string(name)]
        if let index { fields["index"] = .int(Int64(index)) }
        return Command(type: "AddTrack", fields)
    }

    public static func removeTrack(track: UInt64) -> Command {
        Command(type: "RemoveTrack", ["track": .int(Int64(track))])
    }

    public static func setTrackEnabled(track: UInt64, enabled: Bool) -> Command {
        Command(type: "SetTrackEnabled", ["track": .int(Int64(track)), "enabled": .bool(enabled)])
    }

    public static func setTrackMuted(track: UInt64, muted: Bool) -> Command {
        Command(type: "SetTrackMuted", ["track": .int(Int64(track)), "muted": .bool(muted)])
    }

    public static func setTrackLocked(track: UInt64, locked: Bool) -> Command {
        Command(type: "SetTrackLocked", ["track": .int(Int64(track)), "locked": .bool(locked)])
    }

    // Clips.

    /// Place a generated clip; times are frame ticks at `fps`.
    public static func addGenerated(
        track: UInt64, generator: Generator, startTicks: Int64, durationTicks: Int64, fps: Fraction
    ) -> Command {
        Command(
            type: "AddGenerated",
            [
                "track": .int(Int64(track)),
                "generator": generator.json,
                "timeline": .object([
                    "start": Self.time(startTicks, fps),
                    "duration": Self.time(durationTicks, fps),
                ]),
            ])
    }

    /// Replace a generated clip's content (edit a title, recolor a solid).
    public static func setGenerator(clip: UInt64, generator: Generator) -> Command {
        Command(type: "SetGenerator", ["clip": .int(Int64(clip)), "generator": generator.json])
    }

    public static func removeClip(clip: UInt64) -> Command {
        Command(type: "RemoveClip", ["clip": .int(Int64(clip))])
    }

    /// Delete from the main track, closing the gap (later clips slide left).
    public static func rippleDelete(clip: UInt64) -> Command {
        Command(type: "RippleDelete", ["clip": .int(Int64(clip))])
    }

    // Canvas.

    /// `aspect` ∈ `auto | 16:9 | 9:16 | 1:1 | 4:5 | 21:9`; `background` is
    /// opaque `[r, g, b]`.
    public static func setCanvas(aspect: String, background: [UInt8]) -> Command {
        Command(
            type: "SetCanvas",
            [
                "aspect": .string(aspect),
                "background": .array(background.map { .int(Int64($0)) }),
            ])
    }

    // Clip looks (persist-only this milestone; ids come from `Catalogs`).

    /// Set or clear (`nil`) a clip's mask. Visual clips only.
    public static func setClipMask(clip: UInt64, mask: UiMask?) -> Command {
        var fields: [String: JSONValue] = ["clip": .int(Int64(clip))]
        if let mask {
            var object: [String: JSONValue] = ["kind": .string(mask.kind)]
            if mask.feather != 0 { object["feather"] = .number(Double(mask.feather)) }
            if mask.invert { object["invert"] = .bool(true) }
            fields["mask"] = .object(object)
        }
        return Command(type: "SetClipMask", fields)
    }

    /// Set or clear (`nil`) green-screen keying. Media-backed visual clips.
    public static func setClipChroma(clip: UInt64, chroma: UiChromaKey?) -> Command {
        var fields: [String: JSONValue] = ["clip": .int(Int64(clip))]
        if let chroma {
            fields["chroma"] = .object([
                "rgb": .array(chroma.rgb.map { .int(Int64($0)) }),
                "strength": .number(Double(chroma.strength)),
                "shadow": .number(Double(chroma.shadow)),
            ])
        }
        return Command(type: "SetClipChroma", fields)
    }

    /// Set or clear (`nil`) stabilization. Video-media clips only.
    /// `level` ∈ `recommended | smooth | max_smooth`.
    public static func setClipStabilize(clip: UInt64, level: String?) -> Command {
        var fields: [String: JSONValue] = ["clip": .int(Int64(clip))]
        if let level { fields["stabilize"] = .string(level) }
        return Command(type: "SetClipStabilize", fields)
    }

    /// Set or clear (`nil`) a filter preset. Visual clips only.
    public static func setClipFilter(clip: UInt64, filter: UiFilter?) -> Command {
        var fields: [String: JSONValue] = ["clip": .int(Int64(clip))]
        if let filter {
            fields["filter"] = .object([
                "id": .string(filter.id),
                "intensity": .number(Double(filter.intensity)),
            ])
        }
        return Command(type: "SetClipFilter", fields)
    }

    /// Write the color-grade sliders (all-neutral = grade off). Visual clips.
    public static func setClipAdjustments(clip: UInt64, adjust: UiAdjust) -> Command {
        var object: [String: JSONValue] = [:]
        if adjust.brightness != 0 { object["brightness"] = .number(Double(adjust.brightness)) }
        if adjust.contrast != 0 { object["contrast"] = .number(Double(adjust.contrast)) }
        if adjust.saturation != 0 { object["saturation"] = .number(Double(adjust.saturation)) }
        if adjust.exposure != 0 { object["exposure"] = .number(Double(adjust.exposure)) }
        if adjust.temperature != 0 { object["temperature"] = .number(Double(adjust.temperature)) }
        return Command(
            type: "SetClipAdjustments",
            ["clip": .int(Int64(clip)), "adjust": .object(object)])
    }

    /// Set or clear (`nil`) the animation in one slot. `slot` ∈ `in | out |
    /// combo`; a combo evicts in/out and vice versa (engine-enforced).
    public static func setClipAnimation(clip: UInt64, slot: String, animationID: String?)
        -> Command
    {
        var fields: [String: JSONValue] = [
            "clip": .int(Int64(clip)),
            "slot": .string(slot),
        ]
        if let animationID { fields["animation"] = .object(["id": .string(animationID)]) }
        return Command(type: "SetClipAnimation", fields)
    }

    /// Tag or untag (`nil`) an audio-lane clip's role.
    /// `role` ∈ `music | sfx | voiceover | extracted`.
    public static func setAudioRole(clip: UInt64, role: String?) -> Command {
        var fields: [String: JSONValue] = ["clip": .int(Int64(clip))]
        if let role { fields["role"] = .string(role) }
        return Command(type: "SetAudioRole", fields)
    }

    private static func time(_ ticks: Int64, _ fps: Fraction) -> JSONValue {
        .object([
            "value": .int(ticks),
            "rate": .object(["num": .int(Int64(fps.num)), "den": .int(Int64(fps.den))]),
        ])
    }
}

/// Generated clip content for `addGenerated` / `setGenerator`.
public enum Generator: Sendable {
    case solidColor(rgba: [UInt8])
    case text(content: String, style: TextStyle)
    case sticker
    case effect
    case filter
    case adjustment

    var json: JSONValue {
        switch self {
        case .solidColor(let rgba):
            return .object([
                "SolidColor": .object(["rgba": .array(rgba.map { .int(Int64($0)) })])
            ])
        case .text(let content, let style):
            let styleData = (try? WireCoding.encoder.encode(style)) ?? Data("{}".utf8)
            let styleJSON =
                (try? WireCoding.plainDecoder.decode(JSONValue.self, from: styleData))
                ?? .object([:])
            return .object([
                "Text": .object(["content": .string(content), "style": styleJSON])
            ])
        case .sticker: return .string("Sticker")
        case .effect: return .string("Effect")
        case .filter: return .string("Filter")
        case .adjustment: return .string("Adjustment")
        }
    }
}

/// A gesture-level operation (`{"intent": "...", ...}`) — the normal way UI
/// edits reach the engine. Multi-command intents run in one history group and
/// roll back atomically.
public struct Intent: Sendable {
    let object: [String: JSONValue]

    private init(_ name: String, _ fields: [String: JSONValue] = [:]) {
        var object = fields
        object["intent"] = .string(name)
        self.object = object
    }

    /// Import each file and append it to the end of the main track.
    public static func appendMain(paths: [String]) -> Intent {
        Intent("append_main", ["paths": .array(paths.map(JSONValue.string))])
    }

    /// Import a file and ripple-insert it at the main-track boundary nearest
    /// `atSeconds`.
    public static func insertMain(path: String, atSeconds: Double) -> Intent {
        Intent("insert_main", ["path": .string(path), "at_seconds": .number(atSeconds)])
    }

    /// Split any clip at an absolute timeline position.
    public static func split(clip: UInt64, seconds: Double) -> Intent {
        Intent("split", ["clip": .int(Int64(clip)), "seconds": .number(seconds)])
    }

    /// Ripple-trim a main-track clip. `edge` ∈ `leading | trailing`;
    /// `deltaSeconds` is the signed movement of that edge.
    public static func rippleTrimMain(clip: UInt64, edge: String, deltaSeconds: Double) -> Intent {
        Intent(
            "ripple_trim_main",
            [
                "clip": .int(Int64(clip)),
                "edge": .string(edge),
                "delta_seconds": .number(deltaSeconds),
            ])
    }

    /// Re-place a free lane clip (non-ripple).
    public static func trimLane(clip: UInt64, startSeconds: Double, lengthSeconds: Double) -> Intent
    {
        Intent(
            "trim_lane",
            [
                "clip": .int(Int64(clip)),
                "start_seconds": .number(startSeconds),
                "length_seconds": .number(lengthSeconds),
            ])
    }

    /// Move a lane clip (or lift a main clip) to a lane at `startSeconds`.
    /// `track: nil` picks/creates a fitting lane.
    public static func moveLane(clip: UInt64, track: UInt64? = nil, startSeconds: Double) -> Intent
    {
        var fields: [String: JSONValue] = [
            "clip": .int(Int64(clip)),
            "start_seconds": .number(startSeconds),
        ]
        if let track { fields["track"] = .int(Int64(track)) }
        return Intent("move_lane", fields)
    }

    /// Insert a clip into the main track at slot `index` (reorder when it
    /// already lives there).
    public static func insertIntoMain(clip: UInt64, index: Int) -> Intent {
        Intent("insert_into_main", ["clip": .int(Int64(clip)), "index": .int(Int64(index))])
    }

    public static func addText(text: String, atSeconds: Double, durationSeconds: Double = 3)
        -> Intent
    {
        Intent(
            "add_text",
            [
                "text": .string(text),
                "at_seconds": .number(atSeconds),
                "duration_seconds": .number(durationSeconds),
            ])
    }

    public static func addSticker(atSeconds: Double, durationSeconds: Double = 3) -> Intent {
        Intent(
            "add_sticker",
            [
                "at_seconds": .number(atSeconds),
                "duration_seconds": .number(durationSeconds),
            ])
    }

    /// Drop an effect bar. `kind` ∈ `effect | filter | adjustment`.
    public static func addEffect(kind: String, atSeconds: Double, durationSeconds: Double = 3)
        -> Intent
    {
        Intent(
            "add_effect",
            [
                "kind": .string(kind),
                "at_seconds": .number(atSeconds),
                "duration_seconds": .number(durationSeconds),
            ])
    }

    /// Import a file and drop it as picture-in-picture on an overlay lane.
    public static func addPip(path: String, atSeconds: Double) -> Intent {
        Intent("add_pip", ["path": .string(path), "at_seconds": .number(atSeconds)])
    }

    /// Import a file and drop it on an audio lane. `role` tags the clip with
    /// the picker tab it came from (`music | sfx | voiceover`).
    public static func addAudio(path: String, atSeconds: Double, role: String? = nil) -> Intent {
        var fields: [String: JSONValue] = [
            "path": .string(path),
            "at_seconds": .number(atSeconds),
        ]
        if let role { fields["role"] = .string(role) }
        return Intent("add_audio", fields)
    }

    /// Duplicate a clip right after itself.
    public static func duplicate(clip: UInt64) -> Intent {
        Intent("duplicate", ["clip": .int(Int64(clip))])
    }

    /// Swap a media clip's source file, keeping its slot.
    public static func replaceMedia(clip: UInt64, path: String) -> Intent {
        Intent("replace_media", ["clip": .int(Int64(clip)), "path": .string(path)])
    }

    /// CapCut "extract audio": linked audio clip on an audio lane, original
    /// muted via linkage.
    public static func extractAudio(clip: UInt64) -> Intent {
        Intent("extract_audio", ["clip": .int(Int64(clip))])
    }

    /// CapCut "freeze": extract the frame under the playhead to `pngPath`,
    /// import it as a still, and ripple-insert it there (splitting mid-clip).
    public static func freeze(
        clip: UInt64, seconds: Double, pngPath: String, durationSeconds: Double = 3
    ) -> Intent {
        Intent(
            "freeze",
            [
                "clip": .int(Int64(clip)),
                "seconds": .number(seconds),
                "png_path": .string(pngPath),
                "duration_seconds": .number(durationSeconds),
            ])
    }

    /// Retime a media clip.
    public static func setSpeed(clip: UInt64, speed: Double, reversed: Bool = false) -> Intent {
        Intent(
            "set_speed",
            [
                "clip": .int(Int64(clip)),
                "speed": .number(speed),
                "reversed": .bool(reversed),
            ])
    }

    /// Apply a speed-curve preset from the catalog (`nil` restores constant
    /// speed). The clip's duration re-derives from the curve in one undo step.
    public static func setSpeedPreset(clip: UInt64, preset: String?) -> Intent {
        var fields: [String: JSONValue] = ["clip": .int(Int64(clip))]
        if let preset { fields["preset"] = .string(preset) }
        return Intent("set_speed_preset", fields)
    }

    /// Volume slider + fade handles. `volume: nil` keeps the current gain.
    public static func setAudio(
        clip: UInt64, volume: Float? = nil, fadeInSeconds: Double = 0, fadeOutSeconds: Double = 0
    ) -> Intent {
        var fields: [String: JSONValue] = [
            "clip": .int(Int64(clip)),
            "fade_in_seconds": .number(fadeInSeconds),
            "fade_out_seconds": .number(fadeOutSeconds),
        ]
        if let volume { fields["volume"] = .number(Double(volume)) }
        return Intent("set_audio", fields)
    }

    /// Canvas placement in UI coordinates (0..1, 0.5 = center). With
    /// `atSeconds` the edit composes with animation (writes keyframes on
    /// animated properties); without it everything flattens to constants.
    public static func setTransform(
        clip: UInt64, posX: Float, posY: Float, scale: Float, rotationDegrees: Float,
        opacity: Float, atSeconds: Double? = nil
    ) -> Intent {
        var fields: [String: JSONValue] = [
            "clip": .int(Int64(clip)),
            "pos_x": .number(Double(posX)),
            "pos_y": .number(Double(posY)),
            "scale": .number(Double(scale)),
            "rotation_degrees": .number(Double(rotationDegrees)),
            "opacity": .number(Double(opacity)),
        ]
        if let atSeconds { fields["at_seconds"] = .number(atSeconds) }
        return Intent("set_transform", fields)
    }

    /// Stamp or remove a transform keyframe at an absolute timeline position.
    public static func toggleTransformKeyframe(clip: UInt64, seconds: Double) -> Intent {
        Intent(
            "toggle_transform_keyframe",
            [
                "clip": .int(Int64(clip)),
                "seconds": .number(seconds),
            ])
    }

    /// Set, change, or clear (`transitionID: nil`) the transition at a
    /// main-track clip's right junction.
    public static func setTransition(
        clip: UInt64, transitionID: String?, durationSeconds: Double = 0
    ) -> Intent {
        var fields: [String: JSONValue] = [
            "clip": .int(Int64(clip)),
            "duration_seconds": .number(durationSeconds),
        ]
        if let transitionID { fields["transition_id"] = .string(transitionID) }
        return Intent("set_transition", fields)
    }
}
