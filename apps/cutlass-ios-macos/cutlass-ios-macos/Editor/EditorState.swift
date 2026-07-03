import SwiftUI

/// In-memory state for the mock editor: a single sequential video track.
/// All edits are pure array/state manipulation; nothing touches the engine.
@Observable
final class EditorState {
    var clips: [MockClip] = []
    var playhead: TimeInterval = 0
    var selectedClipID: MockClip.ID?
    var isPlaying = false

    /// Timeline zoom: how many seconds one point of track width represents.
    var secondsPerPoint: Double = 1.0 / 44.0

    var isEmpty: Bool { clips.isEmpty }

    var duration: TimeInterval {
        clips.reduce(0) { $0 + $1.length }
    }

    var selectedClip: MockClip? {
        selectedClipID.flatMap { id in clips.first { $0.id == id } }
    }

    // MARK: Time <-> clip mapping

    /// Timeline start time of the given clip.
    func startTime(of id: MockClip.ID) -> TimeInterval {
        var start: TimeInterval = 0
        for clip in clips {
            if clip.id == id { break }
            start += clip.length
        }
        return start
    }

    /// The clip under a timeline position; the playhead is always clamped to
    /// the timeline, so positions at or past the end hold the last clip.
    func clip(at time: TimeInterval) -> MockClip? {
        var start: TimeInterval = 0
        for clip in clips {
            let end = start + clip.length
            if time < end { return clip }
            start = end
        }
        return clips.last
    }

    // MARK: Project lifecycle

    func startProject(with items: [MockMediaItem]) {
        clips = items.map(MockClip.init(from:))
        playhead = 0
        selectedClipID = nil
        isPlaying = false
    }

    func appendMedia(_ items: [MockMediaItem]) {
        clips.append(contentsOf: items.map(MockClip.init(from:)))
    }

    // MARK: Transport

    func stepFrame(by direction: Double) {
        isPlaying = false
        playhead = min(max(0, playhead + direction / 30.0), duration)
    }
}
