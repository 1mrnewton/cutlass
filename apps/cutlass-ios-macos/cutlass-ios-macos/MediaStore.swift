import Foundation

/// On-disk home for one project's media: `Documents/Projects/<id>/media/`.
/// Picker copies land here, freeze-frame PNGs are written here by the
/// engine, and (come the persistence phase) `project.cutlass` will sit next
/// to it. Directories are created on first use.
nonisolated struct ProjectMediaStore {
    let projectID: UUID

    /// Staging area for picker copies made before/outside a project
    /// (`Documents/Incoming/`). `adopt(_:)` moves staged files in.
    static var incomingDirectory: URL {
        ensured(documents.appendingPathComponent("Incoming", isDirectory: true))
    }

    var mediaDirectory: URL {
        Self.ensured(
            Self.documents
                .appendingPathComponent("Projects", isDirectory: true)
                .appendingPathComponent(projectID.uuidString, isDirectory: true)
                .appendingPathComponent("media", isDirectory: true))
    }

    /// Where the next freeze-frame still should be written.
    func freezeFrameURL() -> URL {
        mediaDirectory.appendingPathComponent(
            "freeze-\(UUID().uuidString.prefix(8)).png")
    }

    /// Claim a file for this project: staged picks *move* into the media
    /// directory (same volume, a rename); anything else — bundled fixtures,
    /// files already in place — imports where it lies.
    func adopt(_ url: URL) -> URL {
        guard url.path.hasPrefix(Self.incomingDirectory.path) else { return url }
        let destination = mediaDirectory.appendingPathComponent(url.lastPathComponent)
        do {
            try FileManager.default.moveItem(at: url, to: destination)
            return destination
        } catch {
            print("cutlass: media adopt failed, importing in place: \(error)")
            return url
        }
    }

    private static var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private static func ensured(_ url: URL) -> URL {
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
