import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// Copies PhotosPicker selections into the staging directory as real files
/// the engine can probe. `EditorState` later adopts staged files into the
/// project's media directory (a same-volume rename).
nonisolated enum MediaImporter {
    /// Stage one pick; nil when the asset can't be loaded.
    static func stage(_ item: PhotosPickerItem) async -> URL? {
        if item.supportedContentTypes.contains(where: { $0.conforms(to: .movie) }) {
            return (try? await item.loadTransferable(type: StagedMovie.self))?.url
        }
        // Images travel as data in their original encoding (HEIC/PNG/JPEG —
        // the engine's ImageIO probe handles all of them).
        guard let data = try? await item.loadTransferable(type: Data.self) else { return nil }
        let ext = item.supportedContentTypes.first?.preferredFilenameExtension ?? "jpg"
        let url = stagingURL(extension: ext)
        do {
            try data.write(to: url)
            return url
        } catch {
            print("cutlass: staging image failed: \(error)")
            return nil
        }
    }

    static func stagingURL(extension ext: String) -> URL {
        ProjectMediaStore.incomingDirectory
            .appendingPathComponent("\(UUID().uuidString).\(ext)")
    }
}

/// Movie picks transfer as file copies straight to the staging directory —
/// never loaded into memory.
nonisolated private struct StagedMovie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let ext = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let url = MediaImporter.stagingURL(extension: ext)
            try FileManager.default.copyItem(at: received.file, to: url)
            return Self(url: url)
        }
    }
}
