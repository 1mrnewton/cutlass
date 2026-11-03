import CoreTransferable
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// A picked item, copied out of the Photos library into a temporary file
/// that keeps its own name. Rust reads the file: what it is, its upright
/// size, and its length.
///
/// A movie comes as recorded; the Apple decoder plays HEVC, so nothing is
/// converted. A photo is asked for as PNG or JPEG, so Photos hands a HEIC
/// photo over as JPEG.
struct PickedFile: Transferable {
    let url: URL

    /// Copy `item` out of the Photos library.
    static func load(_ item: PhotosPickerItem) async throws -> URL {
        guard let file = try await item.loadTransferable(type: PickedFile.self) else {
            throw ImportFailure("The item could not be read")
        }
        return file.url
    }

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .movie) { try copy($0) }
        FileRepresentation(importedContentType: .png) { try copy($0) }
        FileRepresentation(importedContentType: .jpeg) { try copy($0) }
    }

    private static func copy(_ received: ReceivedTransferredFile) throws -> PickedFile {
        let copy = try scratchFolder().appending(path: received.file.lastPathComponent)
        try FileManager.default.copyItem(at: received.file, to: copy)
        return PickedFile(url: copy)
    }
}

/// A fresh folder in the temporary directory, so a picked file keeps its own
/// name. The caller removes the folder once the file is imported.
nonisolated private func scratchFolder() throws -> URL {
    let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    return folder
}

struct ImportFailure: LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
}
