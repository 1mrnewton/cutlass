import AVFoundation
import CoreTransferable
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// A picked item, written to a temporary file and probed the way the desktop
/// probes before import: a still's upright size, or a movie's upright size
/// and length.
struct PickedMedia {
    let url: URL
    let media: Media

    /// Read `item` from the Photos library. A photo is redrawn upright as a
    /// JPEG, so its orientation and format are ones the editor reads.
    static func load(_ item: PhotosPickerItem) async throws -> PickedMedia {
        if item.supportedContentTypes.contains(where: { $0.conforms(to: .movie) }) {
            guard let movie = try await item.loadTransferable(type: MovieFile.self) else {
                throw ImportFailure("The movie could not be read")
            }
            return try await probeMovie(movie.url)
        }
        guard let data = try await item.loadTransferable(type: Data.self),
              let image = UIImage(data: data)
        else {
            throw ImportFailure("The photo could not be read")
        }
        return try upright(image)
    }

    private static func probeMovie(_ url: URL) async throws -> PickedMedia {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ImportFailure("The movie has no picture")
        }
        let (size, transform) = try await track.load(.naturalSize, .preferredTransform)
        let upright = size.applying(transform)
        let micros = Int64((duration.seconds * 1_000_000).rounded())
        return PickedMedia(
            url: url,
            media: .movie(
                width: Int32(abs(upright.width).rounded()),
                height: Int32(abs(upright.height).rounded()),
                durationUs: micros
            )
        )
    }

    private static func upright(_ image: UIImage) throws -> PickedMedia {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let drawn = UIGraphicsImageRenderer(size: image.size, format: format).image { _ in
            image.draw(at: .zero)
        }
        guard let jpeg = drawn.jpegData(compressionQuality: 0.92), let pixels = drawn.cgImage else {
            throw ImportFailure("The photo could not be converted")
        }
        let url = try scratchFolder().appending(path: "Photo \(UUID().uuidString.prefix(8)).jpg")
        try jpeg.write(to: url)
        return PickedMedia(
            url: url,
            media: .still(width: Int32(pixels.width), height: Int32(pixels.height))
        )
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

/// A movie copied out of the Photos library into a temporary file.
struct MovieFile: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let copy = try scratchFolder().appending(path: received.file.lastPathComponent)
            try FileManager.default.copyItem(at: received.file, to: copy)
            return MovieFile(url: copy)
        }
    }
}
