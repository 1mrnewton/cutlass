//
//  ProjectStore.swift
//  cutlass-ios-macos
//
//  Enumerates `.cutlass` project files in the app documents directory and backs
//  the home gallery. Projects are plain files on disk; the engine owns their
//  contents (open/save through `CutlassSession`).
//

import Combine
import CoreGraphics
import Foundation
import ImageIO
import SwiftUI
import UniformTypeIdentifiers
import CutlassMobile

/// One project on disk, plus its (optional) thumbnail sidecar.
struct ProjectEntry: Identifiable, Hashable {
    /// The `.cutlass` file URL — also the stable identity.
    let url: URL
    var name: String
    var modified: Date
    var thumbnailURL: URL?

    var id: URL { url }
}

/// Where projects and their caches live, and how the gallery is built.
@MainActor
final class ProjectStore: ObservableObject {
    @Published private(set) var projects: [ProjectEntry] = []

    private let fileManager = FileManager.default

    /// App documents directory (user-visible projects).
    var documentsDir: URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// Engine cache directory (frame cache / scratch). Created on demand.
    var cacheDir: String {
        let dir = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("cutlass-sessions", isDirectory: true)
        try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.path
    }

    /// Re-scan the documents directory for `.cutlass` files.
    func refresh() {
        let urls = (try? fileManager.contentsOfDirectory(
            at: documentsDir,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        projects = urls
            .filter { $0.pathExtension == "cutlass" }
            .map { url in
                let modified = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
                    .contentModificationDate ?? .distantPast
                let thumb = url.deletingPathExtension().appendingPathExtension("png")
                return ProjectEntry(
                    url: url,
                    name: url.deletingPathExtension().lastPathComponent,
                    modified: modified,
                    thumbnailURL: fileManager.fileExists(atPath: thumb.path) ? thumb : nil
                )
            }
            .sorted { $0.modified > $1.modified }
    }

    /// A unique `<base> N.cutlass` URL in the documents directory.
    func uniqueProjectURL(base: String = "Project") -> URL {
        var index = 1
        var candidate = documentsDir.appendingPathComponent("\(base).cutlass")
        while fileManager.fileExists(atPath: candidate.path) {
            index += 1
            candidate = documentsDir.appendingPathComponent("\(base) \(index).cutlass")
        }
        return candidate
    }

    /// The thumbnail sidecar URL for a project file.
    func thumbnailURL(for projectURL: URL) -> URL {
        projectURL.deletingPathExtension().appendingPathExtension("png")
    }

    func delete(_ entry: ProjectEntry) {
        try? fileManager.removeItem(at: entry.url)
        if let thumb = entry.thumbnailURL {
            try? fileManager.removeItem(at: thumb)
        }
        refresh()
    }
}

/// Write a `CGImage` to `url` as PNG (best-effort thumbnail persistence).
func writePNG(_ image: CGImage, to url: URL) {
    guard let dest = CGImageDestinationCreateWithURL(
        url as CFURL, UTType.png.identifier as CFString, 1, nil
    ) else { return }
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
}

/// Holds a `CGImage` so it can cross actor boundaries without a Sendable
/// warning; we never mutate it after construction.
struct FrameBox: @unchecked Sendable {
    let image: CGImage?
}

/// A picked video, copied out of the Photos sandbox into a temp file we own so
/// the native decoder can open it by a stable path.
struct PickedMovie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let ext = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let copy = FileManager.default.temporaryDirectory
                .appendingPathComponent("cutlass-import-\(UUID().uuidString)")
                .appendingPathExtension(ext)
            try? FileManager.default.removeItem(at: copy)
            try FileManager.default.copyItem(at: received.file, to: copy)
            return PickedMovie(url: copy)
        }
    }
}
