import SwiftUI
import UniformTypeIdentifiers

/// One square cell in the samples grid: an engine-rendered thumbnail (or a
/// waveform glyph for audio), a duration badge, and a numbered badge +
/// border when selected.
struct MediaCell: View {
    var url: URL
    /// 1-based position in the selection order, nil when unselected.
    var selectionIndex: Int?

    @State private var thumbnail: CGImage?
    @State private var duration: Double?

    private var isAudio: Bool {
        UTType(filenameExtension: url.pathExtension)?.conforms(to: .audio) == true
    }

    private var isVideo: Bool {
        UTType(filenameExtension: url.pathExtension)?.conforms(to: .movie) == true
    }

    var body: some View {
        ZStack {
            Rectangle().fill(Theme.surfaceElevated)
            if let thumbnail {
                Image(decorative: thumbnail, scale: 1)
                    .resizable()
                    .scaledToFill()
            } else {
                Image(systemName: isAudio ? "waveform" : "photo")
                    .font(.system(size: 22, weight: .medium))
                    .foregroundStyle(Theme.textTertiary)
            }
        }
        .aspectRatio(1, contentMode: .fill)
        .clipped()
        .overlay(alignment: .bottomLeading) {
            if let duration, duration > 0, isVideo {
                DurationBadge(duration: duration)
                    .padding(4)
            }
        }
        .overlay {
            if selectionIndex != nil {
                Rectangle()
                    .fill(.black.opacity(0.35))
                Rectangle()
                    .strokeBorder(.white, lineWidth: 2)
            }
        }
        .overlay(alignment: .topTrailing) {
            if let index = selectionIndex {
                Text("\(index)")
                    .font(.caption2.bold())
                    .foregroundStyle(.white)
                    .frame(width: 20, height: 20)
                    .background(Theme.accent, in: Circle())
                    .padding(5)
            }
        }
        .contentShape(Rectangle())
        .task(id: url) {
            guard !isAudio else { return }
            thumbnail = await ThumbnailCache.shared.thumbnail(
                path: url.path, seconds: 0, height: 256)
            duration = await ThumbnailCache.shared.duration(path: url.path)
        }
    }
}

#Preview {
    HStack(spacing: 2) {
        if let video = FixtureLibrary.video {
            MediaCell(url: video, selectionIndex: nil)
        }
        if let photo = FixtureLibrary.photo {
            MediaCell(url: photo, selectionIndex: 2)
        }
    }
    .background(Theme.background)
}
