import SwiftUI

/// Renders mock artwork: a gradient with an optional ghosted SF Symbol where
/// a real thumbnail or video frame would be.
struct MockArtView: View {
    var art: MockArt
    var symbolSize: CGFloat = 24

    var body: some View {
        ZStack {
            art.gradient
            if let symbol = art.symbol {
                Image(systemName: symbol)
                    .font(.system(size: symbolSize, weight: .medium))
                    .foregroundStyle(.white.opacity(0.35))
            }
        }
    }
}

/// Small translucent capsule used for duration badges on thumbnails.
struct DurationBadge: View {
    var duration: TimeInterval

    var body: some View {
        Text(duration.timecode)
            .font(.caption2.weight(.semibold).monospacedDigit())
            .foregroundStyle(.white)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(.black.opacity(0.55), in: Capsule())
    }
}
