import SwiftUI

/// Default bottom toolbar (nothing selected): the three add-content entry
/// points from the reference design.
struct MediaToolbar: View {
    var onAddMedia: () -> Void

    var body: some View {
        HStack {
            Spacer()
            item("photo.badge.plus", "Videos\nand images", action: onAddMedia)
            Spacer()
            item("waveform.badge.plus", "Music\nand audio") {}
            Spacer()
            item("textformat", "Titles\nand captions") {}
            Spacer()
        }
        .padding(.top, 10)
        .padding(.bottom, 4)
    }

    private func item(
        _ symbol: String,
        _ label: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(spacing: 6) {
                Image(systemName: symbol)
                    .font(.system(size: 21, weight: .regular))
                    .foregroundStyle(.white)
                    .frame(height: 24)
                Text(label)
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.textSecondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
            }
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    MediaToolbar(onAddMedia: {})
        .background(Theme.background)
}
