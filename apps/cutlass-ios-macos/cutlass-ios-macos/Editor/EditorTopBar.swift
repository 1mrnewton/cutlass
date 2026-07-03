import SwiftUI

/// Editor top chrome: home / fullscreen / overlay icons on the left, premium
/// badge and the Export pill on the right.
struct EditorTopBar: View {
    var exportEnabled: Bool
    var onHome: () -> Void
    var onExport: () -> Void

    var body: some View {
        HStack(spacing: 22) {
            Button(action: onHome) {
                Image(systemName: "house")
                    .font(.system(size: 17, weight: .medium))
                    .foregroundStyle(.white)
            }
            .buttonStyle(.plain)

            Image(systemName: "arrow.up.left.and.arrow.down.right")
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(.white)

            Image(systemName: "rectangle.portrait.on.rectangle.portrait")
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(.white)

            Spacer()

            Circle()
                .fill(Theme.premiumBadge)
                .frame(width: 26, height: 26)
                .overlay {
                    Image(systemName: "crown.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(.white)
                }

            Button(action: onExport) {
                Text("Export")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(exportEnabled ? .black : Theme.textTertiary)
                    .padding(.horizontal, 18)
                    .padding(.vertical, 7)
                    .background(
                        exportEnabled ? AnyShapeStyle(.white) : AnyShapeStyle(Theme.surface),
                        in: Capsule()
                    )
            }
            .buttonStyle(.plain)
            .disabled(!exportEnabled)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }
}

#Preview {
    VStack {
        EditorTopBar(exportEnabled: true, onHome: {}, onExport: {})
        EditorTopBar(exportEnabled: false, onHome: {}, onExport: {})
    }
    .background(Theme.background)
}
