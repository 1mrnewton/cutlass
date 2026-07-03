import SwiftUI

/// The two-row grid of creation shortcuts under the home header.
struct QuickActionsGrid: View {
    struct Action: Identifiable {
        var id: String { label }
        var symbol: String
        var label: String
        /// Blank projects skip the media picker and open an empty editor.
        var isBlank = false
    }

    static let actions: [Action] = [
        Action(symbol: "photo.on.rectangle.angled", label: "New from photo library"),
        Action(symbol: "play.square.fill", label: "Create for Shorts"),
        Action(symbol: "doc.on.doc", label: "New from files"),
        Action(symbol: "rectangle", label: "New blank project", isBlank: true),
        Action(symbol: "waveform.badge.plus", label: "Extract audio"),
        Action(symbol: "captions.bubble", label: "Add captions"),
        Action(symbol: "sparkles.rectangle.stack", label: "Image to video"),
        Action(symbol: "text.below.photo", label: "Text to video"),
    ]

    var onNewProject: () -> Void
    var onBlankProject: () -> Void

    private let columns = Array(repeating: GridItem(.flexible(), spacing: 12), count: 4)

    var body: some View {
        LazyVGrid(columns: columns, spacing: 14) {
            ForEach(Self.actions) { action in
                Button {
                    action.isBlank ? onBlankProject() : onNewProject()
                } label: {
                    VStack(spacing: 7) {
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(.white.opacity(0.09))
                            .frame(height: 60)
                            .overlay {
                                Image(systemName: action.symbol)
                                    .font(.system(size: 19, weight: .medium))
                                    .foregroundStyle(.white)
                            }
                        Text(action.label)
                            .font(.system(size: 10.5))
                            .foregroundStyle(Theme.textSecondary)
                            .multilineTextAlignment(.center)
                            .lineLimit(2)
                            .frame(height: 26, alignment: .top)
                    }
                }
                .buttonStyle(.plain)
            }
        }
    }
}

#Preview {
    QuickActionsGrid(onNewProject: {}, onBlankProject: {})
        .padding()
        .background(Theme.background)
}
