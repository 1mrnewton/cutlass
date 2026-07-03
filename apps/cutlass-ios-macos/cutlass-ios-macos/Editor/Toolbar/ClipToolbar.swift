import SwiftUI

/// Bottom toolbar when a clip is selected: fixed add button, then a
/// horizontally scrollable strip of clip operations.
struct ClipToolbar: View {
    var onAdd: () -> Void
    var onSplit: () -> Void
    var onDelete: () -> Void
    var onDuplicate: () -> Void
    var onReplace: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Button(action: onAdd) {
                Circle()
                    .fill(Theme.accent)
                    .frame(width: 44, height: 44)
                    .overlay {
                        Image(systemName: "plus")
                            .font(.system(size: 19, weight: .semibold))
                            .foregroundStyle(.white)
                    }
                    .shadow(color: .black.opacity(0.4), radius: 6, y: 2)
            }
            .buttonStyle(.plain)
            .padding(.leading, 14)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 26) {
                    item("scissors", "Split", action: onSplit)
                    item("trash", "Delete", action: onDelete)
                    item("plus.square.on.square", "Duplicate", action: onDuplicate)
                    item("rectangle.2.swap", "Replace", action: onReplace)
                    item("arrow.up.to.line", "Move up") {}
                    item("speaker.wave.2", "Volume") {}
                }
                .padding(.horizontal, 18)
            }
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
                    .font(.system(size: 19, weight: .regular))
                    .foregroundStyle(.white)
                    .frame(height: 22)
                Text(label)
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.textSecondary)
            }
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    ClipToolbar(onAdd: {}, onSplit: {}, onDelete: {}, onDuplicate: {}, onReplace: {})
        .background(Theme.background)
}
