import SwiftUI

/// Add text, or change a text clip: the words on top with a check that
/// closes, the faces Rust found below. The words show on the preview as
/// they are typed: the first letter adds the clip at the playhead, and the
/// whole typing is one undo step. Picking a face applies it at once, as an
/// undo step of its own.
struct TextPanel: View {
    let model: EditorModel
    /// The clip being changed, or nil to add one.
    let clip: String?
    /// The clip the words go into: `clip`, or the one the first letter added.
    @State private var typed: String?
    let close: () -> Void
    @State private var words: String
    @State private var face: String
    @FocusState private var typing: Bool

    init(model: EditorModel, clip: String?, close: @escaping () -> Void) {
        self.model = model
        self.clip = clip
        self.close = close
        _typed = State(initialValue: clip)
        let text = clip.flatMap { model.text(of: $0) }
        _words = State(initialValue: text?.content ?? "")
        _face = State(initialValue: text?.face ?? model.fonts.first?.name ?? "")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                TextField("Enter text", text: $words, axis: .vertical)
                    .lineLimit(1...3)
                    .focused($typing)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 9)
                    .background(Color(white: 0.18), in: RoundedRectangle(cornerRadius: 8))
                    .submitLabel(.done)
                Button(action: done) {
                    HugeIcon("done", size: 20)
                        .frame(width: 40, height: 36)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Done")
            }
            Text("Fonts")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white.opacity(0.6))
            if model.fonts.isEmpty {
                Text("No fonts were found on this device.")
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.6))
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(model.fonts, id: \.name) { font in
                            Button { pick(font.name) } label: {
                                Text(font.label)
                                    .font(.footnote)
                                    .lineLimit(1)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                                    .background(
                                        font.name == face ? Palette.accent : Color(white: 0.18),
                                        in: Capsule())
                                    .foregroundStyle(font.name == face ? .black : .white)
                            }
                            .accessibilityAddTraits(font.name == face ? .isSelected : [])
                        }
                    }
                }
            }
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 16)
        .padding(.top, 12)
        .padding(.bottom, 16)
        .background(Color(white: 0.11))
        .onAppear { typing = clip == nil }
        .onChange(of: words) { _, words in
            typed = model.typeText(into: typed, words, face: face)
        }
    }

    private func pick(_ name: String) {
        face = name
        if let typed { model.setTextFont(typed, face: name) }
    }

    /// Close. The words are already in the clip; blank words added nothing
    /// and leave a clip's words as they were.
    private func done() {
        typing = false
        close()
    }
}
