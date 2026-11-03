import SwiftUI

/// Add text, or change a text clip: the words on top with a check that
/// saves them, the faces Rust found below. A new clip is placed at the
/// playhead when the check is tapped. On a clip, picking a face applies it
/// at once, so each pick is one undo step.
struct TextPanel: View {
    let model: EditorModel
    /// The clip being changed, or nil to add one.
    let clip: String?
    let close: () -> Void
    @State private var words: String
    @State private var face: String
    @FocusState private var typing: Bool

    init(model: EditorModel, clip: String?, close: @escaping () -> Void) {
        self.model = model
        self.clip = clip
        self.close = close
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
    }

    private func pick(_ name: String) {
        face = name
        if let clip { model.setTextFont(clip, face: name) }
    }

    /// Save the words and close. Blank words add nothing; on a clip they
    /// leave its words as they were.
    private func done() {
        let trimmed = words.trimmingCharacters(in: .whitespacesAndNewlines)
        if let clip {
            if !trimmed.isEmpty, model.text(of: clip)?.content != words {
                model.setText(clip, words)
            }
        } else if !trimmed.isEmpty {
            model.addText(words, face: face)
        }
        typing = false
        close()
    }
}
