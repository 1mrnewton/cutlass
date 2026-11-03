import SwiftUI

/// Volume and speed for the selected clip. A value is sent once the finger
/// lifts, so one drag is one undo step.
struct ClipPanel: View {
    let model: EditorModel
    let clip: Clip
    let close: () -> Void
    @State private var volume: Double
    @State private var speed: Double

    static let speeds: [Double] = [0.25, 0.5, 1, 1.5, 2, 3, 4]

    init(model: EditorModel, clip: Clip, close: @escaping () -> Void) {
        self.model = model
        self.clip = clip
        self.close = close
        _volume = State(initialValue: clip.volume)
        _speed = State(initialValue: clip.speed)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Text(clip.name)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                Spacer()
                Button(action: close) {
                    Image(systemName: "xmark")
                        .frame(width: 32, height: 32)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Close")
            }
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Volume")
                    Spacer()
                    Text("\(Int((volume * 100).rounded()))%")
                        .monospacedDigit()
                        .foregroundStyle(.white.opacity(0.7))
                }
                .font(.footnote)
                Slider(value: $volume, in: 0...2) { editing in
                    if !editing { model.setVolume(clip.id, volume) }
                }
                .accessibilityLabel("Volume")
            }
            VStack(alignment: .leading, spacing: 6) {
                Text("Speed").font(.footnote)
                HStack(spacing: 6) {
                    ForEach(Self.speeds, id: \.self) { choice in
                        Button {
                            speed = choice
                            model.setSpeed(clip.id, choice)
                        } label: {
                            Text(Self.label(choice))
                                .font(.caption.weight(.medium).monospacedDigit())
                                .frame(maxWidth: .infinity, minHeight: 32)
                                .background(
                                    speed == choice ? Color.white : Color.white.opacity(0.12),
                                    in: RoundedRectangle(cornerRadius: 8)
                                )
                                .foregroundStyle(speed == choice ? Color.black : Color.white)
                        }
                        .accessibilityLabel("Speed \(Self.label(choice))")
                    }
                }
            }
        }
        .foregroundStyle(.white)
        .padding(16)
        .background(Color(white: 0.11))
        .onChange(of: clip) { _, shown in
            // Undo, redo, or a refusal: show what the project holds.
            volume = shown.volume
            speed = shown.speed
        }
    }

    static func label(_ speed: Double) -> String {
        let text = speed == speed.rounded() ? String(Int(speed)) : String(speed)
        return text + "×"
    }
}
