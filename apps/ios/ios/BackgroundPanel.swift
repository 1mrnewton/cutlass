import SwiftUI

/// What fills the canvas around the main clips. Color sets the project
/// color; Blur puts every main clip over a blurred copy of itself. A tap
/// applies at once, as one undo step.
struct BackgroundPanel: View {
    let model: EditorModel
    let close: () -> Void
    @State private var tab: Tab

    enum Tab: String, CaseIterable {
        case color = "Color"
        case blur = "Blur"
    }

    private static let colors = backgroundColors()
    private static let blurs = backgroundBlurs()

    init(model: EditorModel, close: @escaping () -> Void) {
        self.model = model
        self.close = close
        if case .blur = model.background {
            _tab = State(initialValue: .blur)
        } else {
            _tab = State(initialValue: .color)
        }
    }

    var body: some View {
        PanelFrame(title: "Background", close: close) {
            Picker("Background", selection: $tab) {
                ForEach(Tab.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    NoneTile(chosen: model.background == .none) {
                        model.setBackground(.none)
                    }
                    switch tab {
                    case .color:
                        ForEach(Self.colors, id: \.self) { color in
                            SwatchTile(color: color, chosen: model.background == .color(color: color)) {
                                model.setBackground(.color(color: color))
                            }
                        }
                    case .blur:
                        ForEach(Array(Self.blurs.enumerated()), id: \.offset) { index, blur in
                            BlurTile(
                                picture: model.picture,
                                level: index + 1,
                                strength: blur,
                                chosen: model.background == .blur(blur: blur)
                            ) {
                                model.setBackground(.blur(blur: blur))
                            }
                        }
                    }
                }
                // Room for the ring around the chosen tile.
                .padding(4)
            }
        }
    }
}

let tile: CGFloat = 52

/// A tile that takes a color or blur off.
struct NoneTile: View {
    let chosen: Bool
    var size: CGFloat = tile
    let tap: () -> Void

    var body: some View {
        Button(action: tap) {
            Image(systemName: "nosign")
                .font(.system(size: 22))
                .foregroundStyle(Palette.muted)
                .frame(width: size, height: size)
                .background(Color(white: 0.17), in: RoundedRectangle(cornerRadius: 8))
                .overlay(ChosenRing(chosen: chosen))
        }
        .accessibilityLabel("None")
        .accessibilityAddTraits(chosen ? .isSelected : [])
    }
}

/// A tile of one color, ringed when chosen.
struct SwatchTile: View {
    let color: String
    let chosen: Bool
    var size: CGFloat = tile
    let tap: () -> Void

    var body: some View {
        Button(action: tap) {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(hex: color))
                .frame(width: size, height: size)
                .overlay(
                    RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.15))
                )
                .overlay(ChosenRing(chosen: chosen))
        }
        .accessibilityLabel(color)
        .accessibilityAddTraits(chosen ? .isSelected : [])
    }
}

/// The picture at the playhead, blurred more at each level.
private struct BlurTile: View {
    let picture: CGImage?
    let level: Int
    let strength: Double
    let chosen: Bool
    let tap: () -> Void

    var body: some View {
        Button(action: tap) {
            ZStack {
                Color(white: 0.17)
                if let picture {
                    Image(decorative: picture, scale: 1)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .blur(radius: strength * 60)
                }
            }
            .frame(width: tile, height: tile)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(ChosenRing(chosen: chosen))
        }
        .accessibilityLabel("Blur \(level)")
        .accessibilityAddTraits(chosen ? .isSelected : [])
    }
}

private struct ChosenRing: View {
    let chosen: Bool

    var body: some View {
        RoundedRectangle(cornerRadius: 8)
            .stroke(chosen ? Palette.accent : .clear, lineWidth: 2)
            .padding(-3)
    }
}

extension Color {
    /// `#RRGGBB`. Anything else is black.
    init(hex: String) {
        let digits = hex.hasPrefix("#") ? String(hex.dropFirst()) : hex
        let value = digits.count == 6 ? UInt32(digits, radix: 16) ?? 0 : 0
        self.init(
            red: Double((value >> 16) & 0xFF) / 255,
            green: Double((value >> 8) & 0xFF) / 255,
            blue: Double(value & 0xFF) / 255
        )
    }
}
