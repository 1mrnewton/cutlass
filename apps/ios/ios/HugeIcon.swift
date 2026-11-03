import SwiftUI

/// A free Hugeicons glyph from `crates/icons`. The name is the one in
/// `icons.txt`; `cargo run -p icongen` writes the image set. The foreground
/// style tints it. The button or the text beside it carries the label.
struct HugeIcon: View {
    let name: String
    var size: CGFloat = 22

    init(_ name: String, size: CGFloat = 22) {
        self.name = name
        self.size = size
    }

    var body: some View {
        Image(Self.asset(name))
            .renderingMode(.template)
            .resizable()
            .frame(width: size, height: size)
            .accessibilityHidden(true)
    }

    /// `textVideo` becomes `icon_text_video`, as in `cutlass_icons::asset_name`.
    static func asset(_ name: String) -> String {
        var asset = "icon_"
        for character in name {
            if character.isUppercase {
                asset += "_" + character.lowercased()
            } else {
                asset.append(character)
            }
        }
        return asset
    }
}
