import SwiftUI

/// One tool bar as Rust lists it: a back chevron when there is somewhere to
/// go back to, then the tools, scrolling sideways. A tool that does not work
/// yet is dimmed but still answers a tap.
struct ToolBarView: View {
    let tools: [ToolItem]
    let back: (() -> Void)?
    let tap: (ToolItem) -> Void

    var body: some View {
        HStack(spacing: 0) {
            if let back {
                Button(action: back) {
                    HugeIcon("chevronLeft", size: 20)
                        .foregroundStyle(.white)
                        .frame(width: 40, height: 52)
                        .background(Color(white: 0.17), in: RoundedRectangle(cornerRadius: 6))
                }
                .accessibilityLabel("Back")
                .padding(.leading, 8)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 4) {
                    ForEach(tools, id: \.kind) { tool in
                        Button { tap(tool) } label: { ToolLabel(tool: tool) }
                            .accessibilityHint(tool.ready ? "" : "Not available yet")
                    }
                }
                .padding(.horizontal, 8)
            }
        }
        .foregroundStyle(.white)
        .padding(.top, 10)
        .padding(.bottom, 4)
        .frame(maxWidth: .infinity)
        .background(Color(white: 0.11))
    }
}

private struct ToolLabel: View {
    let tool: ToolItem

    var body: some View {
        VStack(spacing: 4) {
            HugeIcon(tool.kind.icon)
                .frame(height: 26)
            Text(tool.label).font(.caption2).lineLimit(1).minimumScaleFactor(0.8)
        }
        .opacity(tool.ready ? 1 : 0.4)
        .frame(width: 68, height: 52)
        .contentShape(Rectangle())
    }
}

extension ToolKind {
    /// The SF Symbol drawn for this tool.
    /// The name in `crates/icons/icons.txt`.
    var icon: String {
        switch self {
        case .edit: "scissors"
        case .audio: "audio"
        case .text: "text"
        case .effects: "effects"
        case .overlay: "overlay"
        case .captions: "captions"
        case .filters: "filters"
        case .adjust: "adjustments"
        case .stickers: "stickers"
        case .aspectRatio: "aspectRatio"
        case .background: "background"
        case .split: "split"
        case .volume: "volume"
        case .speed: "speed"
        case .animations: "animations"
        case .delete: "delete"
        case .duplicate: "duplicate"
        case .replace: "replace"
        case .crop: "crop"
        case .mirror: "mirror"
        case .rotate: "rotate"
        case .reverse: "reverse"
        case .freeze: "freeze"
        case .extractAudio: "extractAudio"
        case .fade: "fade"
        case .beats: "beat"
        case .unlink: "unlink"
        case .splice: "splice"
        case .addOverlay: "addOverlay"
        case .importSound: "importSound"
        case .record: "record"
        case .addText: "addText"
        case .editText: "editText"
        case .draw: "draw"
        case .videoEffects: "plugin"
        }
    }
}
