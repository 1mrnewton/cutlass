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
                    Image(systemName: "chevron.left")
                        .font(.body.weight(.semibold))
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
            Image(systemName: tool.kind.symbol)
                .font(.title3)
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
    var symbol: String {
        switch self {
        case .edit: "scissors"
        case .audio: "music.note"
        case .text: "textformat"
        case .effects: "sparkles"
        case .overlay: "square.on.square"
        case .captions: "captions.bubble"
        case .filters: "camera.filters"
        case .adjust: "slider.horizontal.3"
        case .stickers: "face.smiling"
        case .aspectRatio: "aspectratio"
        case .background: "paintpalette"
        case .split: "square.split.2x1"
        case .volume: "speaker.wave.2"
        case .speed: "gauge.with.dots.needle.67percent"
        case .animations: "play.square.stack"
        case .delete: "trash"
        case .duplicate: "plus.square.on.square"
        case .replace: "arrow.triangle.2.circlepath"
        case .crop: "crop"
        case .mirror: "arrow.left.and.right.righttriangle.left.righttriangle.right"
        case .rotate: "rotate.right"
        case .reverse: "backward"
        case .freeze: "snowflake"
        case .extractAudio: "waveform"
        case .fade: "waveform.path"
        case .beats: "metronome"
        case .unlink: "link"
        case .splice: "arrow.down.to.line"
        case .addOverlay: "rectangle.badge.plus"
        case .importSound: "music.note.list"
        case .record: "mic"
        case .addText: "character.textbox"
        case .editText: "pencil"
        case .draw: "scribble"
        case .videoEffects: "wand.and.stars"
        }
    }
}
