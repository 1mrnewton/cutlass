import AVFoundation
import SwiftUI
import UIKit

/// Video effects: a grid of the effects Rust ships, each tile looping its
/// short preview video. Tapping one adds it at the playhead, over every
/// clip below it.
struct EffectsPanel: View {
    let model: EditorModel
    var title = "Video effects"
    /// What a tap does; adding at the playhead when nil.
    var pick: ((String) -> Void)?
    let close: () -> Void

    private let columns = Array(repeating: GridItem(.flexible(), spacing: 10), count: 4)

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Button(action: close) {
                    HugeIcon("done", size: 20)
                        .frame(width: 40, height: 32)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Done")
            }
            ScrollView {
                LazyVGrid(columns: columns, spacing: 12) {
                    ForEach(model.effects, id: \.id) { effect in
                        Button {
                            if let pick { pick(effect.id) } else { model.addEffect(effect.id) }
                            close()
                        } label: {
                            VStack(spacing: 5) {
                                LoopingVideo(name: effect.preview)
                                    .aspectRatio(1, contentMode: .fit)
                                    .clipShape(RoundedRectangle(cornerRadius: 8))
                                Text(effect.name)
                                    .font(.caption2)
                                    .lineLimit(1)
                                    .foregroundStyle(.white.opacity(0.85))
                            }
                        }
                        .accessibilityLabel(effect.name)
                    }
                }
            }
            .frame(height: 230)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 16)
        .padding(.top, 10)
        .padding(.bottom, 12)
        .background(Color(white: 0.11))
    }
}

/// A muted video from the app bundle that plays over and over.
struct LoopingVideo: UIViewRepresentable {
    let name: String

    func makeUIView(context: Context) -> PlayerView {
        let view = PlayerView()
        view.backgroundColor = UIColor(white: 0.18, alpha: 1)
        guard let url = Self.url(of: name) else { return view }
        let player = AVQueuePlayer()
        player.isMuted = true
        // The preview is decoration: it must not stop other sound.
        player.preventsDisplaySleepDuringVideoPlayback = false
        context.coordinator.looper = AVPlayerLooper(
            player: player, templateItem: AVPlayerItem(url: url))
        view.playerLayer.player = player
        view.playerLayer.videoGravity = .resizeAspectFill
        player.play()
        return view
    }

    func updateUIView(_ view: PlayerView, context: Context) {}

    static func dismantleUIView(_ view: PlayerView, coordinator: Coordinator) {
        view.playerLayer.player?.pause()
        view.playerLayer.player = nil
        coordinator.looper = nil
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var looper: AVPlayerLooper?
    }

    /// Xcode may copy the folders' files to the bundle's top level or keep
    /// the folders; look in each.
    static func url(of name: String) -> URL? {
        Bundle.main.url(forResource: name, withExtension: nil)
            ?? Bundle.main.url(forResource: name, withExtension: nil, subdirectory: "EffectPreviews")
            ?? Bundle.main.url(
                forResource: name, withExtension: nil, subdirectory: "TransitionPreviews")
    }

    final class PlayerView: UIView {
        override static var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer {
            // The layer class above makes this cast always succeed.
            layer as? AVPlayerLayer ?? AVPlayerLayer()
        }
    }
}
