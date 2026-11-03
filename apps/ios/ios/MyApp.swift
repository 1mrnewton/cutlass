import AVFAudio
import OSLog
import SwiftUI

@main struct MyApp: App {
    init() {
        // An editor's preview is heard with the ring switch off, like a video.
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .moviePlayback)
        do {
            try start(root: Places.root)
        } catch {
            Logger(subsystem: "cutlass", category: "app").error("\(error.localizedDescription)")
        }
    }

    var body: some Scene {
        WindowGroup {
            AppScreen()
                .preferredColorScheme(.dark)
        }
    }
}
