import AVFAudio
import OSLog
import SwiftUI

@main struct MyApp: App {
    init() {
        // An editor's preview is heard with the ring switch off, like a video.
        let audio = AVAudioSession.sharedInstance()
        try? audio.setCategory(.playback, mode: .moviePlayback)
        try? audio.setActive(true)
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
